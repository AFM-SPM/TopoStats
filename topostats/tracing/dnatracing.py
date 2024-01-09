"""Perform DNA Tracing"""
from collections import OrderedDict
from functools import partial
from itertools import repeat
import logging
import math
from multiprocessing import Pool
import os
from pathlib import Path
from typing import Dict, Union, Tuple
import warnings
import time

import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage, spatial, optimize, interpolate as interp
from scipy.signal import argrelextrema
from skimage.morphology import label, binary_dilation
from skimage.filters import gaussian, threshold_otsu
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from topoly import jones, homfly, params, reduce_structure, translate_code
import skimage.measure as skimage_measure
from tqdm import tqdm
import math as math

from topostats.logs.logs import LOGGER_NAME
from topostats.tracing.tracingfuncs import genTracingFuncs, reorderTrace
from topostats.tracing.skeletonize import getSkeleton, pruneSkeleton
from topostats.utils import convolve_skelly, ResolutionError

LOGGER = logging.getLogger(LOGGER_NAME)

OUTPUT_DIR = Path("/Users/laurawiggins/Desktop/")


class dnaTrace:
    """
    This class gets all the useful functions from the old tracing code and staples
    them together to create an object that contains the traces for each DNA molecule
    in an image and functions to calculate stats from those traces.

    The traces are stored in dictionaries labelled by their gwyddion defined grain
    number and are represented as numpy arrays.

    The object also keeps track of the skeletonised plots and other intermediates
    in case these are useful for other things in the future.

    2023-06-09 : This class has undergone some refactoring so that it works with a single grain. The `trace_grain()`
    helper function runs the class and returns the expected statistics whilst the `trace_image()` function handles
    processing all detected grains within an image. The original methods of skeletonisation are available along with
    additional methods from scikit-image.

    Some bugs have been identified and corrected see commits for further details...

    236750b2
    2a79c4ff
    """

    def __init__(
        self,
        image: np.ndarray,
        grain: np.ndarray,
        filename: str,
        pixel_to_nm_scaling: float,
        min_skeleton_size: int = 10,
        convert_nm_to_m: bool = True,
        skeletonisation_params={"skeletonisation_method": "zhang"},
        pruning_params={"pruning_method": "joe"},
        n_grain: int = None,
    ):
        """Initialise the class.

        Parameters
        ==========
        image: np.ndarray,
            Cropped image, typically padded beyond the bounding box.
        grain: np.ndarray,
            Labelled mask for the grain, typically padded beyond the bounding box.
        filename: str
            Filename being processed
        pixel_to_nm_scaling: float,
            Pixel to nm scaling
        min_skeleton_size: int = 10,
            Minimum skeleton size below which tracing statistics are not calculated.
        convert_nm_to_m: bool = True,
            Convert nanometers to metres.
        skeletonisation_params: dict
            Any parameters associated with the method of skeletonisation.
        pruning_params: dict
            Any parameters associated with the method of pruning.
        n_grain: int
            Grain number being processed (only  used in logging).
        """
        self.image = image  # if convert_nm_to_m else image
        self.grain = grain
        self.filename = filename
        self.pixel_to_nm_scaling = pixel_to_nm_scaling * 1e-9 if convert_nm_to_m else pixel_to_nm_scaling
        self.min_skeleton_size = min_skeleton_size
        self.skeletonisation_params = skeletonisation_params
        self.pruning_params = pruning_params
        self.n_grain = n_grain
        self.number_of_rows = self.image.shape[0]
        self.number_of_columns = self.image.shape[1]
        self.sigma = 2  # 0.7 / (self.pixel_to_nm_scaling * 1e9)

        self.gauss_image = gaussian(self.image, self.sigma)
        self.smoothed_grain = np.zeros_like(image)
        self.skeleton = np.zeros_like(image)
        self.pruned_skeleton = np.zeros_like(image)
        self.disordered_trace = None
        self.node_image = np.zeros_like(image)
        self.node_dict = None
        self.node_image_dict = None
        self.ordered_trace = None
        self.ordered_trace_img = np.zeros_like(image)
        self.num_crossings = None
        self.avg_crossing_confidence = None
        self.topology = [None]
        self.fitted_traces = []
        self.fitted_trace_img = np.zeros_like(image)
        self.splined_traces = []
        self.splined_trace_img = np.zeros_like(image)
        self.contour_lengths = []
        self.new_contour_length = None
        self.end_to_end_distances = []
        self.mol_is_circulars = []
        self.curvatures = []
        self.num_mols = 1

        self.visuals = np.zeros_like(image)

        self.neighbours = 5  # The number of neighbours used for the curvature measurement
        self.step_size = 7e-9

        # supresses scipy splining warnings
        warnings.filterwarnings("ignore")

        LOGGER.debug(f"[{self.filename}] Performing DNA Tracing")

    def trace_dna(self):
        """Perform DNA tracing."""
        self.get_disordered_trace()
        self.new_contour_length = self.fragment_skeleton_and_calc_contour_length()
        if self.disordered_trace is None:
            LOGGER.info(f"[{self.filename}] : Grain {self.n_grain} failed to Skeletonise")
        elif len(self.disordered_trace) >= self.min_skeleton_size:
            self.linear_or_circular(self.disordered_trace)
            nodes = nodeStats(
                filename=self.filename,
                image=self.image,
                grain=self.grain,
                smoothed_grain=self.smoothed_grain,
                skeleton=self.pruned_skeleton,
                px_2_nm=self.pixel_to_nm_scaling,
                n_grain=self.n_grain,
            )
            self.node_dict, self.node_image_dict = nodes.get_node_stats()
            self.avg_crossing_confidence = self.average_crossing_confs(self.node_dict)
            self.node_image = nodes.connected_nodes
            self.num_crossings = nodes.num_crossings  # len(self.node_dict)

            # try: # try to order using nodeStats
            if nodes.check_node_errorless():
                ordered_traces, self.visuals, self.topology = nodes.compile_trace()
                self.num_mols = len(ordered_traces)
                LOGGER.info(f"[{self.filename}] : Grain {self.n_grain} ordered via nodeStats.")
                LOGGER.info(f"[{self.filename}] : Grain {self.n_grain} has {len(ordered_traces)} molecules.")
                self.ordered_trace = ordered_traces
            else:
                LOGGER.info(
                    f"[{self.filename}] : Grain {self.n_grain} couldn't be traced due to errors in analysing the nodes."
                )
                raise ValueError
            """
            except ValueError: # if nodestats fails, use default ordering method
                LOGGER.info(f"[{self.filename}] : Grain {self.n_grain} failed to order through nodeStats, trying standard ordering.")
                self.get_ordered_traces()
            """
            for trace in self.ordered_trace:
                self.ordered_trace_img += self.coords_2_img(trace, self.image, ordered=True)
                if len(trace) >= self.min_skeleton_size:
                    mol_is_circular = self.linear_or_circular(trace)
                    self.mol_is_circulars.append(mol_is_circular)
                    fitted_trace = self.get_fitted_traces(trace, mol_is_circular)
                    self.fitted_trace_img += self.coords_2_img(fitted_trace, self.image)
                    # Propper cleanup needed - ordered trace instead of fitted trace
                    splined_trace = self.get_splined_traces(trace, trace, mol_is_circular)
                    self.splined_traces.append(np.array(splined_trace))
                    self.splined_trace_img += self.coords_2_img(np.array(splined_trace, dtype=np.int32), self.image)
                    # self.find_curvature()
                    # self.saveCurvature()
                    self.contour_lengths.append(self.measure_contour_length(splined_trace, mol_is_circular))
                    self.end_to_end_distances.append(self.measure_end_to_end_distance(splined_trace, mol_is_circular))
                else:  # fill the row with nothing so it can still be joined to grainstats
                    self.num_mols -= 1  # remove this from the num mols indexer
                    LOGGER.info(
                        f"[{self.filename}] [grain {self.n_grain}] : Grain ordered trace pixels < {self.min_skeleton_size}"
                    )
                    self.contour_lengths.append(None)
                    self.mol_is_circulars.append(None)
                    self.end_to_end_distances.append(None)

        else:
            LOGGER.info(f"[{self.filename}] [{self.n_grain}] : Grain skeleton pixels < {self.min_skeleton_size}")
            self.contour_lengths.append([])
            self.mol_is_circulars.append([])
            self.end_to_end_distances.append([])

    @staticmethod
    def average_crossing_confs(node_dict):
        sum_conf = 0
        valid_confs = 0
        for i, (_, values) in enumerate(node_dict.items()):
            conf = values["confidence"]
            if conf is not None:
                sum_conf += conf
                valid_confs += 1
            try:
                return sum_conf / (i + 1)
            except ZeroDivisionError:
                return None

    def smooth_grains(self, grain: np.ndarray) -> None:
        """Smoothes grains based on the lower number added from dilation or gaussian.
        (makes sure gaussian isnt too agressive."""
        dilation = ndimage.binary_dilation(grain, iterations=2).astype(np.int32)
        gauss = gaussian(grain, sigma=max(grain.shape) / 256)
        gauss[gauss > threshold_otsu(gauss) * 1.3] = 1
        gauss[gauss != 1] = 0
        gauss = gauss.astype(np.int32)
        if dilation.sum() - grain.sum() > gauss.sum() - grain.sum():
            print(f"gaussian: {gauss.sum()-grain.sum()}px")
            gauss = self.re_add_holes(grain, gauss)
            return gauss
        else:
            print(f"dilation: {dilation.sum()-grain.sum()}px")
            dilation = self.re_add_holes(grain, dilation)
            return dilation

    def re_add_holes(self, orig_mask, new_mask, holearea_min_max=[4, np.inf]):
        """As gaussian and dilation smoothing methods can close holes in the original mask,
        this function obtains those holes (based on the general background being the first due
        to padding) and adds them back into the smoothed mask. When paired, this essentailly
        just smooths the outer edge of the grains.

        Parameters:
        -----------
            orig_mask (_type_): _description_
            new_mask (_type_): _description_
            holearea_min_max (_type_): _description_
        """
        holesize_min_px = holearea_min_max[0] / ((self.pixel_to_nm_scaling / 1e-9) ** 2)
        holesize_max_px = holearea_min_max[1] / ((self.pixel_to_nm_scaling / 1e-9) ** 2)
        holes = 1 - orig_mask
        holes = label(holes)
        sizes = [holes[holes == i].size for i in range(1, holes.max() + 1)]
        holes[holes == 1] = 0  # set background to 0

        # self.holes = holes.copy()

        for i, size in enumerate(sizes):
            if size < holesize_min_px or size > holesize_max_px:  # small holes may be fake are left out
                holes[holes == i + 1] = 0
        holes[holes != 0] = 1

        # compare num holes in each mask
        holey_smooth = new_mask.copy()
        holey_smooth[holes == 1] = 0

        return holey_smooth

    def get_disordered_trace(self):
        self.smoothed_grain = self.smooth_grains(self.grain)
        self.skeleton = getSkeleton(self.gauss_image, self.smoothed_grain).get_skeleton(
            self.skeletonisation_params.copy()
        )
        # np.savetxt(OUTPUT_DIR / "skel.txt", self.skeleton)
        # np.savetxt(OUTPUT_DIR / "image.txt", self.image)
        # np.savetxt(OUTPUT_DIR / "smooth.txt", self.smoothed_grain)
        self.pruned_skeleton = pruneSkeleton(self.gauss_image, self.skeleton).prune_skeleton(self.pruning_params.copy())
        self.pruned_skeleton = self.remove_touching_edge(self.pruned_skeleton)
        np.savetxt(OUTPUT_DIR / "pruned_skeleton.txt", self.pruned_skeleton)
        self.disordered_trace = np.argwhere(self.pruned_skeleton == 1)

    @staticmethod
    def remove_touching_edge(skeleton: np.ndarray):
        """Removes any skeletons touching the boarder (to prevent errors later).

        Parameters
        ----------
        skeleton: np.ndarray
            A binary array where touching clusters of 1's become 0's if touching the edge of the array.
        """
        for edge in [skeleton[0, :-1], skeleton[:-1, -1], skeleton[-1, 1:], skeleton[1:, 0]]:
            uniques = np.unique(edge)
            for i in uniques:
                skeleton[skeleton == i] = 0
        return skeleton

    # FIXME : It is straight-forward to get bounding boxes for grains, need to then have a dictionary of original image
    #         and label for each grain to then be processed.
    def _get_bounding_box(array: np.ndarray) -> np.ndarray:
        """Calculate bounding box for each grain."""
        rows = grain_array.any(axis=1)
        LOGGER.info(f"[{self.filename}] : Cropping grain")
        if rows.any():
            m, n = grain_array.shape
            cols = grain_array.any(0)
            return (rows.argmax(), m - rows[::-1].argmax(), cols.argmax(), n - cols[::-1].argmax())
            return grain_array[rows.argmax() : m - rows[::-1].argmax(), cols.argmax() : n - cols[::-1].argmax()]
        return np.empty((0, 0), dtype=bool)

    def _crop_array(array: np.ndarray, bounding_box: Tuple) -> np.ndarray:
        """Crop an array.

        Parameters
        ----------
        array: np.ndarray
            2D Numpy array to be cropped.
        bounding_box: Tuple
            Tuple of co-ordinates to crop, should be of form (max_x, min_x, max_y, min_y) as returned by
        _get_bounding_box().

        Returns
        -------
        np.ndarray()
            Cropped array
        """
        return array[bounding_box[0] : bounding_box[1], bounding_box[2] : bounding_box[3]]

    @staticmethod
    def coords_2_img(coords, image, ordered=False):
        comb = np.zeros_like(image)
        if ordered:
            comb[coords[:, 0].astype(np.int32), coords[:, 1].astype(np.int32)] = np.arange(1, len(coords) + 1)
        else:
            comb[np.floor(coords[:, 0]).astype(np.int32), np.floor(coords[:, 1]).astype(np.int32)] = 1
        return comb

    @staticmethod
    def concat_images_in_dict(image_size, image_dict: dict):
        """Concatonates the skeletons in the skeleton dictionary onto one image"""
        skeletons = np.zeros(image_size)
        for skeleton in image_dict.values():
            skeletons += skeleton
        return skeletons

    def linear_or_circular(self, traces: np.ndarray):
        """Determines whether each molecule is circular or linear based on the local environment of each pixel from the trace

        This function is sensitive to branches from the skeleton so might need to implement a function to remove them

        Parameters
        ----------
        traces: dict
            A dictionary of the molecule_number and points within the skeleton.
        """

        """
        points_with_one_neighbour = 0
        fitted_trace_list = traces.tolist()

        # For loop determines how many neighbours a point has - if only one it is an end
        for x, y in fitted_trace_list:
            if genTracingFuncs.count_and_get_neighbours(x, y, fitted_trace_list)[0] == 1:
                points_with_one_neighbour += 1
            else:
                pass

        if points_with_one_neighbour == 0:
            return True
        else:
            return False
        """

        dist = np.sqrt((traces[0][0] - traces[-1][0]) ** 2 + (traces[0][1] - traces[-1][1]) ** 2)
        if dist > np.sqrt(2):
            return False
        else:
            return True

    def get_ordered_traces(self):
        """Depending on whether the mol is circular or linear, order the traces so the points follow."""
        if self.mol_is_circular:
            self.ordered_trace, trace_completed = reorderTrace.circularTrace(self.disordered_trace)

            if not trace_completed:
                self.mol_is_circular = False
                try:
                    self.ordered_trace = reorderTrace.linearTrace(self.ordered_trace.tolist())
                except UnboundLocalError:
                    pass

        elif not self.mol_is_circular:
            self.ordered_trace = reorderTrace.linearTrace(self.disordered_trace.tolist())

    def get_fitted_traces(self, ordered_trace, mol_is_circular):
        """Create trace coordinates (for each identified molecule) that are adjusted to lie
        along the highest points of each traced molecule
        """

        individual_skeleton = ordered_trace
        # This indexes a 3 nm height profile perpendicular to DNA backbone
        # note that this is a hard coded parameter
        index_width = int(3e-9 / (self.pixel_to_nm_scaling))
        if index_width < 2:
            index_width = 2

        for coord_num, trace_coordinate in enumerate(individual_skeleton):
            height_values = None

            # Block of code to prevent indexing outside image limits
            # e.g. indexing self.gauss_image[130, 130] for 128x128 image
            if trace_coordinate[0] < 0:
                # prevents negative number indexing
                # i.e. stops (trace_coordinate - index_width) < 0
                trace_coordinate[0] = index_width
            elif trace_coordinate[0] >= (self.number_of_rows - index_width):
                # prevents indexing above image range causing IndexError
                trace_coordinate[0] = self.number_of_rows - index_width
            # do same for y coordinate
            elif trace_coordinate[1] < 0:
                trace_coordinate[1] = index_width
            elif trace_coordinate[1] >= (self.number_of_columns - index_width):
                trace_coordinate[1] = self.number_of_columns - index_width

            # calculate vector to n - 2 coordinate in trace
            if mol_is_circular:
                nearest_point = individual_skeleton[coord_num - 2]
                vector = np.subtract(nearest_point, trace_coordinate)
                vector_angle = math.degrees(math.atan2(vector[1], vector[0]))
            else:
                try:
                    nearest_point = individual_skeleton[coord_num + 2]
                except IndexError:
                    nearest_point = individual_skeleton[coord_num - 2]
                vector = np.subtract(nearest_point, trace_coordinate)
                vector_angle = math.degrees(math.atan2(vector[1], vector[0]))

            if vector_angle < 0:
                vector_angle += 180

            # if  angle is closest to 45 degrees
            if 67.5 > vector_angle >= 22.5:
                perp_direction = "negative diaganol"
                # positive diagonal (change in x and y)
                # Take height values at the inverse of the positive diaganol
                # (i.e. the negative diaganol)
                y_coords = np.arange(trace_coordinate[1] - index_width, trace_coordinate[1] + index_width)[::-1]
                x_coords = np.arange(trace_coordinate[0] - index_width, trace_coordinate[0] + index_width)

            # if angle is closest to 135 degrees
            elif 157.5 >= vector_angle >= 112.5:
                perp_direction = "positive diaganol"
                y_coords = np.arange(trace_coordinate[1] - index_width, trace_coordinate[1] + index_width)
                x_coords = np.arange(trace_coordinate[0] - index_width, trace_coordinate[0] + index_width)

            # if angle is closest to 90 degrees
            if 112.5 > vector_angle >= 67.5:
                perp_direction = "horizontal"
                x_coords = np.arange(trace_coordinate[0] - index_width, trace_coordinate[0] + index_width)
                y_coords = np.full(len(x_coords), trace_coordinate[1])

            elif 22.5 > vector_angle:  # if angle is closest to 0 degrees
                perp_direction = "vertical"
                y_coords = np.arange(trace_coordinate[1] - index_width, trace_coordinate[1] + index_width)
                x_coords = np.full(len(y_coords), trace_coordinate[0])

            elif vector_angle >= 157.5:  # if angle is closest to 180 degrees
                perp_direction = "vertical"
                y_coords = np.arange(trace_coordinate[1] - index_width, trace_coordinate[1] + index_width)
                x_coords = np.full(len(y_coords), trace_coordinate[0])

            # Use the perp array to index the guassian filtered image
            perp_array = np.column_stack((x_coords, y_coords))
            try:
                height_values = self.gauss_image[perp_array[:, 0], perp_array[:, 1]]
            except IndexError:
                perp_array[:, 0] = np.where(
                    perp_array[:, 0] > self.gauss_image.shape[0], self.gauss_image.shape[0], perp_array[:, 0]
                )
                perp_array[:, 1] = np.where(
                    perp_array[:, 1] > self.gauss_image.shape[1], self.gauss_image.shape[1], perp_array[:, 1]
                )
                height_values = self.image[perp_array[:, 1], perp_array[:, 0]]

            # Grab x,y coordinates for highest point
            # fine_coords = np.column_stack((fine_x_coords, fine_y_coords))
            sorted_array = perp_array[np.argsort(height_values)]
            highest_point = sorted_array[-1]

            try:
                # could use np.append() here
                fitted_coordinate_array = np.vstack((fitted_coordinate_array, highest_point))
            except UnboundLocalError:
                fitted_coordinate_array = highest_point

        return fitted_coordinate_array
        del fitted_coordinate_array  # cleaned up by python anyway?

    def get_splined_traces(self, fitted_trace, ordered_trace, mol_is_circular):
        """Gets a splined version of the fitted trace - useful for finding the radius of gyration etc

        This function actually calculates the average of several splines which is important for getting a good fit on
        the lower res data"""

        step_size_px = int(self.step_size / (self.pixel_to_nm_scaling))  # 3 nm step size
        step_size_px = 2 if step_size_px == 0 else step_size_px
        # Lets see if we just got with the pixel_to_nm_scaling
        # step_size = self.pixel_to_nm_scaling
        # interp_step = self.pixel_to_nm_scaling

        splining_success = True
        nbr = len(fitted_trace[:, 0])

        # Hard to believe but some traces have less than 4 coordinates in total
        if len(fitted_trace[:, 1]) < 4:
            splined_trace = fitted_trace
            # continue

        # The degree of spline fit used is 3 so there cannot be less than 3 points in the splined trace
        while nbr / step_size_px < 4:
            if step_size_px <= 1:
                step_size_px = 1
                break
            step_size_px = -1
        if mol_is_circular:
            smoothness = 2
            periodicity = 2
        else:
            smoothness = 5
            periodicity = 0
        # if nbr/step_size > 4: #the degree of spline fit is 3 so there cannot be less than 3 points in splined trace

        # ev_array = np.linspace(0, 1, nbr * step_size)
        ev_array = np.linspace(0, 1, int(nbr * step_size_px))

        for i in range(step_size_px):
            x_sampled = np.array([fitted_trace[:, 0][j] for j in range(i, len(fitted_trace[:, 0]), step_size_px)])
            y_sampled = np.array([fitted_trace[:, 1][j] for j in range(i, len(fitted_trace[:, 1]), step_size_px)])

            try:
                tck, u = interp.splprep([x_sampled, y_sampled], s=smoothness, per=periodicity, quiet=1, k=3)
                out = interp.splev(ev_array, tck)
                splined_trace = np.column_stack((out[0], out[1]))
            except ValueError:
                # Value error occurs when the "trace fitting" really messes up the traces

                x = np.array([ordered_trace[:, 0][j] for j in range(i, len(ordered_trace[:, 0]), step_size_px)])
                y = np.array([ordered_trace[:, 1][j] for j in range(i, len(ordered_trace[:, 1]), step_size_px)])

                try:
                    tck, u = interp.splprep([x, y], s=smoothness, per=periodicity, quiet=1)
                    out = interp.splev(np.linspace(0, 1, nbr * step_size_px), tck)
                    splined_trace = np.column_stack((out[0], out[1]))
                except ValueError:  # sometimes even the ordered_traces are too bugged out so just delete these traces
                    print("ValueError")
                    # self.mol_is_circular.pop(dna_num)
                    # self.disordered_trace.pop(dna_num)
                    # self.grain.pop(dna_num)
                    # self.ordered_trace.pop(dna_num)
                    splining_success = False
                    try:
                        del spline_running_total
                    except UnboundLocalError:  # happens if splining fails immediately
                        break
                    break

            try:
                spline_running_total = np.add(spline_running_total, splined_trace)
            except NameError:
                spline_running_total = np.array(splined_trace)

        # if not splining_success:
        #     continue

        spline_average = np.divide(spline_running_total, [step_size_px, step_size_px])
        del spline_running_total
        # ensure spline lies within bounds
        spline_average = spline_average[spline_average[:, 0] > 0]
        spline_average = spline_average[spline_average[:, 1] > 0]
        spline_average = spline_average[spline_average[:, 0] < self.image.shape[0]]
        spline_average = spline_average[spline_average[:, 1] < self.image.shape[1]]
        return spline_average

    def show_traces(self):
        plt.pcolormesh(self.image, vmax=-3e-9, vmin=3e-9)
        plt.colorbar()
        plt.plot(self.ordered_trace[:, 0], self.ordered_trace[:, 1], markersize=1)
        plt.plot(self.fitted_trace[:, 0], self.fitted_trace[:, 1], markersize=1)
        plt.plot(self.splined_trace[:, 0], self.splined_trace[:, 1], markersize=1)

        plt.show()
        plt.close()

    def saveTraceFigures(
        self, filename: Union[str, Path], channel_name: str, vmaxval, vminval, output_dir: Union[str, Path] = None
    ):
        if output_dir:
            filename = self._checkForSaveDirectory(filename, output_dir)

        # save_file = filename[:-4]

        vmaxval = 20e-9
        vminval = -10e-9

        plt.pcolormesh(self.image, vmax=vmaxval, vmin=vminval)
        plt.colorbar()
        # plt.savefig("%s_%s_originalImage.png" % (save_file, channel_name))
        plt.savefig(output_dir / filename / f"{channel_name}_original.png")
        plt.close()

        plt.pcolormesh(self.image, vmax=vmaxval, vmin=vminval)
        plt.colorbar()
        # disordered_trace_list = self.ordered_trace[dna_num].tolist()
        # less_dense_trace = np.array([disordered_trace_list[i] for i in range(0,len(disordered_trace_list),5)])
        plt.plot(self.splined_trace[:, 0], self.splined_trace[:, 1], color="c", linewidth=1.0)
        if self.mol_is_circular:
            starting_point = 0
        else:
            starting_point = self.neighbours
        length = len(self.curvature)
        plt.plot(
            self.splined_trace[starting_point, 0],
            self.splined_trace[starting_point, 1],
            color="#D55E00",
            markersize=3.0,
            marker=5,
        )
        plt.plot(
            self.splined_trace[starting_point + int(length / 6), 0],
            self.splined_trace[starting_point + int(length / 6), 1],
            color="#E69F00",
            markersize=3.0,
            marker=5,
        )
        plt.plot(
            self.splined_trace[starting_point + int(length / 6 * 2), 0],
            self.splined_trace[starting_point + int(length / 6 * 2), 1],
            color="#F0E442",
            markersize=3.0,
            marker=5,
        )
        plt.plot(
            self.splined_trace[starting_point + int(length / 6 * 3), 0],
            self.splined_trace[starting_point + int(length / 6 * 3), 1],
            color="#009E74",
            markersize=3.0,
            marker=5,
        )
        plt.plot(
            self.splined_trace[starting_point + int(length / 6 * 4), 0],
            self.splined_trace[starting_point + int(length / 6 * 4), 1],
            color="#0071B2",
            markersize=3.0,
            marker=5,
        )
        plt.plot(
            self.splined_trace[starting_point + int(length / 6 * 5), 0],
            self.splined_trace[starting_point + int(length / 6 * 5), 1],
            color="#CC79A7",
            markersize=3.0,
            marker=5,
        )
        plt.savefig(f"{save_file}_{channel_name}_splinedtrace_with_markers.png")
        plt.close()

        plt.pcolormesh(self.image, vmax=vmaxval, vmin=vminval)
        plt.colorbar()
        plt.plot(self.splined_trace[:, 0], self.splined_trace[:, 1], color="c", linewidth=1.0)
        # plt.savefig("%s_%s_splinedtrace.png" % (save_file, channel_name))
        plt.savefig(output_dir / filename / f"{channel_name}_splinedtrace.png")
        LOGGER.info(f"Splined Trace image saved to : {str(output_dir / filename / f'{channel_name}_splinedtrace.png')}")
        plt.close()

        # plt.pcolormesh(self.image, vmax=vmaxval, vmin=vminval)
        # plt.colorbar()
        # LOOP REMOVED
        # for dna_num in sorted(self.disordered_trace.keys()):
        # disordered_trace_list = self.disordered_trace[dna_num].tolist()
        # less_dense_trace = np.array([disordered_trace_list[i] for i in range(0,len(disordered_trace_list),5)])
        plt.plot(
            self.disordered_trace[:, 0],
            self.disordered_trace[:, 1],
            "o",
            markersize=1.0,
            color="c",
        )
        # plt.savefig("%s_%s_disorderedtrace.png" % (save_file, channel_name))
        # plt.savefig(output_dir / filename / f"{channel_name}_disordered_trace.png")
        plt.savefig(output_dir / f"{filename}.png")
        plt.close()
        LOGGER.info(
            f"Disordered trace image saved to : {str(output_dir / filename / f'{channel_name}_disordered_trace.png')}"
        )

        # plt.pcolormesh(self.image, vmax=vmaxval, vmin=vminval)
        # plt.colorbar()
        # for dna_num in sorted(self.grain.keys()):
        #    grain_plt = np.argwhere(self.grain[dna_num] == 1)
        #    plt.plot(grain_plt[:, 0], grain_plt[:, 1], "o", markersize=2, color="c")
        # plt.savefig("%s_%s_grains.png" % (save_file, channel_name))
        # plt.savefig(output_dir / filename / f"{channel_name}_grains.png")
        # plt.savefig(output_dir / f"{filename}_grain.png")
        # plt.close()
        LOGGER.info(f"Grains image saved to : {str(output_dir / filename / f'{channel_name}_grains.png')}")

    # FIXME : Replace with Path() (.mkdir(parent=True, exists=True) negate need to handle errors.)
    def _checkForSaveDirectory(self, filename, new_output_dir):
        split_directory_path = os.path.split(filename)

        try:
            os.mkdir(os.path.join(split_directory_path[0], new_output_dir))
        except OSError:  # OSError happens if the directory already exists
            pass

        updated_filename = os.path.join(split_directory_path[0], new_output_dir, split_directory_path[1])

        return updated_filename

    def find_curvature(self):
        curve = []
        contour = 0
        coordinates = np.zeros([2, self.neighbours * 2 + 1])
        for i, (x, y) in enumerate(self.splined_trace):
            # Extracts the coordinates for the required number of points and puts them in an array
            if self.mol_is_circular or (self.neighbours < i < len(self.splined_trace) - self.neighbours):
                for j in range(self.neighbours * 2 + 1):
                    coordinates[0][j] = self.splined_trace[i - j][0]
                    coordinates[1][j] = self.splined_trace[i - j][1]

                # Calculates the angles for the tangent lines to the left and the right of the point
                theta1 = math.atan(
                    (coordinates[1][self.neighbours] - coordinates[1][0])
                    / (coordinates[0][self.neighbours] - coordinates[0][0])
                )
                theta2 = math.atan(
                    (coordinates[1][-1] - coordinates[1][self.neighbours])
                    / (coordinates[0][-1] - coordinates[0][self.neighbours])
                )

                left = coordinates[:, : self.neighbours + 1]
                right = coordinates[:, -(self.neighbours + 1) :]

                xa = np.mean(left[0])
                ya = np.mean(left[1])

                xb = np.mean(right[0])
                yb = np.mean(right[1])

                # Calculates the curvature using the change in angle divided by the distance
                dist = math.hypot((xb - xa), (yb - ya))
                dist_real = dist * self.pixel_to_nm_scaling
                curve.append([i, contour, (theta2 - theta1) / dist_real])

                contour = contour + math.hypot(
                    (coordinates[0][self.neighbours] - coordinates[0][self.neighbours - 1]),
                    (coordinates[1][self.neighbours] - coordinates[1][self.neighbours - 1]),
                )
            self.curvature = curve

    def saveCurvature(self):
        # FIXME : Iterate directly over self.splined_trace.values() or self.splined_trace.items()
        # roc_array = np.zeros(shape=(1, 3))
        for i, [n, contour, c] in enumerate(self.curvature):
            try:
                roc_array = np.append(roc_array, np.array([[i, contour, c]]), axis=0)
                # oc_array.append([dna_num, i, contour, c])
            except NameError:
                roc_array = np.array([[i, contour, c]])
            # roc_array = np.vstack((roc_array, np.array([dna_num, i, c])))
        # roc_array = np.delete(roc_array, 0, 0)
        roc_stats = pd.DataFrame(roc_array)

        if not os.path.exists(os.path.join(os.path.dirname(self.filename), "Curvature")):
            os.mkdir(os.path.join(os.path.dirname(self.filename), "Curvature"))
        directory = os.path.join(os.path.dirname(self.filename), "Curvature")
        savename = os.path.join(directory, os.path.basename(self.filename)[:-4])
        roc_stats.to_json(savename + ".json")
        roc_stats.to_csv(savename + ".csv")

    def plotCurvature(self, dna_num):
        """Plot the curvature of the chosen molecule as a function of the contour length (in metres)"""

        curvature = np.array(self.curvature[dna_num])
        length = len(curvature)
        # FIXME : Replace with Path()
        if not os.path.exists(os.path.join(os.path.dirname(self.filename), "Curvature")):
            os.mkdir(os.path.join(os.path.dirname(self.filename), "Curvature"))
        directory = os.path.join(os.path.dirname(self.filename), "Curvature")
        savename = os.path.join(directory, os.path.basename(self.filename)[:-4])

        plt.figure()
        sns.lineplot(curvature[:, 1] * self.pixel_to_nm_scaling, curvature[:, 2], color="k")
        plt.ylim(-1e9, 1e9)
        plt.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
        plt.axvline(curvature[0][1], color="#D55E00")
        plt.axvline(curvature[int(length / 6)][1] * self.pixel_to_nm_scaling, color="#E69F00")
        plt.axvline(curvature[int(length / 6 * 2)][1] * self.pixel_to_nm_scaling, color="#F0E442")
        plt.axvline(curvature[int(length / 6 * 3)][1] * self.pixel_to_nm_scaling, color="#009E74")
        plt.axvline(curvature[int(length / 6 * 4)][1] * self.pixel_to_nm_scaling, color="#0071B2")
        plt.axvline(curvature[int(length / 6 * 5)][1] * self.pixel_to_nm_scaling, color="#CC79A7")
        plt.savefig(f"{savename}_{dna_num}_curvature.png")
        plt.close()

    @staticmethod
    def keep_largest_contour(binary_skeleton):
        # Label connected components with diagonal connections
        labelled_mask = label(binary_skeleton, connectivity=2)

        # Find properties of connected components
        regions = skimage_measure.regionprops(labelled_mask)

        # Find the largest contour
        largest_contour = max(regions, key=lambda x: x.area)

        # Create a new array with only the largest contour
        largest_contour_mask = np.zeros_like(binary_skeleton)
        for coord in largest_contour.coords:
            largest_contour_mask[coord[0], coord[1]] = 1

        return largest_contour_mask

    @staticmethod
    def measure_contour_length_from_skeleton(points, pixel_to_nm_scaling):
        hypotenuse_array = []
        for num in range(len(points) - 1):
            x1, y1 = points[num]
            x2, y2 = points[num + 1]
            
            try:
                hypotenuse_array.append(np.hypot((x1 - x2), (y1 - y2)))
            except NameError:
                hypotenuse_array = [np.hypot((x1 - x2), (y1 - y2))]
        
        contour_length = np.sum(np.array(hypotenuse_array)) * pixel_to_nm_scaling
        return contour_length
    
    def fragment_skeleton_and_calc_contour_length(self):
        import trace_skeleton
        import random

        skeleton_to_use = self.keep_largest_contour(self.pruned_skeleton)

        polys = trace_skeleton.from_numpy(skeleton_to_use.astype(int))

        fig, ax = plt.subplots()
        sum = 0
        for points in polys:
            x_values = [point[0] for point in points]
            y_values = [point[1] for point in points]
            color = (random.random(), random.random(), random.random())
            ax.plot(x_values, y_values, color=color)
        plt.savefig(OUTPUT_DIR / f"{self.n_grain}-image.png")

        sum = 0
        for points in polys:
            x_values = [point[0] for point in points]
            y_values = [point[1] for point in points]
            contour_length = self.measure_contour_length_from_skeleton(points, self.pixel_to_nm_scaling)
            sum += contour_length
        return sum

    def measure_contour_length(self, splined_trace, mol_is_circular):
        """Measures the contour length for each of the splined traces taking into
        account whether the molecule is circular or linear

        Contour length units are nm"""
        if mol_is_circular:
            for num, i in enumerate(splined_trace):
                x1 = splined_trace[num - 1, 0]
                y1 = splined_trace[num - 1, 1]
                x2 = splined_trace[num, 0]
                y2 = splined_trace[num, 1]

                try:
                    hypotenuse_array.append(math.hypot((x1 - x2), (y1 - y2)))
                except NameError:
                    hypotenuse_array = [math.hypot((x1 - x2), (y1 - y2))]

            contour_length = np.sum(np.array(hypotenuse_array)) * self.pixel_to_nm_scaling
            del hypotenuse_array
            return contour_length

        else:
            for num, i in enumerate(splined_trace):
                try:
                    x1 = splined_trace[num, 0]
                    y1 = splined_trace[num, 1]
                    x2 = splined_trace[num + 1, 0]
                    y2 = splined_trace[num + 1, 1]

                    try:
                        hypotenuse_array.append(math.hypot((x1 - x2), (y1 - y2)))
                    except NameError:
                        hypotenuse_array = [math.hypot((x1 - x2), (y1 - y2))]
                except IndexError:  # IndexError happens at last point in array
                    contour_length = np.sum(np.array(hypotenuse_array)) * self.pixel_to_nm_scaling
                    del hypotenuse_array
                    return contour_length
                
    

    def measure_end_to_end_distance(self, splined_trace, mol_is_circular):
        """Calculate the Euclidean distance between the start and end of linear molecules.
        The hypotenuse is calculated between the start ([0,0], [0,1]) and end ([-1,0], [-1,1]) of linear
        molecules. If the molecule is circular then the distance is set to zero (0).
        """
        if mol_is_circular:
            return 0
        else:
            x1 = splined_trace[0, 0]
            y1 = splined_trace[0, 1]
            x2 = splined_trace[-1, 0]
            y2 = splined_trace[-1, 1]
            return math.hypot((x1 - x2), (y1 - y2)) * self.pixel_to_nm_scaling


def trace_image(
    image: np.ndarray,
    grains_mask: np.ndarray,
    filename: str,
    pixel_to_nm_scaling: float,
    min_skeleton_size: int,
    skeletonisation_params: dict,
    pruning_params: dict,
    pad_width: int = 10,
    cores: int = 1,
) -> pd.DataFrame:
    """Processor function for tracing grains individually.

    Parameters
    ----------
    image : np.ndarray
        Full image as Numpy Array.
    grains_mask : np.ndarray
        Full image as Grains that are labelled.
    filename: str
        File being processed
    pixel_to_nm_scaling: float
        Pixel to nm scaling.
    min_skeleton_size: int
        Minimum size of grain in pixels after skeletonisation.
    skeletonisation_params: dict
        Any parameters associated with the method of skeletonisation.
    pruning_params: dict
        Any parameters associated with the method of pruning.
    pad_width: int
        Number of cells to pad arrays by, required to handle instances where grains touch the bounding box edges.
    cores : int
        Number of cores to process with.

    Returns
    -------
    pd.DataFrame
        Statistics from skeletonising and tracing the grains in the image.

    """
    # Check both arrays are the same shape
    assert image.shape == grains_mask.shape

    cropped_images, cropped_masks, bboxs = prep_arrays(image, grains_mask, pad_width)
    n_grains = len(cropped_images)
    LOGGER.info(f"[{filename}] : Calculating statistics for {n_grains} grains.")
    # results = {}
    full_node_dict = {}
    full_node_image_dict = {}
    img_base = np.zeros_like(image)
    # want to get each cropped image, use some anchor coords to match them onto the image,
    #   and compile all the grain images onto a single image
    all_images = {
        "grain": img_base.copy(),
        "smoothed_grain": img_base.copy(),
        "skeleton": img_base.copy(),
        "pruned_skeleton": img_base.copy(),
        "node_img": img_base.copy(),
        "ordered_traces": img_base.copy(),
        "fitted_traces": img_base.copy(),
        "visual": img_base.copy(),
    }
    all_traces = []

    for n_grain, (cropped_image, cropped_mask) in enumerate(zip(cropped_images, cropped_masks)):
        result, node_dict, node_image_dict, images, trace = trace_grain(
            cropped_image,
            cropped_mask,
            pixel_to_nm_scaling,
            skeletonisation_params,
            pruning_params,
            filename,
            min_skeleton_size,
            n_grain,
        )
        LOGGER.info(f"[{filename}] : Traced grain {n_grain + 1} of {n_grains}")
        full_node_dict[n_grain] = node_dict
        full_node_image_dict[n_grain] = node_image_dict
        # results[n_grain] = result
        try:
            pd_result = pd.DataFrame.from_dict(result, orient="index")
            grains_results = pd.concat([grains_results, pd_result])
        except NameError:  # if grain results not present
            grains_results = pd.DataFrame.from_dict(result, orient="index")
        except ValueError as error:
            LOGGER.error("No grains found in any images, consider adjusting your thresholds.")
            LOGGER.error(error)

        for key, value in all_images.items():
            crop = images[key]
            bbox = bboxs[n_grain]
            value[bbox[0] : bbox[2], bbox[1] : bbox[3]] += crop[pad_width:-pad_width, pad_width:-pad_width]

        for mol_trace in trace:
            all_traces.append(mol_trace + [bbox[0] - pad_width, bbox[1] - pad_width])

    print(grains_results)

    return grains_results, full_node_dict, full_node_image_dict, all_images, all_traces


def prep_arrays(image: np.ndarray, labelled_grains_mask: np.ndarray, pad_width: int) -> Tuple[list, list]:
    """Takes an image and labelled mask and crops individual grains and original heights to a list.

    Parameters
    ==========
    image: np.ndarray
        Gaussian filtered image. Typically filtered_image.images["gaussian_filtered"].
    labelled_grains_mask: np.ndarray
        2D Numpy array of labelled grain masks, with each mask being comprised solely of unique integer (not
    zero). Typically this will be output from grains.directions[<direction>["labelled_region_02].
    pad_width: int
        Cells by which to pad cropped regions by.

    Returns
    =======
    Tuple
        Returns a tuple of two lists, each consisting of cropped arrays.
    """
    # Get bounding boxes for each grain
    region_properties = skimage_measure.regionprops(labelled_grains_mask)
    # Subset image and grains then zip them up
    cropped_images = [crop_array(image, grain.bbox, 0) for grain in region_properties]
    cropped_images = [np.pad(grain, pad_width=pad_width) for grain in cropped_images]
    cropped_masks = [crop_array(labelled_grains_mask, grain.bbox, 0) for grain in region_properties]
    cropped_masks = [np.pad(grain, pad_width=pad_width) for grain in cropped_masks]
    # Flip every labelled region to be 1 instead of its label
    cropped_masks = [np.where(grain == 0, 0, 1) for grain in cropped_masks]
    # Get BBOX coords to remap crops to images
    bboxs = [np.array(grain.bbox) for grain in region_properties]

    return (cropped_images, cropped_masks, bboxs)


def trace_grain(
    cropped_image: np.ndarray,
    cropped_mask: np.ndarray,
    pixel_to_nm_scaling: float,
    skeletonisation_params: dict,
    pruning_params: dict,
    filename: str = None,
    min_skeleton_size: int = 10,
    n_grain: int = None,
) -> Dict:
    """Trace an individual grain.

    Tracing involves multiple steps...

    1. Skeletonisation
    2. Pruning of side branch artefacts from skeletonisation.
    3. Ordering of the skeleton.
    4. Determination of molecule shape.
    5. Jiggling/Fitting
    6. Splining to improve resolution of image.

    Pararmeters
    ===========
    cropped_image: np.ndarray
        Cropped array from the original image defined as the bounding box from the labelled mask.
    cropped_mask: np.ndarray
        Cropped array from the labelled image defined as the bounding box from the labelled mask. This should have been
        converted to a binary mask.
    filename: str
        File being processed
    pixel_to_nm_scaling: float
        Pixel to nm scaling.
    min_skeleton_size: int
        Minimum size of grain in pixels after skeletonisation.
    skeletonisation_params: dict
        Any parameters associated with the method of skeletonisation.
    pruning_params: dict
        Any parameters associated with the method of pruning.
    n_grain: int
        Grain number being processed.

    Returns
    =======
    Dictionary
        Dictionary of the contour length, whether the image is circular or linear, the end-to-end distance and an array
    of co-ordinates.
    """
    dnatrace = dnaTrace(
        image=cropped_image,
        grain=cropped_mask,
        skeletonisation_params=skeletonisation_params,
        pruning_params=pruning_params,
        filename=filename,
        pixel_to_nm_scaling=pixel_to_nm_scaling,
        min_skeleton_size=min_skeleton_size,
        n_grain=n_grain,
    )
    dnatrace.trace_dna()
    results = {}

    if dnatrace.num_mols == 0: # incase no mols could be traced
        results[0] = {
            "image": dnatrace.filename,
            "grain_number": n_grain,
            "contour_length": None,
            "circular": None,
            "end_to_end_distance": None,
            "num_crossings": None,
            "topology": None,
            "num_mols": None,
        }
    else:
        for i in range(dnatrace.num_mols):
            results[i] = {
                "image": dnatrace.filename,
                "grain_number": n_grain,
                "contour_length": dnatrace.contour_lengths[i],
                "new_contour_length": dnatrace.new_contour_length,
                "circular": dnatrace.mol_is_circulars[i],
                "end_to_end_distance": dnatrace.end_to_end_distances[i],
                "num_crossings": dnatrace.num_crossings,
                "grain_avg_crossing_confidence": dnatrace.avg_crossing_confidence,
                "topology": dnatrace.topology[i],
                "num_mols": dnatrace.num_mols,
            }

    images = {
        "image": dnatrace.image,
        "grain": dnatrace.grain,
        "smoothed_grain": dnatrace.smoothed_grain,
        "skeleton": dnatrace.skeleton,
        "pruned_skeleton": dnatrace.pruned_skeleton,
        "node_img": dnatrace.node_image,
        "ordered_traces": dnatrace.ordered_trace_img,
        "fitted_traces": dnatrace.fitted_trace_img,
        "visual": dnatrace.visuals,
    }
    return results, dnatrace.node_dict, dnatrace.node_image_dict, images, dnatrace.splined_traces


def crop_array(array: np.ndarray, bounding_box: tuple, pad_width: int = 0) -> np.ndarray:
    """Crop an array.

    Ideally we pad the array that is being cropped so that we have heights outside of the grains bounding box. However,
    in some cases, if an grain is near the edge of the image scan this results in requesting indexes outside of the
    existing image. In which case we get as much of the image padded as possible.

    Parameters
    ----------
    array: np.ndarray
        2D Numpy array to be cropped.
    bounding_box: Tuple
        Tuple of co-ordinates to crop, should be of form (min_row, min_col, max_row, max_col).
    pad_width: int
        Padding to apply to bounding box.

    Returns
    -------
    np.ndarray()
        Cropped array
    """
    bounding_box = list(bounding_box)
    # Top Row : Make this the first column if too close
    bounding_box[0] = 0 if bounding_box[0] - pad_width < 0 else bounding_box[0] - pad_width
    # Bottom Row : Make this the last row if too close
    bounding_box[2] = array.shape[0] if bounding_box[2] + pad_width > array.shape[0] else bounding_box[2] + pad_width
    # Left Column : Make this the first column if too close
    bounding_box[1] = 0 if bounding_box[1] - pad_width < 0 else bounding_box[1] - pad_width
    # Right Column : Make this the last column if too close
    bounding_box[3] = array.shape[1] if bounding_box[3] + pad_width > array.shape[1] else bounding_box[3] + pad_width
    return array[
        bounding_box[0] : bounding_box[2],
        bounding_box[1] : bounding_box[3],
    ]


# 2023-06-09 - Code that runs dnatracing in parallel across grains, left deliberately for use when we remodularise the
#              entry-points/workflow. Will require that the gaussian filtered array is saved and passed in along with
#              the labelled regions. @ns-rse
#
#
# if __name__ == "__main__":
#     cropped_images, cropped_masks = prep_arrays(image, grains_mask, pad_width)
#     n_grains = len(cropped_images)
#     LOGGER.info(f"[{filename}] : Calculating statistics for {n_grains} grains.")
#     # Process in parallel
#     with Pool(processes=cores) as pool:
#         results = {}
#         with tqdm(total=n_grains) as pbar:
#             x = 0
#             for result in pool.starmap(
#                 trace_grain,
#                 zip(
#                     cropped_images,
#                     cropped_masks,
#                     repeat(pixel_to_nm_scaling),
#                     repeat(filename),
#                     repeat(min_skeleton_size),
#                     repeat(skeletonisation_method),
#                 ),
#             ):
#                 LOGGER.info(f"[{filename}] : Traced grain {x + 1} of {n_grains}")
#                 results[x] = result
#                 x += 1
#                 pbar.update()
#     try:
#         results = pd.DataFrame.from_dict(results, orient="index")
#         results.index.name = "molecule_number"
#     except ValueError as error:
#         LOGGER.error("No grains found in any images, consider adjusting your thresholds.")
#         LOGGER.error(error)
#     return results


class nodeStats:
    """Class containing methods to find and analyse the nodes/crossings within a grain"""

    def __init__(
        self,
        filename,
        image: np.ndarray,
        grain: np.ndarray,
        smoothed_grain: np.ndarray,
        skeleton: np.ndarray,
        px_2_nm: float,
        n_grain: int,
    ) -> None:
        self.filename = filename
        self.image = image

        # np.savetxt(OUTPUT_DIR / "image.txt", image)
        # np.savetxt(OUTPUT_DIR / "hess.txt", self.hess)
        self.grain = grain
        self.smoothed_grain = smoothed_grain
        self.skeleton = skeleton
        self.px_2_nm = px_2_nm
        self.n_grain = n_grain

        sigma = (-3.5 / 3) * self.px_2_nm * 1e9 + 15.5 / 3
        self.hess = self.detect_ridges(self.image * 1e9, 4)
        # np.savetxt(OUTPUT_DIR / "hess.txt", self.hess)

        """
        a = np.zeros((100,100))
        a[21:80, 20] = 1
        a[21:80, 50] = 1
        a[21:80, 80] = 1
        a[20, 21:80] = 1
        a[80, 21:80] = 1
        a[20, 50] = 0
        a[80, 50] = 0
        self.grain = ndimage.binary_dilation(a, iterations=3)
        self.image = np.ones((100,100))
        self.skeleton = a
        self.px_2_nm = 1
        """
        """
        a = np.zeros((100,100))
        a[20:80, 50] = 1
        a[50, 20:80] = 1
        self.grain = ndimage.binary_dilation(a, iterations=3)
        self.image = np.ones((100,100))
        self.skeleton = a
        self.px_2_nm = 1
        """
        self.conv_skelly = np.zeros_like(self.skeleton)
        self.connected_nodes = np.zeros_like(self.skeleton)
        self.all_connected_nodes = self.skeleton.copy()
        self.whole_skel_graph = None

        self.node_centre_mask = None
        self.num_crossings = 0
        self.node_dict = {}
        self.node_image_dict = {}
        self.test = None
        self.test2 = None
        self.test3 = None
        self.test4 = None
        self.test5 = None
        self.full_dict = {}
        self.mol_coords = {}
        self.visuals = {}
        self.all_visuals_img = None

    def get_node_stats(self) -> tuple:
        """The workflow for obtaining the node statistics.

        Returns:
        --------
        dict
            Key structure:  <grain_number>
                            |-> <node_number>
                                |-> 'error'
                                |-> 'branch_stats'
                                    |-> <branch_number>
                                        |-> 'ordered_coords', 'heights', 'gaussian_fit', 'fwhm', 'angles'
                                |-> 'node_coords'
        dict
            Key structure <grain_number>
                            |-> 'node_area_image'
                            |-> 'node_area_grain'
                            |-> 'node_area_skeleton'
                            |-> 'node_branch_mask'
                            |-> 'node_avg_mask

        """
        LOGGER.info(f"Node Stats - Processing Grain: {self.n_grain}")
        self.conv_skelly = convolve_skelly(self.skeleton)
        if len(self.conv_skelly[self.conv_skelly == 3]) != 0:  # check if any nodes
            # convolve to see crossing and end points
            self.conv_skelly = self.tidy_branches(self.conv_skelly, self.image)
            # reset skeleton var as tidy branches may have modified it
            self.skeleton = np.where(self.conv_skelly != 0, 1, 0)
            # get graph of skeleton
            self.whole_skel_graph = self.skeleton_image_to_graph(self.skeleton)
            # connect the close nodes
            self.connected_nodes = self.connect_close_nodes(self.conv_skelly, node_width=7e-9)
            # connect the odd-branch nodes
            self.connected_nodes = self.connect_extended_nodes_nearest(
                self.connected_nodes, extend_dist=14e-9 / self.px_2_nm
            )
            # self.connected_nodes = self.connect_extended_nodes(self.connected_nodes)
            self.node_centre_mask = self.highlight_node_centres(self.connected_nodes)
            self.num_crossings = (self.node_centre_mask == 3).sum()
            self.analyse_nodes(max_branch_length=20e-9)
        return self.node_dict, self.node_image_dict
        # self.all_visuals_img = dnaTrace.concat_images_in_dict(self.image.shape, self.visuals)

    def check_node_errorless(self):
        for _, vals in self.node_dict.items():
            if vals["error"]:
                return False
            else:
                pass
        return True

    @staticmethod
    def skeleton_image_to_graph(skel):
        skeImPos = np.argwhere(skel).T
        g = nx.Graph()
        neigh = np.array([[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]])

        for idx in range(skeImPos[0].shape[0]):
            for neighIdx in range(neigh.shape[0]):
                curNeighPos = skeImPos[:, idx] + neigh[neighIdx]
                if np.any(curNeighPos < 0) or np.any(curNeighPos >= skel.shape):
                    continue
                if skel[curNeighPos[0], curNeighPos[1]] > 0:
                    idx_coord = skeImPos[0, idx], skeImPos[1, idx]
                    curNeigh_coord = curNeighPos[0], curNeighPos[1]
                    # assign lower weight to nodes if not a binary image
                    if skel[idx_coord] == 3 and skel[curNeigh_coord] == 3:
                        weight = 0
                    else:
                        weight = 1
                    g.add_edge(idx_coord, curNeigh_coord, weight=weight)
        g.graph["physicalPos"] = skeImPos.T
        return g

    @staticmethod
    def graph_to_skeleton_image(g, im_size):
        im = np.zeros(im_size)
        for node in g:
            im[node] = 1

        return im

    # TODO: Maybe move to skeletonisation
    def tidy_branches(self, connect_node_mask: np.ndarray, image: np.ndarray):
        """Aims to wrangle distant connected nodes back towards the main cluster. By reskeletonising
        soely the node areas.

        Parameters
        ----------
        connect_node_mask : np.ndarray
            The connected node mask - a skeleton where node regions = 3, endpoints = 2, and skeleton = 1.
        image : np.ndarray
            The intensity image.

        Returns
        -------
        np.ndarray
            The wrangled connected_node_mask.
        """
        new_skeleton = np.where(connect_node_mask != 0, 1, 0)
        labeled_nodes = label(np.where(connect_node_mask == 3, 1, 0))
        for node_num in range(1, labeled_nodes.max() + 1):
            solo_node = np.where(labeled_nodes == node_num, 1, 0)
            coords = np.argwhere(solo_node == 1)
            node_centre = coords.mean(axis=0).astype(np.int32)
            node_wid = coords[:, 0].max() - coords[:, 0].min() + 2  # +2 so always 2 by default
            node_len = coords[:, 1].max() - coords[:, 1].min() + 2  # +2 so always 2 by default
            overflow = int(10e-9 / self.px_2_nm) if int(10e-9 / self.px_2_nm) != 0 else 1
            # grain mask fill
            new_skeleton[
                node_centre[0] - node_wid // 2 - overflow : node_centre[0] + node_wid // 2 + overflow,
                node_centre[1] - node_len // 2 - overflow : node_centre[1] + node_len // 2 + overflow,
            ] = self.smoothed_grain[
                node_centre[0] - node_wid // 2 - overflow : node_centre[0] + node_wid // 2 + overflow,
                node_centre[1] - node_len // 2 - overflow : node_centre[1] + node_len // 2 + overflow,
            ]
        # remove any artifacts of thre grain caught in the overflow areas
        new_skeleton = self.keep_biggest_object(new_skeleton)
        # Re-skeletonise
        new_skeleton = getSkeleton(image, new_skeleton).get_skeleton(
            {"skeletonisation_method": "joe", "height_bias": 0.6}
        )
        new_skeleton = pruneSkeleton(image, new_skeleton).prune_skeleton({"pruning_method": "joe", "max_length": -1})
        new_skeleton = getSkeleton(image, new_skeleton).get_skeleton(
            {"skeletonisation_method": "zhang"}
        )  # cleanup around nibs
        # might also need to remove segments that have squares connected

        return convolve_skelly(new_skeleton)

    @staticmethod
    def keep_biggest_object(mask: np.ndarray) -> np.ndarray:
        """Keeps the largest object in a binary mask.

        Parameters
        ----------
        mask : np.ndarray
            Binary mask.

        Returns
        -------
        np.ndarray
            A binary mask with only one object.
        """
        labelled_mask = label(mask)
        idxs, counts = np.unique(mask, return_counts=True)
        try:
            max_idx = idxs[np.argmax(counts[1:]) + 1]
            return np.where(labelled_mask == max_idx, 1, 0)
        except ValueError as e:
            LOGGER.info(f"{e}: mask is empty.")
            return mask

    def connect_close_nodes(self, conv_skelly: np.ndarray, node_width: float = 2.85e-9) -> None:
        """Looks to see if nodes are within the node_width boundary (2.85nm) and thus
        should be connected, also labeling them as part of the same node.

        Parameters
        ----------
        node_width: float
            The width of the dna in the grain, used to connect close nodes.
        """
        self.connected_nodes = conv_skelly.copy()
        nodeless = conv_skelly.copy()
        nodeless[(nodeless == 3) | (nodeless == 2)] = 0  # remove node & termini points
        nodeless_labels = label(nodeless)
        for i in range(1, nodeless_labels.max() + 1):
            if nodeless[nodeless_labels == i].size < (node_width / self.px_2_nm):
                # maybe also need to select based on height? and also ensure small branches classified
                self.connected_nodes[nodeless_labels == i] = 3

        return self.connected_nodes

    def highlight_node_centres(self, mask):
        """Uses the provided mask to calculate the node centres based on
        height. These node centres are then re-plotted on the mask.

            bg = 0, skeleton = 1, endpoints = 2, node_centres = 3.
        """
        small_node_mask = mask.copy()
        small_node_mask[mask == 3] = 1  # remap nodes to skeleton
        big_nodes = mask.copy()
        big_nodes[mask != 3] = 0  # remove non-nodes
        big_nodes[mask == 3] = 1  # set nodes to 1
        big_node_mask = label(big_nodes)

        for i in np.delete(np.unique(big_node_mask), 0):  # get node indecies
            centre = np.unravel_index((self.image * (big_node_mask == i).astype(int)).argmax(), self.image.shape)
            small_node_mask[centre] = 3

        return small_node_mask

    def connect_extended_nodes(self, connected_nodes):
        just_nodes = connected_nodes.copy()
        just_nodes[(connected_nodes == 1) | (connected_nodes == 2)] = 0  # remove branches & termini points
        labelled = label(just_nodes)

        just_branches = connected_nodes.copy()
        just_branches[(connected_nodes == 3) | (connected_nodes == 2)] = 0  # remove node & termini points
        just_branches[connected_nodes == 1] = labelled.max() + 1
        labelled_branches = label(just_branches)

        def bounding_box(points):
            x_coordinates, y_coordinates = zip(*points)

            return [(min(x_coordinates), min(y_coordinates)), (max(x_coordinates), max(y_coordinates))]

        def do_sets_touch(set_A, set_B):
            # Iterate through coordinates in set_A and set_B
            for point_A in set_A:
                for point_B in set_B:
                    # Check if any coordinate in set_A is adjacent to any coordinate in set_B
                    if abs(point_A[0] - point_B[0]) <= 1 and abs(point_A[1] - point_B[1]) <= 1:
                        return True  # Sets touch
            return False  # Sets do not touch

        emanating_branches_by_node = {}  # Dictionary to store emanating branches for each label
        nodes_with_odd_branches = []  # List to store nodes with three branches

        for node in range(1, labelled.max() + 1):
            num_branches = 0
            bounding = bounding_box(np.argwhere(labelled == node))
            cropped_matrix = connected_nodes[
                bounding[0][0] - 1 : bounding[1][0] + 2, bounding[0][1] - 1 : bounding[1][1] + 2
            ]
            node_coords = np.argwhere(cropped_matrix == 3)
            branch_coords = np.argwhere(cropped_matrix == 1)
            for node_coord in node_coords:
                for branch_coord in branch_coords:
                    distance = math.dist(node_coord, branch_coord)
                    if distance <= math.sqrt(2):
                        num_branches = num_branches + 1
            # num_branches = len(np.argwhere(cropped_matrix == 1))

            if num_branches % 2 == 1:
                nodes_with_odd_branches.append(node)
                emanating_branches = []  # List to store emanating branches for the current label
                for branch in range(1, labelled_branches.max() + 1):
                    touching = do_sets_touch(np.argwhere(labelled_branches == branch), np.argwhere(labelled == node))
                    if touching:
                        emanating_branches.append(branch)
                    emanating_branches_by_node[node] = emanating_branches  # Store emanating branches for this label

        # Iterate through the nodes and their emanating branches
        for node1, branches1 in emanating_branches_by_node.items():
            for node2, branches2 in emanating_branches_by_node.items():
                if node1 != node2:  # Avoid comparing a node with itself
                    # Find the common branches between the lists
                    common_branches = set(branches1) & set(branches2)
                    if common_branches:
                        min_length = float("inf")  # Initialize with positive infinity
                        # Find the minimum length among all common branches
                        for shared_branch in common_branches:
                            length = len(np.argwhere(labelled_branches == shared_branch))
                            if length < min_length:
                                min_length = length
                        # Change the value to 3 only when len is minimal
                        for shared_branch in common_branches:
                            length = len(np.argwhere(labelled_branches == shared_branch))
                            if length == min_length:
                                connected_nodes[labelled_branches == shared_branch] = 3

        self.connected_nodes = connected_nodes
        return self.connected_nodes

    def connect_extended_nodes_nearest(self, connected_nodes, extend_dist=-1):
        just_nodes = np.where(connected_nodes == 3, 1, 0)  # remove branches & termini points
        labelled_nodes = label(just_nodes)

        just_branches = np.where(connected_nodes == 1, 1, 0)  # remove node & termini points
        just_branches[connected_nodes == 1] = labelled_nodes.max() + 1
        labelled_branches = label(just_branches)

        def bounding_box(points):
            x_coordinates, y_coordinates = zip(*points)
            return [(min(x_coordinates), min(y_coordinates)), (max(x_coordinates), max(y_coordinates))]

        def do_sets_touch(set_A, set_B):
            # Iterate through coordinates in set_A and set_B
            # TODO: instead of iterate, minus point from array and see if abs(diff) <= 2?
            for point_A in set_A:
                for point_B in set_B:
                    # Check if any coordinate in set_A is adjacent to any coordinate in set_B
                    if abs(point_A[0] - point_B[0]) <= 1 and abs(point_A[1] - point_B[1]) <= 1:
                        return True, point_A  # Sets touch
            return False, None  # Sets do not touch

        emanating_branch_starts_by_node = {}  # Dictionary to store emanating branches for each label
        nodes_with_odd_branches = []  # List to store nodes with three branches

        for node_num in range(1, labelled_nodes.max() + 1):
            num_branches = 0
            # makes lil box around node with 1 overflow
            bounding = bounding_box(np.argwhere(labelled_nodes == node_num))
            cropped_matrix = connected_nodes[
                bounding[0][0] - 1 : bounding[1][0] + 2, bounding[0][1] - 1 : bounding[1][1] + 2
            ]
            # get coords of nodes and branches in box
            node_coords = np.argwhere(cropped_matrix == 3)
            branch_coords = np.argwhere(cropped_matrix == 1)
            # iterate through node coords to see which are within 8 dirs
            for node_coord in node_coords:
                for branch_coord in branch_coords:
                    distance = math.dist(node_coord, branch_coord)
                    if distance <= math.sqrt(2):
                        num_branches = num_branches + 1

            # find the branch start point of odd branched nodes
            if num_branches % 2 == 1:
                nodes_with_odd_branches.append(node_num)
                emanating_branches = []  # List to store emanating branches for the current label
                for branch in range(1, labelled_branches.max() + 1):
                    # technically using labelled_branches when there's an end loop will only cause one
                    #   of the end loop coords to be captured. This shopuldn't matter as the other
                    #   label after the crossing should be closer to another node.
                    touching, branch_start = do_sets_touch(
                        np.argwhere(labelled_branches == branch), np.argwhere(labelled_nodes == node_num)
                    )
                    if touching:
                        emanating_branches.append(branch_start)
                    emanating_branch_starts_by_node[
                        node_num - 1
                    ] = emanating_branches  # Store emanating branches for this label
                    # assert len(emanating_branches) // 2 == 1

        if (
            len(emanating_branch_starts_by_node) <= 1
        ):  # only <1 odd branch so ignore pairing. will nx work with this and return no pairs anyway?
            return self.connected_nodes

        # Iterate through the nodes and their emanating branches
        shortest_node_dists = np.zeros(
            (len(emanating_branch_starts_by_node), len(emanating_branch_starts_by_node))
        )  # initialise the maximal pairing matrix
        shortest_dists_branch_idxs = np.zeros((shortest_node_dists.shape[0], shortest_node_dists.shape[0], 2)).astype(
            np.int64
        )
        shortest_dist_coords = np.zeros((shortest_node_dists.shape[0], shortest_node_dists.shape[0], 2, 2)).astype(
            np.int64
        )

        for i, (node1, branch_starts1) in enumerate(emanating_branch_starts_by_node.items()):
            for j, (node2, branch_starts2) in enumerate(emanating_branch_starts_by_node.items()):
                if node1 != node2:  # Avoid comparing a node with itself
                    # get shortest distance between all branch starts in n1 and n2, on #px not nm length
                    temp_length_matrix = np.zeros((len(branch_starts1), len(branch_starts2)))
                    for ii, bs1 in enumerate(branch_starts1):
                        for jj, bs2 in enumerate(branch_starts2):
                            temp_length_matrix[ii, jj] = nx.shortest_path_length(
                                self.whole_skel_graph, tuple(bs1), tuple(bs2)
                            )
                    # add shortest dist to shortest dist matrix
                    shortest_dist = np.min(temp_length_matrix)
                    shortest_node_dists[i, j] = shortest_dist
                    shortest_dists_branch_idxs[i, j] = np.argwhere(temp_length_matrix == shortest_dist)[0]
                    shortest_dist_coords[i, j] = (
                        branch_starts1[shortest_dists_branch_idxs[i, j][0]],
                        branch_starts2[shortest_dists_branch_idxs[i, j][1]],
                    )

        # get best matches
        matches = self.best_matches(shortest_node_dists, max_weight_matching=False)
        # get paths of best matches. TODO: replace below with using shortest dist coords
        for node_pair_idx in matches:
            shortest_dist = shortest_node_dists[node_pair_idx[0], node_pair_idx[1]]
            if shortest_dist <= extend_dist or extend_dist == -1:
                branch_idxs = shortest_dists_branch_idxs[node_pair_idx[0], node_pair_idx[1]]
                node_nums = list(emanating_branch_starts_by_node.keys())
                source = tuple(emanating_branch_starts_by_node[node_nums[node_pair_idx[0]]][branch_idxs[0]])
                target = tuple(emanating_branch_starts_by_node[node_nums[node_pair_idx[1]]][branch_idxs[1]])
                path = np.array(nx.shortest_path(self.whole_skel_graph, source, target))
                connected_nodes[path[:, 0], path[:, 1]] = 3

        self.connected_nodes = connected_nodes
        return self.connected_nodes

    @staticmethod
    def find_branch_starts(reduced_node_image: np.ndarray) -> np.ndarray:
        """Finds the corrdinates where the branches connect to the node region through binary dilation of the node.

        Parameters
        ----------
        reduced_node_image : np.ndarray
            A 2D numpy array containing a single node region (=3) and its connected branches (=1).

        Returns
        -------
        np.ndarray
            _description_
        """
        node = np.where(reduced_node_image == 3, 1, 0)
        nodeless = np.where(reduced_node_image == 1, 1, 0)
        thicc_node = binary_dilation(node, footprint=np.ones((3, 3)))

        return np.argwhere(thicc_node * nodeless == 1)

    def analyse_nodes(self, max_branch_length: float = 20e-9):
        """This function obtains the main analyses for the nodes of a single molecule. Within a certain box (nm) around the node.

        bg = 0, skeleton = 1, endpoints = 2, nodes = 3.

        Parameters:
        -----------
        max_branch_length: float
            The side length of the box around the node to analyse (in nm).

        """
        # get coordinates of nodes
        xy_arr = np.argwhere(self.node_centre_mask.copy() == 3)

        # check whether average trace resides inside the grain mask
        dilate = ndimage.binary_dilation(self.skeleton, iterations=2)
        average_trace_advised = dilate[self.smoothed_grain == 1].sum() == dilate.sum()
        LOGGER.info(f"[{self.filename}] : Branch height traces will be averaged: {average_trace_advised}")

        # iterate over the nodes to find areas
        # node_dict = {}
        matched_branches = None
        branch_img = None
        avg_img = None

        real_node_count = 0
        for node_no, (x, y) in enumerate(xy_arr):  # get centres
            # get area around node - might need to check if box lies on the edge
            """
            box_lims = self.get_box_lims(x, y, length, self.image)
            image_area = self.image[box_lims[0] : box_lims[1], box_lims[2] : box_lims[3]]
            hess_area = self.hess[box_lims[0] : box_lims[1], box_lims[2] : box_lims[3]]
            node_area = self.connected_nodes.copy()[box_lims[0] : box_lims[1], box_lims[2] : box_lims[3]]
            """
            max_length_px = max_branch_length / self.px_2_nm
            print("ALONG NODE: ", max_length_px, "px")
            image_slices = (
                x - int(max_length_px * 1.2),
                x + int(max_length_px * 1.2),
                y - int(max_length_px * 1.2),
                y + int(max_length_px * 1.2),
            )

            # reduce the skeleton area
            reduced_node_area = self._only_centre_branches(self.connected_nodes, (x, y))
            self.reduced_skel_graph = self.skeleton_image_to_graph(reduced_node_area)
            branch_mask = reduced_node_area.copy()

            branch_mask[branch_mask == 3] = 0
            branch_mask[branch_mask == 2] = 1
            node_coords = np.argwhere(reduced_node_area == 3)

            error = False  # to see if node too complex or region too small

            branch_start_coords = self.find_branch_starts(reduced_node_area)

            # stop processing if nib (node has 2 branches)
            if branch_start_coords.shape[0] <= 2:
                LOGGER.info(f"node {node_no} has only two branches - skipped & nodes removed")
                # sometimes removal of nibs can cause problems when re-indexing nodes
                LOGGER.info(f"{len(node_coords)} pixels in nib node")
                # TODO: node coords might be missaligned
                # self.node_centre_mask[node_coords[:, 0], node_coords[:, 0]] = 1  # remove these from node_centre_mask
                # self.connected_nodes[node_coords[:, 0], node_coords[:, 1]] = 1  # remove these from connected_nodes
            else:
                try:
                    # check wether resolution good enough to trace
                    res = self.px_2_nm <= 1000 / 512
                    if not res:
                        print(f"Resolution {res} is below suggested {1000 / 512}, node difficult to analyse.")
                        # raise ResolutionError
                    # elif x - length < 0 or y - length < 0 or x + length > self.image.shape[0] or y + length > self.image.shape[1]:
                    # LOGGER.info(f"Node lies too close to image boundary, increase 'pad_with' value.")
                    # raise ResolutionError

                    real_node_count += 1
                    print(f"Real node: {real_node_count}")
                    ordered_branches = []
                    vectors = []
                    nodeless = np.where(reduced_node_area == 1, 1, 0)
                    for i, branch_start_coord in enumerate(branch_start_coords):
                        # order branch
                        ordered = self.order_branch_from_start(
                            nodeless.copy(), branch_start_coord, max_length=max_length_px
                        )
                        # identify vector
                        vector = self.get_vector(ordered, branch_start_coord)  # [x, y]
                        # add to list
                        vectors.append(vector)
                        ordered_branches.append(ordered)
                    if node_no == 0:
                        self.test2 = vectors
                    # pair vectors
                    # print(f"NODE {real_node_count}, vectors:\n {vectors}")
                    pairs = self.pair_vectors(np.asarray(vectors))

                    # join matching branches through node
                    matched_branches = {}
                    masked_image = {}
                    branch_img = np.zeros_like(self.skeleton)  # initialising paired branch img
                    avg_img = np.zeros_like(self.skeleton)
                    for i, (branch_1, branch_2) in enumerate(pairs):
                        matched_branches[i] = {}
                        masked_image[i] = {}
                        # find close ends by rearranging branch coords
                        branch_1_coords, branch_2_coords = self.order_branches(
                            ordered_branches[branch_1], ordered_branches[branch_2]
                        )
                        # Get graphical shortest path between branch ends on the skeleton
                        crossing = nx.shortest_path(
                            self.reduced_skel_graph,
                            source=tuple(branch_1_coords[-1]),
                            target=tuple(branch_2_coords[0]),
                            weight="weight",
                        )
                        crossing = np.asarray(crossing[1:-1])  # remove start and end points & turn into array
                        # Branch coords and crossing
                        if crossing.shape == (0,):
                            branch_coords = np.vstack([branch_1_coords, branch_2_coords])
                        else:
                            branch_coords = np.vstack([branch_1_coords, crossing, branch_2_coords])
                        # make images of single branch joined and multiple branches joined
                        single_branch_img = np.zeros_like(self.skeleton)
                        single_branch_img[branch_coords[:, 0], branch_coords[:, 1]] = 1
                        single_branch_coords = self.order_branch(single_branch_img, [0, 0])
                        # calc image-wide coords
                        matched_branches[i]["ordered_coords"] = single_branch_coords
                        # get heights and trace distance of branch
                        try:
                            assert average_trace_advised
                            distances, heights, mask, _ = self.average_height_trace(
                                self.image, single_branch_img, single_branch_coords, [x, y]
                            )  # hess_area
                            masked_image[i]["avg_mask"] = mask
                        except (
                            AssertionError,
                            IndexError,
                        ) as e:  # Assertion - avg trace not advised, Index - wiggy branches
                            LOGGER.info(f"[{self.filename}] : avg trace failed with {e}, single trace only.")
                            average_trace_advised = False
                            distances = self.coord_dist_rad(single_branch_coords, [x, y])
                            # distances = self.coord_dist(single_branch_coords)
                            zero_dist = distances[
                                np.argmin(
                                    np.sqrt(
                                        (single_branch_coords[:, 0] - x) ** 2 + (single_branch_coords[:, 1] - y) ** 2
                                    )
                                )
                            ]
                            heights = self.image[single_branch_coords[:, 0], single_branch_coords[:, 1]]  # self.hess
                            distances = distances - zero_dist
                            distances, heights = self.average_uniques(
                                distances, heights
                            )  # needs to be paired with coord_dist_rad
                        matched_branches[i]["heights"] = heights
                        matched_branches[i]["distances"] = distances  # * self.px_2_nm
                        # identify over/under
                        matched_branches[i]["fwhm2"] = self.fwhm2(
                            heights, distances
                        )  # self.peak_height(heights, distances)

                    # redo fwhms after to get better baselines + same hm matching
                    hms = []
                    for branch_idx, values in matched_branches.items():  # get hms
                        hms.append(values["fwhm2"][1][2])
                    for branch_idx, values in matched_branches.items():  # use same highest hm
                        fwhm2 = self.fwhm2(values["heights"], values["distances"], hm=max(hms))
                        matched_branches[branch_idx]["fwhm2"] = fwhm2

                    # get AUC instead
                    """
                    xmaxs = []
                    xmins = []
                    for branch_idx, values in matched_branches.items():  # get hms
                        xmaxs.append(np.max(values["distances"]))
                        xmins.append(np.min(values["distances"]))
                    for branch_idx, values in matched_branches.items():  # use same highest hm
                        print("IMG MEAN: ", self.image[self.skeleton==1].mean())
                        fwhm2 = self.auc(values["distances"], values["heights"], y_lim=self.image[self.skeleton==1].mean(), xrange=(max(xmins), min(xmaxs))) # self.image[self.skeleton==1].mean()
                        matched_branches[branch_idx]["fwhm2"] = fwhm2
                    """

                    # get confidences
                    crossing_quants = []
                    for branch_idx, values in matched_branches.items():
                        crossing_quants.append(values["fwhm2"][0])
                    if len(crossing_quants) == 1: # from 3 eminnating branches
                        conf = None
                    else:
                        combs = self.get_two_combinations(crossing_quants)
                        conf = self.cross_confidence(combs)

                    # add paired and unpaired branches to image plot
                    fwhms = []
                    for branch_idx, values in matched_branches.items():
                        fwhms.append(values["fwhm2"][0])
                    branch_idx_order = np.array(list(matched_branches.keys()))[np.argsort(np.array(fwhms))]
                    # branch_idx_order = np.arange(0,len(matched_branches)) #uncomment to unorder (will not unorder the height traces)

                    for i, branch_idx in enumerate(branch_idx_order):
                        branch_coords = matched_branches[branch_idx]["ordered_coords"]
                        branch_img[branch_coords[:, 0], branch_coords[:, 1]] = i + 1  # add to branch img
                        if average_trace_advised:  # add avg traces
                            avg_img[masked_image[branch_idx]["avg_mask"] != 0] = i + 1
                        else:
                            avg_img = None

                    unpaired_branches = np.delete(np.arange(0, branch_start_coords.shape[0]), pairs.flatten())
                    LOGGER.info(f"Unpaired branches: {unpaired_branches}")
                    branch_label = branch_img.max()
                    for i in unpaired_branches:  # carries on from loop variable i
                        branch_label += 1
                        branch_img[ordered_branches[i][:, 0], ordered_branches[i][:, 1]] = branch_label

                    if node_no == 0:
                        self.test3 = avg_img

                    # calc crossing angle
                    # get full branch vectors
                    vectors = []
                    for branch_no, values in matched_branches.items():
                        vectors.append(self.get_vector(values["ordered_coords"], [x, y]))
                    # calc angles to first vector i.e. first should always be 0
                    angles = self.calc_angles(np.asarray(vectors))[0]
                    for i, angle in enumerate(angles):
                        matched_branches[i]["angles"] = angle

                    if node_no == 0:
                        self.test4 = vectors
                        self.test5 = angles
                    """
                    except ValueError:
                        LOGGER.error(f"Node {node_no} too complex, see images for details.")
                        error = True
                    """
                except ResolutionError:
                    LOGGER.info(f"Node stats skipped as resolution too low: {self.px_2_nm}nm per pixel")
                    error = True

                if average_trace_advised:
                    avg_img = avg_img  # [image_slices[0] : image_slices[1], image_slices[2] : image_slices[3]]

                print("Error: ", error)
                self.node_dict[real_node_count] = {
                    "error": error,
                    "px_2_nm": self.px_2_nm,
                    "crossing_type": None,
                    "branch_stats": matched_branches,
                    "node_coords": node_coords,
                    "confidence": conf,
                }

                self.node_image_dict[real_node_count] = {
                    "node_area_image": self.image,  # [
                    # image_slices[0] : image_slices[1], image_slices[2] : image_slices[3]
                    # ],  # self.hess
                    "node_area_grain": self.grain,  # [
                    # image_slices[0] : image_slices[1], image_slices[2] : image_slices[3]
                    # ],
                    "node_area_skeleton": reduced_node_area,  # [
                    # image_slices[0] : image_slices[1], image_slices[2] : image_slices[3]
                    # ],
                    "node_branch_mask": branch_img,  # [
                    # image_slices[0] : image_slices[1], image_slices[2] : image_slices[3]
                    # ],
                    "node_avg_mask": avg_img,
                }
            self.all_connected_nodes[self.connected_nodes != 0] = self.connected_nodes[self.connected_nodes != 0]

    @staticmethod
    def sq(curr, nxt, y_lim):
        "Obtains the area of the square between y_lim and the 2 points"
        lowest_y = min(curr[1], nxt[1])
        y_diff = max(0, lowest_y - y_lim)
        return (nxt[0] - curr[0]) * y_diff

    def tri(self, curr, nxt, y_lim):
        "Obtains the area of the triangle between y_lim and the 2 points"
        if y_lim <= min(curr[1], nxt[1]):  # if below triangle
            return (nxt[0] - curr[0]) * (abs(nxt[1] - curr[1])) / 2
        if y_lim >= max(curr[1], nxt[1]):  # if val above triangle
            return 0
        else:  # linearly interpolate
            y_lim_x = self.lin_interp(curr, nxt, yvalue=y_lim)
            if curr[1] > nxt[1]:  # which coord has higher y
                max_coord = curr
            else:
                max_coord = nxt
            return abs(curr[0] - y_lim_x) * (max_coord[1] - y_lim) / 2

    def auc(self, x, y, y_lim, xrange):
        """Get the area under the curve (or points in this case) by summing the
        trapezoid areas between points above the y_lim and between the xrange."""
        xcap = x[(x >= xrange[0]) & (x <= xrange[1])]
        ycap = y[(x >= xrange[0]) & (x <= xrange[1])]
        auc = 0
        for i in range(len(xcap) - 1):
            curr = xcap[i], ycap[i]
            nxt = xcap[i + 1], ycap[i + 1]
            square = self.sq(curr, nxt, y_lim)
            triangle = self.tri(curr, nxt, y_lim)
            auc += square + triangle

        return auc, xcap, ycap, y_lim

    @staticmethod
    def get_two_combinations(fwhm_list):
        """Gets all paired combinations of values in the list."""
        combs = []
        for i in range(len(fwhm_list) - 1):
            [combs.append([fwhm_list[i], j]) for j in fwhm_list[i + 1 :]]
        return combs

    def cross_confidence(self, combs):
        """Obtains the average confidence of the combinations using the function provided."""
        c = 0
        for comb in combs:
            c += self.recip(comb)
        return c / len(combs)

    @staticmethod
    def recip(vals):
        try:
            if min(vals) == 0:  # means fwhm variation hasn't worked
                return 0
            else:
                return 1 - min(vals) / max(vals)
        except ZeroDivisionError:
            return 0

    @staticmethod
    def per_diff(vals):
        try:
            if min(vals) == 0:  # means fwhm variation hasn't worked
                return 0
            else:
                return abs(vals[0] - vals[1]) / (vals[0] + vals[1])
        except ZeroDivisionError:
            return 0

    @staticmethod
    def detect_ridges(gray, sigma=1.0):
        H_elems = hessian_matrix(gray, sigma=sigma, order="rc")
        hess = hessian_matrix_eigvals(H_elems)[1]  # 1 is min eigenvalues
        # Normalise
        hess += -1 * hess.min()  # min = 0
        hess = hess * -1 + hess.max()  # reflect in y
        return hess

    @staticmethod
    def get_box_lims(x: int, y: int, length: int, image: np.ndarray) -> tuple:
        """Gets the box limits of length around x and y. If the length exceeds the limits of the original
            box, these will be set to the limits of the original image.

        Parameters
        ----------
        x : int
            X coordinate.
        y : int
            Y coordinate.
        length : int
            Side length of the box.

        Returns
        -------
        tuple
            The x, and y limits of the box
        """
        box = np.array([x - length, x + length + 1, y - length, y + length + 1])
        box[box < 0] = 0

        if box[1] > image.shape[0]:
            box[1] = image.shape[0] - 1
        if box[3] > image.shape[1]:
            box[3] = image.shape[1] - 1

        return box

    def order_branch(self, binary_image: np.ndarray, anchor: list):
        """Orders a linear branch by identifing an endpoint, and looking at the local area of the point to find the next.

        Parameters
        ----------
        binary_image: np.ndarray
            A binary image of a skeleton segment to order it's points.
        anchor: list
            A list of 2 integers representing the coordinate to order the branch from the endpoint closest to this.

        Returns
        -------
        np.ndarray
            An array of ordered cordinates.
        """
        mask = binary_image.copy()

        if len(np.argwhere(mask == 1)) < 3:  # if < 3 coords just return them
            return np.argwhere(mask == 1)

        # get branch starts
        endpoints_highlight = ndimage.convolve(mask, np.ones((3, 3)))
        endpoints_highlight[mask == 0] = 0
        endpoints = np.argwhere(endpoints_highlight == 2)
        if len(endpoints) != 0:  # if any endpoints, start closest to anchor
            dist_vals = abs(endpoints - anchor).sum(axis=1)
            start = endpoints[np.argmin(dist_vals)]
        else:  # will be circular so pick the first coord (is this always the case?)
            start = np.argwhere(mask == 1)[0]
        # order the points according to what is nearby
        ordered = self.order_branch_from_start(mask, start)

        return np.array(ordered)

    def order_branch_from_start(self, nodeless, start, max_length=np.inf):
        dist = 0
        # add starting point to ordered array
        ordered = []
        ordered.append(start)
        nodeless[start[0], start[1]] = 0  # remove from array

        # iterate to order the rest of the points
        current_point = ordered[-1]  # get last point
        area, _ = self.local_area_sum(nodeless, current_point)  # look at local area
        local_next_point = np.argwhere(
            area.reshape(
                (
                    3,
                    3,
                )
            )
            == 1
        ) - (1, 1)
        dist += np.sqrt(2) if abs(local_next_point).sum() > 1 else 1

        while len(local_next_point) != 0 and dist <= max_length:
            next_point = (current_point + local_next_point)[0]
            # find where to go next
            ordered.append(next_point)
            nodeless[next_point[0], next_point[1]] = 0  # set value to zero
            current_point = ordered[-1]  # get last point
            area, _ = self.local_area_sum(nodeless, current_point)  # look at local area
            local_next_point = np.argwhere(
                area.reshape(
                    (
                        3,
                        3,
                    )
                )
                == 1
            ) - (1, 1)
            dist += np.sqrt(2) if abs(local_next_point).sum() > 1 else 1

        return np.array(ordered)

    @staticmethod
    def local_area_sum(binary_map, point):
        """Evaluates the local area around a point in a binary map.

        Parameters
        ----------
        binary_map: np.ndarray
            A binary array of an image.
        point: Union[list, touple, np.ndarray]
            A single object containing 2 integers relating to a point within the binary_map

        Returns
        -------
        np.ndarray
            An array values of the local coordinates around the point.
        int
            A value corresponding to the number of neighbours around the point in the binary_map.
        """
        x, y = point
        local_pixels = binary_map[x - 1 : x + 2, y - 1 : y + 2].flatten()
        local_pixels[4] = 0  # ensure centre is 0
        return local_pixels, local_pixels.sum()

    @staticmethod
    def get_vector(coords, origin):
        """Calculate the normalised vector of the coordinate means in a branch"""
        vector = coords.mean(axis=0) - origin
        norm = np.sqrt(vector @ vector)
        vector = vector if norm == 0 else vector / norm  # normalise vector so length=1
        return vector

    @staticmethod
    def calc_angles(vectors: np.ndarray):
        """Calculates the angles between vectors in an array.
        Uses the formula: cos(theta) = |a|•|b|/|a||b|

        Parameters
        ----------
        vectors: np.ndarray
            Array of 2x1 vectors.

        Returns
        -------
        np.ndarray
            An array of the cosine of the angles between the vectors.
        """
        dot = vectors @ vectors.T
        norm = np.diag(dot) ** 0.5
        cos_angles = dot / (norm.reshape(-1, 1) @ norm.reshape(1, -1))
        angles = abs(np.arccos(cos_angles) / np.pi * 180)
        return angles

    def pair_vectors(self, vectors: np.ndarray):
        """Takes a list of vectors and pairs them based on the angle between them

        Parameters
        ----------
        vectors: np.ndarray
            Array of 2x1 vectors to be paired.

        Returns:
        --------
        np.ndarray
            An array of the matching pair indicies.
        """
        # calculate cosine of angle
        angles = self.calc_angles(vectors)
        # find highest values
        np.fill_diagonal(angles, 0)  # ensures not paired with itself
        # match angles
        return self.best_matches(angles)

    def best_matches(self, arr: np.ndarray, max_weight_matching=True) -> np.ndarray:
        """Turns a matrix into a graph and calulates the best matching index pairs."""
        if max_weight_matching:
            G = self.create_weighted_graph(arr)
            matching = np.array(list(nx.max_weight_matching(G, maxcardinality=True)))
        else:
            np.fill_diagonal(arr, arr.max() + 1)
            G = self.create_weighted_graph(arr)
            matching = np.array(list(nx.min_weight_matching(G)))
        return matching

    @staticmethod
    def create_weighted_graph(matrix: np.ndarray) -> nx.Graph:
        """From a square matrix of weights (i,j), creates a bipartite graph connecting i <-> j.

        Parameters
        ----------
        matrix : np.ndarray
            Square array of weights between rows and columns.

        Returns
        -------
        nx.Graph
            Bipatrite graph
        """
        n = len(matrix)
        G = nx.Graph()
        for i in range(n):
            for j in range(i + 1, n):
                G.add_edge(i, j, weight=matrix[i, j])
        return G

    @staticmethod
    def pair_angles(angles: np.ndarray) -> list:
        """Pairs angles that are 180 degrees to eachother and removes them before selecting the next pair.

        Parameters
         ----------
         angles : np.ndarray
             Square array (i,j) of angles between i and j.

         Returns
         -------
         list
             A list of paired indexes in a list.
        """
        angles_cp = angles.copy()
        pairs = []
        for _ in range(int(angles.shape[0] / 2)):
            pair = np.unravel_index(np.argmax(angles_cp), angles.shape)
            pairs.append(pair)  # add to list
            angles_cp[[pair]] = 0  # set rows 0 to avoid picking again
            angles_cp[:, [pair]] = 0  # set cols 0 to avoid picking again

        return np.asarray(pairs)

    @staticmethod
    def gaussian(x: np.ndarray, h: float, mean: float, sigma: float):
        """The gaussian function.

        Parameters
        ----------
        h: float
            The peak height of the gaussian.
        x: np.ndarray
            X values to be passed into the gaussian.
        mean: float
            The mean of the x values.
        sigma: float
            The standard deviation of the image.

        Returns
        -------
        np.ndarray
            The y-values of the gaussian performed on the x values.
        """
        return h * np.exp(-((x - mean) ** 2) / (2 * sigma**2))

    def fwhm(self, heights: np.ndarray, distances: np.ndarray) -> tuple:
        """Fits a gaussian to the branch heights, and calculates the FWHM.

        Parameters:
        -----------
        heights: np.ndarray
            Array of height values.
        distances: np.ndarray
            array of distance values.

        Returns:
        --------
        tuple
            A tuple of the FWHM value, and x,y matrix from curve fitting.

        """
        mean = 45.5  # hard coded as middle node value
        sigma = 1 / (200 / self.px_2_nm)  # 1nm / px2nm = px  half a nm as either side of std
        popt, pcov = optimize.curve_fit(
            self.gaussian,
            distances,
            heights - heights.min(),
            p0=[max(heights) - heights.min(), mean, sigma],
            maxfev=8000,
        )

        return 2.3548 * popt[2], popt  # 2*(2ln2)^1/2 * sigma = FWHM

    def fwhm2(self, heights: np.ndarray, distances: np.ndarray, hm: Union[float, None] = None) -> tuple:
        """A second function to caculate the FWHM value by identifying the HM then finding the closest values
        in the distances array and using linear interpolation to calculate the FWHM.

        Parameters
        ----------
        heights : np.ndarray
            Array of heights
        distances : np.ndarray
            Array of distances
        hm : Union[None, float], optional
            The halfmax value to match (if wanting the same HM between curves), by default None

        Returns
        -------
        tuple
            The FWHM value, [distance at hm for 1st half of trace, distance at hm for 2nd half of trace,
            HM value], [index of the highest point, distance at highest point, height at highest point]

        """
        centre_fraction = int(len(heights) * 0.2)  # incase zone approaches another node, look around centre for max
        if centre_fraction == 0:
            high_idx = np.argmax(heights)
        else:
            high_idx = np.argmax(heights[centre_fraction:-centre_fraction]) + centre_fraction

        # heights_norm = heights.copy() - heights.min()  # lower graph so min is 0

        # get array halves to find first points that cross hm
        arr1 = heights[:high_idx][::-1]  # heights_norm[:high_idx][::-1]
        dist1 = distances[:high_idx][::-1]
        arr2 = heights[high_idx:]  # heights_norm[high_idx:]
        dist2 = distances[high_idx:]

        arr1_hm = 0
        arr2_hm = 0

        if hm is None:
            # Get half max
            hm = (
                heights.max() - heights.min()
            ) / 2 + heights.min()  # heights_norm.max() / 2  # half max value -> try to make it the same as other crossing branch?
            # increase make hm = lowest of peak if it doesn't hit one side
            if np.min(arr1) > hm:
                arr1_local_min = argrelextrema(arr1, np.less)[-1]  # closest to end
                try:
                    hm = arr1[arr1_local_min][0]
                except IndexError:  # index error when no local minima
                    hm = np.min(arr1)
            elif np.min(arr2) > hm:
                arr2_local_min = argrelextrema(arr2, np.less)[0]  # closest to start
                try:
                    hm = arr2[arr2_local_min][0]
                except IndexError:  # index error when no local minima
                    hm = np.min(arr2)

        for i in range(len(arr1) - 1):
            if (arr1[i] >= hm) and (arr1[i + 1] <= hm):  # if points cross through the hm value
                arr1_hm = self.lin_interp([dist1[i], arr1[i]], [dist1[i + 1], arr1[i + 1]], yvalue=hm)
                break

        for i in range(len(arr2) - 1):
            if (arr2[i] >= hm) and (arr2[i + 1] <= hm):  # if points cross through the hm value
                arr2_hm = self.lin_interp([dist2[i], arr2[i]], [dist2[i + 1], arr2[i + 1]], yvalue=hm)
                break

        fwhm = abs(arr2_hm - arr1_hm)

        return fwhm, [arr1_hm, arr2_hm, hm], [high_idx, distances[high_idx], heights[high_idx]]

    def peak_height(self, heights: np.ndarray, distances: np.ndarray, hm=None):
        # find low index between centre fraction (should be centre index)
        centre_fraction = int(len(heights) * 0.2)  # incase zone approaches another node, look around centre for min
        # if centre_fraction == 0:
        #    centre_fraction = 1
        low_idx = np.argmin(abs(distances))

        arr1 = heights[:low_idx][::-1]
        dist1 = distances[:low_idx][::-1]
        arr2 = heights[low_idx:]
        dist2 = distances[low_idx:]

        min_heights = []
        min_height_dists = []

        # obtain distances of peak on each side
        for arr, dist in zip([arr1, arr2], [dist1, dist2]):
            try:
                arr_local_min = argrelextrema(arr, np.less)[0][0]  # closest to start
                min_heights.append(arr[arr_local_min])
                min_height_dists.append(dist[arr_local_min])
            except IndexError:  # no minima, get average
                min_heights.append(arr.mean())
                min_height_dists.append(dist[np.argmin(abs(arr - arr.mean()))])

        # make outputs same as fwhm2
        hm = np.min([np.min(arr1), np.min(arr2)])

        return (
            np.mean([min_heights[1], min_heights[0]]),
            [min_height_dists[0], min_height_dists[1], heights[low_idx]],
            [low_idx, distances[low_idx], heights[low_idx]],
        )

    @staticmethod
    def lin_interp(point_1: list, point_2: list, xvalue: Union[float, None] = None, yvalue: Union[float, None] = None):
        """Linear interp 2 points by finding line equation and subbing.

        Parameters
        ----------
        point_1 : list
            List of an x and y coordinate
        point_2 : list
            List of an x and y coordinate
        xvalue : Union[float, None], optional
            Value at which to interpolate to get a y coordinate, by default None
        yvalue : Union[float, None], optional
            Value at which to interpolate to get an x coordinate, by default None

        Returns
        -------
        _type_
            _description_
        """
        m = (point_1[1] - point_2[1]) / (point_1[0] - point_2[0])
        c = point_1[1] - (m * point_1[0])
        if xvalue is not None:
            interp_y = m * xvalue + c
            return interp_y
        if yvalue is not None:
            interp_x = (yvalue - c) / m
            return interp_x

    @staticmethod
    def close_coords(endpoints1: np.ndarray, endpoints2: np.ndarray) -> tuple:
        """Finds the closest coordinates between arrays of two endpoint coordinates.

        Parameters
        ----------
        endpoints1 : np.ndarray
            Array of 2 coordinates.
        endpoints2 : np.ndarray
            Array of 2 coordinates.

        Returns
        -------
        tuple
            The closest two coordinates - one from each array.
        """
        sum1 = abs(endpoints1 - endpoints2).sum(axis=1)
        sum2 = abs(endpoints1[::-1] - endpoints2).sum(axis=1)
        if sum1.min() < sum2.min():
            min_idx = np.argmin(sum1)
            return endpoints1[min_idx], endpoints2[min_idx]
        else:
            min_idx = np.argmin(sum2)
            return endpoints1[::-1][min_idx], endpoints2[min_idx]

    @staticmethod
    def order_branches(branch1: np.ndarray, branch2: np.ndarray) -> tuple:
        """Order the two ordered arrays based on the closest endpoint coordinates.

        Parameters
        ----------
        branch1 : np.ndarray
            An Nx2 array describing coordinates.
        branch2 : np.ndarray
            An Nx2 array describing coordinates.

        Returns
        -------
        tuple
            An tuple with the each coordinate array ordered to follow on from one-another.
        """
        endpoints1 = np.asarray([branch1[0], branch1[-1]])
        endpoints2 = np.asarray([branch2[0], branch2[-1]])
        sum1 = abs(endpoints1 - endpoints2).sum(axis=1)
        sum2 = abs(endpoints1[::-1] - endpoints2).sum(axis=1)
        if sum1.min() < sum2.min():
            if np.argmin(sum1) == 0:
                return branch1[::-1], branch2
            else:
                return branch1, branch2[::-1]
        else:
            if np.argmin(sum2) == 0:
                return branch1, branch2
            else:
                return branch1[::-1], branch2[::-1]

    @staticmethod
    def binary_line(start: np.ndarray, end: np.ndarray) -> np.ndarray:
        """Creates a binary path following the straight line between 2 points.

        Parameters
        ----------
        start : np.ndarray
            A coordinate.
        end : np.ndarray
            Another coordinate.

        Returns
        -------
        np.ndarray
            An Nx2 coordinate array that the line passes thorugh.
        """
        arr = []
        m_swap = False
        x_swap = False
        slope = (end - start)[1] / (end - start)[0]

        if abs(slope) > 1:  # swap x and y if slope will cause skips
            start, end = start[::-1], end[::-1]
            slope = 1 / slope
            m_swap = True

        if start[0] > end[0]:  # swap x coords if coords wrong way arround
            start, end = end, start
            x_swap = True

        # code assumes slope < 1 hence swap
        x_start, y_start = start
        x_end, y_end = end
        for x in range(x_start, x_end + 1):
            y_true = slope * (x - x_start) + y_start
            y_pixel = np.round(y_true)
            arr.append([x, y_pixel])

        if m_swap:  # if swapped due to slope, return
            arr = np.asarray(arr)[:, [1, 0]].reshape(-1, 2).astype(int)
            if x_swap:
                return arr[::-1]
            else:
                return arr
        else:
            arr = np.asarray(arr).reshape(-1, 2).astype(int)
            if x_swap:
                return arr[::-1]
            else:
                return arr

    @staticmethod
    def coord_dist(coords: np.ndarray, px_2_nm: float = 1) -> np.ndarray:
        """Takes a list/array of coordinates (Nx2) and produces an array which
        accumulates a real distance as if traversing from pixel to pixel.

        Parameters
        ----------
        coords: np.ndarray
            A Nx2 integer array corresponding to the ordered coordinates of a binary trace.
        px_2_nm: float
            The pixel to nanometer scaling factor.

        Returns
        -------
        np.ndarray
            An array of length N containing thcumulative sum of the distances.
        """
        dist_list = [0]
        dist = 0
        for i in range(len(coords) - 1):
            if abs(coords[i] - coords[i + 1]).sum() == 2:
                dist += 2**0.5
            else:
                dist += 1
            dist_list.append(dist)
        return np.asarray(dist_list) * px_2_nm

    @staticmethod
    def coord_dist_rad(coords: np.ndarray, centre: np.ndarray, px_2_nm: float = 1) -> np.ndarray:
        """Calculates the distance from the node centre to a point along the branch, rather than
        through the path taken. This also averages any common distace values and makes those in
        the trace before the node index negitive.

        Parameters
        ----------
        coords : np.ndarray
            Nx2 array of branch coordinates
        centre : np.ndarray
            A 1x2 array of the centre coordinates to identify a 0 point for the node
        px_2_nm : float, optional
            The pixel to nanometer scaling factor to provide real units, by default 1

        Returns
        -------
        np.ndarray
            A Nx1 array of the distance from the node centre.
        """
        diff_coords = coords - centre
        if np.all(coords == centre, axis=1).sum() == 0:  # if centre not in coords, reassign centre
            diff_dists = np.sqrt(diff_coords[:, 0] ** 2 + diff_coords[:, 1] ** 2)
            centre = coords[np.argmin(diff_dists)]
        cross_idx = np.argwhere(np.all(coords == centre, axis=1))
        rad_dist = np.sqrt(diff_coords[:, 0] ** 2 + diff_coords[:, 1] ** 2)
        rad_dist[0 : cross_idx[0][0]] *= -1
        return rad_dist

    @staticmethod
    def above_below_value_idx(array, value):
        """Finds index of the points neighbouring the value in an array."""
        idx1 = abs(array - value).argmin()
        try:
            if value < array[idx1 + 1] and array[idx1] < value:
                idx2 = idx1 + 1
            elif value < array[idx1] and array[idx1 - 1] < value:
                idx2 = idx1 - 1
            else:
                raise IndexError  # this will be if the number is the same
            indices = [idx1, idx2]
            indices.sort()
            return indices
        except IndexError:
            return None

    def average_height_trace(
        self, img: np.ndarray, branch_mask: np.ndarray, branch_coords: np.ndarray, centre=[0, 0]
    ) -> tuple:
        """Dilates the original branch to create two additional side-by-side branches
        in order to get a more accurate average of the height traces. This function produces
        the common distances between these 3 branches, and their averaged heights.

        Parameters
        ----------
        img: np.ndarray
            An array of numbers pertaining to an image.
        branch_mask: np.ndarray
            A binary array of the branch, must share the same dimensions as the image.
        centre: Union[float, None]
            The coordinates to centre the branch around.

        Returns
        -------
        tuple
            A tuple of the averaged heights from the linetrace and their corresponding distances
            from the crossing.
        """
        # get heights and dists of the original (middle) branch
        branch_dist = self.coord_dist_rad(branch_coords, centre)
        # branch_dist = self.coord_dist(branch_coords)
        branch_heights = img[branch_coords[:, 0], branch_coords[:, 1]]
        branch_dist, branch_heights = self.average_uniques(
            branch_dist, branch_heights
        )  # needs to be paired with coord_dist_rad
        dist_zero_point = branch_dist[
            np.argmin(np.sqrt((branch_coords[:, 0] - centre[0]) ** 2 + (branch_coords[:, 1] - centre[1]) ** 2))
        ]
        branch_dist_norm = branch_dist - dist_zero_point  # - 0  # branch_dist[branch_heights.argmax()]

        # want to get a 3 pixel line trace, one on each side of orig
        dilate = ndimage.binary_dilation(branch_mask, iterations=1)
        dilate = self.fill_holes(dilate)
        dilate_minus = np.where(dilate != branch_mask, 1, 0)
        dilate2 = ndimage.binary_dilation(dilate)
        dilate2[(dilate == 1) | (branch_mask == 1)] = 0
        labels = label(dilate2)
        # Cleanup stages - re-entering, early terminating, closer traces
        #   if parallel trace out and back in zone, can get > 2 labels
        labels = self._remove_re_entering_branches(labels, remaining_branches=2)
        #   if parallel trace doesn't exit window, can get 1 label
        #       occurs when skeleton has poor connections (extra branches which cut corners)
        if labels.max() == 1:
            conv = convolve_skelly(branch_mask)
            endpoints = np.argwhere(conv == 2)
            for endpoint in endpoints:  # may be >1 endpoint
                para_trace_coords = np.argwhere(labels == 1)
                abs_diff = np.absolute(para_trace_coords - endpoint).sum(axis=1)
                min_idxs = np.where(abs_diff == abs_diff.min())
                trace_coords_remove = para_trace_coords[min_idxs]
                labels[trace_coords_remove[:, 0], trace_coords_remove[:, 1]] = 0
            labels = label(labels)
        #   reduce binary dilation distance
        paralell = np.zeros_like(branch_mask).astype(np.int32)
        for i in range(1, labels.max() + 1):
            single = labels.copy()
            single[single != i] = 0
            single[single == i] = 1
            sing_dil = ndimage.binary_dilation(single)
            paralell[(sing_dil == dilate_minus) & (sing_dil == 1)] = i
        labels = paralell.copy()

        binary = labels.copy()
        binary[binary != 0] = 1
        binary += branch_mask

        # get and order coords, then get heights and distances relitive to node centre / highest point
        heights = []
        distances = []
        for i in np.unique(labels)[1:]:
            trace_img = np.where(labels == i, 1, 0)
            trace_img = getSkeleton(img, trace_img).get_skeleton({"skeletonisation_method": "zhang"})
            trace = self.order_branch(trace_img, branch_coords[0])
            height_trace = img[trace[:, 0], trace[:, 1]]
            dist = self.coord_dist_rad(trace, centre)  # self.coord_dist(trace)
            dist, height_trace = self.average_uniques(dist, height_trace)  # needs to be paired with coord_dist_rad
            heights.append(height_trace)
            distances.append(
                dist - dist_zero_point  # - 0
            )  # branch_dist[branch_heights.argmax()]) #dist[central_heights.argmax()])
        # Make like coord system using original branch
        avg1 = []
        avg2 = []
        for mid_dist in branch_dist_norm:
            for i, (distance, height) in enumerate(zip(distances, heights)):
                # check if distance already in traces array
                if (mid_dist == distance).any():
                    idx = np.where(mid_dist == distance)
                    if i == 0:
                        avg1.append([mid_dist, height[idx][0]])
                    else:
                        avg2.append([mid_dist, height[idx][0]])
                # if not, linearly interpolate the mid-branch value
                else:
                    # get index after and before the mid branches' x coord
                    xidxs = self.above_below_value_idx(distance, mid_dist)
                    if xidxs is None:
                        pass  # if indexes outside of range, pass
                    else:
                        point1 = [distance[xidxs[0]], height[xidxs[0]]]
                        point2 = [distance[xidxs[1]], height[xidxs[1]]]
                        y = self.lin_interp(point1, point2, xvalue=mid_dist)
                        if i == 0:
                            avg1.append([mid_dist, y])
                        else:
                            avg2.append([mid_dist, y])
        avg1 = np.asarray(avg1)
        avg2 = np.asarray(avg2)
        # ensure arrays are same length to average
        temp_x = branch_dist_norm[np.isin(branch_dist_norm, avg1[:, 0])]
        common_dists = avg2[:, 0][np.isin(avg2[:, 0], temp_x)]

        common_avg_branch_heights = branch_heights[np.isin(branch_dist_norm, common_dists)]
        common_avg1_heights = avg1[:, 1][np.isin(avg1[:, 0], common_dists)]
        common_avg2_heights = avg2[:, 1][np.isin(avg2[:, 0], common_dists)]

        average_heights = (common_avg_branch_heights + common_avg1_heights + common_avg2_heights) / 3
        return (
            common_dists,
            average_heights,
            binary,
            [[heights[0], branch_heights, heights[1]], [distances[0], branch_dist_norm, distances[1]]],
        )

    @staticmethod
    def fill_holes(mask: np.ndarray) -> np.ndarray:
        """Fills all holes within a binary mask."""
        inv_mask = np.where(mask != 0, 0, 1)
        lbl_inv = label(inv_mask, connectivity=1)
        idxs, counts = np.unique(lbl_inv, return_counts=True)
        max_idx = idxs[np.argmax(counts)]
        return np.where(lbl_inv != max_idx, 1, 0)

    @staticmethod
    def _remove_re_entering_branches(image: np.ndarray, remaining_branches: int = 1) -> np.ndarray:
        """Looks to see if branches exit and re-enter the viewing area, then removes one-by-one
        the smallest, so that only <remaining_branches> remain.
        """
        rtn_image = image.copy()
        binary_image = image.copy()
        binary_image[binary_image != 0] = 1
        labels = label(binary_image)

        if labels.max() > remaining_branches:
            lens = [labels[labels == i].size for i in range(1, labels.max() + 1)]
            while len(lens) > remaining_branches:
                smallest_idx = min(enumerate(lens), key=lambda x: x[1])[0]
                rtn_image[labels == smallest_idx + 1] = 0
                lens.remove(min(lens))

        return rtn_image

    @staticmethod
    def _only_centre_branches(node_image: np.ndarray, node_coordinate):
        """Looks identifies the node being examined and removes all
        branches not connected to it.

        Parameters
        ----------
        node_image : np.ndarray
            An image of the skeletonised area surrounding the node where
            the background = 0, skeleton = 1, termini = 2, nodes = 3.

        Returns
        -------
        np.ndarray
            The initial node image but only with skeletal branches
            connected to the middle node.
        """
        node_image_cp = node_image.copy()

        # get node-only image
        nodes = node_image_cp.copy()
        nodes[nodes != 3] = 0
        labeled_nodes = label(nodes)

        # find which cluster is closest to the centre
        centre = node_coordinate
        node_coords = np.argwhere(nodes == 3)
        min_coords = node_coords[abs(node_coords - centre).sum(axis=1).argmin()]
        centre_idx = labeled_nodes[min_coords[0], min_coords[1]]

        # get nodeless image
        nodeless = node_image_cp.copy()
        nodeless[nodeless == 3] = 0
        nodeless[nodeless == 2] = 1  # if termini, need this in the labeled branches too
        nodeless[labeled_nodes == centre_idx] = 1  # return centre node
        labeled_nodeless = label(nodeless)

        # apply to return image
        for i in range(1, labeled_nodeless.max() + 1):
            if (node_image_cp[labeled_nodeless == i] == 3).any():
                node_image_cp[labeled_nodeless != i] = 0
                break

        # remove small area around other nodes
        labeled_nodes[labeled_nodes == centre_idx] = 0
        non_central_node_coords = np.argwhere(labeled_nodes != 0)
        for coord in non_central_node_coords:
            for j, coord_val in enumerate(coord):
                if coord_val - 1 < 0:
                    coord[j] = 1
                if coord_val + 2 > node_image_cp.shape[j]:
                    coord[j] = node_image_cp.shape[j] - 2
            node_image_cp[coord[0] - 1 : coord[0] + 2, coord[1] - 1 : coord[1] + 2] = 0

        return node_image_cp

    @staticmethod
    def average_uniques(arr1, arr2):
        "Takes two arrays, gets the uniques of both with the average of common values in the second."
        arr1_uniq, index = np.unique(arr1, return_index=True)
        arr2_new = np.zeros_like(arr1_uniq).astype(np.float64)
        for i, val in enumerate(arr1[index]):
            mean = arr2[arr1 == val].mean()
            arr2_new[i] += mean

        return arr1[index], arr2_new

    def compile_trace(self):
        """This function uses the branches and FWHM's identified in the node_stats dictionary to create a
        continious trace of the molecule.
        """
        LOGGER.info(f"[{self.filename}] : Compiling the trace.")

        # iterate throught the dict to get branch coords, heights and fwhms
        node_coords = []
        crossing_coords = []
        crossing_heights = []
        crossing_distances = []
        fwhms = []
        for _, stats in self.node_dict.items():
            temp_nodes = []
            temp_coords = []
            temp__heights = []
            temp_distances = []
            temp_fwhms = []
            for _, branch_stats in stats["branch_stats"].items():
                temp_coords.append(branch_stats["ordered_coords"])
                temp__heights.append(branch_stats["heights"])
                temp_distances.append(branch_stats["distances"])
                temp_fwhms.append(branch_stats["fwhm2"][0])
                temp_nodes.append(stats["node_coords"])
            node_coords.append(temp_nodes)
            crossing_coords.append(temp_coords)
            crossing_heights.append(temp__heights)
            crossing_distances.append(temp_distances)
            fwhms.append(temp_fwhms)

        # Get the image minus the crossing regions
        minus = self.skeleton.copy()
        for crossings in crossing_coords:
            for crossing in crossings:
                minus[crossing[:, 0], crossing[:, 1]] = 0
        minus = label(minus)

        # Get both image
        both = minus.copy()
        for node_num, crossings in enumerate(crossing_coords):
            for crossing_num, crossing in enumerate(crossings):
                both[crossing[:, 0], crossing[:, 1]] = node_num + crossing_num + minus.max()

        # np.savetxt(OUTPUT_DIR / "both.txt", both)

        # setup z array
        z = []

        # order minus segments
        ordered = []
        for i in range(1, minus.max() + 1):
            arr = np.where(minus, minus == i, 0)
            ordered.append(self.order_branch(arr, [0, 0]))  # orientated later
            z.append(0)

        # add crossing coords to ordered segment list
        for i, node_crossing_coords in enumerate(crossing_coords):
            z_idx = np.argsort(fwhms[i])
            z_idx[z_idx == 0] = -1
            for j, single_cross in enumerate(node_crossing_coords):
                # check current single cross has no duplicate coords with ordered, except crossing points
                uncommon_single_cross = np.array(single_cross).copy()
                for coords in ordered:
                    uncommon_single_cross = self.remove_common_values(
                        uncommon_single_cross, np.array(coords), retain=node_coords[i][j]
                    )
                if len(uncommon_single_cross) > 0:
                    ordered.append(uncommon_single_cross)
                z.append(z_idx[j])

        # get an image of each ordered segment
        cross_add = np.zeros_like(self.image)
        for i, coords in enumerate(ordered):
            single_cross_img = dnaTrace.coords_2_img(np.array(coords), cross_add)
            cross_add[single_cross_img != 0] = i + 1

        # np.savetxt(OUTPUT_DIR / "cross_add.txt", cross_add)
        LOGGER.info(f"[{self.filename}] Getting coordinate trace")

        coord_trace, simple_trace = self.simple_xyz_trace(ordered, cross_add, z, n=100)

        # TODO: Finish / do triv identification via maybe:
        #   one segment 2 branches
        #   topology based -> 0_1 all triv, 2^2_1 -> 2 real (may have to check against no. nodes to see which are real)
        self.identify_trivial_crossings(node_coords)

        im = np.zeros_like(self.skeleton)
        for i in coord_trace:
            im[i[:, 0], i[:, 1]] = 1
        # np.savetxt(OUTPUT_DIR / "trace.txt", coord_trace[0])

        # visual over under img
        visual = self.get_visual_img(coord_trace, fwhms, crossing_coords)

        # np.savetxt(OUTPUT_DIR / "visual.txt", visual)

        # I could use the traced coords, remove the node centre coords, and re-label segments
        #   following 1, 2, 3... around the mol which should look like the Planar Diagram formation
        #   (https://topoly.cent.uw.edu.pl/dictionary.html#codes). Then look at each corssing zone again,
        #   determine which in in-undergoing and assign labels counter-clockwise

        # print("Getting PD Codes:")
        topology = self.get_topology(simple_trace)
        """
        if len(coord_trace) <= 2:
            topology = self.get_pds(coord_trace, node_centre_coords, fwhms, crossing_coords)
            
        else:
            topology = None
        """
        return coord_trace, visual, topology

    def get_minus_img(self, node_area_box, node_centre_coords):
        minus = self.skeleton.copy()
        for i, area in enumerate(node_area_box):
            x, y = node_centre_coords[i]
            area = np.array(area) // 2
            minus[x - area[0] : x + area[0], y - area[1] : y + area[1]] = 0
        return label(minus)

    @staticmethod
    def remove_common_values(arr1, arr2, retain=[]):
        # Convert the arrays to sets for faster common value lookup
        set_arr2 = {tuple(row) for row in arr2}
        set_retain = {tuple(row) for row in retain}
        # Create a new filtered list while maintaining the order of the first array
        filtered_arr1 = []
        for coord in arr1:
            tup_coord = tuple(coord)
            if tup_coord not in set_arr2 or tup_coord in set_retain:
                filtered_arr1.append(coord)

        return np.asarray(filtered_arr1)

    def simple_xyz_trace(self, ordered_segment_coords, both_img, zs, n=100):
        """Obtains a trace and simplified trace of the molecule by following connected segments."""
        np.save(OUTPUT_DIR / "both.txt", both_img)
        mol_coords = []
        simple_coords = []
        remaining = both_img.copy().astype(np.int32)
        endpoints = np.unique(remaining[convolve_skelly(remaining) == 2])  # uniq incase of whole mol
        n_points_p_seg = (n - 2 * remaining.max()) // remaining.max()

        while remaining.max() != 0:
            # select endpoint to start if there is one
            endpoints = [i for i in endpoints if i in np.unique(remaining)]  # remove if removed from remaining
            if endpoints:
                coord_idx = endpoints[0] - 1
            else:  # if no endpoints, just a loop
                coord_idx = np.unique(remaining)[1] - 1  # avoid choosing 0
            coord_trace = np.empty((0, 2)).astype(np.int32)
            simple_trace = np.empty((0, 3)).astype(np.int32)
            while coord_idx > -1:  # either cycled through all or hits terminus -> all will be just background
                remaining[remaining == coord_idx + 1] = 0
                trace_segment = self.get_trace_segment(remaining, ordered_segment_coords, coord_idx)
                if len(coord_trace) > 0:  # can only order when there's a reference point / segment
                    trace_segment = self.remove_duplicates(
                        trace_segment, prev_segment
                    )  # remove overlaps in trace (may be more efficient to do it on the prev segment)
                    trace_segment = self.order_from_end(coord_trace[-1], trace_segment)
                prev_segment = trace_segment.copy()  # update prev segement
                coord_trace = np.append(coord_trace, trace_segment.astype(np.int32), axis=0)

                simple_trace_temp = self.reduce_rows(
                    trace_segment.astype(np.int32), n=n_points_p_seg
                )  # reducing rows here ensures no segments are skipped
                simple_trace_temp_z = np.column_stack(
                    (simple_trace_temp, np.ones((len(simple_trace_temp), 1)) * zs[coord_idx])
                ).astype(
                    np.int32
                )  # add z's
                simple_trace = np.append(simple_trace, simple_trace_temp_z, axis=0)

                x, y = coord_trace[-1]
                coord_idx = remaining[x - 1 : x + 2, y - 1 : y + 2].max() - 1  # should only be one value

            mol_coords.append(coord_trace)

            # Issue in 0_5 where wrong nxyz[0] selected, and == nxyz[-1] so always duplicated
            nxyz = np.column_stack((np.arange(0, len(simple_trace)), simple_trace))
            if len(nxyz) > 2 and (nxyz[0][1] - nxyz[-1][1]) ** 2 + (nxyz[0][2] - nxyz[-1][2]) ** 2 <= 2:
                # single coord traces mean nxyz[0]==[1] so cause issues when duplicating for topoly
                print("Looped so duplicating index 0: ", nxyz[0])
                nxyz = np.append(nxyz, nxyz[0][np.newaxis, :], axis=0)
            simple_coords.append(nxyz)

        simple_coords = self.comb_xyzs(simple_coords)

        print(f"Mols in trace: {len(mol_coords)}")

        return mol_coords, simple_coords

    def identify_trivial_crossings(self, crossings):
        # TODO
        for i, cross in enumerate(crossings):
            self.node_dict[i + 1]["crossing_type"] = "real"

    @staticmethod
    def reduce_rows(array, n=300):
        # removes reduces the number of rows (but keeping the first and last ones)
        if array.shape[0] < n or array.shape[0] < 4:
            return array
        else:
            idxs_to_keep = np.unique(np.linspace(0, array[1:-1].shape[0] - 1, n).astype(np.int32))
            new_array = array[1:-1][idxs_to_keep]
            new_array = np.append(array[0][np.newaxis, :], new_array, axis=0)
            new_array = np.append(new_array, array[-1][np.newaxis, :], axis=0)
            return new_array

    @staticmethod
    def get_trace_segment(remaining_img, ordered_segment_coords, coord_idx):
        """Check the branch of given index to see if it contains an endpoint. If it does,
        the segment coordinates will be returned starting from the endpoint.

        Parameters
        ----------
        remaining_img : np.ndarray
            A 2D array representing an image composed of connected segments of different integers.
        ordered_segment_coords : list
            A list of 2xN coordinates representing each segment
        idx : _type_
            The index of the current segment to look at. There is an index mismatch between the
            remaining_img and ordered_segment_coords by -1.
        """
        start_xy = ordered_segment_coords[coord_idx][0]
        start_max = remaining_img[start_xy[0] - 1 : start_xy[0] + 2, start_xy[1] - 1 : start_xy[1] + 2].max() - 1
        if start_max == -1:
            return ordered_segment_coords[coord_idx]  # start is endpoint
        else:
            return ordered_segment_coords[coord_idx][::-1]  # end is endpoint

    @staticmethod
    def comb_xyzs(nxyz):
        """Appends each mol trace array to a list as a list for use with Topoly"""
        total = []
        for mol in nxyz:
            temp = []
            for row in mol:
                temp.append(list(row))
            total.append(temp)
        return total

    def get_topology(self, nxyz):
        # Topoly doesn't work when 2 mols don't actually cross
        topology = []
        lin_idxs = []
        nxyz_cp = nxyz.copy()
        # remove linear mols as are just reidmiester moves
        for i, mol_trace in enumerate(nxyz):
            if mol_trace[-1][0] != 0:  # mol is not circular
                topology.append("linear")
                lin_idxs.append(i)
        # remove from list in reverse order so no conflicts
        lin_idxs.sort(reverse=True)
        for i in lin_idxs:
            del nxyz_cp[i]
        # classify topology for non-reidmeister moves
        if len(nxyz_cp) != 0:
            try:
                pd = translate_code(
                    nxyz_cp, output_type="pdcode"
                )  # pd code helps prevents freezing and spawning multiple processes
                LOGGER.info(f"{self.filename} : PD Code is: {pd}")
                top_class = jones(pd)
            except (IndexError, KeyError):
                LOGGER.info(f"{self.filename} : PD Code could not be obtained from trace coordinates.")
                top_class = "N/A"
            [
                topology.append(top_class) for i in range(len(nxyz_cp))
            ]  # don't separate U's - used for distribution comparison

        return topology

    @staticmethod
    def remove_duplicates(current_segment, prev_segment):
        # Convert arrays to tuples
        curr_segment_tuples = [tuple(row) for row in current_segment]
        prev_segment_tuples = [tuple(row) for row in prev_segment]
        # Find unique rows
        unique_rows = list(set(curr_segment_tuples) - set(prev_segment_tuples))
        # Remove duplicate rows from array1
        filtered_curr_array = np.array([row for row in curr_segment_tuples if tuple(row) in unique_rows])

        return filtered_curr_array

    @staticmethod
    def order_from_end(last_segment_coord, current_segment):
        """Orders the current segment coordinated to follow from the end of the previous one.

        Parameters
        ----------
        last_segment_coord : np.ndarray
            X and Y coordinates of the end of the last segment.
        current_segment : np.ndarray
            A 2xN array of coordinates of the current segment to order.
        """
        start_xy = current_segment[0]
        dist = np.sum((start_xy - last_segment_coord) ** 2) ** 0.5
        if dist <= np.sqrt(2):
            return current_segment
        else:
            return current_segment[::-1]

    @staticmethod
    def get_trace_idxs(fwhms: list) -> tuple:
        # node fwhms can be a list of different lengths so cannot use np arrays
        under_idxs = []
        over_idxs = []
        for node_fwhms in fwhms:
            order = np.argsort(node_fwhms)
            under_idxs.append(order[0])
            over_idxs.append(order[-1])
        return under_idxs, over_idxs

    def get_visual_img(self, coord_trace, fwhms, crossing_coords):
        # put down traces
        img = np.zeros_like(self.skeleton)
        for mol_no, coords in enumerate(coord_trace):
            temp_img = np.zeros_like(img)
            temp_img[coords[:, 0], coords[:, 1]] = 1
            temp_img = binary_dilation(temp_img)
            img[temp_img != 0] = 1  # mol_no + 1

        # np.savetxt(OUTPUT_DIR / "preimg.txt", img)

        lower_idxs, upper_idxs = self.get_trace_idxs(fwhms)

        if False:  # len(coord_trace) > 1:
            # plots seperate mols
            for type_idxs in [lower_idxs, upper_idxs]:
                for node_crossing_coords, type_idx in zip(crossing_coords, type_idxs):
                    temp_img = np.zeros_like(img)
                    cross_coords = node_crossing_coords[type_idx]
                    # decide which val
                    matching_coords = np.array([])
                    for trace in coord_trace:
                        c = 0
                        # get overlaps between segment coords and crossing under coords
                        for cross_coord in cross_coords:
                            c += ((trace == cross_coord).sum(axis=1) == 2).sum()
                        matching_coords = np.append(matching_coords, c)
                    val = matching_coords.argmax() + 1
                    temp_img[cross_coords[:, 0], cross_coords[:, 1]] = 1
                    temp_img = binary_dilation(temp_img)
                    img[temp_img != 0] = val

        else:
            # plots over/unders
            for i, type_idxs in enumerate([lower_idxs, upper_idxs]):
                for crossing, type_idx in zip(crossing_coords, type_idxs):
                    temp_img = np.zeros_like(img)
                    cross_coords = crossing[type_idx]
                    # decide which val
                    matching_coords = np.array([])
                    c = 0
                    # get overlaps between segment coords and crossing under coords
                    for cross_coord in cross_coords:
                        c += ((coord_trace[0] == cross_coord).sum(axis=1) == 2).sum()
                    matching_coords = np.append(matching_coords, c)
                    val = matching_coords.argmax() + 1
                    temp_img[cross_coords[:, 0], cross_coords[:, 1]] = 1
                    temp_img = binary_dilation(temp_img)
                    img[temp_img != 0] = i + 2

        return img

    def make_pd_skeleton(self, trace_coords, node_centres):
        """Makes a labeled skeleton where each skeletal segment upto a crossing follows on
        from the previous label number.

        Parameters
        ----------
        trace_coords : np.ndarray
            Coordinates of the ordered traces, may contain multiple molecules.
        node_centres : np.ndarray
            Coordinates of the crossing point centres.

        Returns
        -------
        np.ndarray
            A labeled skeleton image represented as the Planar Diargram format.
        """
        pd_image = np.zeros_like(self.skeleton)
        c = 0
        for mol_trace in trace_coords:
            pd_segments = []
            initial_idx = pd_image.max() + 1
            trace_node_idxs = np.array([0]).astype(np.int32)
            for x, y in node_centres:
                try:  # might hit a node from one mol that isn't between them both i.e. self crossing
                    # might also have the node removed upon re-skeletonising & not be found here
                    dists = np.sqrt((mol_trace[:, 0] - x) ** 2 + (mol_trace[:, 1] - y) ** 2)
                    if np.min(dists) <= np.sqrt(2):
                        x, y = mol_trace[dists.argmin()]
                    else:
                        raise ValueError
                    trace_node_idxs = np.append(
                        trace_node_idxs, np.argwhere((mol_trace[:, 0] == x) & (mol_trace[:, 1] == y))
                    )
                except ValueError:
                    print(f"Node at {x, y} not in mol {c} trace.")
                    pass
            trace_node_idxs.sort()
            # for the 3 branches, it will hit nothing at all
            if len(trace_node_idxs) == 0:
                trace_node_idxs = np.append(len(mol_trace))

            # pd_segments = [mol_trace[:trace_node_idxs[1]]]
            for i in range(len(trace_node_idxs) - 1):
                pd_segments.append(mol_trace[trace_node_idxs[i] : trace_node_idxs[i + 1]])
            last = mol_trace[trace_node_idxs[-1] :]  # assuming looped mol so last == start idx

            # check pd image in spyder to see the problem again
            for coords in pd_segments:
                pd_image[coords[:, 0], coords[:, 1]] = pd_image.max() + 1
            pd_image[last[:, 0], last[:, 1]] = initial_idx
            c += 1
        return pd_image

    def get_pds(self, trace_coords, node_centres, fwhms, crossing_coords):
        # find idxs of branches from start
        pd_code = ""
        pd_vals = []
        pd_img = self.make_pd_skeleton(trace_coords, node_centres)
        # np.savetxt(OUTPUT_DIR / "pd_img.txt", pd_img)
        node_centres = np.array([np.array(node_centre) for node_centre in node_centres])
        # pd_img[node_centres[:,0], node_centres[:,1]] = 0
        under_branch_idxs, _ = self.get_trace_idxs(fwhms)

        for i, (x, y) in enumerate(node_centres):
            matching_coords = np.array([])
            # get node area
            node_area = pd_img[x - 2 : x + 3, y - 2 : y + 3]
            # print("CENTRE #: ", i)
            # print("AREA: ", node_area)
            # get overlaps between crossing coords and
            pd_idx_in_area = np.unique(node_area)
            pd_idx_in_area = pd_idx_in_area[pd_idx_in_area != 0]
            # get overlaps between segment coords and crossing under coords
            for pd_idx in pd_idx_in_area:
                c = 0
                for ordered_branch_coord in crossing_coords[i][under_branch_idxs[i]]:
                    c += ((np.stack(np.where(pd_img == pd_idx)).T == ordered_branch_coord).sum(axis=1) == 2).sum()
                matching_coords = np.append(matching_coords, c)
                # print(f"Segment: {pd_idx}, Matches: {c}")
            highest_count_labels = [pd_idx_in_area[i] for i in np.argsort(matching_coords)[-2:]]
            # print("COUNT: ", highest_count_labels)
            if (
                len(highest_count_labels) > 1
            ):  # why are there single labels and therefore only one branch in the first place?
                if (
                    abs(highest_count_labels[0] - highest_count_labels[1]) > 1
                ):  # if in/out is loop return (assumes matched branch)
                    under_in = max(highest_count_labels)  # set under-in to larger value
                else:
                    under_in = min(highest_count_labels)  # otherwise set to lower value
            else:
                under_in = highest_count_labels[0]
            # print(f"Under-in: {under_in}")

            # get the values PD image around the crossing area
            anti_clock = self.vals_anticlock(node_area, under_in)
            if len(anti_clock) != len(np.unique(anti_clock)):
                self.node_dict[i + 1]["crossing_type"] = "trivial"
                print("TRIV: ", anti_clock)
            else:
                self.node_dict[i + 1]["crossing_type"] = "real"
            pd_vals.append(list(anti_clock))
        # for len 2's in

        # compile pd values into string and calc topology
        pd_code = self.make_pd_string(pd_vals)
        # pd_code = "X[1, 7, 6, 2];X[7, 10, 8, 1];X[5, 2, 3, 6];X[9, 4, 10, 5];X[3, 4, 8, 9]"
        print(f"Total PD code: {pd_code}")

        # filter out the trash PD codes and obtain topology
        try:
            # PD code may need to be reduced first for column check to work (as it fp on poke + slide)
            # pd_code = reduce_structure(pd_code, output_type='pdcode')
            no_triv_pd = self.remove_trivial_crossings(np.array(pd_vals))
            assert self.check_uniq_vals_valid(no_triv_pd)
            assert self.check_no_duplicates_in_columns(no_triv_pd)  # may break when 1-4, 1-6 (Node-Branch)
            if len(no_triv_pd) == 2:
                assert (no_triv_pd[0] == no_triv_pd[1, ::-1]).all() or (
                    no_triv_pd[0] == no_triv_pd[1, [1, 0, 3, 2]]
                ).all()
            # assert self.no_tripple_cross(no_triv_pd)
            topology = homfly(pd_code, closure=params.Closure.CLOSED, chiral=False)
        except AssertionError as e:  # triggers on same value in columns
            print(f"{e} : PD Code is nonsense (might be ok but pokes and slides not accounted for in removal).")
            topology = None
        except IndexError:  # triggers on index error in topoly
            print("PD Code is nonsense.")
            topology = None
        except:
            print("PD Code could not be deciphered.")
            topology = None
        print(f"Topology: {topology}")

        return topology

    @staticmethod
    def make_pd_string(pd_code):
        """Creates a PD code from a list of numeric lists."""
        new_pd_code = ""
        for node_crossing_pd in pd_code:
            new_pd_code += f"X{node_crossing_pd};"
        return new_pd_code[:-1]  # removes final ';'

    @staticmethod
    def vals_anticlock(area, start_lbl):
        """Gets the first occurance of values around the edges of an array in an anti-clockwise direction from the start point.

        Parameters
        ----------
        area : np.ndarray
            The labeled image array you want to observe around
        start_lbl : int
            The value to start the anti-clockwise labeling from. Must be an value on the edge of the area array.

        Returns
        -------
        np.ndarray
            An array of the labeled area values in an anti-clockwise direction from the startpoint.
        """
        top = area[0, :-1][::-1]
        _, top_args = np.unique(top, return_index=True)
        top_vals = top[top_args.sort()]
        left = area[1:, 0]
        _, left_args = np.unique(left, return_index=True)
        left_vals = left[left_args.sort()]
        bottom = area[-1, 1:]
        _, bottom_args = np.unique(bottom, return_index=True)
        bottom_vals = bottom[bottom_args.sort()]
        right = area[:-1, -1][::-1]
        _, right_args = np.unique(right, return_index=True)
        right_vals = right[right_args.sort()]

        total = np.concatenate(
            [
                top_vals[top_vals != 0],
                left_vals[left_vals != 0],
                bottom_vals[bottom_vals != 0],
                right_vals[right_vals != 0],
            ]
        )
        start_idx = np.where(total == start_lbl)[0]

        return np.roll(total, -start_idx)

    @staticmethod
    def remove_trivial_crossings(pd_values: np.ndarray):
        # removes trivial crossings from the pd_values not in place.
        #   assumes the loop doesn't cross anything else
        new_pd = pd_values.copy()
        for i, row in enumerate(pd_values):
            if len(row) != len(np.unique(row)):
                new_pd = np.delete(new_pd, i, axis=0)
        return new_pd

    @staticmethod
    def check_uniq_vals_valid(pd_vals):
        # checks (for real nodes) that there are not more unique pd_vals than physically possible
        # False if fails check, True is good
        flat = pd_vals.flatten()
        if len(np.unique(flat)) > 2 * len(pd_vals):
            return False
        return True

    @staticmethod
    def check_no_duplicates_in_columns(array):
        # checks if duplicate values exist within a column in an array
        # False fails check, True is good
        for col_no in range(array.shape[1]):
            if len(array[:, col_no]) != len(np.unique(array[:, col_no])):
                return False
        return True

    @staticmethod
    def no_tripple_cross(array):
        # checks we don't have a tripple crossing (as I can't seperate these out into 3 X's)
        # False fails check, True is good
        for row in array:
            if len(row) == 5:
                return True
        return False
