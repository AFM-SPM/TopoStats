"""Perform DNA Tracing."""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage, interpolate as interp
from skimage.morphology import label
from skimage.filters import gaussian, threshold_otsu
import skimage.measure as skimage_measure

from topostats.logs.logs import LOGGER_NAME
from topostats.tracing.nodestats import nodeStats
from topostats.tracing.skeletonize import getSkeleton
from topostats.tracing.pruning import pruneSkeleton
from topostats.tracing.tracingfuncs import genTracingFuncs, reorderTrace
from topostats.utils import coords_2_img


LOGGER = logging.getLogger(LOGGER_NAME)


class dnaTrace:
    """
    Calculates traces for a DNA molecule and calculates statistics from those traces.

    2023-06-09 : This class has undergone some refactoring so that it works with a single grain. The `trace_grain()`
    helper function runs the class and returns the expected statistics whilst the `trace_image()` function handles
    processing all detected grains within an image. The original methods of skeletonisation are available along with
    additional methods from scikit-image.

    Some bugs have been identified and corrected see commits for further details...

    236750b2
    2a79c4ff

    Parameters
    ----------
    image : npt.NDArray
        Cropped image, typically padded beyond the bounding box.
    grain : npt.NDArray
        Labelled mask for the grain, typically padded beyond the bounding box.
    filename : str
        Filename being processed.
    pixel_to_nm_scaling : float
        Pixel to nm scaling.
    convert_nm_to_m : bool
        Convert nanometers to metres.
    min_skeleton_size : int
        Minimum skeleton size below which tracing statistics are not calculated.
    mask_smoothing_params: dict
        Dictionary of parameters to smooth the grain mask for better quality skeletonisation results. Contains 
        a gaussian 'sigma' and number of dilation iterations.
    skeletonisation_params : dict
        Skeletonisation Parameters. Method of skeletonisation to use 'topostats' is the original TopoStats
        method. Three methods from scikit-image are available 'zhang', 'lee' and 'thin'.
    pruning_params : dict
            Pruning parameters.
    n_grain : int
        Grain number being processed (only  used in logging).
    joining_node_length : float
        The length over which to join skeleton crossing points due to misalignment.
    spline_step_size : float
        Step size for spline evaluation in metres.
    spline_linear_smoothing : float
        Smoothness of linear splines.
    spline_circular_smoothing : float
        Smoothness of circular splines.
    spline_quiet : bool
        Suppresses scipy splining warnings.
    spline_degree : int
        Degree of the spline.
    """

    def __init__(
        self,
        image: npt.NDArray,
        grain: npt.NDArray,
        filename: str,
        pixel_to_nm_scaling: float,
        convert_nm_to_m: bool = True,
        min_skeleton_size: int = 10,
        mask_smoothing_params: dict = {"gaussian_sigma": None, "dilation_iterations": 2},
        skeletonisation_params: dict = {"skeletonisation_method": "zhang"},
        pruning_params: dict = {"pruning_method": "topostats"},
        n_grain: int = None,
        joining_node_length=7e-9,
        spline_step_size: float = 7e-9,
        spline_linear_smoothing: float = 5.0,
        spline_circular_smoothing: float = 0.0,
        spline_quiet: bool = True,
        spline_degree: int = 3,
    ):
        """
        Initialise the class.

        Parameters
        ----------
        image : npt.NDArray
            Cropped image, typically padded beyond the bounding box.
        grain : npt.NDArray
            Labelled mask for the grain, typically padded beyond the bounding box.
        filename : str
            Filename being processed.
        pixel_to_nm_scaling : float
            Pixel to nm scaling.
        convert_nm_to_m : bool
            Convert nanometers to metres.
        min_skeleton_size : int
            Minimum skeleton size below which tracing statistics are not calculated.
        mask_smoothing_params: dict
            Dictionary of parameters to smooth the grain mask for better quality skeletonisation results. Contains 
            a gaussian 'sigma' and number of dilation iterations.
        skeletonisation_params : dict
            Skeletonisation Parameters. Method of skeletonisation to use 'topostats' is the original TopoStats
            method. Three methods from scikit-image are available 'zhang', 'lee' and 'thin'.
        pruning_params : dict
            Pruning parameters.
        n_grain : int
            Grain number being processed (only  used in logging).
        joining_node_length : float
            The length over which to join skeleton crossing points due to misalignment.
        spline_step_size : float
            Step size for spline evaluation in metres.
        spline_linear_smoothing : float
            Smoothness of linear splines.
        spline_circular_smoothing : float
            Smoothness of circular splines.
        spline_quiet : bool
            Suppresses scipy splining warnings.
        spline_degree : int
            Degree of the spline.
        """
        self.image = image * 1e-9 if convert_nm_to_m else image
        self.grain = grain
        self.filename = filename
        self.pixel_to_nm_scaling = pixel_to_nm_scaling * 1e-9 if convert_nm_to_m else pixel_to_nm_scaling
        self.min_skeleton_size = min_skeleton_size
        self.mask_smoothing_params = mask_smoothing_params
        self.skeletonisation_params = skeletonisation_params
        self.pruning_params = pruning_params
        self.n_grain = n_grain
        self.joining_node_length = joining_node_length
        self.number_of_rows = self.image.shape[0]
        self.number_of_columns = self.image.shape[1]
        self.sigma = 0.7 / (self.pixel_to_nm_scaling * 1e9)
        # Images
        self.smoothed_grain = np.zeros_like(image)
        self.skeleton = np.zeros_like(image)
        self.pruned_skeleton = np.zeros_like(image)
        self.node_image = np.zeros_like(image)
        self.ordered_trace_img = np.zeros_like(image)
        self.fitted_trace_img = np.zeros_like(image)
        self.splined_trace_img = np.zeros_like(image)
        self.visuals = np.zeros_like(image)
        # Nodestats objects
        self.node_dict = None
        self.node_image_dict = None
        # Metrics
        self.num_crossings = None
        self.avg_crossing_confidence = None
        self.min_crossing_confidence = None
        self.topology = [None]
        self.topology2 = [None]
        self.contour_lengths = []
        self.end_to_end_distances = []
        self.mol_is_circulars = []
        self.curvatures = []
        self.num_mols = 1
        # Traces
        self.disordered_trace = None
        self.ordered_traces = None
        self.fitted_traces = []
        self.splined_traces = []
        # Splining parameters
        self.spline_step_size: float = spline_step_size
        self.spline_linear_smoothing: float = spline_linear_smoothing
        self.spline_circular_smoothing: float = spline_circular_smoothing
        self.spline_quiet: bool = spline_quiet
        self.spline_degree: int = spline_degree
        # height-trace objects
        self.ordered_trace_heights = []
        self.ordered_trace_cumulative_distances = []

        # suppresses scipy splining warnings
        warnings.filterwarnings("ignore")

        LOGGER.debug(f"[{self.filename}] Performing DNA Tracing")

    def trace_dna(self):
        """Perform the DNA tracing pipeline."""
        print("------", self.mask_smoothing_params)
        self.smoothed_grain += self.smooth_grains(self.grain, **self.mask_smoothing_params)
        self.get_disordered_trace()

        if self.disordered_trace is None:
            LOGGER.info(f"[{self.filename}] : Grain failed to Skeletonise")
        elif len(self.disordered_trace) >= self.min_skeleton_size:
            self.linear_or_circular(self.disordered_trace)
            # self.get_ordered_traces()
            nodes = nodeStats(
                filename=self.filename,
                image=self.image,
                grain=self.grain,
                smoothed_grain=self.smoothed_grain,
                skeleton=self.pruned_skeleton,
                px_2_nm=self.pixel_to_nm_scaling,
                n_grain=self.n_grain,
                node_joining_length=self.joining_node_length,
            )
            self.node_dict, self.node_image_dict = nodes.get_node_stats()
            self.avg_crossing_confidence = nodeStats.average_crossing_confs(self.node_dict)
            self.min_crossing_confidence = nodeStats.minimum_crossing_confs(self.node_dict)
            self.node_image = nodes.connected_nodes
            self.num_crossings = nodes.num_crossings

            # try: # try to order using nodeStats
            if nodes.check_node_errorless():
                self.ordered_traces, self.visuals = nodes.compile_trace()
                self.num_mols = len(self.ordered_traces)
                LOGGER.info(f"[{self.filename}] : Grain {self.n_grain} ordered via nodeStats.")
                LOGGER.info(f"[{self.filename}] : Grain {self.n_grain} has {len(self.ordered_traces)} molecules.")

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

            for trace in self.ordered_traces:  # can be multiple traces from 1 grain like in catenated mols
                self.ordered_trace_img += coords_2_img(trace, self.image, ordered=True)
                if len(trace) >= self.min_skeleton_size:
                    self.ordered_trace_heights.append(self.get_ordered_trace_heights(trace))
                    self.ordered_trace_cumulative_distances.append(self.get_ordered_trace_cumulative_distances(trace))

                    mol_is_circular = self.linear_or_circular(trace)
                    self.mol_is_circulars.append(mol_is_circular)
                    fitted_trace = self.get_fitted_traces(trace, mol_is_circular)
                    self.fitted_trace_img += coords_2_img(fitted_trace, self.image)
                    # Proper cleanup needed - ordered trace instead of fitted trace?

                    splined_trace = self.get_splined_traces(fitted_trace, mol_is_circular)
                    # sometimes traces can get skwiffy
                    splined_trace = splined_trace[
                        (splined_trace[:, 0] < self.image.shape[0])
                        & (splined_trace[:, 1] < self.image.shape[1])
                        & (splined_trace[:, 0] > 0)
                        & (splined_trace[:, 1] > 0)
                    ]
                    self.splined_traces.append(np.array(splined_trace))
                    self.splined_trace_img += coords_2_img(np.array(splined_trace, dtype=np.int32), self.image)

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

    def gaussian_filter(self, **kwargs) -> npt.NDArray:
        """
        Apply Gaussian filter.

        Parameters
        ----------
        **kwargs
            Arguments passed to 'skimage.filter.gaussian(**kwargs)'.
        """
        self.smoothed_grain = gaussian(self.image, sigma=self.sigma, **kwargs)
        LOGGER.info(f"[{self.filename}] [{self.n_grain}] : Gaussian filter applied.")

    def smooth_grains(
        self, grain: npt.NDArray, dilation_iterations: int=2, gaussian_sigma: float | int | None=None
    ) -> npt.NDArray:
        """
        Smooth grains based on the lower number of binary pixels added from dilation or gaussian.

        This method ensures gaussian smoothing isn't too aggressive and covers / creates gaps in the mask.

        Parameters
        ----------
        grain : npt.NDArray
            Numpy array of the grain mask.
        dilation_iterations : int
            Number of times to dilate the grain to smooth it. Default is 2.
        gaussian_sigma : float | None
            Gaussian sigma value to smooth the grains after an Otsu threshold. If None, defaults to max(grain.shape) / 256.

        Returns
        -------
        npt.NDArray
            Numpy array of smmoothed image.
        """
        gaussian_sigma = max(grain.shape) / 256 if gaussian_sigma is None else gaussian_sigma
        print("-------", dilation_iterations, type(dilation_iterations))
        dilation = ndimage.binary_dilation(grain, iterations=dilation_iterations).astype(np.int32)
        gauss = gaussian(grain, sigma=gaussian_sigma)
        gauss[gauss > threshold_otsu(gauss) * 1.3] = 1
        gauss[gauss != 1] = 0
        gauss = gauss.astype(np.int32)
        if dilation.sum() - grain.sum() > gauss.sum() - grain.sum():
            gauss = self.re_add_holes(grain, gauss)
            return gauss
        else:
            dilation = self.re_add_holes(grain, dilation)
            return dilation

    def re_add_holes(
        self, orig_mask: npt.NDArray, new_mask: npt.NDArray, holearea_min_max: list = [4, np.inf]
    ) -> npt.NDArray:
        """
        Restore holes in masks that were occluded by dilation.

        As Gaussian dilation smoothing methods can close holes in the original mask, this function obtains those holes
        (based on the general background being the first due to padding) and adds them back into the smoothed mask. When
        paired, this essentially just smooths the outer edge of the grains.

        Parameters
        ----------
        orig_mask : npt.NDArray
            Original mask.
        new_mask : npt.NDArray
            New mask.
        holearea_min_max : list
            List of minimum and maximum hole area (in pixels).

        Returns
        -------
        npt.NDArray
            Smoothed mask with holes restored.
        """

        holesize_min_px = holearea_min_max[0] / ((self.pixel_to_nm_scaling / 1e-9) ** 2)
        holesize_max_px = holearea_min_max[1] / ((self.pixel_to_nm_scaling / 1e-9) ** 2)
        holes = 1 - orig_mask
        holes = label(holes)
        sizes = [holes[holes == i].size for i in range(1, holes.max() + 1)]
        holes[holes == 1] = 0  # set background to 0

        for i, size in enumerate(sizes):
            if size < holesize_min_px or size > holesize_max_px:  # small holes may be fake are left out
                holes[holes == i + 1] = 0
        holes[holes != 0] = 1

        # compare num holes in each mask
        holey_smooth = new_mask.copy()
        holey_smooth[holes == 1] = 0

        return holey_smooth

    def get_ordered_trace_heights(self, ordered_trace) -> npt.NDArray:
        """
        Sort the smoothed grain array by the ordered trace.

        Gets the heights of each pixel in the ordered trace from the gaussian filtered image. The pixel coordinates
        for the ordered trace are stored in the ordered trace list as part of the class.

        Parameters
        ----------
        ordered_trace : npt.NDArray
            Array of ordered trace coordinates.

        Returns
        -------
        npt.NDArray
            Smoothed array ordered by the ordered trace.
        """
        return np.array(self.smoothed_grain[ordered_trace[:, 0], ordered_trace[:, 1]])

    def get_ordered_trace_cumulative_distances(self, ordered_trace: npt.NDArray) -> npt.NDArray:
        """
        Measure the cumulative distances of each pixel in the ordered_trace.

        Parameters
        ----------
        ordered_trace : npt.NDArray
            An ordered pixelwise path representing a trace.

        Returns
        -------
        npt.NDArray
            An array of cumulative distances from the start of the trace.
        """

        # Get the cumulative distances of each pixel in the ordered trace from the gaussian filtered image
        # the pixel coordinates are stored in the ordered trace list.
        return self.coord_dist(coordinates=ordered_trace, px_to_nm=self.pixel_to_nm_scaling)

    @staticmethod
    def coord_dist(coordinates: npt.NDArray, px_to_nm: float) -> npt.NDArray:
        """
        Calculate the cumulative real distances between each pixel in a trace.

        Take a Nx2 numpy array of (grid adjacent) coordinates and produce a list of cumulative distances in
        nanometres, travelling from pixel to pixel. 1D example: coordinates: [0, 0], [0, 1], [1, 1], [2, 2] cumulative
        distances: [0, 1, 2, 3.4142]. Counts diagonal connections as 1.4142 distance. Converts distances from
        pixels to nanometres using px_to_nm scaling factor.
        Note that the pixels have to be adjacent.

        Parameters
        ----------
        coordinates : npt.NDArray
            A Nx2 integer array of coordinates of the pixels of a trace from a binary trace image.
        px_to_nm : float
            Pixel to nanometre scaling factor to allow for real length measurements of distances rather
            than pixels.

        Returns
        -------
        npt.NDArray
            Numpy array of length N containing the cumulative sum of distances (0 at the first entry,
            full molecule length at the last entry).
        """

        # Shift the array by one coordinate so the end is at the start and the second to last is at the end
        # this allows for the calculation of the distance between each pixel by subtracting the shifted array
        # from the original array
        rolled_coords = np.roll(coordinates, 1, axis=0)

        # Calculate the distance between each pixel in the trace
        pixel_diffs = coordinates - rolled_coords
        pixel_distances = np.linalg.norm(pixel_diffs, axis=1)

        # Set the first distance to zero since we don't want to count the distance from the last pixel to the first
        pixel_distances[0] = 0

        # Calculate the cumulative sum of the distances
        cumulative_distances = np.cumsum(pixel_distances)

        # Convert the cumulative distances from pixels to nanometres
        cumulative_distances_nm = cumulative_distances * px_to_nm

        return cumulative_distances_nm

    def get_disordered_trace(self):
        """
        Derive the disordered trace coordinates from the binary mask and image via skeletonisation and pruning.
        """
        self.skeleton = getSkeleton(
            self.image,
            self.smoothed_grain,
            method=self.skeletonisation_params["skeletonisation_method"],
            height_bias=self.skeletonisation_params["height_bias"],
        ).get_skeleton()
        # self.skeleton = getSkeleton(self.image, self.smoothed_grain).get_skeleton(self.skeletonisation_params.copy())
        # np.savetxt(OUTPUT_DIR / "skel.txt", self.skeleton)
        # np.savetxt(OUTPUT_DIR / "image.txt", self.image)
        # np.savetxt(OUTPUT_DIR / "smooth.txt", self.smoothed_grain)
        self.pruned_skeleton = pruneSkeleton(self.smoothed_grain, self.skeleton).prune_skeleton(
            self.pruning_params.copy()
        )
        self.pruned_skeleton = self.remove_touching_edge(self.pruned_skeleton)
        self.disordered_trace = np.argwhere(self.pruned_skeleton == 1)

    @staticmethod
    def remove_touching_edge(skeleton: npt.NDArray) -> npt.NDArray:
        """
        Remove any skeleton points touching the border (to prevent errors later).

        Parameters
        ----------
        skeleton : npt.NDArray
            A binary array where touching clusters of 1's become 0's if touching the edge of the array.

        Returns
        -------
        npt.NDArray
            Skeleton without points touching the border.
        """
        for edge in [skeleton[0, :-1], skeleton[:-1, -1], skeleton[-1, 1:], skeleton[1:, 0]]:
            uniques = np.unique(edge)
            for i in uniques:
                skeleton[skeleton == i] = 0
        return skeleton

    def linear_or_circular(self, traces) -> bool:
        """
        Determine whether the molecule is circular or linear via >1 points in the local start area.

        This function is sensitive to branches from the skeleton because it is based on whether any given point has zero
        neighbours or not so the traces should be pruned.

        Parameters
        ----------
        traces : npt.NDArray
            The array of coordinates to be assessed.

        Returns
        -------
        bool
            Whether a molecule is linear or not (True if linear, False otherwise).
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
        return False

    def get_ordered_traces(self):
        """
        Obtain ordered traces from disordered traces.
        """
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

    def get_fitted_traces(self, ordered_trace: npt.NDArray, mol_is_circular: bool) -> npt.NDArray:
        """
        Adjust coordinates to lie along the highest points of the molecule.

        Parameters
        ----------
        ordered_trace : npt.NDArray
            Ordered trace.
        mol_is_circular : bool
            Whether molecule is circular.

        Returns
        -------
        npt.NDArray
            Adjusted ordered trace sitting on the highest points of the molecule.
        """
        # This indexes a 3 nm height profile perpendicular to DNA backbone
        # note that this is a hard coded parameter
        index_width = int(3e-9 / (self.pixel_to_nm_scaling))
        if index_width < 2:
            index_width = 2

        for coord_num, trace_coordinate in enumerate(ordered_trace):
            height_values = None

            # Block of code to prevent indexing outside image limits
            # e.g. indexing self.smoothed_grain[130, 130] for 128x128 image
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
                nearest_point = ordered_trace[coord_num - 2]
                vector = np.subtract(nearest_point, trace_coordinate)
                vector_angle = math.degrees(math.atan2(vector[1], vector[0]))
            else:
                try:
                    nearest_point = ordered_trace[coord_num + 2]
                except IndexError:
                    nearest_point = ordered_trace[coord_num - 2]
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

            # Use the perp array to index the gaussian filtered image
            perp_array = np.column_stack((x_coords, y_coords))
            try:
                height_values = self.smoothed_grain[perp_array[:, 0], perp_array[:, 1]]
            except IndexError:
                perp_array[:, 0] = np.where(
                    perp_array[:, 0] > self.smoothed_grain.shape[0], self.smoothed_grain.shape[0], perp_array[:, 0]
                )
                perp_array[:, 1] = np.where(
                    perp_array[:, 1] > self.smoothed_grain.shape[1], self.smoothed_grain.shape[1], perp_array[:, 1]
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

    @staticmethod
    # Perhaps we need a module for array functions?
    def remove_duplicate_consecutive_tuples(tuple_list: list[tuple | npt.NDArray]) -> list[tuple]:
        """
        Remove duplicate consecutive tuples from a list.

        Parameters
        ----------
        tuple_list : list[tuple | npt.NDArray]
            List of tuples or numpy ndarrays to remove consecutive duplicates from.

        Returns
        -------
        list[Tuple]
            List of tuples with consecutive duplicates removed.

        Examples
        --------
        For the list of tuples [(1, 2), (1, 2), (1, 2), (2, 3), (2, 3), (3, 4)], this function will return
        [(1, 2), (2, 3), (3, 4)]
        """

        duplicates_removed = []
        for index, tup in enumerate(tuple_list):
            if index == 0 or not np.array_equal(tuple_list[index - 1], tup):
                duplicates_removed.append(tup)
        return np.array(duplicates_removed)

    def get_splined_traces(
        self,
        fitted_trace: npt.NDArray,
        mol_is_circular: bool,
    ) -> npt.NDArray:
        """
        Get a splined version of the fitted trace - useful for finding the radius of gyration etc.

        This function actually calculates the average of several splines which is important for getting a good fit on
        the lower resolution data.

        Parameters
        ----------
        fitted_trace : npt.NDArray
            Numpy array of the fitted trace.
        mol_is_circular : bool
            Is the molecule circular.

        Returns
        -------
        npt.NDArray
            Splined (smoothed) array of trace.
        """

        # Fitted traces are Nx2 numpy arrays of coordinates
        # All self references are here for easy turning into static method if wanted, also type hints and short documentation
        fitted_trace: npt.NDArray = fitted_trace  # boolean 2d numpy array of fitted traces to spline
        step_size_m: float = self.spline_step_size  # the step size for the splines to skip pixels in the fitted trace
        pixel_to_nm_scaling: float = self.pixel_to_nm_scaling  # pixel to nanometre scaling factor for the image
        mol_is_circular: bool = mol_is_circular  # whether or not the molecule is classed as circular
        n_grain: int = self.n_grain  # the grain index (for logging purposes)

        # Calculate the step size in pixels from the step size in metres.
        # Should always be at least 1.
        # Note that step_size_m is in m and pixel_to_nm_scaling is in m because of the legacy code which seems to almost always have
        # pixel_to_nm_scaling be set in metres using the flag convert_nm_to_m. No idea why this is the case.
        step_size_px = max(int(step_size_m / pixel_to_nm_scaling), 1)

        # Splines will be totalled and then divived by number of splines to calculate the average spline
        spline_sum = None

        # Get the length of the fitted trace
        fitted_trace_length = fitted_trace.shape[0]

        # If the fitted trace is less than the degree plus one, then there is no
        # point in trying to spline it, just return the fitted trace
        if fitted_trace_length < self.spline_degree + 1:
            LOGGER.warning(
                f"Fitted trace for grain {n_grain} too small ({fitted_trace_length}), returning fitted trace"
            )
            self.splined_trace = fitted_trace
            return

        # There cannot be fewer than degree + 1 points in the spline
        # Decrease the step size to ensure more than this number of points
        while fitted_trace_length / step_size_px < self.spline_degree + 1:
            # Step size cannot be less than 1
            if step_size_px <= 1:
                step_size_px = 1
                break
            step_size_px = -1

        # Set smoothness and periodicity appropriately for linear / circular molecules.
        spline_smoothness, spline_periodicity = (
            (self.spline_circular_smoothing, 2) if mol_is_circular else (self.spline_linear_smoothing, 0)
        )

        # Create an array of evenly spaced points between 0 and 1 for the splines to be evaluated at.
        # This is needed to ensure that the splines are all the same length as the number of points
        # in the spline is controlled by the ev_array variable.
        ev_array = np.linspace(0, 1, fitted_trace_length * step_size_px)

        # Find as many splines as there are steps in step size, this allows for a better spline to be obtained
        # by averaging the splines. Think of this like weaving a lot of splines together along the course of
        # the trace. Example spline coordinate indexes: [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4], where spline
        # 1 takes every 4th coordinate, starting at position 0, then spline 2 takes every 4th coordinate
        # starting at position 1, etc...
        for i in range(step_size_px):
            # Sample the fitted trace at every step_size_px pixels
            sampled = [fitted_trace[j, :] for j in range(i, fitted_trace_length, step_size_px)]

            # Scipy.splprep cannot handle duplicate consecutive x, y tuples, so remove them.
            # Get rid of any consecutive duplicates in the sampled coordinates
            sampled = self.remove_duplicate_consecutive_tuples(tuple_list=sampled)

            x_sampled = sampled[:, 0]
            y_sampled = sampled[:, 1]

            # Use scipy's B-spline functions
            # tck is a tuple, (t,c,k) containing the vector of knots, the B-spline coefficients
            # and the degree of the spline.
            # s is the smoothing factor, per is the periodicity, k is the degree of the spline
            tck, _ = interp.splprep(
                [x_sampled, y_sampled],
                s=spline_smoothness,
                per=spline_periodicity,
                quiet=self.spline_quiet,
                k=self.spline_degree,
            )
            # splev returns a tuple (x_coords ,y_coords) containing the smoothed coordinates of the
            # spline, constructed from the B-spline coefficients and knots. The number of points in
            # the spline is controlled by the ev_array variable.
            # ev_array is an array of evenly spaced points between 0 and 1.
            # This is to ensure that the splines are all the same length.
            # Tck simply provides the coefficients for the spline.
            out = interp.splev(ev_array, tck)
            splined_trace = np.column_stack((out[0], out[1]))

            # Add the splined trace to the spline_sum array for averaging later
            if spline_sum is None:
                spline_sum = np.array(splined_trace)
            else:
                spline_sum = np.add(spline_sum, splined_trace)

        # Find the average spline between the set of splines
        # This is an attempt to find a better spline by averaging our candidates
        spline_average = np.divide(spline_sum, [step_size_px, step_size_px])

        return spline_average

    def show_traces(self):
        """Plot traces."""
        plt.pcolormesh(self.smoothed_grain, vmax=-3e-9, vmin=3e-9)
        plt.colorbar()
        plt.plot(self.ordered_trace[:, 0], self.ordered_trace[:, 1], markersize=1)
        plt.plot(self.fitted_trace[:, 0], self.fitted_trace[:, 1], markersize=1)
        plt.plot(self.splined_trace[:, 0], self.splined_trace[:, 1], markersize=1)

        plt.show()
        plt.close()

    def saveTraceFigures(
        self,
        filename: str | Path,
        channel_name: str,
        vmaxval: float | int,
        vminval: float | int,
        output_dir: str | Path = None,
    ) -> None:
        """
        Save the traces.

        Parameters
        ----------
        filename : str | Path
            Filename being processed.
        channel_name : str
            Channel.
        vmaxval : float | int
            Maximum value for height.
        vminval : float | int
            Minimum value for height.
        output_dir : str | Path
            Output directory.
        """
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
    def _checkForSaveDirectory(self, filename: str, new_output_dir: str) -> str:
        """
        Create output directory and updates filename to account for this.

        Parameters
        ----------
        filename : str
            Filename.
        new_output_dir : str
            Target directory.

        Returns
        -------
        str
            Updated output directory.
        """

        split_directory_path = os.path.split(filename)

        try:
            os.mkdir(os.path.join(split_directory_path[0], new_output_dir))
        except OSError:  # OSError happens if the directory already exists
            pass

        updated_filename = os.path.join(split_directory_path[0], new_output_dir, split_directory_path[1])

        return updated_filename

    def find_curvature(self):
        """Calculate curvature of the molecule."""
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

    def saveCurvature(self) -> None:
        """Save curvature statistics."""
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

    def plotCurvature(self, dna_num: int) -> None:
        """
        Plot the curvature of the chosen molecule as a function of the contour length (in metres).

        Parameters
        ----------
        dna_num : int
            Molecule to plot, used for indexing.
        """

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

    def measure_contour_length(self, splined_trace: npt.NDArray, mol_is_circular: bool) -> float:
        """
        Contour length for each of the splined traces accounting  for whether the molecule is circular or linear.

        Contour length units are nm.

        Parameters
        ----------
        splined_trace : npt.NDArray
            The splined trace.
        mol_is_circular : bool
            Whether the molecule is circular or not.

        Returns
        -------
        float
            Length of molecule in nanometres (nm).
        """
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
                    break
        return contour_length

    def measure_end_to_end_distance(self, splined_trace, mol_is_circular):
        """
        Euclidean distance between the start and end of linear molecules.

        The hypotenuse is calculated between the start ([0,0], [0,1]) and end ([-1,0], [-1,1]) of linear
        molecules. If the molecule is circular then the distance is set to zero (0).

        Parameters
        ----------
        splined_trace : npt.NDArray
            The splined trace.
        mol_is_circular : bool
            Whether the molecule is circular or not.

        Returns
        -------
        float
            Length of molecule in nanometres (nm).
        """
        if not mol_is_circular:
            return (
                math.hypot((splined_trace[0, 0] - splined_trace[-1, 0]), (splined_trace[0, 1] - splined_trace[-1, 1]))
                * self.pixel_to_nm_scaling
            )
        return 0


def trace_image(
    image: npt.NDArray,
    grains_mask: npt.NDArray,
    filename: str,
    pixel_to_nm_scaling: float,
    min_skeleton_size: int,
    mask_smoothing_params: dict,
    skeletonisation_params: dict,
    pruning_params: dict,
    joining_node_length: float = 7e-9,
    spline_step_size: float = 7e-9,
    spline_linear_smoothing: float = 5.0,
    spline_circular_smoothing: float = 0.0,
    pad_width: int = 1,
    cores: int = 1,
) -> dict:
    """
    Processor function for tracing image.

    Parameters
    ----------
    image : npt.NDArray
        Full image as Numpy Array.
    grains_mask : npt.NDArray
        Full image as Grains that are labelled.
    filename : str
        File being processed.
    pixel_to_nm_scaling : float
        Pixel to nm scaling.
    min_skeleton_size : int
        Minimum size of grain in pixels after skeletonisation.
    mask_smoothing_params: dict
        Dictionary of parameters to smooth the grain mask for better quality skeletonisation results. Contains 
        a gaussian 'sigma' and number of dilation iterations.
    skeletonisation_params : dict
        Dictionary of options for skeletonisation, options are 'zhang' (scikit-image) / 'lee' (scikit-image) / 'thin'
        (scikitimage) or 'topostats' (original TopoStats method).
    pruning_params : dict
        Dictionary of options for pruning.
    joining_node_length : float
        The length over which to join skeleton crossing points due to misalignment.
    spline_step_size : float
        Step size for spline evaluation in metres.
    spline_linear_smoothing : float
        Smoothness of linear splines.
    spline_circular_smoothing : float
        Smoothness of circular splines.
    pad_width : int
        Number of cells to pad arrays by, required to handle instances where grains touch the bounding box edges.
    cores : int
        Number of cores to process with.

    Returns
    -------
    dict
        Statistics from skeletonising and tracing the grains in the image.
    """
    # Check both arrays are the same shape - should this be a test instead, why should this ever occur?
    if image.shape != grains_mask.shape:
        raise ValueError(f"Image shape ({image.shape}) and Mask shape ({grains_mask.shape}) should match.")

    cropped_images, cropped_masks, bboxs = prep_arrays(image, grains_mask, pad_width)
    region_properties = skimage_measure.regionprops(grains_mask)
    grain_anchors = [grain_anchor(image.shape, list(grain.bbox), pad_width) for grain in region_properties]
    n_grains = len(cropped_images)
    img_base = np.zeros_like(image)

    grain_results = {}
    trace_results = {}
    full_node_image_dict = {}
    all_ordered_trace_heights = {}
    all_ordered_trace_cumulative_distances = {}
    ordered_traces = {}
    splined_traces = {}

    # want to get each cropped image, use some anchor coords to match them onto the image,
    #   and compile all the grain images onto a single image
    all_images = {
        "grain": img_base,
        "smoothed_grain": img_base.copy(),
        "skeleton": img_base.copy(),
        "pruned_skeleton": img_base.copy(),
        "node_img": img_base.copy(),
        "ordered_traces": img_base.copy(),
        "fitted_traces": img_base.copy(),
        "visual": img_base.copy(),
    }
    spline_traces_image_frame = []

    LOGGER.info(f"[{filename}] : Calculating DNA tracing statistics for {n_grains} grains.")

    for cropped_image_index, cropped_image in cropped_images.items():
        cropped_mask = cropped_masks[cropped_image_index]
        result, trace_images, node_image_dict = trace_grain(
            cropped_image=cropped_image,
            cropped_mask=cropped_mask,
            pixel_to_nm_scaling=pixel_to_nm_scaling,
            mask_smoothing_params=mask_smoothing_params,
            skeletonisation_params=skeletonisation_params,
            pruning_params=pruning_params,
            filename=filename,
            min_skeleton_size=min_skeleton_size,
            joining_node_length=joining_node_length,
            spline_step_size=spline_step_size,
            spline_linear_smoothing=spline_linear_smoothing,
            spline_circular_smoothing=spline_circular_smoothing,
            n_grain=cropped_image_index,
        )
        LOGGER.info(f"[{filename}] : Traced grain {cropped_image_index + 1} of {n_grains}")

        grain_results[cropped_image_index] = result["grainstats_results"]
        trace_results[f"grain_{cropped_image_index}"] = result["tracingstats_results"]
        full_node_image_dict[f"grain_{cropped_image_index}"] = node_image_dict

        all_ordered_trace_heights[f"grain_{cropped_image_index}"] = {
            mol_idx: mol_dict.pop("trace_heights")
            for mol_idx, mol_dict in result["tracingstats_results"]["metrics"].items()
        }
        all_ordered_trace_cumulative_distances[f"grain_{cropped_image_index}"] = {
            mol_idx: mol_dict.pop("trace_distances")
            for mol_idx, mol_dict in result["tracingstats_results"]["metrics"].items()
        }

        ordered_traces[f"grain_{cropped_image_index}"] = {
            mol_idx: mol_dict["ordered_trace"]
            for mol_idx, mol_dict in result["tracingstats_results"]["metrics"].items()
        }
        splined_traces[f"grain_{cropped_image_index}"] = {
            mol_idx: mol_dict["splined_trace"]
            for mol_idx, mol_dict in result["tracingstats_results"]["metrics"].items()
        }

        # remap the cropped images back onto the original
        for image_name, full_image in all_images.items():
            crop = trace_images[image_name]
            bbox = bboxs[cropped_image_index]
            full_image[bbox[0] : bbox[2], bbox[1] : bbox[3]] += crop[pad_width:-pad_width, pad_width:-pad_width]

        # remap spline coords to the original image
        for _, mol_dict in result["tracingstats_results"]["metrics"].items():
            spline_traces_image_frame.append(mol_dict.pop("splined_trace") + [bbox[0] - pad_width, bbox[1] - pad_width])

    try:
        grain_results = pd.DataFrame.from_dict(grain_results, orient="index")
        grain_results.index.name = "grain_number"
    except ValueError as error:
        LOGGER.error("No grains found in any images, consider adjusting your thresholds.")
        LOGGER.error(error)

    return {
        "grain_statistics": grain_results,
        "dnatracing_statistics": trace_results,
        "node_images": full_node_image_dict,
        "all_ordered_traces": ordered_traces,
        "all_splined_traces": splined_traces,
        "splined_traces_image_frame": spline_traces_image_frame,
        "cropped_images": cropped_images,
        "all_images": all_images,
        "all_ordered_trace_heights": all_ordered_trace_heights,
        "all_ordered_trace_cumulative_distances": all_ordered_trace_cumulative_distances,
    }


# not used
def round_splined_traces(splined_traces: dict) -> dict:
    """
    Round a Dict of floating point coordinates to integer floating point coordinates.

    Parameters
    ----------
    splined_traces : dict
        Floating point coordinates to be rounded.

    Returns
    -------
    dict
        Dictionary of rounded integer coordinates.
    """
    rounded_splined_traces = {}
    for grain_number, splined_trace in splined_traces.items():
        if splined_trace is not None:
            rounded_splined_traces[grain_number] = np.round(splined_trace).astype(int)
        else:
            rounded_splined_traces[grain_number] = None

    return rounded_splined_traces


# not used
def trim_array(array: npt.NDArray, pad_width: int) -> npt.NDArray:
    """
    Trim an array by the specified pad_width.

    Removes a border from an array. Typically this is the second padding that is added to the image/masks for edge cases
    that are near image borders and means traces will be correctly aligned as a mask for the original image.

    Parameters
    ----------
    array : npt.NDArray
        Numpy array to be trimmed.
    pad_width : int
        Padding to be removed.

    Returns
    -------
    npt.NDArray
        Trimmed array.
    """
    return array[pad_width:-pad_width, pad_width:-pad_width]


def adjust_coordinates(coordinates: npt.NDArray, pad_width: int) -> npt.NDArray:
    """
    Adjust coordinates of a trace by the pad_width.

    A second padding is made to allow for grains that are "edge cases" and close to the bounding box edge. This adds the
    pad_width to the cropped grain array. In order to realign the trace with the original image we need to remove this
    padding so that when the coordinates are combined with the "grain_anchor", which isn't padded twice, the
    coordinates correctly align with the original image.

    Parameters
    ----------
    coordinates : npt.NDArray
        An array of trace coordinates (typically ordered).
    pad_width : int
        The amount of padding used.

    Returns
    -------
    npt.NDArray
        Array of trace coordinates adjusted for secondary padding.
    """
    return coordinates - pad_width


def trace_mask(
    grain_anchors: list[npt.NDArray], ordered_traces: dict[str, npt.NDArray], image_shape: tuple, pad_width: int
) -> npt.NDArray:
    """
    Place the traced skeletons into an array of the original image for plotting/overlaying.

    Adjusts the coordinates back to the original position based on each grains anchor coordinates of the padded
    bounding box. Adjustments are made for the secondary padding that is made.

    Parameters
    ----------
    grain_anchors : List[npt.NDArray]
        List of grain anchors for the padded bounding box.
    ordered_traces : Dict[npt.NDArray]
        Coordinates for each grain trace.
        Dict of coordinates for each grains trace.
    image_shape : tuple
        Shape of original image.
    pad_width : int
        The amount of padding used on the image.

    Returns
    -------
    npt.NDArray
        Mask of traces for all grains that can be overlaid on original image.
    """
    image = np.zeros(image_shape)
    for grain_number, (grain_anchor, ordered_trace) in enumerate(zip(grain_anchors, ordered_traces.values())):
        # Don't always have an ordered_trace for a given grain_anchor if for example the trace was too small
        if ordered_trace is not None:
            ordered_trace = adjust_coordinates(ordered_trace, pad_width)
            # If any of the values in ordered_trace added to their respective grain_anchor are greater than the image
            # shape, then the trace is outside the image and should be skipped.
            if (
                np.max(ordered_trace[:, 0]) + grain_anchor[0] > image_shape[0]
                or np.max(ordered_trace[:, 1]) + grain_anchor[1] > image_shape[1]
            ):
                LOGGER.info(f"Grain {grain_number} has a trace that breaches the image bounds. Skipping.")
                continue
            ordered_trace[:, 0] = ordered_trace[:, 0] + grain_anchor[0]
            ordered_trace[:, 1] = ordered_trace[:, 1] + grain_anchor[1]
            image[ordered_trace[:, 0], ordered_trace[:, 1]] = 1

    return image


def prep_arrays(
    image: npt.NDArray, labelled_grains_mask: npt.NDArray, pad_width: int
) -> tuple[dict[int, npt.NDArray], dict[int, npt.NDArray]]:
    """
    Take an image and labelled mask and crops individual grains and original heights to a list.

    A second padding is made after cropping to ensure for "edge cases" where grains are close to bounding box edges that
    they are traced correctly. This is accounted for when aligning traces to the whole image mask.

    Parameters
    ----------
    image : npt.NDArray
        Gaussian filtered image. Typically filtered_image.images["gaussian_filtered"].
    labelled_grains_mask : npt.NDArray
        2D Numpy array of labelled grain masks, with each mask being comprised solely of unique integer (not
        zero). Typically this will be output from 'grains.directions[<direction>["labelled_region_02]'.
    pad_width : int
        Cells by which to pad cropped regions by.

    Returns
    -------
    Tuple
        Returns a tuple of two dictionaries, each consisting of cropped arrays.
    """
    # Get bounding boxes for each grain
    region_properties = skimage_measure.regionprops(labelled_grains_mask)
    # Subset image and grains then zip them up
    cropped_images = {}
    cropped_masks = {}

    # for index, grain in enumerate(region_properties):
    #    cropped_image, cropped_bbox = crop_array(image, grain.bbox, pad_width)

    cropped_images = {index: crop_array(image, grain.bbox, pad_width) for index, grain in enumerate(region_properties)}
    cropped_images = {index: np.pad(grain, pad_width=pad_width) for index, grain in cropped_images.items()}
    cropped_masks = {
        index: crop_array(labelled_grains_mask, grain.bbox, pad_width) for index, grain in enumerate(region_properties)
    }
    cropped_masks = {index: np.pad(grain, pad_width=pad_width) for index, grain in cropped_masks.items()}
    # Flip every labelled region to be 1 instead of its label
    cropped_masks = {index: np.where(grain == 0, 0, 1) for index, grain in cropped_masks.items()}
    # Get BBOX coords to remap crops to images
    bboxs = [pad_bounding_box(image.shape, list(grain.bbox), pad_width=pad_width) for grain in region_properties]

    return (cropped_images, cropped_masks, bboxs)


def grain_anchor(array_shape: tuple, bounding_box: list, pad_width: int) -> list:
    """
    Extract anchor (min_row, min_col) from labelled regions and align individual traces over the original image.

    Parameters
    ----------
    array_shape : tuple
        Shape of original array.
    bounding_box : list
        A list of region properties returned by 'skimage.measure.regionprops()'.
    pad_width : int
        Padding for image.

    Returns
    -------
    list(Tuple)
        A list of tuples of the min_row, min_col of each bounding box.
    """
    bounding_coordinates = pad_bounding_box(array_shape, bounding_box, pad_width)
    return (bounding_coordinates[0], bounding_coordinates[1])


def trace_grain(
    cropped_image: npt.NDArray,
    cropped_mask: npt.NDArray,
    pixel_to_nm_scaling: float,
    mask_smoothing_params: dict,
    skeletonisation_params: dict,
    pruning_params: dict,
    filename: str = None,
    min_skeleton_size: int = 10,
    joining_node_length: float = 7e-9,
    spline_step_size: float = 7e-9,
    spline_linear_smoothing: float = 5.0,
    spline_circular_smoothing: float = 0.0,
    n_grain: int = None,
) -> dict:
    """
    Trace an individual grain.

    Tracing involves multiple steps...

    1. Skeletonisation
    2. Pruning of side branches (artefacts from skeletonisation).
    3. Ordering of the skeleton.
    4. Determination of molecule shape.
    5. Jiggling/Fitting
    6. Splining to improve resolution of image.

    Parameters
    ----------
    cropped_image : npt.NDArray
        Cropped array from the original image defined as the bounding box from the labelled mask.
    cropped_mask : npt.NDArray
        Cropped array from the labelled image defined as the bounding box from the labelled mask. This should have been
        converted to a binary mask.
    pixel_to_nm_scaling : float
        Pixel to nm scaling.
    mask_smoothing_params: dict
        Dictionary of parameters to smooth the grain mask for better quality skeletonisation results. Contains 
        a gaussian 'sigma' and number of dilation iterations.
    skeletonisation_params : dict
        Dictionary of skeletonisation parameters, options are 'zhang' (scikit-image) / 'lee' (scikit-image) / 'thin'
        (scikitimage) or 'topostats' (original TopoStats method).
    pruning_params : dict
        Dictionary of pruning parameters.
    filename : str
        File being processed.
    min_skeleton_size : int
        Minimum size of grain in pixels after skeletonisation.
    joining_node_length : float
        The length over which to join skeleton crossing points due to misalignment.
    spline_step_size : float
        Step size for spline evaluation in metres.
    spline_linear_smoothing : float
        Smoothness of linear splines.
    spline_circular_smoothing : float
        Smoothness of circular splines.
    n_grain : int
        Grain number being processed.

    Returns
    -------
    dict
        Dictionary of the contour length, whether the image is circular or linear, the end-to-end distance and an array
        of coordinates.
    """
    dnatrace = dnaTrace(
        image=cropped_image,
        grain=cropped_mask,
        filename=filename,
        pixel_to_nm_scaling=pixel_to_nm_scaling,
        min_skeleton_size=min_skeleton_size,
        mask_smoothing_params=mask_smoothing_params,
        skeletonisation_params=skeletonisation_params,
        pruning_params=pruning_params,
        joining_node_length=joining_node_length,
        spline_step_size=spline_step_size,
        spline_linear_smoothing=spline_linear_smoothing,
        spline_circular_smoothing=spline_circular_smoothing,
        n_grain=n_grain,
    )

    dnatrace.trace_dna()
    results = {
        "grainstats_results": {
            "image": dnatrace.filename,
            "num_crossings": dnatrace.num_crossings,
            "num_mols": dnatrace.num_mols,
            "total_contour_length": np.nansum(np.array(dnatrace.contour_lengths, dtype=np.float64)),
            "grain_avg_crossing_confidence": dnatrace.avg_crossing_confidence,
            "grain_min_crossing_confidence": dnatrace.min_crossing_confidence,
        },
        "tracingstats_results": {
            "image": dnatrace.filename,
            "num_mols": dnatrace.num_mols,
            "num_crossings": dnatrace.num_crossings,
            "total_contour_length": np.nansum(np.array(dnatrace.contour_lengths, dtype=np.float64)),
            "grain_avg_crossing_confidence": dnatrace.avg_crossing_confidence,
            "grain_min_crossing_confidence": dnatrace.min_crossing_confidence,
            "metrics": {
                "mol_0": {
                    "contour_length": None,
                    "circular": None,
                    "end_to_end_distance": None,
                    "trace_heights": None,
                    "trace_distances": None,
                    "ordered_trace": None,
                    "splined_trace": None,
                },  # in case no mols could be traced, need empties to attempt to merge on
            },
            "nodeStats": dnatrace.node_dict,
        },
    }

    for i in range(dnatrace.num_mols):
        results["tracingstats_results"]["metrics"][f"mol_{i}"] = {
            "contour_length": dnatrace.contour_lengths[i],
            "circular": dnatrace.mol_is_circulars[i],
            "end_to_end_distance": dnatrace.end_to_end_distances[i],
            "trace_heights": dnatrace.ordered_trace_heights[i],
            "trace_distances": dnatrace.ordered_trace_cumulative_distances[i],
            "ordered_trace": dnatrace.ordered_traces[i],
            "splined_trace": dnatrace.splined_traces[i],
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

    return results, images, dnatrace.node_image_dict


def crop_array(array: npt.NDArray, bounding_box: tuple, pad_width: int = 0) -> npt.NDArray:
    """
    Crop an array.

    Ideally we pad the array that is being cropped so that we have heights outside of the grains bounding box. However,
    in some cases, if an grain is near the edge of the image scan this results in requesting indexes outside of the
    existing image. In which case we get as much of the image padded as possible.

    Parameters
    ----------
    array : npt.NDArray
        2D Numpy array to be cropped.
    bounding_box : Tuple
        Tuple of coordinates to crop, should be of form (min_row, min_col, max_row, max_col).
    pad_width : int
        Padding to apply to bounding box.

    Returns
    -------
    npt.NDArray()
        Cropped array.
    """
    bounding_box = list(bounding_box)
    bounding_box = pad_bounding_box(array.shape, bounding_box, pad_width)
    return array[
        bounding_box[0] : bounding_box[2],
        bounding_box[1] : bounding_box[3],
    ]


def pad_bounding_box(array_shape: tuple, bounding_box: list, pad_width: int) -> list:
    """
    Pad coordinates, if they extend beyond image boundaries stop at boundary.

    Parameters
    ----------
    array_shape : tuple
        Shape of original image.
    bounding_box : list
        List of coordinates 'min_row', 'min_col', 'max_row', 'max_col'.
    pad_width : int
        Cells to pad arrays by.

    Returns
    -------
    list
       List of padded coordinates.
    """
    # Top Row : Make this the first column if too close
    bounding_box[0] = 0 if bounding_box[0] - pad_width < 0 else bounding_box[0] - pad_width
    # Left Column : Make this the first column if too close
    bounding_box[1] = 0 if bounding_box[1] - pad_width < 0 else bounding_box[1] - pad_width
    # Bottom Row : Make this the last row if too close
    bounding_box[2] = array_shape[0] if bounding_box[2] + pad_width > array_shape[0] else bounding_box[2] + pad_width
    # Right Column : Make this the last column if too close
    bounding_box[3] = array_shape[1] if bounding_box[3] + pad_width > array_shape[1] else bounding_box[3] + pad_width
    return bounding_box


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
