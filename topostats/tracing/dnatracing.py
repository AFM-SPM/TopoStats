"""Perform DNA Tracing"""
from collections import OrderedDict, defaultdict
from functools import partial
from itertools import repeat
import logging
import math
from multiprocessing import Pool
import os
from pathlib import Path
from typing import Dict, Union, Tuple
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage, spatial, interpolate as interp
import skimage.morphology as skimage_morphology
from skimage.filters import gaussian
import skimage.measure as skimage_measure
from tqdm import tqdm

from topostats.logs.logs import LOGGER_NAME
from topostats.tracing.skeletonize import get_skeleton
from topostats.tracing.tracingfuncs import genTracingFuncs, getSkeleton, reorderTrace
from topostats.plottingfuncs import Images

LOGGER = logging.getLogger(LOGGER_NAME)


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
        skeletonisation_method: str = "topostats",
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
        skeletonisation_method:
            Method of skeletonisation to use 'topostats' is the original TopoStats method. Three methods from
            scikit-image are available 'zhang', 'lee' and 'thin'
        n_grain: int
            Grain number being processed (only  used in logging).
        """
        self.image = image * 1e-9 if convert_nm_to_m else image
        self.grain = grain
        self.filename = filename
        self.pixel_to_nm_scaling = pixel_to_nm_scaling * 1e-9 if convert_nm_to_m else pixel_to_nm_scaling
        self.min_skeleton_size = min_skeleton_size
        self.skeletonisation_method = skeletonisation_method
        self.n_grain = n_grain
        self.number_of_rows = self.image.shape[0]
        self.number_of_columns = self.image.shape[1]
        self.sigma = 0.7 / (self.pixel_to_nm_scaling * 1e9)

        self.gauss_image = None
        self.grain = grain
        self.disordered_trace = None
        self.ordered_trace = None
        self.fitted_trace = None
        self.splined_trace = None
        self.contour_length = np.nan
        self.end_to_end_distance = np.nan
        self.mol_is_circular = np.nan
        self.curvature = np.nan

        self.neighbours = 5  # The number of neighbours used for the curvature measurement

        # supresses scipy splining warnings
        warnings.filterwarnings("ignore")

        LOGGER.debug(f"[{self.filename}] Performing DNA Tracing")

    def trace_dna(self):
        """Perform DNA tracing."""
        self.gaussian_filter()
        self.get_disordered_trace()
        if self.disordered_trace is None:
            LOGGER.info(f"[{self.filename}] : Grain failed to Skeletonise")
        elif len(self.disordered_trace) >= self.min_skeleton_size:
            self.linear_or_circular(self.disordered_trace)
            self.get_ordered_traces()
            self.get_trace_heights_and_distances()
            self.linear_or_circular(self.ordered_trace)
            self.get_fitted_traces()
            self.get_splined_traces()
            # self.find_curvature()
            # self.saveCurvature()
            self.measure_contour_length()
            self.measure_end_to_end_distance()

            mask = np.zeros(self.image.shape)
            for coord in self.ordered_trace:
                mask[coord[0], coord[1]] = 1
            Images(
                self.image,
                output_dir=Path("./"),
                filename=f"{self.filename}_trace",
                pixel_to_nm_scaling=self.pixel_to_nm_scaling,
                save=True,
                masked_array=mask,
                mask_cmap="blu",
                image_set="all",
            ).plot_and_save()
        else:
            LOGGER.info(f"[{self.filename}] [{self.n_grain}] : Grain skeleton pixels < {self.min_skeleton_size}")

    def gaussian_filter(self, **kwargs) -> np.array:
        """Apply Gaussian filter"""
        self.gauss_image = gaussian(self.image, sigma=self.sigma, **kwargs)
        LOGGER.info(f"[{self.filename}] [{self.n_grain}] : Gaussian filter applied.")

    def get_disordered_trace(self):
        """Create a skeleton for each of the grains in the image.

        Uses my own skeletonisation function from tracingfuncs module. I will
        eventually get round to editing this function to try to reduce the branching
        and to try to better trace from looped molecules"""
        smoothed_grain = ndimage.binary_dilation(self.grain, iterations=1).astype(self.grain.dtype)

        sigma = 0.01 / (self.pixel_to_nm_scaling * 1e9)
        very_smoothed_grain = ndimage.gaussian_filter(smoothed_grain, sigma)

        LOGGER.info(f"[{self.filename}] [{self.n_grain}] : Skeletonising using {self.skeletonisation_method} method.")
        try:
            if self.skeletonisation_method == "topostats":
                dna_skeleton = getSkeleton(
                    self.gauss_image,
                    smoothed_grain,
                    self.number_of_columns,
                    self.number_of_rows,
                    self.pixel_to_nm_scaling,
                )
                self.disordered_trace = dna_skeleton.output_skeleton
            elif self.skeletonisation_method in ["lee", "zhang", "thin"]:
                self.disordered_trace = np.argwhere(
                    get_skeleton(smoothed_grain, method=self.skeletonisation_method) == True
                )
            else:
                raise ValueError
        except IndexError as e:
            # Some gwyddion grains touch image border causing IndexError
            # These grains are deleted
            LOGGER.info(f"[{self.filename}] [{self.n_grain}] : Grain failed to skeletonise.")
            # raise e

    def linear_or_circular(self, traces):
        """Determines whether each molecule is circular or linear based on the local environment of each pixel from the trace

        This function is sensitive to branches from the skeleton so might need to implement a function to remove them"""

        points_with_one_neighbour = 0
        fitted_trace_list = traces.tolist()

        # For loop determines how many neighbours a point has - if only one it is an end
        for x, y in fitted_trace_list:
            if genTracingFuncs.countNeighbours(x, y, fitted_trace_list) == 1:
                points_with_one_neighbour += 1
            else:
                pass

        if points_with_one_neighbour == 0:
            self.mol_is_circular = True
        else:
            self.mol_is_circular = False

    def get_ordered_traces(self):
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

    def get_trace_heights_and_distances(self) -> None:
        self.trace_heights = self.gauss_image[self.ordered_trace[:, 0], self.ordered_trace[:, 1]]
        self.trace_cumulative_distances = dnaTrace.get_cumulative_coordinate_distances(
            coordinates=self.ordered_trace, pixel_to_nm_scaling=self.pixel_to_nm_scaling
        )

    @staticmethod
    def get_cumulative_coordinate_distances(coordinates: np.ndarray, pixel_to_nm_scaling: float = 1) -> np.ndarray:
        distance_list = [0]
        distance = 0
        for i in range(len(coordinates) - 1):
            if abs(coordinates[i] - coordinates[i + 1]).sum() == 2:
                distance += 1.41421356237  # Saves a sqrt
            else:
                distance += 1
            distance_list.append(distance)
        return np.asarray(distance_list) * pixel_to_nm_scaling

    def get_fitted_traces(self):
        """Create trace coordinates (for each identified molecule) that are adjusted to lie
        along the highest points of each traced molecule
        """

        individual_skeleton = self.ordered_trace
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
            if self.mol_is_circular:
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
                height_values = self.gauss_image[perp_array[:, 1], perp_array[:, 0]]

            # Grab x,y coordinates for highest point
            # fine_coords = np.column_stack((fine_x_coords, fine_y_coords))
            sorted_array = perp_array[np.argsort(height_values)]
            highest_point = sorted_array[-1]

            try:
                # could use np.append() here
                fitted_coordinate_array = np.vstack((fitted_coordinate_array, highest_point))
            except UnboundLocalError:
                fitted_coordinate_array = highest_point

        self.fitted_trace = fitted_coordinate_array
        del fitted_coordinate_array  # cleaned up by python anyway?

    def get_splined_traces(self):
        """Gets a splined version of the fitted trace - useful for finding the radius of gyration etc

        This function actually calculates the average of several splines which is important for getting a good fit on
        the lower res data"""

        step_size = int(7e-9 / (self.pixel_to_nm_scaling))  # 3 nm step size
        interp_step = int(1e-10 / self.pixel_to_nm_scaling)
        # Lets see if we just got with the pixel_to_nm_scaling
        # step_size = self.pixel_to_nm_scaling
        # interp_step = self.pixel_to_nm_scaling

        self.splining_success = True
        nbr = len(self.fitted_trace[:, 0])

        # Hard to believe but some traces have less than 4 coordinates in total
        if len(self.fitted_trace[:, 1]) < 4:
            self.splined_trace = self.fitted_trace
            # continue

        # The degree of spline fit used is 3 so there cannot be less than 3 points in the splined trace
        while nbr / step_size < 4:
            if step_size <= 1:
                step_size = 1
                break
            step_size = -1
        if self.mol_is_circular:
            # if nbr/step_size > 4: #the degree of spline fit is 3 so there cannot be less than 3 points in splined trace

            # ev_array = np.linspace(0, 1, nbr * step_size)
            ev_array = np.linspace(0, 1, int(nbr * step_size))

            for i in range(step_size):
                x_sampled = np.array(
                    [self.fitted_trace[:, 0][j] for j in range(i, len(self.fitted_trace[:, 0]), step_size)]
                )
                y_sampled = np.array(
                    [self.fitted_trace[:, 1][j] for j in range(i, len(self.fitted_trace[:, 1]), step_size)]
                )

                try:
                    tck, u = interp.splprep([x_sampled, y_sampled], s=0, per=2, quiet=1, k=3)
                    out = interp.splev(ev_array, tck)
                    splined_trace = np.column_stack((out[0], out[1]))
                except ValueError:
                    # Value error occurs when the "trace fitting" really messes up the traces

                    x = np.array(
                        [self.ordered_trace[:, 0][j] for j in range(i, len(self.ordered_trace[:, 0]), step_size)]
                    )
                    y = np.array(
                        [self.ordered_trace[:, 1][j] for j in range(i, len(self.ordered_trace[:, 1]), step_size)]
                    )

                    try:
                        tck, u = interp.splprep([x, y], s=0, per=2, quiet=1)
                        out = interp.splev(np.linspace(0, 1, nbr * step_size), tck)
                        splined_trace = np.column_stack((out[0], out[1]))
                    except (
                        ValueError
                    ):  # sometimes even the ordered_traces are too bugged out so just delete these traces
                        self.mol_is_circular.pop(dna_num)
                        self.disordered_trace.pop(dna_num)
                        self.grain.pop(dna_num)
                        self.ordered_trace.pop(dna_num)
                        self.splining_success = False
                        try:
                            del spline_running_total
                        except UnboundLocalError:  # happens if splining fails immediately
                            break
                        break

                try:
                    spline_running_total = np.add(spline_running_total, splined_trace)
                except NameError:
                    spline_running_total = np.array(splined_trace)

            # if not self.splining_success:
            #     continue

            spline_average = np.divide(spline_running_total, [step_size, step_size])
            del spline_running_total
            self.splined_trace = spline_average
            # else:
            #    x = self.fitted_trace[:,0]
            #    y = self.fitted_trace[:,1]

            #    try:
            #        tck, u = interp.splprep([x, y], s=0, per = 2, quiet = 1, k = 3)
            #        out = interp.splev(np.linspace(0,1,nbr*step_size), tck)
            #        splined_trace = np.column_stack((out[0], out[1]))
            #        self.splined_trace = splined_trace
            #    except ValueError: #if the trace is really messed up just delete it
            #        self.mol_is_circular.pop(dna_num)
            #        self.disordered_trace.pop(dna_num)
            #        self.grain.pop(dna_num)
            #        self.ordered_trace.pop(dna_num)

        else:
            # can't get splining of linear molecules to work yet
            self.splined_trace = self.fitted_trace

    def show_traces(self):
        plt.pcolormesh(self.gauss_image, vmax=-3e-9, vmin=3e-9)
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

    def measure_contour_length(self):
        """Measures the contour length for each of the splined traces taking into
        account whether the molecule is circular or linear

        Contour length units are nm"""
        if self.mol_is_circular:
            for num, i in enumerate(self.splined_trace):
                x1 = self.splined_trace[num - 1, 0]
                y1 = self.splined_trace[num - 1, 1]
                x2 = self.splined_trace[num, 0]
                y2 = self.splined_trace[num, 1]

                try:
                    hypotenuse_array.append(math.hypot((x1 - x2), (y1 - y2)))
                except NameError:
                    hypotenuse_array = [math.hypot((x1 - x2), (y1 - y2))]

            self.contour_length = np.sum(np.array(hypotenuse_array)) * self.pixel_to_nm_scaling
            del hypotenuse_array

        else:
            for num, i in enumerate(self.splined_trace):
                try:
                    x1 = self.splined_trace[num, 0]
                    y1 = self.splined_trace[num, 1]
                    x2 = self.splined_trace[num + 1, 0]
                    y2 = self.splined_trace[num + 1, 1]

                    try:
                        hypotenuse_array.append(math.hypot((x1 - x2), (y1 - y2)))
                    except NameError:
                        hypotenuse_array = [math.hypot((x1 - x2), (y1 - y2))]
                except IndexError:  # IndexError happens at last point in array
                    self.contour_length = np.sum(np.array(hypotenuse_array)) * self.pixel_to_nm_scaling
                    del hypotenuse_array
                    break

    def measure_end_to_end_distance(self):
        """Calculate the Euclidean distance between the start and end of linear molecules.
        The hypotenuse is calculated between the start ([0,0], [0,1]) and end ([-1,0], [-1,1]) of linear
        molecules. If the molecule is circular then the distance is set to zero (0).
        """
        if self.mol_is_circular:
            self.end_to_end_distance = 0
        else:
            x1 = self.splined_trace[0, 0]
            y1 = self.splined_trace[0, 1]
            x2 = self.splined_trace[-1, 0]
            y2 = self.splined_trace[-1, 1]
            self.end_to_end_distance = math.hypot((x1 - x2), (y1 - y2)) * self.pixel_to_nm_scaling


def trace_image(
    image: np.ndarray,
    grains_mask: np.ndarray,
    filename: str,
    pixel_to_nm_scaling: float,
    min_skeleton_size: int,
    skeletonisation_method: str,
    pad_width: int = 10,
    mask_dilation_strength: int = 4,
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
    skeletonisation_method: str
        Method of skeletonisation, options are 'zhang' (scikit-image) / 'lee' (scikit-image) / 'thin' (scikitimage) or
       'topostats' (original TopoStats method)
    pad_width: int
        Number of cells to pad arrays by, required to handle instances where grains touch the bounding box edges.
    mask_dilation_strength: int
        Number of times to apply binary dilation for the trace binary image mask. Higher will make the trace more apparent
        in the plot, but may obscure finer structure.
    cores : int
        Number of cores to process with.

    Returns
    -------
    pd.DataFrame
        Statistics from skeletonising and tracing the grains in the image.

    """
    # Check both arrays are the same shape
    assert image.shape == grains_mask.shape

    cropped_images, cropped_masks, grain_crop_coordinates = prep_arrays(image, grains_mask, pad_width)
    n_grains = len(cropped_images)
    LOGGER.info(f"[{filename}] : Calculating statistics for {n_grains} grains.")
    n_grain = 0
    results = {}
    all_global_traces = {}
    all_trace_heights = {}
    all_trace_cumulative_distances = {}
    trace_plot_data = {}
    trace_plot_data["cropped_grains"] = {}
    for cropped_image, cropped_mask in zip(cropped_images, cropped_masks):
        result = (
            trace_grain(
                cropped_image,
                cropped_mask,
                pixel_to_nm_scaling,
                filename,
                min_skeleton_size,
                skeletonisation_method,
                n_grain,
            ),
        )
        LOGGER.info(f"[{filename}] : Traced grain {n_grain + 1} of {n_grains}")

        # Get non-single value grain stats from the results
        trace_heights = result[0]["trace_heights"]
        result[0].pop("trace_heights")
        trace_cumulative_distances = result[0]["trace_cumulative_distances"]
        result[0].pop("trace_cumulative_distances")
        local_trace = result[0]["trace"]
        # Because the grains are cropped, we need to add the crop position to restore the global coordinates
        global_trace = local_trace.copy()
        for _, coord in enumerate(global_trace):
            coord[0] += grain_crop_coordinates[n_grain][0]
            coord[1] += grain_crop_coordinates[n_grain][1]
        result[0].pop("trace")

        results[n_grain] = result[0]
        all_global_traces[n_grain] = global_trace
        all_trace_heights[n_grain] = trace_heights
        all_trace_cumulative_distances[n_grain] = trace_cumulative_distances

        # Save the image and mask for the grain to return
        trace_plot_data["cropped_grains"][n_grain] = {}
        trace_plot_data["cropped_grains"][n_grain]["cropped_grain"] = cropped_image
        grain_overlay_mask = np.zeros(cropped_image.shape)
        # print(f"@@@ grain overlay mask shape: {grain_overlay_mask.shape}")
        # print(f"max trace coords: 0: {np.max(local_trace[:, 0])} 1: {np.max(local_trace[:, 1])}")
        grain_overlay_mask[local_trace[:, 0], local_trace[:, 1]] = 1
        trace_plot_data["cropped_grains"][n_grain]["grain_trace_overlay"] = grain_overlay_mask

        n_grain += 1
    try:
        results = pd.DataFrame.from_dict(results, orient="index")
        results.index.name = "molecule_number"
    except ValueError as error:
        LOGGER.error("No grains found in any images, consider adjusting your thresholds.")
        LOGGER.error(error)

    # Package up all the trace data dictionaries to return
    all_trace_data = {
        "global_traces": all_global_traces,
        "trace_heights": all_trace_heights,
        "trace_cumulative_distances": all_trace_cumulative_distances
    }

    # Main trace plot
    image_overlay_mask = np.zeros(image.shape)
    for _, trace in all_global_traces.items():
        image_overlay_mask[trace[:, 0], trace[:, 1]] = 1
    footprint = np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ]
    )
    for i in range(mask_dilation_strength):
        image_overlay_mask = skimage_morphology.binary_dilation(image_overlay_mask, footprint=footprint)
    trace_plot_data["mask"] = image_overlay_mask

    return results, all_trace_data, trace_plot_data


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
    print(f"@@@ PAD WIDTH : {pad_width}")
    # cropped_positions = [(grain.bbox[0] - pad_width, grain.bbox[1] - pad_width) for grain in region_properties]
    # Subset image and grains then zip them up
    cropped_images = []
    cropped_positions = []
    cropped_masks = []
    for grain in region_properties:
        # Crop the array, with context-based padding
        cropped_image, crop_position = crop_array(image, grain.bbox, pad_width)
        # Pad the array again just in case
        cropped_image = np.pad(cropped_image, pad_width=pad_width)
        cropped_images.append(cropped_image)
        # Record where the top left of the cropped region is, to be able to convert from local to global coords.
        cropped_positions.append((crop_position[0] - pad_width, crop_position[1] - pad_width))

        cropped_mask, _ = crop_array(labelled_grains_mask, grain.bbox, pad_width)
        cropped_mask = np.pad(cropped_mask, pad_width=pad_width)
        cropped_masks.append(cropped_mask)
    # cropped_images = [crop_array(image, grain.bbox, pad_width) for grain in region_properties]
    # cropped_images = [np.pad(grain, pad_width=pad_width) for grain in cropped_images]
    # cropped_masks, _ = [crop_array(labelled_grains_mask, grain.bbox, pad_width) for grain in region_properties]
    # cropped_masks = [np.pad(grain, pad_width=pad_width) for grain in cropped_masks]
    # Flip every labelled region to be 1 instead of its label
    cropped_masks = [np.where(grain == 0, 0, 1) for grain in cropped_masks]
    return (cropped_images, cropped_masks, cropped_positions)


def trace_grain(
    cropped_image: np.ndarray,
    cropped_mask: np.ndarray,
    pixel_to_nm_scaling: float,
    filename: str = None,
    min_skeleton_size: int = 10,
    skeletonisation_method: str = "topostats",
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
    skeletonisation_method: str
        Method of skeletonisation, options are 'zhang' (scikit-image) / 'lee' (scikit-image) / 'thin' (scikitimage) or
       'topostats' (original TopoStats method)
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
        filename=filename,
        pixel_to_nm_scaling=pixel_to_nm_scaling,
        min_skeleton_size=min_skeleton_size,
        skeletonisation_method=skeletonisation_method,
        n_grain=n_grain,
    )
    dnatrace.trace_dna()
    return {
        "image": dnatrace.filename,
        "contour_length": dnatrace.contour_length,
        "circular": dnatrace.mol_is_circular,
        "end_to_end_distance": dnatrace.end_to_end_distance,
        "trace": dnatrace.ordered_trace,
        "trace_heights": dnatrace.trace_heights,
        "trace_cumulative_distances": dnatrace.trace_cumulative_distances,
    }


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
    cropped_array = array[
        bounding_box[0] : bounding_box[2],
        bounding_box[1] : bounding_box[3],
    ]
    cropped_position = (bounding_box[0], bounding_box[1])
    return cropped_array, cropped_position


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
