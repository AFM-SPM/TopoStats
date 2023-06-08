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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage, spatial, interpolate as interp
from skimage import morphology
from skimage.filters import gaussian
import skimage.measure as skimage_measure
from tqdm import tqdm

from topostats.logs.logs import LOGGER_NAME
from topostats.tracing.skeletonize import get_skeleton
from topostats.tracing.tracingfuncs import genTracingFuncs, getSkeleton, reorderTrace

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
    """

    def __init__(
        self,
        full_image_data: np.ndarray,
        grains,
        filename: str,
        pixel_size: float,
        min_skeleton_size: int = 10,
        convert_nm_to_m: bool = True,
        skeletonisation_method: str = "zhang",
    ):
        self.full_image_data = full_image_data * 1e-9 if convert_nm_to_m else full_image_data
        # self.grains_orig = [x for row in grains for x in row]
        self.grains_orig = grains
        self.filename = filename
        self.pixel_size = pixel_size * 1e-9 if convert_nm_to_m else pixel_size
        self.min_skeleton_size = min_skeleton_size
        self.skeletonisation_method = skeletonisation_method
        # self.number_of_columns = number_of_columns
        # self.number_of_rows = number_of_rows
        self.number_of_rows = self.full_image_data.shape[0]
        self.number_of_columns = self.full_image_data.shape[1]
        self.sigma = 0.7 / (self.pixel_size * 1e9)

        self.gauss_image = None
        self.grains = grains
        self.dna_masks = None
        self.skeletons = None
        self.disordered_trace = None
        self.ordered_traces = None
        self.fitted_traces = None
        self.splined_traces = None
        self.contour_lengths = np.nan
        self.end_to_end_distance = np.nan
        self.mol_is_circular = np.nan
        self.curvature = np.nan

        self.number_of_traces = 0
        self.num_circular = 0
        self.num_linear = 0
        self.unprocessed_grains = 0
        self.neighbours = 5  # The number of neighbours used for the curvature measurement

        # supresses scipy splining warnings
        warnings.filterwarnings("ignore")

        LOGGER.debug(f"[{self.filename}] Performing DNA Tracing")

    def trace_dna(self):
        """Perform DNA tracing."""
        self.gaussian_filter()
        self.get_disordered_trace()
        # self.isMolLooped()
        # self.purge_obvious_crap()
        if self.disordered_trace is None:
            LOGGER.info(f"[{self.filename}] : Grain failed to Skeletonise")
        elif len(self.disordered_trace) >= self.min_skeleton_size:
            self.linear_or_circular(self.disordered_trace)
            self.get_ordered_traces()
            self.linear_or_circular(self.ordered_traces)
            self.get_fitted_traces()
            self.get_splined_traces()
            # self.find_curvature()
            # self.saveCurvature()
            self.measure_contour_length()
            self.measure_end_to_end_distance()
            # self.report_basic_stats()
        else:
            LOGGER.info(f"[{self.filename}] : Grain skeleton pixels < {self.min_skeleton_size}")

    def gaussian_filter(self, **kwargs) -> np.array:
        """Apply Gaussian filter"""
        self.gauss_image = gaussian(self.full_image_data, sigma=self.sigma, **kwargs)
        LOGGER.info(f"[{self.filename}] : Gaussian filter applied.")

    def get_disordered_trace(self):
        """Create a skeleton for each of the grains in the image.

        Uses my own skeletonisation function from tracingfuncs module. I will
        eventually get round to editing this function to try to reduce the branching
        and to try to better trace from looped molecules"""
        # LOOP REMOVED
        # for grain_num in sorted(self.grains.keys()):
        smoothed_grain = ndimage.binary_dilation(self.grains, iterations=1).astype(self.grains.dtype)

        sigma = 0.01 / (self.pixel_size * 1e9)
        very_smoothed_grain = ndimage.gaussian_filter(smoothed_grain, sigma)

        LOGGER.info(f"[{self.filename}] : Skeletonising using {self.skeletonisation_method} method.")
        try:
            if self.skeletonisation_method == "topostats":
                dna_skeleton = getSkeleton(
                    self.gauss_image, smoothed_grain, self.number_of_columns, self.number_of_rows, self.pixel_size
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
            LOGGER.info(f"[{self.filename}] : Grain removed due to proximity to border, consider increasing pad_width.")
            # raise e

    # def purge_obvious_crap(self):
    #     all_grains = len(self.disordered_trace)
    #     # LOOP REMOVED
    #     # for dna_num in sorted(self.disordered_trace.keys()):
    #     if len(self.disordered_trace) < self.min_skeleton_size:
    #         LOGGER.info(f"[{self.filename}] : Grain skeleton pixels < {self.min_skeleton_size}")
    #         # TODO - How to handle this?
    #     # self.unprocessed_grains = all_grains - len(self.disordered_trace)
    #     # LOGGER.info(f"[{self.filename}] : Crap grains removed : {self.unprocessed_grains}")

    def linear_or_circular(self, traces):
        """Determines whether each molecule is circular or linear based on the local environment of each pixel from the trace

        This function is sensitive to branches from the skeleton so might need to implement a function to remove them"""

        # self.num_circular = 0
        # self.num_linear = 0

        # LOOP REMOVED
        # for dna_num in sorted(traces.keys()):
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
            self.num_circular += 1
        else:
            self.mol_is_circular = False
            self.num_linear += 1

    def get_ordered_traces(self):
        # LOOP REMOVED
        # for dna_num in sorted(self.disordered_trace.keys()):
        # circle_tracing = True

        if self.mol_is_circular:
            self.ordered_traces, trace_completed = reorderTrace.circularTrace(self.disordered_trace)

            if not trace_completed:
                self.mol_is_circular = False
                try:
                    self.ordered_traces = reorderTrace.linearTrace(self.ordered_traces.tolist())
                except UnboundLocalError:
                    pass
                    # self.mol_is_circular.pop(dna_num)
                    # self.disordered_trace.pop(dna_num)
                    # self.grains.pop(dna_num)
                    # self.ordered_traces.pop(dna_num)

        elif not self.mol_is_circular:
            self.ordered_traces = reorderTrace.linearTrace(self.disordered_trace.tolist())

    # def report_basic_stats(self):
    #     """Report number of circular and linear DNA molecules detected."""
    #     LOGGER.info(
    #         f"There are {self.num_circular} circular and {self.num_linear} linear DNA molecules found in the image"
    #     )

    def get_fitted_traces(self):
        """Create trace coordinates (for each identified molecule) that are adjusted to lie
        along the highest points of each traced molecule
        """

        # LOOP REMOVED
        # for dna_num in sorted(self.ordered_traces.keys()):
        individual_skeleton = self.ordered_traces
        # This indexes a 3 nm height profile perpendicular to DNA backbone
        # note that this is a hard coded parameter
        index_width = int(3e-9 / (self.pixel_size))
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

        self.fitted_traces = fitted_coordinate_array
        del fitted_coordinate_array  # cleaned up by python anyway?

    def get_splined_traces(self):
        """Gets a splined version of the fitted trace - useful for finding the radius of gyration etc

        This function actually calculates the average of several splines which is important for getting a good fit on
        the lower res data"""

        step_size = int(7e-9 / (self.pixel_size))  # 3 nm step size
        interp_step = int(1e-10 / self.pixel_size)
        # Lets see if we just got with the pixel_to_nm_scaling
        # step_size = self.pixel_size
        # interp_step = self.pixel_size

        # FIXME : Iterate over self.fitted_traces directly use either self.fitted_traces.values() or
        # self.fitted_trace.items()
        # LOOP REMOVED
        # for dna_num in sorted(self.fitted_traces.keys()):
        self.splining_success = True
        nbr = len(self.fitted_traces[:, 0])

        # Hard to believe but some traces have less than 4 coordinates in total
        if len(self.fitted_traces[:, 1]) < 4:
            self.splined_traces = self.fitted_traces
            # continue

        # The degree of spline fit used is 3 so there cannot be less than 3 points in the splined trace
        # LOGGER.info(f"DNA Number      : {dna_num}")
        # LOGGER.info(f"nbr             : {nbr}")
        # LOGGER.info(f"step_size       : {step_size}")
        # LOGGER.info(f"self.pixel_size : {self.pixel_size}")
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
                    [self.fitted_traces[:, 0][j] for j in range(i, len(self.fitted_traces[:, 0]), step_size)]
                )
                y_sampled = np.array(
                    [self.fitted_traces[:, 1][j] for j in range(i, len(self.fitted_traces[:, 1]), step_size)]
                )

                try:
                    tck, u = interp.splprep([x_sampled, y_sampled], s=0, per=2, quiet=1, k=3)
                    out = interp.splev(ev_array, tck)
                    splined_trace = np.column_stack((out[0], out[1]))
                except ValueError:
                    # Value error occurs when the "trace fitting" really messes up the traces

                    x = np.array(
                        [self.ordered_traces[:, 0][j] for j in range(i, len(self.ordered_traces[:, 0]), step_size)]
                    )
                    y = np.array(
                        [self.ordered_traces[:, 1][j] for j in range(i, len(self.ordered_traces[:, 1]), step_size)]
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
                        self.grains.pop(dna_num)
                        self.ordered_traces.pop(dna_num)
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
            self.splined_traces = spline_average
            # else:
            #    x = self.fitted_traces[:,0]
            #    y = self.fitted_traces[:,1]

            #    try:
            #        tck, u = interp.splprep([x, y], s=0, per = 2, quiet = 1, k = 3)
            #        out = interp.splev(np.linspace(0,1,nbr*step_size), tck)
            #        splined_trace = np.column_stack((out[0], out[1]))
            #        self.splined_traces = splined_trace
            #    except ValueError: #if the trace is really messed up just delete it
            #        self.mol_is_circular.pop(dna_num)
            #        self.disordered_trace.pop(dna_num)
            #        self.grains.pop(dna_num)
            #        self.ordered_traces.pop(dna_num)

        else:
            # can't get splining of linear molecules to work yet
            self.splined_traces = self.fitted_traces

    def show_traces(self):
        plt.pcolormesh(self.gauss_image, vmax=-3e-9, vmin=3e-9)
        plt.colorbar()
        # LOOP REMOVED
        # for dna_num in sorted(self.disordered_trace.keys()):
        plt.plot(self.ordered_traces[:, 0], self.ordered_traces[:, 1], markersize=1)
        plt.plot(self.fitted_traces[:, 0], self.fitted_traces[:, 1], markersize=1)
        plt.plot(self.splined_traces[:, 0], self.splined_traces[:, 1], markersize=1)
        # print(len(self.skeletons[dna_num]), len(self.disordered_trace[dna_num]))
        # plt.plot(self.skeletons[dna_num][:,0], self.skeletons[dna_num][:,1], 'o', markersize = 0.8)

        plt.show()
        plt.close()

    def saveTraceFigures(
        self, filename: Union[str, Path], channel_name: str, vmaxval, vminval, output_dir: Union[str, Path] = None
    ):
        if directory_name:
            filename_with_ext = self._checkForSaveDirectory(filename_with_ext, directory_name)

        save_file = filename_with_ext[:-4]

        vmaxval = 20e-9
        vminval = -10e-9

        plt.pcolormesh(self.full_image_data, vmax=vmaxval, vmin=vminval)
        plt.colorbar()
        # plt.savefig("%s_%s_originalImage.png" % (save_file, channel_name))
        plt.savefig(output_dir / filename / f"{channel_name}_original.png")
        plt.close()

        plt.pcolormesh(self.full_image_data, vmax=vmaxval, vmin=vminval)
        plt.colorbar()
        # disordered_trace_list = self.ordered_traces[dna_num].tolist()
        # less_dense_trace = np.array([disordered_trace_list[i] for i in range(0,len(disordered_trace_list),5)])
        plt.plot(self.splined_traces[:, 0], self.splined_traces[:, 1], color="c", linewidth=1.0)
        if self.mol_is_circular:
            starting_point = 0
        else:
            starting_point = self.neighbours
        length = len(self.curvature)
        plt.plot(
            self.splined_traces[starting_point, 0],
            self.splined_traces[starting_point, 1],
            color="#D55E00",
            markersize=3.0,
            marker=5,
        )
        plt.plot(
            self.splined_traces[starting_point + int(length / 6), 0],
            self.splined_traces[starting_point + int(length / 6), 1],
            color="#E69F00",
            markersize=3.0,
            marker=5,
        )
        plt.plot(
            self.splined_traces[starting_point + int(length / 6 * 2), 0],
            self.splined_traces[starting_point + int(length / 6 * 2), 1],
            color="#F0E442",
            markersize=3.0,
            marker=5,
        )
        plt.plot(
            self.splined_traces[starting_point + int(length / 6 * 3), 0],
            self.splined_traces[starting_point + int(length / 6 * 3), 1],
            color="#009E74",
            markersize=3.0,
            marker=5,
        )
        plt.plot(
            self.splined_traces[starting_point + int(length / 6 * 4), 0],
            self.splined_traces[starting_point + int(length / 6 * 4), 1],
            color="#0071B2",
            markersize=3.0,
            marker=5,
        )
        plt.plot(
            self.splined_traces[starting_point + int(length / 6 * 5), 0],
            self.splined_traces[starting_point + int(length / 6 * 5), 1],
            color="#CC79A7",
            markersize=3.0,
            marker=5,
        )
        plt.savefig(f"{save_file}_{channel_name}_splinedtrace_with_markers.png")
        plt.close()

        plt.pcolormesh(self.full_image_data, vmax=vmaxval, vmin=vminval)
        plt.colorbar()
        plt.plot(self.splined_traces[:, 0], self.splined_traces[:, 1], color="c", linewidth=1.0)
        # plt.savefig("%s_%s_splinedtrace.png" % (save_file, channel_name))
        plt.savefig(output_dir / filename / f"{channel_name}_splinedtrace.png")
        LOGGER.info(f"Splined Trace image saved to : {str(output_dir / filename / f'{channel_name}_splinedtrace.png')}")
        plt.close()

        # plt.pcolormesh(self.full_image_data, vmax=vmaxval, vmin=vminval)
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

        # plt.pcolormesh(self.full_image_data, vmax=vmaxval, vmin=vminval)
        # plt.colorbar()
        # for dna_num in sorted(self.grains.keys()):
        #    grain_plt = np.argwhere(self.grains[dna_num] == 1)
        #    plt.plot(grain_plt[:, 0], grain_plt[:, 1], "o", markersize=2, color="c")
        # plt.savefig("%s_%s_grains.png" % (save_file, channel_name))
        # plt.savefig(output_dir / filename / f"{channel_name}_grains.png")
        # plt.savefig(output_dir / f"{filename}_grain.png")
        # plt.close()
        LOGGER.info(f"Grains image saved to : {str(output_dir / filename / f'{channel_name}_grains.png')}")

    # FIXME : Replace with Path() (.mkdir(parent=True, exists=True) negate need to handle errors.)
    def _checkForSaveDirectory(self, filename, new_directory_name):
        split_directory_path = os.path.split(filename)

        try:
            os.mkdir(os.path.join(split_directory_path[0], new_directory_name))
        except OSError:  # OSError happens if the directory already exists
            pass

        updated_filename = os.path.join(split_directory_path[0], new_directory_name, split_directory_path[1])

        return updated_filename

    def findWrithe(self):
        pass

    def find_curvature(self):
        # FIXME : Iterate directly over self.splined_traces.values() or self.splined_traces.items()
        # LOOP REMOVED
        # for dna_num in sorted(self.splined_traces.keys()):  # the number of molecules identified
        # splined_traces is a dictionary, where the keys are the number of the molecule, and the values are a
        # list of coordinates, in a numpy.ndarray
        # if self.mol_is_circular[dna_num]:
        curve = []
        contour = 0
        coordinates = np.zeros([2, self.neighbours * 2 + 1])
        for i, (x, y) in enumerate(self.splined_traces):
            # Extracts the coordinates for the required number of points and puts them in an array
            if self.mol_is_circular or (self.neighbours < i < len(self.splined_traces) - self.neighbours):
                for j in range(self.neighbours * 2 + 1):
                    coordinates[0][j] = self.splined_traces[i - j][0]
                    coordinates[1][j] = self.splined_traces[i - j][1]

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
                dist_real = dist * self.pixel_size
                curve.append([i, contour, (theta2 - theta1) / dist_real])

                contour = contour + math.hypot(
                    (coordinates[0][self.neighbours] - coordinates[0][self.neighbours - 1]),
                    (coordinates[1][self.neighbours] - coordinates[1][self.neighbours - 1]),
                )
            self.curvature = curve

    def saveCurvature(self):
        # FIXME : Iterate directly over self.splined_traces.values() or self.splined_traces.items()
        # roc_array = np.zeros(shape=(1, 3))
        # LOOP REMOVED
        # for dna_num in sorted(self.curvature.keys()):
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
        sns.lineplot(curvature[:, 1] * self.pixel_size, curvature[:, 2], color="k")
        plt.ylim(-1e9, 1e9)
        plt.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
        plt.axvline(curvature[0][1], color="#D55E00")
        plt.axvline(curvature[int(length / 6)][1] * self.pixel_size, color="#E69F00")
        plt.axvline(curvature[int(length / 6 * 2)][1] * self.pixel_size, color="#F0E442")
        plt.axvline(curvature[int(length / 6 * 3)][1] * self.pixel_size, color="#009E74")
        plt.axvline(curvature[int(length / 6 * 4)][1] * self.pixel_size, color="#0071B2")
        plt.axvline(curvature[int(length / 6 * 5)][1] * self.pixel_size, color="#CC79A7")
        plt.savefig(f"{savename}_{dna_num}_curvature.png")
        plt.close()

    def measure_contour_length(self):
        """Measures the contour length for each of the splined traces taking into
        account whether the molecule is circular or linear

        Contour length units are nm"""
        if self.mol_is_circular:
            for num, i in enumerate(self.splined_traces):
                x1 = self.splined_traces[num - 1, 0]
                y1 = self.splined_traces[num - 1, 1]
                x2 = self.splined_traces[num, 0]
                y2 = self.splined_traces[num, 1]

                try:
                    hypotenuse_array.append(math.hypot((x1 - x2), (y1 - y2)))
                except NameError:
                    hypotenuse_array = [math.hypot((x1 - x2), (y1 - y2))]

            self.contour_lengths = np.sum(np.array(hypotenuse_array)) * self.pixel_size
            del hypotenuse_array

        else:
            for num, i in enumerate(self.splined_traces):
                try:
                    x1 = self.splined_traces[num, 0]
                    y1 = self.splined_traces[num, 1]
                    x2 = self.splined_traces[num + 1, 0]
                    y2 = self.splined_traces[num + 1, 1]

                    try:
                        hypotenuse_array.append(math.hypot((x1 - x2), (y1 - y2)))
                    except NameError:
                        hypotenuse_array = [math.hypot((x1 - x2), (y1 - y2))]
                except IndexError:  # IndexError happens at last point in array
                    self.contour_lengths = np.sum(np.array(hypotenuse_array)) * self.pixel_size
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
            x1 = self.splined_traces[0, 0]
            y1 = self.splined_traces[0, 1]
            x2 = self.splined_traces[-1, 0]
            y2 = self.splined_traces[-1, 1]
            self.end_to_end_distance = math.hypot((x1 - x2), (y1 - y2)) * self.pixel_size
        # self.end_to_end_distance = {
        #     dna_num: math.hypot((trace[0, 0] - trace[-1, 0]), (trace[0, 1] - trace[-1, 1]))
        #     if self.mol_is_circular[dna_num]
        #     else 0
        #     for dna_num, trace in self.splined_traces.items()
        # }


def trace_image(
    image: np.ndarray,
    grains_mask: np.ndarray,
    filename: str,
    pixel_to_nm_scaling: float,
    min_skeleton_size: int,
    skeletonisation_method: str,
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
    skeletonisation_method: str
        Method of skeletonisation, options are 'zhang' (scikit-image) / 'lee' (scikit-image) / 'thin' (scikitimage) or
       'topostats' (original TopoStats method)
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

    # Get bounding boxes for each grain
    region_properties = skimage_measure.regionprops(grains_mask)
    n_grains = len(np.unique(grains_mask)) - 1
    # Subset image and grains then zip them up
    cropped_images = [crop_array(image, grain.bbox) for grain in region_properties]
    cropped_images = [np.pad(grain, pad_width=pad_width) for grain in cropped_images]
    cropped_masks = [crop_array(grains_mask, grain.bbox) for grain in region_properties]
    cropped_masks = [np.pad(grain, pad_width=pad_width) for grain in cropped_masks]
    # Flip every labelled region to be 1 instead of its label
    cropped_masks = [np.where(grain == 0, 0, 1) for grain in cropped_masks]
    LOGGER.info(f"[{filename}] : Calculating statistics for {n_grains} grains.")
    # Process in parallel
    with Pool(processes=cores) as pool:
        results = {}
        with tqdm(total=n_grains) as pbar:
            x = 0
            for result in pool.starmap(
                trace_grain,
                zip(
                    cropped_images,
                    cropped_masks,
                    repeat(pixel_to_nm_scaling),
                    repeat(filename),
                    repeat(min_skeleton_size),
                    repeat(skeletonisation_method),
                ),
            ):
                LOGGER.info(f"[{filename}] : Traced grain {x + 1} of {n_grains}")
                results[x] = result
                x += 1
                pbar.update()
    try:
        results = pd.DataFrame.from_dict(results, orient="index")
        results.index.name = "molecule_number"
    except ValueError as error:
        LOGGER.error("No grains found in any images, consider adjusting your thresholds.")
        LOGGER.error(error)
    return results


def trace_grain(
    cropped_image: np.ndarray,
    cropped_mask: np.ndarray,
    pixel_to_nm_scaling: float,
    filename: str = None,
    min_skeleton_size: int = 10,
    skeletonisation_method: str = "topostats",
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

    Returns
    =======
    Dictionary
        Dictionary of the contour length, whether the image is circular or linear, the end-to-end distance and an array
    of co-ordinates.
    """
    dnatrace = dnaTrace(
        full_image_data=cropped_image,
        grains=cropped_mask,
        filename=filename,
        pixel_size=pixel_to_nm_scaling,
        min_skeleton_size=min_skeleton_size,
        skeletonisation_method=skeletonisation_method,
    )
    dnatrace.trace_dna()
    # dnatrace.saveTraceFigures(
    #     filename=f"{filename}",
    #     channel_name="test",
    #     vmaxval=20e-9,
    #     vminval=-10e-9,
    #     output_dir=Path("/home/neil/work/projects/topostats/tmp/dnatracing_refactor/"),
    # )
    return {
        "image": dnatrace.filename,
        "contour_length": dnatrace.contour_lengths,
        "circular": dnatrace.mol_is_circular,
        "end_to_end_distance": dnatrace.end_to_end_distance,
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
    return array[
        bounding_box[0] : bounding_box[2],
        bounding_box[1] : bounding_box[3],
    ]


class traceStats:
    """Combine and save trace statistics."""

    def __init__(self, trace_object: dnaTrace, image_path: Union[str, Path]) -> None:
        """Initialise the class.

        Parameters
        ----------
        trace_object: dnaTrace
            Object produced from tracing.
        image_path: Union[str, Path]
            Path for saving images to.

        Returns
        -------
        None
        """

        self.trace_object = trace_object
        self.image_path = Path(image_path)
        self.df = []
        self.create_trace_stats()

    def create_trace_stats(self):
        """Creates a pandas dataframe of the contour length, whether its circular and end to end distance
        combined with details of the working directory, directory images were found in and the image name.
        """
        stats = OrderedDict()
        for mol_num, _ in self.trace_object.ordered_traces.items():
            stats[mol_num] = {}
            stats[mol_num]["contour_lengths"] = self.trace_object.contour_lengths[mol_num]
            stats[mol_num]["circular"] = self.trace_object.mol_is_circular[mol_num]
            stats[mol_num]["end_to_end_distance"] = self.trace_object.end_to_end_distance[mol_num]
        self.df = pd.DataFrame.from_dict(data=stats, orient="index")
        # self.df.reset_index(drop=True, inplace=True)
        self.df.index.name = "molecule_number"
        # self.df["Experiment Directory"] = str(Path().cwd())
        self.df["image"] = self.image_path.name

    def save_trace_stats(self, save_path: Union[str, Path], json: bool = True, csv: bool = True) -> None:
        """Write trace statistics to JSON and/or CSV.

        Parameters
        ----------
        save_path: Union[str, Path]
            Directory to save results to.
        json: bool
            Whether to save a JSON version of statistics.
        csv: bool
            Whether to save a CSV version of statistics.
        """
        if json:
            self.df.to_json(save_path / "tracestats.json")
            LOGGER.info(f"Saved trace info for all analysed images to: {str(save_path / 'tracestats.json')}")
        if csv:
            self.df.to_csv(save_path / "tracestats.csv")
            LOGGER.info(f"Saved trace info for all analysed images to: {str(save_path / 'tracestats.csv')}")
