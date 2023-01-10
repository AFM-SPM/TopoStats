"""Perform DNA Tracing"""
from collections import OrderedDict
import logging
from pathlib import Path
import math
import os
from typing import Union, Tuple
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage, stats, spatial, interpolate as interp
from skimage import morphology
from skimage.filters import gaussian

from topostats.logs.logs import LOGGER_NAME
from topostats.tracing.tracingfuncs import genTracingFuncs, getSkeleton, reorderTrace

LOGGER = logging.getLogger(LOGGER_NAME)


class dnaTrace(object):
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
        full_image_data,
        grains,
        filename,
        pixel_size,
        convert_nm_to_m: bool = True,
    ):
        self.full_image_data = full_image_data * 1e-9 if convert_nm_to_m else full_image_data
        # self.grains_orig = [x for row in grains for x in row]
        self.grains_orig = grains
        self.filename = filename
        self.pixel_size = pixel_size * 1e-9 if convert_nm_to_m else pixel_size
        # self.number_of_columns = number_of_columns
        # self.number_of_rows = number_of_rows
        self.number_of_rows = self.full_image_data.shape[0]
        self.number_of_columns = self.full_image_data.shape[1]
        self.sigma = 0.7 / (self.pixel_size * 1e9)

        self.gauss_image = []
        self.grains = {}
        self.dna_masks = {}
        self.skeletons = {}
        self.disordered_traces = {}
        self.ordered_traces = {}
        self.fitted_traces = {}
        self.splined_traces = {}
        self.simplified_splined_traces = {}
        self.contour_lengths = {}
        self.end_to_end_distance = {}
        self.mol_is_circular = {}
        self.curvature = {}
        self.max_curvature = {}
        self.max_curvature_location = {}
        self.mean_curvature = {}
        self.curvature_variance = {}
        self.curvature_variance_abs = {}
        self.central_curvature = {}
        self.central_max_curvature = {}
        self.central_max_curvature_location = {}
        self.bending_angle = {}

        self.number_of_traces = 0
        self.num_circular = 0
        self.num_linear = 0

        self.n_points = 200
        self.displacement = 0
        self.major = float(5)
        self.minor = float(1)
        self.step_size_m = 7e-9  # Used for getting the splines
        # supresses scipy splining warnings
        warnings.filterwarnings("ignore")

        LOGGER.info("Performing DNA Tracing")

    def trace_dna(self):
        """Perform DNA tracing."""
        self.get_numpy_arrays()
        self.gaussian_filter()
        self.get_disordered_traces()
        # self.isMolLooped()
        self.purge_obvious_crap()
        self.linear_or_circular(self.disordered_traces)
        self.get_ordered_traces()
        self.linear_or_circular(self.ordered_traces)
        self.get_fitted_traces()
        self.get_splined_traces()
        self.find_curvature()
        self.save_curvature()
        self.analyse_curvature()
        self.measure_contour_length()
        self.measure_end_to_end_distance()
        self.measure_bending_angle()
        self.report_basic_stats()

    def get_numpy_arrays(self):

        """Function to get each grain as a numpy array which is stored in a
        dictionary

        Currently the grains are unnecessarily large (the full image) as I don't
        know how to handle the cropped versions

        I find using the gwyddion objects clunky and not helpful once the
        grains have been found

        There is some kind of discrepency between the ordering of arrays from
        gwyddion and how they're usually handled in np arrays meaning you need
        to be careful when indexing from gwyddion derived numpy arrays"""
        for grain_num in set(self.grains_orig.flatten()):
            # Skip the background
            if grain_num != 0:
                # Saves each grain as a multidim numpy array
                # single_grain_1d = np.array([1 if i == grain_num else 0 for i in self.grains_orig])
                # self.grains[int(grain_num)] = np.reshape(single_grain_1d, (self.number_of_columns, self.number_of_rows))
                self.grains[int(grain_num)] = self._get_grain_array(grain_num)
        # FIXME : This should be a method of its own, but strange that apparently Gaussian filtered image is filtered again
        # Get a 7 A gauss filtered version of the original image
        # used in refining the pixel positions in fitted_traces()
        # sigma = 0.7 / (self.pixel_size * 1e9)
        # self.gauss_image = filters.gaussian(self.full_image_data, sigma)

    def _get_grain_array(self, grain_number: int) -> np.ndarray:
        """Extract a single grains."""
        return np.where(self.grains_orig == grain_number, 1, 0)

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

    def gaussian_filter(self, **kwargs) -> np.array:
        """Apply Gaussian filter"""
        self.gauss_image = gaussian(self.full_image_data, sigma=self.sigma, **kwargs)
        LOGGER.info(f"[{self.filename}] : Gaussian filter applied.")

    def get_disordered_traces(self):
        """Create a skeleton for each of the grains in the image.

        Uses my own skeletonisation function from tracingfuncs module. I will
        eventually get round to editing this function to try to reduce the branching
        and to try to better trace from looped molecules"""

        for grain_num in sorted(self.grains.keys()):
            # What purpose does binary dilation serve here? Name suggests some sort of smoothing around the edges and
            # the resulting array is used as a mask during the skeletonising process.
            smoothed_grain = ndimage.binary_dilation(self.grains[grain_num], iterations=1).astype(
                self.grains[grain_num].dtype
            )

            sigma = 0.01 / (self.pixel_size * 1e9)
            very_smoothed_grain = ndimage.gaussian_filter(smoothed_grain, sigma)

            try:
                dna_skeleton = getSkeleton(
                    self.gauss_image, smoothed_grain, self.number_of_columns, self.number_of_rows, self.pixel_size
                )
                self.disordered_traces[grain_num] = dna_skeleton.output_skeleton
            except IndexError:
                # Some gwyddion grains touch image border causing IndexError
                # These grains are deleted
                self.grains.pop(grain_num)
            # skel = morphology.skeletonize(self.grains[grain_num])
            # self.skeletons[grain_num] = np.argwhere(skel == 1)

    def purge_obvious_crap(self):

        for dna_num in sorted(self.disordered_traces.keys()):

            if len(self.disordered_traces[dna_num]) < 10:
                self.disordered_traces.pop(dna_num, None)

    def linear_or_circular(self, traces):

        """Determines whether each molecule is circular or linear based on the local environment of each pixel from the trace

        This function is sensitive to branches from the skeleton so might need to implement a function to remove them"""

        self.num_circular = 0
        self.num_linear = 0

        for dna_num in sorted(traces.keys()):

            points_with_one_neighbour = 0
            fitted_trace_list = traces[dna_num].tolist()

            # For loop determines how many neighbours a point has - if only one it is an end
            for x, y in fitted_trace_list:

                if genTracingFuncs.countNeighbours(x, y, fitted_trace_list) == 1:
                    points_with_one_neighbour += 1
                else:
                    pass

            if points_with_one_neighbour == 0:
                self.mol_is_circular[dna_num] = True
                self.num_circular += 1
            else:
                self.mol_is_circular[dna_num] = False
                self.num_linear += 1

    def get_ordered_traces(self):

        for dna_num in sorted(self.disordered_traces.keys()):

            circle_tracing = True

            if self.mol_is_circular[dna_num]:

                self.ordered_traces[dna_num], trace_completed = reorderTrace.circularTrace(
                    self.disordered_traces[dna_num]
                )

                if not trace_completed:
                    self.mol_is_circular[dna_num] = False
                    try:
                        self.ordered_traces[dna_num] = reorderTrace.linearTrace(self.ordered_traces[dna_num].tolist())
                    except UnboundLocalError:
                        self.mol_is_circular.pop(dna_num)
                        self.disordered_traces.pop(dna_num)
                        self.grains.pop(dna_num)
                        self.ordered_traces.pop(dna_num)

            elif not self.mol_is_circular[dna_num]:
                self.ordered_traces[dna_num] = reorderTrace.linearTrace(self.disordered_traces[dna_num].tolist())

    def report_basic_stats(self):
        """Report number of circular and linear DNA molecules detected."""
        LOGGER.info(
            f"There are {self.num_circular} circular and {self.num_linear} linear DNA molecules found in the image"
        )

    def get_fitted_traces(self):
        """Create trace coordinates (for each identified molecule) that are adjusted to lie
        along the highest points of each traced molecule
        """

        for dna_num in sorted(self.ordered_traces.keys()):

            individual_skeleton = self.ordered_traces[dna_num]

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
                if self.mol_is_circular[dna_num]:
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
                height_values = self.gauss_image[perp_array[:, 1], perp_array[:, 0]]

                """
                # Old code that interpolated the height profile for "sub-pixel
                # accuracy" - probably slow and not necessary, can delete

                #Use interpolation to get "sub pixel" accuracy for heighest position
                if perp_direction == 'negative diaganol':
                    int_func = interp.interp1d(perp_array[:,0], np.ndarray.flatten(height_values), kind = 'cubic')
                    interp_heights = int_func(np.arange(perp_array[0,0], perp_array[-1,0], 0.1))

                elif perp_direction == 'positive diaganol':
                    int_func = interp.interp1d(perp_array[:,0], np.ndarray.flatten(height_values), kind = 'cubic')
                    interp_heights = int_func(np.arange(perp_array[0,0], perp_array[-1,0], 0.1))

                elif perp_direction == 'vertical':
                    int_func = interp.interp1d(perp_array[:,1], np.ndarray.flatten(height_values), kind = 'cubic')
                    interp_heights = int_func(np.arange(perp_array[0,1], perp_array[-1,1], 0.1))

                elif perp_direction == 'horizontal':
                    #print(perp_array[:,0])
                    #print(np.ndarray.flatten(height_values))
                    int_func = interp.interp1d(perp_array[:,0], np.ndarray.flatten(height_values), kind = 'cubic')
                    interp_heights = int_func(np.arange(perp_array[0,0], perp_array[-1,0], 0.1))
                else:
                    quit('A fatal error occured in the CorrectHeightPositions function, this was likely caused by miscalculating vector angles')

                #Make "fine" coordinates which have the same number of coordinates as the interpolated height values
                if perp_direction == 'negative diaganol':
                    fine_x_coords = np.arange(perp_array[0,0], perp_array[-1,0], 0.1)
                    fine_y_coords = np.arange(perp_array[-1,1], perp_array[0,1], 0.1)[::-1]
                elif perp_direction == 'positive diaganol':
                    fine_x_coords = np.arange(perp_array[0,0], perp_array[-1,0], 0.1)
                    fine_y_coords = np.arange(perp_array[0,1], perp_array[-1,1], 0.1)
                elif perp_direction == 'vertical':
                    fine_y_coords = np.arange(perp_array[0,1], perp_array[-1,1], 0.1)
                    fine_x_coords = np.full(len(fine_y_coords), trace_coordinate[0], dtype = 'float')
                elif perp_direction == 'horizontal':
                    fine_x_coords = np.arange(perp_array[0,0], perp_array[-1,0], 0.1)
                    fine_y_coords = np.full(len(fine_x_coords), trace_coordinate[1], dtype = 'float')
                """
                # Grab x,y coordinates for highest point
                # fine_coords = np.column_stack((fine_x_coords, fine_y_coords))
                sorted_array = perp_array[np.argsort(height_values)]
                highest_point = sorted_array[-1]

                try:
                    # could use np.append() here
                    fitted_coordinate_array = np.vstack((fitted_coordinate_array, highest_point))
                except UnboundLocalError:
                    fitted_coordinate_array = highest_point

            self.fitted_traces[dna_num] = fitted_coordinate_array
            del fitted_coordinate_array  # cleaned up by python anyway?

    def get_splined_traces(self):
        """Gets a splined version of the fitted trace - useful for finding the radius of gyration etc

        This function actually calculates the average of several splines which is important for getting a good fit on
        the lower res data"""

        step_size_px = int(self.step_size_m / (self.pixel_size))

        # FIXME : Iterate over self.fitted_traces directly use either self.fitted_traces.values() or self.fitted_trace.items()
        for dna_num in sorted(self.fitted_traces.keys()):

            self.splining_success = True
            nbr = len(self.fitted_traces[dna_num][:, 0])

            # Hard to believe but some traces have less than 4 coordinates in total
            if len(self.fitted_traces[dna_num][:, 1]) < 4:
                self.splined_traces[dna_num] = self.fitted_traces[dna_num]
                continue

            # The degree of spline fit used is 3 so there cannot be less than 3 points in the splined trace
            # LOGGER.info(f"DNA Number      : {dna_num}")
            # LOGGER.info(f"nbr             : {nbr}")
            # LOGGER.info(f"step_size       : {step_size}")
            # LOGGER.info(f"self.pixel_size : {self.pixel_size}")
            while nbr / step_size_px < 4:
                if step_size_px <= 1:
                    step_size_px = 1
                    break
                step_size_px = -1

            if self.mol_is_circular[dna_num]:

                ev_array = np.linspace(0, 1, nbr * step_size_px)

                for i in range(step_size_px):
                    x_sampled = np.array(
                        [
                            self.fitted_traces[dna_num][:, 0][j]
                            for j in range(i, len(self.fitted_traces[dna_num][:, 0]), step_size_px)
                        ]
                    )
                    y_sampled = np.array(
                        [
                            self.fitted_traces[dna_num][:, 1][j]
                            for j in range(i, len(self.fitted_traces[dna_num][:, 1]), step_size_px)
                        ]
                    )

                    try:
                        tck, u = interp.splprep([x_sampled, y_sampled], s=2, per=2, quiet=1, k=3)
                        out = interp.splev(ev_array, tck)
                        splined_trace = np.column_stack((out[0], out[1]))
                    except ValueError:
                        # Value error occurs when the "trace fitting" really messes up the traces

                        x = np.array(
                            [
                                self.ordered_traces[dna_num][:, 0][j]
                                for j in range(i, len(self.ordered_traces[dna_num][:, 0]), step_size_px)
                            ]
                        )
                        y = np.array(
                            [
                                self.ordered_traces[dna_num][:, 1][j]
                                for j in range(i, len(self.ordered_traces[dna_num][:, 1]), step_size_px)
                            ]
                        )

                        try:
                            tck, u = interp.splprep([x, y], s=2, per=2, quiet=1)
                            out = interp.splev(np.linspace(0, 1, nbr * step_size_px), tck)
                            splined_trace = np.column_stack((out[0], out[1]))
                        except ValueError:  # sometimes even the ordered_traces are too bugged out so just delete these traces
                            self.mol_is_circular.pop(dna_num)
                            self.disordered_traces.pop(dna_num)
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

                if not self.splining_success:
                    continue

                spline_average = np.divide(spline_running_total, [step_size_px, step_size_px])
                del spline_running_total
                spline_average = np.delete(spline_average, -1, 0)
                self.splined_traces[dna_num] = spline_average

            else:
                # ev_array = np.linspace(0, 1, 1000)
                ev_array = np.linspace(0, 1, nbr * step_size_px)

                for i in range(step_size_px):
                    x_sampled = np.array(
                        [
                            self.fitted_traces[dna_num][:, 0][j]
                            for j in range(i, len(self.fitted_traces[dna_num][:, 0]), step_size_px)
                        ]
                    )
                    y_sampled = np.array(
                        [
                            self.fitted_traces[dna_num][:, 1][j]
                            for j in range(i, len(self.fitted_traces[dna_num][:, 1]), step_size_px)
                        ]
                    )

                    try:
                        tck, u = interp.splprep([x_sampled, y_sampled], s=5, per=0, quiet=1, k=3)
                        out = interp.splev(ev_array, tck)
                        splined_trace = np.column_stack((out[0], out[1]))
                    except ValueError:
                        # Value error occurs when the "trace fitting" really messes up the traces

                        x = np.array(
                            [
                                self.ordered_traces[dna_num][:, 0][j]
                                for j in range(i, len(self.ordered_traces[dna_num][:, 0]), step_size_px)
                            ]
                        )
                        y = np.array(
                            [
                                self.ordered_traces[dna_num][:, 1][j]
                                for j in range(i, len(self.ordered_traces[dna_num][:, 1]), step_size_px)
                            ]
                        )

                        try:
                            tck, u = interp.splprep([x, y], s=5, per=0, quiet=1)
                            out = interp.splev(np.linspace(0, 1, nbr * step_size_px), tck)
                            splined_trace = np.column_stack((out[0], out[1]))
                        except ValueError:  # sometimes even the ordered_traces are too bugged out so just delete these traces
                            self.mol_is_circular.pop(dna_num)
                            self.disordered_traces.pop(dna_num)
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

                if not self.splining_success:
                    continue

                spline_average = np.divide(spline_running_total, [step_size_px, step_size_px])
                del spline_running_total
                self.splined_traces[dna_num] = spline_average
            self.simplified_splined_traces[dna_num] = self.splined_traces[dna_num][::1]

    def show_traces(self):

        plt.pcolormesh(self.gauss_image, vmax=-3e-9, vmin=3e-9)
        plt.colorbar()
        for dna_num in sorted(self.disordered_traces.keys()):
            plt.plot(self.ordered_traces[dna_num][:, 0], self.ordered_traces[dna_num][:, 1], markersize=1)
            plt.plot(self.fitted_traces[dna_num][:, 0], self.fitted_traces[dna_num][:, 1], markersize=1)
            plt.plot(self.splined_traces[dna_num][:, 0], self.splined_traces[dna_num][:, 1], markersize=1)
            # print(len(self.skeletons[dna_num]), len(self.disordered_traces[dna_num]))
            # plt.plot(self.skeletons[dna_num][:,0], self.skeletons[dna_num][:,1], 'o', markersize = 0.8)
        plt.axis("equal")
        plt.show()
        plt.close()

    def saveTraceFigures(
        self, filename: Union[str, Path], channel_name: str, vmaxval, vminval, output_dir: Union[str, Path] = None
    ):

        # vmaxval = 20e-9
        # vminval = -10e-9
        LOGGER.info(f"[{filename}] : Saving trace figures")
        plt.pcolormesh(self.full_image_data, vmax=vmaxval, vmin=vminval)
        plt.colorbar()
        plt.savefig(output_dir / filename / f"{channel_name}_original.png")
        plt.close()

        plt.pcolormesh(self.full_image_data, vmax=vmaxval, vmin=vminval)
        plt.colorbar()
        for dna_num in sorted(self.splined_traces.keys()):
            plt.plot(self.splined_traces[dna_num][:, 0], self.splined_traces[dna_num][:, 1], color="c", linewidth=2.0)
        plt.savefig(output_dir / filename / f"{channel_name}_splinedtrace.png")
        LOGGER.info(f"Splined Trace image saved to : {str(output_dir / filename / f'{channel_name}_splinedtrace.png')}")
        plt.close()

        plt.pcolormesh(self.full_image_data, vmax=vmaxval, vmin=vminval)
        plt.colorbar()
        for dna_num in sorted(self.splined_traces.keys()):
            plt.plot(self.splined_traces[dna_num][:, 0], self.splined_traces[dna_num][:, 1], color="c", linewidth=1.0)
            length = len(self.curvature[dna_num])
            plt.plot(
                self.splined_traces[dna_num][0, 0],
                self.splined_traces[dna_num][0, 1],
                color="#D55E00",
                markersize=3.0,
                marker=5,
            )
            plt.plot(
                self.splined_traces[dna_num][int(length / 6), 0],
                self.splined_traces[dna_num][int(length / 6), 1],
                color="#E69F00",
                markersize=3.0,
                marker=5,
            )
            plt.plot(
                self.splined_traces[dna_num][int(length / 6 * 2), 0],
                self.splined_traces[dna_num][int(length / 6 * 2), 1],
                color="#F0E442",
                markersize=3.0,
                marker=5,
            )
            plt.plot(
                self.splined_traces[dna_num][int(length / 6 * 3), 0],
                self.splined_traces[dna_num][int(length / 6 * 3), 1],
                color="#009E74",
                markersize=3.0,
                marker=5,
            )
            plt.plot(
                self.splined_traces[dna_num][int(length / 6 * 4), 0],
                self.splined_traces[dna_num][int(length / 6 * 4), 1],
                color="#56B4E9",
                markersize=3.0,
                marker=5,
            )
            plt.plot(
                self.splined_traces[dna_num][int(length / 6 * 5), 0],
                self.splined_traces[dna_num][int(length / 6 * 5), 1],
                color="#CC79A7",
                markersize=3.0,
                marker=5,
            )
        plt.savefig(output_dir / filename / f"{channel_name}_splined_trace_with_markers.png")
        plt.close()
        LOGGER.info(
            f"Splined trace with markers image saved to : {str(output_dir / filename / f'{channel_name}_splined_trace_with_markers.png')}"
        )

        plt.pcolormesh(self.full_image_data, vmax=vmaxval, vmin=vminval)
        plt.colorbar()
        for dna_num in sorted(self.ordered_traces.keys()):
            plt.plot(
                self.ordered_traces[dna_num][:, 0], self.ordered_traces[dna_num][:, 1], "o", markersize=0.5, color="c"
            )
        plt.savefig(output_dir / filename / f"{channel_name}_ordered_trace.png")
        plt.close()
        LOGGER.info(
            f"Ordered trace image saved to : {str(output_dir / filename / f'{channel_name}_ordered_trace.png')}"
        )

        plt.pcolormesh(self.full_image_data, vmax=vmaxval, vmin=vminval)
        plt.colorbar()
        for dna_num in sorted(self.disordered_traces.keys()):
            # disordered_traces_list = self.disordered_traces[dna_num].tolist()
            # less_dense_trace = np.array([disordered_traces_list[i] for i in range(0,len(disordered_traces_list),5)])
            plt.plot(
                self.disordered_traces[dna_num][:, 0],
                self.disordered_traces[dna_num][:, 1],
                "o",
                markersize=0.5,
                color="c",
            )
        plt.savefig(output_dir / filename / f"{channel_name}_disordered_traces.png")
        plt.close()
        LOGGER.info(
            f"Disordered trace image saved to : {str(output_dir / filename / f'{channel_name}_disordered_traces.png')}"
        )

        plt.pcolormesh(self.full_image_data, vmax=vmaxval, vmin=vminval)
        plt.colorbar()
        for dna_num in sorted(self.fitted_traces.keys()):
            plt.plot(
                self.fitted_traces[dna_num][:, 0], self.fitted_traces[dna_num][:, 1], "o", markersize=0.5, color="c"
            )
        plt.savefig(output_dir / filename / f"{channel_name}_fitted_trace.png")
        plt.close()
        LOGGER.info(f"Fitted trace image saved to : {str(output_dir / filename / f'{channel_name}_fitted_trace.png')}")

        plt.pcolormesh(self.full_image_data, vmax=vmaxval, vmin=vminval)
        plt.colorbar()
        for dna_num in sorted(self.grains.keys()):
            grain_plt = np.argwhere(self.grains[dna_num] == 1)
            plt.plot(grain_plt[:, 0], grain_plt[:, 1], "o", markersize=2, color="c")
        plt.savefig(output_dir / filename / f"{channel_name}_grains.png")
        plt.close()
        LOGGER.info(f"Grains image saved to : {str(output_dir / filename / f'{channel_name}_grains.png')}")

        for dna_num in sorted(self.ordered_traces.keys()):
            plt.scatter(
                x=self.simplified_splined_traces[dna_num][:, 0],
                y=self.simplified_splined_traces[dna_num][:, 1],
                c=self.curvature[dna_num][:, 2],
                s=1,
            )

        plt.colorbar()
        plt.axis("equal")
        plt.savefig(output_dir / filename / f"{channel_name}_curvature_summary.png")
        plt.close()
        LOGGER.info(
            f"Curvature summary saved to : {str(output_dir / filename / f'{channel_name}_curvature_summary.png')}"
        )

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

        # Testing with a circle
        # radius = float(1)
        # self.splined_traces[0] = np.zeros([self.n_points, 2])
        # for i in range(self.n_points):
        #     theta = 2 * math.pi / self.n_points * i
        #     x = - math.cos(theta) * radius
        #     y = math.sin(theta) * radius
        #     self.splined_traces[0][i][0] = x
        #     self.splined_traces[0][i][1] = y
        # self.mol_is_circular[0] = True
        #
        # self.splined_traces[101] = self.splined_traces[1]*5
        # self.mol_is_circular[101] = True

        # Testing with an ellipse
        # self.splined_traces[0] = np.zeros([self.n_points, 2])
        # for i in range(0, self.n_points):
        #     theta = 2 * math.pi / self.n_points * i
        #     x = - math.cos(theta) * self.major
        #     y = math.sin(theta) * self.minor
        #     self.splined_traces[0][i][0] = x
        #     self.splined_traces[0][i][1] = y
        # self.splined_traces[0] = np.roll(self.splined_traces[0], int(self.n_points * self.displacement), axis=0)
        # self.mol_is_circular[0] = True

        # Testing with a parabola
        # x = np.linspace(-2, 2, num=self.n_points)
        # y = x**2
        # self.splined_traces[0] = np.column_stack((x, y))
        # self.mol_is_circular[0] = False

        # FIXME : Iterate directly over self.splined_traces.values() or self.splined_traces.items()
        for dna_num in sorted(self.simplified_splined_traces.keys()):  # the number of molecules identified
            # splined_traces is a dictionary, where the keys are the number of the molecule, and the values are a
            # list of coordinates, in a numpy.ndarray
            length = len(self.simplified_splined_traces[dna_num])
            curve = []
            contour = 0
            # coordinates = np.zeros([2, self.neighbours * 2 + 1])
            # dxmean = np.zeros(length)
            # dymean = np.zeros(length)
            # gradients = np.zeros([2, self.neighbours * 2 + 1])
            if self.mol_is_circular[dna_num]:
                longlist = np.concatenate(
                    [
                        self.simplified_splined_traces[dna_num],
                        self.simplified_splined_traces[dna_num],
                        self.simplified_splined_traces[dna_num],
                    ]
                )
                dx = np.gradient(longlist, axis=0)[:, 0]
                dy = np.gradient(longlist, axis=0)[:, 1]
                d2x = np.gradient(dx)
                d2y = np.gradient(dy)

                dx = dx[length : 2 * length]
                dy = dy[length : 2 * length]
                d2x = d2x[length : 2 * length]
                d2y = d2y[length : 2 * length]
            else:
                dx = np.gradient(self.simplified_splined_traces[dna_num], axis=0, edge_order=2)[:, 0]
                dy = np.gradient(self.simplified_splined_traces[dna_num], axis=0, edge_order=2)[:, 1]
                d2x = np.gradient(dx)
                d2y = np.gradient(dy)

            for i, (x, y) in enumerate(self.simplified_splined_traces[dna_num]):
                # Extracts the coordinates for the required number of points and puts them in an array
                curvature_local = (d2x[i] * dy[i] - dx[i] * d2y[i]) / (dx[i] ** 2 + dy[i] ** 2) ** 1.5
                curve.append([i, contour, curvature_local, dx[i], dy[i], d2x[i], d2y[i]])
                if i < (length - 1):
                    contour = contour + self.pixel_size * 1e9 * math.hypot(
                        (
                            self.simplified_splined_traces[dna_num][(i + 1), 0]
                            - self.simplified_splined_traces[dna_num][i, 0]
                        ),
                        (
                            self.simplified_splined_traces[dna_num][(i + 1), 1]
                            - self.simplified_splined_traces[dna_num][i, 1]
                        ),
                    )
            curve = np.array(curve)
            # curvature_smoothed = scipy.ndimage.gaussian_filter(curve[:, 2], 10, mode='nearest')
            # curve[:, 2] = curvature_smoothed
            self.curvature[dna_num] = curve

    def save_curvature(self):

        # FIXME : Iterate directly over self.splined_traces.values() or self.splined_traces.items()
        for dna_num in sorted(self.curvature.keys()):
            for i, [n, contour, c, dx, dy, d2x, d2y] in enumerate(self.curvature[dna_num]):
                try:
                    curvature_array = np.append(
                        curvature_array, np.array([[dna_num, i, contour, c, dx, dy, d2x, d2y]]), axis=0
                    )
                except NameError:
                    curvature_array = np.array([[dna_num, i, contour, c, dx, dy, d2x, d2y]])
        curvature_stats = pd.DataFrame(curvature_array)
        curvature_stats.columns = ["DNA number", "Number", "Contour length", "Curvature", "dx", "dy", "d2x", "d2y"]

        # FIXME : Replace with Path()
        if not os.path.exists(os.path.join(os.path.dirname(self.filename), "Curvature")):
            os.mkdir(os.path.join(os.path.dirname(self.filename), "Curvature"))
        directory = os.path.join(os.path.dirname(self.filename), "Curvature")
        savename = os.path.join(directory, os.path.basename(self.filename)[:-4])
        curvature_stats.to_csv(savename + ".csv")

    def analyse_curvature(self):
        """Calculate curvature related statistics for each molecule, including max curvature, max curvature location,
        mean value of absolute curvature, variance of curvature, and variance of absolute curvature"""
        for dna_num in sorted(self.curvature.keys()):
            self.max_curvature[dna_num] = np.amax(np.abs(self.curvature[dna_num][:, 2]))
            max_index = np.argmax(np.abs(self.curvature[dna_num][:, 2]))
            self.max_curvature_location[dna_num] = self.curvature[dna_num][max_index, 1]
            self.mean_curvature[dna_num] = np.average(np.abs(self.curvature[dna_num][:, 2]))
            self.curvature_variance[dna_num] = np.var(self.curvature[dna_num][:, 2])
            self.curvature_variance_abs[dna_num] = np.var(np.abs(self.curvature[dna_num][:, 2]))

    def plot_curvature(
        self, dna_num, filename: Union[str, Path], channel_name: str, output_dir: Union[str, Path] = None
    ):

        """Plot the curvature of the chosen molecule as a function of the contour length (in metres). The molecule
        number needs to be specified when calling the method."""

        curvature = np.array(self.curvature[dna_num])
        length = len(curvature)
        plt.figure()
        # fig, ax = plt.subplots(figsize=(25, 25))

        # if dna_num == 0:
        #     plt.ylim(0, 2)
        if dna_num == 0:
            theory = np.zeros(length)
            for i in range(length):
                # For ellipse
                theory[i] = (
                    self.major
                    * self.minor
                    * (
                        -1
                        / (
                            (self.major**2 - self.minor**2) * math.cos(math.pi * 2 / self.n_points * i) ** 2
                            - self.major**2
                        )
                    )
                    ** 1.5
                )
                theory = np.roll(theory, int(self.n_points * self.displacement), axis=0)
                # For parabola
                # theory[i] = - 2/(1+(2*self.splined_traces[0][i][0])**2)**1.5
            sns.lineplot(x=curvature[:, 1] * self.pixel_size * 1e9, y=theory, color="b")
            sns.lineplot(x=curvature[:, 1] * self.pixel_size * 1e9, y=curvature[:, 2], color="y")
        else:
            # plt.xlim(0, 105)
            # plt.ylim(-0.1, 0.2)
            sns.lineplot(x=curvature[:, 1], y=curvature[:, 2], color="black", linewidth=5)
            plt.ticklabel_format(axis="both", style="sci", scilimits=(-2, 2))
            plt.axvline(curvature[0][1], color="#D55E00", linewidth=5, alpha=0.8)
            plt.axvline(curvature[int(length / 6)][1], color="#E69F00", linewidth=5, alpha=0.8)
            plt.axvline(curvature[int(length / 6 * 2)][1], color="#F0E442", linewidth=5, alpha=0.8)
            plt.axvline(curvature[int(length / 6 * 3)][1], color="#009E74", linewidth=5, alpha=0.8)
            plt.axvline(curvature[int(length / 6 * 4)][1], color="#56B4E9", linewidth=5, alpha=0.8)
            plt.axvline(curvature[int(length / 6 * 5)][1], color="#CC79A7", linewidth=5, alpha=0.8)
        plt.xlabel("Length alongside molecule / nm")
        plt.ylabel("Curvature / $\mathregular{nm^{-1}}$")
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        plt.savefig(output_dir / filename / f"{channel_name}_{dna_num}_curvature.png")
        plt.close()
        LOGGER.info(
            f"Curvature plot for molecule {dna_num} saved to : {str(output_dir / filename / f'{channel_name}_{dna_num}_curvature.png')}"
        )

    def plot_gradient(
        self, dna_num, filename: Union[str, Path], channel_name: str, output_dir: Union[str, Path] = None
    ):

        """Plot the first and second order gradients of the chosen molecule as a function of the contour length (in
        metres). The molecule number needs to be specified when calling the method."""

        curvature = np.array(self.curvature[dna_num])
        plt.figure()
        plt.plot(curvature[:, 1] * self.pixel_size, curvature[:, 3], color="r")
        plt.plot(curvature[:, 1] * self.pixel_size, curvature[:, 4], color="b")
        plt.savefig(output_dir / filename / f"{channel_name}_{dna_num}_gradient.png")
        plt.close()

        plt.figure()
        plt.plot(curvature[:, 1] * self.pixel_size, curvature[:, 5], color="r")
        plt.plot(curvature[:, 1] * self.pixel_size, curvature[:, 6], color="b")
        plt.savefig(output_dir / filename / f"{channel_name}_{dna_num}_second_order.png")
        plt.close()

    def measure_contour_length(self):

        """Measures the contour length for each of the splined traces taking into
        account whether the molecule is circular or linear

        Contour length units are nm"""

        for dna_num in sorted(self.splined_traces.keys()):

            if self.mol_is_circular[dna_num]:
                for num, i in enumerate(self.splined_traces[dna_num]):

                    x1 = self.splined_traces[dna_num][num - 1, 0]
                    y1 = self.splined_traces[dna_num][num - 1, 1]
                    x2 = self.splined_traces[dna_num][num, 0]
                    y2 = self.splined_traces[dna_num][num, 1]

                    try:
                        hypotenuse_array.append(math.hypot((x1 - x2), (y1 - y2)))
                    except NameError:
                        hypotenuse_array = [math.hypot((x1 - x2), (y1 - y2))]

                self.contour_lengths[dna_num] = np.sum(np.array(hypotenuse_array)) * self.pixel_size * 1e9
                del hypotenuse_array

            else:
                for num, i in enumerate(self.splined_traces[dna_num]):
                    try:
                        x1 = self.splined_traces[dna_num][num, 0]
                        y1 = self.splined_traces[dna_num][num, 1]
                        x2 = self.splined_traces[dna_num][num + 1, 0]
                        y2 = self.splined_traces[dna_num][num + 1, 1]

                        try:
                            hypotenuse_array.append(math.hypot((x1 - x2), (y1 - y2)))
                        except NameError:
                            hypotenuse_array = [math.hypot((x1 - x2), (y1 - y2))]
                    except IndexError:  # IndexError happens at last point in array
                        self.contour_lengths[dna_num] = np.sum(np.array(hypotenuse_array)) * self.pixel_size * 1e9
                        del hypotenuse_array
                        break

    def write_contour_lengths(self, filename, channel_name):

        if not self.contour_lengths:
            self.measure_contour_length()

        with open(f"{filename}_{channel_name}_contours.txt", "w") as writing_file:
            writing_file.write("#units: nm\n")
            for dna_num in sorted(self.contour_lengths.keys()):
                writing_file.write("%f \n" % self.contour_lengths[dna_num])

    def write_coordinates(
        self, dna_num, filename: Union[str, Path], channel_name: str, output_dir: Union[str, Path] = None
    ):
        # FIXME: Replace with Path()
        # if not os.path.exists(os.path.join(os.path.dirname(self.filename), "Coordinates")):
        #     os.mkdir(os.path.join(os.path.dirname(self.filename), "Coordinates"))
        # directory = os.path.join(os.path.dirname(self.filename), "Coordinates")
        # savename = os.path.join(directory, os.path.basename(self.filename)[:-4])
        for i, (x, y) in enumerate(self.splined_traces[dna_num]):
            try:
                coordinates_array = np.append(coordinates_array, np.array([[x, y]]), axis=0)
            except NameError:
                coordinates_array = np.array([[x, y]])

        coordinates = pd.DataFrame(coordinates_array)
        # coordinates.to_csv("%s_%s.csv" % (savename, dna_num))
        coordinates.to_csv(output_dir / "Coordinates.csv")

        plt.plot(coordinates_array[:, 0], coordinates_array[:, 1], "k.", markersize=5)
        plt.axis("equal")
        length = len(coordinates_array)
        plt.plot(coordinates_array[0, 0], coordinates_array[0, 1], color="#D55E00", markersize=10, marker="o")
        plt.plot(
            coordinates_array[int(length / 6), 0],
            coordinates_array[int(length / 6), 1],
            color="#E69F00",
            markersize=10,
            marker="o",
        )
        plt.plot(
            coordinates_array[int(length / 6 * 2), 0],
            coordinates_array[int(length / 6 * 2), 1],
            color="#F0E442",
            markersize=10,
            marker="o",
        )
        plt.plot(
            coordinates_array[int(length / 6 * 3), 0],
            coordinates_array[int(length / 6 * 3), 1],
            color="#009E74",
            markersize=10,
            marker="o",
        )
        plt.plot(
            coordinates_array[int(length / 6 * 4), 0],
            coordinates_array[int(length / 6 * 4), 1],
            color="#56B4E9",
            markersize=10,
            marker="o",
        )
        plt.plot(
            coordinates_array[int(length / 6 * 5), 0],
            coordinates_array[int(length / 6 * 5), 1],
            color="#CC79A7",
            markersize=10,
            marker="o",
        )
        # plt.xticks([])
        # plt.yticks([])

        plt.savefig(output_dir / filename / f"{channel_name}_{dna_num}_coordinates.png")
        plt.close()

        # curvature = np.array(self.curvature[dna_num])
        # plt.plot(curvature[:, 1] * self.pixel_size, coordinates_array[:, 0], color='r')
        # plt.plot(curvature[:, 1] * self.pixel_size, coordinates_array[:, 1], color='b')
        # plt.savefig('%s_%s_x_and_y.png' % (savename, dna_num))
        # plt.close()

    def measure_end_to_end_distance(self):
        """Calculate the Euclidean distance between the start and end of linear molecules.
        The hypotenuse is calculated between the start ([0,0], [0,1]) and end ([-1,0], [-1,1]) of linear
        molecules. If the molecule is circular then the distance is set to zero (0).
        """

        for dna_num in sorted(self.splined_traces.keys()):
            if self.mol_is_circular[dna_num]:
                self.end_to_end_distance[dna_num] = 0
            else:
                x1 = self.splined_traces[dna_num][0, 0]
                y1 = self.splined_traces[dna_num][0, 1]
                x2 = self.splined_traces[dna_num][-1, 0]
                y2 = self.splined_traces[dna_num][-1, 1]
                self.end_to_end_distance[dna_num] = math.hypot((x1 - x2), (y1 - y2)) * self.pixel_size * 1e9

    def measure_bending_angle(self):
        """Calculate the bending angle at the point of highest curvature"""
        nm_each_side = 10  # nm each side of the max curvature point used to determine bending angle
        for dna_num in sorted(self.splined_traces.keys()):
            if self.contour_lengths[dna_num] > 80:  # Filtering out molecules that are too small
                length = len(self.curvature[dna_num])
                mid_nm = self.curvature[dna_num][int(length / 2), 1]
                start_nm = mid_nm - 10  # Starting point of the middle section, in nm
                end_nm = mid_nm + 10  # Ending point of the middle section, in nm
                start = np.argmin(np.abs(self.curvature[dna_num][:, 1] - start_nm))
                end = np.argmin(np.abs(self.curvature[dna_num][:, 1] - end_nm))
                self.central_curvature[dna_num] = self.curvature[dna_num][start:end]  # Middle 20 nm of the molecule
                self.central_max_curvature[dna_num] = np.amax(np.abs(self.central_curvature[dna_num][:, 2]))  # in nm
                position_in_central = np.argmax(np.abs(self.central_curvature[dna_num][:, 2]))
                self.central_max_curvature_location[dna_num] = self.central_curvature[dna_num][position_in_central, 1]
                position = position_in_central + start  # The location of the max curvature point in the entire molecule

                # Left line that forms the bending angle
                left_nm = self.central_max_curvature_location[dna_num] - nm_each_side
                left = np.argmin(np.abs(self.curvature[dna_num][:, 1] - left_nm))
                xa = self.splined_traces[dna_num][left : position + 1, 0]
                ya = self.splined_traces[dna_num][left : position + 1, 1]
                ga, _, _, _, _ = stats.linregress(xa, ya)  # Gradient of left line
                vax = self.splined_traces[dna_num][position, 0] - self.splined_traces[dna_num][left, 0]
                vay = ga * vax  # Left line vector

                # Right line that forms the bending angle
                right_nm = self.central_max_curvature_location[dna_num] + 10
                right = np.argmin(np.abs(self.curvature[dna_num][:, 1] - right_nm))
                xb = self.splined_traces[dna_num][position : right + 1, 0]
                yb = self.splined_traces[dna_num][position : right + 1, 1]
                gb, _, _, _, _ = stats.linregress(xb, yb)  # Gradient of right line
                vbx = self.splined_traces[dna_num][position, 0] - self.splined_traces[dna_num][right, 0]
                vby = gb * vbx  # Right line vector

                # Calculates the bending angle
                dot_product = vax * vbx + vay * vby
                mod_of_vector = math.sqrt(vax**2 + vay**2) * math.sqrt(vbx**2 + vby**2)
                bending_angle_r = math.acos(dot_product / mod_of_vector)  # radians
                bending_angle_d = bending_angle_r / math.pi * 180  # degrees
                self.bending_angle[dna_num] = bending_angle_d
            else:  # For molecules that are too short
                self.central_max_curvature_location[dna_num] = 0
                self.bending_angle[dna_num] = 0


class traceStats(object):
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
            stats[mol_num]["Contour Lengths"] = self.trace_object.contour_lengths[mol_num]
            stats[mol_num]["Circular"] = self.trace_object.mol_is_circular[mol_num]
            stats[mol_num]["End to End Distance"] = self.trace_object.end_to_end_distance[mol_num]
            stats[mol_num]["Max Curvature"] = self.trace_object.max_curvature[mol_num]
            stats[mol_num]["Max Curvature Location"] = self.trace_object.max_curvature_location[mol_num]
            stats[mol_num]["Mean Curvature"] = self.trace_object.mean_curvature[mol_num]
            stats[mol_num]["Variance of Curvature"] = self.trace_object.curvature_variance[mol_num]
            stats[mol_num]["Variance of Absolute Curvature"] = self.trace_object.curvature_variance_abs[mol_num]
            stats[mol_num]["Middle Max Curvature Location"] = self.trace_object.central_max_curvature_location[mol_num]
            stats[mol_num]["Bending Angle"] = self.trace_object.bending_angle[mol_num]
        self.df = pd.DataFrame.from_dict(data=stats, orient="index")
        self.df.reset_index(drop=True, inplace=True)
        self.df.index.name = "Molecule Number"
        # self.df["Experiment Directory"] = str(Path().cwd())
        self.df["Image Name"] = self.image_path.name
        self.df["Basename"] = str(self.image_path.parent)

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


class curvatureStats(object):
    """A class for pixel-level curvature related statistics."""

    def __init__(self, trace_object: dnaTrace, image_path: Union[str, Path]) -> None:
        self.trace_object = trace_object
        self.image_path = Path(image_path)
        self.curvature_dataframe = []
        self.create_curvature_object()

    def create_curvature_object(self):
        """Creates a pixel-level pandas dataframe, for the curvature and accumulative contour length at each point of the molecule"""

        stats = OrderedDict()
        for mol_num, _ in self.trace_object.curvature.items():
            if (
                self.trace_object.mol_is_circular[mol_num] == False
                and 80 < self.trace_object.contour_lengths[mol_num] < 130
                and self.trace_object.max_curvature[mol_num] < 2
            ):
                stats[mol_num] = mol_num
                for i, [n, contour, c, dx, dy, d2x, d2y] in enumerate(self.trace_object.curvature[mol_num]):
                    stats[mol_num]["Point Number"] = n
                    stats[mol_num]["Contour Length"] = contour
                    stats[mol_num]["Curvature"] = c
        self.curvature_dataframe = pd.DataFrame.from_dict(data=stats, orient="index")
        self.curvature_dataframe.reset_index(drop=True, inplace=True)
        self.curvature_dataframe.index.name = "Molecule Number"
        self.curvature_dataframe["Image Name"] = self.image_path.name
        self.curvature_dataframe["Basename"] = str(self.image_path.parent)

        # curvature_dict = {}
        #
        # trace_directory_file = self.trace_object.afm_image_name
        # trace_directory = os.path.dirname(trace_directory_file)
        # basename = os.path.basename(trace_directory)
        # img_name = os.path.basename(trace_directory_file)
        #
        # for mol_num, dna_num in enumerate(sorted(self.trace_object.curvature.keys())):
        #     try:
        #         if (
        #                 self.trace_object.mol_is_circular[dna_num] == False
        #                 and 80 < self.trace_object.contour_lengths[dna_num] < 130
        #                 and self.trace_object.max_curvature[dna_num] < 2
        #         ):
        #             for i, [n, contour, c, dx, dy, d2x, d2y] in enumerate(self.trace_object.curvature[dna_num]):
        #                 try:
        #                     curvature_dict["Molecule number"].append(mol_num)
        #                     curvature_dict["Experiment Directory"].append(trace_directory)
        #                     curvature_dict["Basename"].append(basename)
        #                     curvature_dict["Image Name"].append(img_name)
        #                     curvature_dict["Point number"].append(n)
        #                     curvature_dict["Contour length"].append(contour)
        #                     curvature_dict["Curvature"].append(c)
        #
        #                 except KeyError:
        #                     curvature_dict["Molecule number"] = [mol_num]
        #                     curvature_dict["Experiment Directory"] = [trace_directory]
        #                     curvature_dict["Basename"] = [basename]
        #                     curvature_dict["Image Name"] = [img_name]
        #                     curvature_dict["Point number"] = [n]
        #                     curvature_dict["Contour length"] = [contour]
        #                     curvature_dict["Curvature"] = [c]
        #     except KeyError:
        #         continue
        #
        # self.curvature_dataframe = pd.DataFrame(data=curvature_dict)

    # def update_curvature(self, new_traces):
    #
    #     curvature_dict = {}
    #
    #     trace_directory_file = new_traces.afm_image_name
    #     trace_directory = os.path.dirname(trace_directory_file)
    #     basename = os.path.basename(trace_directory)
    #     img_name = os.path.basename(trace_directory_file)
    #
    #     for mol_num, dna_num in enumerate(sorted(new_traces.contour_lengths.keys())):
    #         try:
    #             if (
    #                     new_traces.mol_is_circular[dna_num] == False
    #                     and 80 < new_traces.contour_lengths[dna_num] < 130
    #                     and new_traces.max_curvature[dna_num] < 2
    #             ):
    #                 for i, [n, contour, c, dx, dy, d2x, d2y] in enumerate(new_traces.curvature[dna_num]):
    #                     try:
    #                         curvature_dict["Molecule number"].append(mol_num)
    #                         curvature_dict["Experiment Directory"].append(trace_directory)
    #                         curvature_dict["Basename"].append(basename)
    #                         curvature_dict["Image Name"].append(img_name)
    #                         curvature_dict["Point number"].append(n)
    #                         curvature_dict["Contour length"].append(contour)
    #                         curvature_dict["Curvature"].append(c)
    #
    #                     except KeyError:
    #                         curvature_dict["Molecule number"] = [mol_num]
    #                         curvature_dict["Experiment Directory"] = [trace_directory]
    #                         curvature_dict["Basename"] = [basename]
    #                         curvature_dict["Image Name"] = [img_name]
    #                         curvature_dict["Point number"] = [n]
    #                         curvature_dict["Contour length"] = [contour]
    #                         curvature_dict["Curvature"] = [c]
    #         except KeyError:
    #             continue
    #
    #     pd_new_traces_dframe = pd.DataFrame(data=curvature_dict)
    #
    #     self.curvature_dataframe = self.curvature_dataframe.append(pd_new_traces_dframe, ignore_index=True)

    def save_curvature_stats(self, save_path: Union[str, Path], json: bool = True, csv: bool = True) -> None:
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
            self.curvature_dataframe.to_json(save_path / "curvaturestats.json")
            LOGGER.info(f"Saved curvature stats for all analysed images to: {str(save_path / 'curvaturestats.json')}")
        if csv:
            self.curvature_dataframe.to_csv(save_path / "curvaturestats.csv")
            LOGGER.info(f"Saved curvature stats for all analysed images to: {str(save_path / 'curvaturestats.csv')}")
