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
from scipy import ndimage, spatial, interpolate as interp
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
        self.disordered_trace = {}
        self.ordered_traces = {}
        self.fitted_traces = {}
        self.splined_traces = {}
        self.contour_lengths = {}
        self.end_to_end_distance = {}
        self.mol_is_circular = {}
        self.curvature = {}

        self.number_of_traces = 0
        self.num_circular = 0
        self.num_linear = 0

        self.neighbours = 5  # The number of neighbours used for the curvature measurement

        # supresses scipy splining warnings
        warnings.filterwarnings("ignore")

        LOGGER.debug(f"[{self.filename}] Performing DNA Tracing")

    def trace_dna(self):
        """Perform DNA tracing."""
        self.get_numpy_arrays()
        self.gaussian_filter()
        self.get_disordered_trace()
        # self.isMolLooped()
        self.purge_obvious_crap()
        self.linear_or_circular(self.disordered_trace)
        self.get_ordered_traces()
        self.linear_or_circular(self.ordered_traces)
        self.get_fitted_traces()
        self.get_splined_traces()
        # self.find_curvature()
        # self.saveCurvature()
        self.measure_contour_length()
        self.measure_end_to_end_distance()
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

    def get_disordered_trace(self):
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
                self.disordered_trace[grain_num] = dna_skeleton.output_skeleton
            except IndexError:
                # Some gwyddion grains touch image border causing IndexError
                # These grains are deleted
                self.grains.pop(grain_num)
            # skel = morphology.skeletonize(self.grains[grain_num])
            # self.skeletons[grain_num] = np.argwhere(skel == 1)

    def purge_obvious_crap(self):

        for dna_num in sorted(self.disordered_trace.keys()):

            if len(self.disordered_trace[dna_num]) < 10:
                self.disordered_trace.pop(dna_num, None)

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

        for dna_num in sorted(self.disordered_trace.keys()):

            circle_tracing = True

            if self.mol_is_circular[dna_num]:

                self.ordered_traces[dna_num], trace_completed = reorderTrace.circularTrace(
                    self.disordered_trace[dna_num]
                )

                if not trace_completed:
                    self.mol_is_circular[dna_num] = False
                    try:
                        self.ordered_traces[dna_num] = reorderTrace.linearTrace(self.ordered_traces[dna_num].tolist())
                    except UnboundLocalError:
                        self.mol_is_circular.pop(dna_num)
                        self.disordered_trace.pop(dna_num)
                        self.grains.pop(dna_num)
                        self.ordered_traces.pop(dna_num)

            elif not self.mol_is_circular[dna_num]:
                self.ordered_traces[dna_num] = reorderTrace.linearTrace(self.disordered_trace[dna_num].tolist())

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

        step_size = int(7e-9 / (self.pixel_size))  # 3 nm step size
        interp_step = int(1e-10 / self.pixel_size)
        # Lets see if we just got with the pixel_to_nm_scaling
        # step_size = self.pixel_size
        # interp_step = self.pixel_size

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
            while nbr / step_size < 4:
                if step_size <= 1:
                    step_size = 1
                    break
                step_size = -1
            if self.mol_is_circular[dna_num]:

                # if nbr/step_size > 4: #the degree of spline fit is 3 so there cannot be less than 3 points in splined trace

                # ev_array = np.linspace(0, 1, nbr * step_size)
                ev_array = np.linspace(0, 1, int(nbr * step_size))

                for i in range(step_size):
                    x_sampled = np.array(
                        [
                            self.fitted_traces[dna_num][:, 0][j]
                            for j in range(i, len(self.fitted_traces[dna_num][:, 0]), step_size)
                        ]
                    )
                    y_sampled = np.array(
                        [
                            self.fitted_traces[dna_num][:, 1][j]
                            for j in range(i, len(self.fitted_traces[dna_num][:, 1]), step_size)
                        ]
                    )

                    try:
                        tck, u = interp.splprep([x_sampled, y_sampled], s=0, per=2, quiet=1, k=3)
                        out = interp.splev(ev_array, tck)
                        splined_trace = np.column_stack((out[0], out[1]))
                    except ValueError:
                        # Value error occurs when the "trace fitting" really messes up the traces

                        x = np.array(
                            [
                                self.ordered_traces[dna_num][:, 0][j]
                                for j in range(i, len(self.ordered_traces[dna_num][:, 0]), step_size)
                            ]
                        )
                        y = np.array(
                            [
                                self.ordered_traces[dna_num][:, 1][j]
                                for j in range(i, len(self.ordered_traces[dna_num][:, 1]), step_size)
                            ]
                        )

                        try:
                            tck, u = interp.splprep([x, y], s=0, per=2, quiet=1)
                            out = interp.splev(np.linspace(0, 1, nbr * step_size), tck)
                            splined_trace = np.column_stack((out[0], out[1]))
                        except ValueError:  # sometimes even the ordered_traces are too bugged out so just delete these traces
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

                if not self.splining_success:
                    continue

                spline_average = np.divide(spline_running_total, [step_size, step_size])
                del spline_running_total
                self.splined_traces[dna_num] = spline_average
                # else:
                #    x = self.fitted_traces[dna_num][:,0]
                #    y = self.fitted_traces[dna_num][:,1]

                #    try:
                #        tck, u = interp.splprep([x, y], s=0, per = 2, quiet = 1, k = 3)
                #        out = interp.splev(np.linspace(0,1,nbr*step_size), tck)
                #        splined_trace = np.column_stack((out[0], out[1]))
                #        self.splined_traces[dna_num] = splined_trace
                #    except ValueError: #if the trace is really messed up just delete it
                #        self.mol_is_circular.pop(dna_num)
                #        self.disordered_trace.pop(dna_num)
                #        self.grains.pop(dna_num)
                #        self.ordered_traces.pop(dna_num)

            else:
                """
                start_x = self.fitted_traces[dna_num][0, 0]
                end_x = self.fitted_traces[dna_num][-1, 0]

                for i in range(step_size):
                    x_sampled = np.array([self.fitted_traces[dna_num][:, 0][j] for j in
                                          range(i, len(self.fitted_traces[dna_num][:, 0]), step_size)])
                    y_sampled = np.array([self.fitted_traces[dna_num][:, 1][j] for j in
                                          range(i, len(self.fitted_traces[dna_num][:, 1]), step_size)])

                    interp_f = interp.interp1d(x_sampled, y_sampled, kind='cubic', assume_sorted=False)

                    x_new = np.linspace(start_x, end_x, interp_step)
                    y_new = interp_f(x_new)

                    print(y_new)

                    # tck = interp.splrep(x_sampled, y_sampled, quiet = 0)
                    # out = interp.splev(np.linspace(start_x,end_x, nbr*step_size), tck)
                    splined_trace = np.column_stack((x_new, y_new))

                    try:
                        np.add(spline_running_total, splined_trace)
                    except NameError:
                        spline_running_total = np.array(splined_trace)

                spline_average = spline_running_total
                self.splined_traces[dna_num] = spline_average
                """

                # can't get splining of linear molecules to work yet
                self.splined_traces[dna_num] = self.fitted_traces[dna_num]

    def show_traces(self):

        plt.pcolormesh(self.gauss_image, vmax=-3e-9, vmin=3e-9)
        plt.colorbar()
        for dna_num in sorted(self.disordered_trace.keys()):
            plt.plot(self.ordered_traces[dna_num][:, 0], self.ordered_traces[dna_num][:, 1], markersize=1)
            plt.plot(self.fitted_traces[dna_num][:, 0], self.fitted_traces[dna_num][:, 1], markersize=1)
            plt.plot(self.splined_traces[dna_num][:, 0], self.splined_traces[dna_num][:, 1], markersize=1)
            # print(len(self.skeletons[dna_num]), len(self.disordered_trace[dna_num]))
            # plt.plot(self.skeletons[dna_num][:,0], self.skeletons[dna_num][:,1], 'o', markersize = 0.8)

        plt.show()
        plt.close()

    def saveTraceFigures(
        self, filename: Union[str, Path], channel_name: str, vmaxval, vminval, output_dir: Union[str, Path] = None
    ):

        # if directory_name:
        #     filename_with_ext = self._checkForSaveDirectory(filename_with_ext, directory_name)

        # save_file = filename_with_ext[:-4]

        # vmaxval = 20e-9
        # vminval = -10e-9

        plt.pcolormesh(self.full_image_data, vmax=vmaxval, vmin=vminval)
        plt.colorbar()
        # plt.savefig("%s_%s_originalImage.png" % (save_file, channel_name))
        plt.savefig(output_dir / filename / f"{channel_name}_original.png")
        plt.close()

        # plt.pcolormesh(self.full_image_data, vmax=vmaxval, vmin=vminval)
        # plt.colorbar()
        # for dna_num in sorted(self.splined_traces.keys()):
        #     # disordered_trace_list = self.ordered_traces[dna_num].tolist()
        #     # less_dense_trace = np.array([disordered_trace_list[i] for i in range(0,len(disordered_trace_list),5)])
        #     plt.plot(self.splined_traces[dna_num][:, 0], self.splined_traces[dna_num][:, 1], color='c', linewidth=1.0)
        #     if self.mol_is_circular[dna_num]:
        #         starting_point = 0
        #     else:
        #         starting_point = self.neighbours
        #     length = len(self.curvature[dna_num])
        #     plt.plot(self.splined_traces[dna_num][starting_point, 0],
        #              self.splined_traces[dna_num][starting_point, 1],
        #              color='#D55E00', markersize=3.0, marker=5)
        #     plt.plot(self.splined_traces[dna_num][starting_point + int(length / 6), 0],
        #              self.splined_traces[dna_num][starting_point + int(length / 6), 1],
        #              color='#E69F00', markersize=3.0, marker=5)
        #     plt.plot(self.splined_traces[dna_num][starting_point + int(length / 6 * 2), 0],
        #              self.splined_traces[dna_num][starting_point + int(length / 6 * 2), 1],
        #              color='#F0E442', markersize=3.0, marker=5)
        #     plt.plot(self.splined_traces[dna_num][starting_point + int(length / 6 * 3), 0],
        #              self.splined_traces[dna_num][starting_point + int(length / 6 * 3), 1],
        #              color='#009E74', markersize=3.0, marker=5)
        #     plt.plot(self.splined_traces[dna_num][starting_point + int(length / 6 * 4), 0],
        #              self.splined_traces[dna_num][starting_point + int(length / 6 * 4), 1],
        #              color='#0071B2', markersize=3.0, marker=5)
        #     plt.plot(self.splined_traces[dna_num][starting_point + int(length / 6 * 5), 0],
        #              self.splined_traces[dna_num][starting_point + int(length / 6 * 5), 1],
        #              color='#CC79A7', markersize=3.0, marker=5)
        # plt.savefig('%s_%s_splinedtrace_with_markers.png' % (save_file, channel_name))
        # plt.close()

        plt.pcolormesh(self.full_image_data, vmax=vmaxval, vmin=vminval)
        plt.colorbar()
        for dna_num in sorted(self.splined_traces.keys()):
            plt.plot(self.splined_traces[dna_num][:, 0], self.splined_traces[dna_num][:, 1], color="c", linewidth=1.0)
        # plt.savefig("%s_%s_splinedtrace.png" % (save_file, channel_name))
        plt.savefig(output_dir / filename / f"{channel_name}_splinedtrace.png")
        LOGGER.info(f"Splined Trace image saved to : {str(output_dir / filename / f'{channel_name}_splinedtrace.png')}")
        plt.close()

        """
        plt.pcolormesh(self.full_image_data)
        plt.colorbar()
        for dna_num in sorted(self.ordered_traces.keys()):
            #disordered_trace_list = self.ordered_traces[dna_num].tolist()
            #less_dense_trace = np.array([disordered_trace_list[i] for i in range(0,len(disordered_trace_list),5)])
            plt.plot(self.ordered_traces[dna_num][:,0], self.ordered_traces[dna_num][:,1])
        plt.savefig('%s_%s_splinedtrace.png' % (save_file, channel_name))
        plt.close()
        """

        plt.pcolormesh(self.full_image_data, vmax=vmaxval, vmin=vminval)
        plt.colorbar()
        for dna_num in sorted(self.disordered_trace.keys()):
            # disordered_trace_list = self.disordered_trace[dna_num].tolist()
            # less_dense_trace = np.array([disordered_trace_list[i] for i in range(0,len(disordered_trace_list),5)])
            plt.plot(
                self.disordered_trace[dna_num][:, 0],
                self.disordered_trace[dna_num][:, 1],
                "o",
                markersize=0.5,
                color="c",
            )
        # plt.savefig("%s_%s_disorderedtrace.png" % (save_file, channel_name))
        plt.savefig(output_dir / filename / f"{channel_name}_disordered_trace.png")
        plt.close()
        LOGGER.info(
            f"Disordered trace image saved to : {str(output_dir / filename / f'{channel_name}_disordered_trace.png')}"
        )

        plt.pcolormesh(self.full_image_data, vmax=vmaxval, vmin=vminval)
        plt.colorbar()
        for dna_num in sorted(self.grains.keys()):
            grain_plt = np.argwhere(self.grains[dna_num] == 1)
            plt.plot(grain_plt[:, 0], grain_plt[:, 1], "o", markersize=2, color="c")
        # plt.savefig("%s_%s_grains.png" % (save_file, channel_name))
        plt.savefig(output_dir / filename / f"{channel_name}_grains.png")
        plt.close()
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
        for dna_num in sorted(self.splined_traces.keys()):  # the number of molecules identified
            # splined_traces is a dictionary, where the keys are the number of the molecule, and the values are a
            # list of coordinates, in a numpy.ndarray
            # if self.mol_is_circular[dna_num]:
            curve = []
            contour = 0
            coordinates = np.zeros([2, self.neighbours * 2 + 1])
            for i, (x, y) in enumerate(self.splined_traces[dna_num]):
                # Extracts the coordinates for the required number of points and puts them in an array
                if self.mol_is_circular[dna_num] or (
                    self.neighbours < i < len(self.splined_traces[dna_num]) - self.neighbours
                ):
                    for j in range(self.neighbours * 2 + 1):
                        coordinates[0][j] = self.splined_traces[dna_num][i - j][0]
                        coordinates[1][j] = self.splined_traces[dna_num][i - j][1]

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
                self.curvature[dna_num] = curve

    def saveCurvature(self):

        # FIXME : Iterate directly over self.splined_traces.values() or self.splined_traces.items()
        # roc_array = np.zeros(shape=(1, 3))
        for dna_num in sorted(self.curvature.keys()):
            for i, [n, contour, c] in enumerate(self.curvature[dna_num]):
                try:
                    roc_array = np.append(roc_array, np.array([[dna_num, i, contour, c]]), axis=0)
                    # oc_array.append([dna_num, i, contour, c])
                except NameError:
                    roc_array = np.array([[dna_num, i, contour, c]])
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
        plt.savefig("%s_%s_curvature.png" % (savename, dna_num))
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

    def writeContourLengths(self, filename, channel_name):

        if not self.contour_lengths:
            self.measure_contour_length()

        with open(f"{filename}_{channel_name}_contours.txt", "w") as writing_file:
            writing_file.write("#units: nm\n")
            for dna_num in sorted(self.contour_lengths.keys()):
                writing_file.write("%f \n" % self.contour_lengths[dna_num])

    # FIXME : This method doesn't appear to be used here nor within pygwytracing, can it be removed?
    def writeCoordinates(self, dna_num):
        # FIXME: Replace with Path()
        if not os.path.exists(os.path.join(os.path.dirname(self.filename), "Coordinates")):
            os.mkdir(os.path.join(os.path.dirname(self.filename), "Coordinates"))
        directory = os.path.join(os.path.dirname(self.filename), "Coordinates")
        savename = os.path.join(directory, os.path.basename(self.filename)[:-4])
        for i, (x, y) in enumerate(self.splined_traces[dna_num]):
            try:
                coordinates_array = np.append(coordinates_array, np.array([[x, y]]), axis=0)
            except NameError:
                coordinates_array = np.array([[x, y]])

        coordinates = pd.DataFrame(coordinates_array)
        coordinates.to_csv("%s_%s.csv" % (savename, dna_num))

        plt.plot(coordinates_array[:, 0], coordinates_array[:, 1], "ko")
        plt.savefig("%s_%s_coordinates.png" % (savename, dna_num))

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
        # self.end_to_end_distance = {
        #     dna_num: math.hypot((trace[0, 0] - trace[-1, 0]), (trace[0, 1] - trace[-1, 1]))
        #     if self.mol_is_circular[dna_num]
        #     else 0
        #     for dna_num, trace in self.splined_traces.items()
        # }


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
