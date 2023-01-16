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
from scipy import ndimage, spatial, optimize, interpolate as interp
from skimage.morphology import label, binary_dilation
from skimage.filters import gaussian

from topostats.logs.logs import LOGGER_NAME
from topostats.tracing.tracingfuncs import genTracingFuncs, reorderTrace
from topostats.tracing.skeletonize import getSkeleton, pruneSkeleton

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
        skeletonisation_method: str = "joe",
        pruning_method: str = "joe",
    ):
        self.full_image_data = full_image_data * 1e-9 if convert_nm_to_m else full_image_data
        self.grains_orig = np.where(grains != 0, 1, 0)
        self.filename = filename
        self.pixel_size = pixel_size * 1e-9 if convert_nm_to_m else pixel_size
        self.skeletonisation_method = skeletonisation_method
        self.pruning_method = pruning_method
        self.number_of_rows = self.full_image_data.shape[0]
        self.number_of_columns = self.full_image_data.shape[1]
        self.sigma = 0.7 / (self.pixel_size * 1e9)  # hardset

        self.gauss_image = gaussian(self.full_image_data, self.sigma)
        self.grains = {}
        self.skeleton_dict = {}
        self.skeletons = None
        self.disordered_traces = {}
        self.disordered_trace_img = None
        self.ordered_traces = {}
        self.fitted_traces = {}
        self.splined_traces = {}
        self.contour_lengths = {}
        self.end_to_end_distance = {}
        self.mol_is_circular = {}
        self.curvature = {}

        self.num_circular = 0
        self.num_linear = 0

        self.neighbours = 5  # The number of neighbours used for the curvature measurement

        # supresses scipy splining warnings
        warnings.filterwarnings("ignore")

        LOGGER.debug(f"[{self.filename}] Performing DNA Tracing")

    def trace_dna(self):
        """Perform DNA tracing."""
        self.grains = self.binary_img_to_dict(self.grains_orig)
        # What purpose does binary dilation serve here? Name suggests some sort of smoothing around the edges and
        # the resulting array is used as a mask during the skeletonising process.
        #self.dilate_grains()
        for grain_num, grain in self.grains.items():
            skeleton = getSkeleton(self.gauss_image, grain).get_skeleton(self.skeletonisation_method)
            LOGGER.info(f"[{self.filename}] {label(skeleton).max()-1} breakages in skeleton {grain_num}")
            pruned_skeleton = pruneSkeleton(self.gauss_image, skeleton).prune_skeleton(self.pruning_method)
            self.skeleton_dict[grain_num] = pruned_skeleton
        self.concat_skeletons()
        for grain_num, grain in self.skeleton_dict.items():
            pass
        self.get_disordered_trace()
        self.disordered_trace_img = self.dict_to_binary_image(self.disordered_traces)
        # self.isMolLooped()
        self.linear_or_circular(self.disordered_traces)
        self.get_ordered_traces()
        self.linear_or_circular(self.ordered_traces)
        self.get_fitted_traces()
        self.disordered_trace_img = self.dict_to_binary_image(self.ordered_traces)
        self.get_splined_traces()
        # self.find_curvature()
        # self.saveCurvature()
        self.measure_contour_length()
        self.measure_end_to_end_distance()
        self.report_basic_stats()

    @staticmethod
    def binary_img_to_dict(img: np.ndarray) -> None:
        """Converts a binary image of multiple objects into a dictionary
        of multiple images the same size as the img (nessecary for dialation step).

        Parameters:
        -----------
        img : np.ndarray
            A binary image of multiple molecules

        Returns:
        --------
        dict:
            A dictionary mapping the object (or grain number) to a binary image with
            only that object present.
        """
        dictionary = {}
        labelled_img = label(img)
        for grain_num in range(1, labelled_img.max() + 1):
            dictionary[grain_num] = np.where(labelled_img == grain_num, 1, 0)
        return dictionary

    def dilate_grains(self) -> None:
        """Dilates each individual grain in the grains dictionary."""
        for grain_num, image in self.grains.items():
            self.grains[grain_num] = ndimage.binary_dilation(image, iterations=1).astype(np.int32)

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

    def dict_to_binary_image(self, coord_dict):
        """Construct a binary image from point coordinates.

        Parameters
        ----------
        coord_dict: dict
            A dictionary of x and y coordinates.

        Returns
        -------
        np.ndarray
            Image of the point coordinates.
        """
        img = np.zeros_like(self.full_image_data)
        for grain_num, coords in coord_dict.items():
            img[coords[:, 0], coords[:, 1]] = grain_num
        return img

    def concat_skeletons(self):
        """Concatonates the skeletons in the skeleton dictionary onto one image"""
        self.skeletons = np.zeros_like(self.grains_orig)
        for skeleton in self.skeleton_dict.values():
            self.skeletons += skeleton

    def get_disordered_trace(self):
        """Puts skeletons into dictionary"""
        for grain_num, skeleton in self.skeleton_dict.items():
            try:
                self.disordered_traces[grain_num] = np.argwhere(skeleton == 1)
            except IndexError:
                # Grains touching border (IndexError) are deleted
                self.grains.pop(grain_num)

    def linear_or_circular(self, traces: dict):
        """Determines whether each molecule is circular or linear based on the local environment of each pixel from the trace.
        This function is sensitive to branches from the skeleton so might need to implement a function to remove them

        Parameters
        ----------
        traces: dict
            A dictionary of the molecule_number and points within the skeleton.
        """

        self.num_circular = 0
        self.num_linear = 0

        for dna_num in sorted(traces.keys()):
            points_with_one_neighbour = 0
            fitted_trace_list = traces[dna_num].tolist()
            # For loop determines how many neighbours a point has - if only one it is an end
            for x, y in fitted_trace_list:

                if genTracingFuncs.count_and_get_neighbours(x, y, fitted_trace_list)[0] == 1:
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
        """Depending on whether the mol is circular or linear, order the traces so the points follow."""
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
                    except UnboundLocalError:  # unsure how the ULE appears and why that means we remove the grain?
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
                    y_coords = np.arange(trace_coordinate[1] - index_width, trace_coordinate[1] + index_width + 1)[::-1]
                    x_coords = np.arange(trace_coordinate[0] - index_width, trace_coordinate[0] + index_width + 1)

                # if angle is closest to 135 degrees
                elif 157.5 >= vector_angle >= 112.5:
                    perp_direction = "positive diaganol"
                    y_coords = np.arange(trace_coordinate[1] - index_width, trace_coordinate[1] + index_width + 1)
                    x_coords = np.arange(trace_coordinate[0] - index_width, trace_coordinate[0] + index_width + 1)

                # if angle is closest to 90 degrees
                if 112.5 > vector_angle >= 67.5:
                    perp_direction = "horizontal"
                    x_coords = np.arange(trace_coordinate[0] - index_width, trace_coordinate[0] + index_width + 1)
                    y_coords = np.full(len(x_coords), trace_coordinate[1])

                elif 22.5 > vector_angle:  # if angle is closest to 0 degrees
                    perp_direction = "vertical"
                    y_coords = np.arange(trace_coordinate[1] - index_width, trace_coordinate[1] + index_width + 1)
                    x_coords = np.full(len(y_coords), trace_coordinate[0])

                elif vector_angle >= 157.5:  # if angle is closest to 180 degrees
                    perp_direction = "vertical"
                    y_coords = np.arange(trace_coordinate[1] - index_width, trace_coordinate[1] + index_width + 1)
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
                #        self.disordered_traces.pop(dna_num)
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
        for dna_num in sorted(self.disordered_traces.keys()):
            plt.plot(self.ordered_traces[dna_num][:, 0], self.ordered_traces[dna_num][:, 1], markersize=1)
            plt.plot(self.fitted_traces[dna_num][:, 0], self.fitted_traces[dna_num][:, 1], markersize=1)
            plt.plot(self.splined_traces[dna_num][:, 0], self.splined_traces[dna_num][:, 1], markersize=1)
            # print(len(self.skeletons[dna_num]), len(self.disordered_traces[dna_num]))
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
        for dna_num in sorted(self.disordered_traces.keys()):
            # disordered_trace_list = self.disordered_traces[dna_num].tolist()
            # less_dense_trace = np.array([disordered_trace_list[i] for i in range(0,len(disordered_trace_list),5)])
            plt.plot(
                self.disordered_traces[dna_num][:, 0],
                self.disordered_traces[dna_num][:, 1],
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


class nodeStats():
    """Class containing methods to find and analyse the nodes/crossings within a grain"""

    def __init__(self, image: np.ndarray, grains: np.ndarray, skeletons: np.ndarray, px_2_nm: float) -> None:
        self.image = image
        self.grains = grains
        self.skeletons = skeletons
        self.px_2_nm = px_2_nm

        self.skeleton = None
        self.conv_skelly = None
        self.connected_nodes = None
        self.node_centre_mask = None
        self.node_dict = None
        self.test = None
        self.full_dict = {}

    def get_node_stats(self) -> dict:
        """The workflow for obtaining the node statistics.
        
        Returns:
        --------
        dict
            Structure: <grain_number>
                        |-> <node_number>
                            |-> 'branch_stats'
                                |-> <branch_number>
                                    -> 'ordered_coords', 'heights', 'gaussian_fit', 'fwhm', 'angles'
                                    -> <values>
        """
        for skeleton_no in range(1, label(self.skeletons).max() + 1):
            self.skeleton = self.skeletons.copy()
            self.skeleton[self.skeleton != skeleton_no] = 0
            self.skeleton[self.skeleton != 0] = 1
            self.convolve_skelly()
            if len(self.conv_skelly[self.conv_skelly==3]) != 0: # check if any nodes
                self.connect_close_nodes(node_width=2.85)
                self.highlight_node_centres(self.connected_nodes)
                self.analyse_nodes(box_length=20)
                self.full_dict[skeleton_no] = self.node_dict
            else:
                self.full_dict[skeleton_no] = {}
        return self.full_dict

    def convolve_skelly(self) -> None:
        """Convolves the skeleton with a 3x3 ones kernel to producing an array
        of the skeleton as 1, endpoints as 2, and nodes as 3.
        """
        conv = ndimage.convolve(self.skeleton, np.ones((3, 3)))
        conv[self.skeleton == 0] = 0  # remove non-skeleton points
        conv[conv == 3] = 1  # skelly = 1
        conv[conv > 3] = 3  # nodes = 3
        self.conv_skelly = conv

    def connect_close_nodes(self, node_width: float = 2.85) -> None:
        """Looks to see if nodes are within the node_width boundary (2.85nm) and thus
        are also labeled as part of the node.

        Parameters
        ----------
        node_width: float
            The width of the dna in the grain, used to connect close nodes.
        """
        self.connected_nodes = self.conv_skelly.copy()
        nodeless = self.conv_skelly.copy()
        nodeless[nodeless != 1] = 0  # remove non-skeleton points
        nodeless = label(nodeless)
        for i in range(1, nodeless.max() + 1):
            if nodeless[nodeless == i].size < (node_width / self.px_2_nm):
                self.connected_nodes[nodeless == i] = 3

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

        self.node_centre_mask = small_node_mask

    def analyse_nodes(self, box_length: int = 20):
        """This function obtains the main analyses for the nodes of a single molecule. Within a certain box (nm) around the node.

        bg = 0, skeleton = 1, endpoints = 2, nodes = 3, #branches = 4.
        """
        length = int((box_length / 2) / self.px_2_nm)
        x_arr, y_arr = np.where(self.node_centre_mask.copy() == 3)
        # iterate over the nodes to find areas
        node_dict = {}
        real_node_count = 0
        for node_no, (x, y) in enumerate(zip(x_arr, y_arr)): # get centres
            # get area around node
            node_area = self.connected_nodes.copy()[x-length : x+length+1, y-length : y+length+1]
            node_area = np.pad(node_area, 1)
            node_coords = np.stack(np.where(node_area == 3)).T
            centre = (np.asarray(node_area.shape) / 2).astype(int)
            branch_mask = self.clean_centre_branches(node_area)
            branch_img = np.zeros_like(node_area) # initialising paired branch img
            
            # iterate through branches to order
            labeled_area = label(branch_mask)
            LOGGER.info(f"No. branches from node {node_no}: {labeled_area.max()}")

            # stop processing if nib (node has 2 branches)
            if labeled_area.max() <= 2:
                LOGGER.info(f"node {node_no} has only two branches - skipped & nodes removed")
                node_coords += ([x, y] - centre) # get whole image coords
                self.node_centre_mask[x, y] = 1 # remove these from node_centre_mask
                self.connected_nodes[node_coords[:,0], node_coords[:,1]] = 1 # remove these from connected_nodes
            else:
                real_node_count += 1
                ordered_branches = []
                vectors = []
                for branch_no in range(1, labeled_area.max() + 1):
                    # get image of just branch
                    branch = labeled_area.copy()
                    branch[labeled_area != branch_no] = 0
                    branch[labeled_area == branch_no] = 1
                    # order branch
                    ordered = self.order_branch(branch)
                    # identify vector
                    vector = self.get_vector(ordered)
                    # add to list
                    vectors.append(vector)
                    ordered_branches.append(ordered)

                # pair vectors
                pairs = self.pair_vectors(np.asarray(vectors))

                # join matching branches (through node?)
                matched_branches = {}
                for i, (branch_1, branch_2) in enumerate(pairs):
                    matched_branches[i] = {}
                    branch_1_coords = ordered_branches[branch_1]
                    branch_2_coords = ordered_branches[branch_2]
                    branch_1_coords, branch_2_coords = self.order_branches(branch_1_coords, branch_2_coords)
                    # find close ends by rearranging branch coords
                    crossing = self.binary_line(branch_1_coords[-1], branch_2_coords[0])
                    # find lowest size of branch coords
                    branch_coords = np.append(branch_1_coords, crossing[1:-1], axis=0)
                    branch_coords = np.append(branch_coords, branch_2_coords, axis=0)
                    branch_img[branch_coords[:,0], branch_coords[:,1]] = i + 1
                    # calc image-wide coords
                    branch_coords_img = branch_coords + ([x, y] - centre)
                    matched_branches[i]["ordered_coords"] = branch_coords_img
                    # get heights and line distance of branch
                    heights = self.image[branch_coords_img[:, 0], branch_coords_img[:, 1]]
                    matched_branches[i]["heights"] = heights
                    distances = self.coord_dist(branch_coords)
                    matched_branches[i]["distances"] = distances
                    # identify over/under
                    fwhm2 = self.fwhm2(heights, distances)
                    matched_branches[i]["fwhm2"] = fwhm2
                    try:
                        fwhm, popt = self.fwhm(heights, distances)
                        matched_branches[i]["gaussian_fit"] = popt
                        matched_branches[i]["fwhm"] = fwhm
                    except RuntimeError as error:
                        LOGGER.error(f"{error}. Could not find optimal curvefit params")

                # add unpaired branches to image plot
                unpaired_branches = np.delete(np.arange(0, labeled_area.max()), pairs.flatten())
                print(f"Unpaired branches: {unpaired_branches}")
                branch_label = branch_img.max()
                for i in unpaired_branches: # carries on from loop variable i
                    branch_label += 1
                    branch_img[ordered_branches[i][:,0], ordered_branches[i][:,1]] = branch_label

                # calc crossing angle
                # get full branch vectors
                vectors = []
                for branch_no, values in matched_branches.items():
                    vectors.append(self.get_vector(values["ordered_coords"]))
                # calc angles to first vector i.e. first should always be 0
                cos_angles = self.calc_angles(np.asarray(vectors))[0]
                cos_angles[cos_angles > 1] = 1  # floating point sometimes causes nans for 1's
                angles = np.arccos(cos_angles) / np.pi * 180
                for i, angle in enumerate(angles):
                    matched_branches[i]["angles"] = angle

                nodes_and_branches = np.zeros_like(self.image)
                for branch_num, values in matched_branches.items():
                    coords = values["ordered_coords"]
                    nodes_and_branches[coords[:,0], coords[:,1]] = branch_num + 1
                
                self.test = branch_img

                node_dict[real_node_count] = {
                    "branch_stats": matched_branches,
                    "node_stats": {
                        "node_mid_coords": [x, y],
                        "node_area_image": self.image[x-length : x+length+1, y-length : y+length+1],
                        "node_area_grain": self.grains[x-length : x+length+1, y-length : y+length+1],
                        "node_branch_mask": branch_img[1:-1, 1:-1], # to remove padding
                    }
                }

            self.node_dict = node_dict

    def order_branch(self, binary_image: np.ndarray):
        """Orders the branch by identify an endpoint, and looking at the local area of the point to find the next.

        Parameters
        ----------
        binary_image: np.ndarray
            A binary image of a skeleton segment to order it's points.

        Returns
        -------
        np.ndarray
            An array of ordered cordinates.
        """
        # get branch starts
        endpoints_highlight = ndimage.convolve(binary_image, np.ones((3, 3)))
        endpoints_highlight[binary_image == 0] = 0
        endpoints = np.argwhere(endpoints_highlight == 2)

        # as > 1 endpoint, find one closest to node
        centre = (np.asarray(binary_image.shape) / 2).astype(int)
        dist_vals = abs((endpoints - centre).sum(axis=1))
        endpoint = endpoints[np.argmin(dist_vals)]

        # initialise points
        all_points = np.stack(np.where(binary_image == 1)).T
        no_points = len(all_points)

        # add starting point to ordered array
        ordered = np.zeros_like(all_points)
        ordered[0] += endpoint
        binary_image[endpoint[0], endpoint[1]] = 0  # remove from array
        # iterate to order the rest of the points
        for i in range(no_points - 1):
            current_point = ordered[i]  # get last point
            area, _ = self.local_area_sum(binary_image, current_point)  # look at local area
            next_point = (current_point + np.argwhere(np.reshape(area, (3, 3,)) == 1) - (1, 1))[0]
            # find where to go next
            ordered[i + 1] += next_point  # add to ordered array
            binary_image[next_point[0], next_point[1]] = 0  # set value to zero

        return ordered

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

    def clean_centre_branches(self, node_area: np.ndarray):
        """This function check to see if the node area contains another node. If it does,
        it removes the node and branche pixels associated with only that extra node.

        Parameters
        ----------
        node_area: np.ndarray
            An integer NxM numpy array where 0 = background, 1 = skeleton, 2 = endpoint, 3 = node.

        Returns
        -------
        np.ndarray
            A binary branch mask of the branches around the central node only.
        """
        centre = (np.asarray(node_area.shape) / 2).astype(int)
        # highlight only nodes
        nodes = node_area.copy()
        nodes[nodes!=3] = 0
        nodes[nodes==3] = 1
        labeled_nodes = label(nodes)
        # highlight only branches
        branches = node_area.copy()
        branches[branches==3] = 0
        labeled_branches = label(branches)
        
        if labeled_nodes.max() > 1:
            # gets centre node coords
            for i in range(1, labeled_nodes.max()+1):
                node_coords = np.stack(np.where(labeled_nodes==i)).T
                if centre in node_coords: # find centre node
                    if len(node_coords.shape) < 2: # if only one coord point
                        # find connecting skeleton
                        local_area = self.local_area_sum(branches, node_coords)[0].reshape(3,3)
                        connecting_points = np.stack(np.where(local_area==1)).T - [1,1] + node_coords
                        connecting_branch_points = connecting_points
                    else:
                        # find connecting skeleton points
                        connecting_branch_points = np.asarray([])
                        for coord in node_coords:
                            local_area = self.local_area_sum(branches, coord)[0].reshape(3,3)
                            connecting_points = np.stack(np.where(local_area==1)).T - [1,1] + coord
                            connecting_branch_points = np.append(connecting_branch_points, connecting_points)
                            connecting_branch_points = connecting_branch_points.reshape(-1,2).astype(int)
            # remove branches without connecting_branch_points
            for i in range(1, labeled_branches.max()+1):    
                branch_coords = np.stack(np.where(labeled_branches==i)).T
                point_in_coords = (np.isin(branch_coords, connecting_branch_points).sum(axis=1)==2).any()
                if not point_in_coords:
                    branches[labeled_branches==i] = 0 # remove from view if not connected
        return branches

    @staticmethod
    def get_vector(coords):
        """Calculate the normalised vector of the coordinate means in a branch"""
        vector = coords.mean(axis=0) - coords[0]
        vector /= abs(vector).max()
        return vector

    @staticmethod
    def calc_angles(vectors: np.ndarray):
        """Calculates the cosine of the angles between vectors in an array.
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
        norm = abs(np.diag(dot)) ** 0.5
        angles = abs(dot / (norm.reshape(-1, 1) @ norm.reshape(1, -1)))
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
        return self.pair_angles(angles)

    @staticmethod
    def pair_angles(angles):
        """pairs large values in a symmetric NxN matrix"""
        angles_cp = angles.copy()
        pairs = []
        for _ in range(int(angles.shape[0]/2)):
            pair = np.unravel_index(np.argmax(angles_cp), angles.shape)
            pairs.append(pair) # add to list
            angles_cp[[pair]] = 0 # set rows 0 to avoid picking again
            angles_cp[:,[pair]] = 0 # set cols 0 to avoid picking again
            
        return np.asarray(pairs)

    @staticmethod
    def gaussian(x, h, mean, sigma):
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

    def fwhm(self, heights, distances):
        """Fits a gaussian to the branch heights, and calculates the FWHM"""
        mean = 45.5  # hard coded as middle node value
        sigma = 1 / (200 / 1024)  # 1nm / px2nm = px  half a nm as either side of std
        popt, pcov = optimize.curve_fit(
            self.gaussian, distances, heights - heights.min(), p0=[max(heights) - heights.min(), mean, sigma], maxfev=8000
        )

        return 2.3548 * popt[2], popt  # 2*(2ln2)^1/2 * sigma = FWHM

    def fwhm2(self, heights, distances):
        centre_fraction = int(len(heights)*0.2) # incase zone approaches another node, look at centre for max
        high_idx = np.argmax(heights[centre_fraction:-centre_fraction]) + centre_fraction
        heights_norm = heights.copy() - heights.min() # lower graph so min is 0
        hm = heights_norm.max()/2 # half max value
        
        # get array halves to find first points that cross hm
        arr1 = heights_norm[:high_idx][::-1]
        dist1 = distances[:high_idx][::-1]
        arr2 = heights_norm[high_idx:]
        dist2 = distances[high_idx:]

        arr1_hm = 0
        arr2_hm = 0
        
        for i in range(len(arr1)-1):
            if (arr1[i] > hm) and (arr1[i+1] < hm): # if points cross through the hm value
                arr1_hm = self.lin_interp([dist1[i], arr1[i]], [dist1[i+1], arr1[i+1]], hm)
                break

        for i in range(len(arr2)-1):
            if (arr2[i] > hm) and (arr2[i+1] < hm): # if points cross through the hm value
                arr2_hm = self.lin_interp([dist2[i], arr2[i]], [dist2[i+1], arr2[i+1]], hm)
                break

        fwhm = arr2_hm - arr1_hm
        
        return fwhm, [arr1_hm, arr2_hm, hm], [high_idx, distances[high_idx], heights[high_idx]]

    @staticmethod
    def lin_interp(point_1, point_2, value):
        """Linear interp 2 points by finding line eq and subbing."""
        m = (point_1[1]-point_2[1]) / (point_1[0]-point_2[0])
        c = point_1[1] - (m * point_1[0])
        interp_x = (value - c) / m
        return interp_x

    @staticmethod
    def close_coords(endpoints1, endpoints2):
        """Find the closes coordinates (those at the node crossing) between 2 pairs."""
        sum1 = abs(endpoints1 - endpoints2).sum(axis=1)
        sum2 = abs(endpoints1[::-1] - endpoints2).sum(axis=1)
        if sum1.min() < sum2.min():
            min_idx = np.argmin(sum1)
            return endpoints1[min_idx], endpoints2[min_idx]
        else:
            min_idx = np.argmin(sum2)
            return endpoints1[::-1][min_idx], endpoints2[min_idx]

    @staticmethod
    def order_branches(branch1, branch2):
        """Find the closes coordinates (those at the node crossing) between 2 pairs."""
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
    def binary_line(start, end):
        """Creates a binary path following the straight line between 2 points."""
        arr = []
        swap = False
        slope = (end-start)[1] / (end-start)[0]
        
        if abs(slope) > 1: # swap x and y if slope will cause skips
            start = start[::-1]
            end = end[::-1]

            slope = 1/slope
            swap = True
        
        if start[0] > end[0]: # swap x coords if coords wrong way arround
            start_temp = start
            start = end
            end = start_temp
        
        # code assumes slope < 1 hence swap
        x_start, y_start = start
        x_end, y_end = end
        for x in range(x_start, x_end + 1):
            y_true = slope * (x - x_start) + y_start
            y_pixel = np.round(y_true)
            arr.append([x, y_pixel])
            
        if swap: # if swapped due to slope, return
            return np.asarray(arr)[:,[1,0]].reshape(-1,2).astype(int)
        else:
            return np.asarray(arr).reshape(-1,2).astype(int)

    @staticmethod
    def coord_dist(coords):
        dist_list = [0]
        dist = 0
        for i in range(len(coords)-1):
            if abs(coords[i]-coords[i+1]).sum() == 2:
                dist += 2**0.5
            else:
                dist += 1
            dist_list.append(dist)
        return np.asarray(dist_list)
