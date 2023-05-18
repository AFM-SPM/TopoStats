"""Perform DNA Tracing"""
from collections import OrderedDict
import logging
from pathlib import Path
import math
import os
from typing import Union, Tuple
import warnings
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage, spatial, optimize, interpolate as interp
from skimage.morphology import label, binary_dilation
from skimage.filters import gaussian, threshold_otsu

from topostats.logs.logs import LOGGER_NAME
from topostats.tracing.tracingfuncs import genTracingFuncs, reorderTrace
from topostats.tracing.skeletonize import getSkeleton, pruneSkeleton
from topostats.utils import convolve_skelly, ResolutionError

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
        print(f"Img blur: {0.7 / (self.pixel_size * 1e9)}")

        self.gauss_image = gaussian(self.full_image_data, self.sigma)
        self.grains = {}
        self.smoothed_grains = None
        self.orig_skeleton_dict = {}
        self.orig_skeletons = None
        self.skeleton_dict = {}
        self.skeletons = None
        self.disordered_traces = {}
        self.disordered_trace_img = None
        self.ordered_traces = {}
        self.ordered_trace_img = None
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
        self.smooth_grains()
        self.smoothed_grains = self.concat_images_in_dict(image_dict=self.grains)
        for grain_num, grain in self.grains.items():
            skeleton = getSkeleton(self.gauss_image, grain).get_skeleton(self.skeletonisation_method)
            self.orig_skeleton_dict[grain_num] = skeleton
            LOGGER.info(f"[{self.filename}] {(convolve_skelly(skeleton)==2).sum()} endpoints in skeleton {grain_num}")
            pruned_skeleton = pruneSkeleton(self.gauss_image, skeleton).prune_skeleton(self.pruning_method)
            self.skeleton_dict[grain_num] = pruned_skeleton
            self.purge_obvious_crap(self.skeleton_dict)
        self.orig_skeletons = self.concat_images_in_dict(image_dict=self.orig_skeleton_dict)
        self.skeletons = self.concat_images_in_dict(image_dict=self.skeleton_dict)
        self.get_disordered_trace()
        self.disordered_trace_img = self.dict_to_binary_image(self.disordered_traces)
        # self.isMolLooped()
        self.linear_or_circular(self.disordered_traces)
        self.get_ordered_traces()
        self.ordered_trace_img = self.dict_to_binary_image(self.ordered_traces)
        self.linear_or_circular(self.ordered_traces)
        self.get_fitted_traces()
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

    def smooth_grains(self) -> None:
        """Smoothes grains based on the lower number added from dilation or gaussian.
        (makes sure gaussian isnt too agressive."""
        for grain_num, image in self.grains.items():
            dilation = ndimage.binary_dilation(image, iterations=1).astype(np.int32)
            gauss = gaussian(image, sigma=max(image.shape) / 256)
            gauss[gauss > threshold_otsu(gauss) * 1.3] = 1
            gauss[gauss != 1] = 0
            gauss = gauss.astype(np.int32)
            if dilation.sum() - image.sum() > gauss.sum() - image.sum():
                print(f"gaussian: {gauss.sum()-image.sum()}px")
                # self.grains[grain_num] = self.re_add_holes(image, gauss)
                self.grains[grain_num] = gauss
            else:
                print(f"dilation: {dilation.sum()-image.sum()}px")
                # self.grains[grain_num] = self.re_add_holes(image, dilation)
                self.grains[grain_num] = dilation

    def re_add_holes(self, orig_mask, new_mask, holearea_min_max=[4, 100]):
        """As gaussian and dilation smoothing methods can close holes in the original mask,
        this function obtains those holes (based on the general background being the largest)
        and adds them back into the smoothed mask. When paired, this essentailly just smooths
        the outer edge of the grains.

        Parameters:
        -----------
            orig_mask (_type_): _description_
            new_mask (_type_): _description_
            holearea_min_max (_type_): _description_
        """
        holesize_min_px = holearea_min_max[0] / ((self.pixel_size / 1e-9) ** 2)
        holesize_max_px = holearea_min_max[1] / ((self.pixel_size / 1e-9) ** 2)
        holes = 1 - orig_mask
        holes = label(holes)
        sizes = [holes[holes == i].size for i in range(1, holes.max() + 1)]
        max_idx = max(enumerate(sizes), key=lambda x: x[1])[0] + 1  # identify background
        holes[holes == max_idx] = 0  # set background to 0

        self.holes = holes.copy()

        for i, size in enumerate(sizes):
            if size < holesize_min_px or size > holesize_max_px:  # small holes cause issues so are left out
                holes[holes == i + 1] = 0
        holes[holes != 0] = 1

        holey_gauss = new_mask.copy()
        holey_gauss[holes == 1] = 0

        return holey_gauss

    @staticmethod
    def purge_obvious_crap(skeleton_dict: dict) -> None:
        """Removes skeletons < 10px. Caused when circular objects are skeletonised."""
        for dna_num in sorted(skeleton_dict.keys()):
            if len(skeleton_dict[dna_num]) < 10:
                skeleton_dict.pop(dna_num, None)
                print("Popped :", dna_num)

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

    def concat_images_in_dict(self, image_dict: dict):
        """Concatonates the skeletons in the skeleton dictionary onto one image"""
        skeletons = np.zeros_like(self.grains_orig)
        for skeleton in image_dict.values():
            skeletons += skeleton
        return skeletons

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

                """
                # Issue due to large index width that some perp array values > img dims.
                #   if idx width = 2 and trace coord lies 1 away from edge pixels, x/y_coords then goes to coord + 2
                y_coords[y_coords >= self.full_image_data.shape[1]] = self.full_image_data.shape[1] - 1
                x_coords[x_coords >= self.full_image_data.shape[0]] = self.full_image_data.shape[0] - 1
                """
                # Use the perp array to index the guassian filtered image
                perp_array = np.column_stack((x_coords, y_coords))
                height_values = self.full_image_data[perp_array[:, 0], perp_array[:, 1]]

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

        plt.pcolormesh(self.full_image_data, vmax=-3e-9, vmin=3e-9)
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


class nodeStats:
    """Class containing methods to find and analyse the nodes/crossings within a grain"""

    def __init__(self, image: np.ndarray, grains: np.ndarray, skeletons: np.ndarray, px_2_nm: float) -> None:
        self.image = image
        self.grains = grains
        self.skeletons = skeletons
        self.px_2_nm = 1 #px_2_nm
        """
        a = np.zeros((100,100))
        a[21:80, 20] = 1
        a[21:80, 50] = 1
        a[21:80, 80] = 1
        a[20, 21:80] = 1
        a[80, 21:80] = 1
        a[20, 50] = 0
        a[80, 50] = 0
        self.grains = ndimage.binary_dilation(a, iterations=3)
        self.image = np.ones((100,100))
        self.skeletons = a
        """

        self.skeleton = None
        self.conv_skelly = None
        self.connected_nodes = None
        self.all_connected_nodes = self.skeletons.copy()

        self.node_centre_mask = None
        self.node_dict = {}
        self.test = None
        self.test2 = None
        self.test3 = None
        self.test4 = None
        self.test5 = None
        self.full_dict = {}
        self.mol_coords = {}
        self.visuals = {}

    def get_node_stats(self) -> dict:
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
                                |-> 'node stats'
                                    |-> 'node_area_grain', 'node_area_image', 'node_branch_mask', 'node_mid_coords'
        """
        labelled_skeletons = label(self.skeletons)
        for skeleton_no in range(1, labelled_skeletons.max() + 1):
            LOGGER.info(f"Processing Mol: {skeleton_no}")
            self.skeleton = self.skeletons.copy()
            self.skeleton[labelled_skeletons != skeleton_no] = 0
            self.conv_skelly = convolve_skelly(self.skeleton)
            if len(self.conv_skelly[self.conv_skelly == 3]) != 0:  # check if any nodes
                self.connect_close_nodes(node_width=6)
                self.highlight_node_centres(self.connected_nodes)
                self.analyse_nodes(box_length=20)
                if self.check_node_errorless():
                    self.mol_coords[skeleton_no], self.visuals[skeleton_no] = self.compile_trace()
                self.full_dict[skeleton_no] = self.node_dict
            else:
                self.full_dict[skeleton_no] = {}

    def check_node_errorless(self):
        for _, vals in self.node_dict.items():
            if vals['error']:
                return False
            else:
                pass
        return True

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
        nodeless[(nodeless == 3) | (nodeless == 2)] = 0  # remove node & termini points
        nodeless_labels = label(nodeless)
        for i in range(1, nodeless_labels.max() + 1):
            if nodeless[nodeless_labels == i].size < (node_width / self.px_2_nm):
                # maybe also need to select based on height? and also ensure small branches classified
                self.connected_nodes[nodeless_labels == i] = 3

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

    def analyse_nodes(self, box_length: float = 20):
        """This function obtains the main analyses for the nodes of a single molecule. Within a certain box (nm) around the node.

        bg = 0, skeleton = 1, endpoints = 2, nodes = 3.

        Parameters:
        -----------
        box_length: float
            The side length of the box around the node to analyse (in nm).

        """
        # santity check for box length (too small can cause empty sequence error)
        length = int((box_length / 2) / self.px_2_nm)
        if length < 10:
            LOGGER.info(f"Readapted Box Length from {box_length/2}nm or {length}px to 10px")
            length = 10
        x_arr, y_arr = np.where(self.node_centre_mask.copy() == 3)

        # check whether average trace resides inside the grain mask
        dilate = ndimage.binary_dilation(self.skeleton, iterations=2)
        average_trace_advised = dilate[self.grains == 1].sum() == dilate.sum()
        LOGGER.info(f"Branch height traces will be averaged: {average_trace_advised}")

        # iterate over the nodes to find areas
        #node_dict = {}
        matched_branches = None
        branch_img = None
        avg_img = None

        real_node_count = 0
        for node_no, (x, y) in enumerate(zip(x_arr, y_arr)):  # get centres
            # get area around node - might need to check if box lies on the edge
            image_area = self.image[x - length : x + length + 1, y - length : y + length + 1]
            node_area = self.connected_nodes.copy()[x - length : x + length + 1, y - length : y + length + 1]
            reduced_node_area = self._only_centre_branches(node_area)
            branch_mask = reduced_node_area.copy()
            branch_mask[branch_mask == 3] = 0
            branch_mask[branch_mask == 2] = 1
            node_coords = np.stack(np.where(reduced_node_area == 3)).T
            centre = (np.asarray(node_area.shape) / 2).astype(int)
            error = False  # to see if node too complex or region too small

            # iterate through branches to order
            labeled_area = label(
                branch_mask
            )  # labeling the branch mask may not be the best way to do this due to a pissoble single connection
            LOGGER.info(f"No. branches from node {node_no}: {labeled_area.max()}")

            # for cats paper figures - should be removed
            if node_no == 0:
                self.test = labeled_area
            # stop processing if nib (node has 2 branches)
            if labeled_area.max() <= 2:
                LOGGER.info(f"node {node_no} has only two branches - skipped & nodes removed")
                pass
                # sometimes removal of nibs can cause problems when re-indexing nodes
                # node_coords += ([x, y] - centre) # get whole image coords
                # self.node_centre_mask[x, y] = 1 # remove these from node_centre_mask
                # self.connected_nodes[node_coords[:,0], node_coords[:,1]] = 1 # remove these from connected_nodes
            else:
                # try:
                # check wether resolution good enough to trace
                res = self.px_2_nm <= 1000 / 512
                if not res:
                    print("Res Error")
                    raise ResolutionError

                real_node_count += 1
                print(f"Real node: {real_node_count}")
                ordered_branches = []
                vectors = []
                for branch_no in range(1, labeled_area.max() + 1):
                    # get image of just branch
                    branch = labeled_area.copy()
                    branch[labeled_area != branch_no] = 0
                    branch[labeled_area == branch_no] = 1
                    # order branch
                    ordered = self.order_branch(branch, centre)
                    # identify vector
                    vector = self.get_vector(ordered, centre)
                    # add to list
                    vectors.append(vector)
                    ordered_branches.append(ordered)
                if node_no == 0:
                    self.test2 = vectors
                # pair vectors
                pairs = self.pair_vectors(np.asarray(vectors))

                # join matching branches through node
                print("GOT TO MB")
                matched_branches = {}
                branch_img = np.zeros_like(node_area)  # initialising paired branch img
                avg_img = np.zeros_like(node_area)
                for i, (branch_1, branch_2) in enumerate(pairs):
                    matched_branches[i] = {}
                    branch_1_coords = ordered_branches[branch_1]
                    branch_2_coords = ordered_branches[branch_2]
                    # find close ends by rearranging branch coords
                    branch_1_coords, branch_2_coords = self.order_branches(branch_1_coords, branch_2_coords)
                    # Linearly interpolate across the node
                    #   binary line needs to consider previous pixel as it can make kinks (non-skelly bits)
                    #   which can cause a pixel to be missed when ordering the traces
                    crossing1 = self.binary_line(branch_1_coords[-1], centre)
                    crossing2 = self.binary_line(centre, branch_2_coords[0])
                    crossing = np.append(crossing1, crossing2).reshape(-1, 2)
                    # remove the duplicate crossing coords
                    uniq_cross_idxs = np.unique(crossing, axis=0, return_index=True)[1]
                    crossing = np.array([crossing[i] for i in sorted(uniq_cross_idxs)])
                    # Branch coords and crossing
                    branch_coords = np.append(branch_1_coords, crossing[1:-1], axis=0)
                    branch_coords = np.append(branch_coords, branch_2_coords, axis=0)
                    # make images of single branch joined and multiple branches joined
                    single_branch = np.zeros_like(node_area)
                    single_branch[branch_coords[:, 0], branch_coords[:, 1]] = 1
                    single_branch = getSkeleton(image_area, single_branch).get_skeleton("zhang")
                    # calc image-wide coords
                    branch_coords_img = branch_coords + ([x, y] - centre)
                    matched_branches[i]["ordered_coords"] = branch_coords_img
                    matched_branches[i]["ordered_coords_local"] = branch_coords
                    # get heights and trace distance of branch
                    distances = self.coord_dist(branch_coords)
                    zero_dist = distances[np.where(np.all(branch_coords == centre, axis=1))]
                    if average_trace_advised:
                        # np.savetxt("knot2/area.txt",image_area)
                        # np.savetxt("knot2/single_branch.txt",single_branch)
                        # print("ZD: ", zero_dist)
                        distances, heights, mask, _ = self.average_height_trace(image_area, single_branch, zero_dist)
                        # add in mid dist adjustment
                        matched_branches[i]["avg_mask"] = mask
                    else:
                        heights = self.image[branch_coords_img[:, 0], branch_coords_img[:, 1]]
                        distances = distances - zero_dist
                    matched_branches[i]["heights"] = heights
                    matched_branches[i]["distances"] = distances

                    # identify over/under
                    fwhm2 = self.fwhm2(heights, distances)
                    matched_branches[i]["fwhm2"] = fwhm2

                # add paired and unpaired branches to image plot
                fwhms = []
                for branch_idx, values in matched_branches.items():
                    fwhms.append(values["fwhm2"][0])
                branch_idx_order = np.array(list(matched_branches.keys()))[np.argsort(np.array(fwhms))]

                for i, branch_idx in enumerate(branch_idx_order):
                    branch_coords = matched_branches[branch_idx]["ordered_coords_local"]
                    branch_img[branch_coords[:, 0], branch_coords[:, 1]] = i + 1  # add to branch img
                    if average_trace_advised:  # add avg traces
                        avg_img[matched_branches[branch_idx]["avg_mask"] != 0] = i + 1
                    else:
                        avg_img = None

                unpaired_branches = np.delete(np.arange(0, labeled_area.max()), pairs.flatten())
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
                    vectors.append(self.get_vector(values["ordered_coords"], centre))
                # calc angles to first vector i.e. first should always be 0
                cos_angles = self.calc_angles(np.asarray(vectors))[0]
                cos_angles[cos_angles > 1] = 1  # floating point sometimes causes nans for 1's
                angles = np.arccos(cos_angles) / np.pi * 180
                for i, angle in enumerate(angles):
                    matched_branches[i]["angles"] = angle

                if node_no == 0:
                    self.test4 = vectors
                    self.test5 = angles

                """
                except ValueError:
                    LOGGER.error(f"Node {node_no} too complex, see images for details.")
                    error = True
                except ResolutionError:
                    LOGGER.info(f"Node stats skipped as resolution too low: {self.px_2_nm}nm per pixel")
                    error = True
                """

                self.node_dict[real_node_count] = {
                    "error": error,
                    "branch_stats": matched_branches,
                    "node_stats": {
                        "node_mid_coords": [x, y],
                        "node_area_image": image_area,
                        "node_area_grain": self.grains[x - length : x + length + 1, y - length : y + length + 1],
                        "node_area_skeleton": node_area,
                        "node_branch_mask": branch_img,
                        "node_avg_mask": avg_img,
                    },
                }

                print("MB Keys: ", matched_branches.keys())

            self.all_connected_nodes[self.connected_nodes != 0] = self.connected_nodes[self.connected_nodes != 0]
            self.node_dict = self.node_dict
            print("Node keys: ", self.node_dict.keys())

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
        binary_image = np.pad(binary_image, 1).astype(int)
        # get branch starts
        endpoints_highlight = ndimage.convolve(binary_image, np.ones((3, 3)))
        endpoints_highlight[binary_image == 0] = 0
        endpoints = np.argwhere(endpoints_highlight == 2)

        # as > 1 endpoint, find one closest to anchor
        dist_vals = abs((endpoints - anchor).sum(axis=1))
        start = endpoints[np.argmin(dist_vals)]

        # initialise points
        all_points = np.stack(np.where(binary_image == 1)).T
        no_points = len(all_points)

        # add starting point to ordered array
        #ordered = np.zeros_like(all_points)
        ordered = []
        ordered.append(start)
        binary_image[start[0], start[1]] = 0  # remove from array

        # iterate to order the rest of the points
        #for i in range(no_points - 1):
        current_point = ordered[-1]  # get last point
        area, _ = self.local_area_sum(binary_image, current_point)  # look at local area
        local_next_point =  np.argwhere(area.reshape((3, 3,)) == 1) - (1, 1)
        while len(local_next_point) != 0:
            next_point = (current_point + local_next_point)[0]
            # find where to go next
            #ordered[i + 1] += next_point  # add to ordered array
            ordered.append(next_point)
            binary_image[next_point[0], next_point[1]] = 0  # set value to zero
            
            current_point = ordered[-1]  # get last point
            area, _ = self.local_area_sum(binary_image, current_point)  # look at local area
            local_next_point =  np.argwhere(area.reshape((3, 3,)) == 1) - (1, 1)
        # TODO: remove extra 0's that might be leftover?
        return np.array(ordered) - [1, 1]  # remove padding

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
        start_coord = coords[np.absolute(origin - coords).sum(axis=1).argmin()]
        vector = coords.mean(axis=0) - start_coord
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
        for _ in range(int(angles.shape[0] / 2)):
            pair = np.unravel_index(np.argmax(angles_cp), angles.shape)
            pairs.append(pair)  # add to list
            angles_cp[[pair]] = 0  # set rows 0 to avoid picking again
            angles_cp[:, [pair]] = 0  # set cols 0 to avoid picking again

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
            self.gaussian,
            distances,
            heights - heights.min(),
            p0=[max(heights) - heights.min(), mean, sigma],
            maxfev=8000,
        )

        return 2.3548 * popt[2], popt  # 2*(2ln2)^1/2 * sigma = FWHM

    def fwhm2(self, heights, distances):
        centre_fraction = int(len(heights) * 0.2)  # incase zone approaches another node, look at centre for max
        high_idx = np.argmax(heights[centre_fraction:-centre_fraction]) + centre_fraction
        heights_norm = heights.copy() - heights.min()  # lower graph so min is 0
        hm = heights_norm.max() / 2  # half max value -> try to make it the same as other crossing branch?

        # get array halves to find first points that cross hm
        arr1 = heights_norm[:high_idx][::-1]
        dist1 = distances[:high_idx][::-1]
        arr2 = heights_norm[high_idx:]
        dist2 = distances[high_idx:]

        arr1_hm = 0
        arr2_hm = 0

        for i in range(len(arr1) - 1):
            if (arr1[i] > hm) and (arr1[i + 1] < hm):  # if points cross through the hm value
                arr1_hm = self.lin_interp([dist1[i], arr1[i]], [dist1[i + 1], arr1[i + 1]], yvalue=hm)
                break

        for i in range(len(arr2) - 1):
            if (arr2[i] > hm) and (arr2[i + 1] < hm):  # if points cross through the hm value
                arr2_hm = self.lin_interp([dist2[i], arr2[i]], [dist2[i + 1], arr2[i + 1]], yvalue=hm)
                break

        fwhm = arr2_hm - arr1_hm

        return fwhm, [arr1_hm, arr2_hm, hm], [high_idx, distances[high_idx], heights[high_idx]]

    @staticmethod
    def lin_interp(point_1, point_2, xvalue=None, yvalue=None):
        """Linear interp 2 points by finding line eq and subbing."""
        m = (point_1[1] - point_2[1]) / (point_1[0] - point_2[0])
        c = point_1[1] - (m * point_1[0])
        if xvalue is not None:
            interp_y = m * xvalue + c
            return interp_y
        if yvalue is not None:
            interp_x = (yvalue - c) / m
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
        """Find the closest coordinates between 2 coordinate arrays."""
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

    def average_height_trace(self, img: np.ndarray, branch_mask: np.ndarray, dist_zero_point: float) -> tuple:
        """Dilates the original branch to create two additional side-by-side branches
        in order to get a more accurate average of the height traces. This function produces
        the common distances between these 3 branches, and their averaged heights.

        Parameters
        ----------
        img: np.ndarray
            An array of numbers pertaining to an image.
        branch_mask: np.ndarray
            A binary array of the branch, must share the same dimensions as the image.

        Returns
        -------
        tuple
            A tuple of the averaged heights from the linetrace and their corresponding distances
            from the crossing.
        """
        # get heights and dists of the original (middle) branch
        branch_coords = np.stack(np.where(branch_mask == 1)).T
        branch_dist = self.coord_dist(branch_coords)
        branch_heights = img[branch_coords[:, 0], branch_coords[:, 1]]
        branch_dist_norm = branch_dist - dist_zero_point  # branch_dist[branch_heights.argmax()]

        # want to get a 3 pixel line trace, one on each side of orig
        dilate = ndimage.binary_dilation(branch_mask, iterations=1)
        dilate_minus = dilate.copy()
        dilate_minus[branch_mask == 1] = 0
        dilate2 = ndimage.binary_dilation(dilate)
        dilate2[(dilate == 1) | (branch_mask == 1)] = 0
        labels = label(dilate2)
        # reduce binary dilation distance
        paralell = np.zeros_like(branch_mask).astype(np.int32)
        for i in range(1, labels.max() + 1):
            single = labels.copy()
            single[single != i] = 0
            single[single == i] = 1
            sing_dil = ndimage.binary_dilation(single)
            paralell[(sing_dil == dilate_minus) & (sing_dil == 1)] = i
        labels = paralell.copy()
        # print(np.unique(labels, return_index=True))
        # if parallel trace out and back in zone, can get > 2 labels
        labels = self._remove_re_entering_branches(labels, remaining_branches=2)
        # if parallel trace doesn't exit window, can get 1 label
        #   occurs when skeleton has poor connections (extra branches which cut corners)
        if labels.max() == 1:
            conv = convolve_skelly(branch_mask)
            endpoint = np.stack(np.where(conv == 2)).T
            para_trace_coords = np.stack(np.where(labels == 1)).T
            abs_diff = np.absolute(para_trace_coords - endpoint).sum(axis=1)
            min_idxs = np.where(abs_diff == abs_diff.min())
            trace_coords_remove = para_trace_coords[min_idxs]
            labels[trace_coords_remove[:, 0], trace_coords_remove[:, 1]] = 0
            labels = label(labels)

        binary = labels.copy()
        binary[binary != 0] = 1
        binary += branch_mask

        # get and order coords, then get heights and distances relitive to node centre / highest point
        centre_fraction = 1 - 0.8  # the middle % of data to look for peak - stops peaks being found at edges
        heights = []
        distances = []
        for i in range(1, labels.max() + 1):
            trace = labels.copy()
            trace[trace != i] = 0
            trace[trace != 0] = 1
            trace = getSkeleton(img, trace).get_skeleton("zhang")
            trace = self.order_branch(trace, branch_coords[0])
            height_trace = img[trace[:, 0], trace[:, 1]]
            height_len = len(height_trace)
            central_heights = height_trace[int(height_len * centre_fraction) : int(-height_len * centre_fraction)]
            heights.append(height_trace)
            dist = self.coord_dist(trace)
            distances.append(
                dist - dist_zero_point
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
    def _only_centre_branches(node_image: np.ndarray):
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
        centre = np.asarray(node_image_cp.shape) / 2
        node_coords = np.stack(np.where(nodes == 3)).T
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

        return node_image_cp

    def compile_trace(self):
        """This function uses the branches and FWHM's identified in the node_stats dictionary to create a
        continious trace of the molecule.
        """
        LOGGER.info("Compiling the trace.")

        # iterate throught the dict to get branch coords, heights and fwhms
        node_centre_coords = []
        node_area_box = []
        crossing_coords = []
        crossing_heights = []
        crossing_distances = []
        fwhms = []
        for _, stats in self.node_dict.items():
            node_centre_coords.append(stats['node_stats']['node_mid_coords'])
            node_area_box.append(stats['node_stats']['node_area_image'].shape)
            temp_coords = []
            temp__heights = []
            temp_distances = []
            temp_fwhms = []
            for _, branch_stats in stats['branch_stats'].items():
                temp_coords.append(branch_stats['ordered_coords'])
                temp__heights.append(branch_stats['heights'])
                temp_distances.append(branch_stats['distances'])
                temp_fwhms.append(branch_stats["fwhm2"][0])
            crossing_coords.append(temp_coords)
            crossing_heights.append(temp__heights)
            crossing_distances.append(temp_distances)
            fwhms.append(temp_fwhms)


        # get image minus the crossing areas
        minus = self.get_minus_img(node_area_box, node_centre_coords)
        # get crossing image
        crossings = self.get_crossing_img(crossing_coords, minus.max() + 1)
        print(np.unique(crossings))
        # combine branches and segments
        both_img = self.get_both_img(minus, crossings)

        # order minus segments
        ordered = []
        for i in range(1, minus.max() + 1):
            arr = np.where(minus, minus == i, 0)
            ordered.append(self.order_branch(arr, [0, 0]))  # orientated later
        # combine ordered indexes
        for node_crossing_coords in crossing_coords:
            for single_cross in node_crossing_coords:
                ordered.append(np.array(single_cross))
        print("LEN: ", len(ordered))

        #np.savetxt("/Users/Maxgamill/Desktop/both.txt", both_img)
        print("Getting coord trace")
        coord_trace = self.trace_mol(ordered, both_img)
        #np.savetxt("/Users/Maxgamill/Desktop/trace.txt", coord_trace[0])

        # visual over under img
        visual = self.get_visual_img(coord_trace, fwhms, crossing_coords)

        #np.savetxt("/Users/Maxgamill/Desktop/visual.txt", visual)

        # I could use the traced coords, remove the node centre coords, and re-label segments
        #   following 1, 2, 3... around the mol which should look like the Planar Diagram formation
        #   (https://topoly.cent.uw.edu.pl/dictionary.html#codes). Then look at each corssing zone again,
        #   determine which in in-undergoing and assign labels counter-clockwise
        print("Getting PD Codes:")
        pd_codes = self.get_pds(coord_trace, node_centre_coords, fwhms, crossing_coords)

        return coord_trace, visual

    def get_minus_img(self, node_area_box, node_centre_coords):
        minus = self.skeleton.copy()
        for i, area in enumerate(node_area_box):
            x, y = node_centre_coords[i]
            area = np.array(area) // 2
            minus[x - area[0] : x + area[0], y - area[1] : y + area[1]] = 0
        return label(minus)

    def get_crossing_img(self, crossing_coords, label_offset):
        crossings = np.zeros_like(self.skeleton)
        for i, node_crossing_coords in enumerate(crossing_coords):
            for j, single_cross_coords in enumerate(node_crossing_coords):
                #print(i,j, 2*i + j)
                crossings[single_cross_coords[:, 0], single_cross_coords[:, 1]] = 2*i + j + label_offset
        return crossings

    @staticmethod
    def get_both_img(minus_img, crossing_img):
        both_img = minus_img.copy()
        both_img[crossing_img != 0] = crossing_img[crossing_img != 0]
        return both_img

    @staticmethod
    def trace_mol(ordered_segment_coords, both_img):
        remaining = both_img.copy().astype(np.int32)  # image
        # get first segment
        idx = 0  # set index
        coord_trace = ordered_segment_coords[idx].astype(np.int32).copy()  # add ordered segment to trace
        remaining[remaining == idx + 1] = 0  # remove segment from image
        x, y = coord_trace[-1]  # find end coords of trace
        idx = remaining[x - 1 : x + 2, y - 1 : y + 2].max() - 1  # find local area of end coord to find next index

        mol_coords = []
        mol_num = 0
        while len(np.unique(remaining)) > 1:
            mol_num += 1
            while idx > -1:  # either cycled through all or hits terminus -> all will be just background
                if (
                    abs(coord_trace[-1] - ordered_segment_coords[idx][0]).sum()
                    < abs(coord_trace[-1] - ordered_segment_coords[idx][-1]).sum()
                ):
                    coord_trace = np.append(coord_trace, ordered_segment_coords[idx].astype(np.int32), axis=0)
                else:
                    coord_trace = np.append(coord_trace, ordered_segment_coords[idx][::-1].astype(np.int32), axis=0)
                x, y = coord_trace[-1]
                remaining[remaining == idx + 1] = 0
                idx = remaining[x - 1 : x + 2, y - 1 : y + 2].max() - 1  # should only be one value
            mol_coords.append(coord_trace)
            try:
                idx = np.unique(remaining)[1] - 1  # avoid choosing 0
                coord_trace = ordered_segment_coords[idx].astype(np.int32).copy()
            except:  # index of -1 out of range
                break

        print(f"Mols in trace: {len(mol_coords)}")

        return mol_coords
    
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
            img[temp_img != 0] = mol_no + 1

        np.savetxt("/Users/Maxgamill/Desktop/preimg.txt", img)

        print("Crossings: ")
        lower_idxs, upper_idxs = self.get_trace_idxs(fwhms)

        if len(coord_trace) > 1:
            for type_idxs in [lower_idxs, upper_idxs]:
                for (node_crossing_coords, type_idx) in zip(crossing_coords, type_idxs):
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
            # make plot where overs are one colour and unders another
            for i, type_idxs in enumerate([lower_idxs, upper_idxs]):
                for (crossing, type_idx) in zip(crossing_coords, type_idxs):
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
            

    def get_pds(self, trace_coords, node_centres, fwhms, crossing_coords):
        # find idxs of branches from start
        for mol_trace in trace_coords:
            node_coord_idxs = np.array([]).astype(np.int32)
            global_node_idxs = np.array([]).astype(np.int32)
            img = np.zeros_like(self.skeleton.copy())
            for i, c in enumerate(node_centres):
                node_coord_idx = np.where((mol_trace[:, 0] == c[0]) & (mol_trace[:, 1] == c[1]))
                node_coord_idxs = np.append(node_coord_idxs, node_coord_idx)
                global_node_idx = np.zeros_like(node_coord_idx) + i
                global_node_idxs = np.append(global_node_idxs, global_node_idx)

            ordered_node_coord_idxs, ordered_node_idx_idxs = np.sort(node_coord_idxs), np.argsort(node_coord_idxs)
            global_node_idxs = global_node_idxs[ordered_node_idx_idxs]

            under_branch_idxs, _ = self.get_trace_idxs(fwhms)

            # iterate though nodes and label segments to node
            img[mol_trace[0 : ordered_node_coord_idxs[0], 0], mol_trace[0 : ordered_node_coord_idxs[0], 1]] = 1
            for i in range(0, len(ordered_node_coord_idxs) - 1):
                img[
                    mol_trace[ordered_node_coord_idxs[i] : ordered_node_coord_idxs[i + 1], 0],
                    mol_trace[ordered_node_coord_idxs[i] : ordered_node_coord_idxs[i + 1], 1],
                ] = (
                    i + 2
                )
            img[
                mol_trace[ordered_node_coord_idxs[-1] : -1, 0], mol_trace[ordered_node_coord_idxs[-1] : -1, 1]
            ] = 1  # rejoins start at 1

            # want to generate PD code by looking at each node and decide which
            #   img label is the under-in one, then append anti-clockwise labels
            #   - We'll have to match the node number to the node order
            #   - Then check the FWHMs to see lowest
            #   - Use lowest FWHM index to get the under branch coords
            #   - Count overlapping coords between under branch coords and each ordered segment
            #   - Get img label of two highest count (in and out)
            #   - Under-in = lowest of two indexes

            #print("global node idxs", global_node_idxs)
            for i, global_node_idx in enumerate(global_node_idxs):
                #print(f"\n----Trace Node Num: {i+1}, Global Node Num: {global_node_idx}----")
                under_branch_idx = under_branch_idxs[global_node_idx]
                #print("under_branch_idx: ", under_branch_idx)
                matching_coords = np.array([])
                x, y = node_centres[global_node_idx]
                node_area = img[x - 3 : x + 4, y - 3 : y + 4]
                uniq_labels = np.unique(node_area)
                uniq_labels = uniq_labels[uniq_labels != 0]

                for label2 in uniq_labels:
                    c = 0
                    # get overlaps between segment coords and crossing under coords
                    for ordered_branch_coord in crossing_coords[global_node_idx][
                        under_branch_idx
                    ]:  # for global_node[4] branch index is incorrect
                        c += ((np.stack(np.where(img == label2)).T == ordered_branch_coord).sum(axis=1) == 2).sum()
                    matching_coords = np.append(matching_coords, c)
                    # print(f"Segment: {label}, Matches: {c}")
                highest_count_labels = [uniq_labels[i] for i in np.argsort(matching_coords)[-2:]]
                under_in = min(highest_count_labels)  # under-in for global_node[4] is incorrect
                # print(f"Under-in: {under_in}")
                anti_clock = list(self.vals_anticlock(node_area, under_in))
                
                if len(anti_clock) == 2: # mol passes over/under another mol (maybe && [i]+1 == [i+1])
                    print(f"passive: X{anti_clock}")
                elif len(anti_clock) == 3: # trival crossing (maybe also applies to Y's therefore maybe && consec when sorted)
                    print(f"trivial: X{anti_clock}")
                else:
                    print(f"Real crossing: X{anti_clock}")

        return None

    @staticmethod
    def make_arr_consec(arr):
        for i, val in enumerate(arr):
            if i not in arr:
                arr[arr >= i] += -1
        return arr

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
        top = area[0, :][area[0, :] != 0][::-1]
        left = area[:, 0][area[:, 0] != 0]
        bottom = area[-1, :][area[-1, :] != 0]
        right = area[:-1][area[:-1] != 0][::-1]
        total = np.concatenate([top, left, bottom, right])

        # prevent multiple occurances while retaining order
        uniq_total_idxs = np.unique(total, return_index=True)[1]
        total = np.array([total[i] for i in sorted(uniq_total_idxs)])
        start_idx = np.where(total == start_lbl)[0]

        return np.roll(total, -start_idx)
