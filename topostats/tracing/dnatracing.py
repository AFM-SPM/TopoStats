"""Perform DNA Tracing."""

from __future__ import annotations

from collections import OrderedDict
from functools import partial
from itertools import repeat
import logging
import math
from multiprocessing import Pool
import os
from pathlib import Path
from typing import Dict, List, Union, Tuple
import warnings

import numpy as np
import numpy.typing as npt
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
from topostats.utils import bound_padded_coordinates_to_image

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
    min_skeleton_size : int
        Minimum skeleton size below which tracing statistics are not calculated.
    convert_nm_to_m : bool
        Convert nanometers to metres.
    skeletonisation_method : str
        Method of skeletonisation to use 'topostats' is the original TopoStats method. Three methods from
        scikit-image are available 'zhang', 'lee' and 'thin'.
    n_grain : int
        Grain number being processed (only  used in logging).
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
        min_skeleton_size: int = 10,
        convert_nm_to_m: bool = True,
        skeletonisation_method: str = "topostats",
        n_grain: int = None,
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
        min_skeleton_size : int
            Minimum skeleton size below which tracing statistics are not calculated.
        convert_nm_to_m : bool
            Convert nanometers to metres.
        skeletonisation_method : str
            Method of skeletonisation to use 'topostats' is the original TopoStats method. Three methods from
            scikit-image are available 'zhang', 'lee' and 'thin'.
        n_grain : int
            Grain number being processed (only  used in logging).
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

        # Splining parameters
        self.spline_step_size: float = spline_step_size
        self.spline_linear_smoothing: float = spline_linear_smoothing
        self.spline_circular_smoothing: float = spline_circular_smoothing
        self.spline_quiet: bool = spline_quiet
        self.spline_degree: int = spline_degree

        self.neighbours = 5  # The number of neighbours used for the curvature measurement

        self.ordered_trace_heights = None
        self.ordered_trace_cumulative_distances = None

        # suppresses scipy splining warnings
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
            self.get_ordered_trace_heights()
            self.get_ordered_trace_cumulative_distances()
            self.linear_or_circular(self.ordered_trace)
            self.get_fitted_traces()
            self.get_splined_traces()
            # self.find_curvature()
            # self.saveCurvature()
            self.measure_contour_length()
            self.measure_end_to_end_distance()
        else:
            LOGGER.info(f"[{self.filename}] [{self.n_grain}] : Grain skeleton pixels < {self.min_skeleton_size}")

    def gaussian_filter(self, **kwargs) -> np.array:
        """
        Apply Gaussian filter.

        Parameters
        ----------
        **kwargs
            Arguments passed to 'skimage.filter.gaussian(**kwargs)'.
        """
        self.gauss_image = gaussian(self.image, sigma=self.sigma, **kwargs)
        LOGGER.info(f"[{self.filename}] [{self.n_grain}] : Gaussian filter applied.")

    def get_ordered_trace_heights(self) -> None:
        """
        Derive the pixel heights from the ordered trace `self.ordered_trace` list.

        Gets the heights of each pixel in the ordered trace from the gaussian filtered image. The pixel coordinates
        for the ordered trace are stored in the ordered trace list as part of the class.
        """
        self.ordered_trace_heights = np.array(self.gauss_image[self.ordered_trace[:, 0], self.ordered_trace[:, 1]])

    def get_ordered_trace_cumulative_distances(self) -> None:
        """Calculate the cumulative distances of each pixel in the `self.ordered_trace` list."""

        # Get the cumulative distances of each pixel in the ordered trace from the gaussian filtered image
        # the pixel coordinates are stored in the ordered trace list.
        self.ordered_trace_cumulative_distances = self.coord_dist(
            coordinates=self.ordered_trace, px_to_nm=self.pixel_to_nm_scaling
        )

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

    def get_disordered_trace(self) -> None:
        """
        Create a skeleton for each of the grains in the image.

        Uses my own skeletonisation function from tracingfuncs module. I (Joe) will eventually get round to editing this
        function to try to reduce the branching  and to try to better trace from looped molecules.
        """
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

    def linear_or_circular(self, traces) -> None:
        """
        Determine whether molecule is circular or linear based on the local environment of each pixel from the trace.

        This function is sensitive to branches from the skeleton so might need to implement a function to remove them.

        Parameters
        ----------
        traces : npt.NDArray
            The array of coordinates to be assessed.
        """

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
        """Order a trace."""
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

    def get_fitted_traces(self):
        """Create trace coordinates that are adjusted to lie along the highest points of each traced molecule."""

        individual_skeleton = self.ordered_trace
        # This indexes a 3 nm height profile perpendicular to DNA backbone
        # note that this is a hard coded parameter
        index_width = int(3e-9 / (self.pixel_to_nm_scaling))
        if index_width < 2:
            index_width = 2

        for coord_num, trace_coordinate in enumerate(individual_skeleton):
            height_values = None

            # Ensure that padding will not exceed the image boundaries
            trace_coordinate = bound_padded_coordinates_to_image(
                coordinates=trace_coordinate,
                padding=index_width,
                image_shape=(self.number_of_rows, self.number_of_columns),
            )

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

            # Use the perp array to index the gaussian filtered image
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
    ) -> None:
        """
        Get a splined version of the fitted trace - useful for finding the radius of gyration etc.

        This function actually calculates the average of several splines which is important for getting a good fit on
        the lower resolution data.
        """

        # Fitted traces are Nx2 numpy arrays of coordinates
        # All self references are here for easy turning into static method if wanted, also type hints and short documentation
        fitted_trace: np.ndarray = self.fitted_trace  # boolean 2d numpy array of fitted traces to spline
        step_size_m: float = self.spline_step_size  # the step size for the splines to skip pixels in the fitted trace
        pixel_to_nm_scaling: float = self.pixel_to_nm_scaling  # pixel to nanometre scaling factor for the image
        mol_is_circular: bool = self.mol_is_circular  # whether or not the molecule is classed as circular
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

        self.splined_trace = spline_average

    def show_traces(self):
        """Plot traces."""
        plt.pcolormesh(self.gauss_image, vmax=-3e-9, vmin=3e-9)
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

    def measure_contour_length(self) -> None:
        """
        Contour lengthof the splined trace taking into account whether the molecule is circular or linear.

        Contour length units are nm.
        """
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
        """
        Calculate the Euclidean distance between the start and end of linear molecules.

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
    image: npt.NDArray,
    grains_mask: npt.NDArray,
    filename: str,
    pixel_to_nm_scaling: float,
    min_skeleton_size: int,
    skeletonisation_method: str,
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
    skeletonisation_method : str
        Method of skeletonisation, options are 'zhang' (scikit-image) / 'lee' (scikit-image) / 'thin' (scikitimage) or
       'topostats' (original TopoStats method).
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
    # Check both arrays are the same shape
    if image.shape != grains_mask.shape:
        raise ValueError(f"Image shape ({image.shape}) and Mask shape ({grains_mask.shape}) should match.")

    cropped_images, cropped_masks = prep_arrays(image, grains_mask, pad_width)
    region_properties = skimage_measure.regionprops(grains_mask)
    grain_anchors = [grain_anchor(image.shape, list(grain.bbox), pad_width) for grain in region_properties]
    n_grains = len(cropped_images)
    LOGGER.info(f"[{filename}] : Calculating statistics for {n_grains} grains.")
    results = {}
    ordered_traces = {}
    splined_traces = {}
    all_ordered_trace_heights = {}
    all_ordered_trace_cumulative_distances = {}
    for cropped_image_index, cropped_image in cropped_images.items():
        cropped_mask = cropped_masks[cropped_image_index]

        result = trace_grain(
            cropped_image,
            cropped_mask,
            pixel_to_nm_scaling,
            filename,
            min_skeleton_size,
            skeletonisation_method,
            spline_step_size,
            spline_linear_smoothing,
            spline_circular_smoothing,
            cropped_image_index,
        )
        LOGGER.info(f"[{filename}] : Traced grain {cropped_image_index + 1} of {n_grains}")
        ordered_traces[cropped_image_index] = result.pop("ordered_trace")
        splined_traces[cropped_image_index] = result.pop("splined_trace")
        all_ordered_trace_heights[cropped_image_index] = result.pop("ordered_trace_heights")
        all_ordered_trace_cumulative_distances[cropped_image_index] = result.pop("ordered_trace_cumulative_distances")
        results[cropped_image_index] = result
    try:
        results = pd.DataFrame.from_dict(results, orient="index")
        results.index.name = "molecule_number"
        image_trace = trace_mask(grain_anchors, ordered_traces, image.shape, pad_width)
        rounded_splined_traces = round_splined_traces(splined_traces=splined_traces)
        image_spline_trace = trace_mask(grain_anchors, rounded_splined_traces, image.shape, pad_width)
    except ValueError as error:
        LOGGER.error("No grains found in any images, consider adjusting your thresholds.")
        LOGGER.error(error)
    return {
        "statistics": results,
        "all_ordered_traces": ordered_traces,
        "all_splined_traces": splined_traces,
        "all_cropped_images": cropped_images,
        "image_ordered_trace": image_trace,
        "image_spline_trace": image_spline_trace,
        "all_ordered_trace_heights": all_ordered_trace_heights,
        "all_ordered_trace_cumulative_distances": all_ordered_trace_cumulative_distances,
    }


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
    cropped_images = {index: crop_array(image, grain.bbox, pad_width) for index, grain in enumerate(region_properties)}
    cropped_images = {index: np.pad(grain, pad_width=pad_width) for index, grain in cropped_images.items()}
    cropped_masks = {
        index: crop_array(labelled_grains_mask, grain.bbox, pad_width) for index, grain in enumerate(region_properties)
    }
    cropped_masks = {index: np.pad(grain, pad_width=pad_width) for index, grain in cropped_masks.items()}
    # Flip every labelled region to be 1 instead of its label
    cropped_masks = {index: np.where(grain == 0, 0, 1) for index, grain in cropped_masks.items()}
    return (cropped_images, cropped_masks)


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
    filename: str = None,
    min_skeleton_size: int = 10,
    skeletonisation_method: str = "topostats",
    spline_step_size: float = 7e-9,
    spline_linear_smoothing: float = 5.0,
    spline_circular_smoothing: float = 0.0,
    n_grain: int = None,
) -> dict:
    """
    Trace an individual grain.

    Tracing involves multiple steps...

    1. Skeletonisation
    2. Pruning of side branch artefacts from skeletonisation.
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
    filename : str
        File being processed.
    min_skeleton_size : int
        Minimum size of grain in pixels after skeletonisation.
    skeletonisation_method : str
        Method of skeletonisation, options are 'zhang' (scikit-image) / 'lee' (scikit-image) / 'thin' (scikitimage) or
       'topostats' (original TopoStats method).
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
        skeletonisation_method=skeletonisation_method,
        spline_step_size=spline_step_size,
        spline_linear_smoothing=spline_linear_smoothing,
        spline_circular_smoothing=spline_circular_smoothing,
        n_grain=n_grain,
    )
    dnatrace.trace_dna()
    return {
        "image": dnatrace.filename,
        "contour_length": dnatrace.contour_length,
        "circular": dnatrace.mol_is_circular,
        "end_to_end_distance": dnatrace.end_to_end_distance,
        "ordered_trace": dnatrace.ordered_trace,
        "splined_trace": dnatrace.splined_trace,
        "ordered_trace_heights": dnatrace.ordered_trace_heights,
        "ordered_trace_cumulative_distances": dnatrace.ordered_trace_cumulative_distances,
    }


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
