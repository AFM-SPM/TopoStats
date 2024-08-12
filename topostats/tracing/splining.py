"""Order single pixel skeletons with or without NodeStats Statistics."""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt
import pandas as pd
from skimage.morphology import label
from scipy import ndimage, interpolate as interp
import math

from topostats.logs.logs import LOGGER_NAME
from topostats.tracing.tracingfuncs import coord_dist, genTracingFuncs, order_branch, reorderTrace
from topostats.utils import convolve_skeleton, coords_2_img

LOGGER = logging.getLogger(LOGGER_NAME)


class splineTrace:

    def __init__(
        self,
        image: npt.NDArray,
        mol_ordered_tracing_data: dict,
        pixel_to_nm_scaling: float,
        spline_step_size: float,
        spline_linear_smoothing: float,
        spline_circular_smoothing: float,
        spline_degree: int,
    ) -> None:
        
        self.image = image
        self.mol_ordered_trace = mol_ordered_tracing_data["ordered_coords"]
        self.mol_is_circular = mol_ordered_tracing_data["mol_stats"]["circular"]
        self.pixel_to_nm_scaling = pixel_to_nm_scaling
        self.spline_step_size = spline_step_size
        self.spline_linear_smoothing = spline_linear_smoothing
        self.spline_circular_smoothing = spline_circular_smoothing
        self.spline_degree = spline_degree

        self.tracing_stats = {
            "contour_length": None,
            "end_to_end_distance": None,
        }
    
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
            # e.g. indexing self.smoothed_mask[130, 130] for 128x128 image
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
                height_values = self.smoothed_mask[perp_array[:, 0], perp_array[:, 1]]
            except IndexError:
                perp_array[:, 0] = np.where(
                    perp_array[:, 0] > self.smoothed_mask.shape[0], self.smoothed_mask.shape[0], perp_array[:, 0]
                )
                perp_array[:, 1] = np.where(
                    perp_array[:, 1] > self.smoothed_mask.shape[1], self.smoothed_mask.shape[1], perp_array[:, 1]
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

    def get_splined_traces(
        self,
        fitted_trace: npt.NDArray,
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

        # Calculate the step size in pixels from the step size in metres.
        # Should always be at least 1.
        # Note that step_size_m is in m and pixel_to_nm_scaling is in m because of the legacy code which seems to almost always have
        # pixel_to_nm_scaling be set in metres using the flag convert_nm_to_m. No idea why this is the case.
        step_size_px = max(int(self.spline_step_size / (self.pixel_to_nm_scaling * 1e-9)), 1)
        # Splines will be totalled and then divived by number of splines to calculate the average spline
        spline_sum = None

        # Get the length of the fitted trace
        fitted_trace_length = fitted_trace.shape[0]

        # If the fitted trace is less than the degree plus one, then there is no
        # point in trying to spline it, just return the fitted trace
        if fitted_trace_length < self.spline_degree + 1:
            LOGGER.warning(
                f"Fitted trace for grain {step_size_px} too small ({fitted_trace_length}), returning fitted trace"
            )

            return fitted_trace

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
            (self.spline_circular_smoothing, 2) if self.mol_is_circular else (self.spline_linear_smoothing, 0)
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
                #quiet=self.spline_quiet,
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

    def run_spline_trace(self):
        # fitted trace
        #fitted_trace = self.get_fitted_traces(self.ordered_trace, mol_is_circular)
        # splined trace
        # splined_trace = self.get_splined_traces(fitted_trace, mol_is_circular)
        splined_trace = self.get_splined_traces(self.mol_ordered_trace)
        # compile CL & E2E distance
        self.tracing_stats["contour_length"] = measure_contour_length(splined_trace, self.mol_is_circular, self.pixel_to_nm_scaling)
        self.tracing_stats["end_to_end_distance"] = measure_end_to_end_distance(splined_trace, self.mol_is_circular, self.pixel_to_nm_scaling)
        # compile images?

        return splined_trace, self.tracing_stats
    

class windowTrace:
    
    def __init__(
        self,
        mol_ordered_tracing_data: dict,
        pixel_to_nm_scaling: float,
        rolling_window_size: float,
    ) -> None:
        
        self.mol_ordered_trace = mol_ordered_tracing_data["ordered_coords"]
        self.mol_is_circular = mol_ordered_tracing_data["mol_stats"]["circular"]
        self.pixel_to_nm_scaling = pixel_to_nm_scaling
        self.rolling_window_size = rolling_window_size

        self.tracing_stats = {
            "contour_length": None,
            "end_to_end_distance": None,
        }
    
    @staticmethod
    def pool_trace(pixel_trace: npt.NDArray[np.int32], rolling_window_size: np.float64 = 6.0, pixel_to_nm_scaling: float = 1) -> npt.NDArray[np.float64]:
        # Pool the trace points
        pooled_trace = []

        for i in range(len(pixel_trace)):
            binned_points = []
            current_length = 0
            j = 1
            # compile roalling window
            while current_length < rolling_window_size:
                current_index = i + j
                previous_index = i + j - 1
                while current_index >= len(pixel_trace):
                    current_index -= len(pixel_trace)
                while previous_index >= len(pixel_trace):
                    previous_index -= len(pixel_trace)
                current_length += np.linalg.norm(pixel_trace[current_index] - pixel_trace[previous_index]) * pixel_to_nm_scaling
                binned_points.append(pixel_trace[current_index])
                j += 1

            # Get the mean of the binned points
            pooled_trace.append(np.mean(binned_points, axis=0))

        return np.array(pooled_trace)
    
    def run_window_trace(self):
        # fitted trace
        #fitted_trace = self.get_fitted_traces(self.ordered_trace, mol_is_circular)
        # splined trace
        splined_trace = self.pool_trace(self.mol_ordered_trace, self.rolling_window_size, self.pixel_to_nm_scaling)
        # compile CL & E2E distance
        self.tracing_stats["contour_length"] = measure_contour_length(splined_trace, self.mol_is_circular, self.pixel_to_nm_scaling)
        self.tracing_stats["end_to_end_distance"] = measure_end_to_end_distance(splined_trace, self.mol_is_circular, self.pixel_to_nm_scaling)

        return splined_trace, self.tracing_stats
    

def measure_contour_length(splined_trace: npt.NDArray, mol_is_circular: bool, pixel_to_nm_scaling: float) -> float:
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

        contour_length = np.sum(np.array(hypotenuse_array)) * pixel_to_nm_scaling
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
                contour_length = np.sum(np.array(hypotenuse_array)) * pixel_to_nm_scaling
                del hypotenuse_array
                break
    return contour_length

def measure_end_to_end_distance(splined_trace, mol_is_circular, pixel_to_nm_scaling: float):
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
            * pixel_to_nm_scaling
        )
    return 0


def splining_image(
    image: npt.NDArray,
    ordered_tracing_direction_data: dict,
    pixel_to_nm_scaling: float,
    filename: str,
    method: str,
    rolling_window_size: float,
    spline_step_size: float,
    spline_linear_smoothing: float,
    spline_circular_smoothing: float,
    spline_degree: int,
    pad_width: int,
):
    grainstats_additions = {}
    all_splines_data = {}

    # iterate through disordered_tracing_dict
    for grain_no, ordered_grain_data in ordered_tracing_direction_data.items():
        grain_trace_stats = {"total_contour_length": 0, "average_end_to_end_distance": 0}
        all_splines_data[grain_no] = {}
        for mol_no, mol_trace_data in ordered_grain_data.items():
            LOGGER.info(f"[{filename}] : Splining {grain_no} - {mol_no}")
            # check if want to do nodestats tracing or not
            if method == "spline":
                spline = splineTrace(
                    image=image,
                    mol_ordered_tracing_data=mol_trace_data,
                    pixel_to_nm_scaling=pixel_to_nm_scaling,
                    spline_step_size=spline_step_size,
                    spline_linear_smoothing=spline_linear_smoothing,
                    spline_circular_smoothing=spline_circular_smoothing,
                    spline_degree=spline_degree,
                )
                splined_data, tracing_stats = spline.run_spline_trace()

            # if not doing nodestats ordering, do original TS ordering
            elif method == "rolling_window":
                smooth = windowTrace(
                    mol_ordered_tracing_data=mol_trace_data,
                    pixel_to_nm_scaling=pixel_to_nm_scaling,
                    rolling_window_size=rolling_window_size,
                )
                splined_data, tracing_stats = smooth.run_window_trace()
            else:
                LOGGER.warning("Neither 'spline' or 'rolling_window' methods are being used.")

            # get combined stats for the grains
            grain_trace_stats["total_contour_length"] += tracing_stats["contour_length"]
            grain_trace_stats["average_end_to_end_distance"] += tracing_stats["end_to_end_distance"]
            
            # get individual mol stats
            all_splines_data[grain_no][mol_no] = {
                "spline_coords": splined_data,
                "bbox": mol_trace_data["bbox"],
                "tracing_stats": tracing_stats,
            }
            LOGGER.info(f"[{filename}] : Finished splining {grain_no} - {mol_no}")
            
        # average the e2e dists
        grain_trace_stats["average_end_to_end_distance"] /= int(mol_no.split('_')[-1]) + 1

        # compile metrics
        grainstats_additions[grain_no] = {
            "image": filename,
            "grain_number": int(grain_no.split("_")[-1]),
        }
        grainstats_additions[grain_no].update(grain_trace_stats)

    # convert grainstats metrics to dataframe
    grainstats_additions_df = pd.DataFrame.from_dict(grainstats_additions, orient="index")

    return all_splines_data, grainstats_additions_df
