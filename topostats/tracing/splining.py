"""Order single pixel skeletons with or without NodeStats Statistics."""

import logging
import math

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import interpolate as interp

from topostats.logs.logs import LOGGER_NAME

LOGGER = logging.getLogger(LOGGER_NAME)

# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-locals
# pylint: disable=too-many-positional-arguments


# pylint: disable=too-many-instance-attributes
class splineTrace:
    # pylint: disable=too-many-instance-attributes
    """
    Smooth the ordered trace via an average of splines.

    Parameters
    ----------
    image : npt.NDArray
        Whole image containing all molecules and grains.
    mol_ordered_tracing_data : dict
        Molecule ordered trace dictionary containing Nx2 ordered coords and molecule statistics.
    pixel_to_nm_scaling : float
        The pixel to nm scaling factor, by default 1.
    spline_step_size : float
        Step length in meters to use a coordinate for splining.
    spline_linear_smoothing : float
        Amount of linear spline smoothing.
    spline_circular_smoothing : float
        Amount of circular spline smoothing.
    spline_degree : int
        Degree of the spline. Cubic splines are recommended. Even values of k should be avoided especially with a
        small s-value.
    """

    # pylint: disable=too-many-arguments
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
        # pylint: disable=too-many-arguments
        """
        Initialise the splineTrace class.

        Parameters
        ----------
        image : npt.NDArray
            Whole image containing all molecules and grains.
        mol_ordered_tracing_data : dict
            Nx2 ordered trace coordinates.
        pixel_to_nm_scaling : float
            The pixel to nm scaling factor, by default 1.
        spline_step_size : float
            Step length in meters to use a coordinate for splining.
        spline_linear_smoothing : float
            Amount of linear spline smoothing.
        spline_circular_smoothing : float
            Amount of circular spline smoothing.
        spline_degree : int
            Degree of the spline. Cubic splines are recommended. Even values of k should be avoided especially with a
            small s-value.
        """
        self.image = image
        self.number_of_rows, self.number_of_columns = image.shape
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

        Returns
        -------
        npt.NDArray
            Splined (smoothed) array of trace.
        """
        # Calculate the step size in pixels from the step size in metres.
        # Should always be at least 1.
        # Note that step_size_m is in m and pixel_to_nm_scaling is in m because of the legacy code which seems to almost
        # always have pixel_to_nm_scaling be set in metres using the flag convert_nm_to_m. No idea why this is the case.
        step_size_px = max(int(self.spline_step_size / (self.pixel_to_nm_scaling * 1e-9)), 1)
        # Splines will be totalled and then divived by number of splines to calculate the average spline
        spline_sum = None

        # Get the length of the fitted trace
        fitted_trace_length = fitted_trace.shape[0]

        # If the fitted trace is less than the degree plus one, then there is no
        # point in trying to spline it, just return the fitted trace
        if fitted_trace_length < self.spline_degree + 1:
            LOGGER.debug(
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
            tck = interp.splprep(
                [x_sampled, y_sampled],
                s=spline_smoothness,
                per=spline_periodicity,
                # quiet=self.spline_quiet,
                k=self.spline_degree,
            )[0]
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

        return np.divide(spline_sum, [step_size_px, step_size_px])

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

    def run_spline_trace(self) -> tuple[npt.NDArray, dict]:
        """
        Pipeline to run the splining smoothing and obtaining smoothing stats.

        Returns
        -------
        tuple[npt.NDArray, dict]
            Tuple of Nx2 smoothed trace coordinates, and smoothed trace statistics.
        """
        # fitted trace
        # fitted_trace = self.get_fitted_traces(self.ordered_trace, mol_is_circular)
        # splined trace
        splined_trace = self.get_splined_traces(self.mol_ordered_trace)
        # compile CL & E2E distance
        self.tracing_stats["contour_length"] = (
            measure_contour_length(splined_trace, self.mol_is_circular, self.pixel_to_nm_scaling) * 1e-9
        )
        self.tracing_stats["end_to_end_distance"] = (
            measure_end_to_end_distance(splined_trace, self.mol_is_circular, self.pixel_to_nm_scaling) * 1e-9
        )

        return splined_trace, self.tracing_stats


class windowTrace:
    """
    Obtain a smoothed trace of a molecule.

    Parameters
    ----------
    mol_ordered_tracing_data : dict
        Molecule ordered trace dictionary containing Nx2 ordered coords and molecule statistics.
    pixel_to_nm_scaling : float, optional
        The pixel to nm scaling factor, by default 1.
    rolling_window_size : np.float64, optional
        The length of the rolling window too average over, by default 6.0.
    rolling_window_resampling : bool, optional
        Whether to resample the rolling window, by default False.
    rolling_window_resample_regular_spatial_interval_nm : float, optional
        The regular spatial interval (nm) to resample the rolling window, by default 0.5.
    """

    def __init__(
        self,
        mol_ordered_tracing_data: dict,
        pixel_to_nm_scaling: float,
        rolling_window_size: float,
        rolling_window_resampling: bool = False,
        rolling_window_resample_regular_spatial_interval_nm: float = 0.5,
    ) -> None:
        """
        Initialise the windowTrace class.

        Parameters
        ----------
        mol_ordered_tracing_data : dict
            Molecule ordered trace dictionary containing Nx2 ordered coords and molecule statistics.
        pixel_to_nm_scaling : float, optional
            The pixel to nm scaling factor, by default 1.
        rolling_window_size : np.float64, optional
            The length of the rolling window too average over, by default 6.0.
        rolling_window_resampling : bool, optional
            Whether to resample the rolling window, by default False.
        rolling_window_resample_regular_spatial_interval_nm : float, optional
            The regular spatial interval (nm) to resample the rolling window, by default 0.5.
        """
        self.mol_ordered_trace = mol_ordered_tracing_data["ordered_coords"]
        self.mol_is_circular = mol_ordered_tracing_data["mol_stats"]["circular"]
        self.pixel_to_nm_scaling = pixel_to_nm_scaling
        self.rolling_window_size = rolling_window_size / 1e-9  # for nm scaling factor
        self.rolling_window_resampling = rolling_window_resampling
        self.rolling_window_resample_regular_spatial_interval_nm = (
            rolling_window_resample_regular_spatial_interval_nm / 1e-9
        )  # for nm scaling factor

        self.tracing_stats = {
            "contour_length": None,
            "end_to_end_distance": None,
        }

    @staticmethod
    def pool_trace_circular(
        pixel_trace: npt.NDArray[np.int32], rolling_window_size: np.float64 = 6.0, pixel_to_nm_scaling: float = 1
    ) -> npt.NDArray[np.float64]:
        """
        Smooth a pixelwise ordered trace of circular molecules via a sliding window.

        Parameters
        ----------
        pixel_trace : npt.NDArray[np.int32]
            Nx2 ordered trace coordinates.
        rolling_window_size : np.float64, optional
            The length of the rolling window too average over, by default 6.0.
        pixel_to_nm_scaling : float, optional
            The pixel to nm scaling factor, by default 1.

        Returns
        -------
        npt.NDArray[np.float64]
            MxN Smoothed ordered trace coordinates.
        """
        # Pool the trace points
        pooled_trace = []

        for i in range(len(pixel_trace)):
            binned_points = []
            current_length = 0
            j = 1

            # compile rolling window
            while current_length < rolling_window_size:
                current_index = i + j
                previous_index = i + j - 1
                while current_index >= len(pixel_trace):
                    current_index -= len(pixel_trace)
                while previous_index >= len(pixel_trace):
                    previous_index -= len(pixel_trace)
                current_length += (
                    np.linalg.norm(pixel_trace[current_index] - pixel_trace[previous_index]) * pixel_to_nm_scaling
                )
                binned_points.append(pixel_trace[current_index])
                j += 1

            # Get the mean of the binned points
            pooled_trace.append(np.mean(binned_points, axis=0))

        return np.array(pooled_trace)

    @staticmethod
    def pool_trace_linear(
        pixel_trace: npt.NDArray[np.int32],
        rolling_window_size: np.float64 = 6.0,
        pixel_to_nm_scaling: float = 1,
    ) -> npt.NDArray[np.float64]:
        """
        Smooth a pixelwise ordered trace of linear molecules via a sliding window.

        Parameters
        ----------
        pixel_trace : npt.NDArray[np.int32]
            Nx2 ordered trace coordinates.
        rolling_window_size : np.float64, optional
            The length of the rolling window too average over, by default 6.0.
        pixel_to_nm_scaling : float, optional
            The pixel to nm scaling factor, by default 1.

        Returns
        -------
        npt.NDArray[np.float64]
            MxN Smoothed ordered trace coordinates.
        """
        pooled_trace = [pixel_trace[0]]  # Add first coord as to not cut it off

        # Get average point for trace in rolling window
        for i in range(0, len(pixel_trace)):
            binned_points = []
            current_length = 0
            j = 0
            # Compile rolling window
            while current_length < rolling_window_size:
                current_index = i + j
                previous_index = i + j - 1
                if current_index >= len(pixel_trace):  # exit if exceeding the trace
                    break
                current_length += (
                    np.linalg.norm(pixel_trace[current_index] - pixel_trace[previous_index]) * pixel_to_nm_scaling
                )
                binned_points.append(pixel_trace[current_index])
                j += 1
            else:
                # Get the mean of the binned points
                pooled_trace.append(np.mean(binned_points, axis=0))

            # Exit if reached the end of the trace
            if current_index + 1 >= len(pixel_trace):
                break

        pooled_trace.append(pixel_trace[-1])  # Add last coord as to not cut it off

        # Check if the first two points are the same and remove the first point if they are
        # This can happen if the algorithm happens to add the first point naturally due to having a small
        # rolling window size.
        if np.array_equal(pooled_trace[0], pooled_trace[1]):
            pooled_trace.pop(0)

        # Check if the last two points are the same and remove the last point if they are
        # This can happen if the algorithm happens to add the last point naturally due to having a small
        # rolling window size.
        if np.array_equal(pooled_trace[-1], pooled_trace[-2]):
            pooled_trace.pop(-1)

        return np.array(pooled_trace)

    def run_window_trace(self) -> tuple[npt.NDArray, dict]:
        """
        Pipeline to run the rolling window smoothing and obtaining smoothing stats.

        Returns
        -------
        tuple[npt.NDArray, dict]
            Tuple of Nx2 smoothed trace coordinates, and smoothed trace statistics.
        """
        # fitted trace
        # fitted_trace = self.get_fitted_traces(self.ordered_trace, mol_is_circular)
        # splined trace
        if self.mol_is_circular:
            splined_trace = self.pool_trace_circular(
                self.mol_ordered_trace, self.rolling_window_size, self.pixel_to_nm_scaling
            )
            if self.rolling_window_resampling:
                splined_trace = resample_points_regular_interval(
                    points=splined_trace,
                    interval=self.rolling_window_resample_regular_spatial_interval_nm / self.pixel_to_nm_scaling,
                    circular=True,
                )
        else:
            splined_trace = self.pool_trace_linear(
                self.mol_ordered_trace, self.rolling_window_size, self.pixel_to_nm_scaling
            )
            if self.rolling_window_resampling:
                splined_trace = resample_points_regular_interval(
                    points=splined_trace,
                    interval=self.rolling_window_resample_regular_spatial_interval_nm / self.pixel_to_nm_scaling,
                    circular=False,
                )
        # compile CL & E2E distance
        self.tracing_stats["contour_length"] = (
            measure_contour_length(splined_trace, self.mol_is_circular, self.pixel_to_nm_scaling) * 1e-9
        )
        self.tracing_stats["end_to_end_distance"] = (
            measure_end_to_end_distance(splined_trace, self.mol_is_circular, self.pixel_to_nm_scaling) * 1e-9
        )

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
    pixel_to_nm_scaling : float
        Scaling factor from pixels to nanometres.

    Returns
    -------
    float
        Length of molecule in nanometres (nm).
    """
    if mol_is_circular:
        for num in range(len(splined_trace)):
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
        for num in range(len(splined_trace)):
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
    pixel_to_nm_scaling : float
        Scaling factor from pixels to nanometres.

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


# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
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
    rolling_window_resampling: bool = False,
    rolling_window_resample_regular_spatial_interval: float = 0.5,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals
    """
    Obtain smoothed traces of pixel-wise ordered traces for molecules in an image.

    Parameters
    ----------
    image : npt.NDArray
        Whole image containing all molecules and grains.
    ordered_tracing_direction_data : dict
        Dictionary result from the ordered traces.
    pixel_to_nm_scaling : float
        Scaling factor from pixels to nanometres.
    filename : str
        Name of the image file.
    method : str
        Method of trace smoothing, options are 'splining' and 'rolling_window'.
    rolling_window_size : float
        Length in meters to average coordinates over in the rolling window.
    spline_step_size : float
        Step length in meters to use a coordinate for splining.
    spline_linear_smoothing : float
        Amount of linear spline smoothing.
    spline_circular_smoothing : float
        Amount of circular spline smoothing.
    spline_degree : int
        Degree of the spline. Cubic splines are recommended. Even values of k should be avoided especially with a
        small s-value.
    rolling_window_resampling : bool, optional
        Whether to resample the rolling window, by default False.
    rolling_window_resample_regular_spatial_interval : float, optional
        The regular spatial interval (nm) to resample the rolling window, by default 0.5.

    Returns
    -------
    tuple[dict, pd.DataFrame, pd.DataFrame]
        A spline data dictionary for all molecules, and a grainstats dataframe additions dataframe and molecule
        statistics dataframe.
    """
    grainstats_additions = {}
    molstats = {}
    all_splines_data = {}

    mol_count = 0
    for mol_trace_data in ordered_tracing_direction_data.values():
        mol_count += len(mol_trace_data)
    LOGGER.info(f"[{filename}] : Calculating Splining statistics for {mol_count} molecules...")

    # iterate through disordered_tracing_dict
    for grain_no, ordered_grain_data in ordered_tracing_direction_data.items():
        grain_trace_stats = {"total_contour_length": 0, "average_end_to_end_distance": 0}
        all_splines_data[grain_no] = {}
        mol_no = None
        for mol_no, mol_trace_data in ordered_grain_data.items():
            try:
                LOGGER.debug(f"[{filename}] : Splining {grain_no} - {mol_no}")
                # check if want to do nodestats tracing or not
                if method == "rolling_window":
                    splined_data, tracing_stats = windowTrace(
                        mol_ordered_tracing_data=mol_trace_data,
                        pixel_to_nm_scaling=pixel_to_nm_scaling,
                        rolling_window_size=rolling_window_size,
                        rolling_window_resampling=rolling_window_resampling,
                        rolling_window_resample_regular_spatial_interval_nm=rolling_window_resample_regular_spatial_interval,
                    ).run_window_trace()

                # if not doing nodestats ordering, do original TS ordering
                else:  # method == "spline":
                    splined_data, tracing_stats = splineTrace(
                        image=image,
                        mol_ordered_tracing_data=mol_trace_data,
                        pixel_to_nm_scaling=pixel_to_nm_scaling,
                        spline_step_size=spline_step_size,
                        spline_linear_smoothing=spline_linear_smoothing,
                        spline_circular_smoothing=spline_circular_smoothing,
                        spline_degree=spline_degree,
                    ).run_spline_trace()

                # get combined stats for the grains
                grain_trace_stats["total_contour_length"] += tracing_stats["contour_length"]
                grain_trace_stats["average_end_to_end_distance"] += tracing_stats["end_to_end_distance"]

                # get individual mol stats
                all_splines_data[grain_no][mol_no] = {
                    "spline_coords": splined_data,
                    "bbox": mol_trace_data["bbox"],
                    "tracing_stats": tracing_stats,
                }
                molstats[grain_no.split("_")[-1] + "_" + mol_no.split("_")[-1]] = {
                    "image": filename,
                    "grain_number": int(grain_no.split("_")[-1]),
                    "molecule_number": int(mol_no.split("_")[-1]),
                }
                molstats[grain_no.split("_")[-1] + "_" + mol_no.split("_")[-1]].update(tracing_stats)
                LOGGER.debug(f"[{filename}] : Finished splining {grain_no} - {mol_no}")

            except Exception as e:  # pylint: disable=broad-exception-caught
                LOGGER.error(
                    f"[{filename}] : Splining for {grain_no} failed. Consider raising an issue on GitHub. Error: ",
                    exc_info=e,
                )
                all_splines_data[grain_no] = {}

        if mol_no is None:
            LOGGER.warning(f"[{filename}] : No molecules found for grain {grain_no}")
        else:
            # average the e2e dists -> mol_no should always be in the grain dict
            grain_trace_stats["average_end_to_end_distance"] /= len(ordered_grain_data)

        # compile metrics
        grainstats_additions[grain_no] = {
            "image": filename,
            "grain_number": int(grain_no.split("_")[-1]),
        }
        grainstats_additions[grain_no].update(grain_trace_stats)

    # convert grainstats metrics to dataframe
    splining_stats_df = pd.DataFrame.from_dict(grainstats_additions, orient="index")
    molstats_df = pd.DataFrame.from_dict(molstats, orient="index")
    molstats_df.reset_index(drop=True, inplace=True)
    return all_splines_data, splining_stats_df, molstats_df


def interpolate_between_two_points_distance(
    point1: npt.NDArray[np.float32], point2: npt.NDArray[np.float32], distance: np.float32
) -> npt.NDArray[np.float32]:
    """
    Interpolate between two points to create a new point at a set distance between the two.

    Parameters
    ----------
    point1 : npt.NDArray[np.float32]
        The first point.
    point2 : npt.NDArray[np.float32]
        The second point.
    distance : np.float32
        The distance to interpolate between the two points.

    Returns
    -------
    npt.NDArray[np.float32]
        The new point at the specified distance between the two points.
    """
    distance_between_points = np.linalg.norm(point2 - point1)
    assert (
        distance_between_points > distance
    ), f"distance between points is less than the desired interval: {distance_between_points} < {distance}"
    proportion = distance / distance_between_points
    return point1 + proportion * (point2 - point1)


def resample_points_regular_interval(points: npt.NDArray, interval: float, circular: bool) -> npt.NDArray:
    """
    Resample a set of points to be at regular spatial intervals.

    Note: This is NOT intended to be pure interpolation, as interpolated points would not produce uniformly spaced
    points in cartesian space.

    Parameters
    ----------
    points : npt.NDArray
        The points to resample.
    interval : float
        The distance that all returned points should be apart.
    circular : bool
        If True, the first and last points will be connected to form a closed loop. If False, the first and last points
        will not be connected.

    Returns
    -------
    npt.NDArray
        The resampled points, evenly spaced at the specified interval.
    """
    if circular:
        points = np.concatenate((points, points[0:1]), axis=0)

    resampled_points = []
    resampled_points.append(points[0])
    current_point_index = 1
    while True:
        current_point = resampled_points[-1]
        next_original_point = points[current_point_index]
        distance_to_next_splined_point = np.linalg.norm(next_original_point - current_point)
        # if the distance to the next splined point is less than the interval, then skip to the next point
        if distance_to_next_splined_point < interval:
            current_point_index += 1
            if current_point_index >= len(points):
                break
            continue
        new_interpolated_point = interpolate_between_two_points_distance(
            point1=current_point, point2=next_original_point, distance=interval
        )
        resampled_points.append(new_interpolated_point)

    # if the first and last points are less than 0.5 * the interval apart, then remove the last point
    if np.linalg.norm(resampled_points[0] - resampled_points[-1]) < 0.5 * interval:
        resampled_points = resampled_points[:-1]

    return np.array(resampled_points)
