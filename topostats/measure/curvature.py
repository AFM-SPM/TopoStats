"""Calculate various curvature metrics for traces."""

import logging
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

from topostats.array_manipulation import distances_nm
from topostats.classes import GrainCurvatureStats, MoleculeCurvatureStats, TopoStats
from topostats.logs.logs import LOGGER_NAME

LOGGER = logging.getLogger(LOGGER_NAME)


def angle_diff_signed(v1: npt.NDArray[np.float64], v2: npt.NDArray[np.float64]):
    """
    Calculate the signed angle difference between two point vecrtors in 2D space.

    Positive angles are clockwise, negative angles are counterclockwise.

    Parameters
    ----------
    v1 : npt.NDArray[np.float64]
        First vector.
    v2 : npt.NDArray[np.float64]
        Second vector.

    Returns
    -------
    float
        The signed angle difference in radians.
    """
    if v1.shape != (2,) or v2.shape != (2,):
        raise ValueError("Vectors must be of shape (2,)")

    angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
    if angle > np.pi:
        angle -= 2 * np.pi
    elif angle < -np.pi:
        angle += 2 * np.pi

    return angle


def calculate_discrete_angle_difference_linear(trace: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Calculate the discrete angle difference per point along a linear trace.

    Parameters
    ----------
    trace : npt.NDArray[np.float64]
        The coordinate trace, in any units.

    Returns
    -------
    npt.NDArray[np.float64]
        The discrete angle difference per point in radians.
    """
    angles = np.zeros(trace.shape[0])
    for index, point in enumerate(trace):
        if index == 0:
            # No previous point so cannot calculate angle
            angle = 0.0
        elif index == trace.shape[0] - 1:
            # No next point so cannot calculate angle, end of trace
            v1 = point - trace[index - 1]
            angle = 0.0
        else:
            v1 = point - trace[index - 1]
            v2 = trace[index + 1] - point

            # Normalise vectors to unit length
            norm_v1 = v1 / np.linalg.norm(v1)
            norm_v2 = v2 / np.linalg.norm(v2)

            # Calculate the signed angle difference between the previous direction and the current direction
            angle = angle_diff_signed(norm_v1, norm_v2)
        angles[index] = angle
    return angles


def calculate_discrete_angle_difference_circular(trace: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Calculate the discrete angle difference per point along a circular trace.

    Parameters
    ----------
    trace : npt.NDArray[np.float64]
        The coordinate trace, in any units.

    Returns
    -------
    npt.NDArray[np.float64]
        The discrete angle difference per point in radians.
    """
    angles = np.zeros(trace.shape[0])
    for index, point in enumerate(trace):
        if index == 0:
            v1 = point - trace[-1]
            v2 = trace[index + 1] - point
        elif index == trace.shape[0] - 1:
            v1 = point - trace[index - 1]
            v2 = trace[0] - point
        else:
            v1 = point - trace[index - 1]
            v2 = trace[index + 1] - point

        # Normalise vectors to unit length
        norm_v1 = v1 / np.linalg.norm(v1)
        norm_v2 = v2 / np.linalg.norm(v2)

        # Calculate the signed angle difference between the previous direction and the current direction
        angle = angle_diff_signed(norm_v1, norm_v2)
        angles[index] = angle

    return angles


def discrete_angle_difference_per_nm_circular(
    trace_nm: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Calculate the discrete angle difference per nm along a trace.

    Parameters
    ----------
    trace_nm : npt.NDArray[np.float64]
        The coordinate trace, in nanometre units.

    Returns
    -------
    npt.NDArray[np.float64]
        The discrete angle difference per nm.
    """
    angles_per_nm = np.zeros(trace_nm.shape[0])
    for index, point in enumerate(trace_nm):
        if index == 0:
            v1 = point - trace_nm[-1]
            v2 = trace_nm[index + 1] - point
        elif index == trace_nm.shape[0] - 1:
            v1 = point - trace_nm[index - 1]
            v2 = trace_nm[0] - point
        else:
            v1 = point - trace_nm[index - 1]
            v2 = trace_nm[index + 1] - point

        # Normalise vectors to unit length
        norm_v1 = v1 / np.linalg.norm(v1)
        norm_v2 = v2 / np.linalg.norm(v2)

        # Calculate the signed angle difference between the previous direction and the current direction
        angle = angle_diff_signed(norm_v1, norm_v2)

        # Calculate distance travelled between previous point and the current point
        distance = np.linalg.norm(v1)

        # Calculate the angle difference per nm
        angles_per_nm[index] = angle / distance

    return angles_per_nm


def discrete_angle_difference_per_nm_linear(
    trace_nm: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Calculate the discrete angle difference per nm along a trace.

    Parameters
    ----------
    trace_nm : npt.NDArray[np.float64]
        The coordinate trace, in nanometre units.

    Returns
    -------
    npt.NDArray[np.float64]
        The discrete angle difference per nm.
    """
    angles_per_nm = np.zeros(trace_nm.shape[0])
    for index, point in enumerate(trace_nm):
        if index == 0:
            # No previous point so cannot calculate angle
            v2 = trace_nm[index + 1] - point
            angle = 0.0
            distance = np.linalg.norm(v2)
        elif index == trace_nm.shape[0] - 1:
            # No next point so cannot calculate angle
            v1 = point - trace_nm[index - 1]
            angle = 0.0
            distance = np.linalg.norm(v1)
        else:
            v1 = point - trace_nm[index - 1]
            v2 = trace_nm[index + 1] - point

            # Normalise vectors to unit length
            norm_v1 = v1 / np.linalg.norm(v1)
            norm_v2 = v2 / np.linalg.norm(v2)

            # Calculate the signed angle difference between the previous direction and the current direction
            angle = angle_diff_signed(norm_v1, norm_v2)

            # Calculate distance travelled between previous point and the current point
            distance = np.linalg.norm(v1)

        angles_per_nm[index] = angle / distance

    return angles_per_nm


def smooth_curvature(
    curvatures: npt.NDArray[np.float64],
    point_spacing_nm: np.float64,
    method: Literal["gaussian", "savitzky_golay"],
    gaussian_sigma_nm: float,
    savgol_window_length_nm: int,
    savgol_polyorder: int,
) -> npt.NDArray[np.float64]:
    """
    Smooth the curvature values of a trace.

    Parameters
    ----------
    curvatures : npt.NDArray[np.float64]
        An array of shape (N,) containing the curvature values at each point along the trace.
    point_spacing_nm : float
        The spacing between points along the trace in nanometres - this needs to be pretty consistent.
    method : Literal["gaussian", "savitzky_golay"]
        The method to use for smoothing the curvature values. Options are "gaussian" for a Gaussian filter or
        "savitzky_golay" for a Savitzky-Golay filter.
    gaussian_sigma_nm : float
        The standard deviation of the Gaussian kernel in nanometres.
    savgol_window_length_nm : int
        The length of the filter window in nanometres.
    savgol_polyorder : int
        The order of the polynomial used to fit the samples in the savgol filter.

    Returns
    -------
    npt.NDArray[np.float64]
        An array of shape (N,) containing the smoothed curvature values at each point along the trace.
    """
    if method == "gaussian":
        # adjust the sigma for the gaussian filter
        gaussian_sigma_adjusted = gaussian_sigma_nm / point_spacing_nm
        smoothed_curvatures = gaussian_filter1d(curvatures.copy(), sigma=gaussian_sigma_adjusted)
    elif method == "savitzky_golay":
        # adjust the window length for the savgol filter based on the point spacing
        savgol_window_length_points = int(savgol_window_length_nm / point_spacing_nm)
        assert savgol_window_length_points < len(
            curvatures
        ), "savgol_window_length must be less than the length of the curvature array"
        smoothed_curvatures = savgol_filter(
            curvatures.copy(), window_length=savgol_window_length_points, polyorder=savgol_polyorder
        )
    else:
        raise ValueError(f"Invalid smoothing method: {method}")
    return smoothed_curvatures


def total_turn_in_region_radians(  # noqa: C901
    angles_radians: npt.NDArray[np.float64],
    region_inclusive: tuple[int, int],
    circular: bool = False,
) -> tuple[float, float]:
    """
    Calculate the total turn in radians for a linear trace in a specified region.

    Parameters
    ----------
    angles_radians : npt.NDArray[np.float64]
        The discrete angle differences per point in radians.
    region_inclusive : tuple[int, int]
        The region of the trace to calculate the total turn for, specified as a tuple of two integers (start, end),
        where both indices are inclusive.
    circular : bool, optional
        If True, the trace is considered circular, meaning the first and last points are connected.

    Returns
    -------
    tuple[float, float]
        The total turn in radians for the specified region.

    Raises
    ------
    ValueError
        If the region is not a tuple of two integers or if the indices are out of bounds for the trace.
    """
    if len(region_inclusive) != 2:
        raise ValueError("Region must be a tuple of two integers (start, end).")
    if region_inclusive[0] < 0 or region_inclusive[1] >= angles_radians.shape[0]:
        raise ValueError("Region indices must be within the bounds of the trace.")

    total_left_turn = 0.0
    total_right_turn = 0.0
    if region_inclusive[0] > region_inclusive[1]:
        # The start of the region is after the end, so if the trace is circular, then we wrap around.
        if circular:
            # Grab the angles from the points to the end of the region and then to the start of the region
            for _, angle in enumerate(angles_radians[region_inclusive[0] :]):
                if angle < 0:
                    total_left_turn += abs(angle)
                else:
                    total_right_turn += abs(angle)
            for angle in angles_radians[: region_inclusive[1] + 1]:
                if angle < 0:
                    total_left_turn += abs(angle)
                else:
                    total_right_turn += abs(angle)
        else:
            raise ValueError("Region start must be less than region end for non-circular traces.")
    else:
        # The start of the region is before or the same as the end, so we can just sum the angles in the region
        for angle in angles_radians[region_inclusive[0] : region_inclusive[1] + 1]:
            if angle < 0:
                total_left_turn += abs(angle)
            else:
                total_right_turn += abs(angle)
    return float(total_left_turn), float(total_right_turn)


def quantify_turns(
    curvatures: npt.NDArray[np.float64],
    trace_distances_nm: npt.NDArray[np.float64],
    curvature_turn_threshold_iqr_multiplier: float,
    curvature_turn_minimum_delay_nm: float,
) -> int:
    """
    Calculate the number of turns in trace from its curvatures.

    Parameters
    ----------
    curvatures : npt.NDArray[np.float64]
        Array of curvatures for each point in the trace.
    trace_distances_nm : npt.NDArray[np.float64]
        The distances from each point to the last point in nanometres.
    curvature_turn_threshold_iqr_multiplier : float
        The multiplier to apply to the interquartile range for creating a threshold for identifying turns.
    curvature_turn_minimum_delay_nm : float
        The minimum distance in nanometres before considering if a turn has ended or a new turn has began.

    Returns
    -------
    int
        Number of turns in the trace.
    """
    # Calculate the threshold for a turn based on the nth percentile
    curvature_iqr = np.float64(np.percentile(curvatures, 75) - np.percentile(curvatures, 25))
    curvature_pos_turn_threshold_pernm = np.median(curvatures) + curvature_turn_threshold_iqr_multiplier * curvature_iqr
    curvature_neg_turn_threshold_pernm = np.median(curvatures) - curvature_turn_threshold_iqr_multiplier * curvature_iqr

    # Calculate the number of turns
    turns = 0
    in_pos_turn = False
    in_neg_turn = False
    current_segment_distance = np.inf
    for curvature, distance_to_last_point in zip(curvatures, trace_distances_nm):
        current_segment_distance += distance_to_last_point
        if current_segment_distance > curvature_turn_minimum_delay_nm:
            if curvature >= 0:
                if curvature >= curvature_pos_turn_threshold_pernm:
                    # In positive turn
                    if not in_pos_turn:
                        turns += 1
                        in_pos_turn = True
                        in_neg_turn = False
                        # Start a new segment
                        current_segment_distance = 0
                else:
                    # Not in a turn, reset
                    in_pos_turn = False
                    in_neg_turn = False
                    current_segment_distance = 0
            elif curvature < 0:
                if curvature <= curvature_neg_turn_threshold_pernm:
                    # In negative turn
                    if not in_neg_turn:
                        turns += 1
                        in_neg_turn = True
                        in_pos_turn = False
                        # Start a new segment
                        current_segment_distance = 0
                else:
                    # Not in a turn, reset
                    in_pos_turn = False
                    in_neg_turn = False
                    current_segment_distance = 0

    return turns


def calculate_curvature_metrics(curvatures: npt.NDArray[np.float64], decimals: int = 6) -> dict[str, float]:
    """
    Calculate curvature metrics from an array of curvatures.

    Parameters
    ----------
    curvatures : npt.NDArray[np.float64]
        Array of curvature values.
    decimals : int
        The number of decimal places to round the metrics to.

    Returns
    -------
    dict[str, float]
        Dictionary of curvature metrics.
    """
    return {
        "curvature_mean": float(np.round(np.mean(np.abs(curvatures)), decimals=decimals)),
        "curvature_max": float(np.round(np.max(np.abs(curvatures)), decimals=decimals)),
        "curvature_min": float(np.round(np.min(np.abs(curvatures)), decimals=decimals)),
        "curvature_std": float(np.round(np.std(np.abs(curvatures)), decimals=decimals)),
        "curvature_var": float(np.round(np.var(np.abs(curvatures)), decimals=decimals)),
        "curvature_total": float(np.round(np.sum(np.abs(curvatures)), decimals=decimals)),
        "curvature_median": float(np.round(np.median(np.abs(curvatures)), decimals=decimals)),
        "curvature_iqr": float(
            np.round(np.percentile(np.abs(curvatures), 75) - np.percentile(curvatures, 25), decimals=decimals)
        ),
        "curvature_90th": float(np.round(np.percentile(np.abs(curvatures), 90), decimals=decimals)),
    }


def construct_symmetric_window_of_length(
    distances_to_previous_points: npt.NDArray[np.float64],
    midpoint_index: int,
    window_length: float,
    circular: bool,
) -> tuple[int, int]:
    """Construct a symmetric window around a midpoint index where the len of the window is defined by a given length."""
    left_index = 0
    right_index = 0
    cumulative_distance_left = 0.0
    cumulative_distance_right = 0.0
    half_window_length = window_length / 2
    # can get the reversed direction distances to previous points by just rolling the array by 1 in the negative
    # direction
    reversed_distances_to_previous_points = np.roll(distances_to_previous_points, -1)

    # guard for the length of the window being longer than the total length of the trace
    if window_length >= np.sum(distances_to_previous_points):
        raise ValueError("window length exceeds total length of the trace, cannot construct window")

    # Find the right index by moving forwards until the cumulative distance exceeds the window length, wrapping around
    # if necessary and circular
    for index in range(1, len(distances_to_previous_points) + 1):
        prospective_right_index = midpoint_index + index
        if prospective_right_index >= len(distances_to_previous_points):
            if circular:
                prospective_right_index = prospective_right_index % len(distances_to_previous_points)
            else:
                raise ValueError("required right window bound exceeds the end of the linear trace")
        cumulative_distance_right += distances_to_previous_points[prospective_right_index]
        if cumulative_distance_right > half_window_length:
            right_index = prospective_right_index
            break
    # Find the left index by moving backwards until the cumulative distance exceeds the window length, wrapping around
    # if necessary and circular
    for index in range(1, len(distances_to_previous_points) + 1):
        prospective_left_index = midpoint_index - index
        if prospective_left_index < 0:
            if circular:
                prospective_left_index = prospective_left_index % len(distances_to_previous_points)
            else:
                raise ValueError("required left window bound exceeds the start of the linear trace")
        cumulative_distance_left += reversed_distances_to_previous_points[prospective_left_index]
        if cumulative_distance_left > half_window_length:
            left_index = prospective_left_index
            break
    return left_index, right_index


def calculate_turn_in_distance_for_trace_deg(
    trace_coords_nm: npt.NDArray[np.float64],
    turn_in_distance_window_length_nm: float,
    circular: bool,
    turn_in_distance_window_end_sampling_points: int,
) -> npt.NDArray[np.float64]:
    """Calculate the degrees turned over a given window length for each point in a trace."""
    assert turn_in_distance_window_end_sampling_points >= 2, "need at least 2 points to calculate a vector"
    turn_in_distances_deg = np.zeros(len(trace_coords_nm))
    distances_to_previous_points_nm = distances_nm(trace_coords_nm, circular)

    # for each point, construct a symmetric window around it with the given window length
    for index in range(len(trace_coords_nm)):
        try:
            left_index, right_index = construct_symmetric_window_of_length(
                distances_to_previous_points=distances_to_previous_points_nm,
                midpoint_index=index,
                window_length=turn_in_distance_window_length_nm,
                circular=circular,
            )
        # check for the error where the edge of the window meets the end of the trace in linear traces
        # if this happens, set the value to nan and continue
        except ValueError as e:
            if "required right window bound exceeds the end of the linear trace" in str(
                e
            ) or "required left window bound exceeds the start of the linear trace" in str(e):
                turn_in_distances_deg[index] = np.nan
                continue
            raise e

        # To calculate vectors, we will need to grab another point on either side. Grab the inner points
        # so, for the left, get the next point, and for the right, get the previous point.
        # will need to check for windows with points that have too few points to calculate the vectors
        window_num_points = (
            right_index - left_index + 1
            if right_index >= left_index
            else left_index + (len(trace_coords_nm) - right_index) + 1
        )
        if window_num_points < turn_in_distance_window_end_sampling_points * 2:
            raise ValueError(
                f"window of {window_num_points} points contains too few points to sample the required number of points"
                f" {turn_in_distance_window_end_sampling_points} from each side to calculate the turn in distance"
            )

        # get the indices of the points to sample, wrapping where needed
        window_left_points_indices = []
        window_right_points_indices = []
        for sample_index in range(turn_in_distance_window_end_sampling_points):
            left_sample_index = left_index + sample_index
            if left_sample_index >= len(trace_coords_nm):
                if circular:
                    left_sample_index = left_sample_index % len(trace_coords_nm)
                else:
                    raise ValueError(
                        "required left sample point index exceeds the end of the linear trace, cannot calculate turn in distance"
                    )
            window_left_points_indices.append(left_sample_index)
            right_sample_index = right_index - sample_index
            if right_sample_index < 0:
                if circular:
                    right_sample_index = right_sample_index % len(trace_coords_nm)
                else:
                    raise ValueError(
                        "required right sample point index exceeds the start of the linear trace, cannot calculate turn in distance"
                    )
            window_right_points_indices.append(right_sample_index)

        # calculate the vectors by taking the differences between each sequential point pair and then averaging them
        left_vectors = np.diff(trace_coords_nm[window_left_points_indices], axis=0)
        # flip the right vectors since we are going in the opposite direction and we want the vectors to point
        # in the same direction for angle calculation
        window_right_points_indices_flipped = list(reversed(window_right_points_indices))
        right_vectors = np.diff(trace_coords_nm[window_right_points_indices_flipped], axis=0)
        left_vector = np.mean(left_vectors, axis=0)
        right_vector = np.mean(right_vectors, axis=0)
        # calculate angle difference between the two vectors
        turn_rad = angle_diff_signed(v1=left_vector, v2=right_vector)
        turn_deg = np.degrees(turn_rad)
        turn_in_distances_deg[index] = turn_deg

    return turn_in_distances_deg


# pylint: disable=too-many-locals
def calculate_curvature_stats_image(
    topostats_object: TopoStats,
    smoothing_method: Literal["gaussian", "savitzky_golay"],
    smoothing_gaussian_sigma_nm: float,
    smoothing_savgol_window_length_nm: int,
    smoothing_savgol_polyorder: int,
    curvature_turn_minimum_delay_nm: float,
    curvature_turn_threshold_iqr_multiplier: float,
    turn_in_distance_window_length_nm: float,
    turn_in_distance_end_sampling_points: int,
) -> None:
    """
    Perform curvature analysis for a whole image of grains.

    Curvature statistics are added to the ``Molecule.curvature_stats`` attribute of the traces that are being processed.

    Parameters
    ----------
    topostats_object : TopoStats
        ``TopoStats`` object with attribute ``grain_crop``. Should be post-splining.
    smoothing_method : Literal["gaussian", "savitzky_golay"],
        The method to use for smoothing.
    smoothing_gaussian_sigma_nm : float
        The gaussian sigma to use for smoothing.
    smoothing_savgol_window_length_nm : int
        The window length to use for the savgol window (nanometres).
    smoothing_savgol_polyorder : int
        The polynomial order to use for the savgol filter.
    curvature_turn_minimum_delay_nm : float
        The minimum distance in nanometres before considering if a turn has ended or a new turn has began.
    curvature_turn_threshold_iqr_multiplier : float
        The multiplier to apply to the interquartile range for creating a threshold for identifying turns.
    """
    # Iterate over grains
    for _, grain_crop in topostats_object.require_grain_crops().items():
        # Iterate over molecules
        if grain_crop.ordered_trace is not None and grain_crop.ordered_trace.molecule_data is not None:
            for _, molecule_data in grain_crop.ordered_trace.molecule_data.items():
                # Calculate curvature stats per molecule
                trace_nm = molecule_data.require_splined_coords() * topostats_object.require_pixel_to_nm_scaling()
                is_circular = molecule_data.circular
                assert isinstance(is_circular, bool)

                curvatures = (
                    discrete_angle_difference_per_nm_circular(trace_nm)
                    if is_circular
                    else discrete_angle_difference_per_nm_linear(trace_nm)
                ).astype(np.float64)

                # Smooth the curvatures
                trace_distances_nm = distances_nm(coords_nm=trace_nm, circular=is_circular)
                avg_trace_distance_nm = np.mean(trace_distances_nm)
                curvatures_smoothed = smooth_curvature(
                    curvatures=curvatures,
                    point_spacing_nm=avg_trace_distance_nm,
                    method=smoothing_method,
                    gaussian_sigma_nm=smoothing_gaussian_sigma_nm,
                    savgol_window_length_nm=smoothing_savgol_window_length_nm,
                    savgol_polyorder=smoothing_savgol_polyorder,
                )

                curvature_metrics = calculate_curvature_metrics(curvatures_smoothed)

                num_turns = quantify_turns(
                    curvatures=curvatures_smoothed,
                    trace_distances_nm=trace_distances_nm,
                    curvature_turn_minimum_delay_nm=curvature_turn_minimum_delay_nm,
                    curvature_turn_threshold_iqr_multiplier=curvature_turn_threshold_iqr_multiplier,
                )

                turn_in_distances_deg = calculate_turn_in_distance_for_trace_deg(
                    trace_coords_nm=trace_nm,
                    turn_in_distance_window_length_nm=turn_in_distance_window_length_nm,
                    turn_in_distance_window_end_sampling_points=turn_in_distance_end_sampling_points,
                    circular=is_circular,
                )

                molecule_data.curvature_stats = MoleculeCurvatureStats(
                    curvatures=curvatures_smoothed,
                    turn_in_distances_deg=turn_in_distances_deg,
                    turn_in_distance_window_length_nm=turn_in_distance_window_length_nm,
                    turn_in_distance_end_sampling_points=turn_in_distance_end_sampling_points,
                    is_circular=is_circular,
                    num_turns=num_turns,
                    **curvature_metrics,
                )

            # Calculate curvature stats for the whole grain and update
            curvatures_all_grain = np.concatenate(
                [
                    molecule_data.require_curvature_stats().curvatures
                    for molecule_data in grain_crop.ordered_trace.molecule_data.values()
                ]
            )
            curvature_metrics_grain = calculate_curvature_metrics(curvatures_all_grain)
            num_turns_total_grain = sum(
                molecule_data.require_curvature_stats().num_turns
                for molecule_data in grain_crop.ordered_trace.molecule_data.values()
            )
            grain_crop.ordered_trace.grain_curvature_stats = GrainCurvatureStats(
                num_turns=num_turns_total_grain,
                **curvature_metrics_grain,
            )
            # Add the stats raw to the grain crop stats dictionary to be saved to csv
            for class_number, stats in grain_crop.stats.items():
                for subgrain_index, _ in stats.items():
                    grain_crop.stats[class_number][subgrain_index]["curvature_grain_num_turns"] = np.int64(
                        num_turns_total_grain
                    )
                    grain_crop.stats[class_number][subgrain_index].update(curvature_metrics_grain)
