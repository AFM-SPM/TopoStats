"""Calculate various curvature metrics for traces."""

import logging
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

from topostats.classes import TopoStatsBaseModel
from topostats.damage.array_manipulation import distances_nm
from topostats.logs.logs import LOGGER_NAME

LOGGER = logging.getLogger(LOGGER_NAME)


def angle_diff_signed(v1: npt.NDArray[np.number], v2: npt.NDArray[np.number]):
    """
    Calculate the signed angle difference between two point vecrtors in 2D space.

    Positive angles are clockwise, negative angles are counterclockwise.

    Parameters
    ----------
    v1 : npt.NDArray[np.number]
        First vector.
    v2 : npt.NDArray[np.number]
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
    return total_left_turn, total_right_turn


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
    trace_nm: npt.NDArray[np.number],
) -> npt.NDArray[np.number]:
    """
    Calculate the discrete angle difference per nm along a trace.

    Parameters
    ----------
    trace_nm : npt.NDArray[np.number]
        The coordinate trace, in nanometre units.

    Returns
    -------
    npt.NDArray[np.number]
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
    trace_nm: npt.NDArray[np.number],
) -> npt.NDArray[np.number]:
    """
    Calculate the discrete angle difference per nm along a trace.

    Parameters
    ----------
    trace_nm : npt.NDArray[np.number]
        The coordinate trace, in nanometre units.

    Returns
    -------
    npt.NDArray[np.number]
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


class MoleculeCurvatureStats(TopoStatsBaseModel):
    """Data model for storing curvature statistics for a single molecule."""

    curvatures: npt.NDArray[np.float64]
    is_circular: bool
    num_turns: int
    curvature_mean: float
    curvature_max: float
    curvature_min: float
    curvature_std: float
    curvature_var: float
    curvature_total: float
    curvature_median: float
    curvature_iqr: float
    curvature_90th: float

    # function to turn to a dictionary if needed
    def to_dict(self) -> dict:
        """
        Convert the MoleculeCurvatureStats to a dictionary.

        Returns
        -------
        dict
            Dictionary representation of the MoleculeCurvatureStats.
        """
        return {
            "curvatures": self.curvatures,
            "is_circular": self.is_circular,
            "curvature_mean": self.curvature_mean,
            "curvature_max": self.curvature_max,
            "curvature_min": self.curvature_min,
            "curvature_std": self.curvature_std,
            "curvature_var": self.curvature_var,
            "curvature_total": self.curvature_total,
            "curvature_median": self.curvature_median,
            "curvature_iqr": self.curvature_iqr,
            "curvature_90th": self.curvature_90th,
            "num_turns": self.num_turns,
        }


class GrainCurvatureStats(TopoStatsBaseModel):
    """Data model for storing curvature statistics for a single grain."""

    molecules: dict[str, MoleculeCurvatureStats]
    num_turns: int
    curvature_mean: float
    curvature_max: float
    curvature_min: float
    curvature_std: float
    curvature_var: float
    curvature_total: float
    curvature_median: float
    curvature_iqr: float
    curvature_90th: float

    def to_dict(self) -> dict:
        """
        Convert the GrainCurvatureStats to a dictionary.

        Returns
        -------
        dict
            Dictionary representation of the GrainCurvatureStats.
        """
        return {
            "molecules": {key: value.to_dict() for key, value in self.molecules.items()},
            "curvature_mean": self.curvature_mean,
            "curvature_max": self.curvature_max,
            "curvature_min": self.curvature_min,
            "curvature_std": self.curvature_std,
            "curvature_var": self.curvature_var,
            "curvature_total": self.curvature_total,
            "curvature_median": self.curvature_median,
            "curvature_iqr": self.curvature_iqr,
            "curvature_90th": self.curvature_90th,
            "num_turns": self.num_turns,
        }


class AllGrainCurvatureStats(TopoStatsBaseModel):
    """Data model for storing curvature statistics for all grains."""

    grains: dict[str, GrainCurvatureStats]
    filename: str

    def to_dict(self) -> dict:
        """
        Convert the AllGrainCurvatureStats to a dictionary.

        Returns
        -------
        dict
            Dictionary representation of the AllGrainCurvatureStats.
        """
        return {
            "grains": {key: value.to_dict() for key, value in self.grains.items()},
            "filename": self.filename,
        }

    def create_grain_curvature_stats_dataframe(self) -> pd.DataFrame:
        """
        Create a dataframe of grain curvature statistics.

        Returns
        -------
        pd.DataFrame
            Dataframe of grain curvature statistics.
        """
        # Format: grain_index | <metric>
        records = []
        for grain_index, grain_curvature_stats in self.grains.items():
            entry = {
                "image": self.filename,
                "grain_number": int(grain_index.split("_")[-1]),
                "curvature_mean": grain_curvature_stats.curvature_mean,
                "curvature_max": grain_curvature_stats.curvature_max,
                "curvature_min": grain_curvature_stats.curvature_min,
                "curvature_std": grain_curvature_stats.curvature_std,
                "curvature_var": grain_curvature_stats.curvature_var,
                "curvature_total": grain_curvature_stats.curvature_total,
                "curvature_median": grain_curvature_stats.curvature_median,
                "curvature_iqr": grain_curvature_stats.curvature_iqr,
                "curvature_90th": grain_curvature_stats.curvature_90th,
                "num_turns": grain_curvature_stats.num_turns,
            }
            records.append(entry)
        return pd.DataFrame.from_records(records)


def _calculate_curvature_metrics(curvatures: npt.NDArray[np.float64]) -> dict[str, float]:
    """
    Calculate curvature metrics from an array of curvatures.

    Parameters
    ----------
    curvatures : npt.NDArray[np.float64]
        Array of curvature values.

    Returns
    -------
    dict[str, float]
        Dictionary of curvature metrics.
    """
    return {
        "curvature_mean": float(np.mean(np.abs(curvatures))),
        "curvature_max": float(np.max(np.abs(curvatures))),
        "curvature_min": float(np.min(np.abs(curvatures))),
        "curvature_std": float(np.std(np.abs(curvatures))),
        "curvature_var": float(np.var(np.abs(curvatures))),
        "curvature_total": float(np.sum(np.abs(curvatures))),
        "curvature_median": float(np.median(np.abs(curvatures))),
        "curvature_iqr": float(np.percentile(np.abs(curvatures), 75) - np.percentile(curvatures, 25)),
        "curvature_90th": float(np.percentile(np.abs(curvatures), 90)),
    }


def calculate_curvature_stats_image(
    filename: str,
    all_grain_smoothed_data: dict,
    pixel_to_nm_scaling: float,
    smoothing_method: Literal["gaussian", "savitzky_golay"],
    smoothing_gaussian_sigma_nm: float,
    smoothing_savgol_window_length_nm: int,
    smoothing_savgol_polyorder: int,
    curvature_turn_minimum_delay_nm: float,
    curvature_turn_threshold_iqr_multiplier: float,
) -> tuple[AllGrainCurvatureStats, pd.DataFrame]:
    """
    Perform curvature analysis for a whole image of grains.

    Parameters
    ----------
    filename : str
        Filename of the image.
    all_grain_smoothed_data : dict
        Dictionary containing grain traces in pixel units.
    pixel_to_nm_scaling : float
        Pixel to nm scaling factor.

    Returns
    -------
    tuple[AllGrainCurvatureStats, pd.DataFrame]
        All grain curvature statistics and dataframe of grain curvature statistics.
    """
    grains: dict[str, GrainCurvatureStats] = {}

    # Iterate over grains
    for grain_key, grain_data in all_grain_smoothed_data.items():
        # Iterate over molecules
        molecules: dict[str, MoleculeCurvatureStats] = {}
        for molecule_key, molecule_data in grain_data.items():
            trace_nm = molecule_data["spline_coords"] * pixel_to_nm_scaling
            is_circular = molecule_data["tracing_stats"]["end_to_end_distance"] == 0.0

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

            metrics = _calculate_curvature_metrics(curvatures_smoothed)
            num_turns = quantify_turns(
                curvatures=curvatures_smoothed,
                trace_distances_nm=trace_distances_nm,
                curvature_turn_minimum_delay_nm=curvature_turn_minimum_delay_nm,
                curvature_turn_threshold_iqr_multiplier=curvature_turn_threshold_iqr_multiplier,
            )
            molecules[molecule_key] = MoleculeCurvatureStats(
                curvatures=curvatures_smoothed,
                is_circular=is_circular,
                num_turns=num_turns,
                **metrics,
            )

        # Collate stats
        all_curvatures = np.concatenate([molecule.curvatures for molecule in molecules.values()])
        grain_metrics = _calculate_curvature_metrics(all_curvatures)
        grain_num_turns = sum(molecule.num_turns for molecule in molecules.values())
        grains[grain_key] = GrainCurvatureStats(
            molecules=molecules,
            num_turns=grain_num_turns,
            **grain_metrics,
        )

        all_grain_curvature_stats = AllGrainCurvatureStats(filename=filename, grains=grains)
        all_grain_curvature_stats_df = all_grain_curvature_stats.create_grain_curvature_stats_dataframe()

    return all_grain_curvature_stats, all_grain_curvature_stats_df


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
        assert savgol_window_length_points < len(curvatures), (
            "savgol_window_length must be less than the length of the curvature array"
        )
        smoothed_curvatures = savgol_filter(
            curvatures.copy(), window_length=savgol_window_length_points, polyorder=savgol_polyorder
        )
    else:
        raise ValueError(f"Invalid smoothing method: {method}")
    return smoothed_curvatures
