"""Calculate various curvature metrics for traces."""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt

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


# def find_curvature_defects_simple_threshold(
#     curvature_angle_per_nm: npt.NDArray[np.number],
#     defect_threshold: float,
# ) -> npt.NDArray[np.bool_]:
#     """
#     Find defects in the curvature of a trace.

#     Parameters
#     ----------
#     curvature_angle_per_nm : npt.NDArray[np.number]
#         The curvature angle per nm.
#     defect_threshold : float
#         The threshold for the curvature defect.

#     Returns
#     -------
#     npt.NDArray[np.bool_]
#         The boolean array indicating the defects.
#     """
#     return np.abs(curvature_angle_per_nm) >= defect_threshold


# def calculate_trace_distances_to_last_points_circular(
#     trace_nm: npt.NDArray[np.number],
# ) -> npt.NDArray[np.number]:
#     """
#     Calculate the distances between each point in the trace and the preceding point.

#     Parameters
#     ----------
#     trace_nm : npt.NDArray[np.number]
#         The coordinate trace, in nanometre units.

#     Returns
#     -------
#     npt.NDArray[np.number]
#         The distances between each point in the trace and the preceding point.
#     """
#     if trace_nm.shape[1] != 2:
#         raise ValueError("Trace must be an array of shape (n, 2)")

#     distances_to_last_points = np.zeros(trace_nm.shape[0])

#     for index, point in enumerate(trace_nm):
#         if index == 0:
#             distances_to_last_points[index] = np.linalg.norm(point - trace_nm[-1])
#         else:
#             distances_to_last_points[index] = np.linalg.norm(point - trace_nm[index - 1])

#     return distances_to_last_points


# pylint: disable=too-many-branches
# def calculate_distances_between_defects_circular(
#     curvature_defects: npt.NDArray[np.bool_],
#     trace_distances_to_last_points: npt.NDArray[np.number],
# ) -> npt.NDArray[np.float32]:
#     """
#     Calculate the real distance along the trace between defects.

#     Parameters
#     ----------
#     curvature_defects : npt.NDArray[np.bool_]
#         The boolean array indicating the defects.
#     trace_distances_to_last_points : npt.NDArray[np.number]
#         The distances between each point in the trace and the preceding point.

#     Returns
#     -------
#     npt.NDArray[np.float32]
#         The distances between defects. Eg: [5.2, 3.1].
#     """
#     defect_gap_distances: list[float] = []

#     # State variables
#     in_defect: bool = False
#     current_gap_distance: float | None = None
#     first_gap_distance: float = 0.0

#     # Iterate over the boolean defects array
#     for index, is_defect in enumerate(curvature_defects):
#         if is_defect:
#             if not in_defect:
#                 # Start of new defect
#                 in_defect = True
#                 # If this is the end of the first gap, then don't add it to the defect gaps
#                 # else if this is the end of a standard gap, add it to the defect gaps
#                 if current_gap_distance is not None:
#                     # Add gap distance to defect distances, including the distance of the current (defect) point to
#                     # the previous (non-defect) point
#                     defect_gap_distances.append(current_gap_distance + trace_distances_to_last_points[index])
#                     current_gap_distance = 0.0
#             else:
#                 # Continue defect
#                 pass
#         else:
#             if in_defect:
#                 # End of defect
#                 in_defect = False
#                 current_gap_distance = trace_distances_to_last_points[index]
#             else:
#                 # Continue gap
#                 # If not encountered a defect yet, ignore, since this is the first gap
#                 if current_gap_distance is not None:
#                     # If this is not the first gap, add the distance to the current gap
#                     current_gap_distance += trace_distances_to_last_points[index]
#                 else:
#                     # If this is the first gap, add the distance to the first gap and save it for later
#                     first_gap_distance += trace_distances_to_last_points[index]

#     # If the last point is a defect, then the last gap distance is simply the distance to the first point
#     # from the start of the trace
#     if current_gap_distance is not None:
#         if in_defect:
#             # Check that the current + first gap distance doesn't equal 0.0, since we don't want to add a 0.0 distance
#             if current_gap_distance + first_gap_distance != 0.0:
#                 defect_gap_distances.append(current_gap_distance + first_gap_distance)
#             else:
#                 pass
#         # Else we need to add the distance from the last defect-the end of the trace to the distance from the
#         # start of the trace to the first defect, which will be the first gap distance plus the current gap distance
#         else:
#             defect_gap_distances.append(first_gap_distance + current_gap_distance)
#     else:
#         # No defect was encountered at all so nothing to add
#         pass

#     return np.array(defect_gap_distances).astype(np.float32)


# def calculate_number_of_defects(curvature_defects: npt.NDArray[np.bool_], circular: bool) -> int:
#     """
#     Calculate the number of continuous defects in the curvature of a trace.

#     Parameters
#     ----------
#     curvature_defects : npt.NDArray[np.bool_]
#         The boolean array indicating the defects.
#     circular : bool
#         Whether the trace is circular or not.

#     Returns
#     -------
#     int
#         The number of defects.
#     """
#     # Find the number of continuous 1 regions in the binary curvature defects array
#     labelled_defects = label(curvature_defects, connectivity=1, background=0)
#     defect_count = np.max(labelled_defects)

#     if circular:
#         if curvature_defects[0] and curvature_defects[-1]:
#             # If the first and last points are defects, then the last defect is the same as the first defect
#             # unless there is only one defect, ie the whole trace is a defect.
#             if defect_count > 1:
#                 defect_count -= 1

#     return defect_count


# def calculate_curvature_stats_single_grain_circular(
#     grain_trace_nm: npt.NDArray[np.number],
#     defect_threshold: float,
# ) -> dict:
#     """
#     Calculate curvature statistics for a single grain.

#     Parameters
#     ----------
#     grain_trace_nm : npt.NDArray[np.number]
#         The grain trace in nanometres.
#     defect_threshold : float
#         The threshold for the curvature defect.

#     Returns
#     -------
#     dict
#         The curvature statistics. Indexes are grain indexes.
#     """
#     # Calculate curvature
#     curvature_angle_per_nm = discrete_angle_difference_per_nm_circular(grain_trace_nm)

#     # Find defects
#     curvature_defects_binary_array = find_curvature_defects_simple_threshold(
#         curvature_angle_per_nm=curvature_angle_per_nm, defect_threshold=defect_threshold
#     )

#     # Calculate distances between defects
#     trace_distances_to_last_points = calculate_trace_distances_to_last_points_circular(grain_trace_nm)
#     defect_gap_distances = calculate_distances_between_defects_circular(
#         curvature_defects_binary_array, trace_distances_to_last_points
#     )

#     # Find the number of defects
#     number_of_defects = calculate_number_of_defects(curvature_defects_binary_array, circular=True)

#     return {
#         "grain_trace_nm": grain_trace_nm,
#         "grain_curvature": curvature_angle_per_nm,
#         "curvature_defects_binary_array": curvature_defects_binary_array,
#         "defect_gap_distances": defect_gap_distances,
#         "number_of_defects": number_of_defects,
#     }


def calculate_curvature_stats_image(
    all_grain_smoothed_data: dict,
    pixel_to_nm_scaling: float,
) -> dict:
    """
    Perform curvature analysis for a whole image of grains.

    Parameters
    ----------
    all_grain_smoothed_data : dict
        Dictionary containing grain traces in pixel units.
    pixel_to_nm_scaling : float
        Pixel to nm scaling factor.

    Returns
    -------
    dict
        The curvature statistics for each grain. Indexes are grain indexes.
    """
    grain_curvature_stats: dict = {}

    # Iterate over grains
    for grain_key, grain_data in all_grain_smoothed_data.items():
        # Iterate over molecules
        grain_curvature_stats[grain_key] = {}
        for molecule_key, molecule_data in grain_data.items():
            trace_nm = molecule_data["spline_coords"] * pixel_to_nm_scaling
            # Check if the molecule is circular or linear
            if molecule_data["tracing_stats"]["end_to_end_distance"] == 0.0:
                # Molecule is circular
                grain_curvature_stats[grain_key][molecule_key] = discrete_angle_difference_per_nm_circular(trace_nm)
            else:
                # Molecule is linear
                grain_curvature_stats[grain_key][molecule_key] = discrete_angle_difference_per_nm_linear(trace_nm)

    return grain_curvature_stats