"""Calculate various curvature metrics for traces."""

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
                grain_curvature_stats[grain_key][molecule_key] = np.abs(
                    discrete_angle_difference_per_nm_circular(trace_nm)
                )
            else:
                # Molecule is linear
                grain_curvature_stats[grain_key][molecule_key] = np.abs(
                    discrete_angle_difference_per_nm_linear(trace_nm)
                )

    return grain_curvature_stats
