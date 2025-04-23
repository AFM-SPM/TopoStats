"""functions for trace smoothing"""

import numpy.typing as npt
import numpy as np
from topostats.io import LoadScans
import matplotlib.pyplot as plt
from topostats.unet_masking import make_bounding_box_square, pad_bounding_box


def interpolate_between_two_points_distance(
    point1: npt.NDArray[np.float32], point2: npt.NDArray[np.float32], distance: np.float32
) -> npt.NDArray[np.float32]:
    """Interpolate between two points to create a new point at a set distance between the two.

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
    new_point = point1 + proportion * (point2 - point1)
    return new_point


def resample_points_regular_interval(points: npt.NDArray, interval: float, circular: bool) -> npt.NDArray:
    """Resample a set of points to be at regular spatial intervals.
    
    Note: This is NOT intended to be pure interpolation, as interpolated points would not produce uniformly spaced points."""

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
        new_interpolated_point = interpolate_between_two_points(current_point, next_original_point, interval)
        resampled_points.append(new_interpolated_point)

    # if the first and last points are less than 0.5 * the interval apart, then remove the last point
    if np.linalg.norm(resampled_points[0] - resampled_points[-1]) < 0.5 * interval:
        resampled_points = resampled_points[:-1]

    resampled_points_numpy = np.array(resampled_points)

    return resampled_points_numpy
