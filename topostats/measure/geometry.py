"""Functions for measuring geometric properties of grains."""

import numpy as np
from numpy.typing import NDArray


def bounding_box_cartesian_points_float(
    points: NDArray[np.number],
) -> tuple[np.float64, np.float64, np.float64, np.float64]:
    """
    Calculate the bounding box from a set of points.

    Parameters
    ----------
    points : NDArray[np.number]
        Nx2 numpy array of points.

    Returns
    -------
    Tuple[np.float64, np.float64, np.float64, np.float64]
        Tuple of (min_x, min_y, max_x, max_y).

    Raises
    ------
    ValueError
        If the input array is not Nx2.
    """
    if points.shape[1] != 2:
        raise ValueError("Input array must be Nx2.")
    x_coords, y_coords = points[:, 0].astype(np.float64), points[:, 1].astype(np.float64)
    return (np.min(x_coords), np.min(y_coords), np.max(x_coords), np.max(y_coords))


def bounding_box_cartesian_points_integer(points: NDArray[np.number]) -> tuple[np.int32, np.int32, np.int32, np.int32]:
    """
    Calculate the bounding box from a set of points.

    Parameters
    ----------
    points : NDArray[np.number]
        Nx2 numpy array of points.

    Returns
    -------
    Tuple[np.int32, np.int32, np.int32, np.int32]
        Tuple of (min_x, min_y, max_x, max_y).

    Raises
    ------
    ValueError
        If the input array is not Nx2.
    """
    if points.shape[1] != 2:
        raise ValueError("Input array must be Nx2.")
    x_coords, y_coords = points[:, 0].astype(np.int32), points[:, 1].astype(np.int32)
    return (np.min(x_coords), np.min(y_coords), np.max(x_coords), np.max(y_coords))


def do_points_in_arrays_touch(
    points1: NDArray[np.number], points2: NDArray[np.number]
) -> tuple[bool, NDArray[np.number] | None, NDArray[np.number] | None]:
    """
    Check if any points in two arrays are touching.

    Parameters
    ----------
    points1 : NDArray[np.number]
        Nx2 numpy array of points.
    points2 : NDArray[np.number]
        Mx2 numpy array of points.

    Returns
    -------
    tuple[bool, NDArray[np.number] | None, NDArray[np.number] | None]
        True if any points in the two arrays are touching, False otherwise, followed by the two points that touch.

    Raises
    ------
    ValueError
        If the input arrays are not Nx2 and Mx2.
    """
    if points1.shape[1] != 2 or points2.shape[1] != 2:
        raise ValueError("Input arrays must be Nx2 and Mx2.")

    for point1 in points1:
        for point2 in points2:
            diff = np.abs(point1 - point2)
            if np.all(diff <= 1):
                return (True, point1, point2)
    return (False, None, None)
