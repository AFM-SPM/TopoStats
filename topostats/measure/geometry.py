"""Functions for measuring geometric properties of grains."""

import numpy as np
from numpy.typing import NDArray


def bounding_box_cartesian_points(points: NDArray[np.number]) -> tuple[np.float64, np.float64, np.float64, np.float64]:
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
