"""Derive height profiles across the images."""

import logging
import warnings

import numpy as np
import numpy.typing as npt
from scipy import interpolate

from topostats.logs.logs import LOGGER_NAME
from topostats.measure import feret

LOGGER = logging.getLogger(LOGGER_NAME)

# Handle warnings as exceptions (encountered when gradient of base triangle is zero)
warnings.filterwarnings("error")


def interpolate_height_profile(img: npt.NDArray, mask: npt.NDArray, **kwargs) -> npt.NDArray:
    """
    Interpolate heights along the maximum feret.

    Interpolates the height along the line of the maximum feret using SciPy `scipy.interpolateRegularGridInterpolator
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html`. Arguments can
    be passed using 'kwargs'.

    Parameters
    ----------
    img : npt.NDArray
        Original image with heights.
    mask : npt.NDArray
        Binary skeleton.
    **kwargs : dict
        Keyword arguments passed on to scipy.interpolate.RegularGridInterpolator().

    Returns
    -------
    npt.NDArray
        Interpolated heights between the calculated feret co-cordinates.
    """
    # Extract feret coordinates
    feret_stats = feret.get_feret_from_mask(mask)
    x_coords = feret_stats["max_feret_coords"][:, 0]
    x_diff = np.abs(x_coords[0] - x_coords[1])
    y_coords = feret_stats["max_feret_coords"][:, 1]
    y_diff = np.abs(y_coords[0] - y_coords[1])
    # Interpolate along the longest axis (maximises detail)
    if x_diff > y_diff:
        # Sort in ascending order, required for correct np.linspace()
        order = np.argsort(x_coords)
        x_coords = x_coords[order]
        y_coords = y_coords[order]
        x_points = np.linspace(np.min(x_coords), np.max(x_coords), np.max(x_coords) - np.min(x_coords) + 1)
        y_points = np.interp(x_points, x_coords, y_coords)
        xy_points = np.vstack((x_points, y_points)).T
    else:
        # Sort in ascending order, required for correct np.linspace()
        order = np.argsort(y_coords)
        x_coords = x_coords[order]
        y_coords = y_coords[order]
        y_points = np.linspace(np.min(y_coords), np.max(y_coords), np.max(y_coords) - np.min(y_coords) + 1)
        x_points = np.interp(y_points, y_coords, x_coords)
        xy_points = np.vstack((x_points, y_points)).T
    # Interpolate heights
    interp = interpolate.RegularGridInterpolator(
        points=(np.arange(img.shape[0]), np.arange(img.shape[1])), values=img, **kwargs
    )
    return interp(xy_points)
