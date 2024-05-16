"""Derive height profiles across the images."""

from __future__ import annotations

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


def interpolate_height_profile(img: npt.NDArray, skeleton: npt.NDArray, **kwargs) -> npt.NDArray:
    """
    Interpolate heights along the maximum feret.

    Interpolates the height along the line of the maximum feret using SciPy `scipy.interpolateRegularGridInterpolator
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html`. Arguments can
    be passed using 'kwargs'.

    Parameters
    ----------
    img : npt.NDArray
        Original image with heights.
    skeleton : npt.NDArray
        Binary skeleton.
    **kwargs : dict
        Keyword arguments passed on to scipy.interpolate.RegularGridInterpolator().

    Returns
    -------
    npt.NDArray
        Interpolated heights between the calculated feret co-cordinates.
    """
    # Extract feret coordinates
    feret_stats = feret.get_feret_from_mask(skeleton)
    x_coords = np.sort(feret_stats["max_feret_coords"][:, 0])
    y_coords = np.sort(feret_stats["max_feret_coords"][:, 1])
    # Evenly spaced points on x-axis
    x_points = np.linspace(np.min(x_coords), np.max(x_coords), np.max(x_coords) - np.min(x_coords) + 1)
    # Corresponding y values for each x
    y_points = np.interp(x_points, x_coords, y_coords)
    # Combine into array
    xy_points = np.vstack((x_points, y_points)).T
    # Interpolate
    interp = interpolate.RegularGridInterpolator(
        points=(np.arange(img.shape[0]), np.arange(img.shape[1])), values=img, **kwargs
    )
    return interp(xy_points)
