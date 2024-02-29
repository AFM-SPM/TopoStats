"""Functions for calculating thresholds."""

# pylint: disable=no-name-in-module
import logging
from collections.abc import Callable

import numpy as np
from skimage.filters import threshold_mean, threshold_minimum, threshold_otsu, threshold_triangle, threshold_yen

from topostats.logs.logs import LOGGER_NAME

LOGGER = logging.getLogger(LOGGER_NAME)

# pylint: disable=no-else-return
# pylint: disable=unused-argument


def threshold(image: np.ndarray, method: str = None, otsu_threshold_multiplier: float = None, **kwargs: dict) -> float:
    """Thresholding for producing masks.

    Parameters
    ----------
    method : str
        Method to use for thresholding, currently supported methods are otsu (default), mean and minimum.
    **kwargs : dict
        Additional keyword arguments to pass to skimage methods.

    Returns
    -------
    float
        Threshold of image using specified method.
    """
    thresholder = _get_threshold(method)
    return thresholder(image, otsu_threshold_multiplier=otsu_threshold_multiplier, **kwargs)


def _get_threshold(method: str = "otsu") -> Callable:
    """Creator component which determines which threshold method to use.

    Parameters
    ----------
    method : str
        Threshold method to use, currently supports otsu (default), mean, minimum, mean yen, and triangle.

    Returns
    -------
    function
        Returns function appropriate for the required threshold method.

    Raises
    ------
    ValueError
        Unsupported methods result in ValueError.
    """
    if method == "otsu":
        return _threshold_otsu
    if method == "mean":
        return _threshold_mean
    if method == "minimum":
        return _threshold_minimum
    if method == "yen":
        return _threshold_yen
    if method == "triangle":
        return _threshold_triangle
    raise ValueError(method)


def _threshold_otsu(image: np.ndarray, otsu_threshold_multiplier: float = None, **kwargs) -> float:
    return threshold_otsu(image, **kwargs) * otsu_threshold_multiplier


def _threshold_mean(image: np.ndarray, otsu_threshold_multiplier: float = None, **kwargs) -> float:
    return threshold_mean(image, **kwargs)


def _threshold_minimum(image: np.ndarray, otsu_threshold_multiplier: float = None, **kwargs) -> float:
    return threshold_minimum(image, **kwargs)


def _threshold_yen(image: np.ndarray, otsu_threshold_multiplier: float = None, **kwargs) -> float:
    return threshold_yen(image, **kwargs)


def _threshold_triangle(image: np.ndarray, otsu_threshold_multiplier: float = None, **kwargs) -> float:
    return threshold_triangle(image, **kwargs)
