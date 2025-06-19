"""Functions for calculating thresholds."""

# pylint: disable=no-name-in-module
import logging
from collections.abc import Callable

import numpy.typing as npt
from skimage.filters import threshold_mean, threshold_minimum, threshold_otsu, threshold_triangle, threshold_yen

from topostats.logs.logs import LOGGER_NAME

LOGGER = logging.getLogger(LOGGER_NAME)

# pylint: disable=no-else-return
# pylint: disable=unused-argument


def threshold(image: npt.NDArray, method: str = None, otsu_threshold_multiplier: float = None, **kwargs: dict) -> float:
    """
    Thresholding for producing masks.

    Parameters
    ----------
    image : npt.NDArray
        2-D Numpy array of image for thresholding.
    method : str
        Method to use for thresholding, currently supported methods are otsu (default), mean and minimum.
    otsu_threshold_multiplier : float
        Factor for scaling the Otsu threshold.
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
    """
    Creator component which determines which threshold method to use.

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


def _threshold_otsu(image: npt.NDArray, otsu_threshold_multiplier: float = None, **kwargs) -> float:
    """
    Calculate the Otsu threshold.

    For more information see `skimage.filters.threshold_otsu()
    <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_otsu>`_.

    Parameters
    ----------
    image : npt.NDArray
        2-D Numpy array of image for thresholding.
    otsu_threshold_multiplier : float
        Factor for scaling Otsu threshold.
    **kwargs : dict
        Dictionary of keyword arguments to pass to 'skimage.filters.threshold_otsu(**kwargs)'.

    Returns
    -------
    float
        Threshold to be used in masking heights.
    """
    return threshold_otsu(image, **kwargs) * otsu_threshold_multiplier


def _threshold_mean(image: npt.NDArray, otsu_threshold_multiplier: float = None, **kwargs) -> float:
    """
    Calculate the Mean threshold.

    For more information see `skimage.filters.threshold_mean()
    <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_mean>`_.

    Parameters
    ----------
    image : npt.NDArray
        2-D Numpy array of image for thresholding.
    otsu_threshold_multiplier : float
        Factor for scaling (not used).
    **kwargs : dict
        Dictionary of keyword arguments to pass to 'skimage.filters.threshold_mean(**kwargs)'.

    Returns
    -------
    float
        Threshold to be used in masking heights.
    """
    return threshold_mean(image, **kwargs)


def _threshold_minimum(image: npt.NDArray, otsu_threshold_multiplier: float = None, **kwargs) -> float:
    """
    Calculate the Minimum threshold.

    For more information see `skimage.filters.threshold_minimum()
    <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_minimum>`_.

    Parameters
    ----------
    image : npt.NDArray
        2-D Numpy array of image for thresholding.
    otsu_threshold_multiplier : float
        Factor for scaling (not used).
    **kwargs : dict
        Dictionary of keyword arguments to pass to 'skimage.filters.threshold_minimum(**kwargs)'.

    Returns
    -------
    float
        Threshold to be used in masking heights.
    """
    return threshold_minimum(image, **kwargs)


def _threshold_yen(image: npt.NDArray, otsu_threshold_multiplier: float = None, **kwargs) -> float:
    """
    Calculate the Yen threshold.

    For more information see `skimage.filters.threshold_yen()
    <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_yen>`_.

    Parameters
    ----------
    image : npt.NDArray
        2-D Numpy array of image for thresholding.
    otsu_threshold_multiplier : float
        Factor for scaling (not used).
    **kwargs : dict
        Dictionary of keyword arguments to pass to 'skimage.filters.threshold_yen(**kwargs)'.

    Returns
    -------
    float
        Threshold to be used in masking heights.
    """
    return threshold_yen(image, **kwargs)


def _threshold_triangle(image: npt.NDArray, otsu_threshold_multiplier: float = None, **kwargs) -> float:
    """
    Calculate the triangle threshold.

    For more information see `skimage.filters.threshold_triangle()
    <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_triangle>`_.

    Parameters
    ----------
    image : npt.NDArray
        2-D Numpy array of image for thresholding.
    otsu_threshold_multiplier : float
        Factor for scaling (not used).
    **kwargs : dict
        Dictionary of keyword arguments to pass to 'skimage.filters.threshold_triangle(**kwargs)'.

    Returns
    -------
    float
        Threshold to be used in masking heights.
    """
    return threshold_triangle(image, **kwargs)
