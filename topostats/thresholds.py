"""Functions for calculating thresholds."""
# pylint: disable=no-name-in-module
from typing import Callable
import numpy as np
from skimage.filters import (
    threshold_mean,
    threshold_minimum,
    threshold_otsu,
    threshold_yen,
    threshold_triangle,
)
import logging
from topostats.logs.logs import LOGGER_NAME

LOGGER = logging.getLogger(LOGGER_NAME)


def threshold(image: np.array, method: str = None, otsu_threshold_multiplier: float = None, **kwargs: dict) -> float:
    """Factory method for thresholding.

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
    print(f'THRESHOLD() OTSU MULTIPLIER: {otsu_threshold_multiplier}')
    return thresholder(image, otsu_threshold_multiplier=otsu_threshold_multiplier, **kwargs)


def _get_threshold(method: str = "otsu") -> Callable:
    """Creator component which determines which threshold method to use.

    Parameters
    ----------
    method : str
        Threshold method to use, currently supports otsu (default), std_dev_lower, std_dev_upper, minimum, mean and yen.

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
    elif method == "mean":
        return _threshold_mean
    elif method == "minimum":
        return _threshold_minimum
    elif method == "yen":
        return _threshold_yen
    elif method == "triangle":
        return _threshold_triangle
    # elif method == 'std_dev_lower':
    #     return _threshold_std_dev_lower
    # elif method == 'std_dev_upper':
    #     return _threshold_std_dev_upper
    else:
        raise ValueError(method)


def _threshold_otsu(image: np.array, otsu_threshold_multiplier: float = None, **kwargs) -> float:
    print(f"MULTIPLIER THRESHOLD OTSU: {otsu_threshold_multiplier}")
    return threshold_otsu(image, **kwargs) * otsu_threshold_multiplier


def _threshold_mean(image: np.array, otsu_threshold_multiplier: float = None, **kwargs) -> float:
    return threshold_mean(image)


def _threshold_minimum(image: np.array, otsu_threshold_multiplier: float = None, **kwargs) -> float:
    return threshold_minimum(image, **kwargs)


def _threshold_yen(image: np.array, otsu_threshold_multiplier: float = None, **kwargs) -> float:
    return threshold_yen(image, **kwargs)


def _threshold_triangle(image: np.array, otsu_threshold_multiplier: float = None, **kwargs) -> float:
    return threshold_triangle(image, **kwargs)


# def _threshold_std_dev_lower(image: np.array, threshold_multiplier: float, **kwargs) -> float:
#     mean = np.nanmean(image)
#     std_dev = np.nanstd(image)
#     return mean - (float(threshold_multiplier) * std_dev)

# def _threshold_std_dev_lower(image: np.array, threshold_multiplier: float, **kwargs) -> float:
#     mean = np.nanmean(image)
#     std_dev = np.nanstd(image)
#     return mean - (float(threshold_multiplier) * std_dev)

# def _threshold_std_dev_upper(image: np.array, threshold_multiplier: float, **kwargs) -> float:
#     mean = np.nanmean(image)
#     std_dev = np.nanstd(image)
#     return mean + float(threshold_multiplier) * std_dev
