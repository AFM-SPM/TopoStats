"""Functions for calculating thresholds."""
import numpy as np
from skimage.filters import threshold_mean, threshold_minimum, threshold_otsu, threshold_yen, threshold_triangle
import logging
from topostats.logs.logs import LOGGER_NAME
LOGGER = logging.getLogger(LOGGER_NAME)

def threshold(image: np.array, method: str = 'otsu', threshold_multiplier: float = None, **kwargs: dict) -> float:
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

    Examples
    --------
    FIXME: Add docs.

    """
    thresholder = _get_threshold(method)
    return thresholder(image, threshold_multiplier, **kwargs)


def _get_threshold(method: str = 'otsu'):
    """Creator component which determines which threshold method to use.

    Parameters
    ----------
    method : str
        Threshold method to use, currently supports otsu (default), minimum, mean and yen.

    Returns
    -------
    function
        Returns function appropriate for the required threshold method.

    Raises
    ------
    ValueError
        Unsupported methods result in ValueError.

    Examples
    --------
    FIXME: Add docs.

    """
    if method == 'otsu':
        return _threshold_otsu
    elif method == 'mean':
        return _threshold_mean
    elif method == 'minimum':
        return _threshold_minimum
    elif method == 'yen':
        return _threshold_yen
    elif method == 'triangle':
        return _threshold_triangle
    elif method == 'std_dev_lower':
        return _threshold_std_dev_lower
    elif method == 'std_dev_upper':
        return _threshold_std_dev_upper
    else:
        raise ValueError(method)


def _threshold_otsu(image: np.array, threshold_multiplier: float, **kwargs) -> float:
    return threshold_otsu(image, **kwargs) * float(threshold_multiplier)


def _threshold_mean(image: np.array) -> float:
    return threshold_mean(image)


def _threshold_minimum(image: np.array, **kwargs) -> float:
    return threshold_minimum(image, **kwargs)


def _threshold_yen(image: np.array, **kwargs) -> float:
    return threshold_yen(image, **kwargs)


def _threshold_triangle(image: np.array, **kwargs) -> float:
    return threshold_triangle(image, **kwargs)

def _threshold_std_dev_lower(image: np.array, threshold_multiplier: float, **kwargs) -> float:
    mean = np.nanmean(image)
    std_dev = np.nanstd(image)
    LOGGER.info(f"THRESHOLDING LOWER THRESHOLD MULTIPLIER: {threshold_multiplier}")
    LOGGER.info('THRESHOLDING : mean, std dev: ' + str(mean) + ' ' + str(std_dev))
    threshold = mean - (float(threshold_multiplier) * std_dev)
    LOGGER.info(f'-threshold: {threshold}')
    return threshold

def _threshold_std_dev_upper(image: np.array, threshold_multiplier: float, **kwargs) -> float:
    mean = np.nanmean(image)
    std_dev = np.nanstd(image)
    return mean + float(threshold_multiplier) * std_dev