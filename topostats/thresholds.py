"""Functions for calculating thresholds."""
import numpy as np
from skimage.filters import threshold_mean, threshold_minimum, threshold_otsu, threshold_yen, threshold_triangle


def threshold(image: np.array, method: str = 'otsu', **kwargs: dict) -> float:
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
    return thresholder(image, **kwargs)


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
    else:
        raise ValueError(method)


def _threshold_otsu(image: np.array, **kwargs) -> float:
    return threshold_otsu(image, **kwargs)


def _threshold_mean(image: np.array) -> float:
    return threshold_mean(image)


def _threshold_minimum(image: np.array, **kwargs) -> float:
    return threshold_minimum(image, **kwargs)


def _threshold_yen(image: np.array, **kwargs) -> float:
    return threshold_yen(image, **kwargs)


def _threshold_triangle(image: np.array, **kwargs) -> float:
    return threshold_triangle(image, **kwargs)
