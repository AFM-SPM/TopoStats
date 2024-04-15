"""Skeletonize molecules."""

import logging
from collections.abc import Callable

import numpy.typing as npt
from skimage.morphology import skeletonize, thin

from topostats.logs.logs import LOGGER_NAME

LOGGER = logging.getLogger(LOGGER_NAME)


def get_skeleton(image: npt.NDArray, method: str) -> npt.NDArray:
    """
    Skeletonizing masked molecules.

    Parameters
    ----------
    image : npt.NDArray
        Image of molecule to be skeletonized.
    method : str
        Method to use, default is 'zhang' other options are 'lee', and 'thin'.

    Returns
    -------
    npt.NDArray
        Skeletonised version of the image.all($0).

    Notes
    -----
    This is a thin wrapper to the methods provided
    by the `skimage.morphology
    <https://scikit-image.org/docs/stable/api/skimage.morphology.html?highlight=skeletonize>`_
    module. See also the `examples
    <https://scikit-image.org/docs/stable/auto_examples/edges/plot_skeleton.html>_
    """
    skeletonizer = _get_skeletonize(method)
    return skeletonizer(image)


def _get_skeletonize(method: str = "zhang") -> Callable:
    """
    Creator component which determines which skeletonize method to use.

    Parameters
    ----------
    method : str
        Method to use for skeletonizing, methods are 'zhang' (default), 'lee', and 'thin'.

    Returns
    -------
    Callable
        Returns the function appropriate for the required skeletonizing method.
    """
    if method == "zhang":
        return _skeletonize_zhang
    if method == "lee":
        return _skeletonize_lee
    if method == "thin":
        return _skeletonize_thin
    raise ValueError(method)


def _skeletonize_zhang(image: npt.NDArray) -> npt.NDArray:
    """
    Skeletonize using Zhang method.

    Parameters
    ----------
    image : npt.NDArray
        Numpy array to be skeletonized.

    Returns
    -------
    npt.NDArray
        Skeletonized Numpy array.
    """
    return skeletonize(image, method="zhang")


def _skeletonize_lee(image: npt.NDArray) -> npt.NDArray:
    """
    Skeletonize using Lee method.

    Parameters
    ----------
    image : npt.NDArray
        Numpy array to be skeletonized.

    Returns
    -------
    npt.NDArray
        Skeletonized Numpy array.
    """
    return skeletonize(image, method="lee")


def _skeletonize_thin(image: npt.NDArray) -> npt.NDArray:
    """
    Skeletonize using thinning method.

    Parameters
    ----------
    image : npt.NDArray
        Numpy array to be skeletonized.

    Returns
    -------
    npt.NDArray
        Skeletonized Numpy array.
    """
    return thin(image)
