"""Skeletonize molecules"""
import logging
from typing import Callable
import numpy as np
from skimage.morphology import medial_axis, skeletonize, thin

from topostats.logs.logs import LOGGER_NAME

LOGGER = logging.getLogger(LOGGER_NAME)

# Max notes: Want to separate this module into:
#   the different skeletonisation skimage methods & joe's
#   the different branch pruning methods (mine & joe's)
#   skeleton descriptors (mine)

class getSkeleton():

    def __init__(self, image: np.ndarray, mask: np.ndarray, method: str):
        self.image = image
        self.mask = mask

    def get_skeleton(method: str) -> np.ndarray:
        """Factory method for skeletonizing molecules.

        Parameters
        ----------
        method : str
            Method to use, default is 'zhang' other options are 'lee', 'medial_axis' and 'thin'.

        Returns
        -------
        np.ndarray
            Skeletonised version of the image.all($0)

        Notes
        -----

        This is a thin wrapper to the methods provided
        by the `skimage.morphology
        <https://scikit-image.org/docs/stable/api/skimage.morphology.html?highlight=skeletonize>`_
        module. See also the `examples
        <https://scikit-image.org/docs/stable/auto_examples/edges/plot_skeleton.html>_
        """
        skeletonizer = _get_skeletonize(method)
        return skeletonizer(self.image, self.mask)


    def _get_skeletonize(method: str = "zhang") -> Callable:
        """Creator component which determines which skeletonize method to use.

        Parameters
        ----------
        method: str
            Method to use for skeletonizing, methods are 'zhang' (default), 'lee', 'medial_axis', and 'thin'.

        Returns
        -------
        Callable
            Returns the function appropriate for the required skeletonizing method.
        """
        if method == "zhang":
            return _skeletonize_zhang
        if method == "lee":
            return _skeletonize_lee
        if method == "medial_axis":
            return _skeletonize_medial_axis
        if method == "thin":
            return _skeletonize_thin
        raise ValueError(method)

    @staticmethod
    def _skeletonize_zhang(mask: np.ndarray) -> np.ndarray:
        return skeletonize(mask, method="zhang")

    @staticmethod
    def _skeletonize_lee(mask: np.ndarray) -> np.ndarray:
        return skeletonize(mask, method="lee")

    @staticmethod
    def _skeletonize_medial_axis(image: np.ndarray) -> np.ndarray:
        # don't know how these work - do they need img or mask?
        return medial_axis(image, return_distance=False)

    @staticmethod
    def _skeletonize_thin(image: np.ndarray) -> np.ndarray:
        # don't know how these work - do they need img or mask?
        return thin(image)

    @staticmethod
    def _skeletonize_joe(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        # image height bits don't seem to be used but are there?
        return joe(image, mask)

    def joe(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # need to find a way to make this work with the whole image
    # image height bits don't seem to be used but are there?
    # skimage zhang (what this is based on) seems to produce a different skeleton??

