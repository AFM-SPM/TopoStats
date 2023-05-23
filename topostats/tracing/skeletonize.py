"""Skeletonize molecules"""
import logging
from typing import Callable
import numpy as np
from skimage.morphology import medial_axis, skeletonize, thin

# from skimage.filters import meijering, sato, frangi, hessian

# from topostats.tracing.tracingfuncs import genTracingFuncs
from topostats.logs.logs import LOGGER_NAME

LOGGER = logging.getLogger(LOGGER_NAME)

# Max notes: Want to separate this module into:
#   the different skeletonisation skimage methods & joe's
#   the different branch pruning methods (mine & joe's)
#   skeleton descriptors (mine)


class getSkeleton:  # pylint: disable=too-few-public-methods
    """Class containing skeletonization code from factory methods to functions
    depaendant on the method"""

    def __init__(self, image: np.ndarray, mask: np.ndarray):
        """Initialise the class.
        Parameters
        ----------
        image: np.ndarray
            The image used to generate the mask.
        mask: np.ndarray
            The binary mask of features in the image.
        """
        self.image = image
        self.mask = mask

    def get_skeleton(self, method: str) -> np.ndarray:
        """Factory method for skeletonizing molecules.
        Parameters
        ----------
        method : str
            Method to use, default is 'zhang' other options are 'lee', 'medial_axis', 'thin'.
        Returns
        -------
        np.ndarray
            Skeletonised version of the binary mask (possibly using criteria from the image).
        Notes
        -----
        This is a thin wrapper to the methods provided
        by the `skimage.morphology
        <https://scikit-image.org/docs/stable/api/skimage.morphology.html?highlight=skeletonize>`_
        module. See also the `examples
        <https://scikit-image.org/docs/stable/auto_examples/edges/plot_skeleton.html>_
        """
        return self._get_skeletonize(method)

    def _get_skeletonize(self, method: str = "zhang") -> Callable:
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
            return self._skeletonize_zhang(self.mask)
        if method == "lee":
            return self._skeletonize_lee(self.mask)
        if method == "medial_axis":
            return self._skeletonize_medial_axis(self.mask)
        if method == "thin":
            return self._skeletonize_thin(self.mask)
        raise ValueError(method)

    @staticmethod
    def _skeletonize_zhang(image: np.ndarray) -> np.ndarray:
        """Wrapper for the scikit image implementation of the Zhang skeletonisation method.

        Parameters
        ----------
        image: np.ndarray
            A binary array to skeletonise.

        Returns
        -------
        np.ndarray
            The mask array reduce to a single pixel thickness
        """
        return skeletonize(image, method="zhang")

    @staticmethod
    def _skeletonize_lee(image: np.ndarray) -> np.ndarray:
        """Wrapper for the scikit image implementation of the Lee skeletonisation method.

        Parameters
        ----------
        imagemask: np.ndarray
            A binary array to skeletonise.

        Returns
        -------
        np.ndarray
            The mask array reduce to a single pixel thickness
        """
        return skeletonize(image, method="lee")

    @staticmethod
    def _skeletonize_medial_axis(image: np.ndarray) -> np.ndarray:
        """Wrapper for the scikit image implementation of the medial axis skeletonisation method.

        Parameters
        ----------
        image: np.ndarray
            A binary array to skeletonise.

        Returns
        -------
        np.ndarray
            The mask array reduce to a single pixel thickness
        """
        return medial_axis(image, return_distance=False)

    @staticmethod
    def _skeletonize_thin(image: np.ndarray) -> np.ndarray:
        """Wrapper for the scikit image implementation of the thin skeletonisation method.

        Parameters
        ----------
        image: np.ndarray
            A binary array to skeletonise.

        Returns
        -------
        np.ndarray
            The mask array reduce to a single pixel thickness
        """
        return thin(image)
