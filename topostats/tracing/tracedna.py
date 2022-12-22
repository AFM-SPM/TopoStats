"""Perform DNA Tracing"""
# from collections import OrderedDict
import logging

# from pathlib import Path
# import math
# from typing import Dict , Union, Tuple

# import warnings

import numpy as np

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
from scipy.ndimage import binary_dilation, gaussian_filter

# from scipy.interpolate import splprep, splev
# from skimage import morphology
# from skimage.filters import gaussian
# from skimage.measure import label, regionprops, regionprops_table

from topostats.logs.logs import LOGGER_NAME
from topostats.tracing.skeletonize import getSkeleton
from topostats.tracing.utils import orderTrace

# from topostats.tracing.tracingfuncs import genTracingFuncs, getSkeleton, reorderTrace

LOGGER = logging.getLogger(LOGGER_NAME)


class traceDNA(orderTrace):  # pylint: disable=too-few-public-methods
    """Trace a single grain of DNA."""

    def __init__(
        self,
        grain: dict,
        filename: str,
        pixel_to_nm_scaling: float,
        iterations: int = 2,
        sigma: float = 1,
        skeletonisation_method: str = "zhang",
    ):
        """Initialise the class.

        Parameters
        ----------
        grain: dict
            Single grain dictionary returned by GrainStats, should have key:values for cropped images with 'original',
        'mask' versions of the image.
        filename: str
            Filename
        pixel_to_nm_scaling: float
            Pixel to nanometer sacling factor
        iterations: int
            Number of iterations to perform dilation for.
        sigma: float
            Number of standard deviations to use for Gaussian Filtering.
        skeletonisation_method: str
            Skeletonisation method
        """
        super().__init__("grain")
        self.grain = {}
        self.grain["original"] = grain
        # self.grain["mask"] = self.grain["mask"]
        self.iterations = iterations
        self.sigma = sigma
        self.filename = filename
        self.pixel_to_nm_scaling = pixel_to_nm_scaling
        self.skeletonisation_method = skeletonisation_method

    def trace_dna(self):
        """Perform DNA tracing on a single grain."""
        # Step 1 Binary dilation
        self.grain["dilated"] = self._dilate(grain=self.grain["mask"], iterations=self.iterations)
        # Step 2 Gaussian filter
        self.grain["filtered"] = self._gaussian_filter(grain=self.grain["dilated"], sigma=self.sigma)
        # Step 3 - Skeletonize the image
        self.skeletonize()
        # Don't purge, skeletonise and assess if its nonsense/single pixel
        # self.purge_obvious_crap()
        # self.determine_linear()
        self.order(self._circle)
        # self.determine_morphology()
        # self.get_fitted_traces()
        # self.get_splined_traces()

    @staticmethod
    def _dilate(grain: np.ndarray, iterations: int) -> np.ndarray:
        """Dilate the grain.

        Parameters
        ----------
        grain: np.ndarray
            Mask of image to dilate.
        iterations: int
            Number of iterations to dilate for.

        Returns
        -------
        np.ndarray
            Dilated grain.
        """
        return binary_dilation(grain, iterations=iterations).astype(grain.dtype)

    @staticmethod
    def _gaussian_filter(grain: np.ndarray, sigma: float) -> np.ndarray:
        """Apply Gaussian Filter to grain.

        Parameters
        ----------
        grain : np.ndarray
            Grain to be filtered.
        sigma : float
            Variance for filtering.

        Returns
        -------
        np.ndarray
            Gaussian filtered image.
        """
        return gaussian_filter(grain, sigma)

    def skeletonize(self) -> None:
        """Skeletonise"""
        # try:
        self.grain["skeleton"] = getSkeleton(image=self.grain["dilated"], mask=self.grain["mask"]).get_skeleton(
            method=self.skeletonisation_method
        )
        # except:  # noqua
        #    raise

    def get_disordered_trace(self) -> None:
        """Something"""

    @property
    def circle(self) -> bool:
        """Whether the grain is circular."""
        return self._circle

    # @property.setter
    # def circle(self):
    #     """Set whether circle or not"""

    def get_ordered_trace(self) -> None:
        """Order the trace"""

    def get_fitted_trace(self):
        """Something else"""

    def get_splined_trace(self):
        """Spline the trace."""
