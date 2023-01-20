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
        grain: np.ndarray,
        filename: str,
        pixel_to_nm_scaling: float,
        grain_number: int = 0,
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
        self.iterations = iterations
        self.sigma = sigma
        self.filename = filename
        self.pixel_to_nm_scaling = pixel_to_nm_scaling
        self.skeletonisation_method = skeletonisation_method
        self._circle = None

    def trace_dna(self):
        """Perform DNA tracing on a single grain."""
        # Step 1 Binary dilation
        self.grain["dilated"] = self._dilate(grain=self.grain["mask"], iterations=self.iterations)
        # Step 2 Gaussian filter
        self.grain["filtered"] = self._gaussian_filter(grain=self.grain["dilated"], sigma=self.sigma)
        # Step 3 - Skeletonize the image, there is no get_disordered trace as the docstring for get_disordered_trace()
        # method in dnatracing indicates this is the skeletonisation process.
        self.grain["skeleton"] = self.skeletonize()
        # TODO (2023-01-17) - Set a configurable and minimal size for skeletons to be to continue
        # Don't purge, skeletonise and assess if its nonsense/single pixel
        # self.purge_obvious_crap()
        # self.determine_linear()
        self.circle()
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

    # Disordere trace is just a skeleton as far as I can tell, old method used co-ordinates of these but we have
    # vectorised the approach which should be faster and doesn't rely on ordering so much.
    # def get_disordered_trace(self) -> None:
    #     """Something"""

    @property
    def circle(self) -> bool:
        """Whether the grain is circular."""
        return self._circle

    @circle.setter
    def circle(self):
        """Set whether circle or not"""
        self._count_adjacent
        self._inverse_mask()
        self.grain["adjacent_masked"] = np.ma.masked_array(self.grain["adjacent"], self.grain["skeleton_inverse_mask"])
        self._circle = True if self._count_ends() == 0 else False

    def _count_ends(self):
        """Count how many points of a skeletonised image are ends.

        In this context an end is a non-zero cell that has only one adjacent non-zero cell."""
        return np.count_nonzero(self.grain["adjacent"] == 1)

    def _inverse_mask(self):
        """Mask all cells but those that are part of the skeleton."""
        self.grain["skeleton_inverse_mask"] = np.where(self.grain["skeleton"] == 1, 0, 1)

    def _count_adjacent(self):
        """Count the number of non-zero cells around every x, y co-ordinate in a binary 2-D Numpy array.

        This is done by padding the array and creating eight shifted arrays, one for each adjacent cell.

        These are then summed and because the input is a binary array the total number non-zero adjacent cells is
        obtained.

        If x is a given cell in the array and we want to know how many adjacent cells are non-zero (i.e. the sum of
        a, b, c, d, e f, g and h)

          Original      Padded               Summing

                                        A       B       C
                       0 0 0 0 0      0 0 0   0 0 0   0 0 0
            a b c      0 a b c 0      0 a b   a b c   b c 0
            d x e -->  0 d x e 0 -->  0 d x   d x e   x e 0
            f g h      0 f g h 0
                       0 0 0 0 0        D               E
                                      0 a b           b c 0
                                      0 d x           x e 0
                                      0 f g           g h 0

                                        F       G       H
                                      0 d x   d x e   x e 0
                                      0 f g   f g h   g h 0
                                      0 0 0   0 0 0   0 0 0

        Non-zero adjacent cells to x = A + B + C + D + E + F + G + H
        """
        padded = np.pad(self.grain["skeleton"], 1, mode="constant")
        padded_mask = np.where(self.grain["skeleton"] == 0, 1, 0)
        # print(f"padded :\n{padded}")
        # print(f"padded_mask :\n{padded_mask}")
        # Use slicing to get the 8 surrounding cells of each element in the padded array
        top_left = padded[:-2, :-2]
        top_center = padded[:-2, 1:-1]
        top_right = padded[:-2, 2:]
        middle_left = padded[1:-1, :-2]
        middle_right = padded[1:-1, 2:]
        bottom_left = padded[2:, :-2]
        bottom_center = padded[2:, 1:-1]
        bottom_right = padded[2:, 2:]

        # Get number of adjacent cells for the skeleton only by summing and masking non-skeleton cells
        self.grain["adjacent"] = sum(
            [
                top_left,
                top_center,
                top_right,
                middle_left,
                middle_right,
                bottom_left,
                bottom_center,
                bottom_right,
            ]
        )
        # self.grain["skeleton_adjacent"] =
        self.grain["adjacent_masked"] = np.ma.masked_array(self.grain["adjacent"], padded_mask)
        # print(f"self.grain['adjacent'] :\n{self.grain['adjacent']}")
        # print(f"self.grain['adjacent_masked'] :\n{self.grain['adjacent_masked']}")

    def get_ordered_trace(self) -> None:
        """Order the trace"""

    def get_fitted_trace(self):
        """Something else"""

    def get_splined_trace(self):
        """Spline the trace."""
