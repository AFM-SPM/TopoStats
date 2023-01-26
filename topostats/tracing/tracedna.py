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
from skimage import morphology

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
        dilate: bool = False,
        dilation_iterations: int = 2,
        sigma: float = 1,
        skeletonisation_method: str = "zhang",
        linear_method: str = "ends",
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
        grain_number: int
            Grain number being processed.
        dilate: bool
            Whether to dilate the boolean mask.
        dilation_iterations: int
            Number of iterations to perform dilation for.
        sigma: float
            Number of standard deviations to use for Gaussian Filtering.
        skeletonisation_method: str
            Skeletonisation method
        linear_method: str
            Method to determine whether molecule is linear or circular. Default is 'ends' which attempts to count the
        number of cells in a skeleton that have only one adjacent neighbour. If there are only two then the molecule
        may be considered linear. However, if there are branches then the number of ends can be > 2. An alternative
        method ('regions') is to label the unique contiguous non-skeleton regions of the image. If a molecule is linear
        then the number of such unique regions will be 1. If a molecule is circular or has loops in it (e.g. folded
        into a figure-8) then the number of regions will be > 1. E.g. a circle will have 2 regions, a figure-8 will
        have 3 and so on.
        """
        super().__init__("grain")
        self.grain = {}
        self.grain["original"] = grain
        self.grain_number = grain_number
        self.dilate = dilate
        self.dilation_iterations = dilation_iterations
        self.sigma = sigma
        self.regions = None
        self.filename = filename
        self.pixel_to_nm_scaling = pixel_to_nm_scaling
        self.skeletonisation_method = skeletonisation_method
        self.ends = None
        self._circle = None

    def trace_dna(self):
        """Perform DNA tracing on a single grain."""
        # Step 1 Binary dilation
        self.grain["dilated"] = (
            self._dilate(grain=self.grain["mask"], iterations=self.dilation_iterations)
            if self.dilate
            else self.grain["mask"]
        )
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
        if self.linear_method == "end":
            self._count_adjacent()
            self._count_ends()
            self._circle = False if self.ends == 2 else True
        else:
            self._label_regions()
            self._count_regions()
            self._circle = False if self.regions == 1 else True

    def _count_ends(self):
        """Count how many points of a skeletonised image are ends.

        In this context an end is a non-zero cell that has a sum of adjacent (n = 8) non-zero cells of either 1 or 2 and
        the four adjacent cells in the x and y plane that are non-zero sum to 1. By way of example...

            Original     Adjacent    Abscissa/    Masked      Ends
                         non-zero    Ordinate
                                     non-zero

            0 0 0 0      1 1 1 0                 0 0 0 0     0 0 0 0    A linear skeleton is straight-forward there
            0 1 0 0      2 1 2 0                 0 1 0 0     0 1 0 0    are two ends and based solely on the adjacent
            0 1 0 0 -->  2 2 2 0 -->   N/A   --> 0 2 0 0 --> 0 0 0 0    (n = 8) surrounding cells after masking we have
            0 1 0 0      2 1 2 0                 0 1 0 0     0 1 0 0    only two cells with the value of 1 adjacent cell
            0 0 0 0      1 1 1 0                 0 0 0 0     0 0 0 0    that is non-zero

            0 0 0 0      1 1 1 0                 0 0 0 0     0 0 0 0    Another example of a linear skeleton with a diagonal
            0 0 1 0      1 2 1 0                 0 0 1 0     0 0 1 0    end and based solely on the adjacent (n = 8) cells
            0 1 0 0 -->  2 2 3 0 -->   N/A   --> 0 2 0 0 --> 0 0 0 0    cells after masking we again have only two cells
            0 1 0 0      2 1 2 0                 0 1 0 0     0 1 0 0    with the value of 1 adjacent cell that is non-zero.
            0 0 0 0      1 1 1 0                 0 0 0 0     0 0 0 0

            0 0 0 0      1 1 1 1                 0 0 0 0     0 0 0 0     However, when there is a "kink" at the end we
            0 1 1 0      2 2 2 1                 0 2 2 0     0 0 0 0     can not rely solely on the adjacent (n = 8) as the
            0 1 0 0 -->  2 2 3 1 --------------> 0 2 0 0 --> 0 0 0 0     kink results in there being only one end.
            0 1 0 0      2 1 2 0                 0 1 0 0     0 1 0 0
            0 0 0 0      1 1 1 0                 0 0 0 0     0 0 0 0
               |
               |                     0 1 1 0     0 0 0 0     0 0 0 0     Instead we also have to look at the adjacent cells
               |                     1 2 1 1     0 2 1 0     0 0 1 0     just in the abscissa/ordinate directions to
               |-------------------> 1 2 1 0 --> 0 2 0 0 --> 0 0 0 0     check if there are kinks.
                                     1 1 1 0     0 1 0 0     0 1 0 0
                                     0 1 0 0     0 0 0 0     0 0 0 0

        We then sum the end arrays from the adjacent non-zero and ordinal/abscissa to tell us how many ends there are in
        the image.

        What this approach doesn't deal well with is branching, which occurs frequently with the skeletonisation method
        of 'medial_axis' but also with others.
        """
        adjacent_ends = np.count_nonzero(self.grain["adjacent_masked"] == 1)
        abscissa_ordinate_ends = np.count_nonzero(self.grain["adjacent_abscissa_ordinate_masked"] == 1)
        if adjacent_ends == 2:
            LOGGER.info(f"[{self.filename}] | [{self.grain_number}] : Found two ends via all adjacent.")
            self.ends = adjacent_ends
        elif abscissa_ordinate_ends == 2:
            LOGGER.info(f"[{self.filename}] | [{self.grain_number}] : Found two ends via abscissa/ordinate adjacent.")
            self.ends = abscissa_ordinate_ends
        else:
            LOGGER.info(
                f"[{self.filename}] | [{self.grain_number}] : No ends found based on all adjacent or"
                "abscissa/ordinate adjacent alone, checking determining structure based on labelled regions"
            )
            self._label_regions()
            self._count_regions()
            self.ends = 2 if self.regions == 1 else 0
            LOGGER.info(
                f"[{self.filename}] | [{self.grain_number}] Ends counted ({self.ends}) based  on unique regions"
                f"({self.regions})"
            )

    def _label_regions(self) -> None:
        """Labels a skeletonised image to determine how many contiguous regions there are in an image.

        Linear molecules will have only one contiguous region as all non-skeleton areas will be adjacent. Circular DNA or
        those with twists/loops in will at least two regions (one inside, one out) or more if there are twists that form
        say figure-8 like structures."""
        self.grain["labelled"] = morphology.label(self.grain["skeleton"], background=1)

    def _count_regions(self) -> None:
        """Count the number of unique background regions in an image which is indicative of whether a molecule is linear or
        has a circle/loop in it."""
        self.regions = len(np.unique(self.grain["labelled"])) - 1

    def _inverse_mask(self) -> None:
        """Mask all cells but those that are part of the skeleton."""
        self.grain["skeleton_inverse_mask"] = np.where(self.grain["skeleton"] == 1, 0, 1)

    def _count_adjacent(self) -> None:
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

        Non-zero adjacent cells to x = a + b + c + d + e + f + g + h but since this method is vectorised we get the
        counts across the whole of the Original array at the same time.
        """
        padded = np.pad(self.grain["skeleton"], 1, mode="constant")
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
        self.grain["adjacent_masked"] = np.ma.masked_array(self.grain["adjacent"], self.grain["skeleton_inverse_mask"])
        # print(f"padded_mask            :\n{self.grain['skeleton_inverse_mask']}")
        # print(f"self.grain['adjacent'] :\n{self.grain['adjacent']}")
        print(f"self.grain['adjacent_masked'] :\n{self.grain['adjacent_masked']}")
        LOGGER.debug(f"[{self.filename}] | [{self.grain_number}] : Adjacent grains calculated for all adjacent cells")
        self.grain["adjacent_abscissa_ordinate"] = sum([top_center, middle_left, middle_right, bottom_center])
        self.grain["adjacent_abscissa_ordinate_masked"] = np.ma.masked_array(
            self.grain["adjacent_abscissa_ordinate"], self.grain["skeleton_inverse_mask"]
        )
        print(f"self.grain['adjacent_abscissa_ordinate_masked'] :\n{self.grain['adjacent_abscissa_ordinate_masked']}")
        LOGGER.debug(
            f"[{self.filename}] | [{self.grain_number}] : Adjacent grains calculated for abscissa and ordinate "
            "adjacent cells only."
        )

    def get_ordered_trace(self) -> None:
        """Order the trace"""

    def get_fitted_trace(self):
        """Something else"""

    def get_splined_trace(self):
        """Spline the trace."""
