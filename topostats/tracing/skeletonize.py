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

    def __init__(self, image: np.ndarray, mask: np.ndarray):
        self.image = image
        self.mask = mask

    def get_skeleton(self, method: str) -> np.ndarray:
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
        if method == "joe":
            return self._skeletonize_joe(self.image, self.mask)
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
        return joeFuncs(image, mask).doSkeletonising()


class joeFuncs():
    """Contains all the functions used for Joe's skeletonisation code"""
    def __init__(self, image: np.ndarray, mask: np.ndarray):
        self.image = image
        self.mask = mask.copy()
        self.skeleton_converged = False

    def doSkeletonising(self):
        """Simple while loop to check if the skeletonising is finished"""
        while not self.skeleton_converged:
            self._doSkeletonisingIteration()
        # When skeleton converged do an additional iteration of thinning to remove hanging points
        self.finalSkeletonisationIteration()

        return self.mask

    def _doSkeletonisingIteration(self):
        """Do an iteration of skeletonisation - check for the local binary pixel
        environment and assess the local height values to decide whether to
        delete a point
        """
        pixels_to_delete = []
        # Sub-iteration 1 - binary check
        mask_coordinates = np.argwhere(self.mask == 1).tolist()
        for point in mask_coordinates:
            if self._deletePixelSubit1(point):
                pixels_to_delete.append(point)

        # Check the local height values to determine if pixels should be deleted
        # pixels_to_delete = self._checkHeights(pixels_to_delete)

        for x, y in pixels_to_delete:
            self.mask[x, y] = 0

        # Sub-iteration 2 - binary check - seems like a repeat and could be condensed
        mask_coordinates = np.argwhere(self.mask == 1).tolist()
        for point in mask_coordinates:
            if self._deletePixelSubit2(point):
                pixels_to_delete.append(point)

        # Check the local height values to determine if pixels should be deleted
        # pixels_to_delete = self._checkHeights(pixels_to_delete)
        
        for x, y in pixels_to_delete:
            self.mask[x, y] = 0

        if len(pixels_to_delete) == 0:
            self.skeleton_converged = True

    def _deletePixelSubit1(self, point):
        """Function to check whether a single point should be deleted based
        on both its local binary environment and its local height values"""

        self.p2, self.p3, self.p4, self.p5, self.p6, self.p7, self.p8, self.p9 = self.getLocalPixelsBinary(
            self.mask, point[0], point[1]
        )
        return self._binaryThinCheck_a() and self._binaryThinCheck_b_returncount() == 1 and self._binaryThinCheck_c() and self._binaryThinCheck_d()

    def _deletePixelSubit2(self, point):
        """Function to check whether a single point should be deleted based
        on both its local binary environment and its local height values"""

        self.p2, self.p3, self.p4, self.p5, self.p6, self.p7, self.p8, self.p9 = self.getLocalPixelsBinary(
            self.mask, point[0], point[1]
        )
        # Add in generic code here to protect high points from being deleted
        return self._binaryThinCheck_a() and self._binaryThinCheck_b_returncount() == 1 and self._binaryThinCheck_csharp() and self._binaryThinCheck_dsharp()

    """These functions are ripped from the Zhang et al. paper and do the basic
    skeletonisation steps

    I can use the information from the c,d,c' and d' tests to determine a good
    direction to search for higher height values """

    def _binaryThinCheck_a(self):
        # Condition A protects the endpoints (which will be > 2) - add in code here to prune low height points
        return 2 <= self.p2 + self.p3 + self.p4 + self.p5 + self.p6 + self.p7 + self.p8 + self.p9 <= 6

    def _binaryThinCheck_b_returncount(self):
        # assess local area connectivity? (adding T/F)
        count = [self.p2, self.p3] == [0, 1]
        count += [self.p3, self.p4] == [0, 1]
        count += [self.p4, self.p5] == [0, 1]
        count += [self.p5, self.p6] == [0, 1]
        count += [self.p6, self.p7] == [0, 1]
        count += [self.p7, self.p8] == [0, 1]
        count += [self.p8, self.p9] == [0, 1]
        count += [self.p9, self.p2] == [0, 1]

        return count

    def _binaryThinCheck_c(self):
        # check if p2, p4 or p6 is 0
        return self.p2 * self.p4 * self.p6 == 0

    def _binaryThinCheck_d(self):
        # check if p4, p6 or p8 is 0
        return self.p4 * self.p6 * self.p8 == 0

    def _binaryThinCheck_csharp(self):
        return self.p2 * self.p4 * self.p8 == 0

    def _binaryThinCheck_dsharp(self):
        return self.p2 * self.p6 * self.p8 == 0

    def finalSkeletonisationIteration(self):
        """A final skeletonisation iteration that removes "hanging" pixels.
        Examples of such pixels are:

                    [0, 0, 0]               [0, 1, 0]            [0, 0, 0]
                    [0, 1, 1]               [0, 1, 1]            [0, 1, 1]
            case 1: [0, 1, 0]   or  case 2: [0, 1, 0] or case 3: [1, 1, 0]

        This is useful for the future functions that rely on local pixel environment
        to make assessments about the overall shape/structure of traces"""

        remaining_coordinates = np.argwhere(self.mask).tolist()

        for x, y in remaining_coordinates:

            (
                self.p2,
                self.p3,
                self.p4,
                self.p5,
                self.p6,
                self.p7,
                self.p8,
                self.p9,
            ) = self.getLocalPixelsBinary(self.mask, x, y)

            # Checks for case 1 pixels
            if self._binaryThinCheck_b_returncount() == 2 and self._binaryFinalThinCheck_a():
                self.mask[x, y] = 0
            # Checks for case 2 pixels
            elif self._binaryThinCheck_b_returncount() == 3 and self._binaryFinalThinCheck_b():
                self.mask[x, y] = 0

    def _binaryFinalThinCheck_a(self):
        # assess if local area has 4 connectivity
        return 1 in (self.p2 * self.p4, self.p4 * self.p6, self.p6 * self.p8, self.p8 * self.p2)


    def _binaryFinalThinCheck_b(self):
        # assess if local area has 4 connectivity
        return 1 in (self.p2 * self.p4 * self.p6, self.p4 * self.p6 * self.p8, self.p6 * self.p8 * self.p2, self.p8 * self.p2 * self.p4)

    @staticmethod
    def getLocalPixelsBinary(binary_map, x, y):
        # [[p9, p2, p3],
        #  [p8, na, p4],
        #  [p7, p6, p5]]
        p2 = binary_map[x, y + 1]
        p3 = binary_map[x + 1, y + 1]
        p4 = binary_map[x + 1, y]
        p5 = binary_map[x + 1, y - 1]
        p6 = binary_map[x, y - 1]
        p7 = binary_map[x - 1, y - 1]
        p8 = binary_map[x - 1, y]
        p9 = binary_map[x - 1, y + 1]

        return p2, p3, p4, p5, p6, p7, p8, p9


class pruneSkeleton():

    def __init__(self, image: np.ndarray, mask: np.ndarray) -> None:
        self.image = image
        self.mask = mask


