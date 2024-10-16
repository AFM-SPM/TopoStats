"""Skeletonize molecules."""

import logging
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from skimage.morphology import medial_axis, skeletonize, thin

from topostats.logs.logs import LOGGER_NAME

LOGGER = logging.getLogger(LOGGER_NAME)


class getSkeleton:  # pylint: disable=too-few-public-methods
    """
    Class skeletonising images.

    Parameters
    ----------
    image : npt.NDArray
        Image used to generate the mask.
    mask : npt.NDArray
        Binary mask of features.
    method : str
        Method for skeletonizing. Options 'zhang' (default), 'lee', 'medial_axis', 'thin' and 'topostats'.
    height_bias : float
        Ratio of lowest intensity (height) pixels to total pixels fitting the skeletonisation criteria. 1 is all pixels
        smiilar to Zhang.
    """

    def __init__(self, image: npt.NDArray, mask: npt.NDArray, method: str = "zhang", height_bias: float = 0.6):
        """
        Initialise the class.

        This is a thin wrapper to the methods provided by the `skimage.morphology
        <https://scikit-image.org/docs/stable/api/skimage.morphology.html?highlight=skeletonize>`_
        module. See also the `examples
        <https://scikit-image.org/docs/stable/auto_examples/edges/plot_skeleton.html>_

        Parameters
        ----------
        image : npt.NDArray
            Image used to generate the mask.
        mask : npt.NDArray
            Binary mask of features.
        method : str
            Method for skeletonizing. Options 'zhang' (default), 'lee', 'medial_axis', 'thin' and 'topostats'.
        height_bias : float
            Ratio of lowest intensity (height) pixels to total pixels fitting the skeletonisation criteria. 1 is all
            pixels smiilar to Zhang.
        """
        # Q What benefit is there to having a class getSkeleton over the get_skeleton() function? Ostensibly the class
        # is doing only one thing, we don't need to change state/modify anything here. Beyond encapsulating all
        # functions in a single class this feels like overkill.
        self.image = image
        self.mask = mask
        self.method = method
        self.height_bias = height_bias

    def get_skeleton(self) -> npt.NDArray:
        """
        Skeletonise molecules.

        Returns
        -------
        npt.NDArray
            Skeletonised version of the binary mask (possibly using criteria from the image).
        """
        return self._get_skeletonize()

    def _get_skeletonize(self) -> Callable:
        """
        Determine which skeletonise method to use.

        Returns
        -------
        Callable
            Returns the function appropriate for the required skeletonizing method.
        """
        if self.method == "zhang":
            return self._skeletonize_zhang(mask=self.mask).astype(np.int8)
        if self.method == "lee":
            return self._skeletonize_lee(mask=self.mask).astype(np.int8)
        if self.method == "medial_axis":
            return self._skeletonize_medial_axis(mask=self.mask).astype(np.int8)
        if self.method == "thin":
            return self._skeletonize_thin(mask=self.mask).astype(np.int8)
        if self.method == "topostats":
            return self._skeletonize_topostats(image=self.image, mask=self.mask, height_bias=self.height_bias).astype(
                np.int8
            )
        raise ValueError(self.method)

    @staticmethod
    def _skeletonize_zhang(mask: npt.NDArray) -> npt.NDArray:
        """
        Use scikit-image implementation of the Zhang skeletonisation method.

        Parameters
        ----------
        mask : npt.NDArray
            Binary array to skeletonise.

        Returns
        -------
        npt.NDArray
            Mask array reduced to a single pixel thickness.
        """
        return skeletonize(mask, method="zhang")

    @staticmethod
    def _skeletonize_lee(mask: npt.NDArray) -> npt.NDArray:
        """
        Use scikit-image implementation of the Lee skeletonisation method.

        Parameters
        ----------
        mask : npt.NDArray
            Binary array to skeletonise.

        Returns
        -------
        npt.NDArray
            Mask array reduced to a single pixel thickness.
        """
        return skeletonize(mask, method="lee")

    @staticmethod
    def _skeletonize_medial_axis(mask: npt.NDArray) -> npt.NDArray:
        """
        Use scikit-image implementation of the Medial axis skeletonisation method.

        Parameters
        ----------
        mask : npt.NDArray
            Binary array to skeletonise.

        Returns
        -------
        npt.NDArray
            Mask array reduced to a single pixel thickness.
        """
        return medial_axis(mask, return_distance=False)

    @staticmethod
    def _skeletonize_thin(mask: npt.NDArray) -> npt.NDArray:
        """
        Use scikit-image implementation of the thinning skeletonisation method.

        Parameters
        ----------
        mask : npt.NDArray
            Binary array to skeletonise.

        Returns
        -------
        npt.NDArray
            Mask array reduced to a single pixel thickness.
        """
        return thin(mask)

    @staticmethod
    def _skeletonize_topostats(image: npt.NDArray, mask: npt.NDArray, height_bias: float = 0.6) -> npt.NDArray:
        """
        Use scikit-image implementation of the Zhang skeletonisation method.

        This method is based on Zhang's method but produces different results (less branches but slightly less
        accurate).

        Parameters
        ----------
        image : npt.NDArray
            Original image with heights.
        mask : npt.NDArray
            Binary array to skeletonise.
        height_bias : float
            Ratio of lowest intensity (height) pixels to total pixels fitting the skeletonisation criteria. 1 is all
            pixels smiilar to Zhang.

        Returns
        -------
        npt.NDArray
            Masked array reduced to a single pixel thickness.
        """
        return topostatsSkeletonize(image, mask, height_bias).do_skeletonising()


class topostatsSkeletonize:  # pylint: disable=too-many-instance-attributes
    """
    Skeletonise a binary array following Zhang's algorithm (Zhang and Suen, 1984).

    Modifications are made to the published algorithm during the removal step to remove a fraction of the smallest pixel
    values opposed to all of them in the aforementioned algorithm. All operations are performed on the mask entered.

    Parameters
    ----------
    image : npt.NDArray
        Original 2D image containing the height data.
    mask : npt.NDArray
        Binary image containing the object to be skeletonised. Dimensions should match those of 'image'.
    height_bias : float
        Ratio of lowest intensity (height) pixels to total pixels fitting the skeletonisation criteria. 1 is all pixels
        smiilar to Zhang.
    """

    def __init__(self, image: npt.NDArray, mask: npt.NDArray, height_bias: float = 0.6):
        """
        Initialise the class.

        Parameters
        ----------
        image : npt.NDArray
            Original 2D image containing the height data.
        mask : npt.NDArray
            Binary image containing the object to be skeletonised. Dimensions should match those of 'image'.
        height_bias : float
            Ratio of lowest intensity (height) pixels to total pixels fitting the skeletonisation criteria. 1 is all
            pixels smiilar to Zhang.
        """
        self.image = image
        self.mask = mask.copy()
        self.height_bias = height_bias

        self.skeleton_converged = False
        self.p2 = None
        self.p3 = None
        self.p4 = None
        self.p5 = None
        self.p6 = None
        self.p7 = None
        self.p8 = None
        self.p9 = None
        self.counter = 0

    def do_skeletonising(self) -> npt.NDArray:
        """
        Perform skeletonisation.

        Returns
        -------
        npt.NDArray
            The single pixel thick, skeletonised array.
        """
        while not self.skeleton_converged:
            self._do_skeletonising_iteration()
        # When skeleton converged do an additional iteration of thinning to remove hanging points
        self.final_skeletonisation_iteration()
        self.mask = getSkeleton(
            image=self.image, mask=self.mask, method="zhang"
        ).get_skeleton()  # not sure if this is needed?

        return self.mask

    def _do_skeletonising_iteration(self) -> None:
        """
        Obtain the local binary pixel environment and assess the local pixel values.

        This determines whether to delete a point according to the Zhang algorithm.

        Then removes ratio of lowest intensity (height) pixels to total pixels fitting the skeletonisation criteria. 1
        is all pixels smiilar to Zhang.
        """
        skel_img = self.mask.copy()
        pixels_to_delete = []
        # Sub-iteration 1 - binary check
        mask_coordinates = np.argwhere(self.mask == 1).tolist()
        for point in mask_coordinates:
            if self._delete_pixel_subit1(point):
                pixels_to_delete.append(point)

        # remove points based on height (lowest height_bias%)
        pixels_to_delete = np.asarray(pixels_to_delete)  # turn into array
        if pixels_to_delete.shape != (0,):  # ensure array not empty
            skel_img[pixels_to_delete[:, 0], pixels_to_delete[:, 1]] = 2
            heights = self.image[pixels_to_delete[:, 0], pixels_to_delete[:, 1]]  # get heights of pixels
            height_sort_idx = self.sort_and_shuffle(heights)[1][
                : int(np.ceil(len(heights) * self.height_bias))
            ]  # idx of lowest height_bias%
            self.mask[pixels_to_delete[height_sort_idx, 0], pixels_to_delete[height_sort_idx, 1]] = (
                0  # remove lowest height_bias%
            )

        pixels_to_delete = []
        # Sub-iteration 2 - binary check
        mask_coordinates = np.argwhere(self.mask == 1).tolist()
        for point in mask_coordinates:
            if self._delete_pixel_subit2(point):
                pixels_to_delete.append(point)

        # remove points based on height (lowest height_bias%)
        pixels_to_delete = np.asarray(pixels_to_delete)
        if pixels_to_delete.shape != (0,):
            skel_img[pixels_to_delete[:, 0], pixels_to_delete[:, 1]] = 3
            heights = self.image[pixels_to_delete[:, 0], pixels_to_delete[:, 1]]
            height_sort_idx = self.sort_and_shuffle(heights)[1][
                : int(np.ceil(len(heights) * self.height_bias))
            ]  # idx of lowest height_bias%
            self.mask[pixels_to_delete[height_sort_idx, 0], pixels_to_delete[height_sort_idx, 1]] = (
                0  # remove lowest height_bias%
            )

        if len(pixels_to_delete) == 0:
            self.skeleton_converged = True

    def _delete_pixel_subit1(self, point: list) -> bool:
        """
        Check whether a single point (P1) should be deleted based on its local binary environment.

        (a) 2 ≤ B(P1) ≤ 6, where B(P1) is the number of non-zero neighbours of P1.
        (b) A(P1) = 1, where A(P1) is the # of 01's around P1.
        (C) P2 * P4 * P6 = 0
        (d) P4 * P6 * P8 = 0

        Parameters
        ----------
        point : list
            List of [x, y] coordinate positions.

        Returns
        -------
        bool
            Indicates whether to delete depending on whether the surrounding points have met the criteria of the binary
            thin a, b returncount, c and d checks below.
        """
        self.p7, self.p8, self.p9, self.p6, self.p2, self.p5, self.p4, self.p3 = self.get_local_pixels_binary(
            self.mask, point[0], point[1]
        )
        return (
            self._binary_thin_check_a()
            and self._binary_thin_check_b_returncount() == 1
            # c and d remove only north-west corner points and south-east boundary points.
            and self._binary_thin_check_c()
            and self._binary_thin_check_d()
        )

    def _delete_pixel_subit2(self, point: list) -> bool:
        """
        Check whether a single point (P1) should be deleted based on its local binary environment.

        (a) 2 ≤ B(P1) ≤ 6, where B(P1) is the number of non-zero neighbours of P1.
        (b) A(P1) = 1, where A(P1) is the # of 01's around P1.
        (c') P2 * P4 * P8 = 0
        (d') P2 * P6 * P8 = 0

        Parameters
        ----------
        point : list
            List of [x, y] coordinate positions.

        Returns
        -------
        bool
            Whether surrounding points have met the criteria of the binary thin a, b returncount, csharp and dsharp
            checks below.
        """
        self.p7, self.p8, self.p9, self.p6, self.p2, self.p5, self.p4, self.p3 = self.get_local_pixels_binary(
            self.mask, point[0], point[1]
        )
        # Add in generic code here to protect high points from being deleted
        return (
            self._binary_thin_check_a()
            and self._binary_thin_check_b_returncount() == 1
            # c' and d' remove only north-west boundary points or south-east corner points.
            and self._binary_thin_check_csharp()
            and self._binary_thin_check_dsharp()
        )

    def _binary_thin_check_a(self) -> bool:
        """
        Check the surrounding area to see if the point lies on the edge of the grain.

        Condition A protects the endpoints (which will be < 2)

        Returns
        -------
        bool
            If point lies on edge of graph and isn't an endpoint.
        """
        return 2 <= self.p2 + self.p3 + self.p4 + self.p5 + self.p6 + self.p7 + self.p8 + self.p9 <= 6

    def _binary_thin_check_b_returncount(self) -> int:
        """
        Count local area 01's in order around P1.

        ??? What does this mean?

        Returns
        -------
        int
            The number of 01's around P1.
        """
        return sum(
            [
                [self.p2, self.p3] == [0, 1],
                [self.p3, self.p4] == [0, 1],
                [self.p4, self.p5] == [0, 1],
                [self.p5, self.p6] == [0, 1],
                [self.p6, self.p7] == [0, 1],
                [self.p7, self.p8] == [0, 1],
                [self.p8, self.p9] == [0, 1],
                [self.p9, self.p2] == [0, 1],
            ]
        )

    def _binary_thin_check_c(self) -> bool:
        """
        Check if p2, p4 or p6 is 0.

        Returns
        -------
        bool
            If p2, p4 or p6 is 0.
        """
        return self.p2 * self.p4 * self.p6 == 0

    def _binary_thin_check_d(self) -> bool:
        """
        Check if p4, p6 or p8 is 0.

        Returns
        -------
        bool
            If p4, p6 or p8 is 0.
        """
        return self.p4 * self.p6 * self.p8 == 0

    def _binary_thin_check_csharp(self) -> bool:
        """
        Check if p2, p4 or p8 is 0.

        Returns
        -------
        bool
            If p2, p4 or p8 is 0.
        """
        return self.p2 * self.p4 * self.p8 == 0

    def _binary_thin_check_dsharp(self) -> bool:
        """
        Check if p2, p6 or p8 is 0.

        Returns
        -------
        bool
            If p2, p6 or p8 is 0.
        """
        return self.p2 * self.p6 * self.p8 == 0

    def final_skeletonisation_iteration(self) -> None:
        """
        Remove "hanging" pixels.

        Examples of such pixels are:
                    [0, 0, 0]               [0, 1, 0]              [0, 0, 0]
                    [0, 1, 1]               [0, 1, 1]              [0, 1, 1]
            case 1: [0, 1, 0]   or  case 2: [0, 1, 0]   or case 3: [1, 1, 0]

        This is useful for the future functions that rely on local pixel environment
        to make assessments about the overall shape/structure of traces.
        """
        remaining_coordinates = np.argwhere(self.mask).tolist()

        for x, y in remaining_coordinates:
            self.p7, self.p8, self.p9, self.p6, self.p2, self.p5, self.p4, self.p3 = self.get_local_pixels_binary(
                self.mask, x, y
            )

            # Checks for case 1 and 3 pixels
            if (
                self._binary_thin_check_b_returncount() == 2
                and self._binary_final_thin_check_a()
                and not self.binary_thin_check_diag()
            ):
                self.mask[x, y] = 0
            # Checks for case 2 pixels
            elif self._binary_thin_check_b_returncount() == 3 and self._binary_final_thin_check_b():
                self.mask[x, y] = 0

    def _binary_final_thin_check_a(self) -> bool:
        """
        Assess if local area has 4-connectivity.

        Returns
        -------
        bool
            Logical indicator of whether if any neighbours of the 4-connections have a near pixel.
        """
        return 1 in (self.p2 * self.p4, self.p4 * self.p6, self.p6 * self.p8, self.p8 * self.p2)

    def _binary_final_thin_check_b(self) -> bool:
        """
        Assess if local area 4-connectivity is connected to multiple branches.

        Returns
        -------
        bool
            Logical indicator of whether if any neighbours of the 4-connections have a near pixel.
        """
        return 1 in (
            self.p2 * self.p4 * self.p6,
            self.p4 * self.p6 * self.p8,
            self.p6 * self.p8 * self.p2,
            self.p8 * self.p2 * self.p4,
        )

    def binary_thin_check_diag(self) -> bool:
        """
        Check if opposite corner diagonals are present.

        Returns
        -------
        bool
            Whether a diagonal exists.
        """
        return 1 in (self.p7 * self.p3, self.p5 * self.p9)

    @staticmethod
    def get_local_pixels_binary(binary_map: npt.NDArray, x: int, y: int) -> npt.NDArray:
        """
        Value of pixels in the local 8-connectivity area around the coordinate (P1) described by x and y.

        P1 must not lie on the edge of the binary map.

        [[p7, p8, p9],    [[0,1,2],
         [p6, P1, p2], ->  [3,4,5], -> [0,1,2,3,5,6,7,8]
         [p5, p4, p3]]     [6,7,8]]

        delete P1 to only get local area.

        Parameters
        ----------
        binary_map : npt.NDArray
            Binary mask of image.
        x : int
            X coordinate within the binary map.
        y : int
            Y coordinate within the binary map.

        Returns
        -------
        npt.NDArray
            Flattened 8-long array describing the values in the binary map around the x,y point.
        """
        local_pixels = binary_map[x - 1 : x + 2, y - 1 : y + 2].flatten()
        return np.delete(local_pixels, 4)

    @staticmethod
    def sort_and_shuffle(arr: npt.NDArray, seed: int = 23790101) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Sort array in ascending order and shuffle the order of identical values are the same.

        Parameters
        ----------
        arr : npt.NDArray
            A flattened (1D) array.
        seed : int
            Seed for random number generator.

        Returns
        -------
        npt.NDArray
            An ascending order array where identical value orders are also shuffled.
        npt.NDArray
            An ascending order index array of above where identical value orders are also shuffled.
        """
        # Find unique values
        unique_values_r = np.unique(arr)

        rng = np.random.default_rng(seed)

        # Shuffle the order of elements with the same value
        sorted_and_shuffled_indices: list = []
        for val in unique_values_r:
            indices = np.where(arr == val)[0]
            rng.shuffle(indices)
            sorted_and_shuffled_indices.extend(indices)

        # Rearrange the sorted array according to shuffled indices
        sorted_and_shuffled_arr: list = arr[sorted_and_shuffled_indices]

        return sorted_and_shuffled_arr, sorted_and_shuffled_indices
