"""Skeletonize molecules."""

import logging
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from skimage.morphology import binary_dilation, label, medial_axis, skeletonize, thin

from topostats.logs.logs import LOGGER_NAME
from topostats.tracing.tracingfuncs import genTracingFuncs
from topostats.utils import convolve_skelly

LOGGER = logging.getLogger(LOGGER_NAME)

# pylint: disable=too-many-lines


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
        ??? Needs description.
    """  # numpydoc: ignore=PR01

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
            ??? Needs description.
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
            ??? Needs definition.

        Returns
        -------
        npt.NDArray
            Masked array reduced to a single pixel thickness.
        """
        return topostatsSkeletonize(image, mask, height_bias).do_skeletonising()


class topostatsSkeletonize:  # pylint: disable=too-many-instance-attributes
    """
    Skeletonise a binary array following Zhang's algorithm (Zhang and Suen, 1984).

    Modifications are made to the published algorithm during the removal step to remove the smallest fraction of values
    and not all of them. All operations are performed on the mask entered.

    ??? Could be clearer as to what the "smallest fraction of values and not all of them" means?

    Parameters
    ----------
    image : npt.NDArray
        Original 2D image containing the height data.
    mask : npt.NDArray
        Binary image containing the object to be skeletonised. Dimensions should match those of 'image'.
    height_bias : float
        ??? Needs definition.
    """  # numpydoc: ignore=PR01

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
            ??? Needs definition.
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
        Obtain the local binary pixel environment and assess the local height values.

        This determines whether to delete a point.

        ??? How is this decision made, something to do with height_bias but not clear ???
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
    def sort_and_shuffle(arr: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        """Sort a flat array in ascending order and shuffle the order where the array values are the same.

        Parameters
        ----------
        arr : npt.NDArray
            A flattened (1D) array.

        Returns
        -------
        npt.NDArray
            An ascending order array where identical value orders are also shuffled.
        npt.NDArray
            An ascending order index array of above where identical value orders are also shuffled.
        """
        # Find unique values
        unique_values_r = np.unique(arr)
    
        # Shuffle the order of elements with the same value
        sorted_and_shuffled_indices = []
        for val in unique_values_r:
            indices = np.where(arr == val)[0]
            np.random.shuffle(indices)
            sorted_and_shuffled_indices.extend(indices)
    
        # Rearrange the sorted array according to shuffled indices
        sorted_and_shuffled_arr = arr[sorted_and_shuffled_indices]
    
        return sorted_and_shuffled_arr, sorted_and_shuffled_indices


class pruneSkeleton:  # pylint: disable=too-few-public-methods
    """
    Class containing skeletonization pruning code from factory methods to functions dependent on the method.

    Pruning is the act of removing spurious branches commonly found when implementing skeletonization algorithms.

    Parameters
    ----------
    image : npt.NDArray
        Original image from which the skeleton derives including heights.
    skeleton : npt.NDArray
        Single-pixel-thick skeleton pertaining to features of the image.
    """

    def __init__(self, image: npt.NDArray, skeleton: npt.NDArray) -> None:
        """
        Initialise the class.

        Parameters
        ----------
        image : npt.NDArray
            Original image from which the skeleton derives including heights.
        skeleton : npt.NDArray
            Single-pixel-thick skeleton pertaining to features of the image.
        """
        self.image = image
        self.skeleton = skeleton

    def prune_skeleton(  # pylint: disable=dangerous-default-value
        self, prune_args: dict = {"pruning_method": "topostats"}  # noqa: B006
    ) -> npt.NDArray:
        """
        Pruning skeletons.

        This is a thin wrapper to the methods provided within the pruning classes below.

        Parameters
        ----------
        prune_args : dict
            Method to use, default is 'topostats'.

        Returns
        -------
        npt.NDArray
            An array of the skeleton with spurious branching artefacts removed.
        """
        return self._prune_method(prune_args)

    def _prune_method(self, prune_args: str = None) -> Callable:
        """
        Determine which skeletonize method to use.

        Parameters
        ----------
        prune_args : str
            Method to use for skeletonizing, methods are 'topostats' other options are 'conv'.

        Returns
        -------
        Callable
            Returns the function appropriate for the required skeletonizing method.

        Raises
        ------
        ValueError
            Invalid method passed.
        """
        method = prune_args.pop("pruning_method")
        if method == "topostats":
            return self._prune_topostats(self.image, self.skeleton, prune_args)
        if method == "conv":
            return self._prune_conv(self.image, self.skeleton, prune_args)
        # I've read about a "Discrete Skeleton Evolultion" (DSE) method that might be useful
        raise ValueError(method)

    @staticmethod
    def _prune_topostats(image: npt.NDArray, skeleton: npt.NDArray, prune_args: dict) -> npt.NDArray:
        """
        Prune using the original TopoStats method.

        This is a modified version of the pubhlished Zhang method.

        Parameters
        ----------
        image : npt.NDArray
            Image used to find skeleton, may be original heights or binary mask.
        skeleton : npt.NDArray
            Binary mask of the skeleton.
        prune_args : dict
            Dictionary of pruning arguments. ??? Needs expanding on what the valid arguments are.

        Returns
        -------
        npt.NDArray
            The skeleton with spurious branches removed.
        """
        return topostatsPrune(image, skeleton, **prune_args).prune_all_skeletons()

    @staticmethod
    def _prune_conv(image: npt.NDArray, skeleton: npt.NDArray, prune_args: dict) -> npt.NDArray:
        """
        Prune using a convolutional method.

        Parameters
        ----------
        image : npt.NDArray
            Image used to find skeleton, may be original heights or binary mask.
        skeleton : npt.NDArray
            Binary array containing skeleton.
        prune_args : dict
            Dictionary of pruning arguments for convPrune class. ??? Needs expanding on what the valid arguments are.

        Returns
        -------
        npt.NDArray
            The skeleton with spurious branches removed.
        """
        return convPrune(image, skeleton, **prune_args).prune_all_skeletons()


class topostatsPrune:  # pylint: disable=too-few-public-methods
    """
    Prune spurious skeletal branches based on their length and/or height.

    Contains all the functions used in the original TopoStats pruning code written by Joe Betton.

    Parameters
    ----------
    image : npt.NDArray
        Original image.
    skeleton : npt.NDArray
        Skeleton to be pruned.
    max_length : float
        Maximum length of the branch to prune in nanometres (nm).
    height_threshold : float
        Absolute height value to remove branches below in nanometres (nm).
    method_values : str
        Method for obtaining the height thresholding values. Options are 'min' (minimum value of the branch),
        'median' (median value of the branch) or 'mid' (ordered branch middle coordinate value).
    method_outlier : str
        Method for pruning brancvhes based on height. Options are 'abs' (below absolute value), 'mean_abs' (below the
        skeleton mean - absolute threshold) or 'iqr' (below 1.5 * inter-quartile range).
    """  # numpydoc: ignore=PR01

    def __init__(
        self,
        image: npt.NDArray,
        skeleton: npt.NDArray,
        max_length: float = None,
        height_threshold: float = None,
        method_values: str = None,
        method_outlier: str = None,
    ) -> None:
        """
        Initialise the class.

        Parameters
        ----------
        image : npt.NDArray
            Original image.
        skeleton : npt.NDArray
            Skeleton to be pruned.
        max_length : float
            Maximum length of the branch to prune in nanometres (nm).
        height_threshold : float
            Absolute height value to remove branches below in nanometres (nm).
        method_values : str
            Method for obtaining the height thresholding values. Options are 'min' (minimum value of the branch),
            'median' (median value of the branch) or 'mid' (ordered branch middle coordinate value).
        method_outlier : str
            Method for pruning brancvhes based on height. Options are 'abs' (below absolute value), 'mean_abs' (below
            the skeleton mean - absolute threshold) or 'iqr' (below 1.5 * inter-quartile range).
        """
        self.image = image
        self.skeleton = skeleton.copy()
        self.max_length = max_length
        self.height_threshold = height_threshold
        self.method_values = method_values
        self.method_outlier = method_outlier

    def prune_all_skeletons(self) -> npt.NDArray:
        """
        Prune all skeletons.

        Returns
        -------
        npt.NDArray
            A single mask with all pruned skeletons.
        """
        pruned_skeleton_mask = np.zeros_like(self.skeleton)
        labeled_skel = label(self.skeleton)
        for i in range(1, labeled_skel.max() + 1):
            single_skeleton = np.where(labeled_skel == i, 1, 0)
            if self.max_length is not None:
                single_skeleton = self._prune_by_length(single_skeleton, max_length=self.max_length)
            if self.height_threshold is not None:
                single_skeleton = heightPruning(
                    self.image,
                    single_skeleton,
                    height_threshold=self.height_threshold,
                    method_values=self.method_values,
                    method_outlier=self.method_outlier,
                ).remove_bridges()
            # skeletonise to remove nibs
            pruned_skeleton_mask += getSkeleton(self.image, single_skeleton, method="zhang").get_skeleton()
        return pruned_skeleton_mask

    def _prune_by_length(  # pylint: disable=too-many-locals  # noqa: C901
        self, single_skeleton: npt.NDArray, max_length: float | int = -1
    ) -> npt.NDArray:
        """
        Remove hanging branches from a skeleton.

        This is an iterative process as these are a persistent problem in the overall tracing process.

        Parameters
        ----------
        single_skeleton : npt.NDArray
            Binary array of the skeleton.
        max_length : float | int
            Maximum length of the branch to prune in nanometers (nm). Default is -1 which calculates a value that is 15%
            of the total skeleton length.

        Returns
        -------
        npt.NDArray
            Pruned skeleton as binary array.
        """
        pruning = True
        while pruning:
            single_skeleton = rm_nibs(single_skeleton)
            number_of_branches = 0
            coordinates = np.argwhere(single_skeleton == 1).tolist()

            # The branches are typically short so if a branch is longer than
            #  0.15 * total points, its assumed to be part of the real data
            max_branch_length = max_length if max_length != -1 else int(len(coordinates) * 0.15)
            # first check to find all the end coordinates in the trace
            potential_branch_ends = self._find_branch_ends(coordinates)

            # Now check if its a branch - and if it is delete it
            for branch_x, branch_y in potential_branch_ends:
                branch_coordinates = [[branch_x, branch_y]]
                branch_continues = True
                temp_coordinates = coordinates[:]
                temp_coordinates.pop(temp_coordinates.index([branch_x, branch_y]))

                while branch_continues:
                    no_of_neighbours, neighbours = genTracingFuncs.count_and_get_neighbours(
                        branch_x, branch_y, temp_coordinates
                    )

                    # If branch continues
                    if no_of_neighbours == 1:
                        branch_x, branch_y = neighbours[0]
                        branch_coordinates.append([branch_x, branch_y])
                        temp_coordinates.pop(temp_coordinates.index([branch_x, branch_y]))

                    # If the branch reaches the edge of the main trace
                    elif no_of_neighbours > 1:
                        branch_coordinates.pop(branch_coordinates.index([branch_x, branch_y]))
                        branch_continues = False
                        is_branch = True

                    # Weird case that happens sometimes (would this be linear mols?)
                    elif no_of_neighbours == 0:
                        is_branch = True
                        branch_continues = False

                    # why not `and branch_continues`?
                    if len(branch_coordinates) > max_branch_length:
                        branch_continues = False
                        is_branch = False
                #
                if is_branch:
                    number_of_branches += 1
                    for x, y in branch_coordinates:
                        single_skeleton[x, y] = 0

            if number_of_branches == 0:
                pruning = False

        return single_skeleton

    @staticmethod
    def _find_branch_ends(coordinates: list) -> list:
        """
        Identify branch ends.

        This is achieved by iterating through the coordinates and assessing the local pixel area. Ends have only one
        adjacent pixel.

        Parameters
        ----------
        coordinates : list
            List of x, y coordinates of a branch.

        Returns
        -------
        list
            List of x, y coordinates of the branch ends.
        """
        potential_branch_ends = []

        # Most of the branch ends are just points with one neighbour
        for x, y in coordinates:
            if genTracingFuncs.count_and_get_neighbours(x, y, coordinates)[0] == 1:
                potential_branch_ends.append([x, y])
        return potential_branch_ends


class convPrune:  # pylint: disable=too-few-public-methods
    """
    Prune spurious branches based on their length and/or height using convolutions.

    Parameters
    ----------
    image : npt.NDArray
        The original data, with heights, to aid branch removal.
    skeleton : npt.NDArray
        Skeleton from which unwanted branches are to be removed.
    max_length : float
        Maximum length of branches to prune in nanometres (nm).
    height_threshold : float
        Absolute height value to remove granches below in nanometres (nm). Determined by the value of
        'method_values'.
    method_values : str
        Method for obtaining the height thresholding values. Options are 'min' (minimum value of branch), 'median'
        (median value of branch), 'mid' (ordered branch middle coordinate value).
    method_outlier : str
        Method to prune branches based on height. Options are 'abs' (below absolute value), 'mean_abs' (below the
        skeleton mean), or 'iqr' (below 1.5 * inter-quartile range).
    """  # numpydoc: ignore=PR01

    def __init__(
        self,
        image: npt.NDArray,
        skeleton: npt.NDArray,
        max_length: float = None,
        height_threshold: float = None,
        method_values: str = None,
        method_outlier: str = None,
    ) -> None:
        """
        Initialise the class.

        Parameters
        ----------
        image : npt.NDArray
            The original data, with heights, to aid branch removal.
        skeleton : npt.NDArray
            Skeleton from which unwanted branches are to be removed.
        max_length : float
            Maximum length of branches to prune in nanometres (nm).
        height_threshold : float
            Absolute height value to remove granches below in nanometres (nm). Determined by the value of
            'method_values'.
        method_values : str
            Method for obtaining the height thresholding values. Options are 'min' (minimum value of branch), 'median'
            (median value of branch), 'mid' (ordered branch middle coordinate value).
        method_outlier : str
            Method to prune branches based on height. Options are 'abs' (below absolute value), 'mean_abs' (below the
            skeleton mean), or 'iqr' (below 1.5 * inter-quartile range).
        """
        self.image = image
        self.skeleton = skeleton.copy()
        self.max_length = max_length
        self.height_threshold = height_threshold
        self.method_values = method_values
        self.method_outlier = method_outlier

    def prune_all_skeletons(self) -> npt.NDArray:
        """
        Prune all skeletons.

        Returns
        -------
        npt.NDArray
            A single mask with all pruned skeletons.
        """
        pruned_skeleton_mask = np.zeros_like(self.skeleton)
        labeled_skel = label(self.skeleton)
        for i in range(1, labeled_skel.max() + 1):
            single_skeleton = np.where(labeled_skel == i, 1, 0)
            if self.max_length is not None:
                single_skeleton = self._prune_by_length(single_skeleton, max_length=self.max_length)
            if self.height_threshold is not None:
                single_skeleton = heightPruning(
                    self.image,
                    single_skeleton,
                    height_threshold=self.height_threshold,
                    method_values=self.method_values,
                    method_outlier=self.method_outlier,
                ).remove_bridges()
            # skeletonise to remove nibs
            pruned_skeleton_mask += getSkeleton(self.image, single_skeleton, method="zhang").get_skeleton()

        return pruned_skeleton_mask

    def _prune_by_length(self, single_skeleton: npt.NDArray, max_length: float | int = -1) -> npt.NDArray:
        """
        Remove the hanging branches from a single skeleton via local-area convoluions.

        Parameters
        ----------
        single_skeleton : npt.NDArray
            Binary array containing a single skeleton.
        max_length : float | int
            Maximum length of branch to prune in nanometres (nm). Default is '-1' which sets to the maximum
            branch length to be 15% of the total skeleton length.

        Returns
        -------
        npt.NDArray
            Pruned skeleton.
        """
        total_points = self.skeleton.size
        single_skeleton = self.skeleton.copy()
        conv_skelly = convolve_skelly(self.skeleton)
        nodeless = self.skeleton.copy()
        nodeless[conv_skelly == 3] = 0

        # The branches are typically short so if a branch is longer than
        #  0.15 * total points, its assumed to be part of the real data
        max_branch_length = max_length if max_length != -1 else int(len(total_points) * 0.15)

        # iterate through branches
        nodeless_labels = label(nodeless)
        for i in range(1, nodeless_labels.max() + 1):
            vals = conv_skelly[nodeless_labels == i]
            # check if there is an endpoint and length is below expected
            if (vals == 2).any() and (vals.size < max_branch_length):
                single_skeleton[nodeless_labels == i] = 0

        return single_skeleton


class heightPruning:
    """
    Pruning of branches based on height.

    Parameters
    ----------
    image : npt.NDArray
        Original image, typically the height data.
    skeleton : npt.NDArray
        Skeleton to prune branches from.
    max_length : float
        Maximum length of the branch to prune in nanometres (nm).
    height_threshold : float
        Absolute height value to remove branches below in nanometers (nm).
    method_values : str
        Method of obtaining the height thresholding values. Options are 'min' (minimum value of the branch),
        'median' (median value of the branch) or 'mid' (ordered branch middle coordinate value).
    method_outlier : str
        Method to prune branches based on height. Options are 'abs' (below absolute value), 'mean_abs' (below the
        skeleton mean - absolute threshold) or 'iqr' (below 1.5 * inter-quartile range).
    """  # numpydoc: ignore=PR01

    def __init__(
        self,
        image: npt.NDArray,
        skeleton: npt.NDArray,
        max_length: float = None,
        height_threshold: float = None,
        method_values: str = None,
        method_outlier: str = None,
    ) -> None:
        """
        Initialise the class.

        Parameters
        ----------
        image : npt.NDArray
            Original image, typically the height data.
        skeleton : npt.NDArray
            Skeleton to prune branches from.
        max_length : float
            Maximum length of the branch to prune in nanometres (nm).
        height_threshold : float
            Absolute height value to remove branches below in nanometers (nm).
        method_values : str
            Method of obtaining the height thresholding values. Options are 'min' (minimum value of the branch),
            'median' (median value of the branch) or 'mid' (ordered branch middle coordinate value).
        method_outlier : str
            Method to prune branches based on height. Options are 'abs' (below absolute value), 'mean_abs' (below the
            skeleton mean - absolute threshold) or 'iqr' (below 1.5 * inter-quartile range).
        """
        self.image = image
        self.skeleton = skeleton
        self.max_length = max_length
        self.height_threshold = (height_threshold,)
        self.method_values = (method_values,)
        self.method_outlier = (method_outlier,)

    @staticmethod
    def _get_branch_mins(image: npt.NDArray, segments: npt.NDArray) -> npt.NDArray:
        """
        Collect the minimum height value of each labeled branch.

        Parameters
        ----------
        image : npt.NDArray
            The original image data to help with branch removal.
        segments : npt.NDArray
            Integer labeled array matching the dimensions of the image.

        Returns
        -------
        npt.NDArray
            Array of minimum values of each branch index -1.
        """
        branch_min_heights = [np.min(image[segments == i]) for i in range(1, segments.max() + 1)]
        return np.array(branch_min_heights)

    @staticmethod
    def _get_branch_medians(image: npt.NDArray, segments: npt.NDArray) -> npt.NDArray:
        """
        Collect the median height value of each labeled branch.

        Parameters
        ----------
        image : npt.NDArray
            The original image data to help with branch removal.
        segments : npt.NDArray
            Integer labeled array matching the dimensions of the image.

        Returns
        -------
        npt.NDArray
            Array of median values of each branch index -1.
        """
        branch_median_heights = [np.median(image[segments == i]) for i in range(1, segments.max() + 1)]
        return np.array(branch_median_heights)

    @staticmethod
    def _get_branch_middles(image: npt.NDArray, segments: npt.NDArray) -> npt.NDArray:
        """
        Collect the positionally ordered middle height value of each labeled branch.

        Parameters
        ----------
        image : npt.NDArray
            The original image data to help with branch removal.
        segments : npt.NDArray
            Integer labeled array matching the dimensions of the image.

        Returns
        -------
        npt.NDArray
            Array of middle values of each branch.
        """
        branch_middles = np.zeros(segments.max())
        for i in range(1, segments.max() + 1):
            segment = np.where(segments == i, 1, 0)
            if segment.sum() > 2:
                # sometimes start is not found ?
                start = np.argwhere(convolve_skelly(segment) == 2)[0]
                ordered_coords = order_branch_from_start(segment, start)
                mid_coord = ordered_coords[len(ordered_coords) // 2]
            else:
                mid_coord = np.argwhere(segment == 1)[0]
            branch_middles[i - 1] += image[mid_coord[0], mid_coord[1]]
        return branch_middles

    @staticmethod
    def _get_abs_thresh_idx(height_values: npt.NDArray, threshold: float | int) -> npt.NDArray:
        """
        Identify indices on branches whose height values are less than a given threshold.

        Parameters
        ----------
        height_values : npt.NDArray
            Array of each branches heights.
        threshold : float | int
            Threshold for heights.

        Returns
        -------
        npt.NDArray
            Branch indices which are less than threshold.
        """
        return np.asarray(np.where(height_values < threshold))[0] + 1

    @staticmethod
    def _get_mean_abs_thresh_idx(
        height_values: npt.NDArray, threshold: float | int, image: npt.NDArray, skeleton: npt.NDArray
    ) -> npt.NDArray:
        """
        Identify indices on branch whose height values are less than mean skeleton height - absolute threshold.

        For DNA a threshold of 0.85nm (the depth of the major groove) would ideally remove all segments whose lowest
        point is < mean(height) - 0.85nm, i.e. 1.15nm.

        Parameters
        ----------
        height_values : npt.NDArray
            Array of branches heights.
        threshold : float | int
            Threshold to be subtracted from mean heights.
        image : npt.NDArray
            Original image of heights.
        skeleton : npt.NDArray
            Binary array of skeleton used to identify heights from original image to use.

        Returns
        -------
        npt.NDArray
            Branch indices which are less than mean(height) - threshold.
        """
        avg = image[skeleton == 1].mean()
        return np.asarray(np.where(np.asarray(height_values) < (avg - threshold)))[0] + 1

    @staticmethod
    def _get_iqr_thresh_idx(image: npt.NDArray, segments: npt.NDArray) -> npt.NDArray:
        """
        Identify indices on branches whose height values are less than 1.5 x interquartile range of all heights.

        Parameters
        ----------
        image : npt.NDArray
            Original image with heights.
        segments : npt.NDArray
            Array of skeleton branches.

        Returns
        -------
        npt.NDArray
            Branch indices where heights are < 1.5 * inter-quartile range.
        """
        coords = np.argwhere(segments != 0)
        heights = image[coords[:, 0], coords[:, 1]]  # all skel heights else distribution isn't representitive
        q75, q25 = np.percentile(heights, [75, 25])
        iqr = q75 - q25
        threshold = q25 - 1.5 * iqr
        low_coords = coords[heights < threshold]
        low_segment_idxs = []
        low_segment_mins = []
        # iterate through each branch segment and see if any low_coords are in a branch
        for segment_num in range(1, segments.max() + 1):
            segment_coords = np.argwhere(segments == segment_num)
            for low_coord in low_coords:
                place = np.isin(segment_coords, low_coord).all(axis=1)
                if place.any():
                    low_segment_idxs.append(segment_num)
                    low_segment_mins.append(image[segments == segment_num].min())
                    break
        return np.array(low_segment_idxs)[np.argsort(low_segment_mins)]  # sort in order of ascending mins

    @staticmethod
    def check_skeleton_one_object(skeleton: npt.NDArray) -> bool:
        """
        Ensure that the skeleton hasn't been broken up upon removing a segment.

        Parameters
        ----------
        skeleton : npt.NDArray
            2D single pixel thick array.

        Returns
        -------
        bool
            True or False depending on whether there is 1 or !1 objects.
        """
        skeleton = np.where(skeleton != 0, 1, 0)
        return label(skeleton).max() == 1

    def remove_bridges(self) -> npt.NDArray:
        """
        Identify branches which cross the skeleton in places they shouldn't.

        This occurs due to poor thresholding and holes in the mask, creating false "bridges" which misrepresent the
        skeleton of the molecule.

        Returns
        -------
        npt.NDArray
            A the skeleton with branches removed by height.
        """
        # might need to check that the image *with nodes* is returned
        skeleton_rtn = self.skeleton.copy()
        conv = convolve_skelly(self.skeleton)
        nodeless = np.where(conv == 3, 0, conv)
        segments = label(nodeless)
        # Obtain the height of each branch via the min | median | mid methods
        if self.method_values == "min":
            height_values = self._get_branch_mins(self.image, segments)
        elif self.method_values == "median":
            height_values = self._get_branch_medians(self.image, segments)
        elif self.method_values == "mid":
            height_values = self._get_branch_middles(self.image, segments)

        # threshold heights to obtain indexes of branches to be removed
        if self.method_outlier == "abs":
            idxs = self._get_abs_thresh_idx(height_values, self.height_threshold)
        elif self.method_outlier == "mean_abs":
            idxs = self._get_mean_abs_thresh_idx(height_values, self.height_threshold, self.image, self.skeleton)
        elif self.method_outlier == "iqr":
            idxs = self._get_iqr_thresh_idx(self.image, segments)

        # Only remove the bridge if the skeleton remains a single object.
        for i in idxs:
            temp_skel = skeleton_rtn.copy()
            temp_skel[segments == i] = 0
            if self.check_skeleton_one_object(temp_skel):
                skeleton_rtn[segments == i] = 0

        return skeleton_rtn


def order_branch_from_start(nodeless: npt.NDArray, start: list, max_length: float = np.inf) -> npt.NDArray:
    """
    Take a linear branch and orders its coordinates starting from a specific endpoint.

    Parameters
    ----------
    nodeless : npt.NDArray
        _description_.
    start : list
        _description_.
    max_length : float, optional
        _description_, by default np.inf.

    Returns
    -------
    npt.NDArray
        ??? Needs a description.
    """
    dist = 0
    # add starting point to ordered array
    ordered = []
    ordered.append(start)
    nodeless[start[0], start[1]] = 0  # remove from array

    # iterate to order the rest of the points
    current_point = ordered[-1]  # get last point
    area, _ = local_area_sum(nodeless, current_point)  # look at local area
    local_next_point = np.argwhere(
        area.reshape(
            (
                3,
                3,
            )
        )
        == 1
    ) - (1, 1)
    dist += np.sqrt(2) if abs(local_next_point).sum() > 1 else 1

    while len(local_next_point) != 0 and dist <= max_length:
        next_point = (current_point + local_next_point)[0]
        # find where to go next
        ordered.append(next_point)
        nodeless[next_point[0], next_point[1]] = 0  # set value to zero
        current_point = ordered[-1]  # get last point
        area, _ = local_area_sum(nodeless, current_point)  # look at local area
        local_next_point = np.argwhere(
            area.reshape(
                (
                    3,
                    3,
                )
            )
            == 1
        ) - (1, 1)
        dist += np.sqrt(2) if abs(local_next_point).sum() > 1 else 1

    return np.array(ordered)


def rm_nibs(skeleton):  # pylint: disable=too-many-locals
    """
    Remove single pixel branches (nibs) not identified by nearest neighbour algorithms as there may be >2 neighbours.

    Parameters
    ----------
    skeleton : npt.NDArray
        A single pixel thick trace.

    Returns
    -------
    npt.NDArray
        A skeleton with single pixel nibs removed.
    """
    conv_skel = convolve_skelly(skeleton)
    nodes = np.where(conv_skel == 3, 1, 0)
    labeled_nodes = label(nodes)
    nodeless = np.where((conv_skel == 1) | (conv_skel == 2), 1, 0)
    labeled_nodeless = label(nodeless)
    size_1_idxs = []

    for node_num in range(1, labeled_nodes.max() + 1):
        node = np.where(labeled_nodes == node_num, 1, 0)
        dil = binary_dilation(node, footprint=np.ones((3, 3)))
        minus = np.where(dil != node, 1, 0)

        idxs = labeled_nodeless[minus == 1]
        idxs = idxs[idxs != 0]
        for nodeless_num in np.unique(idxs):
            # if all of the branch is in surrounding node area
            branch_size = (labeled_nodeless == nodeless_num).sum()
            branch_idx_in_surr_area = (idxs == nodeless_num).sum()
            if branch_size == branch_idx_in_surr_area:
                size_1_idxs.append(nodeless_num)

    unique, counts = np.unique(np.array(size_1_idxs), return_counts=True)

    for k, count in enumerate(counts):
        if count == 1:
            skeleton[labeled_nodeless == unique[k]] = 0

    return skeleton


def local_area_sum(binary_map: npt.NDArray, point: list | tuple | npt.NDArray) -> tuple:
    """
    Evaluate the local area around a point in a binary map.

    Parameters
    ----------
    binary_map : npt.NDArray
        Binary array of image.
    point : list | tuple | npt.NDArray
        Coordinates of a point within the binary_map.

    Returns
    -------
    tuple
        Tuple consisting of an array values of the local coordinates around the point and the number of neighbours
        around the point.
    """
    x, y = point
    local_pixels = binary_map[x - 1 : x + 2, y - 1 : y + 2].flatten()
    local_pixels[4] = 0  # ensure centre is 0
    return (local_pixels, local_pixels.sum())
