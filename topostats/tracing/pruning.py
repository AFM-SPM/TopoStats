"""Prune branches from skeletons."""

import logging
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

# from skimage.morphology import binary_dilation, label
from skimage import morphology

from topostats.logs.logs import LOGGER_NAME
from topostats.tracing.skeletonize import getSkeleton
from topostats.tracing.tracingfuncs import coord_dist, genTracingFuncs, order_branch
from topostats.utils import convolve_skeleton

LOGGER = logging.getLogger(LOGGER_NAME)


def prune_skeleton(image: npt.NDArray, skeleton: npt.NDArray, pixel_to_nm_scaling: float, **kwargs) -> npt.NDArray:
    """
    Pruning skeletons using different pruning methods.

    This is a thin wrapper to the methods provided within the pruning classes below.

    Parameters
    ----------
    image : npt.NDArray
        Original image as 2D numpy array.
    skeleton : npt.NDArray
        Skeleton to be pruned.
    pixel_to_nm_scaling : float
        The pixel to nm scaling for pruning by length.
    **kwargs
        Pruning options passed to the respective method.

    Returns
    -------
    npt.NDArray
        An array of the skeleton with spurious branching artefacts removed.
    """
    if image.shape != skeleton.shape:
        raise AttributeError("Error image and skeleton are not the same size.")
    return _prune_method(image, skeleton, pixel_to_nm_scaling, **kwargs)


def _prune_method(image: npt.NDArray, skeleton: npt.NDArray, pixel_to_nm_scaling: float, **kwargs) -> Callable:
    """
    Determine which skeletonize method to use.

    Parameters
    ----------
    image : npt.NDArray
        Original image as 2D numpy array.
    skeleton : npt.NDArray
        Skeleton to be pruned.
    pixel_to_nm_scaling : float
        The pixel to nm scaling for pruning by length.
    **kwargs
        Pruning options passed to the respective method.

    Returns
    -------
    Callable
        Returns the function appropriate for the required skeletonizing method.

    Raises
    ------
    ValueError
        Invalid method passed.
    """
    method = kwargs.pop("method")
    if method == "topostats":
        return _prune_topostats(image, skeleton, pixel_to_nm_scaling, **kwargs)
    # @maxgamill-sheffield I've read about a "Discrete Skeleton Evolultion" (DSE) method that might be useful
    # @ns-rse (2024-06-04) : https://en.wikipedia.org/wiki/Discrete_skeleton_evolution
    #                        https://link.springer.com/chapter/10.1007/978-3-540-74198-5_28
    #                        https://dl.acm.org/doi/10.5555/1780074.1780108
    #                        Python implementation : https://github.com/originlake/DSE-skeleton-pruning
    raise ValueError(f"Invalid pruning method provided ({method}) please use one of 'topostats'.")


def _prune_topostats(img: npt.NDArray, skeleton: npt.NDArray, pixel_to_nm_scaling: float, **kwargs) -> npt.NDArray:
    """
    Prune using the original TopoStats method.

    This is a modified version of the pubhlished Zhang method.

    Parameters
    ----------
    img : npt.NDArray
        Image used to find skeleton, may be original heights or binary mask.
    skeleton : npt.NDArray
        Binary mask of the skeleton.
    pixel_to_nm_scaling : float
        The pixel to nm scaling for pruning by length.
    **kwargs
        Pruning options passed to the topostatsPrune class.

    Returns
    -------
    npt.NDArray
        The skeleton with spurious branches removed.
    """
    return topostatsPrune(img, skeleton, pixel_to_nm_scaling, **kwargs).prune_skeleton()


# class pruneSkeleton:  pylint: disable=too-few-public-methods
#     """
#     Class containing skeletonization pruning code from factory methods to functions dependent on the method.

#     Pruning is the act of removing spurious branches commonly found when implementing skeletonization algorithms.

#     Parameters
#     ----------
#     image : npt.NDArray
#         Original image from which the skeleton derives including heights.
#     skeleton : npt.NDArray
#         Single-pixel-thick skeleton pertaining to features of the image.
#     """

#     def __init__(self, image: npt.NDArray, skeleton: npt.NDArray) -> None:
#         """
#         Initialise the class.

#         Parameters
#         ----------
#         image : npt.NDArray
#             Original image from which the skeleton derives including heights.
#         skeleton : npt.NDArray
#             Single-pixel-thick skeleton pertaining to features of the image.
#         """
#         self.image = image
#         self.skeleton = skeleton

#     def prune_skeleton(  pylint: disable=dangerous-default-value
#         self,
#         prune_args: dict = {"pruning_method": "topostats"},  noqa: B006
#     ) -> npt.NDArray:
#         """
#         Pruning skeletons.

#         This is a thin wrapper to the methods provided within the pruning classes below.

#         Parameters
#         ----------
#         prune_args : dict
#             Method to use, default is 'topostats'.

#         Returns
#         -------
#         npt.NDArray
#             An array of the skeleton with spurious branching artefacts removed.
#         """
#         return self._prune_method(prune_args)

#     def _prune_method(self, prune_args: str = None) -> Callable:
#         """
#         Determine which skeletonize method to use.

#         Parameters
#         ----------
#         prune_args : str
#             Method to use for skeletonizing, methods are 'topostats' other options are 'conv'.

#         Returns
#         -------
#         Callable
#             Returns the function appropriate for the required skeletonizing method.

#         Raises
#         ------
#         ValueError
#             Invalid method passed.
#         """
#         method = prune_args.pop("pruning_method")
#         if method == "topostats":
#             return self._prune_topostats(self.image, self.skeleton, prune_args)
#         I've read about a "Discrete Skeleton Evolultion" (DSE) method that might be useful
#         @ns-rse (2024-06-04) : Citation or link?
#         raise ValueError(method)

#     @staticmethod
#     def _prune_topostats(img: npt.NDArray, skeleton: npt.NDArray, prune_args: dict) -> npt.NDArray:
#         """
#         Prune using the original TopoStats method.

#         This is a modified version of the pubhlished Zhang method.

#         Parameters
#         ----------
#         img : npt.NDArray
#             Image used to find skeleton, may be original heights or binary mask.
#         skeleton : npt.NDArray
#             Binary mask of the skeleton.
#         prune_args : dict
#             Dictionary of pruning arguments. ??? Needs expanding on what the valid arguments are.

#         Returns
#         -------
#         npt.NDArray
#             The skeleton with spurious branches removed.
#         """
#         return topostatsPrune(img, skeleton, **prune_args).prune_skeleton()


# Might be worth renaming this to reflect what it does which is prune by length and height
# pylint: disable=too-many-instance-attributes
class topostatsPrune:
    """
    Prune spurious skeletal branches based on their length and/or height.

    Contains all the functions used in the original TopoStats pruning code written by Joe Betton.

    Parameters
    ----------
    img : npt.NDArray
        Original image.
    skeleton : npt.NDArray
        Skeleton to be pruned.
    pixel_to_nm_scaling : float
        The pixel to nm scaling for pruning by length.
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
    only_height_prune_endpoints : bool
        Whether to only prune endpoints by height, or all skeleton segments.
    """

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        img: npt.NDArray,
        skeleton: npt.NDArray,
        pixel_to_nm_scaling: float,
        max_length: float = None,
        height_threshold: float = None,
        method_values: str = None,
        method_outlier: str = None,
        only_height_prune_endpoints: bool = True,
    ) -> None:
        """
        Initialise the class.

        Parameters
        ----------
        img : npt.NDArray
            Original image.
        skeleton : npt.NDArray
            Skeleton to be pruned.
        pixel_to_nm_scaling : float
            The pixel to nm scaling for pruning by length.
        max_length : float
            Maximum length of the branch to prune in nanometres (nm).
        height_threshold : float
            Absolute height value to remove branches below in nanometres (nm).
        method_values : str
            Method for obtaining the height thresholding values. Options are 'min' (minimum value of the branch),
            'median' (median value of the branch) or 'mid' (ordered branch middle coordinate value).
        method_outlier : str
            Method for pruning branches based on height. Options are 'abs' (below absolute value), 'mean_abs' (below
            the skeleton mean - absolute threshold) or 'iqr' (below 1.5 * inter-quartile range).
        only_height_prune_endpoints : bool
            Whether to only prune endpoints by height, or all skeleton segments.
        """
        self.img = img
        self.skeleton = skeleton.copy()
        self.pixel_to_nm_scaling = pixel_to_nm_scaling
        self.max_length = max_length
        self.height_threshold = height_threshold
        self.method_values = method_values
        self.method_outlier = method_outlier
        self.only_height_prune_endpoints = only_height_prune_endpoints

    # Diverges from the change in layout to apply skeletonisation/pruning/tracing to individual grains and then process
    # all grains in an image (possibly in parallel).
    def prune_skeleton(self) -> npt.NDArray:
        """
        Prune skeleton by length and/or height.

        If the class was initialised with both `max_length is not None` an d `height_threshold is not None` then length
        based pruning is performed prior to height based pruning.

        Returns
        -------
        npt.NDArray
            A pruned skeleton.
        """
        pruned_skeleton_mask = np.zeros_like(self.skeleton, dtype=np.uint8)
        labeled_skel = morphology.label(self.skeleton)
        for i in range(1, labeled_skel.max() + 1):
            single_skeleton = np.where(labeled_skel == i, 1, 0)
            if self.max_length is not None:
                LOGGER.debug(f": pruning.py : Pruning by length < {self.max_length}.")
                single_skeleton = self._prune_by_length(single_skeleton, max_length=self.max_length)
            if self.height_threshold is not None:
                LOGGER.debug(": pruning.py : Pruning by height.")
                single_skeleton = heightPruning(
                    self.img,
                    single_skeleton,
                    height_threshold=self.height_threshold,
                    method_values=self.method_values,
                    method_outlier=self.method_outlier,
                    only_height_prune_endpoints=self.only_height_prune_endpoints,
                ).skeleton_pruned
            # skeletonise to remove nibs
            # Discovered this caused an error when writing tests...
            #
            #  numpy.core._exceptions._UFuncOutputCastingError: Cannot cast ufunc 'add' output from dtype('int8') to
            #  dtype('bool') with casting...
            # pruned_skeleton_mask += getSkeleton(self.img, single_skeleton, method="zhang").get_skeleton()
            pruned_skeleton = getSkeleton(self.img, single_skeleton, method="zhang").get_skeleton()
            pruned_skeleton_mask += pruned_skeleton.astype(dtype=np.uint8)
        return pruned_skeleton_mask

    def _prune_by_length(  # pylint: disable=too-many-locals  # noqa: C901
        self, single_skeleton: npt.NDArray, max_length: float
    ) -> npt.NDArray:
        """
        Remove hanging branches from a skeleton by their length.

        This is an iterative process as these are a persistent problem in the overall tracing process.

        Parameters
        ----------
        single_skeleton : npt.NDArray
            Binary array of the skeleton.
        max_length : float
            Maximum length of the branch to prune in nanometers (nm).

        Returns
        -------
        npt.NDArray
            Pruned skeleton as binary array.
        """
        # get segments via convolution and removing junctions
        conv_skeleton = convolve_skeleton(single_skeleton)
        conv_skeleton[conv_skeleton == 3] = 0
        labeled_segments = morphology.label(conv_skeleton.astype(bool))

        for segment_idx in range(1, labeled_segments.max() + 1):
            # get single segment with endpoints==2
            segment = np.where(labeled_segments == segment_idx, conv_skeleton, 0)
            # get segment length
            ordered_coords = order_branch(np.where(segment != 0, 1, 0), [0, 0])
            segment_length = coord_dist(ordered_coords, self.pixel_to_nm_scaling)[-1]
            # check if endpoint
            if 2 in segment and segment_length < max_length:
                # prune
                single_skeleton[labeled_segments == segment_idx] = 0

        return rm_nibs(single_skeleton)

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
        branch_ends = []

        # Most of the branch ends are just points with one neighbour
        for x, y in coordinates:
            if genTracingFuncs.count_and_get_neighbours(x, y, coordinates)[0] == 1:
                branch_ends.append([x, y])
        return branch_ends


class heightPruning:  # pylint: disable=too-many-instance-attributes
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
    only_height_prune_endpoints : bool
        Whether to only prune endpoints by height, or all skeleton segments. Default is True.
    """  # numpydoc: ignore=PR01

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        image: npt.NDArray,
        skeleton: npt.NDArray,
        max_length: float = None,
        height_threshold: float = None,
        method_values: str = None,
        method_outlier: str = None,
        only_height_prune_endpoints: bool = True,
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
        only_height_prune_endpoints : bool
            Whether to only prune endpoints by height, or all skeleton segments. Default is True.
        """
        self.image = image
        self.skeleton = skeleton
        self.skeleton_convolved = None
        self.skeleton_branches = None
        self.skeleton_branches_labelled = None
        self.max_length = max_length
        self.height_threshold = height_threshold
        self.method_values = method_values
        self.method_outlier = method_outlier
        self.only_height_prune_endpoints = only_height_prune_endpoints
        self.convolve_skeleton()
        self.segment_skeleton()
        self.label_branches()
        self.skeleton_pruned = self.height_prune()

    def convolve_skeleton(self) -> None:
        """Convolve skeleton."""
        self.skeleton_convolved = convolve_skeleton(self.skeleton)

    def segment_skeleton(self) -> None:
        """Convolve skeleton and break into segments at nodes/junctions."""
        self.skeleton_branches = np.where(self.skeleton_convolved == 3, 0, self.skeleton)

    def label_branches(self) -> None:
        """Label segmented branches."""
        self.skeleton_branches_labelled = morphology.label(self.skeleton_branches)

    def _get_branch_mins(self, segments: npt.NDArray) -> npt.NDArray:
        """
        Collect the minimum height value of each individually labeled branch.

        Parameters
        ----------
        segments : npt.NDArray
            Integer labeled array matching the dimensions of the image.

        Returns
        -------
        npt.NDArray
            Array of minimum values of each branch index -1.
        """
        return np.array([np.min(self.image[segments == i]) for i in range(1, segments.max() + 1)])

    def _get_branch_medians(self, segments: npt.NDArray) -> npt.NDArray:
        """
        Collect the median height value of each labeled branch.

        Parameters
        ----------
        segments : npt.NDArray
            Integer labeled array matching the dimensions of the image.

        Returns
        -------
        npt.NDArray
            Array of median values of each branch index -1.
        """
        return np.array([np.median(self.image[segments == i]) for i in range(1, segments.max() + 1)])

    def _get_branch_middles(self, segments: npt.NDArray) -> npt.NDArray:
        """
        Collect the positionally ordered middle height value of each labeled branch.

        Where the branch has an even amount of points, average the two middle heights.

        Parameters
        ----------
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
                start = np.argwhere(convolve_skeleton(segment) == 2)[0]
                ordered_coords = order_branch_from_end(segment, start)
                # if even no. points, average two middles
                middle_idx, middle_remainder = (len(ordered_coords) + 1) // 2 - 1, (len(ordered_coords) + 1) % 2
                mid_coord = ordered_coords[[middle_idx, middle_idx + middle_remainder]]
                # height = image[mid_coord[:, 0], mid_coord[:, 1]].mean()
                height = self.image[mid_coord[:, 0], mid_coord[:, 1]].mean()
            else:
                # if 2 points, need to average them
                height = self.image[segment == 1].mean()
            branch_middles[i - 1] += height
        return branch_middles

    @staticmethod
    def _get_abs_thresh_idx(height_values: npt.NDArray, threshold: float | int) -> npt.NDArray:
        """
        Identify indices of labelled branches whose height values are less than a given threshold.

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
        Identify indices of labelled branch whose height values are less than mean skeleton height - absolute threshold.

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
        LOGGER.debug(f": pruning.py : Avg skeleton height: {avg=}")
        LOGGER.debug(f": pruning.py : mean_abs threshold: {(avg-threshold)=}")
        return np.asarray(np.where(np.asarray(height_values) < (avg - threshold)))[0] + 1

    @staticmethod
    def _get_iqr_thresh_idx(image: npt.NDArray, segments: npt.NDArray) -> npt.NDArray:
        """
        Identify labelled branch indices whose heights are less than 1.5 x interquartile range of all heights.

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
        LOGGER.debug(f": pruning.py : IQR threshold {threshold=}")
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
        return morphology.label(skeleton).max() == 1

    def filter_segments(self, segments: npt.NDArray) -> npt.NDArray:
        """
        Identify and remove segments of a skeleton based on the underlying image height.

        Parameters
        ----------
        segments : npt.NDArray
            A labelled 2D array of skeleton segments.

        Returns
        -------
        npt.NDArray
            The original skeleton without the segments identified by the height criteria.
        """
        # Obtain the height of each branch via the min | median | mid methods
        if self.method_values == "min":
            height_values = self._get_branch_mins(segments)
        elif self.method_values == "median":
            height_values = self._get_branch_medians(segments)
        elif self.method_values == "mid":
            height_values = self._get_branch_middles(segments)
        # threshold heights to obtain indexes of branches to be removed
        if self.method_outlier == "abs":
            idxs = self._get_abs_thresh_idx(height_values, self.height_threshold)
        elif self.method_outlier == "mean_abs":
            idxs = self._get_mean_abs_thresh_idx(height_values, self.height_threshold, self.image, self.skeleton)
        elif self.method_outlier == "iqr":
            idxs = self._get_iqr_thresh_idx(self.image, segments)
        # Only remove the bridge if the skeleton remains a single object.
        skeleton_rtn = self.skeleton.copy()
        for i in idxs:
            temp_skel = self.skeleton.copy()
            temp_skel[segments == i] = 0
            if self.check_skeleton_one_object(temp_skel):
                skeleton_rtn[segments == i] = 0

        return skeleton_rtn

    # def remove_bridges(self) -> npt.NDArray:
    #     """
    #     Identify and remove skeleton bridges using the underlying image height.

    #     Bridges cross the skeleton in places they shouldn't and are defined as an internal branch and thus have no
    #     endpoints. They occur due to poor thresholding creating holes in the mask, creating false "bridges" which
    #     misrepresent the skeleton of the molecule.

    #     Returns
    #     -------
    #     npt.NDArray
    #         A skeleton with internal branches removed by height.
    #     """
    #     conv = convolve_skeleton(self.skeleton)
    #     # Split the skeleton into branches by removing junctions/nodes and label
    #     nodeless = np.where(conv == 3, 0, conv)
    #     segments = morphology.label(np.where(nodeless != 0, 1, 0))
    #     # bridges should not concern endpoints so remove these
    #     for i in range(1, segments.max() + 1):
    #         if (conv[segments == i] == 2).any():
    #             segments[segments == i] = 0
    #     segments = morphology.label(np.where(segments != 0, 1, 0))

    #     # filter the segments based on height criteria
    #     return self.filter_segments(segments)

    def height_prune(self) -> npt.NDArray:
        """
        Identify and remove spurious branches (containing endpoints) using the underlying image height.

        Returns
        -------
        npt.NDArray
            A skeleton with outer branches removed by height.
        """
        conv = convolve_skeleton(self.skeleton)
        segments = self._split_skeleton(conv)
        # if height pruning should only concern endpoints so remove internal connections
        if self.only_height_prune_endpoints:
            for i in range(1, segments.max() + 1):
                if not (conv[segments == i] == 2).any():
                    segments[segments == i] = 0
            segments = morphology.label(np.where(segments != 0, 1, 0))

        # filter the segments based on height criteria
        return self.filter_segments(segments)

    @staticmethod
    def _split_skeleton(skeleton: npt.NDArray) -> npt.NDArray:
        """
        Split the skeleton into branches by removing junctions/nodes and label branches.

        Parameters
        ----------
        skeleton : npt.NDArray
            Convolved skeleton to be split. This should have nodes labelled as 3, ends as 2 and all other points as 1.

        Returns
        -------
        npt.NDArray
            Removes the junctions (3) and returns all remaining sections as labelled segments.
        """
        nodeless = np.where(skeleton == 3, 0, skeleton)
        return morphology.label(np.where(nodeless != 0, 1, 0))


def order_branch_from_end(nodeless: npt.NDArray, start: list, max_length: float = np.inf) -> npt.NDArray:
    """
    Take a linear branch and orders its coordinates starting from a specific endpoint.

    NB - It may be possible to use np.lexsort() to order points, see topostats.measure.feret.sort_coords() for an
    example of how to sort by row or column coordinates, which end of the branch this is from probably doesn't matter
    as one only wants to find the mid-point I think.

    Parameters
    ----------
    nodeless : npt.NDArray
        A 2D binary array where there are no crossing pixels.
    start : list
        A coordinate to start closest to / at.
    max_length : float, optional
        The maximum length to order along the branch, in pixels, by default np.inf.

    Returns
    -------
    npt.NDArray
        The input linear branch ordered from the start coordinate.
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
    conv_skel = convolve_skeleton(skeleton)
    nodes = np.where(conv_skel == 3, 1, 0)
    labeled_nodes = morphology.label(nodes)
    nodeless = np.where((conv_skel == 1) | (conv_skel == 2), 1, 0)
    labeled_nodeless = morphology.label(nodeless)
    size_1_idxs = []

    for node_num in range(1, labeled_nodes.max() + 1):
        node = np.where(labeled_nodes == node_num, 1, 0)
        dil = morphology.binary_dilation(node, footprint=np.ones((3, 3)))
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


def local_area_sum(img: npt.NDArray, point: list | tuple | npt.NDArray) -> tuple:
    """
    Evaluate the local area around a point in a binary map.

    Parameters
    ----------
    img : npt.NDArray
        Binary array of image.
    point : list | tuple | npt.NDArray
        Coordinates of a point within the binary_map.

    Returns
    -------
    tuple
        Tuple consisting of an array values of the local coordinates around the point and the number of neighbours
        around the point.
    """
    if img[point[0], point[1]] > 1:
        raise ValueError("binary_map is not binary!")
    # Capture if point is on the top or left edge or array
    try:
        local_pixels = img[point[0] - 1 : point[0] + 2, point[1] - 1 : point[1] + 2].flatten()
    except IndexError as exc:
        raise IndexError("Point can not be on the edge of an array.") from exc
    # Above does not capture points on right or bottom since slicing arrays beyond their indexes simply extends them
    # Therefore check that we have an array of length 9
    if local_pixels.shape[0] == 9:
        local_pixels[4] = 0  # ensure centre is 0
        if local_pixels.sum() <= 8:
            return local_pixels, local_pixels.sum()
        raise ValueError("'binary_map' is not binary!")
    raise IndexError("'point' is on right or bottom edge of 'binary_map'")
