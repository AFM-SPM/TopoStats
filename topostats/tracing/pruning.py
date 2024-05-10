"""Prune branches from skeletons."""

import logging
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

# from skimage.morphology import binary_dilation, label
from skimage import morphology

from topostats.logs.logs import LOGGER_NAME
from topostats.tracing.skeletonize import getSkeleton
from topostats.tracing.tracingfuncs import genTracingFuncs
from topostats.utils import convolve_skeleton

LOGGER = logging.getLogger(LOGGER_NAME)


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
        self,
        prune_args: dict = {"pruning_method": "topostats"},  # noqa: B006
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
    def _prune_topostats(img: npt.NDArray, skeleton: npt.NDArray, prune_args: dict) -> npt.NDArray:
        """
        Prune using the original TopoStats method.

        This is a modified version of the pubhlished Zhang method.

        Parameters
        ----------
        img : npt.NDArray
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
        return topostatsPrune(img, skeleton, **prune_args).prune_all_skeletons()

    @staticmethod
    def _prune_conv(img: npt.NDArray, skeleton: npt.NDArray, prune_args: dict) -> npt.NDArray:
        """
        Prune using a convolutional method.

        Parameters
        ----------
        img : npt.NDArray
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
        return convPrune(img, skeleton, **prune_args).prune_all_skeletons()


class topostatsPrune:  # pylint: disable=too-few-public-methods
    """
    Prune spurious skeletal branches based on their length and/or height.

    Contains all the functions used in the original TopoStats pruning code written by Joe Betton.

    Parameters
    ----------
    img : npt.NDArray
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
        img: npt.NDArray,
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
        img : npt.NDArray
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
        self.img = img
        self.skeleton = skeleton.copy()
        self.max_length = max_length
        self.height_threshold = height_threshold
        self.method_values = method_values
        self.method_outlier = method_outlier

    # Diverges from the change in layout to apply skeletonisation/pruning/tracing to individual grains and then process
    # all grains in an image (possibly in parallel).
    def prune_all_skeletons(self) -> npt.NDArray:
        """
        Prune all skeletons.

        Returns
        -------
        npt.NDArray
            A single mask with all pruned skeletons.
        """
        pruned_skeleton_mask = np.zeros_like(self.skeleton)
        labeled_skel = morphology.label(self.skeleton)
        for i in range(1, labeled_skel.max() + 1):
            single_skeleton = np.where(labeled_skel == i, 1, 0)
            if self.max_length is not None:
                single_skeleton = self._prune_by_length(single_skeleton, max_length=self.max_length)
            if self.height_threshold is not None:
                single_skeleton = heightPruning(
                    self.img,
                    single_skeleton,
                    height_threshold=self.height_threshold,
                    method_values=self.method_values,
                    method_outlier=self.method_outlier,
                ).remove_bridges()
            # skeletonise to remove nibs
            pruned_skeleton_mask += getSkeleton(self.img, single_skeleton, method="zhang").get_skeleton()
        return pruned_skeleton_mask

    def _prune_by_length(  # pylint: disable=too-many-locals  # noqa: C901
        self, single_skeleton: npt.NDArray, max_length: float | int = -1
    ) -> npt.NDArray:
        """
        Remove hanging branches from a skeleton by their length.

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
            n_branches = 0
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
                    n_neighbours, neighbours = genTracingFuncs.count_and_get_neighbours(
                        branch_x, branch_y, temp_coordinates
                    )

                    # If branch continues
                    if n_neighbours == 1:
                        branch_x, branch_y = neighbours[0]
                        branch_coordinates.append([branch_x, branch_y])
                        temp_coordinates.pop(temp_coordinates.index([branch_x, branch_y]))

                    # If the branch reaches the edge of the main trace
                    elif n_neighbours > 1:
                        branch_coordinates.pop(branch_coordinates.index([branch_x, branch_y]))
                        branch_continues = False
                        is_branch = True

                    # Weird case that happens sometimes (would this be linear mols?)
                    elif n_neighbours == 0:
                        is_branch = True
                        branch_continues = False

                    # why not `and branch_continues`?
                    if len(branch_coordinates) > max_branch_length:
                        branch_continues = False
                        is_branch = False
                #
                if is_branch:
                    n_branches += 1
                    for x, y in branch_coordinates:
                        single_skeleton[x, y] = 0

            if n_branches == 0:
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
    Prune spurious branches based on their length and/or height using sliding window convolutions.

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
        labeled_skel = morphology.label(self.skeleton)
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
        conv_skelly = convolve_skeleton(self.skeleton)
        nodeless = self.skeleton.copy()
        nodeless[conv_skelly == 3] = 0

        # The branches are typically short so if a branch is longer than
        #  0.15 * total points, its assumed to be part of the real data
        max_branch_length = max_length if max_length != -1 else int(len(total_points) * 0.15)

        # iterate through branches
        nodeless_labels = morphology.label(nodeless)
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
        self.skeleton = {"skeleton": skeleton}
        self.max_length = max_length
        self.height_threshold = height_threshold
        self.method_values = method_values
        self.method_outlier = method_outlier
        self.convolve_skeleton()
        self.segment_skeleton()
        self.label_branches()

    def convolve_skeleton(self) -> None:
        """Convolve skeleton."""
        self.skeleton["convolved_skeleton"] = convolve_skeleton(self.skeleton["skeleton"])

    def segment_skeleton(self) -> None:
        """Convolve skeleton and break into segments at nodes/junctions."""
        self.skeleton["branches"] = np.where(self.skeleton["convolved_skeleton"] == 3, 0, self.skeleton["skeleton"])

    def label_branches(self) -> None:
        """Label segmented branches."""
        self.skeleton["labelled_branches"] = morphology.label(self.skeleton["branches"])

    @staticmethod
    def _get_branch_mins(image: npt.NDArray, segments: npt.NDArray) -> npt.NDArray:
        """
        Collect the minimum height value of each individually labeled branch.

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
        return np.array([np.min(image[segments == i]) for i in range(1, segments.max() + 1)])

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
        return np.array([np.median(image[segments == i]) for i in range(1, segments.max() + 1)])

    @staticmethod
    def _get_branch_middles(image: npt.NDArray, segments: npt.NDArray) -> npt.NDArray:
        """
        Collect the positionally ordered middle height value of each labeled branch. Where
        the branch has an even ammount of points, average the two middle hights.

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
                start = np.argwhere(convolve_skeleton(segment) == 2)[0]
                ordered_coords = order_branch_from_end(segment, start)
                # if even no. points, average two middles
                middle_idx, middle_remainder = (len(ordered_coords)+ 1) // 2 - 1, (len(ordered_coords)+ 1) % 2
                mid_coord = ordered_coords[[middle_idx, middle_idx + middle_remainder]]
                height = image[mid_coord[:,0], mid_coord[:,1]].mean()
            else:
                # if 2 points, need to average them
                height = image[segment == 1].mean()
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
        return morphology.label(skeleton).max() == 1

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
        skeleton_rtn = self.skeleton["skeleton"].copy()
        conv = convolve_skeleton(self.skeleton["skeleton"])
        # Split the skeleton into branches by removing junctions/nodes and label
        nodeless = np.where(conv == 3, 0, conv)
        segments = morphology.label(nodeless)
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


def order_branch_from_end(nodeless: npt.NDArray, start: list, max_length: float = np.inf) -> npt.NDArray:
    """
    Take a linear branch and orders its coordinates starting from a specific endpoint.

    NB - It may be possible to use np.lexsort() to order points, see topostats.measure.feret.sort_coords() for an
    example of how to sort by row or column coordinates, which end of the branch this is from probably doesn't matter
    as one only wants to find the mid-point I think.

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
