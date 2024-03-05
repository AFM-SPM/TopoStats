"""Skeletonize molecules"""
import logging
from typing import Callable
import numpy as np
from skimage.morphology import binary_dilation, label, medial_axis, skeletonize, thin

from topostats.tracing.tracingfuncs import genTracingFuncs
from topostats.logs.logs import LOGGER_NAME
from topostats.utils import convolve_skelly

LOGGER = logging.getLogger(LOGGER_NAME)
OUTPUT_DIR = "/Users/maxgamill/Desktop"

# Max notes: Want to separate this module into:
#   the different skeletonisation skimage methods & joe's
#   the different branch pruning methods (mine & joe's)
#   skeleton descriptors (mine)


class getSkeleton:
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

    def get_skeleton(self, params={"method": "zhang"}) -> np.ndarray:
        """Factory method for skeletonizing molecules.

        Parameters
        ----------
        method : str
            Method to use, default is 'zhang' other options are 'lee', 'medial_axis', 'thin' and 'topostats'.
        Parameters
        ----------
        image: np.ndarray
            The image used to generate the mask.
        mask: np.ndarray
            The binary mask of features in the image.
        """
        self.image = image
        self.mask = mask

    def get_skeleton(self, params={"method": "zhang"}) -> np.ndarray:
        """Factory method for skeletonizing molecules.

        Parameters
        ----------
        method : str
            Method to use, default is 'zhang' other options are 'lee', 'medial_axis', 'thin' and 'topostats'.

        Returns
        -------
        np.ndarray
            Skeletonised version of the binary mask (possibly using criteria from the image).
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
        return self._get_skeletonize(params)

    def _get_skeletonize(self, params={"skeletonisation_method": "zhang"}) -> Callable:
        """Creator component which determines which skeletonize method to use.

        Parameters
        ----------
        method: str
            Method to use for skeletonizing, methods are 'zhang' (default), 'lee', 'medial_axis', 'thin' and 'topostats'.

        Returns
        -------
        Callable
            Returns the function appropriate for the required skeletonizing method.
        """
        method = params.pop("skeletonisation_method")
        if method == "zhang":
            return self._skeletonize_zhang(self.mask).astype(np.int32)
        if method == "lee":
            return self._skeletonize_lee(self.mask).astype(np.int32)
        if method == "medial_axis":
            return self._skeletonize_medial_axis(self.mask).astype(np.int32)
        if method == "thin":
            return self._skeletonize_thin(self.mask).astype(np.int32)
        if method == "topostats":
            return self._skeletonize_topostats(self.image, self.mask, params).astype(np.int32)
        raise ValueError(method)

    @staticmethod
    def _skeletonize_zhang(mask: np.ndarray) -> np.ndarray:
        """Wrapper for the scikit image implimentation of the Zhang skeletonisation method.

        Parameters
        ----------
        mask: np.ndarray
            A binary array to skeletonise.

        Returns
        -------
        np.ndarray
            The mask array reduce to a single pixel thickness
        """
        return skeletonize(mask, method="zhang")

    @staticmethod
    def _skeletonize_lee(mask: np.ndarray) -> np.ndarray:
        """Wrapper for the scikit image implimentation of the Lee skeletonisation method.

        Parameters
        ----------
        mask: np.ndarray
            A binary array to skeletonise.

        Returns
        -------
        np.ndarray
            The mask array reduce to a single pixel thickness
        """
        return skeletonize(mask, method="lee")

    @staticmethod
    def _skeletonize_medial_axis(mask: np.ndarray) -> np.ndarray:
        """Wrapper for the scikit image implimentation of the medial axis skeletonisation method.

        Parameters
        ----------
        mask: np.ndarray
            A binary array to skeletonise.

        Returns
        -------
        np.ndarray
            The mask array reduce to a single pixel thickness
        """
        # don't know how these work - do they need img or mask?
        return medial_axis(mask, return_distance=False)

    @staticmethod
    def _skeletonize_thin(image: np.ndarray) -> np.ndarray:
        """Wrapper for the scikit image implimentation of the thin skeletonisation method.

        Parameters
        ----------
        mask: np.ndarray
            A binary array to skeletonise.

        Returns
        -------
        np.ndarray
            The mask array reduce to a single pixel thickness
        """
        # don't know how these work - do they need img or mask?
        return thin(image)

    @staticmethod
    def _skeletonize_topostats(image: np.ndarray, mask: np.ndarray, params) -> np.ndarray:
        """Wrapper for Pyne-lab member Joe's skeletonisation method.

        Parameters
        ----------
        mask: np.ndarray
            A binary array to skeletonise.

        Returns
        -------
        np.ndarray
            The mask array reduce to a single pixel thickness

        Notes
        -----
        This method is based on Zhang's method but produces different results
        (less branches but slightly less accurate).
        """
        return topostatsSkeletonize(image, mask, **params).do_skeletonising()


class topostatsSkeletonize:
    """Contains all the functions used for Joe's topostats skeletonisation code

    Notes
    -----
    This code contains only the minimum viable product code as much of the other code
    relating to skeletonising based on heights was unused. This also means that
    should someone be upto the task, it is possible to include the heights when skeletonising.
    """

    def __init__(self, image: np.ndarray, mask: np.ndarray, height_bias):
        """Initialises the class

        Parameters
        ----------
        image: np.ndarray
            The original image containing the data.
        mask: np.ndarray
            The binary image containing the grain(s) to be skeletonised.
        """
        self.image = image
        self.mask = mask.copy()
        self.height_bias = height_bias
        print("BIAS: ", height_bias)

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

    def do_skeletonising(self) -> np.ndarray:
        """The wrapper for the whole skeletonisation process.

        Returns
        -------
        np.ndarray
            The single pixel thick, skeletonised array.
        """
        # do we need padding because of config padding?
        # self.mask = np.pad(self.mask, 1)  # pad to avoid hitting border
        # self.image = np.pad(self.mask, 1) # pad to make same as mask
        while not self.skeleton_converged:
            self._do_skeletonising_iteration()
        # When skeleton converged do an additional iteration of thinning to remove hanging points
        self.final_skeletonisation_iteration()
        self.mask = getSkeleton(self.image, self.mask).get_skeleton({"skeletonisation_method": "zhang"})

        return self.mask

    def _do_skeletonising_iteration(self) -> None:
        """Do an iteration of skeletonisation - check for the local binary pixel
        environment and assess the local height values to decide whether to
        delete a point.
        """
        self.counter += 1
        skel_img = self.mask.copy()
        pixels_to_delete = []
        # Sub-iteration 1 - binary check
        mask_coordinates = np.argwhere(self.mask == 1).tolist()
        for point in mask_coordinates:
            if self._delete_pixel_subit1(point):
                pixels_to_delete.append(point)

        # remove points based on height (lowest 60%)
        pixels_to_delete = np.asarray(pixels_to_delete)  # turn into array
        if pixels_to_delete.shape != (0,):  # ensure array not empty
            skel_img[pixels_to_delete[:, 0], pixels_to_delete[:, 1]] = 2
            heights = self.image[pixels_to_delete[:, 0], pixels_to_delete[:, 1]]  # get heights of pixels
            hight_sort_idx = np.argsort(heights)[
                : int(np.ceil(len(heights) * self.height_bias))
            ]  # idx of lowest height_bias%
            self.mask[
                pixels_to_delete[hight_sort_idx, 0], pixels_to_delete[hight_sort_idx, 1]
            ] = 0  # remove lowest height_bias%

        pixels_to_delete = []
        # Sub-iteration 2 - binary check
        mask_coordinates = np.argwhere(self.mask == 1).tolist()
        for point in mask_coordinates:
            if self._delete_pixel_subit2(point):
                pixels_to_delete.append(point)

        # remove points based on height (lowest 60%)
        pixels_to_delete = np.asarray(pixels_to_delete)
        if pixels_to_delete.shape != (0,):
            skel_img[pixels_to_delete[:, 0], pixels_to_delete[:, 1]] = 3
            heights = self.image[pixels_to_delete[:, 0], pixels_to_delete[:, 1]]
            hight_sort_idx = np.argsort(heights)[
                : int(np.ceil(len(heights) * self.height_bias))
            ]  # idx of lowest height_bias%
            self.mask[
                pixels_to_delete[hight_sort_idx, 0], pixels_to_delete[hight_sort_idx, 1]
            ] = 0  # remove lowest height_bias%

        if len(pixels_to_delete) == 0:
            self.skeleton_converged = True

        # np.savetxt(f"{OUTPUT_DIR}/Uni/PhD/topo_cats/TopoStats/test/processed/taut/dna_tracing/upper/skel_iters/skel_iter_{self.counter}.txt", skel_img)

    def _delete_pixel_subit1(self, point: list) -> bool:
        """Function to check whether a single point should be deleted based
        on both its local binary environment.

        Parameters
        ----------
        point: list
            List of [x, y] coordinate positions

        Returns
        -------
        bool
            Returns T/F depending if the surrounding points have met the criteria
            of the binary thin a, b returncount, c and d checks below.
        """

        self.p7, self.p8, self.p9, self.p6, self.p2, self.p5, self.p4, self.p3 = self.get_local_pixels_binary(
            self.mask, point[0], point[1]
        )
        return (
            self._binary_thin_check_a()
            and self._binary_thin_check_b_returncount() == 1
            and self._binary_thin_check_c()
            and self._binary_thin_check_d()
        )

    def _delete_pixel_subit2(self, point) -> bool:
        """Function to check whether a single point should be deleted based
        on both its local binary environment.

        Parameters
        ----------
        point: list
            List of [x, y] coordinate positions

        Returns
        -------
        bool
            Returns T/F depending if the surrounding points have met the criteria
            of the binary thin a, b returncount, csharp and dsharp checks below.
        """

        self.p7, self.p8, self.p9, self.p6, self.p2, self.p5, self.p4, self.p3 = self.get_local_pixels_binary(
            self.mask, point[0], point[1]
        )
        # Add in generic code here to protect high points from being deleted
        return (
            self._binary_thin_check_a()
            and self._binary_thin_check_b_returncount() == 1
            and self._binary_thin_check_csharp()
            and self._binary_thin_check_dsharp()
        )

    def _binary_thin_check_a(self) -> bool:
        """Checks the surrounding area to see if the point lies on the edge of the grain.
        Condition A protects the endpoints (which will be > 2)

        Returns
        -------
        bool
            if point lies on edge of graph and isn't an endpoint.
        """
        return 2 <= self.p2 + self.p3 + self.p4 + self.p5 + self.p6 + self.p7 + self.p8 + self.p9 <= 6

    def _binary_thin_check_b_returncount(self) -> bool:
        """Assess local area connectivity?"""
        count = sum(
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

        return count

    def _binary_thin_check_c(self) -> bool:
        """Check if p2, p4 or p6 is 0 - seems very specific

        Returns
        -------
        bool
            if p2, p4 or p6 is 0.
        """
        return self.p2 * self.p4 * self.p6 == 0

    def _binary_thin_check_d(self) -> bool:
        """Check if p4, p6 or p8 is 0 - seems very specific

        Returns
        -------
        bool
            if p4, p6 or p8 is 0.
        """
        return self.p4 * self.p6 * self.p8 == 0

    def _binary_thin_check_csharp(self) -> bool:
        """Check if p2, p4 or p8 is 0 - seems very specific

        Returns
        -------
        bool
            if p2, p4 or p8 is 0.
        """
        return self.p2 * self.p4 * self.p8 == 0

    def _binary_thin_check_dsharp(self) -> bool:
        """Check if p2, p6 or p8 is 0 - seems very specific

        Returns
        -------
        bool
            if p2, p6 or p8 is 0.
        """
        return self.p2 * self.p6 * self.p8 == 0

    def final_skeletonisation_iteration(self) -> None:
        """A final skeletonisation iteration that removes "hanging" pixels.
        Examples of such pixels are:
                    [0, 0, 0]               [0, 1, 0]            [0, 0, 0]
                    [0, 1, 1]               [0, 1, 1]            [0, 1, 1]
            case 1: [0, 1, 0]   or  case 2: [0, 1, 0] or case 3: [1, 1, 0]

        This is useful for the future functions that rely on local pixel environment
        to make assessments about the overall shape/structure of traces"""

        remaining_coordinates = np.argwhere(self.mask).tolist()

        for x, y in remaining_coordinates:
            self.p7, self.p8, self.p9, self.p6, self.p2, self.p5, self.p4, self.p3 = self.get_local_pixels_binary(
                self.mask, x, y
            )

            # Checks for case 1 and 3 pixels
            if self._binary_thin_check_b_returncount() == 2 and self._binary_final_thin_check_a() and not self.binary_thin_check_max():
                self.mask[x, y] = 0
            # Checks for case 2 pixels
            elif self._binary_thin_check_b_returncount() == 3 and self._binary_final_thin_check_b():
                self.mask[x, y] = 0

    def _binary_final_thin_check_a(self) -> bool:
        """Assess if local area has 4-connectivity.

        Returns
        -------
        bool
            Logical indicator of whether if any neighbours of the 4-connections have a near pixel.
        """
        return 1 in (self.p2 * self.p4, self.p4 * self.p6, self.p6 * self.p8, self.p8 * self.p2)

    def _binary_final_thin_check_b(self) -> bool:
        """Assess if local area 4-connectivity is connected to multiple branches.

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
    
    def binary_thin_check_max(self) -> bool:
        """Checks if opposite corner diagonals are present."""
        return 1 in (self.p7*self.p3, self.p5*self.p9)

    @staticmethod
    def get_local_pixels_binary(binary_map, x, y) -> np.ndarray:
        """Get the values of the pixels in the local 8-connectivit area around
        the coordinate described by x and y.

        [[p7, p8, p9],    [[0,1,2],
         [p6, na, p2], ->  [3,4,5], -> [0,1,2,3,5,6,7,8]
         [p5, p4, p3]]     [6,7,8]]
        delete coordinate pixel to only get local area.

        Parameters
        ----------
        binary_map: np.ndarray
            The binary array containing the grains.
        x: int
            An x coordinate within the binary map.
        y: int
            A y coordinate within the binary map.

        Returns
        -------
        np.ndarray
            A flattened 8-long array describing the values in the binary map
            around the x,y point
        """
        local_pixels = binary_map[x - 1 : x + 2, y - 1 : y + 2].flatten()
        return np.delete(local_pixels, 4)


class pruneSkeleton:
    """Class containing skeletonization pruning code from factory methods to functions
    depaendant on the method. Pruning is the act of removing spurious branches commonly
    found when implimenting skeletonization algorithms."""

    def __init__(self, image: np.ndarray, skeleton: np.ndarray) -> None:
        """Initialise the class.

        Parameters
        ----------
        image: np.ndarray
            The original image the skeleton derives from (not the binary mask)
        skeleton: np.ndarray
            The single-pixel-thick skeleton pertaining to features of the image.
        """
        self.image = image
        self.skeleton = skeleton

    def prune_skeleton(self, prune_args: dict = {"pruning_method": "topostats"}) -> np.ndarray:
        """Factory method for pruning skeletons.

        Parameters
        ----------
        method : str
            Method to use, default is 'topostats'.

        Returns
        -------
        np.ndarray
            An array of the skeleton with spurious branching artefacts removed.

        Notes
        -----

        This is a thin wrapper to the methods provided within the pruning classes below.
        """
        return self._prune_method(prune_args)

    def _prune_method(self, prune_args=None) -> Callable:
        """Creator component which determines which skeletonize method to use.

        Parameters
        ----------
        method: str
            Method to use for skeletonizing, methods are 'topostats' other options are 'conv'.

        Returns
        -------
        Callable
            Returns the function appropriate for the required skeletonizing method.
        """
        method = prune_args.pop("pruning_method")
        if method == "topostats":
            return self._prune_topostats(self.image, self.skeleton, prune_args)
        if method == "max":
            return self._prune_max(self.image, self.skeleton, prune_args)
        # I've read about a "Discrete Skeleton Evolultion" (DSE) method that looks useful
        raise ValueError(method)

    @staticmethod
    def _prune_topostats(image: np.ndarray, skeleton: np.ndarray, prune_args: dict) -> np.ndarray:
        """Wrapper for Pyne-lab member Joe's pruning method.

        Parameters
        ----------
        image: np.ndarray
            The image used to find the skeleton (doesn't have to be binary)
        skeleton: np.ndarray
            Binary array containing skelton(s)

        Returns
        -------
        np.ndarray
            The skeleton with spurious branching artefacts removed.
        """
        return topostatsPrune(image, skeleton, **prune_args).prune_all_skeletons()

    @staticmethod
    def _prune_max(image: np.ndarray, skeleton: np.ndarray, prune_args: dict) -> np.ndarray:
        """Wrapper for Pyne-lab member Max's pruning method.

        Parameters
        ----------
        image: np.ndarray
            The image used to find the skeleton (doesn't have to be binary)
        skeleton: np.ndarray
            Binary array containing skelton(s)

        Returns
        -------
        np.ndarray
            The skeleton with spurious branching artefacts removed.
        """
        return maxPrune(image, skeleton, **prune_args).prune_all_skeletons()


class topostatsPrune:
    """Contains all the functions used for Joe's pruning code

    Notes
    -----
    This code contains only the minimum viable product code as much of the other code
    relating to pruning based on heights was unused. This also means that
    should someone be upto the task, it is possible to include the heights when pruning.
    """

    def __init__(
        self,
        image: np.ndarray,
        skeleton: np.ndarray,
        max_length: float = None,
        height_threshold: float = None,
        method_values: str = None,
        method_outlier: str = None,
    ) -> np.ndarray:
        """Initialise the class.

        Parameters
        image: np.ndarray
            The original data to help with branch removal.

        skeleton np.ndarray
            The skeleton to remove unwanted branches from.

        Returns:
            np.ndarray: _description_
        """
        self.image = image
        self.skeleton = skeleton.copy()
        self.max_length = max_length
        self.height_threshold = height_threshold
        self.method_values = method_values
        self.method_outlier = method_outlier

    def prune_all_skeletons(self) -> np.ndarray:
        """Wrapper function to prune all skeletons by labling and iterating through
        each one, binarising, then pruning, then adding up the single skeleton masks
        to make a single mask.

        Returns
        -------
        np.ndarray
            A single mask with all pruned skeletons.
        """
        pruned_skeleton_mask = np.zeros_like(self.skeleton)
        labeled_skel = label(self.skeleton)
        for i in range(1, labeled_skel.max() + 1):
            single_skeleton = np.where(labeled_skel == i, 1, 0)
            if self.max_length is not None:
                single_skeleton = self._prune_single_skeleton(single_skeleton, max_length=self.max_length)
            if self.height_threshold is not None:
                single_skeleton = remove_bridges_abs(
                    single_skeleton,
                    self.image,
                    threshold=self.height_threshold,
                    method_values=self.method_values,
                    method_outlier=self.method_outlier,
                )
            pruned_skeleton_mask += getSkeleton(self.image, single_skeleton).get_skeleton(
                {"skeletonisation_method": "zhang"}
            )  # reskel to remove nibs
        return pruned_skeleton_mask

    def _prune_single_skeleton(self, single_skeleton: np.ndarray, max_length=-1) -> np.ndarray:
        """Function to remove the hanging branches from a single skeleton as this
        function is an iterative process. These are a persistent problem in the
        overall tracing process.

        Parameters
        ---------
        single_skeleton: np.ndarray
            A binary array containing a single skeleton.

        Returns:
        --------
        np.ndarray
            A binary mask of the single skeleton
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
                        print(f"Removed: {len(branch_coordinates)} / {len(coordinates)}, {max_branch_length}")
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
    def _find_branch_ends(coordinates) -> list:
        """Identifies branch ends as they only have one connected point.

        Parameters
        ----------
        coordinates: list
            A list of x, y coordinates of a branch.

        Returns
        -------
        list
            A list of x,y coordinates of the branch ends.
        """
        potential_branch_ends = []

        # Most of the branch ends are just points with one neighbour
        for x, y in coordinates:
            if genTracingFuncs.count_and_get_neighbours(x, y, coordinates)[0] == 1:
                potential_branch_ends.append([x, y])
        return potential_branch_ends


class maxPrune:
    """A class for pruning small branches based on convolutions."""

    def __init__(
        self,
        image: np.ndarray,
        skeleton: np.ndarray,
        max_length: float = None,
        height_threshold: float = None,
        method_values: str = None,
        method_outlier: str = None
    ) -> np.ndarray:
        """Initialise the class.

        Parameters
        image: np.ndarray
            The original data to help with branch removal.

        skeleton np.ndarray
            The skeleton to remove unwanted branches from.

        Returns:
            np.ndarray: _description_
        """
        self.image = image
        self.skeleton = skeleton.copy()
        self.max_length = max_length
        self.height_threshold = height_threshold
        self.method_values = method_values
        self.method_outlier = method_outlier

    def prune_all_skeletons(self) -> np.ndarray:
        """Wrapper function to prune all skeletons by labling and iterating through
        each one, binarising, then pruning, then adding up the single skeleton masks
        to make a single mask.

        Returns
        -------
        np.ndarray
            A single mask with all pruned skeletons.
        """
        pruned_skeleton_mask = np.zeros_like(self.skeleton)
        labeled_skel = label(self.skeleton)
        for i in range(1, labeled_skel.max() + 1):
            single_skeleton = np.where(labeled_skel == i, 1, 0)
            if self.max_length is not None:
                single_skeleton = self._prune_single_skeleton(single_skeleton, max_length=self.max_length)
            if self.height_threshold is not None:
                single_skeleton = remove_bridges_abs(
                    single_skeleton,
                    self.image,
                    threshold=self.height_threshold,
                    method_values=self.method_values,
                    method_outlier=self.method_outlier,
                )
            pruned_skeleton_mask += getSkeleton(self.image, single_skeleton).get_skeleton(
                {"skeletonisation_method": "zhang"}
            )  # reskel to remove nibs

        return pruned_skeleton_mask

    def _prune_single_skeleton(self, single_skeleton: np.ndarray, max_length=-1) -> np.ndarray:
        """Function to remove the hanging branches from a single skeleton via local-area convoluions.

        Parameters
        ---------
        single_skeleton: np.ndarray
            A binary array containing a single skeleton.

        Returns:
        --------
        np.ndarray
            A binary mask of the single skeleton
        """
        total_points = self.skeleton.size
        single_skeleton = self.skeleton.copy()
        conv_skelly = convolve_skelly(self.skeleton)
        nodeless = self.skeleton.copy()
        nodeless[conv_skelly == 3] = 0

        # The branches are typically short so if a branch is longer than
        #  0.15 * total points, its assumed to be part of the real data
        max_branch_length = max_length if max_length != -1 else int(len(total_points) * 0.15)

        nodeless_labels = label(nodeless)
        for i in range(1, nodeless_labels.max() + 1):
            vals = conv_skelly[nodeless_labels == i]
            if (vals == 2).any() and (vals.size < max_branch_length):
                single_skeleton[nodeless_labels == i] = 0
                print("Pruned short branch: ", i)
        return single_skeleton


def remove_bridges_abs(skeleton, image, threshold, method_values, method_outlier) -> np.ndarray:
    """Identifies branches which cross the skeleton in places they shouldn't due to
    poor thresholding and holes in the mask. Segments are removed based on heights lower
    than 1.5 * interquartile range of heights."""
    # might need to check that the image *with nodes* is returned
    skeleton_rtn = skeleton.copy()
    conv = convolve_skelly(skeleton)
    nodeless = np.where(conv == 3, 0, conv)
    segments = label(nodeless)
    if method_values == "min":
        height_values = [np.min(image[segments == i]) for i in range(1, segments.max() + 1)]
    elif method_values == "median":
        height_values = [np.median(image[segments == i]) for i in range(1, segments.max() + 1)]
    elif method_values == "mid":
        height_values = np.zeros(segments.max())
        for i in range(1, segments.max() + 1):
            segment = np.where(segments == i, 1, 0)
            if segment.sum() > 2:
                # sometimes start is not found
                start = np.argwhere(convolve_skelly(segment) == 2)[0]
                ordered_coords = order_branch_from_start(segment, start)
                mid_coord = ordered_coords[len(ordered_coords) // 2]
            else:
                mid_coord = np.argwhere(segment == 1)[0]
            height_values[i - 1] += image[mid_coord[0], mid_coord[1]]
        print("PRUNE: ", height_values)
    else:
        print("Incorrect 'method_values' value.")

    # threshold heights to remove segments
    if method_outlier == "abs":
        idxs = np.asarray(np.where(np.asarray(height_values) < threshold))[0] + 1
    elif method_outlier == "abs_mean":
        avg = image[skeleton == 1].mean()
        idxs = np.asarray(np.where(np.asarray(height_values) < (avg - threshold)))[0] + 1
        print("THRESH: ", avg, threshold, avg - threshold)
    elif method_outlier == "iqr":
        coords = np.argwhere(skeleton == 1)
        heights = image[coords[:, 0], coords[:, 1]]
        q75, q25 = np.percentile(heights, [75, 25])
        iqr = q75 - q25
        threshold = q25 - 1.5 * iqr
        # print("IQR Thresh: ", q25, q75, threshold)
        low_coords = coords[heights < threshold]
        low_segment_idxs = []
        low_segment_mins = []
        for segment_num in range(1, segments.max() + 1):
            segment_coords = np.argwhere(segments == segment_num)
            for low_coord in low_coords:
                place = np.isin(segment_coords, low_coord).all(axis=1)
                if place.any():
                    low_segment_idxs.append(segment_num)
                    low_segment_mins.append(image[segments == segment_num].min())
                    break

        idxs = np.array(low_segment_idxs)[np.argsort(low_segment_mins)]  # sort in order of ascending mins
    else:
        print("Incorrect 'meth_outlier' value.")

    for i in idxs:
        temp_skel = skeleton_rtn.copy()
        temp_skel[segments == i] = 0
        if check_skeleton_one_object(temp_skel):
            print("Removed dud branch: ", i)
            skeleton_rtn[segments == i] = 0
        else:
            print(f"Bridge {i} breaks skeleton, not removed")
    return skeleton_rtn


def remove_bridges_abs_mean(skeleton, image, threshold) -> np.ndarray:
    """Identifies branches which cross the skeleton in places they shouldn't due to
    poor thresholding and holes in the mask. Segments are removed based on their minium
    heights being lower then the mean-threshold.
    I.e. for dna a thresold of 0.85nm would remove all segments whose lowest point is <
    2-0.85nm or 1.15nm. (0.85nm from the depth of the major groove)
    """
    # might need to check that the image *with nodes* is returned
    skeleton_rtn = skeleton.copy()
    avg = image[skeleton == 1].mean()
    conv = convolve_skelly(skeleton)
    nodeless = np.where(conv == 3, 0, conv)
    segments = label(nodeless)
    min_heights = [np.min(image[segments == i]) for i in range(1, segments.max() + 1)]
    # print("Min Heights: ", min_heights)
    # print("VALS: ", avg, threshold, (avg-threshold))
    # threshold heights to remove segments
    idxs = np.asarray(np.where(np.asarray(min_heights) < (avg - threshold)))[0] + 1
    for i in idxs:
        temp_skel = skeleton_rtn.copy()
        temp_skel[segments == i] = 0
        if check_skeleton_one_object(temp_skel):
            print("Removed dud branch: ", i)
            skeleton_rtn[segments == i] = 0
        else:
            print(f"Bridge {i} breaks skeleton, not removed")
    return skeleton_rtn


def remove_bridges_iqr(skeleton, image):
    """Removes bridges via the outlier metric Q1 + 1.5 * IQR of each skeleton pixel's heights.

    Parameters
    ----------
    skeleton : np.ndarray
        _description_
    image : np.ndarray
        _description_
    """
    skeleton_rtn = skeleton.copy()
    conv = convolve_skelly(skeleton)
    nodeless = np.where(conv == 3, 0, conv)
    segments = label(nodeless)

    coords = np.argwhere(skeleton == 1)
    heights = image[coords[:, 0], coords[:, 1]]
    q75, q25 = np.percentile(heights, [75, 25])
    iqr = q75 - q25
    threshold = q25 - 1.5 * iqr
    # print("IQR Thresh: ", q25, q75, threshold)
    low_coords = coords[heights < threshold]

    low_segment_idxs = []
    low_segment_mins = []
    for segment_num in range(1, segments.max() + 1):
        segment_coords = np.argwhere(segments == segment_num)
        for low_coord in low_coords:
            place = np.isin(segment_coords, low_coord).all(axis=1)
            if place.any():
                low_segment_idxs.append(segment_num)
                low_segment_mins.append(image[segments == segment_num].min())
                break

    low_segment_idxs = np.array(low_segment_idxs)[np.argsort(low_segment_mins)]  # sort in order of ascending mins
    for i in low_segment_idxs:
        temp_skel = skeleton_rtn.copy()
        temp_skel[segments == i] = 0
        if check_skeleton_one_object(temp_skel):
            print("Removed dud branch: ", i)
            skeleton_rtn[segments == i] = 0
        else:
            print(f"Bridge {i} breaks skeleton, not removed")

    return skeleton_rtn


def remove_bridges_mid(skeleton, image, threshold) -> np.ndarray:
    """Identifies branches which cross the skeleton in places they shouldn't due to
    poor thresholding and holes in the mask. Segments are removed based on their minium
    heights being lower then the mean-threshold.
    I.e. for dna a thresold of 0.85nm would remove all segments whose lowest point is <
    2-0.85nm or 1.15nm. (0.85nm from the depth of the major groove)
    """
    # might need to check that the image *with nodes* is returned
    skeleton_rtn = skeleton.copy()
    avg = image[skeleton == 1].mean()
    conv = convolve_skelly(skeleton)
    nodeless = np.where(conv == 3, 0, conv)
    segments = label(nodeless)

    mids = np.zeros(len(segments.max()))

    for i in range(1, segments.max() + 1):
        segment = np.where(segments == i, 1, 0)
        start = np.argwhere(convolve_skelly(segment) == 1)[0]
        ordered_coords = order_branch_from_start(segment, start)
        mid_coord = ordered_coords[len(ordered_coords) // 2]
        mids[i - 1] += image[mid_coord[0], mid_coord[1]]

    # threshold heights to remove segments
    idxs = np.asarray(np.where(mids < (avg - threshold)))[0] + 1
    for i in idxs:
        temp_skel = skeleton_rtn.copy()
        temp_skel[segments == i] = 0
        if check_skeleton_one_object(temp_skel):
            print("Removed dud branch: ", i)
            skeleton_rtn[segments == i] = 0
        else:
            print(f"Bridge {i} breaks skeleton, not removed")
    return skeleton_rtn


def check_skeleton_one_object(skeleton):
    """Ensures that the skeleton hasn't been broken up upon removing a segment."""
    skeleton = np.where(skeleton != 0, 1, 0)
    # print("UNIQ_SKELS: ", label(skeleton).max())
    return label(skeleton).max() == 1


def rm_nibs(skeleton):
    """Attempts to remove single pixel branches (nibs) not identified by nearest neighbour
    algorithms as there may be >2 neighbours.

    Parameters
    ----------
    skeleton : np.ndarray
        A single pixel thick trace.

    Returns
    -------
    np.ndarray
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


def order_branch_from_start(nodeless, start, max_length=np.inf):
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


def local_area_sum(binary_map, point):
    """Evaluates the local area around a point in a binary map.

    Parameters
    ----------
    binary_map: np.ndarray
        A binary array of an image.
    point: Union[list, touple, np.ndarray]
        A single object containing 2 integers relating to a point within the binary_map

    Returns
    -------
    np.ndarray
        An array values of the local coordinates around the point.
    int
        A value corresponding to the number of neighbours around the point in the binary_map.
    """
    x, y = point
    local_pixels = binary_map[x - 1 : x + 2, y - 1 : y + 2].flatten()
    local_pixels[4] = 0  # ensure centre is 0
    return local_pixels, local_pixels.sum()
