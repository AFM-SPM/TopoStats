"""Skeletonize molecules"""
import logging
from typing import Callable
import numpy as np
from skimage.morphology import label, medial_axis, skeletonize, thin

from topostats.tracing.tracingfuncs import genTracingFuncs
from topostats.logs.logs import LOGGER_NAME
from topostats.utils import convolve_skelly

LOGGER = logging.getLogger(LOGGER_NAME)

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

    def get_skeleton(self, method: str) -> np.ndarray:
        """Factory method for skeletonizing molecules.

        Parameters
        ----------
        method : str
            Method to use, default is 'zhang' other options are 'lee', 'medial_axis', 'thin' and 'joe'.

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
            Method to use for skeletonizing, methods are 'zhang' (default), 'lee', 'medial_axis', 'thin' and 'joe'.

        Returns
        -------
        Callable
            Returns the function appropriate for the required skeletonizing method.
        """
        if method == "zhang":
            return self._skeletonize_zhang(self.mask).astype(np.int32)
        if method == "lee":
            return self._skeletonize_lee(self.mask).astype(np.int32)
        if method == "medial_axis":
            return self._skeletonize_medial_axis(self.mask).astype(np.int32)
        if method == "thin":
            return self._skeletonize_thin(self.mask).astype(np.int32)
        if method == "joe":
            return self._skeletonize_joe(self.image, self.mask).astype(np.int32)
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
    def _skeletonize_medial_axis(image: np.ndarray) -> np.ndarray:
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
        return medial_axis(image, return_distance=False)

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
    def _skeletonize_joe(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
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
        return joeSkeletonize(image, mask).do_skeletonising()


class joeSkeletonize:
    """Contains all the functions used for Joe's skeletonisation code

    Notes
    -----
    This code contains only the minimum viable product code as much of the other code
    relating to skeletonising based on heights was unused. This also means that
    should someone be upto the task, it is possible to include the heights when skeletonising.
    """

    def __init__(self, image: np.ndarray, mask: np.ndarray):
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
        self.mask = np.pad(self.mask, 1)  # pad to avoid hitting border
        #self.image = np.pad(self.mask, 1) # pad to make same as mask
        while not self.skeleton_converged:
            self._do_skeletonising_iteration()
        # When skeleton converged do an additional iteration of thinning to remove hanging points
        # self.final_skeletonisation_iteration()
        self.mask = getSkeleton(self.image, self.mask).get_skeleton(method="zhang")

        return self.mask[1:-1, 1:-1]  # unpad

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
            skel_img[pixels_to_delete[:,0], pixels_to_delete[:,1]] = 2
            heights = self.image[pixels_to_delete[:, 0], pixels_to_delete[:, 1]]  # get heights of pixels
            hight_sort_idx = np.argsort(heights)[: int(np.ceil(len(heights) * 0.6))]  # idx of lowest 60%
            self.mask[pixels_to_delete[hight_sort_idx, 0], pixels_to_delete[hight_sort_idx, 1]] = 0  # remove lowest 60%

        pixels_to_delete = []
        # Sub-iteration 2 - binary check
        mask_coordinates = np.argwhere(self.mask == 1).tolist()
        for point in mask_coordinates:
            if self._delete_pixel_subit2(point):
                pixels_to_delete.append(point)

        # remove points based on height (lowest 60%)
        pixels_to_delete = np.asarray(pixels_to_delete)
        if pixels_to_delete.shape != (0,):
            skel_img[pixels_to_delete[:,0], pixels_to_delete[:,1]] = 3
            heights = self.image[pixels_to_delete[:, 0], pixels_to_delete[:, 1]]
            hight_sort_idx = np.argsort(heights)[: int(np.ceil(len(heights) * 0.6))]  # idx of lowest 60%
            self.mask[pixels_to_delete[hight_sort_idx, 0], pixels_to_delete[hight_sort_idx, 1]] = 0

        if len(pixels_to_delete) == 0:
            self.skeleton_converged = True
        
        np.savetxt(f"/Users/Maxgamill/Desktop/Uni/PhD/topo_cats/TopoStats/test/processed/taut/dna_tracing/upper/skel_iters/skel_iter_{self.counter}.txt", skel_img)

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

            # Checks for case 1 pixels
            if self._binary_thin_check_b_returncount() == 2 and self._binary_final_thin_check_a():
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

    def prune_skeleton(self, method: str = "joe") -> np.ndarray:
        """Factory method for pruning skeletons.

        Parameters
        ----------
        method : str
            Method to use, default is 'joe'.

        Returns
        -------
        np.ndarray
            An array of the skeleton with spurious branching artefacts removed.

        Notes
        -----

        This is a thin wrapper to the methods provided within the pruning classes below.
        """
        return self._prune_method(method)

    def _prune_method(self, method: str = "joe") -> Callable:
        """Creator component which determines which skeletonize method to use.

        Parameters
        ----------
        method: str
            Method to use for skeletonizing, methods are 'joe' other options are 'conv'.

        Returns
        -------
        Callable
            Returns the function appropriate for the required skeletonizing method.
        """
        if method == "joe":
            return self._prune_joe(self.image, self.skeleton)
        if method == "max":
            return self._prune_max(self.image, self.skeleton)
        # I've read about a "Discrete Skeleton Evolultion" (DSE) method that looks useful
        raise ValueError(method)

    @staticmethod
    def _prune_joe(image: np.ndarray, skeleton: np.ndarray) -> np.ndarray:
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
        return joePrune(image, skeleton).prune_all_skeletons()

    @staticmethod
    def _prune_max(image: np.ndarray, skeleton: np.ndarray) -> np.ndarray:
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
        return maxPrune(image, skeleton).prune_all_skeletons()


class joePrune:
    """Contains all the functions used for Joe's skeletonisation code

    Notes
    -----
    This code contains only the minimum viable product code as much of the other code
    relating to pruning based on heights was unused. This also means that
    should someone be upto the task, it is possible to include the heights when pruning.
    """

    def __init__(self, image: np.ndarray, skeleton: np.ndarray) -> np.ndarray:
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
        for i in range(1, label(self.skeleton).max() + 1):
            single_skeleton = self.skeleton.copy()
            single_skeleton[single_skeleton != i] = 0
            single_skeleton[single_skeleton == i] = 1
            pruned_skeleton_mask += self._prune_single_skeleton(single_skeleton)
            # pruned_skeleton_mask = self._remove_low_dud_branches(pruned_skeleton_mask, self.image)
            # pruned_skeleton_mask = getSkeleton(self.image, pruned_skeleton_mask).get_skeleton('zhang') # reskel to remove nibs
        return pruned_skeleton_mask

    def _prune_single_skeleton(self, single_skeleton: np.ndarray) -> np.ndarray:
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
            number_of_branches = 0
            coordinates = np.argwhere(single_skeleton == 1).tolist()

            # The branches are typically short so if a branch is longer than
            #  0.15 * total points, its assumed to be part of the real data
            max_branch_length = int(len(coordinates) * 0.15)

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

    @staticmethod
    def _remove_low_dud_branches(skeleton, image, threshold=None) -> np.ndarray:
        """Identifies branches which cross the skeleton in places they shouldn't due to
        poor thresholding and holes in the mask. Segments are removed based on heights lower
        than 1.5 * interquartile range of heights."""
        # might need to check that the image *with nodes* is returned
        skeleton_rtn = skeleton.copy()
        conv = convolve_skelly(skeleton)
        nodeless = skeleton.copy()
        nodeless[conv == 3] = 0
        segments = label(nodeless)
        median_heights = [np.median(image[segments == i]) for i in range(1, segments.max() + 1)]
        if threshold is None:
            q75, q25 = np.percentile(median_heights, [75, 25])
            iqr = q75 - q25
            threshold = q25 - 1.5 * iqr
        # threshold heights to remove segments
        idxs = np.asarray(np.where(np.asarray(median_heights) < threshold)) + 1
        for i in idxs:
            print("Removed dud branch: ", i)
            skeleton_rtn[segments == i] = 0
        return skeleton_rtn


class maxPrune:
    """A class for pruning small branches based on convolutions."""

    def __init__(self, image: np.ndarray, skeleton: np.ndarray) -> np.ndarray:
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
        for i in range(1, label(self.skeleton).max() + 1):
            single_skeleton = self.skeleton.copy()
            single_skeleton[single_skeleton != i] = 0
            single_skeleton[single_skeleton == i] = 1
            pruned_skeleton_mask += self._prune_single_skeleton(
                single_skeleton
            )  # maybe need to add other option for large images of like 20px
            # pruned_skeleton_mask = self._remove_low_dud_branches(pruned_skeleton_mask, self.image)
            pruned_skeleton_mask = getSkeleton(self.image, pruned_skeleton_mask).get_skeleton("zhang")
        return pruned_skeleton_mask

    def _prune_single_skeleton(self, single_skeleton: np.ndarray, threshold: float = 0.15) -> np.ndarray:
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

        nodeless_labels = label(nodeless)
        for i in range(1, nodeless_labels.max() + 1):
            vals = conv_skelly[nodeless_labels == i]
            # The branches are typically short so if a branch is longer than
            #  0.15 * total points, its assumed to be part of the real data
            if (vals == 2).any() and vals.size < total_points * threshold:
                single_skeleton[nodeless_labels == i] = 0
                print("Pruned short branch: ", i)
        return single_skeleton

    @staticmethod
    def _remove_low_dud_branches(skeleton, image, threshold=None) -> np.ndarray:
        """Identifies branches which cross the skeleton in places they shouldn't due to
        poor thresholding and holes in the mask. Segments are removed based on heights lower
        than 1.5 * interquartile range of heights."""
        # might need to check that the image *with nodes* is returned
        skeleton_rtn = skeleton.copy()
        conv = convolve_skelly(skeleton)
        nodeless = skeleton.copy()
        nodeless[conv == 3] = 0
        segments = label(nodeless)
        median_heights = [np.median(image[segments == i]) for i in range(1, segments.max() + 1)]
        if threshold is None:
            q75, q25 = np.percentile(median_heights, [75, 25])
            iqr = q75 - q25
            threshold = q25 - 1.5 * iqr
        # threshold heights to remove segments
        idxs = np.asarray(np.where(np.asarray(median_heights) < threshold)) + 1
        for i in idxs:
            print("Removed dud branch: ", i)
            skeleton_rtn[segments == i] = 0
        return skeleton_rtn
