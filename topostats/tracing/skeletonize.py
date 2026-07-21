"""Skeletonize molecules."""

import heapq
import logging
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import skimage as ski
from scipy.ndimage import distance_transform_edt
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
        # return topostatsSkeletonize(image, mask, height_bias).do_skeletonising()

        # return Skeletonisation(image, mask, height_bias).do_skeletonisation()

        og_skel = topostatsSkeletonize(image, mask, height_bias).do_skeletonising().astype(bool)
        new_skel = Skeletonisation(image, mask, height_bias).do_skeletonisation()

        diff = og_skel != new_skel

        _, ax = plt.subplots(1, 3, figsize=(15, 5))

        ax[0].imshow(image, cmap="grey")
        skel_mask = np.ma.masked_where(~og_skel, og_skel)
        ax[0].imshow(skel_mask, alpha=0.5)
        ax[0].set_title("OG")

        ax[1].imshow(image, cmap="grey")
        skel_mask = np.ma.masked_where(~new_skel, new_skel)
        ax[1].imshow(skel_mask, alpha=0.5)
        ax[1].set_title("new")

        ax[2].imshow(image, cmap="grey")
        diff_masked = np.ma.masked_where(~diff, diff)
        ax[2].imshow(diff_masked, alpha=0.5)
        ax[2].set_title(f"Difference {diff.sum()} pixels")

        plt.tight_layout()
        plt.savefig("diff_checker.png")

        return new_skel


class Skeletonisation:
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

    def __init__(self, image: np.ndarray, mask: np.ndarray, height_bias: float = 0.6):
        """
        Initialise the class.

        Parameters
        ----------
        image : npt.NDArray
            Original 2D image containing the height data.
        mask : npt.NDArray
            Binary image containing the object to be skeletonised. Dimensions should match those of 'image'.
        height_bias : float
            Ratio of lowest intensity (height) pixels to total pixels fitting the skeletonisation criteria.
        """
        # Extend the image/ mask by mirroring to avoid edge effects
        self.image = np.pad(image, pad_width=1, mode="edge")
        self.mask = np.pad(mask, pad_width=1, mode="edge")
        self.height_bias = height_bias

    def do_skeletonisation(self) -> np.ndarray:
        """
        Perform skeletonisation.

        Returns
        -------
        npt.NDArray
            The single pixel thick, skeletonised array.
        """
        # Main process, skeletonise the mask
        priority_map = self.calculate_priority_map()
        self.skeletonise_with_bias(priority_map)

        # Remove padding added in __init__() to handle edge pixels
        self.image = self.image[1:-1, 1:-1]
        self.mask = self.mask[1:-1, 1:-1]

        # Final skeletonisation before returning avoids diagonal lines being too thick
        return ski.morphology.skeletonize(self.mask)

        # return self.mask

    def calculate_priority_map(self) -> np.ndarray:
        """
        Create an array of size mask.shape containing priority scores for each pixel.

        The scores are calculated with: score = distance_to_edge + (1.0 - normalised_height) * height_bias
        This means a higher height bias reduces the importance of the pixel being in the centre of the line
        being skeletonised.

        Returns
        -------
        np.ndarray
            The priority map for each pixel in the image. Pixels not part of the mask are marked as 0.
        """
        # Create array of shape mask.shape with score of distance from edge
        dist = distance_transform_edt(self.mask)

        # Normalise the heightmap
        img_min, img_max = self.image.min(), self.image.max()
        norm_height = (self.image - img_min) / (img_max - img_min + 1e-8)

        # Combine the two arrays, balanced by the height bias
        return dist * (1 - self.height_bias) + norm_height * self.height_bias

    def skeletonise_with_bias(self, priority_map):
        """
        Create a skeleton from the mask based on the given priority map.

        Loop through pixels in the mask and queue any pixels on a boundary, then loop through the
        created queue and check each for deletability. If so, delete and add its neighbouring pixels
        to the queue as they are now boundary pixels.

        Parameters
        ----------
        priority_map : np.ndarray
            A 2d array of shape self.mask, each value between 0-1 and representing the 'priority' rating
            of that pixel. The higher the priority, the more it is chosen over other pixels when skeletonising.
        """
        height, width = self.mask.shape
        queue = []
        queue_map = np.zeros_like(self.mask, dtype=bool)  # Boolean map of if the pixel is in queue

        # Find all potential pixels to delete, the very edges of the image can be ignored as this is padding
        for row in range(1, height - 1):
            for col in range(1, width - 1):
                if self.mask[row, col] == 1:
                    # If a 1 touches a 0 it is a boundary pixel
                    if np.min(self.mask[row - 1 : row + 2, col - 1 : col + 2]) == 0:
                        heapq.heappush(queue, (priority_map[row, col], row, col))
                        queue_map[row, col] = True

        while queue:
            _, row, col = heapq.heappop(queue)
            queue_map[row, col] = False
            # Skip if it's been removed from the mask in a previous iteration
            if self.mask[row, col] == 0:
                continue

            if self.is_safe_to_delete(row, col):
                self.mask[row, col] = 0
                # Add neighbours in remaining mask to queue as they have become boundaries
                for dirrow, dircol in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    newrow, newcol = row + dirrow, col + dircol
                    # Check the neighbour exists and is not already in queue
                    if self.mask[newrow, newcol] == 1 and not queue_map[newrow, newcol]:
                        heapq.heappush(queue, (priority_map[newrow, newcol], newrow, newcol))
                        queue_map[newrow, newcol] = True
            else:
                # Not safe
                pass

    def is_safe_to_delete(self, row, col) -> bool:
        """
        Check if a pixel can be safely deleted.

        This is determined by checking neighbouring pixels and confirming the pixel is not
        at the end of a skeleton (only 1 neighbour) or that it isn't in the centre of a blob
        (therefore not an edge pixel).

        Parameters
        ----------
        row : int
            The row index of the pixel being analysed.
        col : int
            The column index of the pixel being analysed.

        Returns
        -------
        bool
            If the pixel is safe to delete or not.
        """
        height, width = self.mask.shape
        if row <= 0 or row >= height - 1 or col <= 0 or col >= width - 1:
            return False

        p = self.get_local_pixels_binary(self.mask, row, col)
        neighbours = [p[1], p[2], p[4], p[7], p[6], p[5], p[3], p[0]]

        # Check that the pixel is not at the end of a line or an isolated dot (num_neighbours < 2)
        # and that the pixel is not surrounded (num_neighbours > 6)
        num_neighbours = sum(neighbours)
        if num_neighbours < 2 or num_neighbours > 6:
            return False

        # Check the pixel's neighbours in a circle, every time a pixel is 0 and the next is
        # 1 count that as a transition. A central pixel on the edge of a block will always have one
        # single transition.
        transitions = 0
        for i, _ in enumerate(neighbours):
            if neighbours[i] == 0 and neighbours[(i + 1) % 8] == 1:
                transitions += 1

        return transitions == 1

    def get_local_pixels_binary(self, binary_map: np.ndarray, x: int, y: int) -> np.ndarray:
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
        counter = 0
        while not self.skeleton_converged:
            self._do_skeletonising_iteration()
            counter += 1
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
        is all pixels similar to Zhang.
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
        if x == 0 or x >= binary_map.shape[0] or y == 0 or y >= binary_map.shape[1]:
            raise IndexError(
                f"One of the pixel coordinates ({x=}, {y=}) are on the edge of the image ({binary_map.shape=})"
            )
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
