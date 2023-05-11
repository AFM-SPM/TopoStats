"""Find grains in an image."""
# pylint: disable=no-name-in-module
from collections import defaultdict
import logging
from typing import List, Dict
import numpy as np

from skimage.segmentation import clear_border
from skimage import morphology
from skimage.measure import regionprops
from skimage.color import label2rgb

from topostats.logs.logs import LOGGER_NAME
from topostats.thresholds import threshold
from topostats.utils import _get_mask, get_thresholds

LOGGER = logging.getLogger(LOGGER_NAME)

# pylint: disable=fixme
# pylint: disable=line-too-long
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-arguments
# pylint: disable=bare-except
# pylint: disable=dangerous-default-value


class Grains:
    """Find grains in an image."""

    def __init__(
        self,
        image: np.ndarray,
        filename: str,
        pixel_to_nm_scaling: float,
        threshold_method: str = None,
        otsu_threshold_multiplier: float = None,
        threshold_std_dev: dict = None,
        threshold_absolute: dict = None,
        absolute_area_threshold: dict = {
            "above": [None, None],
            "below": [None, None],
        },
        grain_height_removal_thresholds_std_dev: dict = {
            "above": None,
            "below": None,
        },
        direction: str = None,
        smallest_grain_size_nm2: float = None,
    ):
        """Initialise the class.

        Parameters
        ----------
        image: np.ndarray
            2D Numpy array of image
        filename: str
            File being processed
        pixel_to_nm_scaling: float
            Sacling of pixels to nanometre.
        threshold_multiplier : Union[int, float]
            Factor by which below threshold is to be scaled prior to masking.
        threshold_method: str
            Method for determining threshold to mask values, default is 'otsu'.
        threshold_std_dev: dict
            Dictionary of 'below' and 'above' factors by which standard deviation is multiplied to derive the threshold if threshold_method is 'std_dev'.
        threshold_absolute: dict
            Dictionary of absolute 'below' and 'above' thresholds for grain finding.
        absolute_area_threshold: dict
            Dictionary of above and below grain's area thresholds
        direction: str
            Direction for which grains are to be detected, valid values are above, below and both.
        """
        self.image = image
        self.filename = filename
        self.pixel_to_nm_scaling = pixel_to_nm_scaling
        self.threshold_method = threshold_method
        self.otsu_threshold_multiplier = otsu_threshold_multiplier
        self.threshold_std_dev = threshold_std_dev
        self.threshold_absolute = threshold_absolute
        self.absolute_area_threshold = absolute_area_threshold
        self.grain_height_removal_thresholds_std_dev = grain_height_removal_thresholds_std_dev
        # Only detect grains for the desired direction
        self.direction = [direction] if direction != "both" else ["above", "below"]
        self.smallest_grain_size_nm2 = smallest_grain_size_nm2
        self.thresholds = None
        self.images = {
            "mask_grains": None,
            "tidied_border": None,
            "tiny_objects_removed": None,
            "objects_removed": None,
            # "labelled_regions": None,
            # "coloured_regions": None,
        }
        self.directions = defaultdict()
        self.minimum_grain_size = None
        self.region_properties = defaultdict()
        self.bounding_boxes = defaultdict()
        self.grainstats = None

    def tidy_border(self, image: np.array, **kwargs) -> np.array:
        """Remove grains touching the border.

        Parameters
        ----------
        image: np.array
            Numpy array representing image.

        Returns
        -------
        np.array
            Numpy array of image with borders tidied.
        """
        LOGGER.info(f"[{self.filename}] : Tidying borders")
        return clear_border(image, **kwargs)

    def label_regions(self, image: np.array) -> np.array:
        """Label regions.

        This method is used twice, once prior to removal of small regions, and again afterwards, hence requiring an
        argument of what image to label.

        Parameters
        ----------
        image: np.array
            Numpy array representing image.

        Returns
        -------
        np.array
            Numpy array of image with objects coloured.
        """
        LOGGER.info(f"[{self.filename}] : Labelling Regions")
        return morphology.label(image, background=0)

    def calc_minimum_grain_size(self, image: np.ndarray) -> float:
        """Calculate the minimum grain size in pixels squared.

        Very small objects are first removed via thresholding before calculating the below extreme.
        """
        region_properties = self.get_region_properties(image)
        grain_areas = np.array([grain.area for grain in region_properties])
        if len(grain_areas > 0):
            # Exclude small objects less than a given threshold first
            grain_areas = grain_areas[
                grain_areas >= threshold(grain_areas, method="otsu", otsu_threshold_multiplier=1.0)
            ]
            self.minimum_grain_size = np.median(grain_areas) - (
                1.5 * (np.quantile(grain_areas, 0.75) - np.quantile(grain_areas, 0.25))
            )
        else:
            self.minimum_grain_size = -1

    def remove_noise(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Removes noise which are objects smaller than the 'smallest_grain_size_nm2'.

        This ensures that the smallest objects ~1px are removed regardless of the size distribution of the grains.

        Parameters
        ----------
        image: np.ndarray
            2D Numpy image to be cleaned.

        Returns
        -------
        np.ndarray
            2D Numpy array of image with objects < smallest_grain_size_nm2 removed.
        """
        LOGGER.info(
            f"[{self.filename}] : Removing noise (< {self.smallest_grain_size_nm2} nm^2"
            "{self.smallest_grain_size_nm2 / (self.pixel_to_nm_scaling**2):.2f} px^2)"
        )
        return morphology.remove_small_objects(
            image, min_size=self.smallest_grain_size_nm2 / (self.pixel_to_nm_scaling**2), **kwargs
        )

    def remove_small_objects(self, image: np.array, **kwargs):
        """Remove small objects from the input image. Threshold determined by the minimum_grain_size variable of the
        Grains class which is in pixels squared.

        Parameters
        ----------
        image: np.ndarray
            2D Numpy image to remove small objects from.
        Returns
        -------
        np.ndarray
            2D Numpy array of image with objects < minimum_grain_size removed.
        """
        # If self.minimum_grain_size is -1, then this means that
        # there were no grains to calculate the minimum grian size from.
        if self.minimum_grain_size != -1:
            small_objects_removed = morphology.remove_small_objects(
                image,
                min_size=self.minimum_grain_size,  # minimum_grain_size is in pixels squared
                **kwargs,
            )
            LOGGER.info(
                f"[{self.filename}] : Removed small objects (< \
{self.minimum_grain_size} px^2 / {self.minimum_grain_size / (self.pixel_to_nm_scaling)**2} nm^2)"
            )
            return small_objects_removed > 0.0
        return image

    def area_thresholding(self, image: np.ndarray, area_thresholds: list):
        """Removes objects larger and smaller than the specified thresholds.

        Parameters
        ----------
        image: np.ndarray
            Image array where the background == 0 and grains are labelled as integers > 0.
        area_thresholds: list
            List of area thresholds (in nanometres squared, not pixels squared), first should be
            the below (smaller) threshold, second above (larger) threshold.

        Returns
        -------
        np.ndarray
            Image where grains outside the thresholds have been removed, as a re-numbered labeled image.

        """
        image_cp = image.copy()
        below, above = area_thresholds
        # if one value is None adjust for comparison
        if above is None:
            above = image.size * self.pixel_to_nm_scaling**2
        if below is None:
            below = 0
        # Get array of grain numbers (discounting zero)
        uniq = np.delete(np.unique(image), 0)
        grain_count = 0
        LOGGER.info(
            f"[{self.filename}] : Area thresholding grains | Thresholds: L: {(below / self.pixel_to_nm_scaling**2):.2f},"
            f"U: {(above / self.pixel_to_nm_scaling**2):.2f} px^2, L: {below:.2f}, U: {above:.2f} nm^2."
        )
        for grain_no in uniq:  # Calculate grian area in nm^2
            grain_area = np.sum(image_cp == grain_no) * (self.pixel_to_nm_scaling**2)
            # Compare area in nm^2 to area thresholds
            if grain_area > above or grain_area < below:
                image_cp[image_cp == grain_no] = 0
            else:
                grain_count += 1
                image_cp[image_cp == grain_no] = grain_count
        return image_cp

    def remove_grain_masks_based_on_height(
        self, image: np.ndarray, labelled_mask: np.ndarray, height_thresholds_std_dev_mult: tuple, direction: str
    ):
        """Removes grains from a labelled mask based on the median height of the grains. Takes two values (lower, upper)
        in a tuple, height_thresholds_std_dev_mult, which get multiplied by the standard deviation of the data image
        to act as thresholds. Grains whose median value is less or greater than these values respectively are removed
        from the mask.

        Parameters
        ----------
        image: np.ndarray
            Numpy 2D image array of the data image that contains the heightmap data for the grains.
        labelled_mask: np.ndarray
            Numpy 2D image array consisting of integer values. 0 is the background and the mask for each grain is an integer
            unique to that grain, eg: grain 1 will have pixel values of all 1, grain 2 will have values of 2, etc.
        height_thresholds_std_dev_mult: list
            A two element tuple whose values correspond to multpliers that get applied to the standard deviation of the
            data image. For example, for the values (1.0, 2.0), and an image whose standard deviation is 3.0, the lower
            grain height threshold would be 1.0 * 3.0 = 3.0, and the upper height threshold would be 2.0 * 3.0 = 6.0.
            So any grain grain mask whose median pixel value is outside of the range 3.0 to 6.0, will be removed.
        direction: str
            A string, either 'above', or 'below', used to ensure the upper / lower thresholds are applied correctly.

        Returns
        -------
        np.ndarray
            Numpy 2D image array of the same format as the labelled_mask input, where the background pixels are 0, and
            grain masks are integers starting at 1, for each grain.
        """

        labelled_mask = labelled_mask.copy()

        # We need to swap the directions if the threshold direction is below, since
        # the upper value will actually be lower. Eg: -4.5 is less than -2.0.
        if direction == "below":
            lower_height_threshold_std_mult = height_thresholds_std_dev_mult[1]
            upper_height_threshold_std_mult = height_thresholds_std_dev_mult[0]
        elif direction == "above":
            lower_height_threshold_std_mult = height_thresholds_std_dev_mult[0]
            upper_height_threshold_std_mult = height_thresholds_std_dev_mult[1]
        else:
            raise ValueError(f"Direction {direction} not valid. Allowed values: above, below")

        # If both threshold multipliers None, simply return the input mask
        if lower_height_threshold_std_mult is None and upper_height_threshold_std_mult is None:
            LOGGER.info(f"[{self.filename}] : threshold for the height based removal of grains is None, skipping.")
            return labelled_mask

        # Contingency for one of the threshold multipliers being None
        if lower_height_threshold_std_mult is None:
            lower_height_threshold = np.inf
        else:
            lower_height_threshold = np.mean(image) + np.std(image) * lower_height_threshold_std_mult
        if upper_height_threshold_std_mult is None:
            upper_height_threshold = -np.inf
        else:
            upper_height_threshold = np.mean(image) + np.std(image) * upper_height_threshold_std_mult

        LOGGER.debug(
            f"[{self.filename}] : remove_grain_masks_based_on_height : upper threshold: {upper_height_threshold}"
        )
        LOGGER.debug(
            f"[{self.filename}] : remove_grain_masks_based_on_height : lower threshold: {lower_height_threshold}"
        )

        # For labelled region in mask
        unique_values = np.delete(np.unique(labelled_mask), 0)
        grain_count = 0
        for grain_number in unique_values:
            LOGGER.debug(f"[{self.filename}] : remove_grain_masks_based_on_height : grain_number: {grain_number}")
            # Get corresponding pixels
            grain_pixels = image[labelled_mask == grain_number]
            # Calculate the median value
            median_grain_value = np.median(grain_pixels)
            # Check if median value is outside of the thresholds
            if median_grain_value > upper_height_threshold:
                LOGGER.debug(
                    f"[{self.filename}] : remove_grain_masks_based_on_height : grain {grain_number} median {median_grain_value} greater than threshold {upper_height_threshold}"
                )
                labelled_mask[labelled_mask == grain_number] = 0
            elif median_grain_value < lower_height_threshold:
                LOGGER.debug(
                    f"[{self.filename}] : remove_grain_masks_based_on_height : grain {grain_number} median {median_grain_value} less than threshold {lower_height_threshold}"
                )
                labelled_mask[labelled_mask == grain_number] = 0
            else:
                LOGGER.debug(
                    f"[{self.filename}] : remove_grain_masks_based_on_height : grain {grain_number} median {median_grain_value} NOT greater than threshold"
                )
                grain_count += 1
                labelled_mask[labelled_mask == grain_number] = grain_count

        return labelled_mask

    def colour_regions(self, image: np.array, **kwargs) -> np.array:
        """Colour the regions.

        Parameters
        ----------
        image: np.array
            Numpy array representing image.

        Returns
        -------
        np.array
            Numpy array of image with objects coloured.
        """

        coloured_regions = label2rgb(image, **kwargs)
        LOGGER.info(f"[{self.filename}] : Coloured regions")
        return coloured_regions

    @staticmethod
    def get_region_properties(image: np.array, **kwargs) -> List:
        """Extract the properties of each region.

        Parameters
        ----------
        image: np.array
            Numpy array representing image

        Returns
        -------
        List
            List of region property objects.
        """
        return regionprops(image, **kwargs)

    def get_bounding_boxes(self, direction) -> Dict:
        """Derive a list of bounding boxes for each region from the derived region_properties

        Parameters
        ----------
        direction: str
            Direction of threshold for which bounding boxes are being calculated.

        Returns
        -------
        dict
            Dictionary of bounding boxes indexed by region area.
        """
        return {region.area: region.area_bbox for region in self.region_properties[direction]}

    def find_grains(self):
        """Find grains."""
        LOGGER.info(f"[{self.filename}] : Thresholding method (grains) : {self.threshold_method}")
        self.thresholds = get_thresholds(
            image=self.image,
            threshold_method=self.threshold_method,
            otsu_threshold_multiplier=self.otsu_threshold_multiplier,
            threshold_std_dev=self.threshold_std_dev,
            absolute=self.threshold_absolute,
        )
        for direction in self.direction:
            LOGGER.info(f"[{self.filename}] : Finding {direction} grains, threshold: ({self.thresholds[direction]})")
            self.directions[direction] = {}
            self.directions[direction]["mask_grains"] = _get_mask(
                self.image,
                thresh=self.thresholds[direction],
                threshold_direction=direction,
                img_name=self.filename,
            )
            self.directions[direction]["labelled_regions_01"] = self.label_regions(
                self.directions[direction]["mask_grains"]
            )
            self.directions[direction]["tidied_border"] = self.tidy_border(
                self.directions[direction]["labelled_regions_01"]
            )
            LOGGER.info(f"[{self.filename}] : Removing noise ({direction})")
            self.directions[direction]["removed_noise"] = self.area_thresholding(
                self.directions[direction]["tidied_border"],
                [self.smallest_grain_size_nm2, None],
            )
            LOGGER.info(f"[{self.filename}] : Removing small / large grains ({direction})")
            # if no area thresholds specified, use otsu
            if self.absolute_area_threshold[direction].count(None) == 2:
                self.calc_minimum_grain_size(self.directions[direction]["removed_noise"])
                self.directions[direction]["removed_small_objects"] = self.remove_small_objects(
                    self.directions[direction]["removed_noise"]
                )
            else:
                self.directions[direction]["removed_small_objects"] = self.area_thresholding(
                    self.directions[direction]["removed_noise"],
                    self.absolute_area_threshold[direction],
                )
            # Ignore grains based on height
            self.directions[direction]["grains_removed_based_on_height"] = self.remove_grain_masks_based_on_height(
                image=self.image,
                labelled_mask=self.directions[direction]["removed_small_objects"],
                height_thresholds_std_dev_mult=self.grain_height_removal_thresholds_std_dev[direction],
                direction=direction,
            )
            self.directions[direction]["labelled_regions_02"] = self.label_regions(
                self.directions[direction]["grains_removed_based_on_height"]
            )
            self.region_properties[direction] = self.get_region_properties(
                self.directions[direction]["labelled_regions_02"]
            )
            LOGGER.info(f"[{self.filename}] : Region properties calculated ({direction})")
            self.directions[direction]["coloured_regions"] = self.colour_regions(
                self.directions[direction]["labelled_regions_02"]
            )
            self.bounding_boxes[direction] = self.get_bounding_boxes(direction=direction)
            LOGGER.info(f"[{self.filename}] : Extracted bounding boxes ({direction})")
