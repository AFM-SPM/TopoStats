"""Find grains in an image."""
# pylint: disable=no-name-in-module
from collections import defaultdict
import logging
from typing import List, Dict
import numpy as np

from skimage.segmentation import clear_border
from skimage.morphology import remove_small_objects, label
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
            "upper": [None, None],
            "lower": [None, None],
        },
        direction: str = None,
        smallest_grain_size_nm: float = None,
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
            Factor by which lower threshold is to be scaled prior to masking.
        threshold_method: str
            Method for determining threshold to mask values, default is 'otsu'.
        threshold_std_dev: dict
            Dictionary of 'lower' and 'upper' factors by which standard deviation is multiplied to derive the threshold if threshold_method is 'std_dev'.
        threshold_absolute: dict
            Dictionary of absolute 'lower' and 'upper' thresholds for grain finding.
        absolute_area_threshold: dict
            Dictionary of upper and lower grain's area thresholds
        direction: str
            Direction for which grains are to be detected, valid values are upper, lower and both.
        """
        self.image = image
        self.filename = filename
        self.pixel_to_nm_scaling = pixel_to_nm_scaling
        self.threshold_method = threshold_method
        self.otsu_threshold_multiplier = otsu_threshold_multiplier
        self.threshold_std_dev = threshold_std_dev
        self.threshold_absolute = threshold_absolute
        self.absolute_area_threshold = absolute_area_threshold
        # Only detect grains for the desired direction
        self.direction = [direction] if direction != "both" else ["upper", "lower"]
        self.smallest_grain_size_nm = smallest_grain_size_nm
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
        return label(image, background=0)

    def calc_minimum_grain_size(self, image: np.ndarray) -> float:
        """Calculate the minimum grain size in pixels squared.

        Very small objects are first removed via thresholding before calculating the lower extreme.
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
        """Removes noise which are objects smaller than the 'smallest_grain_size'.

        This ensures that the smallest objects ~1px are removed regardless of the size distribution of the grains.

        Parameters
        ----------
        image: np.ndarray
            2D Numpy image to be cleaned.

        Returns
        -------
        np.ndarray
            2D Numpy array of image with objects < smallest_grain_size removed.
        """
        LOGGER.info(
            f"[{self.filename}] : Removing noise (< {self.smallest_grain_size} nm^2 \
{self.smallest_grain_size / (self.pixel_to_nm_scaling**2):.2f} px^2)"
        )
        return remove_small_objects(
            image, min_size=self.smallest_grain_size / (self.pixel_to_nm_scaling**2), **kwargs
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
            small_objects_removed = remove_small_objects(
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
            the lower (smaller) threshold, second upper (larger) threshold.

        Returns
        -------
        np.ndarray
            Image where grains outside the thresholds have been removed, as a re-numbered labeled image.

        """
        image_cp = image.copy()
        lower, upper = area_thresholds
        # if one value is None adjust for comparison
        if upper is None:
            upper = image.size * self.pixel_to_nm_scaling**2
        if lower is None:
            lower = 0
        # Get array of grain numbers (discounting zero)
        uniq = np.delete(np.unique(image), 0)
        grain_count = 0
        LOGGER.info(
            f"[{self.filename}] : Area thresholding grains | Thresholds: L:{lower / self.pixel_to_nm_scaling**2:.2f},\
U:{upper / self.pixel_to_nm_scaling**2:.2f} px^2, L:{lower:.2f}, U:{upper:.2f} nm^2."
        )
        for grain_no in uniq:  # Calculate grian area in nm^2
            grain_area = np.sum(image_cp == grain_no) * (self.pixel_to_nm_scaling**2)
            # Compare area in nm^2 to area thresholds
            if grain_area > upper or grain_area < lower:
                image_cp[image_cp == grain_no] = 0
            else:
                grain_count += 1
                image_cp[image_cp == grain_no] = grain_count
        return image_cp

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
        try:
            region_props_count = 0
            for direction in self.direction:
                LOGGER.info(f"[{self.filename}] : Processing {direction} threshold ({self.thresholds[direction]})")
                self.directions[direction] = defaultdict()
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
                    [self.smallest_grain_size, None],
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

                self.directions[direction]["labelled_regions_02"] = self.label_regions(
                    self.directions[direction]["removed_small_objects"]
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
                region_props_count += len(self.region_properties[direction])
            if region_props_count == 0:
                self.region_properties = None
        # FIXME : Identify what exception is raised with images without grains and replace broad except
        except:  # noqa: E722
            LOGGER.info(f"[{self.filename}] : No grains found.")
