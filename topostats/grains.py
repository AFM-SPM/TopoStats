"""Find grains in an image."""
import logging
from os import remove
from pathlib import Path
from typing import Union, List, Dict
import numpy as np

from skimage.filters import gaussian
from skimage.segmentation import clear_border
from skimage.morphology import remove_small_objects, label
from skimage.measure import regionprops
from skimage.color import label2rgb

from topostats.thresholds import threshold
from topostats.logs.logs import LOGGER_NAME
from topostats.utils import get_mask
from topostats.plottingfuncs import plot_and_save

LOGGER = logging.getLogger(LOGGER_NAME)


class Grains:
    """Find grains in an image."""

    def __init__(
        self,
        image: np.array,
        filename: str,
        pixel_to_nm_scaling: float,
        gaussian_size: float = 2,
        gaussian_mode: str = "nearest",
        threshold_method: str = None,
        threshold_multiplier: float = None,
        threshold_multiplier_lower: float = None,
        threshold_multiplier_upper: float = None,
        threshold_absolute_lower: float = None,
        threshold_absolute_upper: float = None,
        absolute_smallest_grain_size: float = None,
        background: float = 0.0,
        output_dir: Union[str, Path] = None,
    ):
        self.image = image
        self.filename = filename
        self.pixel_to_nm_scaling = pixel_to_nm_scaling
        self.threshold_method = threshold_method
        self.threshold_multiplier = threshold_multiplier
        self.threshold_multiplier_lower = threshold_multiplier_lower
        self.threshold_multiplier_upper = threshold_multiplier_upper
        self.threshold_absolute_lower = threshold_absolute_lower
        self.threshold_absolute_upper = threshold_absolute_upper
        self.gaussian_size = gaussian_size
        self.gaussian_mode = gaussian_mode
        self.background = background
        self.output_dir = output_dir
        self.absolute_smallest_grain_size = absolute_smallest_grain_size
        self.threshold = None
        self.images = {
            "gaussian_filtered": None,
            "mask_grains": None,
            "tidied_border": None,
            "tiny_objects_removed": None,
            "objects_removed": None,
            "labelled_regions": None,
            "coloured_regions": None,
        }
        self.minimum_grain_size = None
        self.region_properties = None
        self.bounding_boxes = None
        self.grainstats = None
        Path.mkdir(self.output_dir / self.filename, parents=True, exist_ok=True)

    def get_threshold(self, image: np.array, threshold_method: str, **kwargs) -> float:
        """Sets the value self.threshold that separating the background data out from the data of interest of a 2D heightmap. This data can be either below or above the background data, or both.
        The threshold value that is set, is a tuple, where the first value in the tuple is the lower threshold, and the second value is the upper threshold. If there is not supposed to be a lower or upper threshold, then they are set to be None.

        Parameters
        ----------
        image: np.array
            Image to derive threshold from.
        threshold_method: str
            Method for deriving threshold options are 'otsu' (default), std_dev_lower, std_dev_upper, minimum, mean, yen and triangle

        """

        if "std_dev" in threshold_method:
            if threshold_method == "std_dev_lower":
                thresh = threshold(image, method='mean', **kwargs) - self.threshold_multiplier_lower * np.nanstd(image)
            elif threshold_method == "std_dev_upper":
                thresh = threshold(image, method='mean', **kwargs) + self.threshold_multiplier_upper * np.nanstd(image)
        if "absolute" in threshold_method:
            if threshold_method == "absolute_lower":
                thresh = self.threshold_absolute_lower
            elif threshold_method == "absolute_upper":
                thresh = self.threshold_absolute_upper
        elif threshold_method == "otsu":
            thresh = threshold(image, method='otsu', **kwargs) * self.threshold_multiplier

        LOGGER.info(f"[{self.filename}] : Threshold method: {threshold_method}")
        LOGGER.info(f"[{self.filename}] : Threshold       : {thresh}")
        return thresh

    def gaussian_filter(self, **kwargs) -> np.array:
        """Apply Gaussian filter"""
        LOGGER.info(
            f"[{self.filename}] : Applying Gaussian filter (mode : {self.gaussian_mode}; Gaussian blur (nm) : {self.gaussian_size})."
        )
        self.images["gaussian_filtered"] = gaussian(
            self.image, sigma=(self.gaussian_size * self.pixel_to_nm_scaling), mode=self.gaussian_mode, **kwargs
        )
        plot_and_save(self.images["gaussian_filtered"], self.output_dir / self.filename, 'gaussian_filtered')

    def get_mask(self):
        """Create a boolean array of whether points are greater than the given threshold."""

        # Perhaps I should add a condition that instead checks for 'upper' or 'lower' in the thresholding name to condense this? - Sylvia
        if "std_dev" in self.threshold_method:
            if self.threshold_method == "std_dev_lower":
                mask = get_mask(self.images["gaussian_filtered"], self.threshold, threshold_direction='below', img_name=self.filename)
            elif self.threshold_method == "std_dev_upper":
                mask = get_mask(self.images["gaussian_filtered"], self.threshold, threshold_direction='above', img_name=self.filename)
        elif "absolute" in self.threshold_method:
            if self.threshold_method == "absolute_lower":
                mask = get_mask(self.images["gaussian_filtered"], self.threshold, threshold_direction='below', img_name=self.filename)
            elif self.threshold_method == "absolute_upper":
                mask = get_mask(self.images["gaussian_filtered"], self.threshold, threshold_direction='above', img_name=self.filename)
        elif self.threshold_method == "otsu":
            mask = get_mask(self.images["gaussian_filtered"], self.threshold, threshold_direction='above', img_name=self.filename)
        
        plot_and_save(mask, self.output_dir / self.filename, 'grain_binary_mask')
        LOGGER.info(f"[{self.filename}] : Created boolean image")
        return mask

    def tidy_border(self, **kwargs) -> np.array:
        """Remove grains touching the border

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
        self.images["tidied_border"] = clear_border(self.images["mask_grains"], **kwargs)

    def label_regions(self, image: np.array) -> np.array:
        """Label regions.

        This method is used twice, once prior to removal of small regions, and again afterwards, hence requiring and
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
        self.images["labelled_regions"] = label(image, background=self.background)

    def calc_minimum_grain_size(self, image: np.array) -> float:
        """Calculate the minimum grain size.

        Very small objects are first removed via thresholding before calculating the lower extreme.

        Parameters
        ----------
        image : np.array
            Numpy array of regions of interest after labelling.

        Returns
        -------
        float
            Threshold for minimum grain size.
        """
        self.get_region_properties()
        grain_areas = np.array([grain.area for grain in self.region_properties])
        # grain_areas = grain_areas[grain_areas > threshold(grain_areas, method=self.threshold_method)]
        self.minimum_grain_size = np.median(grain_areas) - (
            1.5 * (np.quantile(grain_areas, 0.75) - np.quantile(grain_areas, 0.25))
        )

    def remove_tiny_objects(self):
        """Removes tiny objects, size set by the config file. This is really important to ensure that the smallest objects ~1px are removed regardless of the size distribution of the grains"""
        self.images["tiny_objects_removed"] = remove_small_objects(self.images["tidied_border"], min_size=self.absolute_smallest_grain_size)
        LOGGER.info(f"[{self.filename}] : Removed tiny objects (< {self.absolute_smallest_grain_size}")

    def remove_small_objects(self, **kwargs):
        """Remove small objects."""
        self.images["objects_removed"] = remove_small_objects(
            self.images["tiny_objects_removed"], min_size=(self.minimum_grain_size * self.pixel_to_nm_scaling), **kwargs
        )
        LOGGER.info(
            f"[{self.filename}] : Removed small objects (< {self.minimum_grain_size * self.pixel_to_nm_scaling})"
        )

    def colour_regions(self, **kwargs) -> np.array:
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
        self.images["coloured_regions"] = label2rgb(self.images["labelled_regions"], **kwargs)
        LOGGER.info(f"[{self.filename}] : Coloured regions")

    def get_region_properties(self, **kwargs) -> List:
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
        self.region_properties = regionprops(self.images["labelled_regions"], **kwargs)
        LOGGER.info(f"[{self.filename}] : Region properties calculated")

    def get_bounding_boxes(self) -> Dict:
        """Derive a list of bounding boxes for each region from the derived region_properties

        Returns
        -------
        dict
            Dictionary of bounding boxes indexed by region area.
        """
        LOGGER.info(f"[{self.filename}] : Extracting bounding boxes")
        self.bounding_boxes = {region.area: region.area_bbox for region in self.region_properties}

    def find_grains(self):
        """Find grains."""
        LOGGER.info(f'thresholding method: {self.threshold_method}')
        self.threshold = self.get_threshold(self.image, self.threshold_method)
        self.gaussian_filter()
        self.images["mask_grains"] = self.get_mask()
        self.tidy_border()
        self.label_regions(self.images["tidied_border"])
        self.calc_minimum_grain_size(image=self.images["labelled_regions"])
        self.remove_tiny_objects()
        self.remove_small_objects()
        self.label_regions(self.images["objects_removed"])
        self.get_region_properties()
        self.colour_regions()
        self.get_bounding_boxes()
