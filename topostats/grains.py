"""Find grains in an image."""
import logging
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
        threshold_method: str = "otsu",
        threshold_multiplier: float = 1.7,
        background: float = 0.0,
        output_dir: Union[str, Path] = None,
    ):
        self.image = image
        self.filename = filename
        self.pixel_to_nm_scaling = pixel_to_nm_scaling
        self.threshold_method = threshold_method
        self.threshold_multiplier = threshold_multiplier
        self.gaussian_size = gaussian_size
        self.gaussian_mode = gaussian_mode
        self.background = background
        self.output_dir = output_dir
        self.threshold = None
        self.images = {
            "gaussian_filtered": None,
            "mask_grains": None,
            "tidied_border": None,
            "objects_removed": None,
            "labelled_regions": None,
            "coloured_regions": None,
        }
        self.minimum_grain_size = None
        self.region_properties = None
        self.bounding_boxes = None
        self.grainstats = None

    def get_threshold(self, **kwargs) -> float:
        """Returns a threshold value based on the stated method multiplied by the threshold multiplier."""
        self.threshold = threshold(self.image, self.threshold_method, **kwargs) * self.threshold_multiplier
        LOGGER.info(f"[{self.filename}] Threshold       : {self.threshold}")

    def gaussian_filter(self, **kwargs) -> np.array:
        """Apply Gaussian filter"""
        LOGGER.info(
            f"[{self.filename}] : Applying Gaussian filter (mode : {self.gaussian_mode}; Gaussian blur (nm) : {self.gaussian_size})."
        )
        self.images["gaussian_filtered"] = gaussian(
            self.image, sigma=(self.gaussian_size * self.pixel_to_nm_scaling), mode=self.gaussian_mode, **kwargs
        )

    def get_mask(self):
        """Create a boolean array of whether points are greater than the given threshold."""
        self.images["mask_grains"] = get_mask(self.images["gaussian_filtered"], self.threshold, self.filename)
        LOGGER.info(f"[{self.filename}] : Created boolean image")

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
        grain_areas = grain_areas[grain_areas >= threshold(grain_areas, method=self.threshold_method)]
        self.minimum_grain_size = np.median(grain_areas) - (
            1.5 * (np.quantile(grain_areas, 0.75) - np.quantile(grain_areas, 0.25))
        )

    def remove_small_objects(self, **kwargs):
        """Remove small objects."""
        self.images["objects_removed"] = remove_small_objects(
            self.images["tidied_border"], min_size=(self.minimum_grain_size * self.pixel_to_nm_scaling), **kwargs
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
        self.get_threshold()
        self.gaussian_filter()
        self.get_mask()
        self.tidy_border()
        self.label_regions(self.images["tidied_border"])
        self.calc_minimum_grain_size(image=self.images["labelled_regions"])
        self.remove_small_objects()
        self.label_regions(self.images["objects_removed"])
        self.get_region_properties()
        self.colour_regions()
        self.get_bounding_boxes()
