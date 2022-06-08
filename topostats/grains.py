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
        absolute_smallest_grain_size: float = None,
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

    def get_threshold(self, **kwargs) -> float:
        """Returns a threshold value based on the stated method multiplied by the threshold multiplier."""
        if self.threshold_method == 'otsu':
            print('getting otsu threshold')
            self.threshold = (None, threshold(self.image, method=self.threshold_method, threshold_multiplier=self.threshold_multiplier, **kwargs))
        elif self.threshold_method == 'std_dev_lower':
            print('getting std dev lower threshold')
            self.threshold = (threshold(self.image, method=self.threshold_method, threshold_multiplier=self.threshold_multiplier, **kwargs), None)
        elif self.threshold_method == 'std_dev_upper':
            print('getting std dev upper threshold')
            self.threshold = (None, threshold(self.image, method=self.threshold_method, threshold_multiplier=self.threshold_multiplier, **kwargs))

        # self.threshold = threshold(self.image, method=self.threshold_method, threshold_multiplier=self.threshold_multiplier, **kwargs)
        LOGGER.info(f"GRAINS THRESHOLDING MULTIPlIER: {self.threshold_multiplier}")
        LOGGER.info(f"[{self.filename}] Threshold       : {self.threshold}")

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
        # if self.threshold_method == 'otsu':
        #     print('otsu')
        #     self.images["mask_grains"] = get_mask(self.images["gaussian_filtered"], (None, self.threshold_multiplier), self.filename)
        #     plot_and_save(self.images['mask_grains'], self.output_dir / self.filename, 'binary_mask_upper')
        # elif self.threshold_method == 'std_dev_lower':
        #     print('std dev lower')
        #     self.images["mask_grains"] = get_mask(self.images["gaussian_filtered"], (self.threshold_multiplier, None), self.filename)
        #     plot_and_save(self.images['mask_grains'], self.output_dir / self.filename, 'binary_mask_lower')
        # elif self.threshold_method == 'std_dev_upper':
        #     print('std dev upper')
        #     self.images["mask_grains"] = get_mask(self.images["gaussian_filtered"], (None, self.threshold_multiplier), self.filename)
        #     plot_and_save(self.images['mask_grains'], self.output_dir / self.filename, 'binary_mask_upper')
        self.images["mask_grains"] = get_mask(self.images["gaussian_filtered"], self.threshold, self.filename)
        plot_and_save(self.images['mask_grains'], self.output_dir / self.filename, 'grain_binary_mask')
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
        # grain_areas = grain_areas[grain_areas > threshold(grain_areas, method=self.threshold_method)]
        self.minimum_grain_size = np.median(grain_areas) - (
            1.5 * (np.quantile(grain_areas, 0.75) - np.quantile(grain_areas, 0.25))
        )

    def remove_tiny_objects(self):
        """Removes tiny objects, size set by the config file"""
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
        self.get_threshold()
        self.gaussian_filter()
        self.get_mask()
        self.tidy_border()
        self.label_regions(self.images["tidied_border"])
        self.calc_minimum_grain_size(image=self.images["labelled_regions"])
        self.remove_tiny_objects()
        self.remove_small_objects()
        self.label_regions(self.images["objects_removed"])
        self.get_region_properties()
        self.colour_regions()
        self.get_bounding_boxes()
