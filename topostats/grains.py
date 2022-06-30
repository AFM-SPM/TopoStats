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

# from topostats.utils import get_grains_thresholds
# from topostats.utils import get_grains_mask
from topostats.utils import _get_mask, get_thresholds
from topostats.plottingfuncs import plot_and_save

from collections import defaultdict

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
        # threshold_multiplier: float = None,
        threshold_std_dev: float = None,
        threshold_absolute_lower: float = None,
        threshold_absolute_upper: float = None,
        absolute_smallest_grain_size: float = None,
        background: float = 0.0,
        output_dir: Union[str, Path] = None,
    ):
        self.image = image
        print(f"image: {image}")
        self.filename = filename
        self.pixel_to_nm_scaling = pixel_to_nm_scaling
        self.threshold_method = threshold_method
        # self.threshold_multiplier = threshold_multiplier
        self.threshold_std_dev = threshold_std_dev
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
        self.directions = defaultdict()
        self.minimum_grain_size = None
        self.region_properties = None
        self.bounding_boxes = None
        self.grainstats = None
        Path.mkdir(self.output_dir / self.filename, parents=True, exist_ok=True)

    def gaussian_filter(self, **kwargs) -> np.array:
        """Apply Gaussian filter"""
        LOGGER.info(
            f"[{self.filename}] : Applying Gaussian filter (mode : {self.gaussian_mode}; Gaussian blur (nm) : {self.gaussian_size})."
        )
        self.images["gaussian_filtered"] = gaussian(
            self.image,
            sigma=(self.gaussian_size * self.pixel_to_nm_scaling),
            mode=self.gaussian_mode,
            **kwargs,
        )
        plot_and_save(
            self.images["gaussian_filtered"],
            self.output_dir / self.filename,
            "gaussian_filtered",
        )

    def tidy_border(self, image: np.array, **kwargs) -> np.array:
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
        return clear_border(image, **kwargs)

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
        return label(image, background=self.background)

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
        self.get_region_properties(image)
        grain_areas = np.array([grain.area for grain in self.region_properties])
        # grain_areas = grain_areas[grain_areas > threshold(grain_areas, method=self.threshold_method)]
        self.minimum_grain_size = np.median(grain_areas) - (
            1.5 * (np.quantile(grain_areas, 0.75) - np.quantile(grain_areas, 0.25))
        )

    def remove_noise(self, image: np.array) -> np.array:
        """Removes tiny objects, size set by the config file. This is really important to ensure that the smallest objects ~1px are removed regardless of the size distribution of the grains"""
        LOGGER.info(f"[{self.filename}] : Removing noise (< {self.absolute_smallest_grain_size}")
        return remove_small_objects(image, min_size=self.absolute_smallest_grain_size)

    def remove_small_objects(self, image: np.array, **kwargs):
        """Remove small objects."""
        small_objects_removed = remove_small_objects(
            image,
            min_size=(self.minimum_grain_size * self.pixel_to_nm_scaling),
            **kwargs,
        )
        LOGGER.info(
            f"[{self.filename}] : Removed small objects (< {self.minimum_grain_size * self.pixel_to_nm_scaling})"
        )
        return small_objects_removed

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

    def get_region_properties(self, image: np.array, **kwargs) -> List:
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
        self.region_properties = regionprops(image, **kwargs)
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
        LOGGER.info(f"thresholding method: {self.threshold_method}")
        # self.threshold = self.get_threshold(self.image, self.threshold_method)
        # self.thresholds_dict = get_grains_thresholds(
        self.thresholds = get_thresholds(
            image=self.image,
            threshold_method=self.threshold_method,
            deviation_from_mean=self.threshold_std_dev,
            absolute=(self.threshold_absolute_lower, self.threshold_absolute_upper),
        )
        self.gaussian_filter()
        # self.images["mask_grains"] = self.get_mask()

        #
        for direction, threshold in self.thresholds.items():
            self.directions[direction] = defaultdict()

            # self.directions[direction]["mask"] = get_grains_mask(
            self.directions[direction]["mask"] = _get_mask(
                self.images["gaussian_filtered"], threshold=threshold, threshold_direction=direction
            )

            plot_and_save(
                data=self.directions[direction]["mask"],
                output_dir=self.output_dir / self.filename,
                # "grain_binary_mask_" + str(direction),
                filename=f"grain_binary_mask_{direction}.png",
            )
            self.directions[direction]["tidied_border"] = self.tidy_border(self.directions[direction]["mask"])
            self.directions[direction]["removed_noise"] = self.remove_noise(self.directions[direction]["tidied_border"])

            plot_and_save(
                data=self.directions[direction]["removed_noise"],
                output_dir=self.output_dir / self.filename,
                # "removed_tiny_objects_" + str(direction),
                filename=f"removed_tiny_objects_{direction}.png",
            )

            self.directions[direction]["labelled_regions_01"] = self.label_regions(
                self.directions[direction]["removed_noise"]
            )

            plot_and_save(
                data=self.directions[direction]["labelled_regions_01"],
                output_dir=self.output_dir / self.filename,
                # "labelled_regions_01_" + str(direction),
                filename=f"labelled_regions_01_{direction}.png",
            )

            self.calc_minimum_grain_size(self.directions[direction]["labelled_regions_01"])

            self.directions[direction]["removed_small_objects"] = self.remove_small_objects(
                self.directions[direction]["labelled_regions_01"]
            )

            plot_and_save(
                data=self.directions[direction]["removed_small_objects"],
                output_dir=self.output_dir / self.filename,
                filename=f"removed_small_objects_{direction}.png",
            )

            self.directions[direction]["labelled_regions_02"] = self.label_regions(
                self.directions[direction]["removed_small_objects"]
            )

            plot_and_save(
                data=self.directions[direction]["labelled_regions_02"],
                output_dir=self.output_dir / self.filename,
                # "labelled_regions_01_" + str(direction),
                filename=f"labelled_regions_02_{direction}.png",
            )

            self.get_region_properties(self.directions[direction]["labelled_regions_02"])

            self.directions[direction]["coloured_regions"] = self.colour_regions(
                self.directions[direction]["labelled_regions_02"]
            )

            plot_and_save(
                data=self.directions[direction]["coloured_regions"],
                output_dir=self.output_dir / self.filename,
                # "removed_small_objects_" + str(direction),
                filename=f"removed_small_objects_{direction}.png",
            )

            self.get_bounding_boxes()
