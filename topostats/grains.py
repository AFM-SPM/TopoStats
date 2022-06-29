"""Find grains in an image."""
from collections import defaultdict
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
from topostats.utils import _get_mask, get_thresholds
from topostats.plottingfuncs import plot_and_save

LOGGER = logging.getLogger(LOGGER_NAME)


class Grains:
    """Find grains in an image."""

    def __init__(
        self,
        image: np.ndarray,
        filename: str,
        pixel_to_nm_scaling: float,
        gaussian_size: float = 2,
        gaussian_mode: str = "nearest",
        threshold_method: str = None,
        otsu_threshold_multiplier: float = None,
        threshold_std_dev: float = None,
        threshold_absolute_lower: float = None,
        threshold_absolute_upper: float = None,
        absolute_smallest_grain_size: float = None,
        background: float = 0.0,
        base_output_dir: Union[str, Path] = None,
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
        gaussian_size : Union[int, float]
            Minimum grain size in nanometers (nm).
        gaussian_mode : str
            Mode for filtering (default is 'nearest').
        threshold_multiplier : Union[int, float]
            Factor by which lower threshold is to be scaled prior to masking.
        threshold_method: str
            Method for determining threshold to mask values, default is 'otsu'.
        background : float
        output_dir : Union[str, Path]
            Output directory.
        """
        self.image = image
        print(f"image: {image}")
        self.filename = filename
        self.pixel_to_nm_scaling = pixel_to_nm_scaling
        self.threshold_method = threshold_method
        self.otsu_threshold_multiplier = otsu_threshold_multiplier
        self.threshold_std_dev = threshold_std_dev
        self.threshold_absolute_lower = threshold_absolute_lower
        self.threshold_absolute_upper = threshold_absolute_upper
        self.gaussian_size = gaussian_size
        self.gaussian_mode = gaussian_mode
        self.background = background
        self.base_output_dir = base_output_dir
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
        Path.mkdir(self.base_output_dir, parents=True, exist_ok=True)

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
            self.base_output_dir,
            "gaussian_filtered",
        )
        plot_and_save(self.images["gaussian_filtered"], self.output_dir / self.filename, "gaussian_filtered")

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
        return label(image, background=self.background)

    def calc_minimum_grain_size(self) -> float:
        """Calculate the minimum grain size.

        Very small objects are first removed via thresholding before calculating the lower extreme.
        """
        self.get_region_properties(image)
        grain_areas = np.array([grain.area for grain in self.region_properties])
        if len(grain_areas > 0):
            # grain_areas = grain_areas[grain_areas > threshold(grain_areas, method=self.threshold_method)]
            self.minimum_grain_size = np.median(grain_areas) - (
                1.5 * (np.quantile(grain_areas, 0.75) - np.quantile(grain_areas, 0.25))
            )
        else:
            self.minimum_grain_size = -1

    def remove_noise(self, image: np.ndarray) -> np.ndarray:
        """Removes noise which are objects smaller than the 'absolute_smallest_grain_size'.

        This ensures that the smallest objects ~1px are removed regardless of the size distribution of the grains.

        Parameters
        ----------
        image: np.ndarray
            2D Numpy image to be cleaned.

        Returns
        -------
        np.ndarray
            2D Numpy array of image with objects > absolute_smallest_grain_size removed.
        """
        LOGGER.info(f"[{self.filename}] : Removing noise (< {self.absolute_smallest_grain_size}")
        return remove_small_objects(image, min_size=self.absolute_smallest_grain_size)

    def remove_small_objects(self, image: np.array, **kwargs):
        """Remove small objects."""
        # If self.minimum_grain_size is -1, then this means that there were no grains to calculate the minimum grian size from.
        if self.minimum_grain_size != -1:
            small_objects_removed = remove_small_objects(
                image,
                min_size=(self.minimum_grain_size * self.pixel_to_nm_scaling),
                **kwargs,
            )
            LOGGER.info(
                f"[{self.filename}] : Removed small objects (< {self.minimum_grain_size * self.pixel_to_nm_scaling})"
            )
            return small_objects_removed
        else:
            return image

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
            otsu_threshold_multiplier=self.otsu_threshold_multiplier,
            deviation_from_mean=self.threshold_std_dev,
            absolute=(self.threshold_absolute_lower, self.threshold_absolute_upper),
        )
        try:
            for direction, threshold in self.thresholds.items():

                # Create sub-directory for the upper / lower grains
                self.output_dir = self.base_output_dir / str(direction)
                Path.mkdir(self.output_dir, parents=True, exist_ok=True)
                LOGGER.info(f"Output dir: {self.output_dir}")
                self.directions[direction] = defaultdict()
                self.gaussian_filter()

                self.directions[direction]["mask"] = _get_mask(
                    self.images["gaussian_filtered"], threshold=threshold, threshold_direction=direction
                )

                plot_and_save(
                    data=self.directions[direction]["mask"],
                    output_dir=self.output_dir,
                    filename=f"grain_binary_mask_{direction}.png",
                )
                self.directions[direction]["tidied_border"] = self.tidy_border(self.directions[direction]["mask"])
                self.directions[direction]["removed_noise"] = self.remove_noise(
                    self.directions[direction]["tidied_border"]
                )

                plot_and_save(
                    data=self.directions[direction]["removed_noise"],
                    output_dir=self.output_dir,
                    filename=f"removed_tiny_objects_{direction}.png",
                )

                self.directions[direction]["labelled_regions_01"] = self.label_regions(
                    self.directions[direction]["removed_noise"]
                )

                plot_and_save(
                    data=self.directions[direction]["labelled_regions_01"],
                    output_dir=self.output_dir,
                    filename=f"labelled_regions_01_{direction}.png",
                )

                self.calc_minimum_grain_size(self.directions[direction]["labelled_regions_01"])

                self.directions[direction]["removed_small_objects"] = self.remove_small_objects(
                    self.directions[direction]["labelled_regions_01"]
                )

                plot_and_save(
                    data=self.directions[direction]["removed_small_objects"],
                    output_dir=self.output_dir,
                    filename=f"removed_small_objects_{direction}.png",
                )

                self.directions[direction]["labelled_regions_02"] = self.label_regions(
                    self.directions[direction]["removed_small_objects"]
                )

                plot_and_save(
                    data=self.directions[direction]["labelled_regions_02"],
                    output_dir=self.output_dir,
                    filename=f"labelled_regions_02_{direction}.png",
                )

                self.get_region_properties(self.directions[direction]["labelled_regions_02"])

                self.directions[direction]["coloured_regions"] = self.colour_regions(
                    self.directions[direction]["labelled_regions_02"]
                )

                plot_and_save(
                    data=self.directions[direction]["coloured_regions"],
                    output_dir=self.output_dir,
                    filename=f"removed_small_objects_{direction}.png",
                )
                self.get_bounding_boxes()
        # FIXME : Identify what exception is raised with images without grains and replace broad except
        except:
            LOGGER.info(f"[{self.filename}] : No grains found.")
