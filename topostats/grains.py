"""Find grains in an image."""

# pylint: disable=no-name-in-module
from __future__ import annotations

import logging
import sys
from collections import defaultdict
from dataclasses import dataclass

import keras
import numpy as np
import numpy.typing as npt
from skimage import morphology
from skimage.color import label2rgb
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation
from skimage.segmentation import clear_border

from topostats.logs.logs import LOGGER_NAME
from topostats.thresholds import threshold
from topostats.unet_masking import (
    iou_loss,
    make_bounding_box_square,
    mean_iou,
    pad_bounding_box,
    predict_unet,
)
from topostats.utils import _get_mask, get_thresholds

LOGGER = logging.getLogger(LOGGER_NAME)

# pylint: disable=fixme
# pylint: disable=line-too-long
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-arguments
# pylint: disable=bare-except
# pylint: disable=dangerous-default-value
# pylint: disable=too-many-lines
# pylint: disable=too-many-public-methods


@dataclass
class GrainCrop:
    """
    Dataclass for storing the crops of grains.

    Attributes
    ----------
    image : npt.NDArray[np.float32]
        2-D Numpy array of the cropped image.
    mask : npt.NDArray[np.bool_]
        3-D Numpy tensor of the cropped mask.
    padding : int
        Padding added to the bounding box of the grain during cropping.
    bbox: tuple[int, int, int, int]
        Bounding box of the crop including padding.
    pixel_to_nm_scaling : float
        Pixel to nanometre scaling factor for the crop
    filename : str
        Filename of the image from which the crop was taken.
    """

    image: npt.NDArray[np.float32]
    mask: npt.NDArray[np.bool_]
    padding: int
    bbox: tuple[int, int, int, int]
    pixel_to_nm_scaling: float
    filename: str

    def __post_init__(self) -> None:
        """Post initialisation checks."""
        # This is where we can add post-initialisation checks but not needed at the moment.

    # Important note about mypy and dataclasses:
    # mypy takes issue with the re-definition of the class attributes into property methods.
    # Eg: how `image` is defined as both a property and a class attribute. I am ignoring this
    # here as we don't currently require mypy for successful PRs but something to be aware of.
    # We could get around this with dummy names and using the `post_init` method but it would be
    # a little messy.

    @property
    def image(self) -> npt.NDArray[np.float32]:
        """
        Getter for the image.

        Returns
        -------
        npt.NDArray
            Numpy array of the image.
        """
        return self._image

    @image.setter
    def image(self, value: npt.NDArray[np.float32]):
        """
        Setter for the image.

        Parameters
        ----------
        value : npt.NDArray
            Numpy array of the image.

        Raises
        ------
        ValueError
            If the image is not square.
        """
        if value.shape[0] != value.shape[1]:
            raise ValueError(f"Image is not square: {value.shape}")
        self._image = value

    @property
    def mask(self) -> npt.NDArray[np.bool_]:
        """
        Getter for the mask.

        Returns
        -------
        npt.NDArray
            Numpy array of the mask.
        """
        return self._mask

    @mask.setter
    def mask(self, value: npt.NDArray[np.bool_]):
        """
        Setter for the mask.

        Parameters
        ----------
        value : npt.NDArray
            Numpy array of the mask.

        Raises
        ------
        ValueError
            If the mask dimensions do not match the image.
        """
        if value.shape[0] != self.image.shape[0] or value.shape[1] != self.image.shape[1]:
            raise ValueError(f"Mask dimensions do not match image: {value.shape} vs {self.image.shape}")
        self._mask = value

    @property
    def padding(self) -> int:
        """
        Getter for the padding.

        Returns
        -------
        int
            The padding amount.
        """
        return self._padding

    @padding.setter
    def padding(self, value: int):
        """
        Setter for the padding.

        Parameters
        ----------
        value : int
            Padding amount.

        Raises
        ------
        ValueError
            If the padding is not an integer or is less than 1.
        """
        if not isinstance(value, int):
            raise ValueError(f"Padding must be an integer, but is {value}")
        if value < 1:
            raise ValueError(f"Padding must be >= 1, but is {value}")
        self._padding = value

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """
        Getter for the bounding box.

        Returns
        -------
        tuple
            Bounding box of the crop.

        Raises
        ------
        ValueError
            If the bounding box is not square.
        """
        return self._bbox

    @bbox.setter
    def bbox(self, value: tuple[int, int, int, int]):
        """
        Setter for the bounding box.

        Parameters
        ----------
        value : tuple[int, int, int, int]
            Bounding box of the crop.
        """
        if len(value) != 4:
            raise ValueError(f"Bounding box must have 4 elements, but has {len(value)}")
        if value[2] - value[0] != value[3] - value[1]:
            raise ValueError(
                f"Bounding box is not square: {value}, size: {value[2] - value[0]} x {value[3] - value[1]}"
            )
        self._bbox = value

    @property
    def pixel_to_nm_scaling(self) -> float:
        """
        Getter for the pixel to nanometre scaling factor.

        Returns
        -------
        float
            Pixel to nanometre scaling factor.
        """
        return self._pixel_to_nm_scaling

    @pixel_to_nm_scaling.setter
    def pixel_to_nm_scaling(self, value: float):
        """
        Setter for the pixel to nanometre scaling factor.

        Parameters
        ----------
        value : float
            Pixel to nanometre scaling factor.
        """
        self._pixel_to_nm_scaling = value

    @property
    def filename(self) -> str:
        """
        Getter for the filename.

        Returns
        -------
        str
            The image filename.
        """
        return self._filename

    @filename.setter
    def filename(self, value: str):
        """
        Setter for the filename.

        Parameters
        ----------
        value : str
            Image filename.
        """
        self._filename = value

    def __eq__(self, other: object) -> bool:
        """
        Check if two GrainCrop objects are equal.

        Parameters
        ----------
        other : object
            Object to compare to.

        Returns
        -------
        bool
            True if the objects are equal, False otherwise.
        """
        if not isinstance(other, GrainCrop):
            return False
        return (
            np.array_equal(self.image, other.image)
            and np.array_equal(self.mask, other.mask)
            and self.padding == other.padding
            and self.bbox == other.bbox
            and self.pixel_to_nm_scaling == other.pixel_to_nm_scaling
            and self.filename == other.filename
        )

    def debug_locate_difference(self, other: object) -> None:
        """
        Debug function to find the culprit when two GrainCrop objects are not equal.

        Parameters
        ----------
        other : object
            Object to compare to.

        Raises
        ------
        ValueError
            If the objects are not equal
        """
        if not isinstance(other, GrainCrop):
            raise ValueError(f"Cannot compare GrainCrop with {type(other)}")
        if not np.array_equal(self.image, other.image):
            raise ValueError("Image is different")
        if not np.array_equal(self.mask, other.mask):
            raise ValueError("Mask is different")
        if self.padding != other.padding:
            raise ValueError("Padding is different")
        if self.bbox != other.bbox:
            raise ValueError("Bounding box is different")
        if self.pixel_to_nm_scaling != other.pixel_to_nm_scaling:
            raise ValueError("Pixel to nm scaling is different")
        if self.filename != other.filename:
            raise ValueError("Filename is different")
        LOGGER.info("Cannot find difference between graincrops")


def validate_full_mask_tensor_shape(array: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    """
    Validate the shape of the full mask tensor.

    Parameters
    ----------
    array : npt.NDArray
        Numpy array to validate.

    Returns
    -------
    npt.NDArray
        Numpy array if valid.
    """
    if len(array.shape) != 3 or array.shape[2] < 2 or array.shape[1] != array.shape[0]:
        raise ValueError(f"Full mask tensor must be NxNxC with C >= 2 but has shape {array.shape}")
    return array


@dataclass
class GrainCropsDirection:
    """
    Dataclass for storing the crops of grains in a particular imaging direction.

    Attributes
    ----------
    full_mask_tensor : npt.NDArray[np.bool_]
        Boolean NxNx3 array of the full mask tensor.
    crops : GrainCrops
        Grain crops.
    """

    crops: dict[int, GrainCrop]
    full_mask_tensor: npt.NDArray[np.bool_]

    def __post_init__(self):
        """
        Validate the full mask tensor shape.

        Raises
        ------
        ValueError
            If the full mask tensor shape is invalid.
        """
        self._full_mask_tensor = validate_full_mask_tensor_shape(self.full_mask_tensor)

    @property
    def full_mask_tensor(self) -> npt.NDArray[np.bool_]:
        """
        Getter for the full mask tensor.

        Returns
        -------
        npt.NDArray
            Numpy array of the full mask tensor.
        """
        return self._full_mask_tensor

    @full_mask_tensor.setter
    def full_mask_tensor(self, value: npt.NDArray[np.bool_]):
        """
        Setter for the full mask tensor.

        Parameters
        ----------
        value : npt.NDArray
            Numpy array of the full mask tensor.
        """
        self._full_mask_tensor = validate_full_mask_tensor_shape(value).astype(np.bool_)

    def __eq__(self, other: object) -> bool:
        """
        Check if two GrainCropsDirection objects are equal.

        Parameters
        ----------
        other : object
            Object to compare to.

        Returns
        -------
        bool
            True if the objects are equal, False otherwise.
        """
        if not isinstance(other, GrainCropsDirection):
            return False
        return self.crops == other.crops and np.array_equal(self.full_mask_tensor, other.full_mask_tensor)

    def locate_difference(self, other: object) -> None:
        """
        Debug function to find the culprit when two GrainCropsDirection objects are not equal.

        Parameters
        ----------
        other : object
            Object to compare to.

        Raises
        ------
        ValueError
            If the objects are not equal.
        """
        if not isinstance(other, GrainCropsDirection):
            raise ValueError(f"Cannot compare GrainCropsDirection with {type(other)}")
        for crop_index, crop in self.crops.items():
            if crop != other.crops[crop_index]:
                LOGGER.info(f"Grain crop {crop_index} is different:")
                crop.locate_difference(other.crops[crop_index])
        if not np.array_equal(self.full_mask_tensor, other.full_mask_tensor):
            raise ValueError("Full mask tensor is different")

        LOGGER.info("Cannot find difference between graincrops")

    def update_full_mask_tensor(self):
        """Update the full mask tensor from the grain crops."""
        self.full_mask_tensor = Grains.construct_full_mask_from_graincrops(
            graincrops=self.crops,
            image_shape=self.full_mask_tensor.shape[:2],
        )


@dataclass
class ImageGrainCrops:
    """
    Dataclass for storing the crops of grains in an image.

    Attributes
    ----------
    above : GrainCropDirection | None
        Grains in the above direction.
    below : GrainCropDirection | None
        Grains in the below direction.
    """

    above: GrainCropsDirection | None
    below: GrainCropsDirection | None

    def __eq__(self, other: object) -> bool:
        """
        Check if two ImageGrainCrops objects are equal.

        Parameters
        ----------
        other : object
            Object to compare to.

        Returns
        -------
        bool
            True if the objects are equal, False otherwise.
        """
        if not isinstance(other, ImageGrainCrops):
            return False
        return self.above == other.above and self.below == other.below

    def locate_difference(self, other: object) -> None:
        """
        Debug function to find the culprit when two ImageGrainCrops objects are not equal.

        Parameters
        ----------
        other : object
            Object to compare to.

        Raises
        ------
        ValueError
            If the objects are not equal.
        """
        if not isinstance(other, ImageGrainCrops):
            raise ValueError(f"Cannot compare ImageGrainCrops with {type(other)}")
        if self.above is not None:
            if self.above != other.above:
                LOGGER.info("Above grains are different")
                self.above.locate_difference(other.above)
        else:
            if other.above is not None:
                raise ValueError("Above grains are different")
        if self.below is not None:
            if self.below != other.below:
                LOGGER.info("Below grains are different")
                self.below.locate_difference(other.below)
        else:
            if other.below is not None:
                raise ValueError("Below grains are different")

        LOGGER.info("Cannot find difference between image grain crops")


class Grains:
    """
    Find grains in an image.

    Parameters
    ----------
    image : npt.NDArray
        2-D Numpy array of image.
    filename : str
        File being processed (used in logging).
    pixel_to_nm_scaling : float
        Scaling of pixels to nanometres.
    grain_crop_padding : int
        Padding to add to the bounding box of the grain during cropping.
    unet_config : dict[str, str | int | float | tuple[int | None, int, int, int] | None]
        Configuration for the UNet model.
        model_path: str
            Path to the UNet model.
        upper_norm_bound: float
            Upper bound for normalising the image.
        lower_norm_bound: float
            Lower bound for normalising the image.
    threshold_method : str
        Method for determining thershold to mask values, default is 'otsu'.
    otsu_threshold_multiplier : float
        Factor by which the below threshold is to be scaled prior to masking.
    threshold_std_dev : dict
        Dictionary of 'below' and 'above' factors by which standard deviation is multiplied to derive the threshold
        if threshold_method is 'std_dev'.
    threshold_absolute : dict
        Dictionary of absolute 'below' and 'above' thresholds for grain finding.
    absolute_area_threshold : dict
        Dictionary of above and below grain's area thresholds.
    direction : str
        Direction for which grains are to be detected, valid values are 'above', 'below' and 'both'.
    smallest_grain_size_nm2 : float
        Whether or not to remove grains that intersect the edge of the image.
    remove_edge_intersecting_grains : bool
        Direction for which grains are to be detected, valid values are 'above', 'below' and 'both'.
    classes_to_merge : list[tuple[int, int]] | None
        List of tuples of classes to merge.
    vetting : dict | None
        Dictionary of vetting parameters.
    """

    # pylint: disable=too-many-locals
    def __init__(
        self,
        image: npt.NDArray,
        filename: str,
        pixel_to_nm_scaling: float,
        grain_crop_padding: int = 1,
        unet_config: dict[str, str | int | float | tuple[int | None, int, int, int] | None] | None = None,
        threshold_method: str | None = None,
        otsu_threshold_multiplier: float | None = None,
        threshold_std_dev: dict | None = None,
        threshold_absolute: dict | None = None,
        absolute_area_threshold: dict | None = None,
        direction: str | None = None,
        smallest_grain_size_nm2: float | None = None,
        remove_edge_intersecting_grains: bool = True,
        classes_to_merge: list[tuple[int, int]] | None = None,
        vetting: dict | None = None,
    ):
        """
        Initialise the class.

        Parameters
        ----------
        image : npt.NDArray
            2-D Numpy array of image.
        filename : str
            File being processed (used in logging).
        pixel_to_nm_scaling : float
            Scaling of pixels to nanometres.
        grain_crop_padding : int
            Padding to add to the bounding box of grains during cropping.
        unet_config : dict[str, str | int | float | tuple[int | None, int, int, int] | None]
            Configuration for the UNet model.
            model_path: str
                Path to the UNet model.
            upper_norm_bound: float
                Upper bound for normalising the image.
            lower_norm_bound: float
                Lower bound for normalising the image.
        threshold_method : str
            Method for determining thershold to mask values, default is 'otsu'.
        otsu_threshold_multiplier : float
            Factor by which the below threshold is to be scaled prior to masking.
        threshold_std_dev : dict
            Dictionary of 'below' and 'above' factors by which standard deviation is multiplied to derive the threshold
            if threshold_method is 'std_dev'.
        threshold_absolute : dict
            Dictionary of absolute 'below' and 'above' thresholds for grain finding.
        absolute_area_threshold : dict
            Dictionary of above and below grain's area thresholds.
        direction : str
            Direction for which grains are to be detected, valid values are 'above', 'below' and 'both'.
        smallest_grain_size_nm2 : float
            Whether or not to remove grains that intersect the edge of the image.
        remove_edge_intersecting_grains : bool
            Direction for which grains are to be detected, valid values are 'above', 'below' and 'both'.
        classes_to_merge : list[tuple[int, int]] | None
            List of tuples of classes to merge.
        vetting : dict | None
            Dictionary of vetting parameters.
        """
        if unet_config is None:
            unet_config = {
                "model_path": None,
                "grain_crop_padding": 0,
                "upper_norm_bound": 1.0,
                "lower_norm_bound": 0.0,
            }
        if absolute_area_threshold is None:
            absolute_area_threshold = {"above": [None, None], "below": [None, None]}
        self.image = image
        self.filename = filename
        self.pixel_to_nm_scaling = pixel_to_nm_scaling
        self.threshold_method = threshold_method
        self.otsu_threshold_multiplier = otsu_threshold_multiplier
        self.threshold_std_dev = threshold_std_dev
        self.threshold_absolute = threshold_absolute
        self.absolute_area_threshold = absolute_area_threshold
        # Only detect grains for the desired direction
        self.direction = [direction] if direction != "both" else ["above", "below"]
        self.smallest_grain_size_nm2 = smallest_grain_size_nm2
        self.remove_edge_intersecting_grains = remove_edge_intersecting_grains
        self.thresholds: dict[str, float] | None = None
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
        self.grain_crop_padding = grain_crop_padding
        self.unet_config = unet_config
        self.vetting = vetting
        self.classes_to_merge = classes_to_merge

        # Hardcoded minimum pixel size for grains. This should not be able to be changed by the user as this is
        # determined by what is processable by the rest of the pipeline.
        self.minimum_grain_size_px = 10
        self.minimum_bbox_size_px = 5

        self.image_grain_crops = ImageGrainCrops(
            above=None,
            below=None,
        )

    def tidy_border(self, image: npt.NDArray, **kwargs) -> npt.NDArray:
        """
        Remove grains touching the border.

        Parameters
        ----------
        image : npt.NDarray
            2-D Numpy array representing the image.
        **kwargs
            Arguments passed to 'skimage.segmentation.clear_border(**kwargs)'.

        Returns
        -------
        npt.NDarray
            2-D Numpy array of image without objects touching the border.
        """
        LOGGER.debug(f"[{self.filename}] : Tidying borders")
        return clear_border(image, **kwargs)

    @staticmethod
    def label_regions(image: npt.NDArray, background: int = 0) -> npt.NDArray:
        """
        Label regions.

        This method is used twice, once prior to removal of small regions and again afterwards which is why an image
        must be supplied rather than using 'self'.

        Parameters
        ----------
        image : npt.NDArray
            2-D Numpy array of image.
        background : int
            Value used to indicate background of image. Default = 0.

        Returns
        -------
        npt.NDArray
            2-D Numpy array of image with regions numbered.
        """
        return morphology.label(image, background)

    def calc_minimum_grain_size(self, image: npt.NDArray) -> float:
        """
        Calculate the minimum grain size in pixels squared.

        Very small objects are first removed via thresholding before calculating the below extreme.

        Parameters
        ----------
        image : npt.NDArray
            2-D Numpy image from which to calculate the minimum grain size.

        Returns
        -------
        float
            Minimum grains size in pixels squared. If there are areas a value of -1 is returned.
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

    def remove_noise(self, image: npt.NDArray, **kwargs) -> npt.NDArray:
        """
        Remove noise which are objects smaller than the 'smallest_grain_size_nm2'.

        This ensures that the smallest objects ~1px are removed regardless of the size distribution of the grains.

        Parameters
        ----------
        image : npt.NDArray
            2-D Numpy array to be cleaned.
        **kwargs
            Arguments passed to 'skimage.morphology.remove_small_objects(**kwargs)'.

        Returns
        -------
        npt.NDArray
            2-D Numpy array of image with objects < smallest_grain_size_nm2 removed.
        """
        LOGGER.debug(
            f"[{self.filename}] : Removing noise (< {self.smallest_grain_size_nm2} nm^2"
            "{self.smallest_grain_size_nm2 / (self.pixel_to_nm_scaling**2):.2f} px^2)"
        )
        return morphology.remove_small_objects(
            image, min_size=self.smallest_grain_size_nm2 / (self.pixel_to_nm_scaling**2), **kwargs
        )

    def remove_small_objects(self, image: np.array, **kwargs) -> npt.NDArray:
        """
        Remove small objects from the input image.

        Threshold determined by the minimum grain size, in pixels squared, of the classes initialisation.

        Parameters
        ----------
        image : np.array
            2-D Numpy array to remove small objects from.
        **kwargs
            Arguments passed to 'skimage.morphology.remove_small_objects(**kwargs)'.

        Returns
        -------
        npt.NDArray
            2-D Numpy array of image with objects < minimumm_grain_size removed.
        """
        # If self.minimum_grain_size is -1, then this means that
        # there were no grains to calculate the minimum grian size from.
        if self.minimum_grain_size != -1:
            small_objects_removed = morphology.remove_small_objects(
                image.astype(bool),
                min_size=self.minimum_grain_size,  # minimum_grain_size is in pixels squared
                **kwargs,
            )
            LOGGER.debug(
                f"[{self.filename}] : Removed small objects (< \
{self.minimum_grain_size} px^2 / {self.minimum_grain_size / (self.pixel_to_nm_scaling)**2} nm^2)"
            )
            return small_objects_removed > 0.0
        return image

    def remove_objects_too_small_to_process(
        self, image: npt.NDArray, minimum_size_px: int, minimum_bbox_size_px: int
    ) -> npt.NDArray[np.bool_]:
        """
        Remove objects whose dimensions in pixels are too small to process.

        Parameters
        ----------
        image : npt.NDArray
            2-D Numpy array of image.
        minimum_size_px : int
            Minimum number of pixels for an object.
        minimum_bbox_size_px : int
            Limit for the minimum dimension of an object in pixels. Eg: 5 means the object's bounding box must be at
            least 5x5.

        Returns
        -------
        npt.NDArray
            2-D Numpy array of image with objects removed that are too small to process.
        """
        labelled_image = label(image)
        region_properties = self.get_region_properties(labelled_image)
        for region in region_properties:
            # If the number of true pixels in the region is less than the minimum number of pixels, remove the region
            if region.area < minimum_size_px:
                labelled_image[labelled_image == region.label] = 0
            bbox_width = region.bbox[2] - region.bbox[0]
            bbox_height = region.bbox[3] - region.bbox[1]
            # If the minimum dimension of the bounding box is less than the minimum dimension, remove the region
            if min(bbox_width, bbox_height) < minimum_bbox_size_px:
                labelled_image[labelled_image == region.label] = 0

        return labelled_image.astype(bool)

    def area_thresholding(self, image: npt.NDArray, area_thresholds: tuple) -> npt.NDArray:
        """
        Remove objects larger and smaller than the specified thresholds.

        Parameters
        ----------
        image : npt.NDArray
            Image array where the background == 0 and grains are labelled as integers >0.
        area_thresholds : tuple
            List of area thresholds (in nanometres squared, not pixels squared), first is the lower limit for size,
            second is the upper.

        Returns
        -------
        npt.NDArray
            Array with small and large objects removed.
        """
        image_cp = image.copy()
        lower_size_limit, upper_size_limit = area_thresholds
        # if one value is None adjust for comparison
        if upper_size_limit is None:
            upper_size_limit = image.size * self.pixel_to_nm_scaling**2
        if lower_size_limit is None:
            lower_size_limit = 0
        # Get array of grain numbers (discounting zero)
        uniq = np.delete(np.unique(image), 0)
        grain_count = 0
        LOGGER.debug(
            f"[{self.filename}] : Area thresholding grains | Thresholds: L: {(lower_size_limit / self.pixel_to_nm_scaling**2):.2f},"
            f"U: {(upper_size_limit / self.pixel_to_nm_scaling**2):.2f} px^2, L: {lower_size_limit:.2f}, U: {upper_size_limit:.2f} nm^2."
        )
        for grain_no in uniq:  # Calculate grian area in nm^2
            grain_area = np.sum(image_cp == grain_no) * (self.pixel_to_nm_scaling**2)
            # Compare area in nm^2 to area thresholds
            if grain_area > upper_size_limit or grain_area < lower_size_limit:
                image_cp[image_cp == grain_no] = 0
            else:
                grain_count += 1
                image_cp[image_cp == grain_no] = grain_count
        return image_cp

    def colour_regions(self, image: npt.NDArray, **kwargs) -> npt.NDArray:
        """
        Colour the regions.

        Parameters
        ----------
        image : npt.NDArray
            2-D array of labelled regions to be coloured.
        **kwargs
            Arguments passed to 'skimage.color.label2rgb(**kwargs)'.

        Returns
        -------
        np.array
            Numpy array of image with objects coloured.
        """
        coloured_regions = label2rgb(image, **kwargs)
        LOGGER.debug(f"[{self.filename}] : Coloured regions")
        return coloured_regions

    @staticmethod
    def get_region_properties(image: np.array, **kwargs) -> list:
        """
        Extract the properties of each region.

        Parameters
        ----------
        image : np.array
            Numpy array representing image.
        **kwargs :
            Arguments passed to 'skimage.measure.regionprops(**kwargs)'.

        Returns
        -------
        list
            List of region property objects.
        """
        return regionprops(image, **kwargs)

    def get_bounding_boxes(self, direction: str) -> dict:
        """
        Derive a list of bounding boxes for each region from the derived region_properties.

        Parameters
        ----------
        direction : str
            Direction of threshold for which bounding boxes are being calculated.

        Returns
        -------
        dict
            Dictionary of bounding boxes indexed by region area.
        """
        return {region.area: region.area_bbox for region in self.region_properties[direction]}

    # Sylvia: This function is more readable and easier to work on if we don't split it up into smaller functions.
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements
    def find_grains(self) -> None:
        """Find grains."""
        LOGGER.debug(f"[{self.filename}] : Thresholding method (grains) : {self.threshold_method}")
        self.thresholds = get_thresholds(
            image=self.image,
            threshold_method=self.threshold_method,
            otsu_threshold_multiplier=self.otsu_threshold_multiplier,
            threshold_std_dev=self.threshold_std_dev,
            absolute=self.threshold_absolute,
        )

        # Create an ImageGrainCrops object to store the grain crops
        image_grain_crops = ImageGrainCrops(above=None, below=None)

        for direction in self.direction:
            LOGGER.debug(f"[{self.filename}] : Finding {direction} grains, threshold: ({self.thresholds[direction]})")
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

            if self.remove_edge_intersecting_grains:
                self.directions[direction]["tidied_border"] = self.tidy_border(
                    self.directions[direction]["labelled_regions_01"]
                )
            else:
                self.directions[direction]["tidied_border"] = self.directions[direction]["labelled_regions_01"]

            LOGGER.debug(f"[{self.filename}] : Removing noise ({direction})")
            self.directions[direction]["removed_noise"] = self.area_thresholding(
                self.directions[direction]["tidied_border"],
                [self.smallest_grain_size_nm2, None],
            )

            LOGGER.debug(f"[{self.filename}] : Removing small / large grains ({direction})")
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
            self.directions[direction]["removed_objects_too_small_to_process"] = (
                self.remove_objects_too_small_to_process(
                    image=self.directions[direction]["removed_small_objects"],
                    minimum_size_px=self.minimum_grain_size_px,
                    minimum_bbox_size_px=self.minimum_bbox_size_px,
                )
            )
            self.directions[direction]["labelled_regions_02"] = self.label_regions(
                self.directions[direction]["removed_objects_too_small_to_process"]
            )

            self.region_properties[direction] = self.get_region_properties(
                self.directions[direction]["labelled_regions_02"]
            )
            LOGGER.debug(f"[{self.filename}] : Region properties calculated ({direction})")
            self.bounding_boxes[direction] = self.get_bounding_boxes(direction=direction)
            LOGGER.debug(f"[{self.filename}] : Extracted bounding boxes ({direction})")

            # If there are no grains, then later steps will fail, so skip the stages if no grains are found.
            if len(self.region_properties[direction]) > 0:
                # Grains found

                # Create a tensor out of the grain mask of shape NxNx2, where the two classes are a binary background
                # mask and the second is a binary grain mask. This is because we want to support multiple classes, and
                # so we standardise so that the first layer is background mask, then feature mask 1, then feature mask
                # 2 etc.

                # Get a binary mask where 1s are background and 0s are grains
                labelled_regions_background_mask = np.where(
                    self.directions[direction]["labelled_regions_02"] == 0, 1, 0
                )

                # Create a tensor out of the background and foreground masks
                full_mask_tensor = np.stack(
                    [
                        labelled_regions_background_mask,
                        self.directions[direction]["labelled_regions_02"],
                    ],
                    axis=-1,
                ).astype(np.int32)

                # Extract tensor mask crops of each grain.
                graincrops = self.extract_grains_from_full_image_tensor(
                    image=self.image,
                    full_mask_tensor=full_mask_tensor,
                    padding=self.grain_crop_padding,
                    pixel_to_nm_scaling=self.pixel_to_nm_scaling,
                    filename=self.filename,
                )

                # Optionally run a user-supplied u-net model on the grains to improve the segmentation
                if self.unet_config["model_path"] is not None:
                    # Run unet segmentation on only the class 1 layer of the labelled_regions_02. Need to make this configurable
                    # later on along with all the other hardcoded class 1s.
                    graincrops = Grains.improve_grain_segmentation_unet(
                        filename=self.filename,
                        direction=direction,
                        unet_config=self.unet_config,
                        graincrops=graincrops,
                    )
                    # Construct full masks from the crops
                    full_mask_tensor = Grains.construct_full_mask_from_graincrops(
                        graincrops=graincrops,
                        image_shape=self.image.shape,
                    )

                # Set the unet tensor regardless of if the unet model was run, since the plotting expects it
                # can be changed when we do a plotting overhaul
                self.directions[direction]["unet_tensor"] = full_mask_tensor

                # Vet the grains
                if self.vetting is not None:
                    graincrops_vetted = Grains.vet_grains(
                        graincrops=graincrops,
                        **self.vetting,
                    )
                else:
                    graincrops_vetted = graincrops
                graincrops_vetted = Grains.graincrops_update_background_class(graincrops=graincrops_vetted)

                full_mask_tensor_vetted = Grains.construct_full_mask_from_graincrops(
                    graincrops=graincrops_vetted,
                    image_shape=self.image.shape,
                )
                self.directions[direction]["vetted_tensor"] = full_mask_tensor_vetted

                # Mandatory check to remove any objects in any classes that are too small to process
                graincrops_removed_too_small_to_process = Grains.graincrops_remove_objects_too_small_to_process(
                    graincrops=graincrops_vetted,
                    min_object_size=self.minimum_grain_size_px,
                    min_object_bbox_size=self.minimum_bbox_size_px,
                )
                graincrops_removed_too_small_to_process = Grains.graincrops_update_background_class(
                    graincrops=graincrops_removed_too_small_to_process
                )

                # Merge classes as specified by the user
                graincrops_merged_classes = Grains.graincrops_merge_classes(
                    graincrops=graincrops_removed_too_small_to_process,
                    classes_to_merge=self.classes_to_merge,
                )
                graincrops_merged_classes = Grains.graincrops_update_background_class(
                    graincrops=graincrops_merged_classes
                )
                graincrops_merged_classes = Grains.graincrops_update_background_class(
                    graincrops=graincrops_merged_classes
                )

                full_mask_tensor_merged_classes = Grains.construct_full_mask_from_graincrops(
                    graincrops=graincrops_merged_classes,
                    image_shape=self.image.shape,
                )
                self.directions[direction]["merged_classes_tensor"] = full_mask_tensor_merged_classes

                # Store the grain crops
                if direction == "above":
                    image_grain_crops.above = GrainCropsDirection(
                        crops=graincrops_merged_classes,
                        full_mask_tensor=full_mask_tensor_merged_classes,
                    )
                elif direction == "below":
                    image_grain_crops.below = GrainCropsDirection(
                        crops=graincrops_merged_classes,
                        full_mask_tensor=full_mask_tensor_merged_classes,
                    )
                else:
                    raise ValueError(f"Invalid direction: {direction}. Allowed values are 'above' and 'below'")
                self.image_grain_crops = image_grain_crops
            else:
                # No grains found
                self.image_grain_crops = ImageGrainCrops(above=None, below=None)

    # pylint: disable=too-many-locals
    @staticmethod
    def improve_grain_segmentation_unet(
        graincrops: dict[int, GrainCrop],
        filename: str,
        direction: str,
        unet_config: dict[str, str | int | float | tuple[int | None, int, int, int] | None],
    ) -> dict[int, GrainCrop]:
        """
        Use a UNet model to re-segment existing grains to improve their accuracy.

        Parameters
        ----------
        graincrops : dict[int, GrainCrop]
            Dictionary of grain crops.
        filename : str
            File being processed (used in logging).
        direction : str
            Direction of threshold for which bounding boxes are being calculated.
        unet_config : dict[str, str | int | float | tuple[int | None, int, int, int] | None]
            Configuration for the UNet model.
            model_path: str
                Path to the UNet model.
            grain_crop_padding: int
                Padding to add to the bounding box of the grain before cropping.
            upper_norm_bound: float
                Upper bound for normalising the image.
            lower_norm_bound: float
                Lower bound for normalising the image.

        Returns
        -------
        dict[int, GrainCrop]
            Dictionary of (hopefully) improved grain crops.
        """
        LOGGER.debug(f"[{filename}] : Running UNet model on {direction} grains")

        # When debugging, you might find that the custom_objects are incorrect. This is entirely based on what the model used
        # for its loss during training and so this will need to be changed a lot.
        # Once the group has gotten used to training models, this can be made configurable, but currently it's too changeable.
        # unet_model = keras.models.load_model(
        #     self.unet_config["model_path"], custom_objects={"dice_loss": dice_loss, "iou_loss": iou_loss}
        # )
        # You may also get an error referencing a "group_1" parameter, this is discussed in this issue:
        # https://github.com/keras-team/keras/issues/19441 which also has an experimental fix that we can try but
        # I haven't tested it yet.

        try:
            unet_model = keras.models.load_model(
                unet_config["model_path"], custom_objects={"mean_iou": mean_iou, "iou_loss": iou_loss}, compile=False
            )
        except Exception as e:
            LOGGER.debug(f"Python executable: {sys.executable}")
            LOGGER.debug(f"Keras version: {keras.__version__}")
            LOGGER.debug(f"Model path: {unet_config['model_path']}")
            raise e

        # unet_model = keras.models.load_model(unet_config["model_path"], custom_objects={"mean_iou": mean_iou})
        LOGGER.debug(f"Output shape of UNet model: {unet_model.output_shape}")

        new_graincrops: dict[int, GrainCrop] = {}
        num_empty_removed_grains = 0
        for grain_number, graincrop in graincrops.items():
            LOGGER.debug(f"Unet predicting mask for grain {grain_number} of {len(graincrops)}")
            # Run the UNet on the region. This is allowed to be a single channel
            # as we can add a background channel afterwards if needed.
            # Remember that this region is cropped from the original image, so it's not
            # the same size as the original image.
            predicted_mask = predict_unet(
                image=graincrop.image,
                model=unet_model,
                confidence=0.1,
                model_input_shape=unet_model.input_shape,
                upper_norm_bound=unet_config["upper_norm_bound"],
                lower_norm_bound=unet_config["lower_norm_bound"],
            )
            assert len(predicted_mask.shape) == 3
            LOGGER.debug(f"Predicted mask shape: {predicted_mask.shape}")

            # Check if all of the non-background classes are empty
            if np.sum(predicted_mask[:, :, 1:]) == 0:
                num_empty_removed_grains += 1
            else:
                new_graincrops[grain_number] = GrainCrop(
                    image=graincrop.image,
                    mask=predicted_mask,
                    padding=graincrop.padding,
                    bbox=graincrop.bbox,
                    pixel_to_nm_scaling=graincrop.pixel_to_nm_scaling,
                    filename=graincrop.filename,
                )

        LOGGER.debug(f"Number of empty removed grains: {num_empty_removed_grains}")

        return new_graincrops

    @staticmethod
    def keep_largest_labelled_region(
        labelled_image: npt.NDArray[np.int32],
    ) -> npt.NDArray[np.bool_]:
        """
        Keep only the largest region in a labelled image.

        Parameters
        ----------
        labelled_image : npt.NDArray
            2-D Numpy array of labelled regions.

        Returns
        -------
        npt.NDArray
            2-D Numpy boolean array of labelled regions with only the largest region.
        """
        # Check if there are any labelled regions
        if labelled_image.max() == 0:
            return np.zeros_like(labelled_image).astype(np.bool_)
        # Get the sizes of the regions
        sizes = np.array([(labelled_image == label).sum() for label in range(1, labelled_image.max() + 1)])
        # Keep only the largest region
        return np.where(labelled_image == sizes.argmax() + 1, labelled_image, 0).astype(bool)

    @staticmethod
    def flatten_multi_class_tensor(grain_mask_tensor: npt.NDArray) -> npt.NDArray:
        """
        Flatten a multi-class image tensor to a single binary mask.

        The returned tensor is of boolean type in case there are multiple hits in the same pixel. We dont want to have
        2s, 3s etc because this would cause issues in labelling and cause erroneous grains within grains.

        Parameters
        ----------
        grain_mask_tensor : npt.NDArray
            Multi class grain mask tensor tensor of shape (N, N, C).

        Returns
        -------
        npt.NDArray
            Combined binary mask of all but the background class (:, :, 0).
        """
        assert len(grain_mask_tensor.shape) == 3, f"Tensor not 3D: {grain_mask_tensor.shape}"
        return np.sum(grain_mask_tensor[:, :, 1:], axis=-1).astype(bool)

    @staticmethod
    def get_multi_class_grain_bounding_boxes(grain_mask_tensor: npt.NDArray) -> dict:
        """
        Get the bounding boxes for each grain in a multi-class image tensor.

        Finds the bounding boxes for each grain in a multi-class image tensor. Grains can span multiple classes, so the
        bounding boxes are found for the combined binary mask of contiguous grains across all classes.

        Parameters
        ----------
        grain_mask_tensor : npt.NDArray
            3-D Numpy array of grain mask tensor.

        Returns
        -------
        dict
            Dictionary of bounding boxes indexed by grain number.
        """
        flattened_mask = Grains.flatten_multi_class_tensor(grain_mask_tensor)
        labelled_regions = Grains.label_regions(flattened_mask)
        region_properties = Grains.get_region_properties(labelled_regions)
        bounding_boxes = {index: region.bbox for index, region in enumerate(region_properties)}
        return {
            index: pad_bounding_box(
                crop_min_row=bbox[0],
                crop_min_col=bbox[1],
                crop_max_row=bbox[2],
                crop_max_col=bbox[3],
                image_shape=(grain_mask_tensor.shape[0], grain_mask_tensor.shape[1]),
                padding=1,
            )
            for index, bbox in bounding_boxes.items()
        }

    @staticmethod
    def update_background_class(
        grain_mask_tensor: npt.NDArray,
    ) -> npt.NDArray:
        """
        Update the background class to reflect the other classes.

        Parameters
        ----------
        grain_mask_tensor : npt.NDArray
            3-D Numpy array of the grain mask tensor.

        Returns
        -------
        npt.NDArray
            3-D Numpy array of image tensor with updated background class.
        """
        flattened_mask = Grains.flatten_multi_class_tensor(grain_mask_tensor)
        new_background = np.where(flattened_mask == 0, 1, 0)
        grain_mask_tensor[:, :, 0] = new_background
        return grain_mask_tensor.astype(bool)

    @staticmethod
    def vet_class_sizes_single_grain(
        single_grain_mask_tensor: npt.NDArray,
        pixel_to_nm_scaling: float,
        class_size_thresholds: list[tuple[int, int, int]] | None,
    ) -> tuple[npt.NDArray, bool]:
        """
        Remove regions of particular classes based on size thresholds.

        Regions of classes that are too large or small may need to be removed for many reasons (eg removing noise
        erroneously detected by the model or larger-than-expected molecules that are obviously erroneous), this method
        allows for the removal of these regions based on size thresholds.

        Parameters
        ----------
        single_grain_mask_tensor : npt.NDArray
            3-D Numpy array of the mask tensor.
        pixel_to_nm_scaling : float
            Scaling of pixels to nanometres.
        class_size_thresholds : list[list[int, int, int]] | None
            List of class size thresholds. Structure is [(class_index, lower, upper)].

        Returns
        -------
        npt.NDArray
            3-D Numpy array of the mask tensor with grains removed based on size thresholds.
        bool
            True if the grain passes the vetting, False if it fails.
        """
        if class_size_thresholds is None:
            return single_grain_mask_tensor, True

        # Iterate over the classes and check the sizes
        for class_index in range(1, single_grain_mask_tensor.shape[2]):
            class_size = np.sum(single_grain_mask_tensor[:, :, class_index]) * pixel_to_nm_scaling**2
            # Check the size against the thresholds

            classes_to_vet = [vetting_criteria[0] for vetting_criteria in class_size_thresholds]

            if class_index not in classes_to_vet:
                continue

            lower_threshold, upper_threshold = [
                vetting_criteria[1:] for vetting_criteria in class_size_thresholds if vetting_criteria[0] == class_index
            ][0]

            if lower_threshold is not None:
                if class_size < lower_threshold:
                    # Return empty tensor
                    empty_crop_tensor = np.zeros_like(single_grain_mask_tensor)
                    # Fill the background class with 1s
                    empty_crop_tensor[:, :, 0] = 1
                    return empty_crop_tensor, False
            if upper_threshold is not None:
                if class_size > upper_threshold:
                    # Return empty tensor
                    empty_crop_tensor = np.zeros_like(single_grain_mask_tensor)
                    # Fill the background class with 1s
                    empty_crop_tensor[:, :, 0] = 1
                    return empty_crop_tensor, False

        return single_grain_mask_tensor, True

    @staticmethod
    def get_individual_grain_crops(
        grain_mask_tensor: npt.NDArray,
        padding: int = 1,
    ) -> tuple[list[npt.NDArray], list[npt.NDArray], int]:
        """
        Get individual grain crops from an image tensor.

        Fetches individual grain crops from an image tensor, but zeros any non-connected grains
        in the crop region. This is to ensure that other grains do not affect further processing
        steps.

        Parameters
        ----------
        grain_mask_tensor : npt.NDArray
            3-D Numpy array of image tensor.
        padding : int
            Padding to add to the bounding box of the grain before cropping. Default is 1.

        Returns
        -------
        list[npt.NDArray]
            List of individual grain crops.
        list[npt.NDArray]
            List of bounding boxes for each grain.
        int
            Padding used for the bounding boxes.
        """
        grain_crops = []
        bounding_boxes = []

        # Label the regions
        flattened_multi_class_mask = Grains.flatten_multi_class_tensor(grain_mask_tensor)
        labelled_regions = Grains.label_regions(flattened_multi_class_mask)

        # Iterate over the regions and return the crop, but zero any non-connected grains
        for region in Grains.get_region_properties(labelled_regions):

            binary_labelled_regions = labelled_regions == region.label

            # Zero any non-connected grains
            # For each class, set all pixels to zero that are not in the current region
            this_region_only_grain_tensor = np.copy(grain_mask_tensor)
            # Iterate over the non-background classes
            for class_index in range(1, grain_mask_tensor.shape[2]):
                # Set all pixels to zero that are not in the current region
                this_region_only_grain_tensor[:, :, class_index] = (
                    binary_labelled_regions * grain_mask_tensor[:, :, class_index]
                )

            # Update background class to reflect the removal of any non-connected grains
            this_region_only_grain_tensor = Grains.update_background_class(
                grain_mask_tensor=this_region_only_grain_tensor
            )

            # Get the bounding box
            bounding_box = region.bbox

            # Pad the bounding box
            bounding_box = pad_bounding_box(
                crop_min_row=bounding_box[0],
                crop_min_col=bounding_box[1],
                crop_max_row=bounding_box[2],
                crop_max_col=bounding_box[3],
                image_shape=(grain_mask_tensor.shape[0], grain_mask_tensor.shape[1]),
                padding=padding,
            )

            # Crop the grain
            grain_crop = this_region_only_grain_tensor[
                bounding_box[0] : bounding_box[2],
                bounding_box[1] : bounding_box[3],
                :,
            ]

            # Add the crop to the list
            grain_crops.append(grain_crop.astype(bool))
            bounding_boxes.append(bounding_box)

        return grain_crops, bounding_boxes, padding

    @staticmethod
    def vet_numbers_of_regions_single_grain(
        grain_mask_tensor: npt.NDArray,
        class_region_number_thresholds: list[tuple[int, int, int]] | None,
    ) -> tuple[npt.NDArray, bool]:
        """
        Check if the number of regions of different classes for a single grain is within thresholds.

        Parameters
        ----------
        grain_mask_tensor : npt.NDArray
            3-D Numpy array of the grain mask tensor, should be of only one grain.
        class_region_number_thresholds : list[list[int, int, int]]
            List of class region number thresholds. Structure is [(class_index, lower, upper)].

        Returns
        -------
        npt.NDArray
            3-D Numpy array of the grain mask tensor with grains removed based on region number thresholds.
        bool
            True if the grain passes the vetting, False if it fails.
        """
        if class_region_number_thresholds is None:
            return grain_mask_tensor, True

        # Iterate over the classes and check the number of regions
        for class_index in range(1, grain_mask_tensor.shape[2]):
            # Get the number of regions
            class_labelled_regions = Grains.label_regions(grain_mask_tensor[:, :, class_index])
            number_of_regions = np.unique(class_labelled_regions).shape[0] - 1
            # Check the number of regions against the thresholds, skip if no thresholds provided
            # Get the classes we are trying to vet (the first element of each tuple)
            classes_to_vet = [vetting_criteria[0] for vetting_criteria in class_region_number_thresholds]

            if class_index not in classes_to_vet:
                continue

            lower_threshold, upper_threshold = [
                vetting_criteria[1:]
                for vetting_criteria in class_region_number_thresholds
                if vetting_criteria[0] == class_index
            ][0]

            # Check the number of regions against the thresholds
            if lower_threshold is not None:
                if number_of_regions < lower_threshold:
                    # Return empty tensor
                    empty_crop_tensor = np.zeros_like(grain_mask_tensor)
                    # Fill the background class with 1s
                    empty_crop_tensor[:, :, 0] = 1
                    return empty_crop_tensor, False
            if upper_threshold is not None:
                if number_of_regions > upper_threshold:
                    # Return empty tensor
                    empty_crop_tensor = np.zeros_like(grain_mask_tensor)
                    # Fill the background class with 1s
                    empty_crop_tensor[:, :, 0] = 1
                    return empty_crop_tensor, False

        return grain_mask_tensor, True

    @staticmethod
    def convert_classes_to_nearby_classes(
        grain_mask_tensor: npt.NDArray,
        classes_to_convert: list[tuple[int, int]] | None,
        class_touching_threshold: int = 1,
    ) -> npt.NDArray:
        """
        Convert all but the largest regions of one class into another class provided the former touches the latter.

        Specifically, it takes a list of tuples of two integers (dubbed class A and class B). For each class A, class B
        pair, it will find the largest region of class A and flag it to be ignored. Then for each non-largest region of
        class A, it will check if it touches any class B region (within the ``class_touching_threshold`` distance). If it
        does, it will convert the region to class B.

        This is useful for situations where you want just one region of class A and the model has a habit of producing
        small regions of class A interspersed in the class B regions, which should be class B instead.

        Parameters
        ----------
        grain_mask_tensor : npt.NDArray
            3-D Numpy array of the grain mask tensor.
        classes_to_convert : list
            List of tuples of classes to convert. Structure is [(class_a, class_b)].
        class_touching_threshold : int
            Number of dilation passes to do to determine class A connectivity with class B.

        Returns
        -------
        npt.NDArray
            3-D Numpy array of the grain mask tensor with classes converted.
        """
        # If no classes to convert, return the original tensor
        if not classes_to_convert:
            return grain_mask_tensor

        # Iterate over class pairs
        for class_a, class_b in classes_to_convert:
            # Get the binary mask for class A and class B
            class_a_mask = grain_mask_tensor[:, :, class_a]
            class_b_mask = grain_mask_tensor[:, :, class_b]

            # Skip if no regions of class A
            if np.max(class_a_mask) == 0:
                continue

            # Find the largest region of class A
            class_a_labelled_regions = Grains.label_regions(class_a_mask)
            class_a_region_properties = Grains.get_region_properties(class_a_labelled_regions)
            class_a_areas = [region.area for region in class_a_region_properties]
            largest_class_a_region = class_a_region_properties[np.argmax(class_a_areas)]

            # For all other regions, check if they touch the class B region
            for region in class_a_region_properties:
                if region.label == largest_class_a_region.label:
                    continue
                # Get only the pixels in the region
                region_mask = class_a_labelled_regions == region.label
                # Dilate the region
                dilated_region_mask = region_mask
                for _ in range(class_touching_threshold):
                    dilated_region_mask = binary_dilation(dilated_region_mask)
                # Get the intersection with the class B mask
                intersection = dilated_region_mask & class_b_mask
                # If there is any intersection, turn the region into class B
                if np.any(intersection):
                    # Add to the class B mask
                    class_b_mask = np.where(region_mask, class_b, class_b_mask)
                    # Remove from the class A mask
                    class_a_mask = np.where(region_mask, 0, class_a_mask)

            # Update the tensor
            grain_mask_tensor[:, :, class_a] = class_a_mask
            grain_mask_tensor[:, :, class_b] = class_b_mask

        return grain_mask_tensor.astype(bool)

    @staticmethod
    def keep_largest_labelled_region_classes(
        single_grain_mask_tensor: npt.NDArray,
        keep_largest_labelled_regions_classes: list[int] | None,
    ) -> npt.NDArray:
        """
        Keep only the largest region in specific classes.

        Parameters
        ----------
        single_grain_mask_tensor : npt.NDArray
            3-D Numpy array of the grain mask tensor.
        keep_largest_labelled_regions_classes : list[int]
            List of classes to keep only the largest region.

        Returns
        -------
        npt.NDArray
            3-D Numpy array of the grain mask tensor with only the largest regions in specific classes.
        """
        if keep_largest_labelled_regions_classes is None:
            return single_grain_mask_tensor

        # Iterate over the classes
        for class_index in keep_largest_labelled_regions_classes:
            # Get the binary mask for the class
            class_mask = single_grain_mask_tensor[:, :, class_index]

            # Skip if no regions
            if np.max(class_mask) == 0:
                continue

            # Label the regions
            labelled_regions = Grains.label_regions(class_mask)
            # Get the region properties
            region_properties = Grains.get_region_properties(labelled_regions)
            # Get the region areas
            region_areas = [region.area for region in region_properties]
            # Keep only the largest region
            largest_region = region_properties[np.argmax(region_areas)]
            class_mask_largest_only = np.where(labelled_regions == largest_region.label, labelled_regions, 0)
            # Update the tensor
            single_grain_mask_tensor[:, :, class_index] = class_mask_largest_only.astype(bool)

        # Update the background class
        return Grains.update_background_class(single_grain_mask_tensor)

    @staticmethod
    def calculate_region_connection_regions(
        grain_mask_tensor: npt.NDArray,
        classes: tuple[int, int],
    ) -> tuple[int, npt.NDArray, dict[int, npt.NDArray[int]]]:
        """
        Get a list of connection regions between two classes.

        Parameters
        ----------
        grain_mask_tensor : npt.NDArray
            3-D Numpy array of the grain mask tensor.
        classes : tuple[int, int]
            Tuple pair of classes to calculate the connection regions.

        Returns
        -------
        int
            Number of connection regions.
        npt.NDArray
            2-D Numpy array of the intersection labels.
        dict
            Dictionary of connection points indexed by region label.
        """
        # Get the binary masks for the classes
        class_a_mask = grain_mask_tensor[:, :, classes[0]]
        class_b_mask = grain_mask_tensor[:, :, classes[1]]

        # Dilate class A mask
        dilated_class_a_mask = binary_dilation(class_a_mask)
        # Get the intersection with the class B mask
        intersection = dilated_class_a_mask & class_b_mask

        # Get number of separate intersection regions
        intersection_labels = label(intersection)
        intersection_regions = regionprops(intersection_labels)
        num_connection_regions = len(intersection_regions)
        # Create a dictionary of the connection points
        intersection_points = {region.label: region.coords for region in intersection_regions}

        return num_connection_regions, intersection_labels, intersection_points

    @staticmethod
    def vet_class_connection_points(
        grain_mask_tensor: npt.NDArray,
        class_connection_point_thresholds: list[tuple[tuple[int, int], tuple[int, int]]] | None,
    ) -> bool:
        """
        Vet the number of connection points between regions in specific classes.

        Parameters
        ----------
        grain_mask_tensor : npt.NDArray
            3-D Numpy array of the grain mask tensor.
        class_connection_point_thresholds : list[tuple[tuple[int, int], tuple[int, int]]] | None
            List of tuples of classes and connection point thresholds. Structure is [(class_pair, (lower, upper))].

        Returns
        -------
        bool
            True if the grain passes the vetting, False if it fails.
        """
        if class_connection_point_thresholds is None:
            return True

        # Iterate over the class pairs
        for class_pair, connection_point_thresholds in class_connection_point_thresholds:
            # Get the connection regions
            num_connection_regions, _, _ = Grains.calculate_region_connection_regions(
                grain_mask_tensor=grain_mask_tensor,
                classes=class_pair,
            )
            # Check the number of connection regions against the thresholds
            lower_threshold, upper_threshold = connection_point_thresholds
            if lower_threshold is not None:
                if num_connection_regions < lower_threshold:
                    return False
            if upper_threshold is not None:
                if num_connection_regions > upper_threshold:
                    return False

        return True

    @staticmethod
    def assemble_grain_mask_tensor_from_crops(
        grain_mask_tensor_shape: tuple[int, int, int],
        grain_crops_and_bounding_boxes: list[dict[str, npt.NDArray]],
    ) -> npt.NDArray:
        """
        Combine individual grain crops into a single grain mask tensor.

        Parameters
        ----------
        grain_mask_tensor_shape : tuple
            Shape of the grain mask tensor.
        grain_crops_and_bounding_boxes : list
            List of dictionaries containing the grain crops and bounding boxes.
            Structure: [{"grain_tensor": npt.NDArray, "bounding_box": tuple, "padding": int}].

        Returns
        -------
        npt.NDArray
            3-D Numpy array of the grain mask tensor.
        """
        # Initialise the grain mask tensor
        grain_mask_tensor = np.zeros(grain_mask_tensor_shape).astype(np.int32)

        # Iterate over the grain crops
        for grain_crop_and_bounding_box in grain_crops_and_bounding_boxes:
            # Get the grain crop and bounding box
            grain_crop = grain_crop_and_bounding_box["grain_tensor"]
            bounding_box = grain_crop_and_bounding_box["bounding_box"]
            padding = grain_crop_and_bounding_box["padding"]

            # Get the bounding box coordinates
            min_row, min_col, max_row, max_col = bounding_box

            # Crop the grain
            cropped_grain = grain_crop[
                padding:-padding,
                padding:-padding,
                :,
            ]

            # Update the grain mask tensor
            grain_mask_tensor[
                min_row + padding : max_row - padding,
                min_col + padding : max_col - padding,
                :,
            ] = cropped_grain

        # Update the background class
        grain_mask_tensor = Grains.update_background_class(grain_mask_tensor)

        return grain_mask_tensor.astype(bool)

    # Ignore too complex, to break the function down into smaller functions would make it more complex.
    # ruff: noqa: C901
    @staticmethod
    def convert_classes_when_too_big_or_small(
        grain_mask_tensor: npt.NDArray,
        pixel_to_nm_scaling: float,
        class_conversion_size_thresholds: list[tuple[tuple[int, int, int], tuple[int, int]]] | None,
    ) -> npt.NDArray:
        """
        Convert classes when they are too big or too small based on size thresholds.

        Parameters
        ----------
        grain_mask_tensor : npt.NDArray
            3-D Numpy array of the grain mask tensor.
        pixel_to_nm_scaling : float
            Scaling of pixels to nanometres.
        class_conversion_size_thresholds : list
            List of class conversion size thresholds.
            Structure is [(class_index, class_to_convert_to_if_to_small, class_to_convert_to_if_too_big),
            (lower_threshold, upper_threshold)].

        Returns
        -------
        npt.NDArray
            3-D Numpy array of the grain mask tensor with classes converted based on size thresholds.
        """
        if class_conversion_size_thresholds is None:
            return grain_mask_tensor

        new_grain_mask_tensor = np.copy(grain_mask_tensor)
        classes_to_vet = [vetting_criteria[0][0] for vetting_criteria in class_conversion_size_thresholds]
        for class_index in range(1, grain_mask_tensor.shape[2]):
            if class_index not in classes_to_vet:
                continue

            lower_threshold, upper_threshold = [
                vetting_criteria[1]
                for vetting_criteria in class_conversion_size_thresholds
                if vetting_criteria[0][0] == class_index
            ][0]

            class_to_convert_to_if_too_small, class_to_convert_to_if_too_big = [
                vetting_criteria[0][1:]
                for vetting_criteria in class_conversion_size_thresholds
                if vetting_criteria[0][0] == class_index
            ][0]

            # For each region in the class, check its size and convert if needed
            labelled_regions = Grains.label_regions(grain_mask_tensor[:, :, class_index])
            region_properties = Grains.get_region_properties(labelled_regions)
            for region in region_properties:
                region_mask = labelled_regions == region.label
                region_size = np.sum(region_mask) * pixel_to_nm_scaling**2
                if lower_threshold is not None:
                    if region_size < lower_threshold:
                        if class_to_convert_to_if_too_small is not None:
                            # Add the region to the class to convert to in the new tensor
                            new_grain_mask_tensor[:, :, class_to_convert_to_if_too_small] = np.where(
                                region_mask,
                                class_to_convert_to_if_too_small,
                                new_grain_mask_tensor[:, :, class_to_convert_to_if_too_small],
                            )
                        # Remove the region from the original class
                        new_grain_mask_tensor[:, :, class_index] = np.where(
                            region_mask,
                            0,
                            new_grain_mask_tensor[:, :, class_index],
                        )
                if upper_threshold is not None:
                    if region_size > upper_threshold:
                        if class_to_convert_to_if_too_big is not None:
                            # Add the region to the class to convert to in the new tensor
                            new_grain_mask_tensor[:, :, class_to_convert_to_if_too_big] = np.where(
                                region_mask,
                                class_to_convert_to_if_too_big,
                                new_grain_mask_tensor[:, :, class_to_convert_to_if_too_big],
                            )
                        # Remove the region from the original class
                        new_grain_mask_tensor[:, :, class_index] = np.where(
                            region_mask,
                            0,
                            new_grain_mask_tensor[:, :, class_index],
                        )

        # Update the background class
        new_grain_mask_tensor = Grains.update_background_class(new_grain_mask_tensor)

        return new_grain_mask_tensor.astype(bool)

    @staticmethod
    def vet_grains(
        graincrops: dict[int, GrainCrop],
        class_conversion_size_thresholds: list[tuple[tuple[int, int, int], tuple[int, int]]] | None,
        class_size_thresholds: list[tuple[int, int, int]] | None,
        class_region_number_thresholds: list[tuple[int, int, int]] | None,
        nearby_conversion_classes_to_convert: list[tuple[int, int]] | None,
        class_touching_threshold: int,
        keep_largest_labelled_regions_classes: list[int] | None,
        class_connection_point_thresholds: list[tuple[tuple[int, int], tuple[int, int]]] | None,
    ) -> dict[int, GrainCrop]:
        """
        Vet grains in a grain mask tensor based on a variety of criteria.

        Parameters
        ----------
        graincrops : dict[int, GrainCrop]
            Dictionary of grain crops.
        class_conversion_size_thresholds : list
            List of class conversion size thresholds. Structure is [(class_index, class_to_convert_to_if_too_small,
            class_to_convert_to_if_too_big), (lower_threshold, upper_threshold)].
        class_size_thresholds : list
            List of class size thresholds. Structure is [(class_index, lower, upper)].
        class_region_number_thresholds : list
            List of class region number thresholds. Structure is [(class_index, lower, upper)].
        nearby_conversion_classes_to_convert : list
            List of tuples of classes to convert. Structure is [(class_a, class_b)].
        class_touching_threshold : int
            Number of dilation passes to do to determine class A connectivity with class B.
        keep_largest_labelled_regions_classes : list
            List of classes to keep only the largest region.
        class_connection_point_thresholds : list
            List of tuples of classes and connection point thresholds. Structure is [(class_pair, (lower, upper))].

        Returns
        -------
        dict[int, GrainCrop]
            Dictionary of grain crops that passed the vetting.
        """
        passed_graincrops: dict[int, GrainCrop] = {}

        # Iterate over the grain crops
        for grain_number, graincrop in graincrops.items():

            single_grain_mask_tensor = graincrop.mask
            pixel_to_nm_scaling = graincrop.pixel_to_nm_scaling

            # Convert small / big areas to other classes
            single_grain_mask_tensor = Grains.convert_classes_when_too_big_or_small(
                grain_mask_tensor=single_grain_mask_tensor,
                pixel_to_nm_scaling=pixel_to_nm_scaling,
                class_conversion_size_thresholds=class_conversion_size_thresholds,
            )

            # Vet number of regions (foreground and background)
            _, passed = Grains.vet_numbers_of_regions_single_grain(
                grain_mask_tensor=single_grain_mask_tensor,
                class_region_number_thresholds=class_region_number_thresholds,
            )
            if not passed:
                continue

            # Vet size of regions (foreground and background)
            _, passed = Grains.vet_class_sizes_single_grain(
                single_grain_mask_tensor=single_grain_mask_tensor,
                pixel_to_nm_scaling=pixel_to_nm_scaling,
                class_size_thresholds=class_size_thresholds,
            )
            if not passed:
                continue

            # Turn all but largest region of class A into class B provided that the class A region touched a class B
            # region
            converted_single_grain_mask_tensor = Grains.convert_classes_to_nearby_classes(
                grain_mask_tensor=single_grain_mask_tensor,
                classes_to_convert=nearby_conversion_classes_to_convert,
                class_touching_threshold=class_touching_threshold,
            )

            # Remove all but largest region in specific classes
            largest_only_single_grain_mask_tensor = Grains.keep_largest_labelled_region_classes(
                single_grain_mask_tensor=converted_single_grain_mask_tensor,
                keep_largest_labelled_regions_classes=keep_largest_labelled_regions_classes,
            )

            # Vet number of connection points between regions in specific classes
            if not Grains.vet_class_connection_points(
                grain_mask_tensor=largest_only_single_grain_mask_tensor,
                class_connection_point_thresholds=class_connection_point_thresholds,
            ):
                continue

            # If passed all vetting steps, add to the dictionary of passed grain crops
            passed_graincrops[grain_number] = GrainCrop(
                image=graincrop.image,
                mask=largest_only_single_grain_mask_tensor,
                padding=graincrop.padding,
                bbox=graincrop.bbox,
                pixel_to_nm_scaling=graincrop.pixel_to_nm_scaling,
                filename=graincrop.filename,
            )

        return passed_graincrops

    @staticmethod
    def merge_classes(
        grain_mask_tensor: npt.NDArray,
        classes_to_merge: list[tuple[int]] | None,
    ) -> npt.NDArray:
        """
        Merge classes in a grain mask tensor and add them to the grain tensor.

        Parameters
        ----------
        grain_mask_tensor : npt.NDArray
            3-D Numpy array of the grain mask tensor.
        classes_to_merge : list | None
            List of tuples for classes to merge, can be any number of classes.

        Returns
        -------
        npt.NDArray
            3-D Numpy array of the grain mask tensor with classes merged.
        """
        if classes_to_merge is None:
            return grain_mask_tensor
        # For each set of classes to merge:
        for classes in classes_to_merge:
            # Get the binary masks for all the classes
            class_masks = [grain_mask_tensor[:, :, class_index] for class_index in classes]
            # Combine the masks
            combined_mask = np.logical_or.reduce(class_masks)

            # Add new class to the grain tensor with the combined mask
            grain_mask_tensor = np.dstack([grain_mask_tensor, combined_mask])

        return grain_mask_tensor.astype(bool)

    @staticmethod
    def construct_full_mask_from_graincrops(
        graincrops: dict[int, GrainCrop], image_shape: tuple[int, int]
    ) -> npt.NDArray[np.bool_]:
        """
        Construct a full mask tensor from the grain crops.

        Parameters
        ----------
        graincrops : dict[int, GrainCrop]
            Dictionary of grain crops.
        image_shape : tuple[int, int, int]
            Shape of the original image.

        Returns
        -------
        npt.NDArray[np.bool_]
            NxNxC Numpy array of the full mask tensor.
        """
        # Calculate the number of classes from the first grain crop
        # Check if graincrops is empty
        if not graincrops:
            raise ValueError("No grain crops provided to construct the full mask tensor.")
        num_classes: int = list(graincrops.values())[0].mask.shape[2]
        full_mask_tensor = np.zeros((image_shape[0], image_shape[1], num_classes), dtype=np.bool_)
        for _grain_number, graincrop in graincrops.items():
            bounding_box = graincrop.bbox
            crop_tensor = graincrop.mask

            # Add the crop to the full mask tensor without overriding anything else, for all classes
            for class_index in range(crop_tensor.shape[2]):
                full_mask_tensor[
                    bounding_box[0] : bounding_box[2],
                    bounding_box[1] : bounding_box[3],
                    class_index,
                ] += crop_tensor[:, :, class_index]

        # Update background class and return
        return Grains.update_background_class(full_mask_tensor)

    @staticmethod
    def extract_grains_from_full_image_tensor(
        image: npt.NDArray[np.float32],
        full_mask_tensor: npt.NDArray[np.bool_],
        padding: int,
        pixel_to_nm_scaling: float,
        filename: str,
    ) -> dict[int, GrainCrop]:
        """
        Extract grains from the full image mask tensor.

        Grains are detected using connected components across all classes in the full mask tensor.

        Parameters
        ----------
        image : npt.NDArray[np.float32]
            2-D Numpy array of the image.
        full_mask_tensor : npt.NDArray[np.bool_]
            3-D NxNxC boolean numpy array of all the class masks for the image.
        padding : int
            Padding added to the bounding box of the grain before cropping.
        pixel_to_nm_scaling : float
            Pixel to nanometre scaling factor.
        filename : str
            Filename of the image.

        Returns
        -------
        dict[int, GrainCrop]
            Dictionary of grain crops.
        """
        # Flatten the mask tensor
        flat_mask = Grains.flatten_multi_class_tensor(full_mask_tensor)
        labelled_flat_full_mask = label(flat_mask)
        flat_regionprops_full_mask = regionprops(labelled_flat_full_mask)
        graincrops = {}
        for grain_number, flat_region in enumerate(flat_regionprops_full_mask):
            # Get a flattened binary mask for the whole grain and no other grains
            flattened_grain_binary_mask = labelled_flat_full_mask == flat_region.label

            # For each class, set all pixels to zero that are not in the current region
            grain_tensor_full_mask = np.zeros_like(full_mask_tensor).astype(bool)
            for class_index in range(1, full_mask_tensor.shape[2]):
                # Set all pixels to zero that are not in the current region's pixels by multiplying by a binary mask
                # for the whole flattened grain mask
                grain_tensor_full_mask[:, :, class_index] = (
                    flattened_grain_binary_mask * full_mask_tensor[:, :, class_index]
                ).astype(bool)

            # Crop the tensor
            # Get the bounding box for the region
            flat_bounding_box: tuple[int, int, int, int] = tuple(flat_region.bbox)  # min_row, min_col, max_row, max_col

            # Pad the mask
            padded_flat_bounding_box = pad_bounding_box(
                crop_min_row=flat_bounding_box[0],
                crop_min_col=flat_bounding_box[1],
                crop_max_row=flat_bounding_box[2],
                crop_max_col=flat_bounding_box[3],
                image_shape=(full_mask_tensor.shape[0], full_mask_tensor.shape[1]),
                padding=padding,
            )

            # Make the mask square
            square_flat_bounding_box = make_bounding_box_square(
                crop_min_row=padded_flat_bounding_box[0],
                crop_min_col=padded_flat_bounding_box[1],
                crop_max_row=padded_flat_bounding_box[2],
                crop_max_col=padded_flat_bounding_box[3],
                image_shape=(full_mask_tensor.shape[0], full_mask_tensor.shape[1]),
            )

            assert (
                square_flat_bounding_box[0] - square_flat_bounding_box[2]
                == square_flat_bounding_box[1] - square_flat_bounding_box[3]
            )

            print(
                f"square flat bounding box {square_flat_bounding_box} shape {square_flat_bounding_box[2] - square_flat_bounding_box[0], square_flat_bounding_box[3] - square_flat_bounding_box[1]}"
            )

            # Grab image and mask for the cropped region
            grain_cropped_image = image[
                square_flat_bounding_box[0] : square_flat_bounding_box[2],
                square_flat_bounding_box[1] : square_flat_bounding_box[3],
            ]

            print(f"grain cropped image shape {grain_cropped_image.shape}")

            grain_cropped_tensor = grain_tensor_full_mask[
                square_flat_bounding_box[0] : square_flat_bounding_box[2],
                square_flat_bounding_box[1] : square_flat_bounding_box[3],
                :,
            ]

            print(f"grain cropped tensor shape {grain_cropped_tensor.shape}")

            # Update background class to reflect the removal of any non-connected grains
            grain_cropped_tensor = Grains.update_background_class(grain_mask_tensor=grain_cropped_tensor)

            assert grain_cropped_image.shape[0] == grain_cropped_image.shape[1]
            print(f"grain cropped tensor shape {grain_cropped_tensor.shape}")
            assert grain_cropped_tensor.shape[0] == grain_cropped_tensor.shape[1]
            # Check that the bounding box is square
            bounding_box_shape = (
                square_flat_bounding_box[2] - square_flat_bounding_box[0],
                square_flat_bounding_box[3] - square_flat_bounding_box[1],
            )
            print(f"bounding box shape {bounding_box_shape}")
            assert bounding_box_shape[0] == bounding_box_shape[1]
            # Check bounding box shape is same as image shape and first two dimensions of tensor
            assert bounding_box_shape == grain_cropped_image.shape
            assert bounding_box_shape == (grain_cropped_tensor.shape[0], grain_cropped_tensor.shape[1])

            graincrops[grain_number] = GrainCrop(
                image=grain_cropped_image,
                mask=grain_cropped_tensor,
                padding=padding,
                bbox=square_flat_bounding_box,
                pixel_to_nm_scaling=pixel_to_nm_scaling,
                filename=filename,
            )

        return graincrops

    @staticmethod
    def graincrops_remove_objects_too_small_to_process(
        graincrops: dict[int, GrainCrop],
        min_object_size: int,
        min_object_bbox_size: int,
    ) -> dict[int, GrainCrop]:
        """
        Remove objects that are too small to process from each class of the grain crops.

        Parameters
        ----------
        graincrops : dict[int, GrainCrop]
            Dictionary of grain crops.
        min_object_size : int
            Minimum object size to keep (pixels).
        min_object_bbox_size : int
            Minimum object bounding box size to keep (pixels^2).

        Returns
        -------
        dict[int, GrainCrop]
            Dictionary of grain crops with objects too small to process removed.
        """
        for _grain_number, graincrop in graincrops.items():
            # Iterate over the classes
            for class_index in range(1, graincrop.mask.shape[2]):
                # Get the binary mask for the class
                class_mask = graincrop.mask[:, :, class_index]

                # Label the regions
                labelled_regions = Grains.label_regions(class_mask)
                region_properties = Grains.get_region_properties(labelled_regions)

                # Iterate over the regions
                for region in region_properties:
                    # Get the region mask
                    region_mask = labelled_regions == region.label
                    region_bbox = region.bbox

                    # Check the region size
                    if (
                        region.area < min_object_size
                        or (region_bbox[2] - region_bbox[0]) < min_object_bbox_size
                        or (region_bbox[3] - region_bbox[1]) < min_object_bbox_size
                    ):
                        # Remove the region from the class
                        graincrop.mask[:, :, class_index] = np.where(
                            region_mask,
                            0,
                            graincrop.mask[:, :, class_index],
                        )

            # Update the background class
            graincrop.mask = Grains.update_background_class(graincrop.mask)

        return graincrops

    @staticmethod
    def graincrops_merge_classes(
        graincrops: dict[int, GrainCrop],
        classes_to_merge: list[tuple[int]] | None,
    ) -> dict[int, GrainCrop]:
        """
        Merge classes in the grain crops.

        Parameters
        ----------
        graincrops : dict[int, GrainCrop]
            Dictionary of grain crops.
        classes_to_merge : list | None
            List of tuples for classes to merge, can be any number of classes.

        Returns
        -------
        dict[int, GrainCrop]
            Dictionary of grain crops with classes merged.
        """
        if classes_to_merge is None:
            return graincrops

        for _grain_number, graincrop in graincrops.items():
            graincrop.mask = Grains.merge_classes(
                grain_mask_tensor=graincrop.mask,
                classes_to_merge=classes_to_merge,
            )

        return graincrops

    @staticmethod
    def graincrops_update_background_class(
        graincrops: dict[int, GrainCrop],
    ) -> dict[int, GrainCrop]:
        """
        Update the background class in the grain crops.

        Parameters
        ----------
        graincrops : dict[int, GrainCrop]
            Dictionary of grain crops.

        Returns
        -------
        dict[int, GrainCrop]
            Dictionary of grain crops with updated background class.
        """
        for _grain_number, graincrop in graincrops.items():
            graincrop.mask = Grains.update_background_class(graincrop.mask)

        return graincrops
