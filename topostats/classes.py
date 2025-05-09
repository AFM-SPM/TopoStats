"""Define custom classes for TopoStats."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from topostats.logs.logs import LOGGER_NAME
from topostats.utils import construct_full_mask_from_graincrops, update_background_class

LOGGER = logging.getLogger(LOGGER_NAME)

# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-instance-attributes


class GrainCrop:
    """
    Class for storing the crops of grains.

    Parameters
    ----------
    image : npt.NDArray[np.float32]
        2-D Numpy array of the cropped image.
    mask : npt.NDArray[np.bool_]
        3-D Numpy tensor of the cropped mask.
    padding : int
        Padding added to the bounding box of the grain during cropping.
    bbox : tuple[int, int, int, int]
        Bounding box of the crop including padding.
    pixel_to_nm_scaling : float
        Pixel to nanometre scaling factor for the crop.
    filename : str
        Filename of the image from which the crop was taken.
    height_profiles : dict[int, [int, npt.NDArray[np.float32]]] | None
        3-D Numpy tensor of the height profiles.
    stats : dict[int, dict[int, Any]] | None
        Dictionary of grain statistics.
    """

    def __init__(
        self,
        image: npt.NDArray[np.float32],
        mask: npt.NDArray[np.bool_],
        padding: int,
        bbox: tuple[int, int, int, int],
        pixel_to_nm_scaling: float,
        filename: str,
        height_profiles: dict[int, dict[int, npt.NDArray[np.float32]]] | None = None,
        stats: dict[int, dict[int, Any]] | None = None,
    ):
        """
        Initialise the class.

        Parameters
        ----------
        image : npt.NDArray[np.float32]
            2-D Numpy array of the cropped image.
        mask : npt.NDArray[np.bool_]
            3-D Numpy tensor of the cropped mask.
        padding : int
            Padding added to the bounding box of the grain during cropping.
        bbox : tuple[int, int, int, int]
            Bounding box of the crop including padding.
        pixel_to_nm_scaling : float
            Pixel to nanometre scaling factor for the crop.
        filename : str
            Filename of the image from which the crop was taken.
        height_profiles : dict[int, [int, npt.NDArray[np.float32]]] | None
            3-D Numpy tensor of the height profiles.
        stats : dict[int, dict[int, Any]] | None
            Dictionary of grain statistics.
        """
        self.padding = padding
        self.image = image
        # This part of the constructor must go after padding since the setter
        # for mask requires the padding.
        self.mask = mask
        self.bbox = bbox
        self.pixel_to_nm_scaling = pixel_to_nm_scaling
        self.filename = filename
        self.height_profiles = height_profiles
        self.stats = stats
        self.disordered_traces: dict[str:Any] = {}

    @property
    def image(self) -> npt.NDArray[np.float32]:
        """
        Getter for the ``image`` attribute.

        Returns
        -------
        npt.NDArray
            Numpy array of the image.
        """
        return self._image

    @image.setter
    def image(self, value: npt.NDArray[np.float32]):
        """
        Setter for the ``image`` attribute.

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
        Getter for the ``mask`` attribute.

        Returns
        -------
        npt.NDArray[np.bool_]
            Numpy array of the mask.
        """
        return self._mask

    @mask.setter
    def mask(self, value: npt.NDArray[np.bool_]) -> None:
        """
        Setter for the ``mask`` attribute.

        Parameters
        ----------
        value : npt.NDArray
            Numpy array of the ``mask`` attribute.

        Raises
        ------
        ValueError
            If the mask dimensions do not match the image.
        """
        if value.shape[0] != self.image.shape[0] or value.shape[1] != self.image.shape[1]:
            raise ValueError(f"Mask dimensions do not match image: {value.shape} vs {self.image.shape}")
        # Ensure that the padding region is blank, set it to be blank if not
        for class_index in range(1, value.shape[2]):
            class_mask = value[:, :, class_index]

            padded_region_top = class_mask[: self.padding, :]
            padded_region_bottom = class_mask[-self.padding :, :]
            padded_region_left = class_mask[:, : self.padding]
            padded_region_right = class_mask[:, -self.padding :]
            if (
                np.any(padded_region_top)
                or np.any(padded_region_bottom)
                or np.any(padded_region_left)
                or np.any(padded_region_right)
            ):
                LOGGER.warning("Padding region is not blank, setting to blank")
                value[: self.padding, :, class_index] = 0
                value[-self.padding :, :, class_index] = 0
                value[:, : self.padding, class_index] = 0
                value[:, -self.padding :, class_index] = 0

        # Update background class in case the mask has been edited
        value = update_background_class(value)
        self._mask: npt.NDArray[np.bool_] = value

    @property
    def padding(self) -> int:
        """
        Getter for the ``padding`` attribute.

        Returns
        -------
        int
            The padding amount.
        """
        return self._padding

    @padding.setter
    def padding(self, value: int) -> None:
        """
        Setter for the ``padding`` attribute.

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
        Getter for the bounding box attribute.

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
    def bbox(self, value: tuple[int, int, int, int]) -> None:
        """
        Setter for the bounding box attribute.

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
        Getter for the pixel to nanometre scaling factor attribute.

        Returns
        -------
        float
            Pixel to nanometre scaling factor.
        """
        return self._pixel_to_nm_scaling

    @pixel_to_nm_scaling.setter
    def pixel_to_nm_scaling(self, value: float) -> None:
        """
        Setter for the pixel to nanometre scaling factor attribute.

        Parameters
        ----------
        value : float
            Pixel to nanometre scaling factor.
        """
        self._pixel_to_nm_scaling = value

    @property
    def filename(self) -> str:
        """
        Getter for the ``filename`` attribute.

        Returns
        -------
        str
            The image ``filename`` attribute.
        """
        return self._filename

    @filename.setter
    def filename(self, value: str) -> None:
        """
        Setter for the ``filename`` attribute.

        Parameters
        ----------
        value : str
            Image ``filename`` attribute.
        """
        self._filename = value

    @property
    def height_profiles(self) -> npt.NDArray:
        """
        Getter for the ``height_profile`` attribute.

        Returns
        -------
        str
            The image height_profile.
        """
        return self._height_profiles

    @height_profiles.setter
    def height_profiles(self, value: npt.NDArray) -> None:
        """
        Setter for the ``height_profile`` attribute.

        Parameters
        ----------
        value : str
            Image ``height_profile``.
        """
        self._height_profiles = value

    @property
    def stats(self) -> dict[str, Any]:
        """
        Getter for the stats.

        Returns
        -------
        str
            Dictionary of image statistics.
        """
        return self._stats

    @stats.setter
    def stats(self, value: dict[str, Any]) -> None:
        """
        Setter for the stats.

        Parameters
        ----------
        value : dict[str, Any]
            Image stats.
        """
        self._stats = value

    @property
    def disordered_traces(self) -> dict[str:Any]:
        """
        Getter for the ``disordered_traces`` attribute.

        Returns
        -------
        dict[str: Any]
            Returns the value of ``disordered_traces``.
        """
        return self._disordered_traces

    @disordered_traces.setter
    def disordered_traces(self, value: dict[str:Any]) -> None:
        """
        Setter for the ``disordered_traces`` attribute.

        Parameters
        ----------
        value : dict[str: Any]
            Value to set for ``disordered_traces``.
        """
        self._disordered_traces = value

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
            and self.stats == other.stats
            and self.height_profiles == other.height_profiles
        )

    def grain_crop_to_dict(self) -> dict[str, Any]:
        """
        Convert GrainCrop to dictionary indexed by attributes.

        Returns
        -------
        dict[str, Any]
            Dictionary indexed by attribute of the grain attributes.
        """
        return {re.sub(r"^_", "", key): value for key, value in self.__dict__.items()}

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
    if len(array.shape) != 3 or array.shape[2] < 2:
        raise ValueError(f"Full mask tensor must be WxHxC with C >= 2 but has shape {array.shape}")
    return array


@dataclass
class GrainCropsDirection:
    """
    Dataclass for storing the crops of grains in a particular imaging direction.

    Attributes
    ----------
    full_mask_tensor : npt.NDArray[np.bool_]
        Boolean WxHxC array of the full mask tensor (W = width ; H = height; C = class >= 2).
    crops : dict[int, GrainCrops]
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

    def grain_crops_direction_to_dict(self) -> dict[str, npt.NDArray[np.bool_] | dict[str:Any]]:
        """
        Convert GrainCropsDirection to dictionary indexed by attributes.

        Returns
        -------
        dict[str, Any]
            Dictionary indexed by attribute of the grain attributes.
        """
        return {re.sub(r"^_", "", key): value for key, value in self.__dict__.items()}

    def debug_locate_difference(self, other: object) -> None:
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
                crop.debug_locate_difference(other.crops[crop_index])
        if not np.array_equal(self.full_mask_tensor, other.full_mask_tensor):
            raise ValueError("Full mask tensor is different")

        LOGGER.info("Cannot find difference between graincrops")

    def update_full_mask_tensor(self):
        """Update the full mask tensor from the grain crops."""
        self.full_mask_tensor = construct_full_mask_from_graincrops(
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

    def image_grain_crops_to_dict(self) -> dict[str, npt.NDArray[np.bool_] | dict[str:Any]]:
        """
        Convert ImageGrainCrops to dictionary indexed by attributes.

        Returns
        -------
        dict[str, Any]
            Dictionary indexed by attribute of the grain attributes.
        """
        return {re.sub(r"^_", "", key): value for key, value in self.__dict__.items()}

    def debug_locate_difference(self, other: object) -> None:
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
                self.above.debug_locate_difference(other.above)
        else:
            if other.above is not None:
                raise ValueError("Above grains are different")
        if self.below is not None:
            if self.below != other.below:
                LOGGER.info("Below grains are different")
                self.below.debug_locate_difference(other.below)
        else:
            if other.below is not None:
                raise ValueError("Below grains are different")

        LOGGER.info("Cannot find difference between image grain crops")


@dataclass
class TopoStats:
    """
    Class for storing TopoStats objects.

    Attributes
    ----------
    image_grain_crops : ImageGrainCrops | None
        ImageGrainCrops of processed image.
    filename : str | None
        Filename.
    pixel_to_nm_scaling : str | None
        Pixel to nanometre scaling.
    img_path : str | None
        Original path to image.
    image : npt.NDArray | None
        Flattened image (post ``Filter()``).
    image_original : npt.NDArray | None
        Original image.
    topostats_version : str | None
        TopoStats version.
    """

    image_grain_crops: ImageGrainCrops | None
    filename: str | None
    pixel_to_nm_scaling: str | None
    img_path: Path | str | None
    image: npt.NDArray | None
    image_original: npt.NDArray | None
    topostats_version: str | None

    def __eq__(self, other: object) -> bool:
        """
        Check if two TopoStats objects are equal.

        Parameters
        ----------
        other : object
          Object to compare to.

        Returns
        -------
        bool
          True if the objects are equal, False otherwise.
        """
        if not isinstance(other, TopoStats):
            return False
        return (
            self.image_grain_crops == other.image_grain_crops
            and self.filename == other.filename
            and self.pixel_to_nm_scaling == other.pixel_to_nm_scaling
            and self.topostats_version == other.topostats_version
            and self.img_path == other.img_path
            and np.all(self.image == other.image)
            and np.all(self.image_original == other.image_original)
        )

    @property
    def image_grain_crops(self) -> ImageGrainCrops:
        """
        Getter for the Image Grain Crops.

        Returns
        -------
        ImageGrainCrops
            Image Grain Crops.
        """
        return self._image_grain_crops

    @image_grain_crops.setter
    def image_grain_crops(self, value: ImageGrainCrops) -> None:
        """
        Setter for the ``image_grain_crops`` attribute.

        Parameters
        ----------
        value : ImageGrainCrops
            Image Grain Crops for the image.
        """
        self._image_grain_crops = value

    @property
    def filename(self) -> str:
        """
        Getter for the ``filename`` attribute.

        Returns
        -------
        str
            Image filename.
        """
        return self._filename

    @filename.setter
    def filename(self, value: str) -> None:
        """
        Setter for the ``filename`` attribute.

        Parameters
        ----------
        value : str
            Filename for the image.
        """
        self._filename = value

    @property
    def pixel_to_nm_scaling(self) -> str:
        """
        Getter for the ``pixel_to_nm_scaling`` attribute.

        Returns
        -------
        str
            Image ``pixel_to_nm_scaling``.
        """
        return self._pixel_to_nm_scaling

    @pixel_to_nm_scaling.setter
    def pixel_to_nm_scaling(self, value: str) -> None:
        """
        Setter for the ``pixel_to_nm_scaling`` attribute.

        Parameters
        ----------
        value : str
            Pixel to nanometre scaling for the image.
        """
        self._pixel_to_nm_scaling = value

    @property
    def img_path(self) -> Path:
        """
        Getter for the ``img_path`` attribute.

        Returns
        -------
        Path
            Path to original image on disk.
        """
        return self._img_path

    @img_path.setter
    def img_path(self, value: str | Path | None) -> None:
        """
        Setter for the ``img_path`` attribute.

        Parameters
        ----------
        value : str | Path | None
            Image Path for the image.
        """
        self._img_path = Path.cwd() if value is None else Path(value)

    @property
    def image(self) -> str:
        """
        Getter for the ``image`` attribute, post filtering.

        Returns
        -------
        str
            Image image.
        """
        return self._image

    @image.setter
    def image(self, value: str) -> None:
        """
        Setter for the ``image`` attribute.

        Parameters
        ----------
        value : str
            Filtered image.
        """
        self._image = value

    @property
    def image_original(self) -> str:
        """
        Getter for the ``image_original`` attribute.

        Returns
        -------
        str
            Original image.
        """
        return self._image_original

    @image_original.setter
    def image_original(self, value: str) -> None:
        """
        Setter for the ``image_original`` attribute.

        Parameters
        ----------
        value : str
            Original image.
        """
        self._image_original = value

    @property
    def topostats_version(self) -> str:
        """
        Getter for the ``topostats_version`` attribute, post filtering.

        Returns
        -------
        str
            Version of TopoStats the class was created with.
        """
        return self._topostats_version

    @topostats_version.setter
    def topostats_version(self, value: str) -> None:
        """
        Setter for the ``topostats_version`` attribute.

        Parameters
        ----------
        value : str
            Topostats version.
        """
        self._topostats_version = value

    def topostats_to_dict(self) -> dict[str, str | ImageGrainCrops | npt.NDArray]:
        """
        Convert ``TopoStats`` object to dictionary.

        Returns
        -------
        dict[str, str | ImageGrainCrops | npt.NDArray]
            Dictionary of ``TopoStats`` object.
        """
        return {re.sub(r"^_", "", key): value for key, value in self.__dict__.items()}
