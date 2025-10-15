"""Define custom classes for TopoStats."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from topostats.logs.logs import LOGGER_NAME
from topostats.utils import construct_full_mask_from_graincrops, update_background_class

LOGGER = logging.getLogger(LOGGER_NAME)

# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-lines
# F811 - redefined-while-unused : we disable this as we want to have getter/setter methods for each attribute
# ruff: noqa: F811


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
    thresholds : float
        Thresholds used to find the grain.
    filename : str
        Filename of the image from which the crop was taken.
    skeleton : npt.NDArray[np.bool_]
        3-D Numpy tensor of the skeletonised mask.
    height_profiles : dict[int, [int, npt.NDArray[np.float32]]] | None
        3-D Numpy tensor of the height profiles.
    stats : dict[int, dict[int, Any]] | None
        Dictionary of grain statistics.
    disordered_trace : DisorderedTrace
        A disordered trace for the grain.
    nodes : dict[str, Nodes]
        Dictionary of grain nodes.
    ordered_trace : OrderedTrace
        An ordered trace for the grain.
    threshold_method : str
        Threshold method used to find grains.
    """

    def __init__(
        self,
        image: npt.NDArray[np.float32],
        mask: npt.NDArray[np.bool_],
        padding: int,
        bbox: tuple[int, int, int, int],
        pixel_to_nm_scaling: float,
        thresholds: list[float],
        filename: str,
        skeleton: npt.NDArray[np.bool_] | None = None,
        height_profiles: dict[int, dict[int, npt.NDArray[np.float32]]] | None = None,
        stats: dict[int, dict[int, Any]] | None = None,
        disordered_trace: DisorderedTrace | None = None,
        nodes: dict[str, Node] | None = None,
        ordered_trace: OrderedTrace | None = None,
        threshold_method: str | None = None,
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
        thresholds : list[float]
            A list of thresholds used to identify the grain.
        filename : str
            Filename of the image from which the crop was taken.
        skeleton : npt.NDArray[np.bool_]
            3-D Numpy tensor of the skeletonised mask.
        height_profiles : dict[int, [int, npt.NDArray[np.float32]]] | None
            3-D Numpy tensor of the height profiles.
        stats : dict[int, dict[int, Any]] | None
            Dictionary of grain statistics.
        disordered_trace : DisorderedTrace
            A disordered trace for the current grain.
        nodes : dict[int, Node] | None
            Grain nodes.
        ordered_trace : OrderedTrace | None
            An ordered trace for the grain.
        threshold_method : str
            Threshold method used to find grains.
        """
        self.padding = padding
        self.image = image
        # This part of the constructor must go after padding since the setter
        # for mask requires the padding.
        self.mask = mask
        self.bbox = bbox
        self.pixel_to_nm_scaling = pixel_to_nm_scaling
        self.thresholds = thresholds
        self.filename = filename
        self.height_profiles = height_profiles
        self.stats = stats
        self.skeleton: npt.NDArray[np.bool_] | None = skeleton
        self.disordered_trace: DisorderedTrace | None = disordered_trace
        self.nodes: dict[int, Node] | None = nodes
        self.ordered_trace: OrderedTrace | None = ordered_trace
        self.threshold_method: str = threshold_method

    def __getstate__(self) -> dict[str, Any]:
        """
        Get the state of the class.

        Returns
        -------
        dict[str, Any]
            Dictionary of the current state of the classes attributes.
        """
        return {
            "_padding": self._padding,
            "_image": self._image,
            "_mask": self._mask,
            "_bbox": self._bbox,
            "_pixel_to_nm_scaling": self._pixel_to_nm_scaling,
            "_thresholds": self._thresholds,
            "_filename": self._filename,
            "_height_profiles": self._height_profiles,
            "_stats": self._stats,
            "_skeleton": self._skeleton,
            "_disordered_trace": self._disordered_trace,
            "_nodes": self._nodes,
            "_ordered_trace": self._ordered_trace,
            "_threshold_method": self._threshold_method,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Set the state of the class.

        Parameters
        ----------
        state : dict[str, Any]
            A dictionary holding the state of all parameters for the class.
        """
        self._padding = state["_padding"]
        self._image = state["_image"]
        self._mask = state["_mask"]
        self._bbox = state["_bbox"]
        self._pixel_to_nm_scaling = state["_pixel_to_nm_scaling"]
        self._thresholds = state["_thresholds"]
        self._filename = state["_filename"]
        self._height_profiles = state["_height_profiles"]
        self._stats = state["_stats"]
        self._skeleton = state["_skeleton"]
        self._disordered_trace = state["_disordered_trace"]
        self._nodes = state["_nodes"]
        self._ordered_trace = state["_ordered_trace"]
        self._threshold_method = state["_threshold_method"]

    def __eq__(self, other: GrainCrop) -> bool:
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
            and self.thresholds == other.thresholds
            and self.filename == other.filename
            and self.stats == other.stats
            and self.height_profiles == other.height_profiles
            and self.disordered_trace == other.disordered_trace
            and np.array_equal(self.skeleton, other.skeleton)
            and self.nodes == other.nodes
            and self.threshold_method == other.threshold_method
        )

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
    def thresholds(self) -> list[float]:
        """
        Getter for the ``thresholds`` attribute.

        Returns
        -------
        list[float]
            Returns the value of ``thresholds``.
        """
        return self._thresholds

    @thresholds.setter
    def thresholds(self, value: list[float]) -> None:
        """
        Setter for the ``thresholds`` attribute.

        Parameters
        ----------
        value : list[float]
            Value to set for ``thresholds``.
        """
        self._thresholds = value

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
    def skeleton(self) -> npt.NDArray:
        """
        Getter for the ``skeleton`` attribute.

        Returns
        -------
        npt.NDArray
            Returns the value of ``skeleton``.
        """
        return self._skeleton

    @skeleton.setter
    def skeleton(self, value: npt.NDArray) -> None:
        """
        Setter for the ``skeleton`` attribute.

        Parameters
        ----------
        value : npt.NDArray
            Value to set for ``skeleton``.
        """
        self._skeleton = value

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
    def disordered_trace(self) -> DisorderedTrace:
        """
        Getter for the ``disordered_trace`` attribute.

        Returns
        -------
        dict[str: Any]
            Returns the value of ``disordered_trace``.
        """
        return self._disordered_trace

    @disordered_trace.setter
    def disordered_trace(self, value: DisorderedTrace) -> None:
        """
        Setter for the ``disordered_trace`` attribute.

        Parameters
        ----------
        value : dict[str: Any]
            Value to set for ``disordered_trace``.
        """
        self._disordered_trace = value

    @property
    def nodes(self) -> dict[int, Node]:
        """
        Getter for the ``nodes`` attribute.

        Returns
        -------
        dict[int, Nodes]
            Returns ``nodes``, a dictionary of Nodes.
        """
        return self._nodes

    @nodes.setter
    def nodes(self, value: Node) -> None:
        """
        Setter for the ``nodes`` attribute.

        Parameters
        ----------
        value : Nodes
            Value to set for ``nodes``.
        """
        self._nodes = value

    @property
    def ordered_trace(self) -> OrderedTrace:
        """
        Getter for the ``ordered_trace`` attribute.

        Returns
        -------
        dict[str: Any]
            Returns the value of ``ordered_trace``.
        """
        return self._ordered_trace

    @ordered_trace.setter
    def ordered_trace(self, value: OrderedTrace) -> None:
        """
        Setter for the ``ordered_trace`` attribute.

        Parameters
        ----------
        value : dict[str: Any]
            Value to set for ``ordered_trace``.
        """
        self._ordered_trace = value

    @property
    def threshold_method(self) -> list[float]:
        """
        The ``threshold_method`` used to find the grain.

        Returns
        -------
        list[float]
            Returns the value of ``threshold_method``.
        """
        return self._threshold_method

    @threshold_method.setter
    def threshold_method(self, value: list[float]) -> None:
        """
        Setter for the ``threshold_method`` attribute.

        Parameters
        ----------
        value : list[float]
            Value to set for ``threshold_method``.
        """
        self._threshold_method = value

    def grain_crop_to_dict(self) -> dict[str, Any]:
        """
        Convert GrainCrop to dictionary indexed by attributes.

        Returns
        -------
        dict[str, Any]
            Dictionary indexed by attribute of the grain attributes.
        """
        return {re.sub(r"^_", "", key): value for key, value in self.__dict__.items()}

    def debug_locate_difference(self, other: object) -> None:  # noqa: C901 # pylint: disable=too-many-branches
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
            raise ValueError(f"Image is different\n self.image  : {self.image}\n other.image : {other.image}")
        if not np.array_equal(self.mask, other.mask):
            raise ValueError(f"Mask is different\n self.mask  : {self.mask}\n other.mask : {other.mask}")
        if self.padding != other.padding:
            raise ValueError(f"Padding is different\n self.padding  : {self.padding}\n other.padding : {other.padding}")
        if self.bbox != other.bbox:
            raise ValueError(f"Bounding box is different\n self.bbox  : {self.bbox}\n other.bbox : {other.bbox}")
        if self.pixel_to_nm_scaling != other.pixel_to_nm_scaling:
            raise ValueError(
                "Pixel to nm scaling is different\n"
                f" self.pixel_to_nm_scaling  : {self.pixel_to_nm_scaling}\n"
                f" other.pixel_to_nm_scaling : {other.pixel_to_nm_scaling}"
            )
        if self.thresholds != other.thresholds:
            raise ValueError(
                f"Thresholds differ\n self.thresholds  : {self.thresholds}\n other.thresholds : {other.thresholds}"
            )
        if self.filename != other.filename:
            raise ValueError(
                f"Filename is different\n self.filename  : {self.filename}\n other.filename : {other.filename}"
            )
        if self.height_profiles != other.height_profiles:
            raise ValueError(
                "Height profiles are different\n"
                f" self.height_profiles  : {self.height_profiles}\n"
                f" other.height_profiles : {other.height_profiles}"
            )
        if self.skeleton != other.skeleton:
            raise ValueError(
                f"Skeleton is different\n self.skeleton  : {self.skeleton}\n other.skeleton : {other.skeleton}"
            )
        if self.disordered_trace != other.disordered_trace:
            raise ValueError(
                "Disordered traces are different\n"
                f" self.disordered_trace  : {self.disordered_trace}\n"
                f" other.disordered_trace : {other.disordered_trace}"
            )
        if self.nodes != other.nodes:
            raise ValueError(f"Nodes are different\n self.nodes  : {self.nodes}\n other.nodes : {other.nodes}")
        if self.threshold_method != other.threshold_method:
            raise ValueError(
                "Threshold Method is different\n"
                f" self.threshold_method  : {self.threshold_method}\n"
                f" other.threshold_method : {other.threshold_method}"
            )
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
class DisorderedTrace:
    """
    Dataclass for storing the disordered tracing data.

    Attributes
    ----------
    images : dict[str: npt.NDArray]
        Dictionary of images generated during disordered tracing, should include ''pruned_skeleton''.
    grain_endpoints : npt.int64
        Number of Grain endpoints.
    grain_junctions : npt.int64
        Number of Grain junctions.
    total_branch_length : float
        Total branch length in nanometres.
    grain_width_mean : float
        Mean grain width in nanometres.
    """

    images: dict[str : npt.NDArray] = None
    grain_endpoints: npt.int64 = None
    grain_junctions: npt.int64 = None
    total_branch_length: float = None
    grain_width_mean: float = None

    def __getstate__(self) -> dict[str, Any]:
        """
        Get the state of the class.

        Returns
        -------
        dict[str, Any]
            Dictionary of the current state of the classes attributes.
        """
        return {
            "_images": self._images,
            "_grain_endpoints": self._grain_endpoints,
            "_grain_junctions": self._grain_junctions,
            "_total_branch_length": self._total_branch_length,
            "_grain_width_mean": self._grain_width_mean,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Set the state of the class.

        Parameters
        ----------
        state : dict[str, Any]
            A dictionary holding the state of all parameters for the class.
        """
        self._images = state["_images"]
        self._grain_endpoints = state["_grain_endpoints"]
        self._grain_junctions = state["_grain_junctions"]
        self._total_branch_length = state["_total_branch_length"]
        self._grain_width_mean = state["_grain_width_mean"]

    def __str__(self) -> str:
        """
        Readable attributes.

        Returns
        -------
        str
            Set of formatted statistics on matched branches.
        """
        return (
            f"images : {self.images}\n"
            f"grain_endpoints : {self.grain_endpoints}\n"
            f"grain_junctions : {self.grain_junctions}\n"
            f"total_branch_length : {self.total_branch_length}\n"
            f"grain_width_mean : {self.grain_width_mean}"
        )

    def __eq__(self, other: DisorderedTrace) -> bool:
        """
        Check if two DisorderedTrace objects are equal.

        Parameters
        ----------
        other : object
            Object to compare to.

        Returns
        -------
        bool
            True if the objects are equal, False otherwise.
        """
        if not isinstance(other, DisorderedTrace):
            return False
        # Check whether all images are equal
        if len(self.images) != len(other.images):
            return False
        images_equal: dict[int, bool] = {}
        for image, array in self.images.items():
            images_equal[image] = np.array_equal(array, other.images[image])
        if not all(images_equal):
            return False
        all_images_equal = True
        return (
            all_images_equal
            and self.grain_endpoints == other.grain_endpoints
            and self.grain_junctions == other.grain_junctions
            and self.total_branch_length == other.total_branch_length
            and self.grain_width_mean == other.grain_width_mean
        )

    @property
    def images(self) -> dict[str, npt.NDArray]:
        """
        Getter for the ``images`` attribute.

        Returns
        -------
        dict[str, npt.NDArray]
            Returns the value of ``images``.
        """
        return self._images

    @images.setter
    def images(self, value: dict[str, npt.NDArray]) -> None:
        """
        Setter for the ``images`` attribute.

        Parameters
        ----------
        value : dict[str, npt.NDArray]
            Value to set for ``images``.
        """
        self._images = value

    @property
    def grain_endpoints(self) -> npt.int64:
        """
        Getter for the ``grain_endpoints`` attribute.

        Returns
        -------
        npt.int64
            Returns the value of ``grain_endpoints``.
        """
        return self._grain_endpoints

    @grain_endpoints.setter
    def grain_endpoints(self, value: npt.int64) -> None:
        """
        Setter for the ``grain_endpoints`` attribute.

        Parameters
        ----------
        value : npt.int64
            Value to set for ``grain_endpoints``.
        """
        self._grain_endpoints = value

    @property
    def grain_junctions(self) -> npt.int64:
        """
        Getter for the ``grain_junctions`` attribute.

        Returns
        -------
        npt.int64
            Returns the value of ``grain_junctions``.
        """
        return self._grain_junctions

    @grain_junctions.setter
    def grain_junctions(self, value: npt.int64) -> None:
        """
        Setter for the ``grain_junctions`` attribute.

        Parameters
        ----------
        value : npt.int64
            Value to set for ``grain_junctions``.
        """
        self._grain_junctions = value

    @property
    def total_branch_length(self) -> float:
        """
        Getter for the ``total_branch_length`` attribute.

        The length of all branches in nanometres.

        Returns
        -------
        float
            Returns the value of ``total_branch_length``.
        """
        return self._total_branch_length

    @total_branch_length.setter
    def total_branch_length(self, value: float) -> None:
        """
        Setter for the ``total_branch_length`` attribute.

        Parameters
        ----------
        value : float
            Value to set for ``total_branch_length``.
        """
        self._total_branch_length = value

    @property
    def grain_width_mean(self) -> float:
        """
        Getter for the ``grain_width_mean`` attribute.

        The mean grain width in nanometers.

        Returns
        -------
        float
            Returns the value of ``grain_width_mean``.
        """
        return self._grain_width_mean

    @grain_width_mean.setter
    def grain_width_mean(self, value: float) -> None:
        """
        Setter for the ``grain_width_mean`` attribute.

        Parameters
        ----------
        value : float
            Value to set for ``grain_width_mean``.
        """
        self._grain_width_mean = value

    def disordered_trace_to_dict(self) -> dict[str, Any]:
        """
        Convert DisorderedTrace to dictionary indexed by attributes.

        Returns
        -------
        dict[str, Any]
            Dictionary indexed by attribute of the grain attributes.
        """
        return {re.sub(r"^_", "", key): value for key, value in self.__dict__.items()}


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

    crops: dict[int, GrainCrop] = None
    full_mask_tensor: npt.NDArray[np.bool_] = None

    def __getstate__(self) -> dict[str, Any]:
        """
        Get the state of the class.

        Returns
        -------
        dict[str, Any]
            Dictionary of the current state of the classes attributes.
        """
        return {"_crops": self._crops, "_full_mask_tensor": self._full_mask_tensor}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Set the state of the class.

        Parameters
        ----------
        state : dict[str, Any]
            A dictionary holding the state of all parameters for the class.
        """
        self._crops = state["_crops"]
        self._full_mask_tensor = state["_full_mask_tensor"]

    def __post_init__(self):
        """
        Validate the full mask tensor shape.

        Raises
        ------
        ValueError
            If the full mask tensor shape is invalid.
        """
        self._full_mask_tensor = validate_full_mask_tensor_shape(self.full_mask_tensor)

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

    @property
    def crops(self) -> dict[int, GrainCrop]:
        """
        Getter for the ``crops`` attribute.

        Returns
        -------
        dict[int, GrainCrop]
            Returns the value of ``crops``.
        """
        return self._crops

    @crops.setter
    def crops(self, value: dict[int, GrainCrop]) -> None:
        """
        Setter for the ``crops`` attribute.

        Parameters
        ----------
        value : dict[int, GrainCrop]
            Value to set for ``crops``.
        """
        self._crops = value

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
    above : GrainCropsDirection | None
        Grains in the above direction.
    below : GrainCropsDirection | None
        Grains in the below direction.
    """

    above: GrainCropsDirection | None = None
    below: GrainCropsDirection | None = None

    def __getstate__(self) -> dict[str, Any]:
        """
        Get the state of the class.

        Returns
        -------
        dict[str, Any]
            Dictionary of the current state of the classes attributes.
        """
        return {"_above": self._above, "_below": self._below}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Set the state of the class.

        Parameters
        ----------
        state : dict[str, Any]
            A dictionary holding the state of all parameters for the class.
        """
        self._above = state["_above"]
        self._below = state["_below"]

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

    @property
    def above(self) -> GrainCropsDirection:
        """
        Getter for the ``above`` attribute.

        Returns
        -------
        GrainCropsDirection
            Returns the value of ``above``.
        """
        return self._above

    @above.setter
    def above(self, value: GrainCropsDirection) -> None:
        """
        Setter for the ``above`` attribute.

        Parameters
        ----------
        value : GrainCropsDirection
            Value to set for ``above``.
        """
        self._above = value

    @property
    def below(self) -> GrainCropsDirection:
        """
        Getter for the ``below`` attribute.

        Returns
        -------
        GrainCropsDirection
            Returns the value of ``below``.
        """
        return self._below

    @below.setter
    def below(self, value: GrainCropsDirection) -> None:
        """
        Setter for the ``below`` attribute.

        Parameters
        ----------
        value : GrainCropsDirection
            Value to set for ``below``.
        """
        self._below = value

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
    full_mask_tensor : npt.NDArray
        Tensor mask for the full image.
    topostats_version : str | None
        TopoStats version.
    config : dict[str, Any] | None
        Configuration used when processing the grain.
    """

    # @ns-rse 2025-10-10 : This will switch to dict[int, GrainCrop] when we remove ImageGrainCrops and
    #                      GrainCropsDirection
    image_grain_crops: ImageGrainCrops | None = None
    filename: str | None = None
    pixel_to_nm_scaling: str | None = None
    img_path: Path | str | None = None
    image: npt.NDArray | None = None
    image_original: npt.NDArray | None = None
    full_mask_tensor: npt.NDArray | None = None
    topostats_version: str | None = None
    config: dict[str, Any] | None = None

    def __getstate__(self) -> dict[str, Any]:
        """
        Get the state of the class.

        Returns
        -------
        dict[str, Any]
            Dictionary of the current state of the classes attributes.
        """
        return {
            "_image_grain_crops": self._image_grain_crops,
            "_filename": self._filename,
            "_pixel_to_nm_scaling": self._pixel_to_nm_scaling,
            "_img_path": self._img_path,
            "_image": self._image,
            "_image_original": self._image_original,
            "_full_mask_tensor": self._full_mask_tensor,
            "_topostats_version": self._topostats_version,
            "_config": self._config,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Set the state of the class.

        Parameters
        ----------
        state : dict[str, Any]
            A dictionary holding the state of all parameters for the class.
        """
        self._image_grain_crops = state["_image_grain_crops"]
        self._filename = state["_filename"]
        self._pixel_to_nm_scaling = state["_pixel_to_nm_scaling"]
        self._img_path = state["_img_path"]
        self._image = state["_image"]
        self._image_original = state["_image_original"]
        self._full_mask_tensor = state["_full_mask_tensor"]
        self._topostats_version = state["_topostats_version"]
        self._config = state["_config"]

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
            and self.topostats_version == other.topostats_version
            and self.config == other.config
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
        Pixel to nm scaling.

        Returns
        -------
        str
            Image ``pixel_to_nm_scaling``.
        """
        return self._pixel_to_nm_scaling

    @pixel_to_nm_scaling.setter
    def pixel_to_nm_scaling(self, value: str) -> None:
        """
        Pixel to nm scaling.

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
    def full_mask_tensor(self) -> npt.NDArray:
        """
        Getter for the ``full_mask_tensor`` attribute.

        Returns
        -------
        npt.NDArray
            Returns the value of ``full_mask_tensor``.
        """
        return self._full_mask_tensor

    @full_mask_tensor.setter
    def full_mask_tensor(self, value: npt.NDArray) -> None:
        """
        Setter for the ``full_mask_tensor`` attribute.

        Parameters
        ----------
        value : npt.NDArray
            Value to set for ``full_mask_tensor``.
        """
        self._full_mask_tensor = value

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

    @property
    def config(self) -> dict[str, Any]:
        """
        Configuration used when processing the image.

        This includes the method and thresholds for filtering, Gaussian blurring and scar removal.

        Returns
        -------
        dict[str, Any]
            Returns the value of ``filter_config``.
        """
        return self._config

    @config.setter
    def config(self, value: dict[str, Any]) -> None:
        """
        Setter for the ``config`` attribute.

        Parameters
        ----------
        value : dict[str, Any]
            Value to set for ``config``.
        """
        self._config = value

    def topostats_to_dict(self) -> dict[str, str | ImageGrainCrops | npt.NDArray]:
        """
        Convert ``TopoStats`` object to dictionary.

        Returns
        -------
        dict[str, str | ImageGrainCrops | npt.NDArray]
            Dictionary of ``TopoStats`` object.
        """
        return {re.sub(r"^_", "", key): value for key, value in self.__dict__.items()}


@dataclass
class MatchedBranch:
    """
    Class for storing matched branches data and attributes.

    Attributes
    ----------
    ordered_coords : npt.NDArray[np.int32]
        Numpy array of ordered coordinates.
    heights : npt.NDArray[np.number]
        Numpy array of heights.
    distances : npt.NDArray[np.number]
        Numpy array of distances.
    fwhm : dict[str, np.float64 | tuple[np.float64]]
        ???
    angles : np.float64
        Angle between branches ???
    """

    ordered_coords: npt.NDArray[np.int32] = None
    heights: npt.NDArray[np.number] = None
    distances: npt.NDArray[np.number] = None
    fwhm: dict[str, np.float64 | tuple[np.float64]] = None
    angles: np.float64 = None
    # ns-rse 2025-10-07 : Need to add check types of attributes and checks that they are valid

    def __getstate__(self) -> dict[str, Any]:
        """
        Get the state of the class.

        Returns
        -------
        dict[str, Any]
            Dictionary of the current state of the classes attributes.
        """
        return {
            "_ordered_coords": self._ordered_coords,
            "_heights": self._heights,
            "_distances": self._distances,
            "_fwhm": self._fwhm,
            "_angles": self._angles,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Set the state of the class.

        Parameters
        ----------
        state : dict[str, Any]
            A dictionary holding the state of all parameters for the class.
        """
        self._ordered_coords = state["_ordered_coords"]
        self._heights = state["_heights"]
        self._distances = state["_distances"]
        self._fwhm = state["_fwhm"]
        self._angles = state["_angles"]

    def __str__(self) -> str:
        """
        Readable attributes.

        Returns
        -------
        str
            Set of formatted statistics on matched branches.
        """
        return (
            f"ordered_coords : {self.ordered_coords}\n"
            f"heights : {self.heights}\n"
            f"distances : {self.distances}\n"
            f"fwhm : {self.fwhm}\n"
            f"angles : {self.angles}"
        )

    @property
    def ordered_coords(self) -> npt.NDArray[np.int32]:
        """
        Getter for the ``ordered_coords`` attribute.

        Returns
        -------
        npt.NDArray[np.int32]
            Returns the value of ``ordered_coords``.
        """
        return self._ordered_coords

    @ordered_coords.setter
    def ordered_coords(self, value: npt.NDArray[np.int32]) -> None:
        """
        Setter for the ``ordered_coords`` attribute.

        Parameters
        ----------
        value : npt.NDArray[np.int32]
            Value to set for ``ordered_coords``.
        """
        self._ordered_coords = value

    @property
    def heights(self) -> npt.NDArray[np.number]:
        """
        Getter for the ``heights`` attribute.

        Returns
        -------
        npt.NDArray[np.number]
            Returns the value of ``heights``.
        """
        return self._heights

    @heights.setter
    def heights(self, value: npt.NDArray[np.number]) -> None:
        """
        Setter for the ``heights`` attribute.

        Parameters
        ----------
        value : npt.NDArray[np.number]
            Value to set for ``heights``.
        """
        self._heights = value

    @property
    def distances(self) -> npt.NDArray[np.number]:
        """
        Getter for the ``distances`` attribute.

        Returns
        -------
        npt.NDArray[np.number]
            Returns the value of ``distances``.
        """
        return self._distances

    @distances.setter
    def distances(self, value: npt.NDArray[np.number]) -> None:
        """
        Setter for the ``distances`` attribute.

        Parameters
        ----------
        value : npt.NDArray[np.number]
            Value to set for ``distances``.
        """
        self._distances = value

    @property
    def fwhm(self) -> dict[str, np.float64 | tuple[np.float64]]:
        """
        Getter for the ``fwhm`` attribute.

        Returns
        -------
        dict[str, np.float64 | tuple[np.float64]]
            Returns the value of ``fwhm``.
        """
        return self._fwhm

    @fwhm.setter
    def fwhm(self, value: dict[str, np.float64 | tuple[np.float64]]) -> None:
        """
        Setter for the ``fwhm`` attribute.

        Parameters
        ----------
        value : dict[str, np.float64 | tuple[np.float64]]
            Value to set for ``fwhm``.
        """
        self._fwhm = value

    @property
    def angles(self) -> np.float64:
        """
        Getter for the ``angles`` attribute.

        Returns
        -------
        np.float64
            Returns the value of ``angles``.
        """
        return self._angles

    @angles.setter
    def angles(self, value: np.float64) -> None:
        """
        Setter for the ``angles`` attribute.

        Parameters
        ----------
        value : np.float64
            Value to set for ``angles``.
        """
        self._angles = value

    def matched_branch_to_dict(self) -> dict[str, Any]:
        """
        Convert ``MatchedBranch`` to dictionary indexed by attributes.

        Returns
        -------
        dict[str, Any]
            Dictionary indexed by attribute of the grain attributes.
        """
        return {re.sub(r"^_", "", key): value for key, value in self.__dict__.items()}


@dataclass
class Node:
    """
    Class for storing Node data and attributes.

    Attributes
    ----------
    error : bool
        Whether an error occurred calculating statistics for this node.
    pixel_to_nm_scaling : np.float64
        Pixel to nanometre scaling.
    branch_stats : dict[int, MatchedBranch]
        Dictionary of branch statistics.
    unmatched_branch_stats : dict[int, Any]
        Dictionary of unmatched branch statistics.
    node_coords : dict[str, dict[str, npt.NDArray[np.int32]]]
        Nested dictionary of node coordinates
    confidence : np.float64
        Confidence in ???.
    reduced_node_area : ???
        Reduced node area.
    node_area_skeleton : npt.NDArray[np.int32]
        Numpy array of skeleton.
    node_branch_mask : npt.NDArray[np.int32]
        Numpy array of branch mask.
    node_avg_mask : npt.NDArray[np.int32]
        Numpy array of averaged mask.
    """

    error: bool | None = None
    pixel_to_nm_scaling: np.float64 | None = None
    branch_stats: dict[int, MatchedBranch] | None = None
    unmatched_branch_stats: dict | None = None
    node_coords: dict[str, dict[str, npt.NDArray[np.int32]]] | None = None
    confidence: np.float64 | None = None
    reduced_node_area: np.float64 | None = None
    node_area_skeleton: npt.NDArray[np.int32] | None = None
    node_branch_mask: npt.NDArray[np.int32] | None = None
    node_avg_mask: npt.NDArray[np.int32] | None = None

    def __getstate__(self) -> dict[str, Any]:
        """
        Get the state of the class.

        Returns
        -------
        dict[str, Any]
            Dictionary of the current state of the classes attributes.
        """
        return {
            "_error": self._error,
            "_pixel_to_nm_scaling": self._pixel_to_nm_scaling,
            "_branch_stats": self._branch_stats,
            "_unmatched_branch_stats": self._unmatched_branch_stats,
            "_node_coords": self._node_coords,
            "_confidence": self._confidence,
            "_reduced_node_area": self._reduced_node_area,
            "_node_area_skeleton": self._node_area_skeleton,
            "_node_branch_mask": self._node_branch_mask,
            "_node_avg_mask": self._node_avg_mask,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Set the state of the class.

        Parameters
        ----------
        state : dict[str, Any]
            A dictionary holding the state of all parameters for the class.
        """
        self._error = state["_error"]
        self._pixel_to_nm_scaling = state["_pixel_to_nm_scaling"]
        self._branch_stats = state["_branch_stats"]
        self._unmatched_branch_stats = state["_unmatched_branch_stats"]
        self._node_coords = state["_node_coords"]
        self._confidence = state["_confidence"]
        self._reduced_node_area = state["_reduced_node_area"]
        self._node_area_skeleton = state["_node_area_skeleton"]
        self._node_branch_mask = state["_node_branch_mask"]
        self._node_avg_mask = state["_node_avg_mask"]

    def __str__(self) -> str:
        """
        Readable attributes.

        Returns
        -------
        str
            Set of formatted statistics on matched branches.
        """
        return (
            f"branch_stats : {self.branch_stats}\n"
            f"unmatched_branch_stats : {self.unmatched_branch_stats}\n"
            f"distances : {self.confidence}\n"
            f"node_coords : {self.node_coords}\n"
            f"confidence : {self.confidence}"
            f"pixel_to_nm_scaling : {self.pixel_to_nm_scaling}\n"
            f"error : {self.error}"
        )

    @property
    def error(self) -> bool:
        """
        Getter for the ``error`` attribute.

        Returns
        -------
        bool
            Returns the value of ``error``.
        """
        return self._error

    @error.setter
    def error(self, value: bool) -> None:
        """
        Setter for the ``error`` attribute.

        Parameters
        ----------
        value : bool
            Value to set for ``error``.
        """
        self._error = value

    @property
    def pixel_to_nm_scaling(self) -> np.float64:
        """
        Pixel to nm scaling.

        Returns
        -------
        np.float64
            Returns the value of ``pixel_to_nm_scaling``.
        """
        return self._pixel_to_nm_scaling

    @pixel_to_nm_scaling.setter
    def pixel_to_nm_scaling(self, value: np.float64) -> None:
        """
        Pixel to nm scaling.

        Parameters
        ----------
        value : np.float64
            Value to set for ``pixel_to_nm_scaling``.
        """
        self._pixel_to_nm_scaling = value

    @property
    def branch_stats(self) -> dict[int, MatchedBranch]:
        """
        Getter for the ``branch_stats`` attribute.

        Returns
        -------
        dict[int, MatchedBranch]
            Returns the value of ``branch_stats``.
        """
        return self._branch_stats

    @branch_stats.setter
    def branch_stats(self, value: dict[int, MatchedBranch]) -> None:
        """
        Setter for the ``branch_stats`` attribute.

        Parameters
        ----------
        value : dict[int, MatchedBranch]
            Value to set for ``branch_stats``.
        """
        self._branch_stats = value

    @property
    def unmatched_branch_stats(self) -> dict[int, MatchedBranch]:
        """
        Getter for the ``unmatched_branch_stats`` attribute.

        Returns
        -------
        dict[int, MatchedBranch]
            Returns the value of ``unmatched_branch_stats``.
        """
        return self._unmatched_branch_stats

    @unmatched_branch_stats.setter
    def unmatched_branch_stats(self, value: dict[int, MatchedBranch]) -> None:
        """
        Setter for the ``unmatched_branch_stats`` attribute.

        Parameters
        ----------
        value : dict[int, MatchedBranch]
            Value to set for ``unmatched_branch_stats``.
        """
        self._unmatched_branch_stats = value

    @property
    def node_coords(self) -> npt.NDArray[np.int32]:
        """
        Getter for the ``node_coords`` attribute.

        Returns
        -------
        npt.NDArray[np.int32]
            Returns the value of ``node_coords``.
        """
        return self._node_coords

    @node_coords.setter
    def node_coords(self, value: npt.NDArray[np.int32]) -> None:
        """
        Setter for the ``node_coords`` attribute.

        Parameters
        ----------
        value : npt.NDArray[np.int32]
            Value to set for ``node_coords``.
        """
        self._node_coords = value

    @property
    def confidence(self) -> np.float64:
        """
        Getter for the ``confidence`` attribute.

        Returns
        -------
        np.float64
            Returns the value of ``confidence``.
        """
        return self._confidence

    @confidence.setter
    def confidence(self, value: np.float64) -> None:
        """
        Setter for the ``confidence`` attribute.

        Parameters
        ----------
        value : np.float64
            Value to set for ``confidence``.
        """
        self._confidence = value

    @property
    def reduced_node_area(self) -> dict[str, dict[str, npt.NDArray[np.int32]]]:
        """
        Getter for the ``reduced_node_area`` attribute.

        Returns
        -------
        dict[str, dict[str, npt.NDArray[np.int32]]]
            Returns the value of ``reduced_node_area``.
        """
        return self._reduced_node_area

    @reduced_node_area.setter
    def reduced_node_area(self, value: dict[str, dict[str, npt.NDArray[np.int32]]]) -> None:
        """
        Setter for the ``reduced_node_area`` attribute.

        Parameters
        ----------
        value : dict[str, dict[str, npt.NDArray[np.int32]]]
            Value to set for ``reduced_node_area``.
        """
        self._reduced_node_area = value

    @property
    def node_area_skeleton(self) -> type:
        """
        Getter for the ``node_area_skeleton`` attribute.

        Returns
        -------
        type
            Returns the value of ``node_area_skeleton``.
        """
        return self._node_area_skeleton

    @node_area_skeleton.setter
    def node_area_skeleton(self, value: type) -> None:
        """
        Setter for the ``node_area_skeleton`` attribute.

        Parameters
        ----------
        value : type
            Value to set for ``node_area_skeleton``.
        """
        self._node_area_skeleton = value

    @property
    def node_branch_mask(self) -> type:
        """
        Getter for the ``node_branch_mask`` attribute.

        Returns
        -------
        type
            Returns the value of ``node_branch_mask``.
        """
        return self._node_branch_mask

    @node_branch_mask.setter
    def node_branch_mask(self, value: type) -> None:
        """
        Setter for the ``node_branch_mask`` attribute.

        Parameters
        ----------
        value : type
            Value to set for ``node_branch_mask``.
        """
        self._node_branch_mask = value

    @property
    def node_avg_mask(self) -> type:
        """
        Getter for the ``node_avg_mask`` attribute.

        Returns
        -------
        type
            Returns the value of ``node_avg_mask``.
        """
        return self._node_avg_mask

    @node_avg_mask.setter
    def node_avg_mask(self, value: type) -> None:
        """
        Setter for the ``node_avg_mask`` attribute.

        Parameters
        ----------
        value : type
            Value to set for ``node_avg_mask``.
        """
        self._node_avg_mask = value

    def node_to_dict(self) -> dict[str, Any]:
        """
        Convert ``Node`` to dictionary indexed by attributes.

        Returns
        -------
        dict[str, Any]
            Dictionary indexed by attribute of the grain attributes.
        """
        return {re.sub(r"^_", "", key): value for key, value in self.__dict__.items()}


@dataclass
class OrderedTrace:
    """
    Class for Ordered Trace data and attributes.

    tracing_stats : dict | None
        Tracing statistics.
    grain_molstats : Any | None
        Grain molecule statistics.
    ordered_trace_data : dict[int, Molecule]
        Ordered trace data for the grain indexed by molecule number.
    molecules : int
        Number of molecules within the grain.
    writhe : str
        The writhe sign, can be either `+`, `-` or `0` for positive, negative or no writhe.
    pixel_to_nm_scaling: np.float64 | None
        Pixel to nm scaling.
    images: dict[str, npt.NDArray] | None
        Images of ???
    error: bool | None
        Errors encountered?
    """

    ordered_trace_data: dict[int, Molecule] | None = None
    tracing_stats: dict | None = None
    grain_molstats: Any | None = None
    molecules: int | None = None
    writhe: str | None = None
    pixel_to_nm_scaling: np.float64 | None = None
    images: dict[str, npt.NDArray] | None = None
    error: bool | None = None

    def __getstate__(self) -> dict[str, Any]:
        """
        Get the state of the class.

        Returns
        -------
        dict[str, Any]
            Dictionary of the current state of the classes attributes.
        """
        return {
            "_ordered_trace_data": self.ordered_trace_data,
            "_tracing_stats": self.tracing_stats,
            "_grain_molstats": self.grain_molstats,
            "_molecules": self.molecules,
            "_writhe": self.writhe,
            "_pixel_to_nm_scaling": self.pixel_to_nm_scaling,
            "_images": self.images,
            "_error": self.error,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Set the state of the class.

        Parameters
        ----------
        state : dict[str, Any]
            A dictionary holding the state of all parameters for the class.
        """
        self._ordered_trace_data = state["ordered_trace_data"]
        self._tracing_stats = state["tracing_stats"]
        self._grain_molstats = state["grain_molstats"]
        self._molecules = state["molecules"]
        self._writhe = state["writhe"]
        self._pixel_to_nm_scaling = state["pixel_to_nm_scaling"]
        self._images = state["images"]
        self._error = state["error"]

    def __str__(self) -> str:
        """
        Readable attributes.

        Returns
        -------
        str
            Set of formatted statistics on matched branches.
        """
        return (
            f"ordered_trace_data : {self.ordered_trace_data}\n"
            f"tracing_stats : {self.tracing_stats}\n"
            f"grain_molstats : {self.grain_molstats}\n"
            f"pixel_to_nm_scaling : {self.pixel_to_nm_scaling}\n"
            f"images : {self.images}"
            f"error : {self.error}"
        )

    @property
    def ordered_trace_data(self) -> dict[str, dict[str, Any]]:
        """
        Getter for the ``ordered_trace_data`` attribute.

        Returns
        -------
        dict[str, dict[str, Any]]
            Returns the value of ``ordered_trace_data``.
        """
        return self._ordered_trace_data

    @ordered_trace_data.setter
    def ordered_trace_data(self, value: dict[str, dict[str, Any]]) -> None:
        """
        Setter for the ``ordered_trace_data`` attribute.

        Parameters
        ----------
        value : dict[str, dict[str, Any]]
            Value to set for ``ordered_trace_data``.
        """
        self._ordered_trace_data = value

    @property
    def tracing_stats(self) -> pd.DataFrame:
        """
        Getter for the ``tracing_stats`` attribute.

        Returns
        -------
        pd.DataFrame
            Returns the value of ``tracing_stats``.
        """
        return self._tracing_stats

    @tracing_stats.setter
    def tracing_stats(self, value: pd.DataFrame) -> None:
        """
        Setter for the ``tracing_stats`` attribute.

        Parameters
        ----------
        value : pd.DataFrame
            Value to set for ``tracing_stats``.
        """
        self._tracing_stats = value

    @property
    def grain_molstats(self) -> pd.DataFrame:
        """
        Getter for the ``grain_molstats`` attribute.

        Returns
        -------
        pd.DataFrame
            Returns the value of ``grain_molstats``.
        """
        return self._grain_molstats

    @grain_molstats.setter
    def grain_molstats(self, value: pd.DataFrame) -> None:
        """
        Setter for the ``grain_molstats`` attribute.

        Parameters
        ----------
        value : pd.DataFrame
            Value to set for ``grain_molstats``.
        """
        self._grain_molstats = value

    @property
    def molecules(self) -> int:
        """
        Getter for the ``molecules`` attribute.

        Returns
        -------
        int
            Returns the value of ``molecules``.
        """
        return len(self.ordered_trace_data)

    @molecules.setter
    def molecules(self, value: int) -> None:
        """
        Setter for the ``molecules`` attribute.

        Parameters
        ----------
        value : int
            Value to set for ``molecules``.
        """
        self._molecules = value

    @property
    def writhe(self) -> str:
        """
        Getter for the ``writhe`` attribute.

        Returns
        -------
        str
            Returns the value of ``writhe``.
        """
        return self._writhe

    @writhe.setter
    def writhe(self, value: str) -> None:
        """
        Setter for the ``writhe`` attribute.

        Parameters
        ----------
        value : str
            Value to set for ``writhe``.
        """
        self._writhe = value

    @property
    def pixel_to_nm_scaling(self) -> np.float64:
        """
        Pixel to nm scaling.

        Returns
        -------
        np.float64
            Returns the value of ``pixel_to_nm_scaling``.
        """
        return self._pixel_to_nm_scaling

    @pixel_to_nm_scaling.setter
    def pixel_to_nm_scaling(self, value: np.float64) -> None:
        """
        Pixel to nm scaling.

        Parameters
        ----------
        value : np.float64
            Value to set for ``pixel_to_nm_scaling``.
        """
        self._pixel_to_nm_scaling = value

    @property
    def images(self) -> dict[int, npt.NDArray]:
        """
        Getter for the ``images`` attribute.

        Returns
        -------
        dict[int, npt.NDArray]
            Returns the value of ``images``.
        """
        return self._images

    @images.setter
    def images(self, value: dict[int, npt.NDArray]) -> None:
        """
        Setter for the ``images`` attribute.

        Parameters
        ----------
        value : dict[int, npt.NDArray]
            Value to set for ``images``.
        """
        self._images = value

    @property
    def error(self) -> bool:
        """
        Getter for the ``error`` attribute.

        Returns
        -------
        bool
            Returns the value of ``error``.
        """
        return self._error

    @error.setter
    def error(self, value: bool) -> None:
        """
        Setter for the ``error`` attribute.

        Parameters
        ----------
        value : bool
            Value to set for ``error``.
        """
        self._error = value

    def ordered_trace_to_dict(self) -> dict[str, Any]:
        """
        Convert ``OrderedTrace`` to dictionary indexed by attributes.

        Returns
        -------
        dict[str, Any]
            Dictionary indexed by attribute of the grain attributes.
        """
        return {re.sub(r"^_", "", key): value for key, value in self.__dict__.items()}


@dataclass
class Molecule:
    """Class for Molecules identified during ordered tracing."""

    circular: str | None = None
    topology: str | None = None
    topology_flip: Any | None = None
    ordered_coords: npt.NDArray | None = None
    heights: npt.NDArray | None = None
    distances: npt.NDArray | None = None

    def __getstate__(self) -> dict[str, Any]:
        """
        Get the state of the class.

        Returns
        -------
        dict[str, Any]
            Dictionary of the current state of the classes attributes.
        """
        return {
            "_circular": self._circular,
            "_topology": self._topology,
            "_topology_flip": self._topology_flip,
            "_ordered_coords": self._ordered_coords,
            "_heights": self._heights,
            "_distances": self._distances,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Set the state of the class.

        Parameters
        ----------
        state : dict[str, Any]
            A dictionary holding the state of all parameters for the class.
        """
        self._circular = state["_circular"]
        self._topology = state["_topology"]
        self._topology_flip = state["_topology_flip"]
        self._ordered_coords = state["_ordered_coords"]
        self._heights = state["_heights"]
        self._distances = state["_distances"]

    @property
    def circular(self) -> bool:
        """
        Getter for the ``circular`` attribute.

        Returns
        -------
        bool
            Returns the value of ``circular``.
        """
        return self._circular

    @circular.setter
    def circular(self, value: bool) -> None:
        """
        Setter for the ``circular`` attribute.

        Parameters
        ----------
        value : bool
            Value to set for ``circular``.
        """
        self._circular = value

    @property
    def topology(self) -> str:
        """
        Getter for the ``topology`` attribute.

        Returns
        -------
        str
            Returns the value of ``topology``.
        """
        return self._topology

    @topology.setter
    def topology(self, value: str) -> None:
        """
        Setter for the ``topology`` attribute.

        Parameters
        ----------
        value : str
            Value to set for ``topology``.
        """
        self._topology = value

    @property
    def topology_flip(self) -> Any:
        """
        Getter for the ``topology_flip`` attribute.

        Returns
        -------
        Any
            Returns the value of ``topology_flip``.
        """
        return self._topology_flip

    @topology_flip.setter
    def topology_flip(self, value: Any) -> None:
        """
        Setter for the ``topology_flip`` attribute.

        Parameters
        ----------
        value : Any
            Value to set for ``topology_flip``.
        """
        self._topology_flip = value

    @property
    def ordered_coords(self) -> npt.NDArray:
        """
        Getter for the ``ordered_coords`` attribute.

        Returns
        -------
        npt.NDArray
            Returns the value of ``ordered_coords``.
        """
        return self._ordered_coords

    @ordered_coords.setter
    def ordered_coords(self, value: npt.NDArray) -> None:
        """
        Setter for the ``ordered_coords`` attribute.

        Parameters
        ----------
        value : npt.NDArray
            Value to set for ``ordered_coords``.
        """
        self._ordered_coords = value

    @property
    def heights(self) -> npt.NDArray:
        """
        Getter for the ``heights`` attribute.

        Returns
        -------
        npt.NDArray
            Returns the value of ``heights``.
        """
        return self._heights

    @heights.setter
    def heights(self, value: npt.NDArray) -> None:
        """
        Setter for the ``heights`` attribute.

        Parameters
        ----------
        value : npt.NDArray
            Value to set for ``heights``.
        """
        self._heights = value

    @property
    def distances(self) -> npt.NDArray:
        """
        Getter for the ``distances`` attribute.

        Returns
        -------
        npt.NDArray
            Returns the value of ``distances``.
        """
        return self._distances

    @distances.setter
    def distances(self, value: npt.NDArray) -> None:
        """
        Setter for the ``distances`` attribute.

        Parameters
        ----------
        value : npt.NDArray
            Value to set for ``distances``.
        """
        self._distances = value

    def molecule_to_dict(self) -> dict[str, Any]:
        """
        Convert ``Molecule`` to dictionary indexed by attributes.

        Returns
        -------
        dict[str, Any]
            Dictionary indexed by attribute of the grain attributes.
        """
        return {re.sub(r"^_", "", key): value for key, value in self.__dict__.items()}
