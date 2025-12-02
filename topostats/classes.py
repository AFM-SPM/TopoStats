"""Define custom classes for TopoStats."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from topostats.logs.logs import LOGGER_NAME
from topostats.utils import update_background_class

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
    threshold : str
        Direction of the molecule from the threshold (above / below).
    grain_number : int
        Index of the grain.
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
        threshold: str | None = None,
        grain_number: int | None = None,
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
        threshold : str
            Direction of the molecule from the threshold (above / below).
        grain_number : int
            Index of the grain.
        skeleton : npt.NDArray[np.bool_]
            3-D Numpy tensor of the skeletonised mask.
        height_profiles : dict[int, [int, npt.NDArray[np.float32]]] | None
            3-D Numpy tensor of the height profiles.
        stats : dict[str, int | float] | None
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
        self.threshold: str | None = threshold
        self.grain_number: int | None = grain_number
        self.height_profiles = height_profiles
        self.stats = {} if stats is None else stats
        self.skeleton: npt.NDArray[np.bool_] | None = skeleton
        self.disordered_trace: DisorderedTrace | None = disordered_trace
        self.nodes: dict[int, Node] | None = nodes
        self.ordered_trace: OrderedTrace | None = ordered_trace
        self.threshold_method: str | None = threshold_method

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

    def __str__(self) -> str:
        """
        Representation function for useful statistics for the class.

        Returns
        -------
        str
            Set of formatted statistics on matched branches.
        """
        return (
            f"filename : {self.filename}\n"
            f"image shape (px) : {self.image.shape}\n"
            f"skeleton shape (px) : {self.skeleton.shape}\n"
            f"mask shape (px) : {self.mask.shape}\n"
            f"padding : {self.padding}\n"
            f"thresholds : {self.thresholds}\n"
            f"threshold method : {self.threshold_method}\n"
            f"bounding box coords : {self.bbox}\n"
            f"pixel to nm scaling : {self.pixel_to_nm_scaling}\n"
            f"number of nodes : {len(self.nodes)}"
        )

    # Unfinished - currently only uses Node class names rather than including their content
    def stats_to_df(self) -> pd.DataFrame:
        """
        Convert class attributes to a pandas dataframe.

        Returns
        -------
        pd.DataFrame
            Dataframe of the classes attributes and their data.
        """
        node_names = {k: v.__class__.__name__ for k, v in self.nodes.items()}
        data = {
            "padding": self.padding,
            "image": self.image,
            "mask": self.mask,
            "bbox": self.bbox,
            "pixel_to_nm_scaling": self.pixel_to_nm_scaling,
            "thresholds": self.thresholds,
            "threshold_method": self.threshold_method,
            "filename": self.filename,
            "height_profiles": self.height_profiles,
            "stats": self.stats,
            "skeleton": self.skeleton,
            "disordered_trace": self.disordered_trace.__class__.__name__,
            "nodes": node_names,
            "ordered_trace": self.ordered_trace.__class__.__name__,
        }
        return pd.DataFrame([data])

    @property
    def image(self) -> npt.NDArray[float]:
        """
        Getter for the ``image`` attribute.

        Returns
        -------
        npt.NDArray
            Numpy array of the image.
        """
        return self._image

    @image.setter
    def image(self, value: npt.NDArray[float]):
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


@dataclass(
    repr=True, eq=True, config=ConfigDict(arbitrary_types_allowed=True, validate_assignment=True), validate_on_init=True
)
class DisorderedTrace:
    """
    Dataclass for storing the disordered tracing data.

    Attributes
    ----------
    images : dict[str: npt.NDArray]
        Dictionary of images generated during disordered tracing, should include ''pruned_skeleton''.
    grain_endpoints : int
        Number of Grain endpoints.
    grain_junctions : int
        Number of Grain junctions.
    total_branch_length : float
        Total branch length in nanometres.
    grain_width_mean : float
        Mean grain width in nanometres.
    stats_dict : dict[int, Any]
        Dictionary of stats for each branch of a grain.
    """

    images: dict[str, npt.NDArray] | None = None
    grain_endpoints: int | None = None
    grain_junctions: int | None = None
    total_branch_length: float | None = None
    grain_width_mean: float | None = None
    stats_dict: dict[int, Any] | None = None

    def __str__(self) -> str:
        """
        Representation function for useful statistics for the class.

        Returns
        -------
        str
            Set of formatted statistics on matched branches.
        """
        image_gens = ", ".join(self.images.keys())
        return (
            f"generated images : {image_gens}\n"
            f"grain endpoints : {self.grain_endpoints}\n"
            f"grain junctions : {self.grain_junctions}\n"
            f"total branch length (nm) : {self.total_branch_length}\n"
            f"mean grain width (nm) : {self.grain_width_mean}\n"
            f"number of branches : {len(self.stats_dict)}"
        )

    def stats_to_df(self) -> pd.DataFrame:
        """
        Convert class attributes to a pandas dataframe.

        Returns
        -------
        pd.DataFrame
            Dataframe of the classes attributes and their data.
        """
        data = {
            "images": self.images,
            "grain_endpoints": self.grain_endpoints,
            "grain_junctions": self.grain_junctions,
            "total_branch_length": self.total_branch_length,
            "grain_width_mean": self.grain_width_mean,
        }
        return pd.DataFrame([data])


@dataclass(
    repr=True, eq=True, config=ConfigDict(arbitrary_types_allowed=True, validate_assignment=True), validate_on_init=True
)
class TopoStats:
    """
    Class for storing TopoStats objects.

    Attributes
    ----------
    grain_crops : dict[int, GrainCrop] | None
        Dictionary of ``GrainCrop`` objects.
    filename : str | None
        Filename.
    image_name : str | None
        Filename without its extension.
    pixel_to_nm_scaling : str | None
        Pixel to nanometre scaling.
    img_path : str | None
        Original path to image.
    image : npt.NDArray | None
        Flattened image (post ``Filter()``).
    image_original : npt.NDArray | None
        Original image.
    image_stats : dict[str, Any] | None
        Dictionary of various image statistics, e.g. size in m and px.
    full_mask_tensor : npt.NDArray
        Tensor mask for the full image.
    topostats_version : str | None
        TopoStats version.
    config : dict[str, Any] | None
        Configuration used when processing the grain.
    basename : str
        Basename of image locations.
    """

    grain_crops: dict[int, GrainCrop] | None = None
    filename: str | None = None
    image_name: str | None = None
    pixel_to_nm_scaling: float | None = None
    img_path: Path | str | None = None
    image: npt.NDArray | None = None
    image_original: npt.NDArray | None = None
    image_stats: dict[str, Any] | None = None
    full_mask_tensor: npt.NDArray | None = None
    topostats_version: str | None = None
    config: dict[str, Any] | None = None
    basename: str | None = None

    def __str__(self) -> str:
        """
        Representation function for useful statistics for the class.

        Returns
        -------
        str
            Set of formatted and user-friendly statistics.
        """
        return (
            f"number of grain crops : {len(self.grain_crops)}\n"
            f"filename : {self.filename}\n"
            f"pixel to nm scaling : {self.pixel_to_nm_scaling}\n"
            f"image shape (px) : {self.image.shape}\n"
            f"image path : {self.img_path}\n"
            f"TopoStats version : {self.topostats_version}\n"
            f"Basename : {self.basename}"
        )

    def __eq__(self, other: object) -> bool:
        """
        Check if two ``TopoStats`` objects are equal.

        Parameters
        ----------
        other : object
            Other ``TopoStats`` object to compare to.

        Returns
        -------
        bool
            ``True`` if the objects are equal, ``False`` otherwise.
        """
        if not isinstance(other, TopoStats):
            return False
        return (
            self.grain_crops == other.grain_crops
            and self.filename == other.filename
            and self.pixel_to_nm_scaling == other.pixel_to_nm_scaling
            and self.img_path == other.img_path
            and self.topostats_version == other.topostats_version
            and self.config == other.config
            and np.array_equal(self.image, other.image)
            and np.array_equal(self.image_original, other.image_original)
            and np.array_equal(self.full_mask_tensor, other.full_mask_tensor)
        )

    # Unfinished - currently only uses GrainCrop class names rather than including their content
    def stats_to_df(self) -> pd.DataFrame:
        """
        Convert class attributes to a pandas dataframe.

        Returns
        -------
        pd.DataFrame
            Dataframe of the classes attributes and their data.
        """
        graincrop_names = {k: v.__class__.__name__ for k, v in self.grain_crops.items()}
        data = {
            "grain_crops": graincrop_names,
            "filename": self.filename,
            "pixel_to_nm_scaling": self.pixel_to_nm_scaling,
            "img_path": self.img_path,
            "image": self.image,
            "image_original": self.image_original,
            "full_mask_tensor": self.full_mask_tensor,
            "topostats_version": self.topostats_version,
            "config": json.dumps(self.config, indent=2, sort_keys=True),
            "basename": self.basename,
        }
        return pd.DataFrame([data])


@dataclass(
    repr=True, eq=True, config=ConfigDict(arbitrary_types_allowed=True, validate_assignment=True), validate_on_init=True
)
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
    fwhm : float
        Full-width half maximum.
    fwhm_half_maxs : list[float]
        Half-maximums from a matched branch.
    fwhm_peaks : list[int | float]
        Peaks from a matched branch.
    angles : float
        Angle between branches.
    """

    ordered_coords: npt.NDArray[np.int32] | None = None
    heights: npt.NDArray[np.number] | None = None
    distances: npt.NDArray[np.number] | None = None
    fwhm: float | None = None
    fwhm_half_maxs: list[float] | None = None
    fwhm_peaks: list[float] | None = None
    angles: float | list[float] | None = None

    def __str__(self) -> str:
        """
        Representation function for useful statistics for the class.

        Returns
        -------
        str
            Set of formatted and user-friendly statistics.
        """
        return (
            f"number of coords : {self.ordered_coords.shape[0]}\n"
            f"full width half maximum : {self.fwhm}\n"
            f"full width half maximum maximums : {self.fwhm_half_maxs}\n"
            f"full width half maximum peaks : {self.fwhm_peaks}\n"
            f"angles : {self.angles}"
        )

    def stats_to_df(self) -> pd.DataFrame:
        """
        Convert class attributes to a pandas dataframe.

        Returns
        -------
        pd.DataFrame
            Dataframe of the classes attributes and their data.
        """
        data = {
            "ordered_coords": self.ordered_coords,
            "heights": self.heights,
            "distances": self.distances,
            "fwhm": self.fwhm,
            "fwhm_half_maxs": self.fwhm_half_maxs,
            "fwhm_peaks": self.fwhm_peaks,
            "angles": self.angles,
        }
        return pd.DataFrame([data])


@dataclass(
    repr=True, eq=True, config=ConfigDict(arbitrary_types_allowed=True, validate_assignment=True), validate_on_init=True
)
class UnMatchedBranch:
    """
    Class for storing matched branches data and attributes.

    Attributes
    ----------
    angles : float
        Angle between branches.
    """

    angles: float | list[float] | None = None

    def __str__(self) -> str:
        """
        Representation function for useful statistics for the class.

        Returns
        -------
        str
            Set of formatted and user-friendly statistics.
        """
        return f"angles : {self.angles}"

    def stats_to_df(self) -> pd.DataFrame:
        """
        Convert class attributes to a pandas dataframe.

        Returns
        -------
        pd.DataFrame
            Dataframe of the classes attributes and their data.
        """
        data = {"angles": self.angles}
        return pd.DataFrame([data])


@dataclass(
    repr=True, eq=True, config=ConfigDict(arbitrary_types_allowed=True, validate_assignment=True), validate_on_init=True
)
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
    unmatched_branch_stats : dict[int, UnMatchedBranch]
        Dictionary of unmatched branch statistics.
    node_coords : npt.NDArray[np.int32]
        Numpy array of node coordinates.
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
    pixel_to_nm_scaling: float | None = None
    branch_stats: dict[int, MatchedBranch] | None = None
    unmatched_branch_stats: dict[int, UnMatchedBranch] | None = None
    node_coords: npt.NDArray[np.int32] | None = None
    confidence: float | None = None
    reduced_node_area: float | None = None
    node_area_skeleton: npt.NDArray[np.int32] | None = None
    node_branch_mask: npt.NDArray[np.int32] | None = None
    node_avg_mask: npt.NDArray[np.int32] | None = None

    def __str__(self) -> str:
        """
        Representation function for useful statistics for the class.

        Returns
        -------
        str
            Set of formatted and user-friendly statistics.
        """
        return (
            f"error : {self.error}\n"
            f"pixel to nm scaling (nm) : {self.pixel_to_nm_scaling}\n"
            f"number of matched branches : {len(self.branch_stats)}\n"
            f"number of unmatched branches : {len(self.unmatched_branch_stats)}\n"
            f"number of coords : {self.node_coords.shape[0]}\n"
            f"confidence : {self.confidence}\n"
            f"reduced node area : {self.reduced_node_area}"
        )

    def stats_to_df(self) -> pd.DataFrame:
        """
        Convert class attributes to a pandas dataframe.

        Returns
        -------
        pd.DataFrame
            Dataframe of the classes attributes and their data.
        """
        data = {
            "error": self.error,
            "pixel_to_nm_scaling": self.pixel_to_nm_scaling,
            "branch_stats": self.branch_stats,
            "unmatched_branch_stats": self.unmatched_branch_stats,
            "node_coords": self.node_coords,
            "confidence": self.confidence,
            "reduced_node_area": self.reduced_node_area,
            "node_area_skeleton": self.node_area_skeleton,
            "node_branch_mask": self.node_branch_mask,
            "node_avg_mask": self.node_avg_mask,
        }
        return pd.DataFrame([data])


@dataclass(
    repr=True, eq=True, config=ConfigDict(arbitrary_types_allowed=True, validate_assignment=True), validate_on_init=True
)
class OrderedTrace:
    """
    Class for Ordered Trace data and attributes.

    molecule_data : dict[int, Molecule]
        Dictionary of ordered trace data for individual molecules within the grain indexed by molecule number.
    tracing_stats : dict | None
        Tracing statistics.
    grain_molstats : Any | None
        Grain molecule statistics.
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

    molecule_data: dict[int, Molecule] | None = None
    tracing_stats: dict | None = None
    grain_molstats: Any | None = None
    molecules: int | None = None
    writhe: str | None = None
    pixel_to_nm_scaling: float | None = None
    images: dict[str, npt.NDArray] | None = None
    error: bool | None = None

    def __str__(self) -> str:
        """
        Representation function for useful statistics for the class.

        Returns
        -------
        str
            Set of formatted and user-friendly statistics.
        """
        writhe = {"+": "positive", "-": "negative", "0": "no writhe"}.get(self.writhe)
        return (
            f"number of molecules : {self.molecules}\n"
            f"number of images : {len(self.images)}\n"
            f"writhe : {writhe}\n"
            f"pixel to nm scaling : {self.pixel_to_nm_scaling}\n"
            f"error : {self.error}"
        )

    # Unfinished - currently only uses Molecule class names rather than including their content
    def stats_to_df(self) -> pd.DataFrame:
        """
        Convert class attributes to a pandas dataframe.

        Returns
        -------
        pd.DataFrame
            Dataframe of the classes attributes and their data.
        """
        molecule_names = {k: v.__class__.__name__ for k, v in self.molecule_data.items()}
        data = {
            "molecule_data": molecule_names,
            "tracing_stats": self.tracing_stats,
            "grain_molstats": self.grain_molstats,
            "molecules": self.molecules,
            "writhe": self.writhe,
            "pixel_to_nm_scaling": self.pixel_to_nm_scaling,
            "images": self.images,
            "error": self.error,
        }
        return pd.DataFrame([data])


@dataclass(
    repr=True, eq=True, config=ConfigDict(arbitrary_types_allowed=True, validate_assignment=True), validate_on_init=True
)
class Molecule:
    """
    Class for Molecules identified during ordered tracing.

    threshold : str
        Direction from threshold of molecule (above / below)
    molecule_number : int
        Index of the molecule (per grain)
    circular : str, bool, optional
        Whether the molecule is circular or linear.
    processing : str
        Which processing type was used, topostats or nodestats.
    topology : str, optional
        Unknown?
    topology_flip : Any, optional
        Unknown?
    ordered_coords : npt.NDArray, optional
        Ordered coordinates for the molecule.
    splined_coords : npt.NDArray, optional
        Smoothed (aka splined) coordinates for the molecule.
    contour_length : float
        Length of the molecule.
    end_to_end_distance : float
        Distance between ends of molecule. Will be ``0.0`` for circular molecules which don't have ends.
    heights : npt.NDArray
        Height along molecule.
    distances : npt.NDArray
        Distance between points on the molecule.
    curvature_stats : npt.NDArray, optional
        Angle changes along molecule. NB - These are always positive due to use of ``np.abs()`` during calculation.
    bbox : tuple[int, int, int, int], optional
        Bounding box.
    """

    threshold: str | None = None
    molecule_number: int | None = None
    circular: str | bool | None = None
    processing: str | None = None
    topology: str | None = None
    topology_flip: Any | None = None
    ordered_coords: npt.NDArray | None = None
    splined_coords: npt.NDArray | None = None
    contour_length: float | None = None
    end_to_end_distance: float | None = None
    heights: npt.NDArray | None = None
    distances: npt.NDArray | None = None
    curvature_stats: npt.NDArray | None = None
    bbox: tuple[int, int, int, int] | None = None

    def __str__(self) -> str:
        """
        Representation function for useful statistics for the class.

        Returns
        -------
        str
            Set of formatted and user-friendly statistics.
        """
        return (
            f"circular : {self.circular}\n"
            f"topology : {self.topology}\n"
            f"topology flip : {self.topology_flip}\n"
            f"number of ordered coords : {self.ordered_coords.shape}\n"
            f"number of spline coords : {self.splined_coords}\n"
            f"contour length : {self.contour_length}\n"
            f"end to end distance : {self.end_to_end_distance}\n"
            f"bounding box coords : {self.bbox}"
        )

    def stats_to_df(self) -> pd.DataFrame:
        """
        Convert class attributes to a pandas dataframe.

        Returns
        -------
        pd.DataFrame
            Dataframe of the classes attributes and their data.
        """
        data = {
            "circular": self.circular,
            "topology": self.topology,
            "topology_flip": self.topology_flip,
            "ordered_coords": self.ordered_coords,
            "splined_coords": self.splined_coords,
            "contour_length": self.contour_length,
            "end_to_end_distance": self.end_to_end_distance,
            "heights": self.heights,
            "distances": self.distances,
            "bbox": self.bbox,
        }
        return pd.DataFrame([data])


def convert_to_dict(to_convert: Any) -> Any:
    """
    Convert a class to a dictionary representation of itself.

    Parameters
    ----------
    to_convert : Any
        An object to be converted to a dictionary / dictionary item.

    Returns
    -------
    Any
        Input parameter as a dictionary / dictionary item.
    """
    if isinstance(to_convert, (int | float | str | Path | bool | np.ndarray | tuple | None)):
        return to_convert
    if isinstance(to_convert, list):
        return [convert_to_dict(item) for item in to_convert]
    if isinstance(to_convert, dict):
        return {key: convert_to_dict(value) for key, value in to_convert.items()}
    if hasattr(to_convert, "__dict__"):
        return {re.sub(r"^_", "", key): convert_to_dict(value) for key, value in to_convert.__dict__.items()}

    return to_convert


def prepare_data_for_df(class_object: object, mapping: dict[str, list[str]]) -> list[dict[str, Any]]:
    """
    Gather required data for creating pd.DataFrames for CSVs.

    A class object is given, and the values under the class name in `mapping` are cycled through.
    Basic data types are directly added to the resultant data structure, and attributes that're classes
    will re-call the method with that class as the `class_object`, returning data to be added to the final structure.

    If the attribute is a dictionary, the data types inside the dictionary are checked; if they're
    class objects this method will be re-called for each object inside the dictionary, and again the result
    will be added to the final structure (the data already there will also be duplicated so it exists on every row).
    If the dictionary only contains basic data types, this data will be moved up one layer and directly
    added to the final structure.

    Parameters
    ----------
    class_object : object
        Class object to be iterated through to find required data.
    mapping : dict[str, list[str]]
        Dictionary of required data for the output, with class names as keys and lists containing strings of
        required attributes from each class.

    Returns
    -------
    list[dict[str, Any]]
        A list of dictionaries containing required data, ready to be converted into a pd.DataFrame.
    """
    class_name = class_object.__class__.__name__
    target_attributes = mapping.get(class_name, [])
    base_row = {}
    child_row_groups = []

    for attribute_name in target_attributes:
        if not hasattr(class_object, attribute_name):
            continue

        value = getattr(class_object, attribute_name)

        # Basic attr type
        if isinstance(value, (int, float, str, bool, type(None))):
            base_row[attribute_name] = value
            continue
        # Attr is a single class object
        if value.__class__.__name__ in mapping:
            child_rows = prepare_data_for_df(value, mapping)
            if len(child_rows) == 1:
                base_row.update(child_rows[0])
            else:
                child_row_groups.append(child_rows)
            continue
        # Attr is a dict
        if isinstance(value, dict):
            # Attr is a dict of dicts
            if all(isinstance(v, dict) for v in value.values()):
                nested_rows = []
                for class_no, subdict in value.items():
                    if not isinstance(subdict, dict):
                        continue
                    for subgrain_no, stat_dict in subdict.items():
                        if not isinstance(stat_dict, dict):
                            continue
                        if not all(isinstance(x, (int, float, str, bool, type(None))) for x in stat_dict.values()):
                            continue

                        row = {"class_number": class_no, "subgrain_number": subgrain_no, **stat_dict}
                        nested_rows.append(row)
                if nested_rows:
                    child_row_groups.append(nested_rows)
                    continue

                for _, v in value.items():
                    for statkey, statval in v.items():
                        base_row[statkey] = statval
                continue
            
            expanded = []
            for subkey, subval in value.items():
                # Attr is a dict of classes
                if subval.__class__.__name__ in mapping:
                    expanded.extend(prepare_data_for_df(subval, mapping))
                # Attr is a dict of basic data types
                else:
                    base_row[subkey] = subval

            if len(expanded) == 1:
                base_row.update(expanded[0])
            else:
                if len(expanded) != 0:
                    child_row_groups.append(expanded)
            continue

    if not child_row_groups:
        return [base_row]
    final_rows = []
    for group in child_row_groups:
        for child_row in group:
            combined = {**base_row, **child_row}
            final_rows.append(combined)
    return final_rows
