"""Define custom classes for TopoStats."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
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
    """

    images: dict[str, npt.NDArray] | None = None
    grain_endpoints: int | None = None
    grain_junctions: int | None = None
    total_branch_length: float | None = None
    grain_width_mean: float | None = None

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

    def disordered_trace_to_dict(self) -> dict[str, Any]:
        """
        Convert DisorderedTrace to dictionary indexed by attributes.

        Returns
        -------
        dict[str, Any]
            Dictionary indexed by attribute of the grain attributes.
        """
        return dict(self.__dict__)


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

    grain_crops: dict[int, GrainCrop] | None = None
    filename: str | None = None
    pixel_to_nm_scaling: float | None = None
    img_path: Path | str | None = None
    image: npt.NDArray | None = None
    image_original: npt.NDArray | None = None
    full_mask_tensor: npt.NDArray | None = None
    topostats_version: str | None = None
    config: dict[str, Any] | None = None

    def topostats_to_dict(self) -> dict[str, str | npt.NDArray | dict[str | int, GrainCrop | Any]]:
        """
        Convert ``TopoStats`` object to dictionary.

        Returns
        -------
        dict[str, str | npt.NDArray | dict[str | int, GrainCrop | Any]]
            Dictionary of ``TopoStats`` object.
        """
        return dict(self.__dict__)

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

    def matched_branch_to_dict(self) -> dict[str, Any]:
        """
        Convert ``MatchedBranch`` to dictionary indexed by attributes.

        Returns
        -------
        dict[str, Any]
            Dictionary indexed by attribute of the grain attributes.
        """
        return dict(self.__dict__)


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

    def unmatched_branch_to_dict(self) -> dict[str, Any]:
        """
        Convert ``MatchedBranch`` to dictionary indexed by attributes.

        Returns
        -------
        dict[str, Any]
            Dictionary indexed by attribute of the grain attributes.
        """
        return dict(self.__dict__)


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
    pixel_to_nm_scaling: float | None = None
    branch_stats: dict[int, MatchedBranch] | None = None
    unmatched_branch_stats: dict[int, UnMatchedBranch] | None = None
    node_coords: npt.NDArray[np.int32] | None = None
    confidence: float | None = None
    reduced_node_area: float | None = None
    node_area_skeleton: npt.NDArray[np.int32] | None = None
    node_branch_mask: npt.NDArray[np.int32] | None = None
    node_avg_mask: npt.NDArray[np.int32] | None = None

    def node_to_dict(self) -> dict[str, Any]:
        """
        Convert ``Node`` to dictionary indexed by attributes.

        Returns
        -------
        dict[str, Any]
            Dictionary indexed by attribute of the grain attributes.
        """
        return dict(self.__dict__)


@dataclass(
    repr=True, eq=True, config=ConfigDict(arbitrary_types_allowed=True, validate_assignment=True), validate_on_init=True
)
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
    pixel_to_nm_scaling: float | None = None
    images: dict[str, npt.NDArray] | None = None
    error: bool | None = None

    def ordered_trace_to_dict(self) -> dict[str, Any]:
        """
        Convert ``OrderedTrace`` to dictionary indexed by attributes.

        Returns
        -------
        dict[str, Any]
            Dictionary indexed by attribute of the grain attributes.
        """
        return dict(self.__dict__)


@dataclass(
    repr=True, eq=True, config=ConfigDict(arbitrary_types_allowed=True, validate_assignment=True), validate_on_init=True
)
class Molecule:
    """Class for Molecules identified during ordered tracing."""

    circular: str | bool | None = None
    topology: str | None = None
    topology_flip: Any | None = None
    ordered_coords: npt.NDArray | None = None
    heights: npt.NDArray | None = None
    distances: npt.NDArray | None = None

    def molecule_to_dict(self) -> dict[str, Any]:
        """
        Convert ``Molecule`` to dictionary indexed by attributes.

        Returns
        -------
        dict[str, Any]
            Dictionary indexed by attribute of the grain attributes.
        """
        return dict(self.__dict__)


def convert_to_dict(something: Any) -> dict[str, Any]:
    """
    Convert an a class to a dictionary representation of itself.

    Parameters
    ----------
    something : Any
        An object to be converted to a dictionary.

    Returns
    -------
    dict[str, Any]
        Dictionary representation of the object.
    """
    return dict(something.__dict__)
