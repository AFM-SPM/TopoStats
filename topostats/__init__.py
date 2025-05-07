"""Topostats."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import numpy.typing as npt
import snoop
from matplotlib import colormaps

from .grains import ImageGrainCrops
from .logs.logs import setup_logger
from .theme import Colormap

# Disable TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

LOGGER = setup_logger()

__version__ = version("topostats")
__release__ = ".".join(__version__.split(".")[:-2])

colormaps.register(cmap=Colormap("nanoscope").get_cmap())
colormaps.register(cmap=Colormap("gwyddion").get_cmap())

# Disable snoop
snoop.install(enabled=False)

# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments


@dataclass
class TopoStats:
    """
    Class for storing TopoStats objects.

    Parameters
    ----------
    image_grain_crops : ImageGrainCrops | None
        ImageGrainCrops of processed image.
    filename : str | None
        Filename.
    pixel_to_nm_scaling : str | None
        Pixel to nanometre scaling.
    img_path : str | None
        Original path to image, if not specified defaults to current working directory.
    image : npt.NDArray | None
        Flattened image (post ''Filter()'').
    image_original : npt.NDArray | None
        Original image.
    topostats_version : str | None
        TopoStats version.
    """

    def __init__(
        self,
        image_grain_crops: ImageGrainCrops | None = None,
        filename: str | None = None,
        pixel_to_nm_scaling: str | None = None,
        img_path: Path | str | None = None,
        image: npt.NDArray | None = None,
        image_original: npt.NDArray | None = None,
        topostats_version: str | None = None,
    ):
        """
        Initialise the class.

        Parameters
        ----------
        image_grain_crops : ImageGrainCrops | None
            ImageGrainCrops of processed image.
        filename : str | None
            Filename.
        pixel_to_nm_scaling : str | None
            Pixel to nanometre scaling.
        img_path : str | None
            Original path to image, if not specified defaults to current working directory.
        image : npt.NDArray | None
            Flattened image (post ''Filter()'').
        image_original : npt.NDArray | None
            Original image.
        topostats_version : str | None
            TopoStats version.
        """
        self.image_grain_crops = image_grain_crops
        self.filename = filename
        self.pixel_to_nm_scaling = pixel_to_nm_scaling
        self.img_path = Path.cwd() if img_path is None else Path(img_path)
        self.image = image
        self.image_original = image_original
        self.topostats_version = topostats_version

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
        Setter for the ''image_grain_crops'' attribute.

        Parameters
        ----------
        value : ImageGrainCrops
            Image Grain Crops for the image.
        """
        self._image_grain_crops = value

    @property
    def filename(self) -> str:
        """
        Getter for the ''filename'' attribute.

        Returns
        -------
        str
            Image filename.
        """
        return self._filename

    @filename.setter
    def filename(self, value: str) -> None:
        """
        Setter for the ''filename'' attribute.

        Parameters
        ----------
        value : str
            Filename for the image.
        """
        self._filename = value

    @property
    def pixel_to_nm_scaling(self) -> str:
        """
        Getter for the ''pixel_to_nm_scaling'' attribute.

        Returns
        -------
        str
            Image ''pixel_to_nm_scaling''.
        """
        return self._pixel_to_nm_scaling

    @pixel_to_nm_scaling.setter
    def pixel_to_nm_scaling(self, value: str) -> None:
        """
        Setter for the ''pixel_to_nm_scaling'' attribute.

        Parameters
        ----------
        value : str
            Pixel to nanometre scaling for the image.
        """
        self._pixel_to_nm_scaling = value

    @property
    def img_path(self) -> Path:
        """
        Getter for the ''img_path'' attribute.

        Returns
        -------
        Path
            Path to original image on disk.
        """
        return self._img_path

    @img_path.setter
    def img_path(self, value: str | Path) -> None:
        """
        Setter for the ''img_path'' attribute.

        Parameters
        ----------
        value : str | Path
            Image Path for the image.
        """
        print(f"\n{value=}\n")
        self._img_path = Path(value)

    @property
    def image(self) -> str:
        """
        Getter for the ''image'' attribute, post filtering.

        Returns
        -------
        str
            Image image.
        """
        return self._image

    @image.setter
    def image(self, value: str) -> None:
        """
        Setter for the ''image'' attribute.

        Parameters
        ----------
        value : str
            Filtered image.
        """
        self._image = value

    @property
    def image_original(self) -> str:
        """
        Getter for the ''image_original'' attribute.

        Returns
        -------
        str
            Original image.
        """
        return self._image_original

    @image_original.setter
    def image_original(self, value: str) -> None:
        """
        Setter for the ''image_original'' attribute.

        Parameters
        ----------
        value : str
            Original image.
        """
        self._image_original = value

    @property
    def topostats_version(self) -> str:
        """
        Getter for the ''topostats_version'' attribute, post filtering.

        Returns
        -------
        str
            Version of TopoStats the class was created with.
        """
        return self._topostats_version

    @topostats_version.setter
    def topostats_version(self, value: str) -> None:
        """
        Setter for the ''topostats_version'' attribute.

        Parameters
        ----------
        value : str
            Topostats version.
        """
        self._topostats_version = value

    def topostats_to_dict(self) -> dict[str, str | ImageGrainCrops | npt.NDArray]:
        """
        Convert ''TopoStats'' object to dictionary.

        Returns
        -------
        dict[str, str | ImageGrainCrops | npt.NDArray]
            Dictionary of ''TopoStats'' object.
        """
        return {re.sub(r"^_", "", key): value for key, value in self.__dict__.items()}
