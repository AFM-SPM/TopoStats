"""Functions for reading and writing data."""
import logging
from datetime import datetime
from pathlib import Path
from typing import Union, Dict
import numpy as np

from pySPM.Bruker import Bruker
from igor import binarywave # this is already imported with afmformats but is being specialised for our use
from afmformats import load_data
from afmformats.mod_creep_compliance import AFMCreepCompliance
from ruamel.yaml import YAML, YAMLError
from ruamel.yaml.main import round_trip_load as yaml_load, round_trip_dump as yaml_dump
from topostats.logs.logs import LOGGER_NAME

LOGGER = logging.getLogger(LOGGER_NAME)

# pylint: disable=broad-except


def read_yaml(filename: Union[str, Path]) -> Dict:
    """Read a YAML file.

    Parameters
    ----------
    filename: Union[str, Path]
        YAML file to read.

    Returns
    -------
    Dict
        Dictionary of the file."""

    with Path(filename).open(encoding="utf-8") as f:
        try:
            yaml_file = YAML(typ="safe")
            return yaml_file.load(f)
        except YAMLError as exception:
            LOGGER.error(exception)
            return {}


def write_yaml(config: dict, output_dir: Union[str, Path]) -> None:
    """Write a configuration (stored as a dictionary) to a YAML file.

    Parameters
    ----------
    config: dict
        Configuration dictionary.
    output_dir: Union[str, Path]
        Path to save the dictionary to as a YAML file (it will be called 'config.yaml').
    """
    # Save the configuration to output directory
    output_config = Path(output_dir) / "config.yaml"
    # Revert PosixPath items to string
    config["base_dir"] = str(config["base_dir"])
    config["output_dir"] = str(config["output_dir"])
    config_yaml = yaml_load(yaml_dump(config))
    config_yaml.yaml_set_start_comment(
        f"Configuration from TopoStats run completed : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    with output_config.open("w") as f:
        try:
            f.write(yaml_dump(config_yaml))
        except YAMLError as exception:
            LOGGER.error(exception)


class LoadScan:
    """Load the image and image parameters from a file path."""

    def __init__(
        self,
        img_path: Union[str, Path],
        channel: str,
    ):

        """Initialise the class.

        Parameters
        ----------
        img_path: Union[str, Path]
            Path to a valid AFM scan to load.
        channel: str
            Image channel to extract from the scan.
        """
        self.img_path = Path(img_path)
        self.channel = channel
        self.channel_data = None
        self.file_path = self.img_path
        self.filename = self.img_path.stem
        self.suffix = self.img_path.suffix
        self.image = None
        self.pixel_to_nm_scaling = None

    def _load_spm(self) -> tuple:
        """Extract image and pixel to nm scaling from the Bruker .spm file."""
        LOGGER.info(f"Loading image from : {self.img_path}")
        try:
            scan = Bruker(self.img_path)
            LOGGER.info(f"[{self.filename}] : Loaded image from : {self.img_path}")
            self.channel_data = scan.get_channel(self.channel)
            LOGGER.info(f"[{self.filename}] : Extracted channel {self.channel}")
            image = np.flipud(np.array(self.channel_data.pixels))
        except FileNotFoundError:
            LOGGER.info(f"[{self.filename}] File not found : {self.img_path}")
        except Exception as exception:
            LOGGER.error(f"[{self.filename}] : {exception}")

        return (image, self._spm_pixel_to_nm_scaling(self.channel_data))

    def _spm_pixel_to_nm_scaling(self, channel_data) -> float:
        """Extract pixel to nm scaling from the SPM image metadata.

        Parameters
        ----------
        channel_data:
            Channel data

        Returns
        -------
        float
            Pixel to nm scaling factor.
        """
        unit_dict = {
            "nm": 1,
            "um": 1e3,
        }
        px_to_real = channel_data.pxs()
        # Has potential for non-square pixels but not yet implimented
        pixel_to_nm_scaling = (
            px_to_real[0][0] * unit_dict[px_to_real[0][1]],
            px_to_real[1][0] * unit_dict[px_to_real[1][1]],
        )[0]
        LOGGER.info(f"[{self.filename}] : Pixel to nm scaling : {pixel_to_nm_scaling}")
        return pixel_to_nm_scaling

    # def load_jpk(self) -> None:
    #     """Load and extract image from .jpk files."""
    #     jpk = self._load_jpk()
    #     data = self._extract_jpk(jpk)
    #     return (jpk, None)

    def _load_ibw(self) -> None:
        """Loads image from Asylum Research (Igor) .ibw files"""
        
        LOGGER.info(f"Loading image from : {self.img_path}")
        try:
            scan = binarywave.load(self.img_path)
            LOGGER.info(f"[{self.filename}] : Loaded image from : {self.img_path}")
            
            labels = []
            for label_list in scan["wave"]["labels"]:
                for label in label_list:
                    if label:
                        labels.append(label.decode())
            channel_idx = labels.index(self.channel)
            image = - scan["wave"]["wData"][:, :, channel_idx].T * 1e9 # Looks to be in m
            image = np.flipud(image)
            LOGGER.info(f"[{self.filename}] : Extracted channel {self.channel}")
        except FileNotFoundError:
            LOGGER.info(f"[{self.filename}] File not found : {self.img_path}")
        except ValueError:
            LOGGER.error(f"[{self.filename}] : {self.channel} not in channel list: {labels}")
        except Exception as exception:
            LOGGER.error(f"[{self.filename}] : {exception}")
        
        return (image, self._ibw_pixel_to_nm_scaling(scan))

    def _ibw_pixel_to_nm_scaling(self, scan) -> None:
        """Extract pixel to nm scaling from the IBW image metadata."""
        # Get metadata
        notes = {}
        for line in str(scan["wave"]["note"]).split("\\r"):
            if line.count(":"):
                key, val = line.split(":", 1)
                notes[key] = val.strip()
        # Has potential for non-square pixels but not yet implimented
        pixel_to_nm_scaling = (
            float(notes['SlowScanSize']) / scan['wave']['wData'].shape[0] * 1e9, # as in m
            float(notes['FastScanSize']) / scan['wave']['wData'].shape[1] * 1e9, # as in m
        )[0]
        LOGGER.info(f"[{self.filename}] : Pixel to nm scaling : {pixel_to_nm_scaling}")
        return pixel_to_nm_scaling

    def _load_jpk(self) -> None:
        try:
            jpk = load_data(self.img_path)
        except FileNotFoundError:
            LOGGER.info(f"[{self.filename}] File not found : {self.img_path}")
        except Exception as exception:
            LOGGER.error(f"[{self.filename}] : {exception}")
        return jpk

    @staticmethod
    def _extract_jpk(jpk: AFMCreepCompliance) -> np.ndarray:
        """Extract data from jpk object"""

    def get_data(self) -> None:
        """Method to extract image and pixel to nm scaling."""
        LOGGER.info(f"Extracting image from {self.suffix}")
        if self.suffix == ".spm":
            self.image, self.pixel_to_nm_scaling = self._load_spm()
        if self.suffix == ".jpk":
            self.image, self.pixel_to_nm_scaling = self.load_jpk()
        if self.suffix == ".ibw":
            self.image, self.pixel_to_nm_scaling = self._load_ibw()
