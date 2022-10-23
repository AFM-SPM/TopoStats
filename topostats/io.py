"""Functions for reading and writing data."""
import logging
from datetime import datetime
from pathlib import Path
from typing import Union, Dict
import numpy as np

from pySPM.Bruker import Bruker
from afmformats import load_data
import libasd
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


class LoadScans:
    """Load the image and image parameters from a file path."""

    def __init__(
        self,
        img_paths: list,
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
        self.img_paths = img_paths
        self.img_path = None
        self.channel = channel
        self.channel_data = None
        self.filename = None
        self.suffix = None
        self.image = None
        self.pixel_to_nm_scaling = None
        self.img_dic = {
            "images": [],
            "img_paths": [],
            "px_2_nms": []
        }

    def load_spm(self) -> tuple:
        """Extract image and pixel to nm scaling from the Bruker .spm file.
        
        Returns
        -------
        tuple(np.ndarray, float)
            A tuple containing the image and its pixel to nanometre scaling value.
        """
        LOGGER.info(f"Loading image from : {self.img_path}")
        try:
            scan = Bruker(self.img_path)
            LOGGER.info(f"[{self.filename}] : Loaded image from : {self.img_path}")
            self.channel_data = scan.get_channel(self.channel)
            LOGGER.info(f"[{self.filename}] : Extracted channel {self.channel}")
            image = np.flipud(np.array(self.channel_data.pixels))
        except FileNotFoundError:
            LOGGER.info(f"[{self.filename}] File not found : {self.img_path}")
            raise
        except Exception as exception:
            LOGGER.error(f"[{self.filename}] : {exception}")
            raise

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

    def _load_jpk(self) -> None:
        try:
            jpk = load_data(self.img_path)
        except FileNotFoundError:
            LOGGER.info(f"[{self.filename}] File not found : {self.img_path}")
        except Exception as exception:
            LOGGER.error(f"[{self.filename}] : {exception}")
        return jpk

    def load_asd(self) -> tuple:
        """Extract image and pixel to nm scaling from .asd files.
        
        Returns
        -------
        tuple(np.ndarray, float)
            A tuple containing the image and its pixel to nanometre scaling value.
        """
        try:
            scan = libasd.read_asd(self.img_path)
            print(type(scan))
            LOGGER.info(f"[{self.filename}] : Loaded image from : {self.img_path}")
        except FileNotFoundError:
            LOGGER.info(f"[{self.filename}] File not found : {self.img_path}")
        try:
            # libasd docs seem like there is only 2 channels i.e. help(scan)
            # libasd.Header_v0 uses "type" and _v1 uses "kind"
            ch1_name = str(scan.header.data_type_1ch).split('.')[1]
            ch2_name = str(scan.header.data_type_2ch).split('.')[1]
        except AttributeError:
            ch1_name = str(scan.header.data_kind_1ch).split('.')[1]
            ch2_name = str(scan.header.data_kind_2ch).split('.')[1]
        try:
            if self.channel == ch1_name:
                channel_data = scan.channels[0]
            elif self.channel == ch2_name:
                channel_data = scan.channels[1]
            else:
                raise ValueError
            LOGGER.info(f"[{self.filename}] : Extracted channel {self.channel} with {len(channel_data)} frames")
            images = [channel_data[i].image() for i in range(len(channel_data))]
        except ValueError:
            LOGGER.info(f"[{self.filename}] : {self.channel} not found in {self.suffix} channel list: [{ch1_name}, {ch2_name}")
            raise ValueError(f"[{self.filename}] : {self.channel} not found in {self.suffix} channel list: [{ch1_name}, {ch2_name}")

        return (images, self._asd_px_to_nm_scaling(scan))

    def _asd_px_to_nm_scaling(self, scan: Union[libasd.Data2ch_v0, libasd.Data2ch_v1]) -> float:
        """Extracts the pixel to nanometre scaling value from an .asd file object.

        Parameters
        ----------
        scan: Union[libasd.Data2ch_v0, libasd.Data2ch_v1]
            The object resulting from loading the filepath via the libasd library.
        Returns
        -------
        px_to_m_scaling: float
            The length of a single pixel in real length units (nm).
        
        """
        px_to_m_scaling = (
            scan.header.x_scanning_range / scan.header.x_pixel, # scan range always in nm
            scan.header.y_scanning_range / scan.header.y_pixel,
        )[0]
        return px_to_m_scaling

    @staticmethod
    def _extract_jpk(jpk: AFMCreepCompliance) -> np.ndarray:
        """Extract data from jpk object"""

    def get_data(self) -> None:
        """Method to extract image, filepath and pixel to nm scaling value, and append these to the 
        img_dic object.
        """
        for img_path in self.img_paths:
            self.img_path = img_path
            self.filename = img_path.stem
            self.suffix = img_path.suffix
            LOGGER.info(f"Extracting image from {self.img_path}")
            if self.suffix == ".spm":
                self.image, self.pixel_to_nm_scaling = self.load_spm()
                self.add_to_dic(self.image, self.img_path, self.pixel_to_nm_scaling)
            if self.suffix == ".jpk":
                self.image, self.pixel_to_nm_scaling = self.load_jpk()
            if self.suffix == ".ibw":
                self.image, self.pixel_to_nm_scaling = self.load_ibw()
                self.add_to_dic(self.image, self.img_path, self.pixel_to_nm_scaling)
            if self.suffix == ".asd":
                self.img_path = str(self.img_path)
                self.image, self.pixel_to_nm_scaling = self.load_asd()
                for i, frame in enumerate(self.image):
                    pathname = list(Path(self.img_path).parts)
                    pathname[-1] = self.filename+"_frame_"+str(i)
                    pathname = Path(*pathname)
                    self.add_to_dic(frame, pathname, self.pixel_to_nm_scaling)

    def add_to_dic(self, image, img_path, px_2_nm):
        "Adds the image, image path and pixel to nanometre scaling value to the img_dic dictionary"
        self.img_dic["images"].append(image)
        self.img_dic["img_paths"].append(img_path)
        self.img_dic["px_2_nms"].append(px_2_nm)