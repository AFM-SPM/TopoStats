"""Functions for reading and writing data."""
import logging
from datetime import datetime
from pathlib import Path
from typing import Union, Dict
import numpy as np

import pySPM
from igor import binarywave
import libasd
import tifffile
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
        self.image = None
        self.pixel_to_nm_scaling = None
        self.img_dic = {}

    def load_spm(self) -> tuple:
        """Extract image and pixel to nm scaling from the Bruker .spm file.

        Returns
        -------
        tuple(np.ndarray, float)
            A tuple containing the image and its pixel to nanometre scaling value.
        """
        LOGGER.info(f"Loading image from : {self.img_path}")
        try:
            scan = pySPM.Bruker(self.img_path)
            LOGGER.info(f"[{self.filename}] : Loaded image from : {self.img_path}")
            self.channel_data = scan.get_channel(self.channel)
            LOGGER.info(f"[{self.filename}] : Extracted channel {self.channel}")
            image = np.flipud(np.array(self.channel_data.pixels))
        except FileNotFoundError:
            LOGGER.info(f"[{self.filename}] File not found : {self.img_path}")
            raise
        except Exception:
            # trying to return the error with options of possible channel values
            labels = []
            for channel in [layer[b"@2:Image Data"][0] for layer in scan.layers]:
                channel_name = channel.decode("latin1").split(" ")[1][1:-1]
                # channel_description = channel.decode('latin1').split('"')[1] # incase the blank field raises quesions?
                labels.append(channel_name)
            LOGGER.error(f"[{self.filename}] : {self.channel} not in {self.img_path.suffix} channel list: {labels}")
            raise

        return (image, self._spm_pixel_to_nm_scaling(self.channel_data))

    def _spm_pixel_to_nm_scaling(self, channel_data: pySPM.SPM.SPM_image) -> float:
        """Extract pixel to nm scaling from the SPM image metadata.

        Parameters
        ----------
        channel_data: pySPM.SPM.SPM_image
            Channel data from PySPM.

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
        if px_to_real[0][0] == 0 and px_to_real[1][0] == 0:
            pixel_to_nm_scaling = 1
            LOGGER.warning(f"[{self.filename}] : Pixel size not found in metadata, defaulting to 1nm")
        LOGGER.info(f"[{self.filename}] : Pixel to nm scaling : {pixel_to_nm_scaling}")
        return pixel_to_nm_scaling

    def load_ibw(self) -> tuple:
        """Loads image from Asylum Research (Igor) .ibw files
        
        Returns
        -------
        tuple(np.ndarray, float)
            A tuple containing the image and its pixel to nanometre scaling value.
        """

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
            image = scan["wave"]["wData"][:, :, channel_idx].T * 1e9  # Looks to be in m
            image = np.flipud(image)
            LOGGER.info(f"[{self.filename}] : Extracted channel {self.channel}")
        except FileNotFoundError:
            LOGGER.info(f"[{self.filename}] File not found : {self.img_path}")
        except ValueError:
            LOGGER.error(f"[{self.filename}] : {self.channel} not in {self.img_path.suffix} channel list: {labels}")
            raise
        except Exception as exception:
            LOGGER.error(f"[{self.filename}] : {exception}")

        return (image, self._ibw_pixel_to_nm_scaling(scan))

    def _ibw_pixel_to_nm_scaling(self, scan: dict) -> float:
        """Extract pixel to nm scaling from the IBW image metadata.
        
        Parameters
        ----------
        scan: dict
            The loaded binary wave object.

        Returns
        -------
        float
            A value corresponding to the real length of a single pixel.
        """
        # Get metadata
        notes = {}
        for line in str(scan["wave"]["note"]).split("\\r"):
            if line.count(":"):
                key, val = line.split(":", 1)
                notes[key] = val.strip()
        # Has potential for non-square pixels but not yet implimented
        pixel_to_nm_scaling = (
            float(notes["SlowScanSize"]) / scan["wave"]["wData"].shape[0] * 1e9,  # as in m
            float(notes["FastScanSize"]) / scan["wave"]["wData"].shape[1] * 1e9,  # as in m
        )[0]
        LOGGER.info(f"[{self.filename}] : Pixel to nm scaling : {pixel_to_nm_scaling}")
        return pixel_to_nm_scaling

    def load_jpk(self) -> tuple:
        """Loads image from JPK Instruments .jpk files.
        
        Returns
        -------
        tuple(np.ndarray, float)
            A tuple containing the image and its pixel to nanometre scaling value.
        """
        # Load the file
        img_path = str(self.img_path)
        try:
            tif = tifffile.TiffFile(img_path)
        except FileNotFoundError:
            LOGGER.info(f"[{self.filename}] File not found : {self.img_path}")
            raise
        # Obtain channel list for all channels in file
        channel_list = {}
        for i, page in enumerate(tif.pages[1:]):  # [0] is thumbnail
            available_channel = page.tags["32848"].value  # keys are hexidecimal vals
            if page.tags["32849"].value == 0:  # wether img is trace or retrace
                tr_rt = "trace"
            else:
                tr_rt = "retrace"
            channel_list[f"{available_channel}_{tr_rt}"] = i + 1
        try:
            channel_idx = channel_list[self.channel]
        except KeyError:
            LOGGER.error(f"{self.channel} not in channel list: {channel_list}")
            raise
        # Get image and if applicable, scale it
        channel_page = tif.pages[channel_idx]
        image = channel_page.asarray()
        scaling_type = channel_page.tags["33027"].value
        if scaling_type == "LinearScaling":
            scaling = channel_page.tags["33028"].value
            offset = channel_page.tags["33029"].value
            image = (image * scaling) + offset
        elif scaling_type == "NullScaling":
            pass
        else:
            raise ValueError(f"Scaling type {scaling_type} is not 'NullScaling' or 'LinearScaling'")
        # Get page for common metadata between scans
        metadata_page = tif.pages[0]
        return (image * 1e9, self._jpk_pixel_to_nm_scaling(metadata_page))

    def _jpk_pixel_to_nm_scaling(self, tiff_page: tifffile.tifffile.TiffPage) -> float:
        """Extract pixel to nm scaling from the JPK image metadata.

        Parameters
        ----------
        tiff_page: tifffile.tifffile.TiffPage
            An image file directory (IFD) of .jpk files.

        Returns
        -------
        float
            A value corresponding to the real length of a single pixel.
        """
        length = tiff_page.tags["32834"].value  # Grid-uLength (fast)
        width = tiff_page.tags["32835"].value  # Grid-vLength (slow)
        length_px = tiff_page.tags["32838"].value  # Grid-iLength (fast)
        width_px = tiff_page.tags["32839"].value  # Grid-jLength (slow)

        px_to_nm = (length / length_px, width / width_px)[0]

        LOGGER.info(px_to_nm)

        return px_to_nm * 1e9
        
    def load_asd(self) -> tuple:
        """Extract image and pixel to nm scaling from .asd files.

        Returns
        -------
        tuple(np.ndarray, float)
            A tuple containing the image and its pixel to nanometre scaling value.
        """
        try:
            img_path_str = str(self.img_path)
            scan = libasd.read_asd(img_path_str)
            LOGGER.info(f"[{self.filename}] : Loaded image from : {self.img_path}")
            scan_header = scan.header
        except FileNotFoundError:
            LOGGER.info(f"[{self.filename}] File not found : {self.img_path}")

        try:
            if type(scan_header) == libasd.Header_v0:
                # libasd docs seem like there is only 2 channels i.e. help(scan)
                # libasd.Header_v0 uses "type" func and _v1 uses "kind" func
                ch1_name = str(scan.header.data_type_1ch).split(".")[1]
                ch2_name = str(scan.header.data_type_2ch).split(".")[1]
            elif type(scan_header) == libasd.Header_v1:
                ch1_name = str(scan.header.data_kind_1ch).split(".")[1]
                ch2_name = str(scan.header.data_kind_2ch).split(".")[1]
            else:
                raise AttributeError
        except AttributeError:
            LOGGER.error(
                f"[{self.filename}] : File header not found. Header of type {type(scan_header)}, not [{libasd.Header_v0, libasd.Header_v1}]"
            )
            raise
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
            LOGGER.error(
                f"[{self.filename}] : {self.channel} not found in {self.img_path.suffix} channel list: [{ch1_name}, {ch2_name}"
            )
            raise ValueError(
                f"[{self.filename}] : {self.channel} not found in {self.img_path.suffix} channel list: [{ch1_name}, {ch2_name}"
            )

        return (images, self._asd_px_to_nm_scaling(scan))

    def _asd_px_to_nm_scaling(self, scan: Union[libasd.Data2ch_v0, libasd.Data2ch_v1]) -> float:
        """Extracts the pixel to nanometre scaling value from an .asd file object.

        Parameters
        ----------
        scan: Union[libasd.Data2ch_v0, libasd.Data2ch_v1]
            The object resulting from loading the filepath via the libasd library.
        Returns
        -------
        px_to_nm_scaling: float
            The length of a single pixel in real length units (nm).

        """
        px_to_nm_scaling = (
            scan.header.x_scanning_range / scan.header.x_pixel,  # scan range always in nm
            scan.header.y_scanning_range / scan.header.y_pixel,
        )[0]
        return px_to_nm_scaling

    def get_data(self) -> None:
        """Method to extract image, filepath and pixel to nm scaling value, and append these to the
        img_dic object.
        """
        for img_path in self.img_paths:
            self.img_path = img_path
            self.filename = img_path.stem
            suffix = img_path.suffix
            LOGGER.info(f"Extracting image from {self.img_path}")
            if suffix == ".spm":
                self.image, self.pixel_to_nm_scaling = self.load_spm()
                self.add_to_dic(
                    self.filename, self.image, self.img_path.with_name(self.filename), self.pixel_to_nm_scaling
                )
            if suffix == ".jpk":
                self.image, self.pixel_to_nm_scaling = self.load_jpk()
                self.add_to_dic(
                    self.filename, self.image, self.img_path.with_name(self.filename), self.pixel_to_nm_scaling
                )
            if suffix == ".ibw":
                self.image, self.pixel_to_nm_scaling = self.load_ibw()
                self.add_to_dic(
                    self.filename, self.image, self.img_path.with_name(self.filename), self.pixel_to_nm_scaling
                )
            if suffix == ".asd":
                self.image, self.pixel_to_nm_scaling = self.load_asd()
                for i, frame in enumerate(self.image):
                    filename = self.filename + f"_frame_{str(i)}"
                    pathname = self.img_path.with_name(filename)
                    self.add_to_dic(filename, frame, pathname, self.pixel_to_nm_scaling)

    def add_to_dict(self, filename: str, image: np.ndarray, img_path: Path, px_2_nm: float) -> None:
        """Adds the image, image path and pixel to nanometre scaling value to the img_dic dictionary under 
        the key filename.

        Parameters
        ----------
        filename: str
            The filename, idealy without an extension.
        image: np.ndarray
            An array of the extracted AFM image.
        img_path: str
            The path to the AFM file (with a frame number if applicable)
        px_2_nm: float
            The length of a pixel in nm.
        """
        self.img_dic[filename] = {"image": image, "img_path": img_path, "px_2_nm": px_2_nm}
