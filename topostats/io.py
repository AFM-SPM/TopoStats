"""Functions for reading and writing data."""
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import pySPM
from igor import binarywave
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


def write_yaml(
    config: dict, output_dir: Union[str, Path], config_file: str = "config.yaml", header_message: str = None
) -> None:
    """Write a configuration (stored as a dictionary) to a YAML file.

    Parameters
    ----------
    config: dict
        Configuration dictionary.
    output_dir: Union[str, Path]
        Path to save the dictionary to as a YAML file (it will be called 'config.yaml').
    config_file: str
        Filename to write to.
    header_message: str
        String to write to the header message of the YAML file
    """
    # Save the configuration to output directory
    output_config = Path(output_dir) / config_file
    # Revert PosixPath items to string
    config["base_dir"] = str(config["base_dir"])
    config["output_dir"] = str(config["output_dir"])
    config_yaml = yaml_load(yaml_dump(config))
    if header_message:
        config_yaml.yaml_set_start_comment(f"{header_message} : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        config_yaml.yaml_set_start_comment(
            f"Configuration from TopoStats run completed : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
    with output_config.open("w") as f:
        try:
            f.write(yaml_dump(config_yaml))
        except YAMLError as exception:
            LOGGER.error(exception)


def get_out_path(
    image_path: Union[str, Path] = None, base_dir: Union[str, Path] = None, output_dir: Union[str, Path] = None
) -> Path:
    """Adds the image path relative to the base directory to the output directory.

    Parameters
    ----------
    image_path: Path
        The path of the current image.
    base_dir: Path
        Directory to recursively search for files.
    output_dir: Path
        The output directory specified in the configuration file.

    Returns
    -------
    Path
        The output path that mirrors the input path structure.
    """
    # If image_path is relative and doesn't include base_dir then a ValueError is raised, in which
    # case we just want to append the image_path to the output_dir
    try:
        # Remove the filename if there is a suffix, not always the case as
        # get_out_path is called from save_folder_grainstats()
        if image_path.suffix:
            return output_dir / image_path.relative_to(base_dir).parent / image_path.stem
        return output_dir / image_path.relative_to(base_dir)
    except ValueError:
        if image_path.suffix:
            return output_dir / image_path.parent / image_path.stem
        return Path(str(output_dir) + "/" + str(image_path))
    # AttributeError is raised if image_path is a string (since it isn't a Path() object with a .suffix)
    except AttributeError:
        LOGGER.error("A string form of a Path has been passed to 'get_out_path()' for image_path")
        raise


def find_images(base_dir: Union[str, Path] = None, file_ext: str = ".spm") -> List:
    """Recursively scan the specified directory for images with the given file extension.

    Parameters
    ----------
    base_dir: Union[str, Path]
        Directory to recursively search for files, if not specified the current directory is scanned.
    file_ext: str
        File extension to search for.

    Returns
    -------
    List
        List of files found with the extension in the given directory.
    """
    base_dir = Path("./") if base_dir is None else Path(base_dir)
    return list(base_dir.glob("**/*" + file_ext))


def save_folder_grainstats(
    output_dir: Union[str, Path], base_dir: Union[str, Path], all_stats_df: pd.DataFrame
) -> None:
    """Saves a data frame of grain and tracing statictics at the folder level.

    Parameters
    ----------
    output_dir: Union[str, Path]
        Path of the output directory head.
    base_dir: Union[str, Path]
        Path of the base directory where files were found.
    all_stats_df: pd.DataFrame
        The dataframe containing all sample statistics run.

    Returns
    -------
    None
        This only saves the dataframes and does not retain them.
    """
    dirs = set(all_stats_df["basename"].values)
    LOGGER.debug(f"Statistics :\n{all_stats_df}")
    for _dir in dirs:
        LOGGER.debug(f"Statistics ({_dir}) :\n{all_stats_df}")
        try:
            out_path = get_out_path(Path(_dir), base_dir, output_dir)
            # Ensure "processed" directory exists at the stem of out_path, creating if needed
            if out_path.stem != "processed":
                out_path_processed = out_path / "processed"
                out_path_processed.mkdir(parents=True, exist_ok=True)
            all_stats_df[all_stats_df["basename"] == _dir].to_csv(
                out_path / "processed" / "folder_grainstats.csv", index=True
            )
            LOGGER.info(f"Folder-wise statistics saved to: {str(out_path)}/folder_grainstats.csv")
        except TypeError:
            LOGGER.info(f"No folder-wise statistics for directory {_dir}, no grains detected in any images.")


# pylint: disable=too-many-instance-attributes
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

    @staticmethod
    def _jpk_pixel_to_nm_scaling(tiff_page: tifffile.tifffile.TiffPage) -> float:
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

    def add_to_dic(self, filename: str, image: np.ndarray, img_path: Path, px_2_nm: float) -> None:
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
