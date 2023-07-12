"""Functions for reading and writing data."""
import os
import logging
from datetime import datetime
import io
import struct
from pathlib import Path
import pickle as pkl
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import pySPM
from igor import binarywave
import tifffile
from ruamel.yaml import YAML, YAMLError
from ruamel.yaml.main import round_trip_load as yaml_load, round_trip_dump as yaml_dump

from topostats.logs.logs import LOGGER_NAME

LOGGER = logging.getLogger(LOGGER_NAME)


CONFIG_DOCUMENTATION_REFERENCE = """For more information on configuration and how to use it:
# https://afm-spm.github.io/TopoStats/main/configuration.html\n"""

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
        Dictionary of the file.
    """

    with Path(filename).open(encoding="utf-8") as f:
        try:
            yaml_file = YAML(typ="safe")
            return yaml_file.load(f)
        except YAMLError as exception:
            LOGGER.error(exception)
            return {}


def get_date_time() -> str:
    """
    Get a date and time for adding to generated files or logging.

    Parameters
    ----------
    None

    Returns
    -------
    str
        A string of the current date and time, formatted appropriately.
    """

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


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
    config = path_to_str(config)
    config_yaml = yaml_load(yaml_dump(config))

    if header_message:
        config_yaml.yaml_set_start_comment(f"{header_message} : {get_date_time()}\n" + CONFIG_DOCUMENTATION_REFERENCE)
    else:
        config_yaml.yaml_set_start_comment(
            f"Configuration from TopoStats run completed : {get_date_time()}\n" + CONFIG_DOCUMENTATION_REFERENCE
        )
    with output_config.open("w") as f:
        try:
            f.write(yaml_dump(config_yaml))
        except YAMLError as exception:
            LOGGER.error(exception)


def write_config_with_comments(config: str, output_dir: Path, filename: str = "config.yaml") -> None:
    """
    Create a config file, retaining the comments by writing it as a string
    rather than using a yaml handling package.

    Parameters
    ----------
    config: str
        A string of the entire configuration file to be saved.
    output_dir: Path
        A pathlib path of where to create the config file.
    filename: str
        A name for the configuration file. Can have a ".yaml" on the end.
    """

    if ".yaml" not in filename and ".yml" not in filename:
        create_config_path = output_dir / f"{filename}.yaml"
    else:
        create_config_path = output_dir / filename

    with open(f"{create_config_path}", "w", encoding="utf-8") as f:
        f.write(f"# Config file generated {get_date_time()}\n")
        f.write(f"# {CONFIG_DOCUMENTATION_REFERENCE}")
        f.write(config)
    LOGGER.info(f"A sample configuration has been written to : {str(create_config_path)}")
    LOGGER.info(CONFIG_DOCUMENTATION_REFERENCE)


def save_array(array: np.ndarray, outpath: Path, filename: str, array_type: str) -> None:
    """Save a Numpy array to disk.

    Parameters
    ----------
    array : np.ndarray
        Numpy array to be saved.
    outpath : Path
        Location array should be saved
    filename : str
        Filename of the current image from which the array is derived.
    array_type : str
        Short string describing the array type e.g. z_threshold. Ideally should not have periods or spaces in (use
    underscores '_' instead).
    """
    np.save(outpath / f"{filename}_{array_type}.npy", array)
    LOGGER.info(f"[{filename}] Numpy array saved to : {outpath}/{filename}_{array_type}.npy")


def load_array(array_path: Union[str, Path]) -> np.ndarray:
    """Load a Numpy array from file.

    Should have been saved using save_array() or numpy.save().

    Parameters
    ----------
    array_path : Union[str, Path]
        Path to the Numpy array on disk.

    Returns
    -------
    np.ndarray
        Returns the loaded Numpy array.
    """
    try:
        return np.load(Path(array_path))
    except FileNotFoundError as e:
        raise e


def path_to_str(config: dict) -> Dict:
    """Recursively traverse a dictionary and convert any Path() objects to strings for writing to YAML.

    Parameters
    ----------
    config: dict
        Dictionary to be converted.

    Returns
    -------
    Dict:
        The same dictionary with any Path() objects converted to string.
    """
    for key, value in config.items():
        if isinstance(value, dict):
            path_to_str(value)
        elif isinstance(value, Path):
            config[key] = str(value)

    return config


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


def find_files(base_dir: Union[str, Path] = None, file_ext: str = ".spm") -> List:
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


def read_null_terminated_string(open_file: io.TextIOWrapper) -> str:
    """Read an open file from the current position in the open binary file,
    until the next null value.

    Parameters
    ----------
    open_file: io.TextIOWrapper
        An open file object.

    Returns
    -------
    str
        String of the ASCII decoded bytes before the next null byte.
    """
    byte = open_file.read(1)
    value = b""
    while byte != b"\x00":
        value += byte
        byte = open_file.read(1)
    return str(value.decode("utf-8"))


def read_u32i(open_file: io.TextIOWrapper) -> str:
    """Read an unsigned 32 bit integer from an open binary file (in little-endian form).

    Parameters
    ----------
    open_file: io.TextIOWrapper
        An open file object.

    Returns
    -------
    int
        Python integer type cast from the unsigned 32 bit integer.
    """
    return int(struct.unpack("<i", open_file.read(4))[0])


def read_64d(open_file: io.TextIOWrapper) -> str:
    """Read a 64-bit double from an open binary file.

    Parameters
    ----------
    open_file:
        An open file object.

    Returns
    -------
    float
        Python float type cast from the double.
    """
    return float(struct.unpack("d", open_file.read(8))[0])


def read_char(open_file: io.TextIOWrapper) -> str:
    """Read a character from an open binary file.

    Parameters
    ----------
    open_file: io.TextIOWrapper
        An open file object.

    Returns
    -------
    str
        A string type cast from the decoded character.
    """
    return open_file.read(1).decode("ascii")


def read_gwy_component_dtype(open_file: io.TextIOWrapper) -> str:
    """Read the data type of a `.gwy` file component.
    Possible data types are as follows:
    - 'b': boolean
    - 'c': character
    - 'i': 32-bit integer
    - 'q': 64-bit integer
    - 'd': double
    - 's': string
    - 'o': `.gwy` format object
    Capitalised versions of some of these data types represent arrays of values of that
    data type. Arrays are stored as an unsigned 32 bit integer, describing the size of the array,
    followed by the unseparated array values.
    - 'C': array of characters
    - 'I': array of 32-bit integers
    - 'Q': array of 64-bit integers
    - 'D': array of doubles
    - 'S': array of strings
    - 'O': array of objects

    Parameters
    ----------
    open_file: io.TextIOWrapper
        An open file object.

    Returns
    -------
    str
        Python string (one character long) of the data type of the
        component's value.
    """
    return open_file.read(1).decode("ascii")


def get_relative_paths(paths: List[Path]) -> List[str]:
    """From a list of paths, create a list of these paths but where
    each path is relative to all path's closest common parent. For
    example, ['a/b/c', 'a/b/d', 'a/b/e/f'] would return ['c', 'd', 'e/f']

    Parameters
    ----------
    paths: list
        List of string or pathlib paths.

    Returns
    -------
    relative_paths: list
        List of string paths, relative to the common parent.
    """

    # Ensure paths are all pathlib paths, and not strings
    paths = [Path(path) for path in paths]

    # If the paths list consists of all the same path, then the relative path will
    # be '.', which we don't want. we want the relative path to be the full path probably.
    # len(set(my_list)) == 1 determines if all the elements in a list are the same.
    if len(set(paths)) == 1:
        return [str(path.as_posix()) for path in paths]

    deepest_common_path = os.path.commonpath(paths)
    # Have to convert to strings else the dataframe values will be slightly different
    # to what is expected.
    return [str(path.relative_to(deepest_common_path).as_posix()) for path in paths]


def convert_basename_to_relative_paths(df: pd.DataFrame):
    """Converts the paths in the 'basename' column in a dataframe from being
    absolute paths, to paths relative to the deepest common parent. For example
    if the 'basename' column has the following paths: ['/usr/topo/data/a/b', '/usr
    /topo/data/c/d'], the output will be: ['a/b', 'c/d'].

    Parameters
    ----------
    df: pd.DataFrame
        A pandas dataframe containing a column 'basename' which contains the paths
        indicating the locations of the image data files.

    Returns
    -------
    df: pd.DataFrame
        A pandas dataframe where the 'basename' column has paths relative to a common
        parent.
    """

    paths = df["basename"].tolist()
    paths = [Path(path) for path in paths]
    relative_paths = get_relative_paths(paths=paths)
    df["basename"] = relative_paths

    return df


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
        self.img_dict = {}
        self.MINIMUM_IMAGE_SIZE = 10

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
        except Exception as e:
            # trying to return the error with options of possible channel values
            labels = []
            for channel in [layer[b"@2:Image Data"][0] for layer in scan.layers]:
                #channel_name = channel.decode("latin1").split(" ")[1][1:-1]
                channel_description = channel.decode('latin1').split('"')[1] # incase the blank field raises quesions?
                labels.append(channel_description)
            LOGGER.error(f"[{self.filename}] : {self.channel} not in {self.img_path.suffix} channel list: {labels}")
            raise e

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

    @staticmethod
    def _gwy_read_object(open_file: io.TextIOWrapper, data_dict: dict) -> None:
        """Parse and extract data from a `.gwy` file object, starting at the current
        open file read position.

        Parameters
        ----------
        open_file: io.TextIOWrapper
            An open file object.
        data_dict: dict
            Dictionary of `.gwy` file image properties.

        Returns
        -------
        None
        """
        object_name = read_null_terminated_string(open_file=open_file)
        data_size = read_u32i(open_file)
        LOGGER.debug(f"OBJECT | name: {object_name} | data_size: {data_size}")
        # Read components
        read_data_size = 0
        while read_data_size < data_size:
            component_data_size = LoadScans._gwy_read_component(
                open_file=open_file,
                initial_byte_pos=open_file.tell(),
                data_dict=data_dict,
            )
            read_data_size += component_data_size

    @staticmethod
    def _gwy_read_component(open_file: io.TextIOWrapper, initial_byte_pos: int, data_dict: dict) -> int:
        """Parse and extract data from a `.gwy` file object, starting at the current
        open file read position.

        Parameters
        ----------
        open_file: io.TextIOWrapper,
            An open file object.
        data_dict: dict
            Dictionary of `.gwy` file image properties.

        Returns
        -------
        int
            Size of the component in bytes.
        """
        component_name = read_null_terminated_string(open_file=open_file)
        data_type = read_gwy_component_dtype(open_file=open_file)

        if data_type == "o":
            LOGGER.debug(f"component name: {component_name} | dtype: {data_type} |")
            sub_dict = {}
            LoadScans._gwy_read_object(open_file=open_file, data_dict=sub_dict)
            data_dict[component_name] = sub_dict
        elif data_type == "c":
            value = read_char(open_file=open_file)
            LOGGER.debug(f"component name: {component_name} | dtype: {data_type} | value: {value}")
            data_dict[component_name] = value
        elif data_type == "i":
            value = read_u32i(open_file=open_file)
            LOGGER.debug(f"component name: {component_name} | dtype: {data_type} | value: {value}")
            data_dict[component_name] = value
        elif data_type == "d":
            value = read_64d(open_file=open_file)
            LOGGER.debug(f"component name: {component_name} | dtype: {data_type} | value: {value}")
            data_dict[component_name] = value
        elif data_type == "s":
            value = read_null_terminated_string(open_file=open_file)
            LOGGER.debug(f"component name: {component_name} | dtype: {data_type} | value: {value}")
            data_dict[component_name] = value
        elif data_type == "D":
            array_size = read_u32i(open_file=open_file)
            LOGGER.debug(f"component name: {component_name} | dtype: {data_type}")
            LOGGER.debug(f"array size: {array_size}")
            data = np.zeros(array_size)
            for index in range(array_size):
                data[index] = read_64d(open_file=open_file)
            if "xres" in data_dict and "yres" in data_dict:
                data = data.reshape((data_dict["xres"], data_dict["yres"]))
            data_dict["data"] = data

        return open_file.tell() - initial_byte_pos

    @staticmethod
    def _gwy_print_dict(gwy_file_dict: dict, pre_string: str) -> None:
        """A developer function to print the nested object / component structure. Can
        be used to find labels and values of objects / components in the `.gwy` file.

        Parameters
        ----------
        gwy_file_dict: dict
            Dictionary of the nested object / component structure of a `.gwy` file.
        """
        for key, value in gwy_file_dict.items():
            if isinstance(value, dict):
                print(pre_string + f"OBJECT: {key}")
                pre_string += "  "
                LoadScans._gwy_print_dict(gwy_file_dict=value, pre_string=pre_string)
                pre_string = pre_string[:-2]
            else:
                print(pre_string + f"component: {key} | value: {value}")

    @staticmethod
    def _gwy_print_dict_wrapper(gwy_file_dict: dict) -> None:
        """Wrapper for the `_print_gwy_dict` function.

        Parameters
        ----------
        gwy_file_dict: dict
            Dictionary of the nested object / component structure of a `.gwy` file.
        """
        pre_string = ""
        LoadScans._gwy_print_dict(gwy_file_dict=gwy_file_dict, pre_string=pre_string)

    def load_gwy(self) -> tuple:
        """Extract image and pixel to nm scaling from the Gwyddion .gwy file.

        Returns
        -------
        tuple(np.ndarray, float)
            A tuple containing the image and its pixel to nanometre scaling value.
        """
        LOGGER.info(f"Loading image from : {self.img_path}")
        try:
            image_data_dict = {}
            with open(self.img_path, "rb") as open_file:
                # Read header
                header = open_file.read(4)
                LOGGER.debug(f"Gwy file header: {header}")

                LoadScans._gwy_read_object(open_file, data_dict=image_data_dict)

            # For development - uncomment to have an indentation based nested
            # dictionary output showing the object - component structure and
            # available keys:
            # LoadScans._gwy_print_dict_wrapper(gwy_file_dict=image_data_dict)

            if "/0/data" in image_data_dict:
                image = image_data_dict["/0/data"]["data"]
                units = image_data_dict["/0/data"]["si_unit_xy"]["unitstr"]
                px_to_nm = image_data_dict["/0/data"]["xreal"] * 1e9 / image.shape[1]
            elif "/1/data" in image_data_dict:
                image = image_data_dict["/1/data"]["data"]
                px_to_nm = image_data_dict["/1/data"]["xreal"] * 1e9 / image.shape[1]
                units = image_data_dict["/1/data"]["si_unit_xy"]["unitstr"]
            else:
                raise KeyError(
                    "Data location not defined in the .gwy file. Please locate it and add to the load_gwy() function."
                )

            # Convert image heights to nanometresQ
            if units == "m":
                image = image * 1e9
            else:
                raise ValueError(
                    f"Units '{units}' have not been added for .gwy files. Please add \
                    an SI to nanometre conversion factor for these units in _gwy_read_component in \
                    io.py."
                )

        except FileNotFoundError:
            LOGGER.info(f"[{self.filename}] File not found : {self.img_path}")
            raise

        return (image, px_to_nm)

    def get_data(self) -> None:
        """Method to extract image, filepath and pixel to nm scaling value, and append these to the
        img_dic object.
        """

        suffix_to_loader = {
            ".spm": self.load_spm,
            ".jpk": self.load_jpk,
            ".ibw": self.load_ibw,
            ".gwy": self.load_gwy,
        }

        for img_path in self.img_paths:
            self.img_path = img_path
            self.filename = img_path.stem
            suffix = img_path.suffix
            LOGGER.info(f"Extracting image from {self.img_path}")
            LOGGER.debug(f"File extension : {suffix}")

            # Check that the file extension is supported
            if suffix in suffix_to_loader:
                try:
                    self.image, self.pixel_to_nm_scaling = suffix_to_loader[suffix]()
                except Exception as e:
                    if "Channel" in str(e) and "not found" in str(e):
                        LOGGER.warning(f"[{self.filename}] Channel {self.channel} not found, skipping image.")
                    else:
                        raise
                else:
                    self._check_image_size_and_add_to_dict()
            else:
                raise ValueError(
                    f"File type {suffix} not yet supported. Please make an issue at \
                https://github.com/AFM-SPM/TopoStats/issues, or email topostats@sheffield.ac.uk to request support for \
                this file type."
                )

    def _check_image_size_and_add_to_dict(self) -> None:
        """Check the image is above a minimum size in both dimensions.

        Images that do not meet the minimum size are not included for processing.
        """
        if self.image.shape[0] < self.MINIMUM_IMAGE_SIZE or self.image.shape[1] < self.MINIMUM_IMAGE_SIZE:
            LOGGER.warning(f"[{self.filename}] Skipping, image too small: {self.image.shape}")
        else:
            self.add_to_dic(self.filename, self.image, self.img_path.with_name(self.filename), self.pixel_to_nm_scaling)
            LOGGER.info(f"[{self.filename}] Image added to processing.")

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
        self.img_dict[filename] = {"image": image, "img_path": img_path, "px_2_nm": px_2_nm}


def save_pkl(outfile: Path, to_pkl: dict) -> None:
    """Pickle objects for working with later.

    Parameters
    ----------
    outfile: Path
        Path and filename to save pickle to.
    to_pkl: dict
        Object to be picled.

    Returns
    -------
    None
    """
    with outfile.open(mode="wb", encoding=None) as f:
        pkl.dump(to_pkl, f)


def load_pkl(infile: Path) -> Any:
    """Load data from a pickle.

    Parameters
    ----------
    infile: Path
        Path to a valid pickle.

    Returns
    -------
    dict:
        Dictionary of generated images.

    Example
    -------

    from pathlib import Path

    from topostats.io import load_plots

    pkl_path = "output/distribution_plots.pkl"
    my_plots = load_pkl(pkl_path)
    # Show the type of my_plots which is a dictionary of nested dictionaries
    type(my_plots)
    # Show the keys are various levels of nesting.
    my_plots.keys()
    my_plots["area"].keys()
    my_plots["area"]["dist"].keys()
    # Get the figure and axis object for a given metrics distribution plot
    figure, axis = my_plots["area"]["dist"].values()
    # Get the figure and axis object for a given metrics violin plot
    figure, axis = my_plots["area"]["violin"].values()

    """
    with infile.open("rb", encoding=None) as f:
        images = pkl.load(f)
    return images
