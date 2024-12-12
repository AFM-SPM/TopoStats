"""Functions for reading and writing data."""

from __future__ import annotations

import io
import json
import logging
import os
import pickle as pkl
import struct
from collections.abc import MutableMapping
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Any, TypeVar

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
from AFMReader import asd, gwy, ibw, jpk, spm, topostats
from numpyencoder import NumpyEncoder
from ruamel.yaml import YAML, YAMLError

from topostats.logs.logs import LOGGER_NAME

LOGGER = logging.getLogger(LOGGER_NAME)


CONFIG_DOCUMENTATION_REFERENCE = """# For more information on configuration and how to use it:
# https://afm-spm.github.io/TopoStats/main/configuration.html\n"""

# pylint: disable=broad-except
# pylint: disable=too-many-lines

MutableMappingType = TypeVar("MutableMappingType", bound="MutableMapping")


def merge_mappings(map1: MutableMappingType, map2: MutableMappingType) -> MutableMappingType:
    """
    Merge two mappings (dictionaries), with priority given to the second mapping.

    Note: Using a Mapping should make this robust to any mapping type, not just dictionaries. MutableMapping was needed
    as Mapping is not a mutable type, and this function needs to be able to change the dictionaries.

    Parameters
    ----------
    map1 : MutableMapping
        First mapping to merge, with secondary priority.
    map2 : MutableMapping
        Second mapping to merge, with primary priority.

    Returns
    -------
    dict
        Merged dictionary.
    """
    # Iterate over the second mapping
    for key, value in map2.items():
        # If the value is another mapping, then recurse
        if isinstance(value, MutableMapping):
            # If the key is not in the first mapping, add it as an empty dictionary before recursing
            map1[key] = merge_mappings(map1.get(key, {}), value)
        else:
            # Else simply add / override the key value pair
            map1[key] = value
    return map1


# Sylvia: Ruff says too complex but I think breaking this out would be more complex.
def dict_almost_equal(dict1: dict, dict2: dict, abs_tol: float = 1e-9):  # noqa: C901
    """
    Recursively check if two dictionaries are almost equal with a given absolute tolerance.

    Parameters
    ----------
    dict1 : dict
        First dictionary to compare.
    dict2 : dict
        Second dictionary to compare.
    abs_tol : float
        Absolute tolerance to check for equality.

    Returns
    -------
    bool
        True if the dictionaries are almost equal, False otherwise.
    """
    if dict1.keys() != dict2.keys():
        return False

    LOGGER.debug("Comparing dictionaries")

    for key in dict1:
        LOGGER.debug(f"Comparing key {key}")
        if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            if not dict_almost_equal(dict1[key], dict2[key], abs_tol=abs_tol):
                return False
        elif isinstance(dict1[key], np.ndarray) and isinstance(dict2[key], np.ndarray):
            if not np.allclose(dict1[key], dict2[key], atol=abs_tol):
                LOGGER.debug(f"Key {key} type: {type(dict1[key])} not equal: {dict1[key]} != {dict2[key]}")
                return False
        elif isinstance(dict1[key], float) and isinstance(dict2[key], float):
            # Skip if both values are NaN
            if not (np.isnan(dict1[key]) and np.isnan(dict2[key])):
                # Check if both values are close
                if not np.isclose(dict1[key], dict2[key], atol=abs_tol):
                    LOGGER.debug(f"Key {key} type: {type(dict1[key])} not equal: {dict1[key]} != {dict2[key]}")
                    return False

        elif dict1[key] != dict2[key]:
            LOGGER.debug(f"Key {key} not equal: {dict1[key]} != {dict2[key]}")
            return False

    return True


def read_yaml(filename: str | Path) -> dict:
    """
    Read a YAML file.

    Parameters
    ----------
    filename : Union[str, Path]
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

    Returns
    -------
    str
        A string of the current date and time, formatted appropriately.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_yaml(
    config: dict,
    output_dir: str | Path,
    config_file: str = "config.yaml",
    header_message: str = None,
) -> None:
    """
    Write a configuration (stored as a dictionary) to a YAML file.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    output_dir : Union[str, Path]
        Path to save the dictionary to as a YAML file (it will be called 'config.yaml').
    config_file : str
        Filename to write to.
    header_message : str
        String to write to the header message of the YAML file.
    """
    # Save the configuration to output directory
    output_config = Path(output_dir) / config_file
    # Revert PosixPath items to string
    config = path_to_str(config)

    if header_message:
        header = f"# {header_message} : {get_date_time()}\n" + CONFIG_DOCUMENTATION_REFERENCE
    else:
        header = f"# Configuration from TopoStats run completed : {get_date_time()}\n" + CONFIG_DOCUMENTATION_REFERENCE
    output_config.write_text(header, encoding="utf-8")

    yaml = YAML(typ="safe")
    with output_config.open("a", encoding="utf-8") as f:
        try:
            yaml.dump(config, f)
        except YAMLError as exception:
            LOGGER.error(exception)


def write_config_with_comments(args=None) -> None:
    """
    Write a sample configuration with in-line comments.

    This function is not designed to be used interactively but can be, just call it without any arguments and it will
    write a configuration to './config.yaml'.

    Parameters
    ----------
    args : Namespace
        A Namespace object parsed from argparse with values for 'filename'.
    """
    filename = "config" if args.filename is None else args.filename
    output_dir = Path("./") if args.output_dir is None else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger_msg = "A sample configuration has been written to"
    # If no config or default is requested we load the default_config.yaml
    if args.config is None or args.config == "default":
        if args.simple:
            config_path = resources.files(__package__) / "simple_config.yaml"
        else:
            config_path = resources.files(__package__) / "default_config.yaml"
        config = config_path.read_text()
    elif args.config == "topostats.mplstyle":
        config = (resources.files(__package__) / "topostats.mplstyle").read_text()
        logger_msg = "A sample matplotlibrc parameters file has been written to"
    # Otherwise we have scope for loading different configs based on the argument, add future dictionaries to
    # topostats/<sample_type>_config.yaml
    else:
        try:
            config = (resources.files(__package__) / f"{args.config}_config.yaml").read_text()
        except FileNotFoundError as e:
            raise UserWarning(f"There is no configuration for samples of type : {args.config}") from e

    if ".yaml" not in str(filename) and ".yml" not in str(filename) and ".mplstyle" not in str(filename):
        create_config_path = output_dir / f"{filename}.yaml"
    else:
        create_config_path = output_dir / filename

    with create_config_path.open("w", encoding="utf-8") as f:
        f.write(f"# Config file generated {get_date_time()}\n")
        f.write(f"# {CONFIG_DOCUMENTATION_REFERENCE}")
        f.write(config)
    LOGGER.info(f"{logger_msg} : {str(create_config_path)}")
    LOGGER.info(CONFIG_DOCUMENTATION_REFERENCE)


def save_array(array: npt.NDArray, outpath: Path, filename: str, array_type: str) -> None:
    """
    Save a Numpy array to disk.

    Parameters
    ----------
    array : npt.NDArray
        Numpy array to be saved.
    outpath : Path
        Location array should be saved.
    filename : str
        Filename of the current image from which the array is derived.
    array_type : str
        Short string describing the array type e.g. z_threshold. Ideally should not have periods or spaces in (use
        underscores '_' instead).
    """
    np.save(outpath / f"{filename}_{array_type}.npy", array)
    LOGGER.info(f"[{filename}] Numpy array saved to : {outpath}/{filename}_{array_type}.npy")


def load_array(array_path: str | Path) -> npt.NDArray:
    """
    Load a Numpy array from file.

    Should have been saved using save_array() or numpy.save().

    Parameters
    ----------
    array_path : Union[str, Path]
        Path to the Numpy array on disk.

    Returns
    -------
    npt.NDArray
        Returns the loaded Numpy array.
    """
    try:
        return np.load(Path(array_path))
    except FileNotFoundError as e:
        raise e


def path_to_str(config: dict) -> dict:
    """
    Recursively traverse a dictionary and convert any Path() objects to strings for writing to YAML.

    Parameters
    ----------
    config : dict
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


def get_out_path(image_path: str | Path = None, base_dir: str | Path = None, output_dir: str | Path = None) -> Path:
    """
    Add the image path relative to the base directory to the output directory.

    Parameters
    ----------
    image_path : Path
        The path of the current image.
    base_dir : Path
        Directory to recursively search for files.
    output_dir : Path
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


def find_files(base_dir: str | Path = None, file_ext: str = ".spm") -> list:
    """
    Recursively scan the specified directory for images with the given file extension.

    Parameters
    ----------
    base_dir : Union[str, Path]
        Directory to recursively search for files, if not specified the current directory is scanned.
    file_ext : str
        File extension to search for.

    Returns
    -------
    List
        List of files found with the extension in the given directory.
    """
    base_dir = Path("./") if base_dir is None else Path(base_dir)
    return list(base_dir.glob("**/*" + file_ext))


def save_folder_grainstats(
    output_dir: str | Path, base_dir: str | Path, all_stats_df: pd.DataFrame, stats_filename: str
) -> None:
    """
    Save a data frame of grain and tracing statistics at the folder level.

    Parameters
    ----------
    output_dir : Union[str, Path]
        Path of the output directory head.
    base_dir : Union[str, Path]
        Path of the base directory where files were found.
    all_stats_df : pd.DataFrame
        The dataframe containing all sample statistics run.
    stats_filename : str
        The name of the type of statistics dataframe to be saved.

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
                out_path / "processed" / f"folder_{stats_filename}.csv", index=True
            )
            LOGGER.info(f"Folder-wise statistics saved to: {str(out_path)}/folder_{stats_filename}.csv")
        except TypeError:
            LOGGER.info(f"No folder-wise statistics for directory {_dir}, no grains detected in any images.")


def read_null_terminated_string(open_file: io.TextIOWrapper, encoding: str = "utf-8") -> str:
    """
    Read an open file from the current position in the open binary file, until the next null value.

    Parameters
    ----------
    open_file : io.TextIOWrapper
        An open file object.
    encoding : str
        Encoding to use when decoding the bytes.

    Returns
    -------
    str
        String of the ASCII decoded bytes before the next null byte.

    Examples
    --------
    >>> with open("test.txt", "rb") as f:
    ...     print(read_null_terminated_string(f), encoding="utf-8")
    """
    byte = open_file.read(1)
    value = b""
    while byte != b"\x00":
        value += byte
        byte = open_file.read(1)
    # Sometimes encodings cannot decode a byte that is not defined in the encoding.
    # Try 'latin1' in this case as it is able to handle symbols such as micro (µ).
    try:
        return str(value.decode(encoding=encoding))
    except UnicodeDecodeError as e:
        if "codec can't decode byte" in str(e):
            bad_byte = str(e).split("byte ")[1].split(":")[0]
            LOGGER.debug(
                f"Decoding error while reading null terminated string. Encoding {encoding} encountered"
                f" a byte that could not be decoded: {bad_byte}. Trying 'latin1' encoding."
            )
            return str(value.decode(encoding="latin1"))
        raise e


def read_u32i(open_file: io.TextIOWrapper) -> str:
    """
    Read an unsigned 32 bit integer from an open binary file (in little-endian form).

    Parameters
    ----------
    open_file : io.TextIOWrapper
        An open file object.

    Returns
    -------
    int
        Python integer type cast from the unsigned 32 bit integer.
    """
    return int(struct.unpack("<i", open_file.read(4))[0])


def read_64d(open_file: io.TextIOWrapper) -> str:
    """
    Read a 64-bit double from an open binary file.

    Parameters
    ----------
    open_file : io.TextIOWrapper
        An open file object.

    Returns
    -------
    float
        Python float type cast from the double.
    """
    return float(struct.unpack("d", open_file.read(8))[0])


def read_char(open_file: io.TextIOWrapper) -> str:
    """
    Read a character from an open binary file.

    Parameters
    ----------
    open_file : io.TextIOWrapper
        An open file object.

    Returns
    -------
    str
        A string type cast from the decoded character.
    """
    return open_file.read(1).decode("ascii")


def read_gwy_component_dtype(open_file: io.TextIOWrapper) -> str:
    """
    Read the data type of a `.gwy` file component.

    Possible data types are as follows:

    - 'b': boolean
    - 'c': character
    - 'i': 32-bit integer
    - 'q': 64-bit integer
    - 'd': double
    - 's': string
    - 'o': `.gwy` format object

    Capitalised versions of some of these data types represent arrays of values of that data type. Arrays are stored as
    an unsigned 32 bit integer, describing the size of the array, followed by the unseparated array values:

    - 'C': array of characters
    - 'I': array of 32-bit integers
    - 'Q': array of 64-bit integers
    - 'D': array of doubles
    - 'S': array of strings
    - 'O': array of objects.

    Parameters
    ----------
    open_file : io.TextIOWrapper
        An open file object.

    Returns
    -------
    str
        Python string (one character long) of the data type of the component's value.
    """
    return open_file.read(1).decode("ascii")


def get_relative_paths(paths: list[Path]) -> list[str]:
    """
    Extract a list of relative paths, removing the common suffix.

    From a list of paths, create a list where each path is relative to all path's closest common parent. For example,
    ['a/b/c', 'a/b/d', 'a/b/e/f'] would return ['c', 'd', 'e/f'].

    Parameters
    ----------
    paths : list
        List of string or pathlib paths.

    Returns
    -------
    list
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
    """
    Convert paths in the 'basename' column of a dataframe to relative paths.

    If the 'basename' column has the following paths: ['/usr/topo/data/a/b', '/usr/topo/data/c/d'], the output will be:
    ['a/b', 'c/d'].

    Parameters
    ----------
    df : pd.DataFrame
        A pandas dataframe containing a column 'basename' which contains the paths
        indicating the locations of the image data files.

    Returns
    -------
    pd.DataFrame
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
    """
    Load the image and image parameters from a file path.

    Parameters
    ----------
    img_paths : list[str, Path]
        Path to a valid AFM scan to load.
    channel : str
        Image channel to extract from the scan.
    """

    def __init__(
        self,
        img_paths: list[str | Path],
        channel: str,
    ):
        """
        Initialise the class.

        Parameters
        ----------
        img_paths : list[str | Path]
            Path to a valid AFM scan to load.
        channel : str
            Image channel to extract from the scan.
        """
        self.img_paths = img_paths
        self.img_path = None
        self.channel = channel
        self.channel_data = None
        self.filename = None
        self.image = None
        self.pixel_to_nm_scaling = None
        self.grain_masks = {}
        self.grain_trace_data = {}
        self.img_dict = {}
        self.MINIMUM_IMAGE_SIZE = 10

    def load_spm(self) -> tuple[npt.NDArray, float]:
        """
        Extract image and pixel to nm scaling from the Bruker .spm file.

        Returns
        -------
        tuple[npt.NDArray, float]
            A tuple containing the image and its pixel to nanometre scaling value.
        """
        try:
            LOGGER.debug(f"Loading image from : {self.img_path}")
            return spm.load_spm(file_path=self.img_path, channel=self.channel)
        except FileNotFoundError:
            LOGGER.error(f"File Not Found : {self.img_path}")
            raise

    def load_topostats(self) -> tuple[npt.NDArray, float]:
        """
        Load a .topostats file (hdf5 format).

        Loads and extracts the image, pixel to nanometre scaling factor and any grain masks.

        Note that grain masks are stored via self.grain_masks rather than returned due to how we extract information for
        all other file loading functions.

        Returns
        -------
        tuple[npt.NDArray, float]
            A tuple containing the image and its pixel to nanometre scaling value.
        """
        try:
            LOGGER.debug(f"Loading image from : {self.img_path}")
            return topostats.load_topostats(self.img_path)
        except FileNotFoundError:
            LOGGER.error(f"File Not Found : {self.img_path}")
            raise

    def load_asd(self) -> tuple[npt.NDArray, float]:
        """
        Extract image and pixel to nm scaling from .asd files.

        Returns
        -------
        tuple[npt.NDArray, float]
            A tuple containing the image and its pixel to nanometre scaling value.
        """
        try:
            frames: np.ndarray
            pixel_to_nm_scaling: float
            _: dict
            frames, pixel_to_nm_scaling, _ = asd.load_asd(file_path=self.img_path, channel=self.channel)
            LOGGER.debug(f"[{self.filename}] : Loaded image from : {self.img_path}")
        except FileNotFoundError:
            LOGGER.error(f"File not found. Path: {self.img_path}")
            raise

        return (frames, pixel_to_nm_scaling)

    def load_ibw(self) -> tuple[npt.NDArray, float]:
        """
        Load image from Asylum Research (Igor) .ibw files.

        Returns
        -------
        tuple[npt.NDArray, float]
            A tuple containing the image and its pixel to nanometre scaling value.
        """
        try:
            LOGGER.debug(f"Loading image from : {self.img_path}")
            return ibw.load_ibw(file_path=self.img_path, channel=self.channel)
        except FileNotFoundError:
            LOGGER.error(f"File not found : {self.img_path}")
            raise

    def load_jpk(self) -> tuple[npt.NDArray, float]:
        """
        Load image from JPK Instruments .jpk files.

        Returns
        -------
        tuple[npt.NDArray, float]
            A tuple containing the image and its pixel to nanometre scaling value.
        """
        try:
            return jpk.load_jpk(file_path=self.img_path, channel=self.channel)
        except FileNotFoundError:
            LOGGER.error(f"[{self.filename}] File not found : {self.img_path}")
            raise

    def load_gwy(self) -> tuple[npt.NDArray, float]:
        """
        Extract image and pixel to nm scaling from the Gwyddion .gwy file.

        Returns
        -------
        tuple[npt.NDArray, float]
            A tuple containing the image and its pixel to nanometre scaling value.
        """
        LOGGER.debug(f"Loading image from : {self.img_path}")
        try:
            return gwy.load_gwy(file_path=self.img_path, channel=self.channel)
        except FileNotFoundError:
            LOGGER.error(f"File not found : {self.img_path}")
            raise

    def get_data(self) -> None:
        """Extract image, filepath and pixel to nm scaling value, and append these to the img_dic object."""
        suffix_to_loader = {
            ".spm": self.load_spm,
            ".jpk": self.load_jpk,
            ".ibw": self.load_ibw,
            ".gwy": self.load_gwy,
            ".topostats": self.load_topostats,
            ".asd": self.load_asd,
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
                    if suffix == ".topostats":
                        self.image, self.pixel_to_nm_scaling, self.img_dict = suffix_to_loader[suffix]()
                    else:
                        self.image, self.pixel_to_nm_scaling = suffix_to_loader[suffix]()
                except Exception as e:
                    if "Channel" in str(e) and "not found" in str(e):
                        LOGGER.warning(e)  # log the specific error message
                        LOGGER.warning(f"[{self.filename}] Channel {self.channel} not found, skipping image.")
                    else:
                        raise
                else:
                    if suffix == ".asd":
                        for index, frame in enumerate(self.image):
                            self._check_image_size_and_add_to_dict(image=frame, filename=f"{self.filename}_{index}")
                    else:
                        self._check_image_size_and_add_to_dict(image=self.image, filename=self.filename)
            else:
                raise ValueError(
                    f"File type {suffix} not yet supported. Please make an issue at \
                https://github.com/AFM-SPM/TopoStats/issues, or email topostats@sheffield.ac.uk to request support for \
                this file type."
                )

    def _check_image_size_and_add_to_dict(self, image: npt.NDArray, filename: str) -> None:
        """
        Check the image is above a minimum size in both dimensions.

        Images that do not meet the minimum size are not included for processing.

        Parameters
        ----------
        image : npt.NDArray
            An array of the extracted AFM image.
        filename : str
            The name of the file.
        """
        if image.shape[0] < self.MINIMUM_IMAGE_SIZE or image.shape[1] < self.MINIMUM_IMAGE_SIZE:
            LOGGER.warning(f"[{filename}] Skipping, image too small: {image.shape}")
        else:
            self.add_to_dict(image=image, filename=filename)
            LOGGER.debug(f"[{filename}] Image added to processing.")

    def add_to_dict(self, image: npt.NDArray, filename: str) -> None:
        """
        Add an image and metadata to the img_dict dictionary under the key filename.

        Adds the image and associated metadata such as any grain masks, and pixel to nanometere
        scaling factor to the img_dict dictionary which is used as a place to store the image
        information for processing.

        Parameters
        ----------
        image : npt.NDArray
            An array of the extracted AFM image.
        filename : str
            The name of the file.
        """
        self.img_dict[filename] = {
            "filename": filename,
            "img_path": self.img_path.with_name(filename),
            "pixel_to_nm_scaling": self.pixel_to_nm_scaling,
            "image_original": image,
            "image_flattened": None,
            "grain_masks": self.grain_masks,
            "grain_trace_data": self.grain_trace_data,
        }


def dict_to_hdf5(open_hdf5_file: h5py.File, group_path: str, dictionary: dict) -> None:
    """
    Recursively save a dictionary to an open hdf5 file.

    Parameters
    ----------
    open_hdf5_file : h5py.File
        An open hdf5 file object.
    group_path : str
        The path to the group in the hdf5 file to start saving data from.
    dictionary : dict
        A dictionary of the data to save.
    """
    for key, item in dictionary.items():
        # LOGGER.info(f"Saving key: {key}")

        if item is None:
            LOGGER.debug(f"Item '{key}' is None. Skipping.")
        # Make sure the key is a string
        key = str(key)

        # Check if the item is a known datatype
        # Ruff wants us to use the pipe operator here but it isn't supported by python 3.9
        if isinstance(item, (list, str, int, float, np.ndarray, Path, dict)):  # noqa: UP038
            # Lists need to be converted to numpy arrays
            if isinstance(item, list):
                item = np.array(item)
                open_hdf5_file[group_path + key] = item
            # Strings need to be encoded to bytes
            elif isinstance(item, str):
                open_hdf5_file[group_path + key] = item.encode("utf8")
            # Integers, floats and numpy arrays can be added directly to the hdf5 file
            # Ruff wants us to use the pipe operator here but it isn't supported by python 3.9
            elif isinstance(item, (int, float, np.ndarray)):  # noqa: UP038
                open_hdf5_file[group_path + key] = item
            # Path objects need to be encoded to bytes
            elif isinstance(item, Path):
                open_hdf5_file[group_path + key] = str(item).encode("utf8")
            # Dictionaries need to be recursively saved
            elif isinstance(item, dict):  # a sub-dictionary, so we need to recurse
                dict_to_hdf5(open_hdf5_file, group_path + key + "/", item)
        else:  # attempt to save an item that is not a numpy array or a dictionary
            try:
                open_hdf5_file[group_path + key] = item
            except Exception as e:
                LOGGER.debug(f"Cannot save key '{key}' to HDF5. Item type: {type(item)}. Skipping. {e}")


def hdf5_to_dict(open_hdf5_file: h5py.File, group_path: str) -> dict:
    """
    Read a dictionary from an open hdf5 file.

    Parameters
    ----------
    open_hdf5_file : h5py.File
        An open hdf5 file object.
    group_path : str
        The path to the group in the hdf5 file to start reading data from.

    Returns
    -------
    dict
        A dictionary of the hdf5 file data.
    """
    data_dict = {}
    for key, item in open_hdf5_file[group_path].items():
        LOGGER.debug(f"Loading hdf5 key: {key}")
        if isinstance(item, h5py.Group):
            LOGGER.debug(f" {key} is a group")
            data_dict[key] = hdf5_to_dict(open_hdf5_file, group_path + key + "/")
        # Decode byte strings to utf-8. The data type "O" is a byte string.
        elif isinstance(item, h5py.Dataset) and item.dtype == "O":
            LOGGER.debug(f" {key} is a byte string")
            data_dict[key] = item[()].decode("utf-8")
            LOGGER.debug(f" {key} type: {type(data_dict[key])}")
        else:
            LOGGER.debug(f" {key} is other type of dataset")
            data_dict[key] = item[()]
            LOGGER.debug(f" {key} type: {type(data_dict[key])}")
    return data_dict


def save_topostats_file(output_dir: Path, filename: str, topostats_object: dict) -> None:
    """
    Save a topostats dictionary object to a .topostats (hdf5 format) file.

    Parameters
    ----------
    output_dir : Path
        Directory to save the .topostats file in.
    filename : str
        File name of the .topostats file.
    topostats_object : dict
        Dictionary of the topostats data to save. Must include a flattened image and pixel to nanometre scaling
        factor. May also include grain masks.
    """
    LOGGER.info(f"[{filename}] : Saving image to .topostats file")

    if ".topostats" not in filename:
        save_file_path = output_dir / f"{filename}.topostats"
    else:
        save_file_path = output_dir / filename

    with h5py.File(save_file_path, "w") as f:
        # It may be possible for topostats_object["image_flattened"] to be None.
        # Make sure that this is not the case.
        if topostats_object["image_flattened"] is not None:
            topostats_object["topostats_file_version"] = 0.2
            # Rename the key to "image" for backwards compatibility
            topostats_object["image"] = topostats_object.pop("image_flattened")

            # Recursively save the topostats object dictionary to the .topostats file
            dict_to_hdf5(open_hdf5_file=f, group_path="/", dictionary=topostats_object)

        else:
            raise ValueError(
                "TopoStats object dictionary does not contain an 'image_flattened'. \
                 TopoStats objects must be saved with a flattened image."
            )


def save_pkl(outfile: Path, to_pkl: dict) -> None:
    """
    Pickle objects for working with later.

    Parameters
    ----------
    outfile : Path
        Path and filename to save pickle to.
    to_pkl : dict
        Object to be picled.
    """
    with outfile.open(mode="wb", encoding=None) as f:
        pkl.dump(to_pkl, f)


def load_pkl(infile: Path) -> Any:
    """
    Load data from a pickle.

    Parameters
    ----------
    infile : Path
        Path to a valid pickle.

    Returns
    -------
    dict:
        Dictionary of generated images.

    Examples
    --------
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
        return pkl.load(f)  # noqa: S301


def dict_to_json(data: dict, output_dir: str | Path, filename: str | Path, indent: int = 4) -> None:
    """
    Write a dictionary to a JSON file at the specified location with the given name.

    NB : The `NumpyEncoder` class is used as the default encoder to ensure Numpy dtypes are written as strings (they are
         not serialisable to JSON using the default JSONEncoder).

    Parameters
    ----------
    data : dict
        Data as a dictionary that is to be written to file.
    output_dir : str | Path
        Directory the file is to be written to.
    filename : str | Path
        Name of output file.
    indent : int
        Spaces to indent JSON with, default is 4.
    """
    output_file = output_dir / filename
    with output_file.open("w") as f:
        json.dump(data, f, indent=indent, cls=NumpyEncoder)
