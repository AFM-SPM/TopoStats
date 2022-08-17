"""Utilities"""
from argparse import Namespace
import logging
from pathlib import Path
from typing import Union, List, Dict
from collections import defaultdict

import numpy as np
import pandas as pd

from topostats.thresholds import threshold
from topostats.logs.logs import LOGGER_NAME

LOGGER = logging.getLogger(LOGGER_NAME)


ALL_STATISTICS_COLUMNS = (
    "Molecule Number",
    "centre_x",
    "centre_y",
    "radius_min",
    "radius_max",
    "radius_mean",
    "radius_median",
    "height_min",
    "height_max",
    "height_median",
    "height_mean",
    "volume",
    "area",
    "area_cartesian_bbox",
    "smallest_bounding_width",
    "smallest_bounding_length",
    "smallest_bounding_area",
    "aspect_ratio",
    "Contour Lengths",
    "Circular",
    "End to End Distance",
    "Image Name",
    "Basename",
)


def convert_path(path: Union[str, Path]) -> Path:
    """Ensure path is Path object.

    Parameters
    ----------
    path: Union[str, Path]
        Path to be converted.

    Returns
    -------
    Path
        pathlib Path
    """
    return Path().cwd() if path == "./" else Path(path).expanduser()


def find_images(base_dir: Union[str, Path] = None, file_ext: str = ".spm") -> List:
    """Scan the specified directory for images with the given file extension.

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


def get_out_path(
    image_path: Union[str, Path] = None, base_dir: Union[str, Path] = None, output_dir: Union[str, Path] = None
) -> Path:
    """Replaces the base directory part of the image path with the output directory.

    Parameters
    ----------
    image_path: Union[str, Path]
        The path of the current image.
    base_dir: Union[str, Path]
        Directory to recursively search for files, if not specified the current directory is scanned.
    output_dir: Union[str, Path]
        The output directory specified in the configuration file.

    Returns
    -------
    Path
        The output path that mirrors the input path structure.
    """
    pathparts = list(image_path.parts)
    inparts = list(base_dir.parts)
    return output_dir / Path(*pathparts[len(inparts) :])


def update_config(config: dict, args: Union[dict, Namespace]) -> Dict:
    """Update the configuration with any arguments

    Parameters
    ----------
    config: dict
        Dictionary of configuration (typically read from YAML file specified with '-c/--config <filename>')
    args: Namespace
        Command line arguments
    Returns
    -------
    Dict
        Dictionary updated with command arguments.
    """
    args = vars(args) if isinstance(args, Namespace) else args

    config_keys = config.keys()
    for arg_key, arg_value in args.items():
        if isinstance(arg_value, dict):
            update_config(config, arg_value)
        else:
            if arg_key in config_keys and arg_value is not None:
                original_value = config[arg_key]
                config[arg_key] = arg_value
                LOGGER.info(f"Updated config config[{arg_key}] : {original_value} > {arg_value} ")
    config["base_dir"] = convert_path(config["base_dir"])
    config["output_dir"] = convert_path(config["output_dir"])
    return config


def _get_mask(image: np.ndarray, threshold: float, threshold_direction: str, img_name: str = None) -> np.ndarray:
    """Calculate a mask for pixels that exceed the threshold

    Parameters
    ----------
    image: np.array
        Numpy array representing image.
    threshold: float
        A float representing the threshold
    threshold_direction: str
        A string representing the direction that should be thresholded. ("above", "below")
    img_name: str
        Name of image being processed

    Returns
    -------
    np.array
        Numpy array of image with objects coloured.
    """
    if threshold_direction == "upper":
        LOGGER.info(f"[{img_name}] : Masking (upper) Threshold: {threshold}")
        return image > threshold
    if threshold_direction == "lower":
        LOGGER.info(f"[{img_name}] : Masking (lower) Threshold: {threshold}")
        return image < threshold
    # LOGGER.fatal(f"[{img_name}] : Threshold direction invalid: {threshold_direction}")


def get_mask(image: np.ndarray, thresholds: dict, img_name: str = None) -> np.ndarray:
    """Mask data that should not be included in flattening.

    Parameters
    ----------
    image: np.ndarray
        2D Numpy array of the image to have a mask derived for.

    thresholds: dict
        Dictionary of thresholds, at a bare minimum must have key 'lower' with an associated value, second key is
        to have an 'upper' threshold.
    img_name: str
        Image name that is being masked.

    Returns
    -------
    np.ndarray
        2D Numpy boolean array of points to mask.
    """
    # Both thresholds are applicable
    if "lower" in thresholds and "upper" in thresholds:
        mask_upper = _get_mask(image, threshold=thresholds["upper"], threshold_direction="upper", img_name=img_name)
        mask_lower = _get_mask(image, threshold=thresholds["lower"], threshold_direction="lower", img_name=img_name)
        # Masks are combined to remove both the extreme high and extreme low data points.
        return mask_upper + mask_lower
    # Only lower threshold is applicable
    if "lower" in thresholds:
        return _get_mask(image, threshold=thresholds["lower"], threshold_direction="lower", img_name=img_name)
    # Only upper threshold is applicable
    return _get_mask(image, threshold=thresholds["upper"], threshold_direction="upper", img_name=img_name)


def get_thresholds(
    image: np.ndarray,
    threshold_method: str,
    otsu_threshold_multiplier: float = None,
    threshold_std_dev: float = None,
    absolute: tuple = None,
    **kwargs,
) -> Dict:
    """Obtain thresholds for masking data points.

    Parameters
    ----------
    image : np.ndarray
        2D Numpy array of image to be masked
    threshold_method : str
        Method for thresholding, 'otsu', 'std_dev' or 'absolute' are valid options.
    threshold_std_dev : float
        Scaling of standard deviation from the mean for lower and upper thresholds.
    absolute : tuple
        Tuple of lower and upper thresholds.
    **kwargs:

    Returns
    -------
    Dict
        Dictionary of thresholds, contains keys 'lower' and optionally 'upper'.
    """
    thresholds = defaultdict()
    if threshold_method == "otsu":
        thresholds["upper"] = threshold(image, method="otsu", otsu_threshold_multiplier=otsu_threshold_multiplier)
    elif threshold_method == "std_dev":
        try:
            thresholds["lower"] = threshold(image, method="mean") - threshold_std_dev * np.nanstd(image)
            thresholds["upper"] = threshold(image, method="mean") + threshold_std_dev * np.nanstd(image)
        except TypeError as typeerror:
            raise typeerror
    elif threshold_method == "absolute":
        if absolute[0] is not None:
            thresholds["lower"] = absolute[0]
        if absolute[1] is not None:
            thresholds["upper"] = absolute[1]
    else:
        if not isinstance(threshold_method, str):
            raise TypeError(
                f"threshold_method ({threshold_method}) should be a string. Valid values : 'otsu' 'std_dev' 'absolute'"
            )
        if threshold_method not in ["otsu", "std_dev", "absolute"]:
            raise ValueError(
                f"threshold_method ({threshold_method}) is invalid. Valid values : 'otsu' 'std_dev' 'absolute'"
            )
    return thresholds


def create_empty_dataframe(columns: set = ALL_STATISTICS_COLUMNS) -> pd.DataFrame:
    """Create an empty data frame for returning when no results are found.

    Parameters
    ----------
    columns: list
        Columns of the empty dataframe.

    Returns
    -------
    pd.DataFrame
        Empty Pandas DataFrame.
    """
    return pd.DataFrame([np.repeat(np.nan, len(columns))], columns=columns)


def folder_grainstats(output_dir: Union[str, Path], base_dir: Union[str, Path], all_stats_df: pd.DataFrame) -> None:
    """Creates saves a data frame of grain and tracing statictics at the folder level.

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
    dirs = set(all_stats_df["Basename"].values)
    for _dir in dirs:
        out_path = get_out_path(Path(_dir), base_dir, output_dir)
        all_stats_df[all_stats_df["Basename"] == _dir].to_csv(out_path / "Processed" / "folder_grainstats.csv")
        LOGGER.info(f"Folder-wise statistics saved to: {str(out_path)}/Processed/folder_grainstats.csv")
