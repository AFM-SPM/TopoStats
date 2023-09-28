"""Utilities"""
from argparse import Namespace
import logging
from pathlib import Path
from typing import Union, Dict
from collections import defaultdict

import json
import numpy as np
import pandas as pd
from scipy.ndimage import convolve

from topostats.thresholds import threshold
from topostats.logs.logs import LOGGER_NAME

LOGGER = logging.getLogger(LOGGER_NAME)


ALL_STATISTICS_COLUMNS = (
    "image",
    "basename",
    "molecule_number",
    "area",
    "area_cartesian_bbox",
    "aspect_ratio",
    "bending_angle",
    "centre_x",
    "centre_y",
    "circular",
    "contour_length",
    "end_to_end_distance",
    "height_max",
    "height_mean",
    "height_median",
    "height_min",
    "max_feret",
    "min_feret",
    "radius_max",
    "radius_mean",
    "radius_median",
    "radius_min",
    "smallest_bounding_area",
    "smallest_bounding_length",
    "smallest_bounding_width",
    "threshold",
    "volume",
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
    if "base_dir" in config.keys():
        config["base_dir"] = convert_path(config["base_dir"])
    if "output_dir" in config.keys():
        config["output_dir"] = convert_path(config["output_dir"])
    return config


def update_plotting_config(plotting_config: dict) -> dict:
    """Update the plotting config for each of the plots in plot_dict to ensure that each
    entry has all the plotting configuration values that are needed."""

    main_config = plotting_config.copy()
    for opt in ["plot_dict", "run"]:
        main_config.pop(opt)
    for image, options in plotting_config["plot_dict"].items():
        plotting_config["plot_dict"][image] = {**options, **main_config}
        # Make it so that binary images do not have the user-defined z-scale
        # applied, but non-binary images do.
        if plotting_config["plot_dict"][image]["image_type"] == "binary":
            plotting_config["plot_dict"][image]["zrange"] = [None, None]

    return plotting_config


def _get_mask(image: np.ndarray, thresh: float, threshold_direction: str, img_name: str = None) -> np.ndarray:
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
    if threshold_direction == "above":
        LOGGER.info(f"[{img_name}] : Masking (above) Threshold: {thresh}")
        return image > thresh
    LOGGER.info(f"[{img_name}] : Masking (below) Threshold: {thresh}")
    return image < thresh
    # LOGGER.fatal(f"[{img_name}] : Threshold direction invalid: {threshold_direction}")


def get_mask(image: np.ndarray, thresholds: dict, img_name: str = None) -> np.ndarray:
    """Mask data that should not be included in flattening.

    Parameters
    ----------
    image: np.ndarray
        2D Numpy array of the image to have a mask derived for.

    thresholds: dict
        Dictionary of thresholds, at a bare minimum must have key 'below' with an associated value, second key is
        to have an 'above' threshold.
    img_name: str
        Image name that is being masked.

    Returns
    -------
    np.ndarray
        2D Numpy boolean array of points to mask.
    """
    # Both thresholds are applicable
    if "below" in thresholds and "above" in thresholds:
        mask_above = _get_mask(image, thresh=thresholds["above"], threshold_direction="above", img_name=img_name)
        mask_below = _get_mask(image, thresh=thresholds["below"], threshold_direction="below", img_name=img_name)
        # Masks are combined to remove both the extreme high and extreme low data points.
        return mask_above + mask_below
    # Only below threshold is applicable
    if "below" in thresholds:
        return _get_mask(image, thresh=thresholds["below"], threshold_direction="below", img_name=img_name)
    # Only above threshold is applicable
    return _get_mask(image, thresh=thresholds["above"], threshold_direction="above", img_name=img_name)


# pylint: disable=unused-argument
def get_thresholds(
    image: np.ndarray,
    threshold_method: str,
    otsu_threshold_multiplier: float = None,
    threshold_std_dev: dict = None,
    absolute: dict = None,
    **kwargs,
) -> Dict:
    """Obtain thresholds for masking data points.

    Parameters
    ----------
    image : np.ndarray
        2D Numpy array of image to be masked
    threshold_method : str
        Method for thresholding, 'otsu', 'std_dev' or 'absolute' are valid options.
    threshold_std_dev : dict
        Dict of above and below thresholds for the standard deviation method.
    absolute : tuple
        Dict of below and above thresholds.
    **kwargs:

    Returns
    -------
    Dict
        Dictionary of thresholds, contains keys 'below' and optionally 'above'.
    """
    thresholds = defaultdict()
    if threshold_method == "otsu":
        thresholds["above"] = threshold(image, method="otsu", otsu_threshold_multiplier=otsu_threshold_multiplier)
    elif threshold_method == "std_dev":
        try:
            if threshold_std_dev["below"] is not None:
                thresholds["below"] = threshold(image, method="mean") - threshold_std_dev["below"] * np.nanstd(image)
            if threshold_std_dev["above"] is not None:
                thresholds["above"] = threshold(image, method="mean") + threshold_std_dev["above"] * np.nanstd(image)
        except TypeError as typeerror:
            raise typeerror
    elif threshold_method == "absolute":
        if absolute["below"] is not None:
            thresholds["below"] = absolute["below"]
        if absolute["above"] is not None:
            thresholds["above"] = absolute["above"]
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


def create_empty_dataframe(columns: set = ALL_STATISTICS_COLUMNS, index: tuple = "molecule_number") -> pd.DataFrame:
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
    empty_df = pd.DataFrame(columns=columns)
    empty_df = empty_df.set_index(index)
    return pd.DataFrame([np.repeat(np.nan, len(columns))], columns=columns)


def folder_grainstats(output_dir: Union[str, Path], base_dir: Union[str, Path], all_stats_df: pd.DataFrame) -> None:
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
    dirs = set(all_stats_df["Basename"].values)
    try:
        for _dir in dirs:
            out_path = get_out_path(Path(_dir), base_dir, output_dir)
            all_stats_df[all_stats_df["Basename"] == _dir].to_csv(out_path / "processed" / "folder_grainstats.csv")
            LOGGER.info(f"Folder-wise statistics saved to: {str(out_path)}/folder_grainstats.csv")
    except TypeError:
        LOGGER.info("Unable to generate folderwise statistics as 'all_stats_df' is empty")


def convolve_skelly(skeleton) -> np.ndarray:
    """Convolves the skeleton with a 3x3 ones kernel to produce an array
    of the skeleton as 1, endpoints as 2, and nodes as 3.

    Parameters
    ----------
    skeleton: np.ndarray
        Single pixel thick binary trace(s) within an array.

    Returns
    -------
    np.ndarray
        The skeleton (=1) with endpoints (=2), and crossings (=3) highlighted.
    """
    conv = convolve(skeleton.astype(np.int32), np.ones((3, 3)))
    conv[skeleton == 0] = 0  # remove non-skeleton points
    conv[conv == 3] = 1  # skelly = 1
    conv[conv > 3] = 3  # nodes = 3
    return conv


class ResolutionError(Exception):
    "Raised when the image resolution is too small for accuurate tracing."
    pass


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
