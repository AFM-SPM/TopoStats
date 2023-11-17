"""Utilities."""
from __future__ import annotations
from argparse import Namespace
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

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


def convert_path(path: str | Path) -> Path:
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


def update_config(config: dict, args: dict | Namespace) -> dict:
    """Update the configuration with any arguments.

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
    """Update the plotting config for each of the plots in plot_dict.

    Ensures that each entry has all the plotting configuration values that are needed.
    """
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


def _get_mask(image: np.ndarray, thresholds: dict, threshold_direction: str, img_name: str = None) -> np.ndarray:
    """Calculate a mask for pixels that exceed the threshold.

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
    minimum = thresholds["minimum"]
    maximum = thresholds["maximum"]

    if threshold_direction == "above":
        LOGGER.info(f"[{img_name}] : Masking (above) Threshold: {thresholds}")
        minimum_thresholded_mask = image > minimum
        maximum_thresholded_mask = image < maximum
        return minimum_thresholded_mask & maximum_thresholded_mask
    LOGGER.info(f"[{img_name}] : Masking (below) Threshold: {thresholds}")
    minimum_thresholded_mask = image < minimum
    maximum_thresholded_mask = image > maximum
    return minimum_thresholded_mask & maximum_thresholded_mask
    # LOGGER.fatal(f"[{img_name}] : Threshold direction invalid: {threshold_direction}")


def get_and_combine_directional_masks(image: np.ndarray, thresholds: dict, img_name: str = None) -> np.ndarray:
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
    # If both above and below thresholds are available
    if thresholds["above"] is not None and thresholds["below"] is not None:
        mask_above = _get_mask(image, thresholds=thresholds["above"], threshold_direction="above", img_name=img_name)
        mask_below = _get_mask(image, thresholds=thresholds["below"], threshold_direction="below", img_name=img_name)
        # Combine the masks
        return mask_above + mask_below
    # If only above threshold is available
    if thresholds["above"] is not None:
        return _get_mask(image, thresholds=thresholds["above"], threshold_direction="above", img_name=img_name)
    # If only below threshold is available
    return _get_mask(image, thresholds=thresholds["below"], threshold_direction="below", img_name=img_name)


# pylint: disable=unused-argument
def get_thresholds(  # noqa: C901
    image: np.ndarray,
    threshold_method: str,
    otsu_threshold_multiplier: float = None,
    threshold_std_dev: dict = None,
    absolute: dict = None,
    **kwargs,
) -> dict:
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
        thresholds["below"], thresholds["above"] = _get_otsu_threshold_min_max(
            otsu_threshold_multiplier=otsu_threshold_multiplier, image=image
        )

    elif threshold_method == "std_dev":
        thresholds["below"], thresholds["above"] = _get_std_dev_threshold_min_max(
            threshold_std_dev=threshold_std_dev, image=image
        )

    elif threshold_method == "absolute":
        thresholds["below"], thresholds["above"] = _get_absolute_threshold_min_max(absolute=absolute)

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


# pylint: disable=too-many-branches
def _get_std_dev_threshold_min_max(threshold_std_dev: dict, image: np.ndarray) -> tuple:
    """Get the minimum and maximum values for the standard deviation threshold.

    Parameters
    ----------
    threshold_std_dev : dict
        Dictionary of standard deviation thresholds. Keys are "above" and "below".
    image : np.ndarray
        2D Numpy array of image to base the standard deviation threshold on.

    Returns
    -------
    tuple
        Tuple of minimum and maximum values for thresholding.
    """
    try:
        if threshold_std_dev["below"] is not None:
            if threshold_std_dev["below"][0] is not None:
                minimum = threshold(image, method="mean") - threshold_std_dev["below"][0] * np.nanstd(image)
            else:
                minimum = np.Infinity
            if threshold_std_dev["below"][1] is not None:
                maximum = threshold(image, method="mean") - threshold_std_dev["below"][1] * np.nanstd(image)
            else:
                maximum = -np.Infinity
            thresholds_below = {"minimum": minimum, "maximum": maximum}
        else:
            thresholds_below = None

        if threshold_std_dev["above"] is not None:
            if threshold_std_dev["above"][0] is not None:
                minimum = threshold(image, method="mean") + threshold_std_dev["above"][0] * np.nanstd(image)
            else:
                minimum = -np.Infinity
            if threshold_std_dev["above"][1] is not None:
                maximum = threshold(image, method="mean") + threshold_std_dev["above"][1] * np.nanstd(image)
            else:
                maximum = np.Infinity
            thresholds_above = {"minimum": minimum, "maximum": maximum}
        else:
            thresholds_above = None

        return thresholds_below, thresholds_above
    except TypeError as typeerror:
        raise typeerror


def _get_absolute_threshold_min_max(absolute: dict) -> tuple:
    """Get the minimum and maximum values for the absolute threshold.

    Parameters
    ----------
    absolute : dict
        Dictionary of absolute thresholds. Keys are "above" and "below".

    Returns
    -------
    tuple
        Tuple of minimum and maximum values for thresholding.
    """
    if absolute["below"] is not None:
        if absolute["below"][0] is not None:
            minimum = absolute["below"][0]
        else:
            minimum = np.Infinity
        if absolute["below"][1] is not None:
            maximum = absolute["below"][1]
        else:
            maximum = -np.Infinity
        thresholds_below = {"minimum": minimum, "maximum": maximum}
    else:
        thresholds_below = None

    if absolute["above"] is not None:
        if absolute["above"][0] is not None:
            minimum = absolute["above"][0]
        else:
            minimum = -np.Infinity
        if absolute["above"][1] is not None:
            maximum = absolute["above"][1]
        else:
            maximum = np.Infinity
        thresholds_above = {"minimum": minimum, "maximum": maximum}
    else:
        thresholds_above = None

    return thresholds_below, thresholds_above


def _get_otsu_threshold_min_max(otsu_threshold_multiplier: float, image: np.ndarray) -> tuple:
    """Get the minimum and maximum threshold values for the otsu threshold.

    Parameters
    ----------
    otsu_threshold_multiplier : float
        Multiplier for the otsu threshold.
    image : np.ndarray
        2D Numpy array of image to base the otsu threshold on.

    Returns
    -------
    tuple
        Tuple of minimum and maximum values for thresholding.
    """
    thresholds_below = None
    thresholds_above = {
        "minimum": threshold(image, method="otsu", otsu_threshold_multiplier=otsu_threshold_multiplier),
        "maximum": np.Infinity,
    }

    return thresholds_below, thresholds_above


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
    return empty_df.set_index(index)
