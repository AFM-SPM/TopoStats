"""Utilities."""

from __future__ import annotations

import logging
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from pprint import pformat

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.ndimage import convolve

from topostats.logs.logs import LOGGER_NAME
from topostats.thresholds import threshold

LOGGER = logging.getLogger(LOGGER_NAME)


COLUMN_SETS = {
    "grainstats": (
        "image",
        "basename",
        "grain_number",
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
    ),
    "disordered_tracing_statistics": (
        "image",
        "basename",
        "threshold",
        "grain_number",
        "index",
        "branch_distance",
        "branch_type",
        "connected_segments",
        "mean_pixel_value",
        "stdev_pixel_value",
        "min_value",
        "median_value",
        "middle_value",
    ),
    "mol_statistics": (
        "image",
        "threshold",
        "basename",
        "grain_number",
        "molecule_number",
        "circular",
        "topology",
        "processing",
    ),
}


def convert_path(path: str | Path) -> Path:
    """
    Ensure path is Path object.

    Parameters
    ----------
    path : str | Path
        Path to be converted.

    Returns
    -------
    Path
        Pathlib object of path.
    """
    return Path().cwd() if path == "./" else Path(path).expanduser()


def update_config(config: dict, args: dict | Namespace) -> dict:
    """
    Update the configuration with any arguments.

    Parameters
    ----------
    config : dict
        Dictionary of configuration (typically read from YAML file specified with '-c/--config <filename>').
    args : Namespace
        Command line arguments.

    Returns
    -------
    dict
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
                LOGGER.debug(f"Updated config config[{arg_key}] : {original_value} > {arg_value} ")
    if "base_dir" in config.keys():
        config["base_dir"] = convert_path(config["base_dir"])
    if "output_dir" in config.keys():
        config["output_dir"] = convert_path(config["output_dir"])
    return config


def update_plotting_config(plotting_config: dict) -> dict:
    """
    Update the plotting config for each of the plots in plot_dict.

    Ensures that each entry has all the plotting configuration values that are needed.

    Parameters
    ----------
    plotting_config : dict
        Plotting configuration to be updated.

    Returns
    -------
    dict
        Updated plotting configuration.
    """
    main_config = plotting_config.copy()
    for opt in ["plot_dict", "run"]:
        main_config.pop(opt)
    LOGGER.debug(
        f"Main plotting options that need updating/adding to plotting dict :\n{pformat(main_config, indent=4)}"
    )
    for image, options in plotting_config["plot_dict"].items():
        main_config_temp = main_config.copy()
        LOGGER.debug(f"Dictionary for image : {image}")
        LOGGER.debug(f"{pformat(options, indent=4)}")
        # First update options with values that exist in main_config
        # We must however be careful not to update the colourmap for diagnostic traces
        if (
            not plotting_config["plot_dict"][image]["core_set"]
            and "mask_cmap" in plotting_config["plot_dict"][image].keys()
        ):
            main_config_temp.pop("mask_cmap")
        plotting_config["plot_dict"][image] = update_config(options, main_config_temp)
        LOGGER.debug(f"Updated values :\n{pformat(plotting_config['plot_dict'][image])}")
        # Then combine the remaining key/values we need from main_config that don't already exist
        for key_main, value_main in main_config_temp.items():
            if key_main not in plotting_config["plot_dict"][image]:
                plotting_config["plot_dict"][image][key_main] = value_main
        LOGGER.debug(f"After adding missing configuration options :\n{pformat(plotting_config['plot_dict'][image])}")
        # Make it so that binary images do not have the user-defined z-scale
        # applied, but non-binary images do.
        if plotting_config["plot_dict"][image]["image_type"] == "binary":
            plotting_config["plot_dict"][image]["zrange"] = [None, None]

    return plotting_config


def _get_mask(image: npt.NDArray, thresh: float, threshold_direction: str, img_name: str = None) -> npt.NDArray:
    """
    Calculate a mask for pixels that exceed the threshold.

    Parameters
    ----------
    image : np.array
        Numpy array representing image.
    thresh : float
        A float representing the threshold.
    threshold_direction : str
        A string representing the direction that should be thresholded. ("above", "below").
    img_name : str
        Name of image being processed.

    Returns
    -------
    npt.NDArray
        Numpy array of image with objects coloured.
    """
    if threshold_direction == "above":
        LOGGER.debug(f"[{img_name}] : Masking (above) Threshold: {thresh}")
        return image > thresh
    LOGGER.debug(f"[{img_name}] : Masking (below) Threshold: {thresh}")
    return image < thresh
    # LOGGER.fatal(f"[{img_name}] : Threshold direction invalid: {threshold_direction}")


def get_mask(image: npt.NDArray, thresholds: dict, img_name: str = None) -> npt.NDArray:
    """
    Mask data that should not be included in flattening.

    Parameters
    ----------
    image : npt.NDArray
        2D Numpy array of the image to have a mask derived for.
    thresholds : dict
        Dictionary of thresholds, at a bare minimum must have key 'below' with an associated value, second key is
        to have an 'above' threshold.
    img_name : str
        Image name that is being masked.

    Returns
    -------
    npt.NDArray
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
def get_thresholds(  # noqa: C901
    image: npt.NDArray,
    threshold_method: str,
    otsu_threshold_multiplier: float = None,
    threshold_std_dev: dict = None,
    absolute: dict = None,
    **kwargs,
) -> dict:
    """
    Obtain thresholds for masking data points.

    Parameters
    ----------
    image : npt.NDArray
        2D Numpy array of image to be masked.
    threshold_method : str
        Method for thresholding, 'otsu', 'std_dev' or 'absolute' are valid options.
    otsu_threshold_multiplier : float
        Scaling value for Otsu threshold.
    threshold_std_dev : dict
        Dict of above and below thresholds for the standard deviation method.
    absolute : tuple
        Dict of below and above thresholds.
    **kwargs :
        Dictionary passed to 'topostats.threshold(**kwargs)'.

    Returns
    -------
    dict
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


def create_empty_dataframe(column_set: str = "grainstats", index_col: str = "grain_number") -> pd.DataFrame:
    """
    Create an empty data frame for returning when no results are found.

    Parameters
    ----------
    column_set : str
        The name of the set of columns for the empty dataframe.
    index_col : str
        Column to set as index of empty dataframe.

    Returns
    -------
    pd.DataFrame
        Empty Pandas DataFrame.
    """
    empty_df = pd.DataFrame(columns=COLUMN_SETS[column_set])
    return empty_df.set_index(index_col)


def bound_padded_coordinates_to_image(coordinates: npt.NDArray, padding: int, image_shape: tuple) -> tuple:
    """
    Ensure the padding of coordinates points does not fall outside of the image shape.

    This function is primarily used in the dnaTrace.get_fitted_traces() method which aims to adjust the points of a
    skeleton to sit on the highest points of a traced molecule. In order to do so it takes the ordered skeleton, which
    may not lie on the highest points as it is generated from a binary mask that is unaware of the heights, and then
    defines a padded boundary of 3nm profile perpendicular to the backbone of the DNA (which at this point is the
    skeleton based on a mask). Each point along the skeleton therefore needs padding by a minimum of 2 pixels (in this
    case each pixel equates to a cell in a NumPy array). If a point is within 2 pixels (i.e. 2 cells) of the border then
    we can not pad beyond this region, we have to stop at the edge of the image and so the coordinates is adjusted such
    that the padding will lie on the edge of the image/array.

    Parameters
    ----------
    coordinates : npt.NDArray
        Coordinates of a point on the mask based skeleton.
    padding : int
        Number of pixels/cells to pad around the point.
    image_shape : tuple
        The shape of the original image from which the pixel is obtained.

    Returns
    -------
    tuple
        Returns a tuple of coordinates that ensure that when the point is padded by the noted padding width in
        subsequent calculations it will not be outside of the image shape.
    """
    # Calculate the maximum row and column indexes
    max_row = image_shape[0] - 1
    max_col = image_shape[1] - 1
    row_coord, col_coord = coordinates

    def check(coord: npt.NDArray, max_val: int, padding: int) -> npt.NDArray:
        """
        Check coordinates are within the bounds of the padding.

        Parameters
        ----------
        coord : npt.NDArray
            Coordinates (length = 2).
        max_val : int
            Maximum width in the dimension being checked (max_row or max_col).
        padding : int
            Padding used in the image.

        Returns
        -------
        npt.NDArray
            Coordinates adjusted for padding.
        """
        if coord - padding < 0:
            coord = padding
        elif coord + padding > max_val:
            coord = max_val - padding
        return coord

    return check(row_coord, max_row, padding), check(col_coord, max_col, padding)


def convolve_skeleton(skeleton: npt.NDArray) -> npt.NDArray:
    """
    Convolve skeleton with a 3x3 kernel.

    This produces an array where the branches of the skeleton are denoted with '1', endpoints are denoted as '2', and
    pixels at nodes as '3'.

    Parameters
    ----------
    skeleton : npt.NDArray
        Single pixel thick binary trace(s) within an array.

    Returns
    -------
    npt.NDArray
        The skeleton (=1) with endpoints (=2), and crossings (=3) highlighted.
    """
    conv = convolve(skeleton.astype(np.int32), np.ones((3, 3)))
    conv[skeleton == 0] = 0  # remove non-skeleton points
    conv[conv == 3] = 1  # skelly = 1
    conv[conv > 3] = 3  # nodes = 3
    return conv


class ResolutionError(Exception):
    """Raised when the image resolution is too small for accuurate tracing."""

    pass  # pylint: disable=unnecessary-pass


def coords_2_img(coords, image, ordered=False) -> np.ndarray:
    """
    Convert coordinates to a binary image.

    Parameters
    ----------
    coords : np.ndarray
        An array of 2xN integer coordinates.
    image : np.ndarray
        An MxL array to assign the above coordinates onto.
    ordered : bool, optional
        If True, incremements the value of each coord to show order.

    Returns
    -------
    np.ndarray
        An array the same shape as 'image' with the coordinates highlighted.
    """
    comb = np.zeros_like(image)
    if ordered:
        comb[coords[:, 0].astype(np.int32), coords[:, 1].astype(np.int32)] = np.arange(1, len(coords) + 1)
    else:
        coords = coords[
            (coords[:, 0] < image.shape[0]) & (coords[:, 1] < image.shape[1]) & (coords[:, 0] > 0) & (coords[:, 1] > 0)
        ]
        comb[np.floor(coords[:, 0]).astype(np.int32), np.floor(coords[:, 1]).astype(np.int32)] = 1
    return comb
