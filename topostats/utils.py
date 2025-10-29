"""Utilities."""

import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.ndimage import convolve

from topostats.logs.logs import LOGGER_NAME

LOGGER = logging.getLogger(LOGGER_NAME)


COLUMN_SETS = {
    "grainstats": (
        "image",
        "basename",
        "grain_number",
        "class_number",
        "class_name",
        "subgrain_number",
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


def get_mask(image: npt.NDArray, thresh: float, img_name: str = None) -> npt.NDArray:
    """
    Calculate a mask for pixels that exceed the threshold.

    Parameters
    ----------
    image : np.array
        Numpy array representing image.
    thresh : float
        A float representing the threshold.
    img_name : str
        Name of image being processed.

    Returns
    -------
    npt.NDArray
        Numpy array of image with objects coloured.
    """
    if thresh > 0:
        LOGGER.debug(f"[{img_name}] : Masking (above) threshold: {thresh}")
        return image > thresh
    LOGGER.debug(f"[{img_name}] : Masking (below) threshold: {thresh}")
    return image < thresh


def create_empty_dataframe(column_set: str = "grainstats") -> pd.DataFrame:
    """
    Create an empty data frame for returning when no results are found.

    Parameters
    ----------
    column_set : str
        The name of the set of columns for the empty dataframe.

    Returns
    -------
    pd.DataFrame
        Empty Pandas DataFrame.
    """
    return pd.DataFrame(columns=COLUMN_SETS[column_set])


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
        If True, increments the value of each coord to show order.

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
