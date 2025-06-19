"""Function for calculating statistics about a whole image, for example number of grains or surface roughness."""

import logging

import numpy as np
import pandas as pd

from topostats.logs.logs import LOGGER_NAME

LOGGER = logging.getLogger(LOGGER_NAME)


def image_statistics(
    image: np.ndarray,
    filename: str,
    pixel_to_nm_scaling: float,
    results_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate statistics pertaining to the whole image.

    Calculates the size of the image in pixels and metres, the root-mean-squared roughness and the grains per metre
    squared.

    Parameters
    ----------
    image : np.ndarray
        Numpy 2D image array of the image to calculate stats for.
    filename : str
        The name of the file being processed.
    pixel_to_nm_scaling : float
        Float of the scaling factor between pixels and nanometres.
    results_df : pd.DataFrame
        Pandas DataFrame of statistics pertaining to individual grains including from grainstats and
        dna tracing.

    Returns
    -------
    dict
        Dictionary of image statistics.
    """
    image_stats = {
        "image": filename,
        "image_size_x_m": None,
        "image_size_y_m": None,
        "image_area_m2": None,
        "image_size_x_px": image.shape[1],
        "image_size_y_px": image.shape[0],
        "image_area_px2": None,
        "grains_number_above": None,
        "grains_per_m2_above": None,
        "grains_number_below": None,
        "grains_per_m2_below": None,
        "rms_roughness": None,
    }

    # Calculate dimensions of the image
    image_stats["image_size_x_m"] = image.shape[1] * pixel_to_nm_scaling * 1e-9
    image_stats["image_size_y_m"] = image.shape[0] * pixel_to_nm_scaling * 1e-9
    image_stats["image_area_m2"] = image_stats["image_size_x_m"] * image_stats["image_size_y_m"]
    image_stats["image_area_px2"] = image_stats["image_size_x_px"] * image_stats["image_size_y_px"]

    # Calculate the RMS roughness of the sample on the flattened image.
    image_stats["rms_roughness"] = roughness_rms(image=image) * 1e-9

    # Calculate image stats relating to grain statistics. Note that the existence of any of these stats
    # is not guaranteed
    try:
        image_stats["grains_number_below"] = results_df["threshold"].value_counts().get("below", 0)
        image_stats["grains_per_m2_below"] = image_stats["grains_number_below"] / image_stats["image_area_m2"]
    except KeyError:
        pass
    try:
        image_stats["grains_number_above"] = results_df["threshold"].value_counts().get("above", 0)
        image_stats["grains_per_m2_above"] = image_stats["grains_number_above"] / image_stats["image_area_m2"]
    except KeyError:
        pass

    image_stats_df = pd.DataFrame([image_stats])
    image_stats_df.set_index("image", inplace=True)

    return image_stats_df


def roughness_rms(image: np.ndarray) -> float:
    """
    Calculate the root-mean-square roughness of a heightmap image.

    Parameters
    ----------
    image : np.ndarray
        2-D numpy array of heightmap data to calculate roughness.

    Returns
    -------
    float
        The RMS roughness of the input array.
    """
    return np.sqrt(np.mean(np.square(image)))
