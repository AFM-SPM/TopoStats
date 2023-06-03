"""Function for calculating statistics about a whole image, for example number of grains or surface roughness."""

import numpy as np
import pandas as pd

from topostats.roughness import roughness_rms


def image_statistics(image: np.ndarray, grainstats: pd.DataFrame, pixel_to_nm_scaling: float) -> pd.DataFrame:
    """Calculate statistics pertaining to the whole image, for example the size of the image in pixels and
    metres, the root-mean-squared roughness and the grains per metre squared.

    Parameters
    ----------
    image: np.ndarray
        Numpy 2D image array of the image to calculate stats for.
    grainstats: pd.DataFrame
        Pandas DataFrame of grain statistics.
    pixel_to_nm_scaling: float
        Float of the scaling factor between pixels and nanometres.

    Returns
    -------
    dict
        Dictionary of image statistics.
    """

    image_stats = {
        "image_size_x_m": image.shape[1] * pixel_to_nm_scaling * 1e-9,
        "image_size_y_m": image.shape[0] * pixel_to_nm_scaling * 1e-9,
        "image_area_m2": None,
        "image_size_x_px": image.shape[1],
        "image_size_y_px": image.shape[0],
        "image_area_px2": None,
        "grains_number_above": grainstats["threshold"].value_counts().get("above", 0),
        "grains_per_m2_above": None,
        "grains_number_below": grainstats["threshold"].value_counts().get("below", 0),
        "grains_per_m2_below": None,
        "rms_roughness": None,
    }

    # Calculate area of the image
    image_stats["image_area_m2"] = image_stats["image_size_x_m"] * image_stats["image_size_y_m"]
    image_stats["image_area_px2"] = image_stats["image_size_x_px"] * image_stats["image_size_y_px"]

    # Calculate the RMS roughness of the sample on the flattened image.
    image_stats["rms_roughness"] = roughness_rms(image=image) * 1e-9

    # Calculate numbers of grains per metre squared
    image_stats["grains_per_m2_above"] = image_stats["grains_number_above"] / image_stats["image_area_m2"]
    image_stats["grains_per_m2_below"] = image_stats["grains_number_below"] / image_stats["image_area_m2"]

    return image_stats
