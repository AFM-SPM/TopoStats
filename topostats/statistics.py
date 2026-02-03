"""Function for calculating statistics about a whole image, for example number of grains or surface roughness."""

import logging

import numpy as np
import numpy.typing as npt

from topostats.logs.logs import LOGGER_NAME

LOGGER = logging.getLogger(LOGGER_NAME)


def image_statistics(
    image: npt.NDArray, filename: str, pixel_to_nm_scaling: float, n_grains: int
) -> dict[str, int | str | float]:
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
    n_grains : int
        Number of grains in image.

    Returns
    -------
    dict[str, int | str | float]
        Dictionary of image statistics.
    """
    image_stats = {
        "image": filename,
        "image_size_x_m": image.shape[1] * pixel_to_nm_scaling * 1e-9,
        "image_size_y_m": image.shape[0] * pixel_to_nm_scaling * 1e-9,
        "image_area_m2": None,
        "image_size_x_px": image.shape[1],
        "image_size_y_px": image.shape[0],
        "image_area_px2": None,
        "grains": n_grains,
        "grains_per_m2": None,
        "rms_roughness": None,
    }
    # Calculate areas
    image_stats["image_area_m2"] = image_stats["image_size_x_m"] * image_stats["image_size_y_m"]
    image_stats["image_area_px2"] = image_stats["image_size_x_px"] * image_stats["image_size_y_px"]
    # Calculate the RMS roughness of the sample on the flattened image.
    image_stats["rms_roughness"] = roughness_rms(image=image) * 1e-9

    # ns-rse 2025-12-19 Need to reconcile grain density per threshold level, challenging as conceivably there could be >
    # 2
    image_stats["grains_per_m2"] = image_stats["grains"] / image_stats["image_area_m2"]

    return image_stats


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
