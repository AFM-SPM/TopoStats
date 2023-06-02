"""Contains functions pertaining to the roughness of AFM images."""

import numpy as np


def roughness_rms(image: np.ndarray) -> float:
    """Calculate the root-mean-square roughness of a heightmap image.

    Parameters
    ----------
    image: np.ndarray
        2D numpy array of heightmap data to calculate the roughness of

    Returns:
    float
        The RMS roughness of the input array.
    """
    return np.sqrt(np.mean(np.square(image)))
