"""Find grains in images."""
# pylint: disable=no-name-in-module
# pylint: disable=invalid-name
from pathlib import Path
import logging
from typing import Dict, List, Union

import numpy as np
import pandas as pd

from skimage.filters import gaussian
from skimage.segmentation import clear_border
from skimage.morphology import remove_small_objects, label
from skimage.measure import regionprops
from skimage.color import label2rgb

from topostats.utils import get_threshold
from topostats.logs.logs import LOGGER_NAME

LOGGER = logging.getLogger(LOGGER_NAME)


def quadratic(x, a, b, c):
    """Calculate the result of the quadratic equation."""
    LOGGER.info('Calculating quadratic.')
    return (a * x**2) + (b * x) + c


def gaussian_filter(image: np.array,
                    gaussian_size: float = 2,
                    dx: float = 1,
                    mode: str = 'nearest',
                    **kwargs) -> np.array:
    """Apply Gaussian filter

    Parameters
    ----------
    image : np.array
        Numpy array of image.
    gaussian_size : float
        Gaussian blur size in nanometers (nm).
    dx : float
        Pixel to nanometer scale.
    mode : str
        Mode for filtering (default is 'nearest')

    Returns
    -------
    np.array
        Numpy array of filtered image.

    Examples
    --------
    FIXME: Add docs.
    """
    LOGGER.info('Applying Gaussian filter (mode : {mode}; Gaussian blur (nm) : {gaussian_size}).')
    return gaussian(image, sigma=(gaussian_size / dx), mode=mode, **kwargs)


# def boolean_image(image: np.array, threshold: float) -> np.array:
#     """Create a boolean array of whether points are greater than the given threshold.

#     Parameters
#     ----------
#     image: np.array
#         Numpy array representing image.
#     threshold: float
#         Threshold for masking points in the image.
#     Returns
#     -------
#     np.array
#         Numpy array for masking
#     """
#     LOGGER.info('Created boolean image')
#     return np.array(np.copy(image) > threshold, dtype='bool')


def tidy_border(image: np.array, **kwargs) -> np.array:
    """Remove grains touching the border
    Parameters
    ----------
    image: np.array
        Numpy array representing image.

    Returns
    -------
    np.array
        Numpy array of image with borders tidied.
    """
    LOGGER.info('Tidying borders')
    return clear_border(np.copy(image), **kwargs)


# def calc_minimum_grain_size(minimum_grain_size: float = None, dx: float = 1) -> float:
#     """Calculate minimum grain size in pixels.
#
#     """


def remove_objects(image: np.array, minimum_grain_size: float, dx: float, **kwargs) -> np.array:
    """Remove small objects

    Parameters
    ----------
    image: np.array
        Numpy array representing image.
    minimum_grain_size: float
        Minimum grain size in nanometers.
    dx: float
        Pixel to nanometer scale.
    Returns
    -------
    np.array
        Numpy array of image with objects coloured.
    """
    LOGGER.info(f'Removing small objects (< {minimum_grain_size / dx})')
    return remove_small_objects(image, min_size=(minimum_grain_size / dx), **kwargs)


def label_regions(image: np.array, background: float = 0.0, **kwargs) -> np.array:
    """Label regions.

    Parameters
    ----------
    image: np.array
        Numpy array representing image.
    background: float

    Returns
    -------
    np.array
        Numpy array of image with objects coloured.
    """
    LOGGER.info('Labelling Regions')
    return label(image, background=background, **kwargs)


def colour_regions(image: np.array, **kwargs) -> np.array:
    """Colour the regions.

    Parameters
    ----------
    image: np.array
        Numpy array representing image.

    Returns
    -------
    np.array
        Numpy array of image with objects coloured.
    """
    LOGGER.info('Colouring regions')
    return label2rgb(image, **kwargs)


def region_properties(image: np.array, **kwargs) -> List:
    """Extract the properties of each region.

    Parameters
    ----------
    image: np.array
        Numpy array representing image

    Returns
    -------
    List
        List of region property objects.
    """
    LOGGER.info('Extracting region properties.')
    return regionprops(image, **kwargs)


def get_bounding_boxes(region_prop: List) -> Dict:
    """Derive a list of bounding boxes for each region.

    Parameters
    ----------
    region_prop : skimage
        Dictionary of region properties

    Returns
    -------
    dict
        Dictionary of bounding boxes indexed by region area.


    Examples
    --------
    FIXME: Add docs.

    """
    LOGGER.info('Extracting bounding boxes')
    return {region.area: region.area_bbox for region in region_prop}


def save_region_stats(bounding_boxes: dict, output_dir: Union[str, Path]) -> None:
    """Save the bounding box statistics.

    Parameters
    ----------
    bounding_boxes: dict
        Dictionary of bounding boxes
    output_dir: Union[str, Path]
        Where to save the statistics to as a CSV file called 'grainstats.csv'."""
    grainstats = pd.DataFrame.from_dict(data=bounding_boxes, orient='index')
    grainstats.to_csv(output_dir / 'grainstats.csv', index=True)
