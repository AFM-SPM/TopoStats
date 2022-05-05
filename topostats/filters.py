"""Contains filter functions that take a 2D array representing an image as an input, as well as necessary parameters,
and return a 2D array of the same size representing the filtered image."""
# pylint: disable=no-name-in-module
# pylint: disable=invalid-name
# pylint: disable=fixme
from pathlib import Path
from typing import Union
import logging
import numpy as np
import pySPM
from pySPM.SPM import SPM_image

from topostats.logs.logs import LOGGER_NAME

LOGGER = logging.getLogger(LOGGER_NAME)


def extract_img_name(img_path: Union[str, Path]) -> str:
    """Extract the image name from the image path.

    Parameters
    ----------
    img_path : Union[str, Path]
        Path to image being processed.

    Returns
    -------
    str
        Filename of processed image.

    Examples
    --------
    FIXME: Add docs.

    """
    LOGGER.info(f'Extracting filename from : {img_path}')
    return Path(img_path).stem


def extract_channel(scan_raw: pySPM.Bruker, channel: str = 'Height') -> SPM_image:
    """Extract the given channel from the image.

    Parameters
    ----------
    scan_raw : pySPM.Bruker
        Raw scan loaded by pySPM.
    channel : str
        Channel to extract (default is 'Height').
    filename: str
        File being processed.

    Returns
    -------
    SPM_image
        SPM image.

    Examples
    --------
    FIXME: Add docs.

    """
    return scan_raw.get_channel(channel)


def extract_pixel_to_nm_scaling(extracted_channel: SPM_image) -> float:
    """Extract the pixel to nanometer scaling from the image metadata.

    Parameters
    ----------
    extracted_channel : SPM_image
        Channel extracted from an image.

    Returns
    -------
    float
        Scaling factor for pixels to nanometers.
    """
    return extracted_channel.get_extent()[1] / len(extracted_channel.pixels)


def extract_pixels(extracted_channel: SPM_image) -> np.array:
    """Flatten the scan to a Numpy Array

    Parameters
    ----------
    extracted_channel : SPM_image
        Channel extracted from an image.

    Returns
    -------
    np.array
        Numpy array representation of the channel of interest.

    Examples
    --------
    FIXME: Add docs.

    """
    return np.flipud(np.array(extracted_channel.pixels))


def amplify(image: np.array, level: float) -> np.array:
    """The amplify filter mulitplies the value of all pixels by the `level` argument.

     Parameters
    ----------
    image: np.array
        Numpy array representing image.
    level: np.array
        Factor by which to amplify the array.

    Returns
    -------
    np.array
        Numpy array of image amplified by level.
    """
    LOGGER.info(f'[amplify] Level : {level}')
    return image * level


def row_col_quantiles(image: np.array, mask: np.array = None) -> np.array:
    """Returns the height value quantiles for the rows and columns.

    Parameters
    ----------
    image: np.array
        Numpy array representing image.
    mask: np.array
        Mask array to apply.

    Returns
    -------
    np.array
        Numpy array of image but with any linear plane slant removed.
    """

    # Mask the data if applicable
    if mask is not None:
        image = np.ma.masked_array(image, mask=mask, fill_value=np.nan)
        LOGGER.info('[row_col_quantiles] Masking enabled')
    else:
        LOGGER.info('[row_col_quantiles] Masking disabled')

    row_quantiles = np.quantile(image, [0.25, 0.5, 0.75], axis=1).T
    col_quantiles = np.quantile(image, [0.25, 0.5, 0.75], axis=0).T
    LOGGER.info('Row and column quantiles calculated.')
    return row_quantiles, col_quantiles


def align_rows(image: np.array, mask: np.array = None) -> np.array:
    """Returns the input image with rows aligned by median height

    Parameters
    ----------
    image: np.array
        Numpy array representing image.
    mask: np.array
        Mask array to apply.

    Returns
    -------
    np.array
        Numpy array of image but with any linear plane slant removed.
    """

    # Get row height quantiles for the image.
    row_quantiles, _ = row_col_quantiles(image, mask)

    # Calculate median row height
    row_medians = row_quantiles[:, 1]
    # Calculate median row height
    median_row_height = np.quantile(row_medians, 0.5)
    LOGGER.info(f'[align_rows] median_row_height: {median_row_height}')

    # Calculate the differences between the row medians and the median row height
    row_median_diffs = row_medians - median_row_height

    # Adjust the row medians accordingly
    for i in range(image.shape[0]):
        if np.isnan(row_median_diffs[i]):
            logging.info(f'{i} row_median is nan! : {row_median_diffs[i]}')
        image[i] -= row_median_diffs[i]
    return image


def remove_x_y_tilt(image: np.array, mask: np.array = None) -> np.array:
    """Returns the input image after removing any linear plane slant

    Parameters
    ----------
    image: np.array
        Numpy array representing image.
    mask: np.array
        Mask array to apply.

    Returns
    -------
    np.array
        Numpy array of image but with any linear plane slant removed.
    """

    # Get the row and column quantiles of the data
    row_quantiles, col_quantiles = row_col_quantiles(image, mask)

    # Calculate the x and y gradient from the left to the right
    x_grad = calc_gradient(row_quantiles, row_quantiles.shape[0])
    logging.info(f'[remove_x_y_tilt] x_grad: {x_grad}')
    y_grad = calc_gradient(col_quantiles, col_quantiles.shape[0])
    logging.info(f'[remove_x_y_tilt] y_grad: {y_grad}')

    # Add corrections to the data
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i, j] -= x_grad * i
            image[i, j] -= y_grad * j
    LOGGER.info('[remove_x_y_tilt] Linear plane slant removed')
    return image


def calc_diff(array: np.array) -> np.array:
    """Calculate the difference of an array.

    Parameters
    ----------
    image: np.array
        Numpy array representing image.

    Returns
    -------
    np.array
    """
    LOGGER.info('[calc_diff] Calculating difference in array.')
    return array[-1][1] - array[0][1]


def calc_gradient(array: np.array, shape: int) -> np.array:
    """Calculate the gradient of an array.

    Parameters
    ----------
    image: np.array
        Numpy array representing image.
    shape: int
        Shape of the array in terms of the number of rows, typically it will be array[0].

    Returns
    -------
    np.array
    """
    LOGGER.info('[calc_gradient] Calculating gradient')
    return calc_diff(array) / shape


def average_background(image: np.array, mask: np.array = None) -> np.array:
    """Zero the background based on the median.

    Parameters
    ----------
    image: np.array
        Numpy array representing image.
    mask: np.array
        Mask of the array, should have the same dimensions as image.

    Returns
    -------
    np.array
        Numpy array of image zero averaged.
    """
    row_quantiles, _ = row_col_quantiles(image, mask)
    LOGGER.info('[average_background] Zero averaging background')
    return image - np.array(row_quantiles[:, 1], ndmin=2).T
