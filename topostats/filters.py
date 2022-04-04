from statistics import median
import numpy as np
from skimage.filters import threshold_otsu
import logging

"""Contains filter functions that take a 2D array representing an image as an input, as well as necessary parameters, and return a 2D array of the same size representing the filtered image."""


def amplify(image: np.array, level: float) -> np.array:
    """The amplify filter mulitplies the value of all pixels by the `level` argument.

    :param image: A 2D raster image
    :param level: Filter level
    :return: A filtered 2D raster image
    """

    return image * level

def row_col_quantiles(image: np.array, binary_mask=None) -> np.array:
    """Returns the height value quantiles for the rows and columns.

    :param image: A 2D raster image
    :param binary_mask: (Array) Optional parameter that allows the use of a mask to ignore data
    :return: Two arrays, one of row height value quantiles, the second of column height value quantiles
    """

    # Mask the data if applicable
    if binary_mask is not None:
        image = np.ma.masked_array(image, mask=binary_mask, fill_value=np.nan)
        logging.info('masking enabled')
    else:
        logging.info('masking disabled')

    # Initialise arrays
    row_quantiles = np.zeros((image.shape[0], 3))
    col_quantiles = np.zeros((image.shape[1], 3))

    # Populate the row array with quantile tuples
    for i in range(image.shape[0]):
        row = image[i, :]
        row_quantiles[i] = np.array([np.quantile(row, 0.25),
        np.quantile(row, 0.5),
        np.quantile(row, 0.75)])

    # Populate the column array with quantile tuples
    for j in range(image.shape[1]):
        col = image[:, j]
        col_quantiles[j] = np.array([np.quantile(col, 0.25),
        np.quantile(col, 0.5),
        np.quantile(col, 0.75)])

    return row_quantiles, col_quantiles

<<<<<<< HEAD
def align_rows(image: np.array, binary_mask=None) -> np.array:
    """Returns the input image with rows aligned by median height

    :param image: A 2D raster image
    :param binary_mask: (Array) Optional parameter that allows the use of a mask to ignore data.
    :return: The same image but with the rows aligned in median height
    """

    # Get row and column height quantiles for the image. Does nothing if binary_mask = None
    row_quantiles, col_quantiles = row_col_quantiles(image, binary_mask)

    # Align row medians
    # Calculate median row height
    row_medians = row_quantiles[:, 1]
    median_row_height = np.quantile(row_medians, 0.5)
    logging.info(f'median_row_height: {median_row_height}')
    
    # Calculate the differences between the row medians and the median row height
    row_median_diffs = row_medians - median_row_height

    # Adjust the row medians accordingly
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i, j] -= row_median_diffs[i]
    
    return image
=======

>>>>>>> 1d66b93... Add row_col_quantiles function to filters.py

def remove_x_y_tilt(image: np.array, binary_mask=None) -> np.array:
    """Returns the input image after removing any linear plane slant

    :param image: A 2D raster image
    :param binary_mask: (Array) Optional parameter that allows the use of a mask to ignore data.
    :return: The same image but with any linear plane slant removed.
    """

    # Get the row and column quantiles of the data
    row_quantiles, col_quantiles = row_col_quantiles(image, binary_mask)

    # Calculate the x gradient from the left to the right
    x_diff = row_quantiles[-1][1] - row_quantiles[0][1]
    logging.info(f'x_diff: {x_diff}')
    # Calculate the gradient of brightness per pixel
    x_grad = x_diff / image.shape[0]
    logging.info(f'x_grad: {x_grad}')

    # Repeat for y
    y_diff = col_quantiles[-1][1] - col_quantiles[0][1]
    logging.info(f'y_diff: {y_diff}')
    y_grad = y_diff / image.shape[1]
    logging.info(f'y_grad: {y_grad}')

    # Add corrections to the data
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i, j] -= x_grad * i
            image[i, j] -= y_grad * j

    return image


def get_threshold(image: np.array) -> float:
    """Returns a threshold value separating the background and foreground of a 2D heightmap.

    :param image: A 2D raster image
    :return: Float - the threshold between background and foreground heights. 
    """

    threshold = threshold_otsu(image)
    
    return threshold


