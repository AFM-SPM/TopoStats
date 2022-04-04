from statistics import median
import numpy as np

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
    if binary_mask != None:
        image = np.ma.masked_array(image, mask=binary_mask, fill_value=np.nan)

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
        col_quantiles = np.array([np.quantile(col, 0.25),
        np.quantile(col, 0.5),
        np.quantile(col, 0.75)])

    return row_quantiles, col_quantiles




