"""Contains filter functions that take a 2D array representing an image as an input, as well as necessary parameters,
and return a 2D array of the same size representing the filtered image."""
import logging
from scipy.optimize import curve_fit
from statistics import median, stdev
import numpy as np
from skimage.filters import threshold_otsu


def amplify(image: np.array, level: float) -> np.array:
    """The amplify filter mulitplies the value of all pixels by the `level` argument.

    :param image: A 2D raster image
    :param level: Filter level
    :return: A filtered 2D raster image
    """

    return image * level


def row_col_quantiles(image: np.array, binary_mask = None) -> np.array:
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
        row_quantiles[i] = np.quantile(row, [0.25, 0.5, 0.75])

    # Populate the column array with quantile tuples
    for j in range(image.shape[1]):
        col = image[:, j]
        col_quantiles[j] = np.quantile(col, [0.25, 0.5, 0.75])

    return row_quantiles, col_quantiles


def align_rows(image: np.array, binary_mask: bool=False) -> np.array:
    """Returns the input image with rows aligned by median height

    :param image: A 2D raster image
    :param binary_mask: (Array) Optional parameter that allows the use of a mask to ignore data.
    :return: The same image but with the rows aligned in median height
    """

    # Get row and column height quantiles for the image. Does nothing if binary_mask = None
    row_quantiles, _ = row_col_quantiles(image, binary_mask)

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


def remove_x_y_tilt(image: np.array, binary_mask: bool = False) -> np.array:
    """Returns the input image after removing any linear plane slant

    :param image: A 2D raster image
    :param binary_mask: (Array) Optional parameter that allows the use of a mask to ignore data.
    :return: The same image but with any linear plane slant removed.
    """

    # Get the row and column quantiles of the data
    row_quantiles, col_quantiles = row_col_quantiles(image, binary_mask)

    # Calculate the x and y gradient from the left to the right
    # FIXME : I've abstracted out the calculation of diff and gradient to avoid code-duplication, but I'm curious if
    #         in the broader scheme of things these parameters are something that the user might want to collate
    #         across all files for assessing data quality?
    #         The values are logged, but if processing thousands of files you don't want to
    #         parse the log-files, it would be relativcely straight-forward, with some refactoring into a class,
    #         to have these statistics returned so they can be collated, but as I've written it the diff isn't
    #         directly available at the moment (because on the grad is subsequently used).
    x_grad = calc_gradient(row_quantiles, row_quantiles.shape[0])
    logging.info(f'x_grad: {x_grad}')
    y_grad = calc_gradient(col_quantiles, col_quantiles.shape[0])
    logging.info(f'y_grad: {y_grad}')

    # Add corrections to the data
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i, j] -= x_grad * i
            image[i, j] -= y_grad * j

    return image


def calc_diff(array: np.array):
    """Calculate the difference of an array."""
    return array[-1][1] - array[0][1]


def calc_gradient(array: np.array, shape: int) -> np.array:
    """Calculate the gradient of an array."""
    return calc_diff(array) / shape


def get_threshold(image: np.array) -> float:
    """Returns a threshold value separating the background and foreground of a 2D heightmap.

    :param image: A 2D raster image
    :return: Float - the threshold between background and foreground heights.
    """
    return threshold_otsu(image)


def remove_x_bowing(image: np.array, binary_mask):

    def quadratic(x, a, b, c):
        return a * x**2 + b * x + c

    # Use the mask to mask the data
    image_masked = np.ma.masked_array(image,
                                      mask=binary_mask,
                                      fill_value=np.nan)

    # Get the average row
    average_row = np.zeros((image_masked.shape[0]))
    for i in range(image_masked.shape[0]):
        average_row[i] = np.mean(image_masked[:, i])

    print(average_row)

    # Plotting setup
    import matplotlib.pyplot as plt
    import matplotlib
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))

    current_cmap = matplotlib.cm.get_cmap()
    current_cmap.set_bad(color='red')
    # ax.imshow(average_row)
    x_data = np.array(range(0, average_row.shape[0]))
    print('xdata shape: ', x_data.shape)
    print('average row shape: ', average_row.shape)

    # Create a quadratic fit
    parameters, covariance = curve_fit(f=quadratic,
                                       xdata=x_data,
                                       ydata=average_row)
    print('parameters:')
    print(parameters)
    print('covariance:')
    print(covariance)

    # Plot the data
    ax = axes[0]
    ax.scatter(x_data, average_row, s=20, color='#008080', label='data')
    ax.plot(x_data,
            quadratic(x_data, *parameters),
            linestyle='--',
            linewidth=2,
            color='black',
            label='fit')
    ax.legend()

    # Calculate standard deviations
    stdevs = np.sqrt(np.diag(covariance))
    print('stdevs: ')
    print(np.mean(stdevs))

    # Calculate the residuals
    residuals = average_row - quadratic(x_data, *parameters)

    # Plot the residuals
    ax = axes[1]
    ax.plot(x_data, residuals, color='#602020', label='residuals')
    ax.legend()
    plt.savefig('experiment_average_row')
