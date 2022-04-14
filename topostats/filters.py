"""Contains filter functions that take a 2D array representing an image as an input, as well as necessary parameters,
and return a 2D array of the same size representing the filtered image."""
from pathlib import Path
from typing import Union
import logging
from scipy.optimize import curve_fit
from statistics import median, stdev
import numpy as np
from skimage.filters import threshold_otsu
from skimage import filters as skimage_filters
from skimage import segmentation as skimage_segmentation
from skimage import measure as skimage_measure
from skimage import morphology as skimage_morphology
from skimage import color as skimage_color
from scipy import ndimage

from topostats.plottingfuncs import plot_and_save


def amplify(image: np.array, level: float) -> np.array:
    """The amplify filter mulitplies the value of all pixels by the `level` argument.

    :param image: A 2D raster image
    :param level: Filter level
    :return: A filtered 2D raster image
    """

    return image * level


def row_col_quantiles(image: np.array, mask: np.array = None) -> np.array:
    """Returns the height value quantiles for the rows and columns.

    :param image: A 2D raster image
    :param mask: (Array) Optional parameter that allows the use of a mask to ignore data
    :return: Two arrays, one of row height value quantiles, the second of column height value quantiles
    """

    # Mask the data if applicable
    if mask is not None:
        # image = np.ma.masked_array(image, mask=mask, fill_value=np.nan)
        image = np.ma.masked_array(image, mask=mask, fill_value=np.nan)
        logging.info('[row_col_quantiles] masking enabled')
    else:
        logging.info('[row_col_quantiles] masking disabled')

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


def align_rows(image: np.array, mask: np.array = None) -> np.array:
    """Returns the input image with rows aligned by median height

    :param image: A 2D raster image
    :param mask: (Array) Optional parameter that allows the use of a mask to ignore data.
    :return: The same image but with the rows aligned in median height
    """

    # Get row height quantiles for the image.
    row_quantiles, _ = row_col_quantiles(image, mask)

    # Align row medians
    # Calculate median row height
    row_medians = row_quantiles[:, 1]
    # logging.info(row_medians)
    median_row_height = np.quantile(row_medians, 0.5)
    logging.info(median_row_height)
    logging.info(f'[align_rows] median_row_height: {median_row_height}')

    # Calculate the differences between the row medians and the median row height
    row_median_diffs = row_medians - median_row_height

    # Adjust the row medians accordingly
    for i in range(image.shape[0]):
        if np.isnan(row_median_diffs[i]):
            LOGGER.info(f'{i} row_median is nan! : {row_median_diffs[i]}')
        # for j in range(image.shape[1]):
        #     image[i, j] -= row_median_diffs[i]
        image[i] -= row_median_diffs[i]
    return image


def remove_x_y_tilt(image: np.array, mask: np.array = None) -> np.array:
    """Returns the input image after removing any linear plane slant

    :param image: A 2D raster image
    :param mask: (Array) Optional parameter that allows the use of a mask to ignore data.
    :return: The same image but with any linear plane slant removed.
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

    return image


def calc_diff(array: np.array):
    """Calculate the difference of an array."""
    return array[-1][1] - array[0][1]


def calc_gradient(array: np.array, shape: int) -> np.array:
    """Calculate the gradient of an array."""
    # logging.info(array)
    return calc_diff(array) / shape


def get_threshold(image: np.array) -> float:
    """Returns a threshold value separating the background and foreground of a 2D heightmap.

    :param image: A 2D raster image
    :return: Float - the threshold between background and foreground heights.
    """
    return threshold_otsu(image)


def get_mask(image: np.array, threshold: float) -> np.array:
    """Calculate a mask for pixels that exceed the threshold

    :param image: A 2D raster image
    :threshold float: Threshold for masking pixels"""
    return image > threshold


# def remove_x_bowing(image: np.array, mask: np.array) -> tuple:
#     """Remove X bowing.

#     :param image: A 2D raster image
#     :param binary_mask: A 2D binary array of points that should be masked.
#     """
#     masked_image = np.ma.masked_array(image,
#                                       mask=mask,
#                                       fill_value=np.nan)
#     row_mean = np.mean(masked_image, axis=1).T
#     parameters, covariance = curve_fit(row_mean)
#     return parameters, covariance

# def curve_fit(row_mean: np.array) -> tuple:
#     """Fit a quadratic curve to the data.

#     :param row_mean: Mean of row. """
#     parmeters, covariance = curve_fit(quadratic,
#                                       np.arange(0, row_mean.shape[0]),
#                                       row_mean)
#     return parameters, covariance


def quadratic(x, a, b, c):
    """Calculate the result of the quadratic equation."""
    return (a * x**2) + (b * x) + c


    # Gaussian filter
def gaussian_filter(image: np.array,
                    gaussian_size: float = 2,
                    dx: float = 1,
                    mode: str = 'nearest') -> np.array:
    """Apply gaussian filter to data.
    :param image:
    :param gaussian_size: Gaussian glur size in nanometers (nm).
    :param dx: Pixel to nm scale.
    :param mode:"""
    return skimage_filters.gaussian(image,
                                    sigma=(gaussian_size / dx),
                                    mode=mode)


def boolean_image(image: np.array, threshold: float) -> np.array:
    """Create a boolean array of whether points are greater than the given threshold.

    :param image:
    :param threshold"""
    return np.array(np.copy(image) > threshold, dtype='bool')


def tidy_border(image: np.array) -> np.array:
    """Remove grains touching the border"""
    return skimage_segmentation.clear_border(np.copy(image))


def remove_small_objects(image: np.array, minimum_grain_size: float,
                         dx: float) -> np.array:
    """Remove small objects

    :param minimum_grain_size: Minimum grain size in nanometers (nm).
    :param dx: Pixel to nm scale.
    :return: Image with small objects removed.
    """
    return skimage_morphology.remove_small_objects(
        data_boolean, min_size=(minimum_grain_size / dx))


def label_regions(image: np.array, background: float = 0.0) -> np.array:
    """Label regions.

    :param image:
    :param background:
    """
    return skimage_morphology.label(image, brackground=background)


def colour_regions(image: np.array) -> np.array:
    """Colour the regions.

    :param image:
    """
    return skimage_color.label2rgb(image)


def region_properties(image: np.array) -> np.array:
    """
    :param image:
    """
    return skimage_measure.regionprops(image)


def find_grains(image: np.array,
                gaussian_size: Union[int, float] = 2,
                dx: Union[int, float] = 1,
                upper_height_threshold_rms_multiplier: Union[int, float] = 1,
                lower_threshold_otsu_multiplier: Union[int, float] = 1.7,
                minimum_grain_size: Union[int, float] = 800,
                mode: str = 'nearest',
                outdir: Union[str, Path] = 'grain_finding') -> np.array:
    """Find grains within an image.

    :param gaussian_size: Gaussian glur size in nanometers (nm).
    :param dx: Pixel to nm scale.
    :param upper_height_threshold_rms_multiplier: Sets the rms multiplier for high-value outlier detection. The higher
    the value the less sensitive grain finding will be to outliers.
    :param lower_thershold_otsu_multiplier: Otsu threshold multiplier for lower thresholding the data. Higher values
    cull lower values around the edges of grains.
    :param minimum_grain_size: Minimum grain size in nanometers (nm).
    :param mode: Method for Gaussian filtering
    :param outdir: Output directory to save images to.
    :return:
    """
    if Path(outdir).is_dir() is False:
        Path(outdir).mkdir(parents=True, exist_ok=True)
    data = np.copy(gaussican_filtered)
    lower_threshold = get_threshold(data) * lower_threshold_otsu_multiplier
    logging.info(f'Lower threshold : {lower_threshold}')
    gaussian_filtered = gaussian_filter(image, gaussian_size, dx1, mode)
    plot_and_save(gaussian_filtered, outdir / 'gaussian_filter.png')
    boolean_image = get_boolean(np.copy(image), lower_threshold)
    border_cleared = clear_border(boolean_image)
    plot_and_save(boolean_image, outdir / 'boolean_border_clear.png')
    small_objects_removed = remove_small_objects(border_cleared,
                                                 minimum_grain_size, dx)
    plot_and_save(boolean_image, outdir / 'small_objects_removed.png')
    labelled_regions = label_regions(small_objects_removed)
    plot_and_save(boolean_image, outdir / 'labelled_regions.png')
    coloured_regions = colour_regions(labelled_regions)
    plot_and_save(boolean_image, outdir / 'coloured_regions.png')
    regional_properties = region_properties(coloured_regions)
    plot_and_save(boolean_image, outdir / 'regional_properties.png')
