"""Contains filter functions that take a 2D array representing an image as an input, as well as necessary parameters,
and return a 2D array of the same size representing the filtered image."""
# pylint: disable=no-name-in-module
# pylint: disable=invalid-name
from pathlib import Path
from typing import Dict, List, Union
import logging
import numpy as np
import pySPM
from pySPM.SPM import SPM_image
from pySPM.Bruker import Bruker

from skimage.filters import threshold_otsu, gaussian
from skimage.segmentation import clear_border
from skimage.morphology import remove_small_objects, label
from skimage.measure import regionprops
from skimage.color import label2rgb

from topostats.plottingfuncs import plot_and_save
from topostats.logs.logs import LOGGER_NAME

LOGGER = logging.getLogger(LOGGER_NAME)


def load_scan(img_path: Union[str, Path]) -> Bruker:
    """Load the image from file.

    Parameters
    ----------
    img_path : Union[str, Path]
        Path to image that needs loading.

    Returns
    -------
    Bruker

    Examples
    --------
    FIXME: Add docs.

    """
    LOGGER.info(f'Loading image from : {img_path}')
    return pySPM.Bruker(Path(img_path))

# def extract_filename(img_path: Union[str, Path]) -> str:
def extract_filename(img_path) -> str:
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
    print('#######')
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

    Returns
    -------
    SPM_image
        SPM image.

    Examples
    --------
    FIXME: Add docs.

    """
    LOGGER.info(f'Extracting filename from : {img_path}')
    return scan_raw.get_channel(channel)


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
    LOGGER.info(f'[extract_pixels] Extracting pixels')
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
        LOGGER.info('[row_col_quantiles] masking enabled')
    else:
        LOGGER.info('[row_col_quantiles] masking disabled')

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

    # Align row medians
    # Calculate median row height
    row_medians = row_quantiles[:, 1]
    # logging.info(row_medians)
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
    LOGGER.info(f'[remove_x_y_tilt] Linear plane slant removed')
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
    LOGGER.info(f'Calculating difference in array.')
    return array[-1][1] - array[0][1]


def calc_gradient(array: np.array, shape: int) -> np.array:
    """Calculate the gradient of an array.

    Parameters
    ----------
    image: np.array
        Numpy array representing image.
    shape: int
        Snape of the array.

    Returns
    -------
    np.array
    """
    LOGGER.info(f'Calculating gradient')
    return calc_diff(array) / shape


def get_threshold(image: np.array) -> float:
    """Returns a threshold value separating the background and foreground of a 2D heightmap.

    Parameters
    ----------
    image: np.array
        Numpy array representing image.

    Returns
    -------
    float
        Otsu threshold.

    Notes
    -----

    The `Otsu method <https://en.wikipedia.org/wiki/Otsu%27s_method>`_ is used for threshold derivation.
    """
    LOGGER.info(f'Calculating Otsu threshold.')
    return threshold_otsu(image)


def get_mask(image: np.array, threshold: float) -> np.array:
    """Calculate a mask for pixels that exceed the threshold

    Parameters
    ----------
    image: np.array
        Numpy array representing image.
    threshold: float
        Factor for defining threshold.

    Returns
    -------
    np.array
        Numpy array of image with objects coloured.
    """
    LOGGER.info(f'Deriving mask.')
    return image > threshold


def average_background(image: np.array, mask: np.array = None) -> np.array:
    """Zero the background

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
    :param image: a 2D raster image"""
    row_quantiles, _ = row_col_quantiles(image, mask)
    LOGGER.info(f'Zero averaging background')
    return image - np.array(row_quantiles[:, 1], ndmin=2).T


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
    LOGGER.info(f'Calculating quadratic.')
    return (a * x**2) + (b * x) + c


# Gaussian filtering
def get_lower_threshold(image: np.array, lower_threshold_multiplier: float = 1.7) -> float:
    """Calculate the lower threshold for filtering.

    Parameters
    ----------
    image: np.array
        Numpy array representing image.
    lower_threshold_multiplier: float
        Factor by which threshold is to be scaled by.
    Returns
    -------
    float
        Lower threshold
    """
    LOGGER.info(f'Calculating lower threshold for grain identification.')
    return get_threshold(image) * lower_threshold_multiplier


def gaussian_filter(image: np.array, gaussian_size: float = 2, dx: float = 1, mode: str = 'nearest') -> np.array:
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
    return gaussian(image, sigma=(gaussian_size / dx), mode=mode)


# FIXME : This is duplicated code and the same as the get_mask() function
def boolean_image(image: np.array, threshold: float) -> np.array:
    """Create a boolean array of whether points are greater than the given threshold.

    Parameters
    ----------
    image: np.array
        Numpy array representing image.
    threshold: float
        Threshold for masking points in the image.
    Returns
    -------
    np.array
        Numpy array for masking
    """
    LOGGER.info(f'Created boolean image')
    return np.array(np.copy(image) > threshold, dtype='bool')


def tidy_border(image: np.array) -> np.array:
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
    LOGGER.info(f'Tidying borders')
    return clear_border(np.copy(image))

# def calc_minimum_grain_size(minimum_grain_size: float = None, dx: float = 1) -> float:
#     """Calculate minimum grain size in pixels.
#
#     """

def remove_objects(image: np.array, minimum_grain_size: float, dx: float) -> np.array:
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
    return remove_small_objects(image, min_size=(minimum_grain_size / dx))


def label_regions(image: np.array, background: float = 0.0) -> np.array:
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
    LOGGER.info(f'Labelling Regions')
    return label(image, background=background)


def colour_regions(image: np.array) -> np.array:
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
    LOGGER.info(f'Colouring regions')
    return label2rgb(image)


def region_properties(image: np.array) -> List:
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
    LOGGER.info(f'Extracting region properties.')
    return regionprops(image)

def get_bounding_boxes(region_properties: List) -> Dict:
    """Derive a list of bounding boxes for each region.

    Parameters
    ----------
    region_properties : skimage
        Dictionary of region properties

    Returns
    -------
    dict
        Dictionary of bounding boxes indexed by region area.


    Examples
    --------
    FIXME: Add docs.

    """
    LOGGER.info(f'Extracting bounding boxes')
    return {region.area: region.area_bbox for region in region_properties}

def save_region_stats(bounding_boxes: dict, outdir: Union[str, Path]) -> None:
    """Save the bounding box statistics.

    :param bounding_boxes: A dictionary of bounding boxes.
    :param outdir: Where to save the statistics to."""
    grainstats = pd.DataFrame.from_dict(data=bounding_boxes, orient='index')
    grainstats.to_csv(outdir / 'grainstats.csv', index=True)
    LOGGER.info(f'Saved region statistics to {str(outdir / "grainstats.csv")}')

def find_images(base_dir: Union[str, Path] = None, file_ext: str = '.spm') -> List:
    """Scan the specified directory for images with the given file extension.

    Parameters
    ----------
    base_dir: Union[str, Path]
        Directory to recursively search for files, if not specified the current directory is scanned.
    file_ext: str
        File extension to search for.

    Returns
    -------
    List
        List of files found with the extension in the given directory.
    """
    base_dir = Path('./') if base_dir is None else Path(base_dir)
    return list(base_dir.glob('**/*' + file_ext))

def process_scan(message = 'WTF?',
                 image: Union[str, Path] = None,
                 output_dir: Union[str, Path] = 'output') -> None:
    LOGGER.info(f'{message}')
    LOGGER.info(f'Processing : {image}')
    print(f'type(image) : {type(image)}')
    # Create output directory
#    filename = extract_filename(image_path)
    LOGGER.info(f'filename : {filename}')
    LOGGER.info(f'output_dir : {output_dir}')
    output_dir = Path(output_dir)
#    output_dir = output_dir / f'{filename}'
    LOGGER.info(f'output_dir : {output_dir}')
#    output_dir.mkdir(parents=True, exist_ok=True)
    # Create output directory
#    filename = extract_filename(image)
#    print(f'#### {filename}')
    # output_dir = Path(output_dir) / f'{filename}'
    # output_dir.mkdir(parents=True, exist_ok=True)

# def process_scan(image_path: Union[str, Path],
#                  channel: str = 'Height',
#                  amplify_level: float = 1.0,
#                  gaussian_size: Union[int, float] = 2,
#                  dx: Union[int, float] = 1,
#                  upper_height_threshold_rms_multiplier: Union[int, float] = 1,
#                  lower_threshold_otsu_multiplier: Union[int, float] = 1.7,
#                  minimum_grain_size: Union[int, float] = 800,
#                  mode: str = 'nearest',
#                  background: float = 0,
#                  output_dir: Union[str, Path] = 'output') -> np.array:
#     """Find grains within an image.

#     This function processes a single image file, applying several different passes of filtering to remove noise and then
#     identifies grains within an image.

#     Parameters
#     ----------
#     image_path : Union[str, Path]
#         Path to .spm image to process.
#     channel: str
#         Channel to extract and process.
#     amplify_level: float
#         Level to amplify channel by.
#     gaussian_size : Union[int, float]
#         Gaussian blur size in nanometers (nm).
#     dx : Union[int, float]
#         Pixel to nanometre scale.
#     upper_height_threshold_rms_multiplier : Union[int, float]
#         Sets the RMS multiplier for high-value outlier detection The higher the value the less sensitive grain finding
#         will be to outliers.
#     lower_threshold_otsu_multiplier : Union[int, float]
#         Otsu threshold multiplier for lower thresholding the data. Higher values cull lower values around the edges of
#         grains.
#     minimum_grain_size : Union[int, float]
#         Minimum grain size in nanometers (nm).
#     mode : str
#         Method for Gaussian filtering (default is 'nearest').
#     background: float

#     output_dir : Union[str, Path]
#         Output directory to save images and files to. A sub-directory with the image filename will be created within
#         this location to hold all output.

#     Returns
#     -------
#     np.array

#     Examples
#     --------
#     FIXME: Add docs.


#     """
#     # Create output directory
#     filename = extract_filename(image_path)
#     output_dir = Path(output_dir) / f'{filename}'
#     output_dir.mkdir(parents=True, exist_ok=True)

#     # Load image and extract channel
#     image = load_scan(image_path)
#     extracted_channel = extract_channel(image)
#     pixels = extract_pixels(extracted_channel)
#     plot_and_save(pixels,
#                   output_dir / '01-raw_heightmap.png',
#                   title='Raw Height')

#     # First pass filtering (no mask)
#     initial_align = align_rows(pixels, mask=None)
#     plot_and_save(initial_align,
#                   output_dir / '02-initial_align_unmasked.png',
#                   title='Initial Alignment (Unmasked)')
#     initial_tilt_removal = remove_x_y_tilt(initial_align, mask=None)
#     plot_and_save(initial_tilt_removal,
#                   output_dir / '03-initial_tilt_removal_unmasked.png',
#                   title='Initial Tilt Removal (Unmasked)')

#     # Create mask
#     threshold = get_threshold(initial_tilt_removal)
#     mask = get_mask(initial_tilt_removal, threshold)
#     plot_and_save(mask,
#                   output_dir / '04-binary_mask.png',
#                   title='Binary Mask')

#     # Second pass filtering (with mask based on threshold)
#     second_align = align_rows(initial_tilt_removal, mask=mask)
#     plot_and_save(second_align,
#                   output_dir / '05-secondary_align_masked.png',
#                   title='Secondary Alignment (Masked)')
#     second_tilt_removal = remove_x_y_tilt(second_align, mask=mask)
#     plot_and_save(second_tilt_removal,
#                   output_dir / '06-secondary_tilt_removal_masked.png',
#                   title='Secondary Tilt Removal (Masked)')

#     # Average background
#     averaged_background = average_background(second_tilt_removal, mask=mask)
#     plot_and_save(averaged_background,
#                   output_dir / '07-zero_average_background.png',
#                   title='Zero Average Background')

#     # Find grains, first apply a gaussian filter
#     gaussian_filtered = gaussian_filter(averaged_background,
#                                         gaussian_size=gaussian_size,
#                                         dx=dx,
#                                         mode=mode)
#     plot_and_save(gaussian_filtered,
#                   output_dir / '08-gaussian_filtered.png',
#                   title='Gaussian Filtered')

#     # Create a boolean image
#     boolean_image = boolean_image(gaussian_filtered, lower_threshold=lower_threshold_otsu_multiplier)
#     plot_and_save(boolean_image,
#                   output_dir / '09-boolean.png',
#                   title='Boolean Mask')

#     # Tidy borders
#     tidied_borders = tidy_border(boolean_image)
#     plot_and_save(tidied_borders,
#                   output_dir / '10-tidy_borders.png',
#                   title='Tidied Borders')

#     # Remove objects
#     small_objects_removed = remove_objects(tidied_borders,
#                                            minimum_grain_size=minimum_grain_size,
#                                            dx=dx)
#     plot_and_save(small_objects_removed,
#                   output_dir / '11-small_objects_removed.png',
#                   title='Small Objects Removed')

#     # Label regions
#     regions_labelled = label_regions(small_objects_removed, background=grain_config['background'])
#     plot_and_save(regions_labelled,
#                   output_dir / '12-labelled_regions.png',
#                   title='Labelled Regions')

#     coloured_regions = colour_regions(regions_labelled)
#     plot_and_save(coloured_regions,
#                   output_dir / '13-coloured_regions.png',
#                   title='Coloured Regions')
#     image_region_properties = region_properties(regions_labelled)
#     plot_and_save(colured_regions,
#                   output_dir / '14-bouinding_boxes.png',
#                   title='Bounding Boxes')

#     bounding_boxes = get_bounding_boxes(image_region_properties)
#     save_region_stats(bounding_boxes, output_dir = output_dir)
