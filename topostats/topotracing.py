# Imports
from pathlib import Path
from datetime import datetime

import filters
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import pySPM
import os
import logging
import plottingfuncs
from skimage import filters as skimage_filters
from skimage import segmentation as skimage_segmentation
from skimage import measure as skimage_measure
from skimage import morphology as skimage_morphology
from skimage import color as skimage_color
from scipy import ndimage
# Fetch base path
basepath = os.getcwd()

# ----------------- CONFIGURATION --------------------------

# Configure logging functionality
now = datetime.now()
date_time = now.strftime("%Y-%m-%d--%H-%M-%S")
print(date_time)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(filename)s: %(message)s', datefmt='%a, %d %b %Y %H:%M:%S', filename=str(basepath + '/topostats/logs/' + date_time + '.log'), filemode='w')
logging.getLogger('pySPM').setLevel(logging.CRITICAL)
logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
logging.getLogger('skimage').setLevel(logging.CRITICAL)
logging.getLogger('skimage_filters').setLevel(logging.CRITICAL)
logging.getLogger('numpy').setLevel(logging.CRITICAL)
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.CRITICAL)
logging.info(f'pySPM version: {pySPM.__version__}')

# Misc config
# Matplotlib configuration
mpl.rcParams['figure.dpi'] = 150
OUT_DIR = Path('/home/neil/tmp/TopoStats/tmp/original/')
#  ------------------ MAIN -------------------------------

# Create main output folder
if not os.path.exists(basepath + '/plot_data/'):
            os.makedirs(basepath + '/plot_data')

# Iterate through all directories searching for spm files
file_list = []
for root, dirs, files in os.walk(basepath):
    # Add all spm files to file list
    for file in files:
        if file.endswith('.spm'):
            logging.info('File found: ' + os.path.join(root, file))
            file_list.append(file)

for file in file_list:
    filename = os.path.splitext(file)[0]

    logging.info('Processing file: ' + str(filename))

    # Create plot data folder for each spm image
    if not os.path.exists(basepath + '/plot_data/' + filename):
        os.makedirs(basepath + '/plot_data/' + filename)
    data_folder = basepath + '/plot_data/' + filename + '/'

    # Create flattening sub folder
    if not os.path.exists(data_folder + 'flattening/'):
        os.makedirs(data_folder + 'flattening/')
    flattening_folder = data_folder + 'flattening/'


    # Fetch the data
    scan = pySPM.Bruker(file)
    # scan.list_channels()
    height = scan.get_channel("Height")
    plottingfuncs.plot_and_save(height, flattening_folder + 'raw_heightmap.png')
    plottingfuncs.plot_and_save(height, OUT_DIR / '01_raw_heightmap.png')

    # Initial processing

    # Copy data into numpy array format
    data_initial_flatten = np.flipud(np.array(height.pixels))

    # Initial flattening
    logging.info('initial flattening')
    logging.info('initial align rows')
    data_initial_flatten = filters.align_rows(data_initial_flatten)
    plottingfuncs.plot_and_save(data_initial_flatten, flattening_folder + 'initial_align_rows.png')
    plottingfuncs.plot_and_save(data_initial_flatten, OUT_DIR / '02_initial_align_rows.png')
    logging.info('initial x-y tilt')
    data_initial_flatten = filters.remove_x_y_tilt(data_initial_flatten)
    plottingfuncs.plot_and_save(data_initial_flatten, flattening_folder + 'initial_x_y_tilt.png')
    plottingfuncs.plot_and_save(data_initial_flatten, OUT_DIR / '03_initial_x_y_tilt.png')

    # Thresholding
    logging.info('otsu thresholding')
    threshold = filters.get_threshold(data_initial_flatten)
    logging.info(f'threshold: {threshold}')

    # Create a mask that defines what data is used
    mask = filters.get_mask(data_initial_flatten, threshold)
    logging.info(f'values exceeding threshold {mask.sum()}')
    plottingfuncs.plot_and_save(mask, flattening_folder + 'binary_mask.png')
    plottingfuncs.plot_and_save(mask, OUT_DIR / '04_binary_mask.png')

    # Masked flattening
    logging.info('masked flattening')
    logging.info('masked align rows')
    data_second_flatten = filters.align_rows(data_initial_flatten, mask=mask)
    plottingfuncs.plot_and_save(data_second_flatten, flattening_folder + 'masked_align_rows.png')
    plottingfuncs.plot_and_save(data_second_flatten, OUT_DIR / '05_masked_align_rows.png')
    logging.info('masked x-y tilt')
    data_second_flatten = filters.remove_x_y_tilt(data_second_flatten, mask=mask)
    plottingfuncs.plot_and_save(data_second_flatten, flattening_folder + 'masked_x_y_tilt.png')
    plottingfuncs.plot_and_save(data_second_flatten, OUT_DIR / '06_masked_x_y_tilt.png')

    # Zero the average background
    logging.info('adjust medians')
    row_quantiles, col_quantiles = filters.row_col_quantiles(data_second_flatten, mask=mask)
    for row_index in range(data_second_flatten.shape[0]):
        row_zero_offset = row_quantiles[row_index, 1]
        data_second_flatten[row_index, :] -= row_zero_offset
    row_quantiles, col_quantiles = filters.row_col_quantiles(data_second_flatten, mask=mask)
    logging.info(f'mean row median: {np.mean(row_quantiles)}')
    plottingfuncs.plot_and_save(data_second_flatten, flattening_folder + 'final_output.png')
    plottingfuncs.plot_and_save(data_second_flatten, OUT_DIR / '07_final_output.png')

    # Remove x bowing
    # filters.remove_x_bowing(data_second_flatten, mask=mask)

    # Grain finding

    gaussian_size = 2
    dx = 1
    gaussian = gaussian_size / dx
    upper_height_threshold_rms_multiplier = 1
    lower_threshold_otsu_multiplier = 1.7
    minimum_grain_size_nm = 800

    # Create grain imaging sub folder
    if not os.path.exists(data_folder + 'grain_finding'):
        os.makedirs(data_folder + 'grain_finding')
    grain_finding_folder = data_folder + 'grain_finding/'

    data = np.copy(data_second_flatten)

    # Lower threshold
    lower_threshold = filters.get_threshold(data) * lower_threshold_otsu_multiplier
    logging.info('lower threshold: ' + str(lower_threshold))

    # Gaussian filter
    gaussian = gaussian_size / dx
    data = skimage_filters.gaussian(data, sigma=gaussian, output=None, mode='nearest')
    plottingfuncs.plot_and_save(data, grain_finding_folder + 'gaussian_filter.png')
    plottingfuncs.plot_and_save(data, OUT_DIR / '08_gaussian_filter.png')

    # Create copy of the data of boolean type where nonzero values are True
    data_boolean = np.copy(data)
    data_boolean[data_boolean <= lower_threshold] = False
    data_boolean[data_boolean > lower_threshold] = True
    data_boolean = data_boolean.astype(bool)

    # Remove grains touching border
    data_boolean = skimage_segmentation.clear_border(data_boolean)
    plottingfuncs.plot_and_save(data_boolean, grain_finding_folder + 'data_boolean_border_cleared.png')
    plottingfuncs.plot_and_save(data_boolean, OUT_DIR / '09_data_boolean_border_cleared.png')

    # Remove small objects
    # Calculate pixel are equivalent of minimum size in square nanometers
    minimum_grain_size_pixels = np.round(minimum_grain_size_nm / dx)
    data_boolean = skimage_morphology.remove_small_objects(data_boolean, min_size=minimum_grain_size_pixels)
    plottingfuncs.plot_and_save(data_boolean, grain_finding_folder + 'data_boolean_cull_small_objects.png')
    plottingfuncs.plot_and_save(data_boolean, OUT_DIR / '10_data_boolean_cull_small_objects.png')

    # Label regions
    labelled_data = skimage_morphology.label(data_boolean, background=0)
    plottingfuncs.plot_and_save(labelled_data, grain_finding_folder + 'data_boolean_labelled_image.png')
    plottingfuncs.plot_and_save(labelled_data, OUT_DIR / '11_data_boolean_labelled_image.png')

    # Colour the regions
    labelled_data_colour_overlay = skimage_color.label2rgb(labelled_data)
    plottingfuncs.plot_and_save(labelled_data_colour_overlay, grain_finding_folder + 'data_boolean_color_labelled_image.png')
    plottingfuncs.plot_and_save(labelled_data, OUT_DIR / '12_data_boolean_color_labelled_image.png')


     # Calculate region properties
    region_properties = skimage_measure.regionprops(labelled_data)

    # Add bounding boxes to the grains and save their stats into a dataframe
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(labelled_data, interpolation='nearest', cmap='afmhot')
    stats_array = []
    for region in region_properties:
        stats = {
            'area'      : region.area,
            'area_bbox' : region.area_bbox
        }
        stats_array.append(stats)

        min_row, min_col, max_row, max_col = region.bbox
        rectangle = mpl.patches.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row,
                                            fill=False, edgecolor='white', linewidth=2)
        ax.add_patch(rectangle)

    grainstats = pd.DataFrame(data=stats_array)
    grainstats.to_csv(grain_finding_folder + 'grainstats.csv')

    plt.savefig(grain_finding_folder + 'labelled_image_bboxes.png')
    plottingfuncs.plot_and_save(labelled_data, OUT_DIR / '13_labelled_image_bboxes.png')
