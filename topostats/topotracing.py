# Imports
from datetime import datetime

import filters
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pySPM
import os
import logging
import plottingfuncs
from skimage import filters as skimage_filters
from skimage import segmentation as skimage_segmentation
from skimage import measure as skimage_measure
from skimage import color as skimage_color

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

    # Initial processing

    # Copy data into numpy array format
    data_initial_flatten = np.flipud(np.array(height.pixels))

    # Initial flattening
    logging.info('initial flattening')
    logging.info('initial align rows')
    data_initial_flatten = filters.align_rows(data_initial_flatten)
    plottingfuncs.plot_and_save(data_initial_flatten, flattening_folder + 'initial_align_rows.png')
    logging.info('initial x-y tilt')
    data_initial_flatten = filters.remove_x_y_tilt(data_initial_flatten)
    plottingfuncs.plot_and_save(data_initial_flatten, flattening_folder + 'initial_x_y_tilt.png')

    # Thresholding
    logging.info('otsu thresholding')
    threshold = filters.get_threshold(data_initial_flatten)
    logging.info(f'threshold: {threshold}')

    # Create a mask that defines what data is used
    mask = filters.get_mask(data_initial_flatten, threshold)
    logging.info(f'values exceeding threshold {mask.sum()}')
    plottingfuncs.plot_and_save(mask, flattening_folder + 'binary_mask.png')

    # Masked flattening
    logging.info('masked flattening')
    logging.info('masked align rows')
    data_second_flatten = filters.align_rows(data_initial_flatten, binary_mask=mask)
    plottingfuncs.plot_and_save(data_second_flatten, flattening_folder + 'masked_align_rows.png')
    logging.info('masked x-y tilt')
    data_second_flatten = filters.remove_x_y_tilt(data_second_flatten, binary_mask=mask)
    plottingfuncs.plot_and_save(data_second_flatten, flattening_folder + 'masked_x_y_tilt.png')

    # Zero the average background
    logging.info('adjust medians')
    row_quantiles, col_quantiles = filters.row_col_quantiles(data_second_flatten, binary_mask=mask)
    for row_index in range(data_second_flatten.shape[0]):
        row_zero_offset = row_quantiles[row_index, 1]
        data_second_flatten[row_index, :] -= row_zero_offset
    row_quantiles, col_quantiles = filters.row_col_quantiles(data_second_flatten, binary_mask=mask)
    logging.info(f'mean row median: {np.mean(row_quantiles)}')
    plottingfuncs.plot_and_save(data_second_flatten, flattening_folder + 'final_output.png')

    # Remove x bowing
    # filters.remove_x_bowing(data_second_flatten, binary_mask=mask)

    # Grain finding

    gaussian_size = 2
    dx = 1
    gaussian = gaussian_size / dx

    # Create grain imaging sub folder
    if not os.path.exists(data_folder + 'grain_finding'):
        os.makedirs(data_folder + 'grain_finding')
    grain_finding_folder = data_folder + 'grain_finding/'

    data = np.copy(data_second_flatten)

    # Gaussian filter
    data = skimage_filters.gaussian(data, sigma=gaussian, output=None, mode='nearest')
    plottingfuncs.plot_and_save(data, grain_finding_folder + 'gaussian_filter.png')

    # Create inverted mask for the grain mask
    grain_mask = np.invert(mask)
    plottingfuncs.plot_and_save(grain_mask, grain_finding_folder + 'grain_mask.png')

    # Mask the data
    masked_data = np.ma.masked_array(data, mask=grain_mask, fill_value=np.nan)
    plottingfuncs.plot_and_save(masked_data, grain_finding_folder + 'masked_data.png')

    # Calculate the RMS
    rms_height = np.sqrt(np.mean(masked_data**2))
    logging.info('rms_height: ' + str(rms_height))

    # Calculate the mean
    mean_height = np.mean(masked_data)

    # Set outlier threshold value TODO: Add to config file.
    threshold = 1

    # Mask out any data that is above a threshold value * sigma above the average height. 
    for i in range(masked_data.shape[0]):
        for j in range(masked_data.shape[1]):
            value = masked_data[i, j]
            if value - mean_height >= threshold * rms_height:
                grain_mask[i, j] = True

    # plottingfuncs.plot_and_save(masked_data, grain_finding_folder + 'masked_data_thresholded.png')
    plottingfuncs.plot_and_save(grain_mask, grain_finding_folder + 'grain_mask_thresholded.png')
    # Apply the mask
    masked_data = np.ma.masked_array(masked_data, mask=grain_mask, fill_value=0.0).filled()
    plottingfuncs.plot_and_save(masked_data, grain_finding_folder + 'masked_data_thresholded.png')

    # Remove grains touching border
    grain_mask = np.invert(skimage_segmentation.clear_border(np.invert(grain_mask)))

    plottingfuncs.plot_and_save(grain_mask, grain_finding_folder + 'grain_mask_border_cleared.png')
    # Apply the mask
    masked_data = np.ma.masked_array(masked_data, mask=grain_mask, fill_value=0.0).filled()
    plottingfuncs.plot_and_save(masked_data, grain_finding_folder + 'masked_data_border_cleared.png')
