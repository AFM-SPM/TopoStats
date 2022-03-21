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

# Fetch base path
basepath = os.getcwd()

# ----------------- CONFIGURATION --------------------------

# Configure logging functionality
now = datetime.now()
date_time = now.strftime("%Y-%m-%d--%H-%M-%S")
print(date_time)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%a, %d %b %Y %H:%M:%S', filename=str(basepath + '/topostats/logs/' + date_time + '.log'), filemode='w')

logging.info(f'pySPM version: {pySPM.__version__}')

# Misc config
# Matplotlib configuration
mpl.rcParams['figure.dpi'] = 150

#  ------------------ MAIN -------------------------------

# Iterate through all directories searching for spm files
data_files = []
for root, dirs, files in os.walk(basepath):
    file_list = []
    # Add all spm files to file list
    for file in files:
        if file.endswith('.spm'):
            file_list.append(file)
        
        # Create plot data folder if there are any spm files
        if len(file_list) > 0:
            if not os.path.exists(root + '/plot_data'):
                os.makedirs(root + '/plot_data')

    for file in file_list:
        scan = pySPM.Bruker(file)
        # scan.list_channels()
        height = scan.get_channel("Height")
        plottingfuncs.plot_and_save(height, 'plot_data/raw_heightmap.png')

        # Initial processing

        # Copy data into numpy array format
        data_initial_flatten = np.flipud(np.array(height.pixels))

        # Initial flattening
        logging.info('initial flattening')
        logging.info('initial align rows')
        data_initial_flatten = filters.align_rows(data_initial_flatten)
        plottingfuncs.plot_and_save(data_initial_flatten, 'plot_data/initial_align_rows.png')
        logging.info('initial x-y tilt')
        data_initial_flatten = filters.remove_x_y_tilt(data_initial_flatten)
        plottingfuncs.plot_and_save(data_initial_flatten, 'plot_data/initial_x_y_tilt.png')

        # Thresholding
        logging.info('otsu thresholding')
        threshold = filters.get_threshold(data_initial_flatten)
        logging.info(f'threshold: {threshold}')
        mask = data_initial_flatten > threshold
        plottingfuncs.plot_and_save(mask, 'plot_data/binary_mask.png')

        # Masked flattening
        logging.info('masked flattening')
        logging.info('masked align rows')
        data_second_flatten = filters.align_rows(data_initial_flatten, binary_mask=mask)
        plottingfuncs.plot_and_save(data_second_flatten, 'plot_data/masked_align_rows.png')
        logging.info('masked x-y tilt')
        data_second_flatten = filters.remove_x_y_tilt(data_second_flatten, binary_mask=mask)
        plottingfuncs.plot_and_save(data_second_flatten, 'plot_data/masked_x_y_tilt.png')

        # Zero the average background
        logging.info('adjust medians')
        row_quantiles, col_quantiles = filters.row_col_quantiles(data_second_flatten, binary_mask=mask)
        for row_index in range(data_second_flatten.shape[0]):
            row_zero_offset = row_quantiles[row_index, 1]
            data_second_flatten[row_index, :] -= row_zero_offset
        row_quantiles, col_quantiles = filters.row_col_quantiles(data_second_flatten, binary_mask=mask)
        logging.info(f'mean row median: {np.mean(row_quantiles)}')
        plottingfuncs.plot_and_save(data_second_flatten, 'plot_data/final_output.png')

    
