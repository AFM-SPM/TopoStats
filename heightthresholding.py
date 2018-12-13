#!/usr/bin/env python2

import gwy

# Set the settings for each function from the saved settings file (~/.gwyddion/settings)
s = gwy.gwy_app_settings_get()
# Turn colour bar off
s["/module/pixmap/ztype"] = 0
# Define the settings for image processing functions e.g. align rows here
s['/module/linematch/method'] = 1
# 'align_rows' function
s["/module/linematch/direction"] = 0
s["/module/linematch/do_extract"] = False
s["/module/linematch/do_plot"] = False
s["/module/linematch/masking"] = 2
s["/module/linematch/max_degree"] = 0
s["/module/linematch/method"] = 0  # uses median
s["/module/linematch/trim_fraction"] = float(0.05)


def otsuthresholdgrainfinding(data, k):
    # 'align_rows' function
    # s["/module/linematch/direction"] = 0
    # s["/module/linematch/do_extract"] = False
    # s["/module/linematch/do_plot"] = False
    # s["/module/linematch/masking"] = 2
    # s["/module/linematch/max_degree"] = 0
    # s["/module/linematch/method"] = 0  # uses median
    # s["/module/linematch/trim_fraction"] = float(0.05)

    # Select channel 'k' of the file
    gwy.gwy_app_data_browser_select_data_field(data, k)
    datafield = gwy.gwy_app_data_browser_get_current(gwy.APP_DATA_FIELD)
    mask = gwy.DataField.new_alike(datafield, False)

    # Apply a 1.5 pixel gaussian filter
    data_field = gwy.gwy_app_data_browser_get_current(gwy.APP_DATA_FIELD)
    data_field.filter_gaussian(1.5)

    # # Mask data that are above thresh*sigma from average height.
    # # Sigma denotes root-mean square deviation of heights.
    # # This criterium corresponds to the usual Gaussian distribution outliers detection if thresh is 3.
    # datafield.mask_outliers(mask, 1)

    # # Shift contrast - equivalent to 'fix zero' - essential for next step to work
    datafield.add(-datafield.get_min())
    # Calculate min, max and range of data to allow calculation of relative value for grain thresholding
    min_datarange = datafield.get_min()
    max_datarange = datafield.get_max()
    datarange = max_datarange + min_datarange
    # Calculate Otsu threshold for data
    o_threshold = datafield.otsu_threshold()
    o_threshold = o_threshold + min_datarange
    # Calculate relative threshold for grain determination
    rel_threshold = 100 * (o_threshold / datarange)
    # Mask grains using either relative threshold of the data, i.e. a percentage
    # this can be set manually i.e. 35 or
    # as the otsu threshold expressed as a percentage which is rel_threshold
    # this will fail unless the min_datarange is negative
    datafield.grains_mark_height(mask, rel_threshold, False)

    # # Invert mask to maks things below the membrane
    mask.grains_invert()

    gwy.gwy_process_func_run("align_rows", data, gwy.RUN_IMMEDIATE)
    # # gwy.gwy_process_func_run("level", data, gwy.RUN_IMMEDIATE)
    # gwy.gwy_process_func_run('flatten_base', data, gwy.RUN_IMMEDIATE)

    # Select channel 'k' of the file
    gwy.gwy_app_data_browser_select_data_field(data, k)
    datafield = gwy.gwy_app_data_browser_get_current(gwy.APP_DATA_FIELD)
    mask = gwy.DataField.new_alike(datafield, False)
    # # Mask data that are above thresh*sigma from average height.
    # # Sigma denotes root-mean square deviation of heights.
    # # This criterium corresponds to the usual Gaussian distribution outliers detection if thresh is 3.
    # datafield.mask_outliers(mask, 1)

    # # Shift contrast - equivalent to 'fix zero' - essential for next step to work
    datafield.add(-datafield.get_min())
    # Calculate min, max and range of data to allow calculation of relative value for grain thresholding
    min_datarange = datafield.get_min()
    max_datarange = datafield.get_max()
    datarange = max_datarange + min_datarange
    # Calculate Otsu threshold for data
    o_threshold = datafield.otsu_threshold()
    o_threshold = o_threshold + min_datarange
    # Calculate relative threshold for grain determination
    rel_threshold = 100 * (o_threshold / datarange)
    # Mask grains using either relative threshold of the data, i.e. a percentage
    # this can be set manually i.e. 35 or
    # as the otsu threshold expressed as a percentage which is rel_threshold
    # this will fail unless the min_datarange is negative
    datafield.grains_mark_height(mask, rel_threshold, False)

    gwy.gwy_process_func_run('zero_mean', data, gwy.RUN_IMMEDIATE)

    # # Invert mask to make things below the membrane
    mask.grains_invert()

    # Calculate pixel width in nm
    dx = datafield.get_dx()
    # Calculate minimum feature size in pixels (integer)
    minsize = int(2000e-9 / dx)
    # Remove grains smaller than (size) in integer pixels
    mask.grains_remove_by_size(minsize)

    mask.grains_invert()
    gwy.gwy_process_func_run('zero_mean', data, gwy.RUN_IMMEDIATE)
    mask.grains_invert()

    # Numbering grains for grain analysis
    grains = mask.number_grains()
    print max(grains)

    return data, mask, datafield, grains