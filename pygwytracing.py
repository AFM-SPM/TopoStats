#!/usr/bin/env python2

import gwy
import gwyutils
import os
import fnmatch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

### Set seaborn to override matplotlib for plot output
sns.set()
# ###The four preset contexts, in order of relative size, are paper, notebook, talk, and poster.
# ### The notebook style is the default
sns.set_context("notebook")
# ### This can be customised further here
# sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 2.5})

# sys.path.append("/usr/local/Cellar/gwyddion/2.52/share/gwyddion/pygwy")

# Set the settings for each function from the saved settings file (~/.gwyddion/settings)
s = gwy.gwy_app_settings_get()

# Generate a settings file - should be found at /Users/alice/.gwyddion/settings
# a = gwy.gwy_app_settings_get_settings_filename()
# # Location of the settings file - edit to change values
# print a

# Turn colour bar off
s["/module/pixmap/ztype"] = 0

# Define the settings for image processing functions e.g. align rows here
s['/module/linematch/method'] = 1


def traversedirectories(fileend, filetype, path):
    # This function finds all the files with the file ending set in the main script as fileend (usually.spm)
    # in the path directory, and all subfolder

    # initialise the list
    spmfiles = []
    # use os.walk to search folders and subfolders and append each file with the correct filetype to the list spmfiles
    for dirpath, subdirs, files in os.walk(path):
        # Looking for files ending in fileend
        for filename in files:
            if filename.endswith(fileend):
                spmfiles.append(os.path.join(dirpath, filename))
        # Looking for files of a certain filetype
        for filename in fnmatch.filter(files, filetype):
            spmfiles.append(os.path.join(dirpath, filename))
        # print the number of files found
    print 'Files found: ' + str(len(spmfiles))
    # return a list of files including their root and the original path specified
    return spmfiles


def getdata(filename):
    # Open file and add data to browser
    data = gwy.gwy_file_load(filename, gwy.RUN_NONINTERACTIVE)
    # Add data to browser
    gwy.gwy_app_data_browser_add(data)
    return data


def choosechannels(data):
    # Obtain the data field ids for the data file (i.e. the various channels within the file)
    ids = gwy.gwy_app_data_browser_get_data_ids(data)
    # Make an empty array for the chosen channels to be saved into
    chosen_ids = []
    # Find channels with the title ZSensor, if these do not exist, find channels with the title Height
    chosen_ids = gwy.gwy_app_data_browser_find_data_by_title(data, 'ZSensor')
    if not chosen_ids:
        chosen_ids = gwy.gwy_app_data_browser_find_data_by_title(data, 'Height')
    else:
        chosen_ids = chosen_ids
    # Save out chosen_ids as a list of the channel ids for ZSensor or height if ZSensor doesnt exist
    return chosen_ids


def imagedetails(data):
    # select the channel file of chosen_ids
    gwy.gwy_app_data_browser_select_data_field(data, k)
    datafield = gwy.gwy_app_data_browser_get_current(gwy.APP_DATA_FIELD)

    xres = datafield.get_xres()
    yres = datafield.get_yres()
    xreal = datafield.get_xreal()
    yreal = datafield.get_yreal()
    dx = datafield.get_dx()
    dy = datafield.get_dy()
    return xres, yres, xreal, yreal, dx, dy


def editfile(data, minheightscale, maxheightscale):
    # select each channel of the file in turn
    # this is run within the for k in chosen_ids loop so k refers to the index of each chosen channel to analyse
    gwy.gwy_app_data_browser_select_data_field(data, k)
    # flatten the data
    gwy.gwy_process_func_run('flatten_base', data, gwy.RUN_IMMEDIATE)
    # level the data
    gwy.gwy_process_func_run("level", data, gwy.RUN_IMMEDIATE)
    # align the rows
    gwy.gwy_process_func_run("align_rows", data, gwy.RUN_IMMEDIATE)  # NONINTERACTIVE is only for file modules
    # Fix zero
    gwy.gwy_process_func_run('zero_mean', data, gwy.RUN_IMMEDIATE)
    # remove scars
    gwy.gwy_process_func_run('scars_remove', data, gwy.RUN_IMMEDIATE)
    # Apply a 1.5 pixel gaussian filter
    data_field = gwy.gwy_app_data_browser_get_current(gwy.APP_DATA_FIELD)
    data_field.filter_gaussian(1)
    # # Shift contrast - equivalent to 'fix zero'
    # datafield.add(-data_field.get_min())

    # Set the image display to fized range and the colour scale for the images
    maximum_disp_value = data.set_int32_by_name("/" + str(k) + "/base/range-type", int(1))
    minimum_disp_value = data.set_double_by_name("/" + str(k) + "/base/min", float(minheightscale))
    maximum_disp_value = data.set_double_by_name("/" + str(k) + "/base/max", float(maxheightscale))
    return data


def grainfinding(data, minarea, k):
    # Select channel 'k' of the file
    gwy.gwy_app_data_browser_select_data_field(data, k)
    datafield = gwy.gwy_app_data_browser_get_current(gwy.APP_DATA_FIELD)
    mask = gwy.DataField.new_alike(datafield, False)
    # Mask data that are above thresh*sigma from average height.
    # Sigma denotes root-mean square deviation of heights.
    # This criterium corresponds to the usual Gaussian distribution outliers detection if thresh is 3.
    datafield.mask_outliers(mask, 1)

    ### Editing grain mask
    # Remove grains touching the edge of the mask
    mask.grains_remove_touching_border()
    # Calculate pixel width in nm
    dx = datafield.get_dx()
    # Calculate minimum feature size in pixels (integer) from a real size specified in the main
    minsize = int(minarea / dx)
    # Remove grains smaller than the minimum size in integer pixels
    mask.grains_remove_by_size(minsize)

    # Numbering grains for grain analysis
    grains = mask.number_grains()

    # Update data to show mask, comment out to remove mask
    # data['/%d/mask' % k] = mask

    return data, mask, datafield, grains


def removelargeobjects(datafield, mask, median_pixel_area, maxdeviation):
    mask2 = gwy.DataField.new_alike(datafield, False)
    # Mask data that are above thresh*sigma from average height.
    # Sigma denotes root-mean square deviation of heights.
    # This criterium corresponds to the usual Gaussian distribution outliers detection if thresh is 3.
    datafield.mask_outliers(mask2, 1)
    # Calculate pixel width in nm
    dx = datafield.get_dx()
    # Calculate minimum feature size in pixels (integer)
    # here this is calculated as 2* the median grain size, as calculated in find_median_pixel_area()
    maxsize = int(maxdeviation * median_pixel_area)
    # Remove grains smaller than the maximum feature size in integer pixels
    # This should remove everything that you do want to keep
    # i.e. everything smaller than aggregates/junk
    mask2.grains_remove_by_size(maxsize)
    # Invert mask2 so everything smaller than aggregates/junk is masked
    mask2.grains_invert()
    # Make mask equalto the intersection of mask and mask 2, i.e. rmeove large objects unmasked by mask2
    mask.grains_intersect(mask2)

    # Numbering grains for grain analysis
    grains = mask.number_grains()

    return mask, grains


def removesmallobjects(datafield, mask, median_pixel_area, mindeviation):
    mask2 = gwy.DataField.new_alike(datafield, False)
    # Mask data that are above thresh*sigma from average height.
    # Sigma denotes root-mean square deviation of heights.
    # This criterium corresponds to the usual Gaussian distribution outliers detection if thresh is 3.
    datafield.mask_outliers(mask2, 1)
    # Calculate pixel width in nm
    dx = datafield.get_dx()
    # Calculate minimum feature size in pixels (integer)
    # here this is calculated as 2* the median grain size, as calculated in find_median_pixel_area()
    minsize = int(mindeviation * median_pixel_area)
    # Remove grains smaller than the maximum feature size in integer pixels
    # This should remove everything that you do want to keep
    # i.e. everything smaller than aggregates/junk
    mask2.grains_remove_by_size(minsize)
    # Make mask equalto the intersection of mask and mask 2, i.e. rmeove large objects unmasked by mask2
    mask.grains_intersect(mask2)

    # Numbering grains for grain analysis
    grains = mask.number_grains()
    print 'There were ' + str(max(grains)) + ' grains found'

    return mask, grains


def grainanalysis(directory, filename, datafield, grains):
    ### Calculating grain statistics using numbered grains file
    # Statistics to be computed should be specified here as a dictionary
    values_to_compute = {'grain_proj_area': gwy.GRAIN_VALUE_PROJECTED_AREA,
                         'grain_maximum': gwy.GRAIN_VALUE_MAXIMUM,
                         'grain_mean': gwy.GRAIN_VALUE_MEAN,
                         'grain_median': gwy.GRAIN_VALUE_MEDIAN,
                         'grain_pixel_area': gwy.GRAIN_VALUE_PIXEL_AREA,
                         'grain_half_height_area': gwy.GRAIN_VALUE_HALF_HEIGHT_AREA,
                         'grain_bound_len': gwy.GRAIN_VALUE_FLAT_BOUNDARY_LENGTH,
                         'grain_min_bound_size': gwy.GRAIN_VALUE_MINIMUM_BOUND_SIZE,
                         'grain_max_bound_size': gwy.GRAIN_VALUE_MAXIMUM_BOUND_SIZE,
                         'grain_center_x': gwy.GRAIN_VALUE_CENTER_X,
                         'grain_center_y': gwy.GRAIN_VALUE_CENTER_Y,
                         'grain_curvature1': gwy.GRAIN_VALUE_CURVATURE1,
                         'grain_curvature2': gwy.GRAIN_VALUE_CURVATURE2,
                         'grain_mean_radius': gwy.GRAIN_VALUE_MEAN_RADIUS,
                         'grain_ellipse_angle': gwy.GRAIN_VALUE_EQUIV_ELLIPSE_ANGLE,
                         'grain_ellipse_major': gwy.GRAIN_VALUE_EQUIV_ELLIPSE_MAJOR,
                         'grain_ellipse_minor': gwy.GRAIN_VALUE_EQUIV_ELLIPSE_MINOR,
                         }
    ### Create empty dictionary for grain data
    grain_data_to_save = {}

    for key in values_to_compute.keys():
        # here we stave the gran stats to both a dictionary and an array in that order
        # these are basically duplicate steps - but are both included as we arent sure which to use later
        # Save grain statistics to a dictionary: grain_data_to_save
        grain_data_to_save[key] = datafield.grains_get_values(grains, values_to_compute[key])
        # Delete 0th value in all arrays - this corresponds to the background
        del grain_data_to_save[key][0]

    grainstats = pd.DataFrame.from_dict(grain_data_to_save, orient='index').transpose()

    return values_to_compute, grainstats


def grainstatistics(datafield, grains, filename, result):
    ### Get only last part of filename without extension
    directory = str(os.path.dirname(filename))
    directory = str(os.path.splitext(os.path.basename(directory))[0])
    filename = os.path.splitext(os.path.basename(filename))[0]

    ### Calculate grain statistics
    # grain_bound_len = datafield.grains_get_values(grains, gwy.GRAIN_VALUE_FLAT_BOUNDARY_LENGTH)
    grain_min_bound = datafield.grains_get_values(grains, gwy.GRAIN_VALUE_MINIMUM_BOUND_SIZE)
    grain_max_bound = datafield.grains_get_values(grains, gwy.GRAIN_VALUE_MAXIMUM_BOUND_SIZE)
    grain_mean_rad = datafield.grains_get_values(grains, gwy.GRAIN_VALUE_MEAN_RADIUS)
    grain_proj_area = datafield.grains_get_values(grains, gwy.GRAIN_VALUE_PROJECTED_AREA)
    grain_max = datafield.grains_get_values(grains, gwy.GRAIN_VALUE_MAXIMUM)
    grain_med = datafield.grains_get_values(grains, gwy.GRAIN_VALUE_MEDIAN)

    ### Delete 0th value in all arrays - this corresponds to the background
    del grain_max_bound[0]
    del grain_min_bound[0]
    del grain_mean_rad[0]
    del grain_proj_area[0]
    del grain_max[0]
    del grain_med[0]

    ### Loop over list to get filename, grain number, and grain min and max bounding sizes
    for i in range(len(grain_min_bound)):
        resultsheader = 'filename, i, grain_min_bound[i], grain_max_bound[i], grain_mean_rad[i], grain_proj_area[i], grain_max[i], grain_med[i]'
        result.append(
            [directory, filename, i, grain_min_bound[i], grain_max_bound[i], grain_mean_rad[i], grain_proj_area[i],
             grain_max[i], grain_med[i]])

    ### Convert results to a pandas dataframe with column headings to save out
    grainstats_df = pd.DataFrame.from_records(result,
                                              columns=['directory', 'filename', 'i', 'grain_min_bound',
                                                       'grain_max_bound', 'grain_mean_rad',
                                                       'grain_proj_area', 'grain_max', 'grain_med'])

    return grainstats_df


def plotall(dataframe, bins, directory, outname, extension):
    ### Create a saving name format/directory
    savedir = os.path.join(directory, 'Plots')
    savename = os.path.join(savedir, os.path.splitext(os.path.basename(directory))[0])
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    df = dataframe
    for i, col in enumerate(df.columns):
        plt.figure(i)
        sns.distplot(df[col], bins=bins)
        # plt.hist(df[col])
        # df[col].plot.hist()
        plt.show()
        plt.savefig(savename + str(i) + extension)


def plotting(dataframe, arg1, grouparg, bins, directory, extension):
    ### Create a saving name format/directory
    savedir = os.path.join(directory, 'Plots')
    savename = os.path.join(savedir, os.path.splitext(os.path.basename(directory))[0])
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    df = dataframe

    ### Change from m to nm units for plotting
    df[arg1] = df[arg1] * 1e9

    ### Generating min and max axes based on datasets
    min_ax = df[arg1].min()
    min_ax = round(min_ax, 9)
    max_ax = df[arg1].max()
    max_ax = round(max_ax, 9)

    ### Plot using MatPlotLib separated by filetype on two separate graphs with stacking
    # Create a figure of given size
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111)
    # Set title
    ttl = 'Histogram of %s' % arg1
    # Pivot dataframe to get required variables in correct format for plotting
    df1 = df.pivot(columns=grouparg, values=arg1)
    # Plot histogram
    df1.plot.hist(ax=ax, legend=True, bins=bins, range=(min_ax, max_ax), alpha=.3, stacked=True)
    # Set x axis label
    plt.xlabel('%s (nm)' % arg1)
    # Set tight borders
    plt.tight_layout()
    # Set legend options
    # plt.legend(ncol=2, loc='upper right')
    # Save plot
    plt.savefig(savename + '_' + arg1 + '_a' + extension)

    ### Plot each argument together using MatPlotLib
    # Create a figure of given size
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111)
    # Set title
    ttl = 'Histogram of %s' % arg1
    # Melt dataframe to leave only columns we are interested in
    df3 = pd.melt(df, id_vars=[arg1])
    # Plot histogram
    df3.plot.hist(ax=ax, bins=bins, range=(min_ax, max_ax), alpha=.3)
    plt.xlabel('%s (nm)' % arg1)
    # # Set legend options
    # plt.legend(ncol=2, loc='upper right')
    # Set tight borders
    plt.tight_layout()
    # Save plot
    plt.savefig(savename + '_' + arg1 + '_b' + extension)


def seaplotting(df, arg1, arg2, grouparg, bins, directory, outname, extension):
    ### Create a saving name format/directory
    savedir = os.path.join(directory, 'Plots')
    savename = os.path.join(savedir, os.path.splitext(os.path.basename(directory))[0])
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    ### Change from m to nm units for plotting
    df[arg1] = df[arg1] * 1e9
    df[arg2] = df[arg2] * 1e9

    ### Generating min and max axes based on datasets
    min_ax = min(df[arg1].min(), df[arg2].min())
    min_ax = round(min_ax, 9)
    max_ax = max(df[arg1].max(), df[arg2].max())
    max_ax = round(max_ax, 9)

    ### Plot data
    with sns.axes_style('white'):
        sns.jointplot("grain_min_bound", "grain_max_bound", data=grainstats_df, kind='hex')
        sns.jointplot("grain_min_bound", "grain_max_bound", data=grainstats_df, kind='reg')


def plotting2(df, arg1, arg2, grouparg, bins, directory, extension):
    ### Create a saving name format/directory
    savedir = os.path.join(directory, 'Plots/')
    savename = os.path.join(savedir, os.path.splitext(os.path.basename(directory))[0])
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    ### Change from m to nm units for plotting
    df[arg1] = df[arg1] * 1e9
    df[arg2] = df[arg2] * 1e9

    ### Generating min and max axes based on datasets
    min_ax = min(df[arg1].min(), df[arg2].min())
    min_ax = round(min_ax, 9)
    max_ax = max(df[arg1].max(), df[arg2].max())
    max_ax = round(max_ax, 9)

    ### Plot data

    ### Plot each type using MatPlotLib separated by filetype on two separate graphs with stacking
    # Create a figure of given size
    fig = plt.figure(figsize=(28, 8))
    # First dataframe
    # Add a subplot to lot both sets on the same figure
    ax = fig.add_subplot(121)
    # Set title
    ttl = 'Histogram of %s and %s' % (arg1, arg2)
    # Pivot dataframe to get required variables in correct format for plotting
    df1 = df.pivot(columns=grouparg, values=arg1)
    # Plot histogram
    df1.plot.hist(legend=True, ax=ax, bins=bins, range=(min_ax, max_ax), alpha=.3, stacked=True)
    # Set x axis label
    plt.xlabel('%s (nm)' % arg1)
    # Set tight borders
    plt.tight_layout()
    # Set legend options
    # plt.legend(ncol=2, loc='upper right')
    # Second dataframe
    # Add a subplot
    ax = fig.add_subplot(122)
    # Pivot second dataframe to get required variables in correct format for plotting
    df2 = df.pivot(columns=grouparg, values=arg2)
    # Plot histogram
    df2.plot.hist(legend=True, ax=ax, bins=bins, range=(min_ax, max_ax), alpha=.3, stacked=True)
    # Set x axis label
    plt.xlabel('%s (nm)' % arg2)
    # Set tight borders
    plt.tight_layout()
    # Set legend options
    # plt.legend(ncol=2, loc='upper right')
    # Save plot
    plt.savefig(savename + '_' + arg1 + '_' + arg2 + '_' + 'a' + extension)

    # Create a figure of given size
    fig = plt.figure(figsize=(18, 12))
    # Add a subplot
    ax = fig.add_subplot(111)
    # Set title
    ttl = 'Histogram of %s and %s' % (arg1, arg2)
    ### Plot each argument together using MatPlotLib
    df3 = pd.melt(df, id_vars=[arg1, arg2])
    df3.plot.hist(legend=True, ax=ax, bins=bins, range=(min_ax, max_ax), alpha=.3)
    # plt.xlabel('%s %s (nm)' % (arg1, arg2))
    plt.xlabel('nm')
    # # Set legend options
    # plt.legend(ncol=2, loc='upper right')
    # Set tight borders
    plt.tight_layout()
    # Save plot
    plt.savefig(savename + '_' + arg1 + '_' + arg2 + '_' + 'b' + extension)


def find_median_pixel_area(datafield, grains):
    # print values_to_compute.keys()
    grain_pixel_area = datafield.grains_get_values(grains, gwy.GRAIN_VALUE_PIXEL_AREA)
    grain_pixel_area = np.array(grain_pixel_area)
    median_pixel_area = np.median(grain_pixel_area)
    return median_pixel_area


def boundbox(cropwidth, datafield, grains, dx, dy, xreal, yreal, xres, yres):
    # Function to return the coordinates of the bounding box for all grains.
    # contains 4 coordinates per image
    bbox = datafield.get_grain_bounding_boxes(grains)
    # Remove all data up to index 4 (i.e. the 0th, 1st, 2nd, 3rd).
    # These are the bboxes for grain zero which is the background and should be ignored
    del bbox[:4]

    # Find the center of each grain in x
    center_x = datafield.grains_get_values(grains, gwy.GRAIN_VALUE_CENTER_X)
    # Delete the background grain center
    del center_x[0]
    # Find the center of each grain in y
    center_y = datafield.grains_get_values(grains, gwy.GRAIN_VALUE_CENTER_Y)
    # Delete the background grain center
    del center_y[0]

    # Find the centre of the grain in pixels
    # get active container
    data = gwy.gwy_app_data_browser_get_current(gwy.APP_CONTAINER)
    # get current number of files
    orig_ids = gwy.gwy_app_data_browser_get_data_ids(data)

    # Define the width of the image to crop to
    cropwidth = int((cropwidth / xreal) * xres)

    for i in range(len(center_x)):
        px_center_x = int((center_x[i] / xreal) * xres)
        px_center_y = int((center_y[i] / yreal) * yres)
        ULcol = px_center_x - cropwidth
        if ULcol < 0:
            ULcol = 0
        ULrow = px_center_y - cropwidth
        if ULrow < 0:
            ULrow = 0
        BRcol = px_center_x + cropwidth
        if BRcol > xres:
            BRcol = xres
        BRrow = px_center_y + cropwidth
        if BRrow > yres:
            BRrow = yres
        crop_datafield_i = datafield.duplicate()
        crop_datafield_i.resize(ULcol, ULrow, BRcol, BRrow)
        # add cropped datafield to active container
        gwy.gwy_app_data_browser_add_data_field(crop_datafield_i, data, i + (len(orig_ids)))
    # Generate list of datafields including cropped fields
    crop_ids = gwy.gwy_app_data_browser_get_data_ids(data)

    return bbox, orig_ids, crop_ids, data


def grainthinning(data, mask, dx):
    # Calculate gaussian width in pixels from real value using pixel size
    Gaussiansize = 2e-9 / dx
    # Gaussiansize = 10
    # Gaussian filter data
    datafield.filter_gaussian(Gaussiansize)
    # Thin (skeletonise) gaussian filtered grains to get traces
    mask.grains_thin()
    return data, mask


def exportasnparray(datafield, mask):
    # Export the current datafield (channel) and mask (grains) as numpy arrays
    npdata = gwyutils.data_field_data_as_array(datafield)
    npmask = gwyutils.data_field_data_as_array(mask)
    return npdata, npmask


def savestats(directory, outname, dataframetosave):
    # Generate a filepath to save the files to using the directory and the 'outname' i.e. what you;d like to append to it
    # directory = os.getcwd()
    # savename = directory + '/' + str(os.path.splitext(os.path.basename(directory))[0]) + outname
    savedir = directory + '/' + outname + '/'
    savename = savedir + str(os.path.splitext(os.path.basename(directory))[0])
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    dataframetosave.to_json(savename + '.json')
    dataframetosave.to_csv(savename + '.txt')

    print 'Saving stats for: ' + str(os.path.splitext(os.path.basename(directory))[0])


def savefiles(data, filename, extension):
    # Turn rulers on
    s["/module/pixmap/xytype"] = 1
    # Save the data for the channels found above i.e. ZSensor/Height, as chosen_ids
    # Data is exported to a file of extension set in the main script
    # Data is exported with the string '_processed' added to the end of its filename
    gwy.gwy_app_data_browser_select_data_field(data, k)
    # change the colour map for all channels (k) in the image:
    palette = data.set_string_by_name("/" + str(k) + "/base/palette", "Nanoscope")
    # Determine the title of each channel
    title = data["/%d/data/title" % k]
    # Determine the filename for each file including path
    filename = os.path.splitext(filename)[0]
    # Generate a filename to save to by removing the extension to the file, adding the suffix '_processed'
    # and an extension set in the main file
    savename = filename + '_' + str(k) + '_' + str(title) + '_processed' + str(extension)
    # Save the file
    gwy.gwy_file_save(data, savename, gwy.RUN_NONINTERACTIVE)
    # Show the mask
    data['/%d/mask' % k] = mask
    # Add the sufix _masked to the previous filename
    savename = filename + '_' + str(k) + '_' + str(title) + '_processed_masked' + str(extension)
    # Save the data
    gwy.gwy_file_save(data, savename, gwy.RUN_NONINTERACTIVE)
    # Print the name of the file you're saving to the command line
    print 'Saving file: ' + str((os.path.splitext(os.path.basename(savename))[0]))


def savecroppedfiles(directory, data, filename, extension, orig_ids, crop_ids, minheightscale, maxheightscale):
    # Turn rulers off
    s["/module/pixmap/xytype"] = 0
    # Save the data for the cropped channels
    # Data is exported to a file of extension set in the main script
    # Data is exported with the string '_cropped' added to the end of its filename

    ### Get only last part of filename without extension
    directory = str(os.path.dirname(filename))
    filename = os.path.splitext(os.path.basename(filename))[0]
    # If the folder Cropped doest exist make it here
    if not os.path.exists(directory + '/Cropped/'):
        os.makedirs(directory + '/Cropped/')
        crop_directory = directory + '/Cropped/'
    # Otherwise set the existing Cropped directory as the directory to write to
    else:
        crop_directory = directory + '/Cropped/'

    # For each cropped file, save out the data with the suffix _Cropped_#
    for i in range(len(orig_ids), len(crop_ids), 1):
        # select each channel fo the file in turn
        gwy.gwy_app_data_browser_select_data_field(data, i)
        # change the colour map for all channels (k) in the image:
        palette = data.set_string_by_name("/" + str(i) + "/base/palette", "Nanoscope")
        # Set the image display to fized range and the colour scale for the images
        maximum_disp_value = data.set_int32_by_name("/" + str(i) + "/base/range-type", int(1))
        minimum_disp_value = data.set_double_by_name("/" + str(i) + "/base/min", float(minheightscale))
        maximum_disp_value = data.set_double_by_name("/" + str(i) + "/base/max", float(maxheightscale))
        # Generate a filename to save to by removing the extension to the file and a numerical identifier (the crop no)
        # The 'crop number' is the channel number minus the original number of channels, less one to avoid starting at 0
        savenumber = i - (len(orig_ids) - 1)
        # adding the suffix '_cropped' and adding the extension set in the main file
        savename = crop_directory + filename + '_cropped_' + str(savenumber) + str(extension)
        # Save the file
        gwy.gwy_file_save(data, savename, gwy.RUN_NONINTERACTIVE)
        # Print the name of the file you're saving to the command line
        # print 'Saving file: ' + str((os.path.splitext(os.path.basename(savename))[0]))


# This the main script
if __name__ == '__main__':
    # Set various options here:

    # Set the file path, i.e. the directory where the files are here
    path = '/Users/alice/Dropbox/UCL/DNA MiniCircles/Minicircle Data/Test'
    # Set file type to look for here
    fileend = '.spm'
    filetype = '.*[0-9]'
    # Set extension to export files as here e.g. '.tiff'
    extension = '.tiff'
    # Set height scale values to save out
    minheightscale = -2e-9
    maxheightscale = 4e-9
    # Set minimum size for grain determination:
    minarea = 500e-9
    # Set allowable deviation from the median pixel size for removal of large and small objects
    maxdeviation = 1.5
    mindeviation = 0.5
    # Set size of the cropped window/2 in pixels
    cropwidth = 40e-9
    # Set number of bins
    bins = 25

    # Declare variables used later
    # Placed outside for loop in order that they don't overwrite data to be appended
    result = []

    # Look through the current directory and all subdirectories for files ending in .spm and add to flist
    spmfiles = traversedirectories(fileend, filetype, path)
    # Iterate over all files found
    for i, filename in enumerate(spmfiles):
        print 'Analysing ' + str(os.path.basename(filename))
        # Load the data for the specified filename
        data = getdata(filename)
        # Find the channels of data you wish to use within the finle e.g. ZSensor or height
        chosen_ids = choosechannels(data)
        # Iterate over the chosen channels in your file e.g. the ZSensor channel
        for k in chosen_ids:
            # Get all the image details eg resolution for your chosen channel
            xres, yres, xreal, yreal, dx, dy = imagedetails(data)
            # Perform basic image processing, to align rows, flatten and set the mean value to zero
            data = editfile(data, minheightscale, maxheightscale)
            # Find all grains in the mask which are both above a height threshold
            # and bigger than the min size set in the main codegrain_mean_rad
            data, mask, datafield, grains = grainfinding(data, minarea, k)
            # Calculate the mean pixel area for all grains to use for renmoving small and large objects from the mask
            median_pixel_area = find_median_pixel_area(datafield, grains)
            # Remove all large objects defined as 1.2* the median grain size (in pixel area)
            mask, grains = removelargeobjects(datafield, mask, median_pixel_area, maxdeviation)
            # Remove all small objects defined as less than 0.5x the median grain size (in pixel area)
            mask, grains = removesmallobjects(datafield, mask, median_pixel_area, mindeviation)
            # Compute all grain statistics in in the 'values to compute' dictionary for grains in the file
            # Not currently used - replaced by grainstatistics function
            values_to_compute, grainstats = grainanalysis(path, filename, datafield, grains)
            # Create cropped datafields for every grain of size set in the main directory
            bbox, orig_ids, crop_ids, data = boundbox(cropwidth, datafield, grains, dx, dy, xreal, yreal, xres, yres)
            # Save out cropped files as images with no scales to a subfolder
            savecroppedfiles(path, data, filename, extension, orig_ids, crop_ids, minheightscale, maxheightscale)
            # Skeletonise data after performing an aggressive gaussian to improve skeletonisation
            # data, mask = grainthinning(data, mask, dx)
            # Save data as 2 images, with and without mask
            savefiles(data, filename, extension)
            # Export the channels data and mask as numpy arrays
            npdata, npmask = exportasnparray(datafield, mask)
            # Determine the grain statistics
            # Append those stats to one file to get all stats in a directory
            # Save out as a pandas dataframe
            grainstats_df = grainstatistics(datafield, grains, filename, result)
    # Plot a single variable from the dataframe
    plotting(grainstats_df, 'grain_mean_rad', 'directory', bins, path, extension)
    # Plot two variables from the dataframe - outputs both stacked by filename and full distributions
    # plotting2(grainstats_df, 'grain_min_bound', 'grain_max_bound', 'directory', bins, path, extension)
    # plotting2(grainstats_df, 'grain_max', 'grain_med', 'directory', bins, path, extension)
    # Plot all output from bigger dataframe grainstats for initial visualisation as KDE plots
    # plotall(grainstats, bins, directory, '_grainstats', '.png')
    # Saving stats to text and JSON files named by master path
    savestats(path, 'GrainStatistics', grainstats_df)
