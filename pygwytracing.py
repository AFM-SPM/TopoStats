''' **pygwytracing.py**
This is the main script, containing modules for basic image processing. '''

import sys
#sys.path.append('/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages') # location of gwy.so file (Macports install)
#sys.path.append('/opt/local/share/gwyddion/pygwy') # # location of gwyutils.py file (Macports install)


#sys.path.append('/usr/local/opt/python@2/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages') # Homebrew install on Mac
#sys.path.append('/usr/local/Cellar/gwyddion/2.53_2/share/gwyddion/pygwy') # Homebrew install on Mac

#sys.path.append('/usr/share/gwyddion/pygwy/') # Ubuntu

sys.path.append('C:\Program Files (x86)\Gwyddion\\bin')    #Windows install
sys.path.append('C:\Program Files (x86)\Gwyddion\share\gwyddion\pygwy') #pygwy location on Windows

import pygtk
pygtk.require20() # adds gtk-2.0 folder to sys.path
import gwy
import fnmatch
import gwyutils
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import dnatracing
import time

# Import height thresholding.py for processing bilayer removal images

### Set seaborn to override matplotlib for plot output
sns.set()
# The four preset contexts, in order of relative size, are paper, notebook, talk, and poster.
# The notebook style is the default
sns.set_context("notebook")
# This can be customised further here
# sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 2.5})

# sys.path.append("/usr/local/Cellar/gwyddion/2.52/share/gwyddion/pygwy")

# Set the settings for each function from the saved settings file (~/.gwyddion/settings)
s = gwy.gwy_app_settings_get()

# Generate a settings file - should be found at /Users/alice/.gwyddion/settings
# a = gwy.gwy_app_settings_get_settings_filename()
# Location of the settings file - edit to change values
# print a

# Turn colour bar off
s["/module/pixmap/ztype"] = 0

# Define the settings for image processing functions e.g. align rows here
s['/module/linematch/method'] = 1  # uses median
s["/module/linematch/max_degree"] = 2
s["/module/polylevel/col_degree"] = 2


def traversedirectories(fileend, filetype, path):
    # This function finds all the files with the file ending set in the main script as fileend (usually.spm)
    # in the path directory, and all subfolder

    # initialise the list
    spmfiles = []
    # use os.walk to search folders and subfolders and append each file with the correct filetype to the list spmfiles
    for dirpath, subdirs, files in os.walk(path):
        # Looking for files ending in fileend
        for filename in files:
            # ignore any files containing '_cs'
            if not fnmatch.fnmatch(filename, '*_cs*'):
                # Find files ending in 'fileend'
                if filename.endswith(fileend):
                    spmfiles.append(os.path.join(dirpath, filename))
                # Find files of a certain 'filetype'
                if fnmatch.fnmatch(filename, '*.*[0-9]'):
                    spmfiles.append(os.path.join(dirpath, filename))
        #
        # for filename in fnmatch.filter(files, filetype):
        #     spmfiles.append(os.path.join(dirpath, filename))
        # print the number of files found
    print('Files found: ' + str(len(spmfiles)))
    # return a list of files including their root and the original path specified
    return spmfiles


def getdata(filename):
    # Open file and add data to browser
    data = gwy.gwy_file_load(filename, gwy.RUN_NONINTERACTIVE)
    # Add data to browser
    gwy.gwy_app_data_browser_add(data)
    return data


def choosechannels(data, channel1, channel2):
    # Obtain the data field ids for the data file (i.e. the various channels within the file)
    ids = gwy.gwy_app_data_browser_get_data_ids(data)
    # Make an empty array for the chosen channels to be saved into
    chosen_ids = []
    # Find channels with the title ZSensor, if these do not exist, find channels with the title Height
    chosen_ids = gwy.gwy_app_data_browser_find_data_by_title(data, channel1)
    if not chosen_ids:
        chosen_ids = gwy.gwy_app_data_browser_find_data_by_title(data, channel2)
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
    dx = xreal / xres
    dy = yreal / yres
    return xres, yres, xreal, yreal, dx, dy

def heightediting(data, k):
    gwy.gwy_app_data_browser_select_data_field(data, k)

    # Flatten the data
    gwy.gwy_process_func_run('flatten_base', data, gwy.RUN_IMMEDIATE)

    datafield = gwy.gwy_app_data_browser_get_current(gwy.APP_DATA_FIELD)
    mask = gwy.DataField.new_alike(datafield, False)
    datafield.grains_mark_height(mask, 30, False)

    # Re-do polynomial correction with masked height
    s["/module/polylevel/masking"] = 1
    gwy.gwy_process_func_run('polylevel', data, gwy.RUN_IMMEDIATE)

    # Re-do align rows with masked heights
    s["/module/linematch/masking"] = 1
    gwy.gwy_process_func_run('align_rows', data, gwy.RUN_IMMEDIATE)

    # # Remove scars
    # gwy.gwy_process_func_run('scars_remove', data, gwy.RUN_IMMEDIATE)

    # Gaussian filter to remove noise
    current_data = gwy.gwy_app_data_browser_get_current(gwy.APP_DATA_FIELD)

    filter_width = 5 * dx * 1e9

    current_data.filter_gaussian(filter_width)

    # Set zero to mean value
    gwy.gwy_process_func_run('zero_mean', data, gwy.RUN_IMMEDIATE)

    return data

def editfile(data, k):
    # select each channel of the file in turn
    # this is run within the for k in chosen_ids loop so k refers to the index of each chosen channel to analyse
    # NONINTERACTIVE is only for file modules
    gwy.gwy_app_data_browser_select_data_field(data, k)

    # align rows (b)
    gwy.gwy_process_func_run("align_rows", data, gwy.RUN_IMMEDIATE)

    # flatten the data (c)
    gwy.gwy_process_func_run("level", data, gwy.RUN_IMMEDIATE)

    # align rows (d)
    gwy.gwy_process_func_run("align_rows", data, gwy.RUN_IMMEDIATE)

    datafield = gwy.gwy_app_data_browser_get_current(gwy.APP_DATA_FIELD)
    mask = gwy.DataField.new_alike(datafield, False)
    datafield.grains_mark_height(mask, 10, False)

    # Re-do polynomial correction with masked height (e)
    s["/module/polylevel/masking"] = 0
    gwy.gwy_process_func_run('polylevel', data, gwy.RUN_IMMEDIATE)

    # Re-do align rows with masked heights (f)
    s["/module/linematch/masking"] = 0
    gwy.gwy_process_func_run('align_rows', data, gwy.RUN_IMMEDIATE)

    # Re-do level with masked heights (g)
    s["/module/polylevel/masking"] = 1
    gwy.gwy_process_func_run('polylevel', data, gwy.RUN_IMMEDIATE)

    # flatten base (h)
    gwy.gwy_process_func_run('flatten_base', data, gwy.RUN_IMMEDIATE)

    # remove scars (i)
    gwy.gwy_process_func_run('scars_remove', data, gwy.RUN_IMMEDIATE)

    # Fix zero (j)
    gwy.gwy_process_func_run('zero_mean', data, gwy.RUN_IMMEDIATE)

    # Apply a 1.5 pixel gaussian filter (k)
    data_field = gwy.gwy_app_data_browser_get_current(gwy.APP_DATA_FIELD)
    data_field.filter_gaussian(1)
    # # Shift contrast - equivalent to 'fix zero'
    # datafield.add(-data_field.get_min())

    return data


def grainfinding(data, minarea, k, thresholdingcriteria, dx):
    # Select channel 'k' of the file
    gwy.gwy_app_data_browser_select_data_field(data, k)
    datafield = gwy.gwy_app_data_browser_get_current(gwy.APP_DATA_FIELD)
    # Gaussiansize = 0.25e-9 / dx
    # datafield.filter_gaussian(Gaussiansize)

    mask = gwy.DataField.new_alike(datafield, False)

    Gaussiansize = 0.1e-9 / dx
    datafield.filter_gaussian(Gaussiansize)

    # Mask data that are above thresh*sigma from average height.
    # Sigma denotes root-mean square deviation of heights.
    # This criterium corresponds to the usual Gaussian distribution outliers detection if thresh is 3.
    # For MAC ~2.1 works and DNA ~0.75 and NPC 0.5
    # datafield.mask_outliers(mask, 2.1)
    # datafield.mask_outliers(mask, 0.5)
    datafield.mask_outliers(mask, 0.75)

    # excluding mask, zero mean
    stats = datafield.area_get_stats_mask(mask, gwy.MASK_EXCLUDE, 0, 0, datafield.get_xres(), datafield.get_yres())
    datafield.add(-stats[0])

    # Set the image display to fixed range and the colour scale for the images
    maximum_disp_value = data.set_int32_by_name("/" + str(k) + "/base/range-type", int(1))
    minimum_disp_value = data.set_double_by_name("/" + str(k) + "/base/min", float(minheightscale))
    maximum_disp_value = data.set_double_by_name("/" + str(k) + "/base/max", float(maxheightscale))

    ### Editing grain mask
    # Remove grains touching the edge of the mask
    mask.grains_remove_touching_border()

    # Calculate pixel width in nm
    # dx = datafield.get_dx()
    # Calculate minimum feature size in pixels (integer) from a real size specified in the main
    minsize = int(minarea / dx)
    # Remove grains smaller than the minimum size in integer pixels
    mask.grains_remove_by_size(minsize)

    # Numbering grains for grain analysis
    grains = mask.number_grains()

    # Update data to show mask, comment out to remove mask
    # data['/%d/mask' % k] = mask

    return data, mask, datafield, grains


def removelargeobjects(datafield, mask, median_pixel_area, maxdeviation, dx):
    mask2 = gwy.DataField.new_alike(datafield, False)

    # Mask data that are above thresh*sigma from average height.
    # Sigma denotes root-mean square deviation of heights.
    # This criterium corresponds to the usual Gaussian distribution outliers detection if thresh is 3.
    datafield.mask_outliers(mask2, 1)
    # Calculate pixel width in nm
    # dx = datafield.get_dx()
    # Calculate minimum feature size in pixels (integer)
    # here this is calculated as 2* the median grain size, as calculated in find_median_pixel_area()
    maxsize = int(maxdeviation * median_pixel_area)
    # Remove grains smaller than the maximum feature size in integer pixels
    # This should remove everything that you do want to keep
    # i.e. everything smaller than aggregates/junk
    mask2.grains_remove_by_size(maxsize)
    # Invert mask2 so everything smaller than aggregates/junk is masked
    mask2.grains_invert()
    # Make mask equal to the intersection of mask and mask 2, i.e. remove large objects unmasked by mask2
    mask.grains_intersect(mask2)

    # Numbering grains for grain analysis
    grains = mask.number_grains()

    return mask, grains


def removesmallobjects(datafield, mask, median_pixel_area, mindeviation, dx):
    mask2 = gwy.DataField.new_alike(datafield, False)
    # Mask data that are above thresh*sigma from average height.
    # Sigma denotes root-mean square deviation of heights.
    # This criterium corresponds to the usual Gaussian distribution outliers detection if thresh is 3.
    datafield.mask_outliers(mask2, 1)
    # Calculate pixel width in nm
    # dx = datafield.get_dx()
    # Calculate minimum feature size in pixels (integer)
    # here this is calculated as 2* the median grain size, as calculated in find_median_pixel_area()
    minsize = int(mindeviation * median_pixel_area)
    # Remove grains smaller than the maximum feature size in integer pixels
    # This should remove everything that you do want to keep
    # i.e. everything smaller than aggregates/junk
    mask2.grains_remove_by_size(minsize)
    # Make mask equalto the intersection of mask and mask 2, i.e. remove large objects unmasked by mask2
    mask.grains_intersect(mask2)

    # Numbering grains for grain analysis
    grains = mask.number_grains()
    number_of_grains = max(grains)
    print('There were %i grains found' % (number_of_grains))

    return mask, grains, number_of_grains


def grainanalysis(appended_data, filename, datafield, grains):
    # Calculating grain statistics using numbered grains file
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
    # Create empty dictionary for grain data
    grain_data_to_save = {}

    for key in values_to_compute.keys():
        # here we stave the gran stats to both a dictionary and an array in that order
        # these are basically duplicate steps - but are both included as we arent sure which to use later
        # Save grain statistics to a dictionary: grain_data_to_save
        grain_data_to_save[key] = datafield.grains_get_values(grains, values_to_compute[key])
        # Delete 0th value in all arrays - this corresponds to the background
        del grain_data_to_save[key][0]

    # Create pandas dataframe of stats
    grainstats = pd.DataFrame.from_dict(grain_data_to_save)

    # Determine directory, filename and grain number to append to dataframe
    directory = str(os.path.dirname(filename))
    directory = str(os.path.splitext(os.path.basename(directory))[0])
    filename = os.path.splitext(os.path.basename(filename))[0]
    # Append filename, directory and grain ID to dataframe
    grainstats['filename'] = pd.Series(filename, index=grainstats.index)
    grainstats['directory'] = pd.Series(directory, index=grainstats.index)
    grainstats['grain no'] = (grainstats.reset_index().index) + 1
    grainstats['image no'] = pd.Series(k, index=grainstats.index)

    # Sort dataframe columns to appear alphabetically - ensures consistency in columns order
    # when using dictionary to generate columns
    grainstats.sort_index(axis=1, inplace=True)

    # Append dataframe to appended_data as list to collect statistics on multiple files
    appended_data.append(grainstats)

    grainstatsarguments = list(values_to_compute.keys())
    grainstatsarguments = sorted(grainstatsarguments)

    return grainstatsarguments, grainstats, appended_data


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

    #make 2d array for grains
    multidim_grain_array = np.reshape(np.array(grains), (xres,yres))

    for i in range(len(center_x)):
        px_center_x = int((center_x[i] / xreal) * xres)
        px_center_y = int((center_y[i] / yreal) * yres)
        # ULcol = px_center_x - cropwidth
        xmin = px_center_x - cropwidth
        # ULrow = px_center_y - cropwidth
        ymin = px_center_y - cropwidth
        # BRcol = px_center_x + cropwidth
        xmax = px_center_x + cropwidth
        # BRrow = px_center_y + cropwidth
        ymax = px_center_y + cropwidth

        # making sure cropping boxes dont run outside the image dimensions
        if xmin < 0:
            xmin = 0
            xmax = 2 * cropwidth
        if ymin < 0:
            ymin = 0
            ymax = 2 * cropwidth
        if xmax > xres:
            xmax = xres
            xmin = xres - 2*cropwidth
        if ymax > yres:
            ymax = yres
            ymin = yres - 2 * cropwidth

        # print ULcol, ULrow, BRcol, BRrow
        # crop the data
        crop_datafield_i = datafield.duplicate()
        crop_datafield_i.resize(xmin, ymin, xmax, ymax)

        # add cropped datafield to active container
        gwy.gwy_app_data_browser_add_data_field(crop_datafield_i, data, i + (len(orig_ids)))

        # cropping the grain array:
        grain_num = i + 1
        cropped_np_grain = multidim_grain_array[ymin:ymax, xmin:xmax]
        cropped_grain = [1 if i == grain_num else 0 for i in cropped_np_grain.flatten()]

        # make a list containing each of the cropped grains
        try:
            cropped_grains.append(cropped_grain)
        except NameError:
            cropped_grains = [cropped_grain]

    # Generate list of datafields including cropped fields
    crop_ids = gwy.gwy_app_data_browser_get_data_ids(data)

    return bbox, orig_ids, crop_ids, data, cropped_grains, cropwidth


def splitimage(data, splitwidth, datafield, xreal, yreal, xres, yres):
    # get current number of images within file
    orig_ids = gwy.gwy_app_data_browser_get_data_ids(data)
    # Define the width of the image to crop to in pixels
    splitwidth_px = int((splitwidth / xreal) * xres)
    # Define the number of images to split into based on the resolution (in one direction)
    no_splits = int(round(xreal / splitwidth))
    # iterate in both x and y to get coordinates to crop image to
    for x in range(no_splits):
        xmin = splitwidth_px * x
        if xmin < 0:
            xmin = 0
        xmax = splitwidth_px * x + splitwidth_px
        if xmax > xres:
            xmax = xres
        for y in range(no_splits):
            ymin = splitwidth_px * y
            if ymin < 0:
                ymin = 0
            ymax = splitwidth_px * y + splitwidth_px
            if ymax > yres:
                ymax = yres
            # coordinates to crop to
            tiles = xmin, ymin, xmax, ymax
            # duplicate image
            crop_datafield_i = datafield.duplicate()
            # crop image
            crop_datafield_i.resize(xmin, ymin, xmax, ymax)
            # add cropped datafield to active container
            gwy.gwy_app_data_browser_add_data_field(crop_datafield_i, data, i + (len(orig_ids)))
    # Generate list of datafields including cropped fields
    crop_ids = gwy.gwy_app_data_browser_get_data_ids(data)
    return orig_ids, crop_ids, data


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


def savestats(directory, dataframetosave):
    directoryname = os.path.splitext(os.path.basename(directory))[0]
    print('Saving stats for: ' + str(directoryname))

    savedir = os.path.join(directory)
    savename = os.path.join(savedir, directoryname)
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    dataframetosave.to_json(savename + '.json')
    dataframetosave.to_csv(savename + '.txt')


def saveindividualstats(filename, dataframetosave, k):

    # Get directory path and filename (including extension to avoid overwriting .000 type Bruker files)
    filedirectory, filename = os.path.split(filename)

    # print 'Saving stats for: ' + str(filename)

    savedir = os.path.join(filedirectory, 'GrainStatistics')
    savename = os.path.join(savedir, filename)
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    dataframetosave.to_json(savename + str(k) + '.json')
    dataframetosave.to_csv(savename + str(k) + '.txt')


def savefiles(data, filename, extension):
    # Turn rulers on
    s["/module/pixmap/xytype"] = 1

    # Get directory path and filename (including extension to avoid overwriting .000 type Bruker files)
    directory, filename = os.path.split(filename)

    # Create a saving name format/directory
    savedir = os.path.join(directory, 'Processed')

    # If the folder Processed doest exist make it here
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # Save the data for the channels found above i.e. ZSensor/Height, as chosen_ids
    # Data is exported to a file of extension set in the main script
    # Data is exported with the string '_processed' added to the end of its filename
    gwy.gwy_app_data_browser_select_data_field(data, k)
    # change the colour map for all channels (k) in the image:
    palette = data.set_string_by_name("/%s/base/palette" % k, "Nanoscope.txt")
    # Determine the title of each channel
    title = data["/%d/data/title" % k]
    # Generate a filename to save to by removing the extension to the file, adding the suffix '_processed'
    # and an extension set in the main file
    savename = os.path.join(savedir, filename) + str(k) + '_' + str(title) + '_processed' + str(extension)
    # Save the file
    gwy.gwy_file_save(data, savename, gwy.RUN_NONINTERACTIVE)
    # Show the mask
    data['/%d/mask' % k] = mask
    # Add the sufix _masked to the previous filename
    savename = os.path.join(savedir, filename) + str(k) + '_' + str(title) + '_processed_masked' + str(extension)
    # Save the data
    gwy.gwy_file_save(data, savename, gwy.RUN_NONINTERACTIVE)
    gwy.gwy_app_file_close()
    # Print the name of the file you're saving to the command line
    # print 'Saving file: ' + str((os.path.splitext(os.path.basename(savename))[0]))


def saveunknownfiles(data, filename, extension):
    # Turn rulers on
    s["/module/pixmap/xytype"] = 1

    # Get directory path and filename (including extension to avoid overwriting .000 type Bruker files)
    directory, filename = os.path.split(filename)

    # Create a saving name format/directory
    savedir = os.path.join(directory, 'Processed_Unknown')

    # If the folder Processed doest exist make it here
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # Save the data for the channels found above i.e. ZSensor/Height, as chosen_ids
    # Data is exported to a file of extension set in the main script
    # Data is exported with the string '_processed' added to the end of its filename
    gwy.gwy_app_data_browser_select_data_field(data, k)
    # change the colour map for all channels (k) in the image:
    palette = data.set_string_by_name("/" + str(k) + "/base/palette", "Nanoscope.txt")
    # Generate a filename to save to by removing the extension to the file, adding the suffix '_processed'
    # and an extension set in the main file
    savename = os.path.join(savedir, filename) + str(k) + '_' + '_processed' + str(extension)
    # Save the file
    gwy.gwy_file_save(data, savename, gwy.RUN_NONINTERACTIVE)
    # Show the mask
    data['/%d/mask' % k] = mask
    # Add the sufix _masked to the previous filename
    savename = os.path.join(savedir, filename) + str(k) + '_' + '_processed_masked' + str(extension)
    # Save the data
    gwy.gwy_file_save(data, savename, gwy.RUN_NONINTERACTIVE)
    # Print the name of the file you're saving to the command line
    # print 'Saving file: ' + str((os.path.splitext(os.path.basename(savename))[0]))


def savefilesasgwy(data, filename):
    # Turn rulers on
    s["/module/pixmap/xytype"] = 1

    # Get directory path and filename (including extension to avoid overwriting .000 type Bruker files)
    directory, filename = os.path.split(filename)

    # Create a saving name format/directory
    savedir = os.path.join(directory)

    # If the folder Processed doest exist make it here
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # Save the data for the channels found above i.e. ZSensor/Height, as chosen_ids
    # Data is exported to a file of extension set in the main script
    # Data is exported with the string '_processed' added to the end of its filename
    gwy.gwy_app_data_browser_select_data_field(data, k)
    # change the colour map for all channels (k) in the image:
    palette = data.set_string_by_name("/" + str(k) + "/base/palette", "Nanoscope.txt")
    # Determine the title of each channel
    # title = data["/%d/data/title" % k]
    # if not title:
    #     title = 'unknown'
    # Generate a filename to save to by removing the extension to the file, adding the suffix '_processed'
    # and an extension set in the main file
    savename = os.path.join(savedir, filename) + str(k) + '.gwy'
    # Save the file
    gwy.gwy_file_save(data, savename, gwy.RUN_NONINTERACTIVE)
    # Print the name of the file you're saving to the command line
    # print 'Saving file: ' + str((os.path.splitext(os.path.basename(savename))[0]))


def savecroppedfiles(directory, data, filename, extension, orig_ids, crop_ids, minheightscale, maxheightscale):
    # Save the data for the cropped channels
    # Data is exported to a file of extension set in the main script
    # Data is exported with the string '_cropped' added to the end of its filename

    # Turn rulers off
    s["/module/pixmap/xytype"] = 0

    # Get directory path and filename (including extension to avoid overwriting .000 type Bruker files)
    directory, filename = os.path.split(filename)

    # Create a saving name format/directory
    savedir = os.path.join(directory, 'Cropped')

    # If the folder Cropped doest exist make it here
    if not os.path.exists(savedir):
        os.makedirs(savedir)
        crop_directory = savedir
    # Otherwise set the existing Cropped directory as the directory to write to
    else:
        crop_directory = savedir

    # For each cropped file, save out the data with the suffix _Cropped_#
    for i in range(len(orig_ids), len(crop_ids), 1):
        # select each channel fo the file in turn
        gwy.gwy_app_data_browser_select_data_field(data, i)
        # change the colour map for all channels (k) in the image:
        palette = data.set_string_by_name("/" + str(i) + "/base/palette", "Nanoscope.txt")
        # Set the image display to fixed range and the colour scale for the images
        maximum_disp_value = data.set_int32_by_name("/" + str(i) + "/base/range-type", int(1))
        minimum_disp_value = data.set_double_by_name("/" + str(i) + "/base/min", float(minheightscale))
        maximum_disp_value = data.set_double_by_name("/" + str(i) + "/base/max", float(maxheightscale))
        # Generate a filename to save to by removing the extension to the file and a numerical identifier (the crop no)
        # The 'crop number' is the channel number minus the original number of channels, less one to avoid starting at 0
        savenumber = i - (len(orig_ids) - 1)
        # adding the suffix '_cropped' and adding the extension set in the main file
        savename = os.path.join(crop_directory, filename) + '_cropped_' + str(savenumber) + str(extension)
        # Save the file
        gwy.gwy_file_save(data, savename, gwy.RUN_NONINTERACTIVE)
        # Print the name of the file you're saving to the command line
        # print 'Saving file: ' + str((os.path.splitext(os.path.basename(savename))[0]))

def getdataforallfiles(appended_data):
    # Get dataframe of all files within folder from appended_data list file
    grainstats_df = pd.concat(appended_data, ignore_index=True)

    return grainstats_df

    # Get dataframe of only files containing a certain string
def searchgrainstats(df, dfargtosearch, searchvalue1, searchvalue2):
    df1 = df[df[dfargtosearch].str.contains(searchvalue1)]
    df2 = df[df[dfargtosearch].str.contains(searchvalue2)]
    grainstats_searched = pd.concat([df1, df2])

    return grainstats_searched

# This the main script
if __name__ == '__main__':
    # Set various options here:

    # Set the file path, i.e. the directory where the files are here'

    # path = '/Volumes/GoogleDrive/My Drive/AFM research group /Methods paper/Data/Circular'
    # path = '/Users/alicepyne/Dropbox/UCL/DNA MiniCircles/Minicircle Data Edited/Minicircle Manuscript/Nickel'
    # path = '/Users/alicepyne/Dropbox/UCL/DNA MiniCircles/Paper/Pyne et al/Figure 1/aspectratioanalysis'
    # path = '/Volumes/GoogleDrive/My Drive/AFM research group /Methods paper/Data/Circular/194 bp'
    # path = '/Volumes/GoogleDrive/My Drive/AFM research group /Methods paper/Data/Circular'
    # path = '/Volumes/GoogleDrive/My Drive/AFM research group /Methods paper/Data/NPC'
    # path = '/Volumes/GoogleDrive/My Drive/AFM research group /Methods paper/Data/Archive/'
    #path = '/Volumes/GoogleDrive/My Drive/AFM research group /Methods paper/Data/Fortracing'
    path = 'C:\Users\Work\OneDrive\Documents\Uni\Research\Code\DNA minicircles'

    # Set file type to look for here
    fileend = '.spm', '.gwy', '*.[0-9]'
    filetype = '*.[0-9]'
    # Set extension to export files as here e.g. '.tiff'
    extension = '.tiff'
    # Set height scale values to save out
    minheightscale = -0e-9
    maxheightscale = 3e-9
    # maxheightscale = 20e-9
    # maxheightscale = 50e-9
    # Set minimum size for grain determination:
    minarea = 300e-9
    # minarea = 50e-9
    # minarea = 1000e-9
    # Set allowable deviation from the median pixel size for removal of large and small objects
    maxdeviation = 1.3
    mindeviation = 0.7
    # maxdeviation = 1.5
    # mindeviation = 0.5
    # Set size of the cropped window/2 in pixels
    # cropwidth = 100e-9
    # cropwidth = 60e-9
    cropwidth = 40e-9
    splitwidth = 2e-6
    # Set number of bins
    bins = 25

    # Declare variables used later
    # Placed outside for loop in order that they don't overwrite data to be appended
    appended_data = []

    # Look through the current directory and all subdirectories for files ending in .spm and add to flist
    # spmfiles = traversedirectories(fileend, filetype, path)
    spmfiles = traversedirectories(fileend, filetype, path)

    if len(spmfiles) == 0:
        quit('No .spm files were found in the folder ' + path)

    # Iterate over all files found
    for i, filename in enumerate(spmfiles):
        print('Analysing ' + str(os.path.basename(filename)))
        # Load the data for the specified filename
        data = getdata(filename)
        # Find the channels of data you wish to use within the file e.g. ZSensor or height
        channels = ['ZSensor', 'Height']
        chosen_ids = choosechannels(data, 'ZSensor', 'Height')
        # chosen_ids = choosechannels(data,'U*', 'X')
        # chosen_ids = [chosen_ids[0]]

        # Iterate over the chosen channels in your file e.g. the ZSensor channel
        # for k in chosen_ids:
        # Or just use first height/height sensor channel to avoid duplicating
        for k in chosen_ids[:1]:
            # Option if you want to only choose one channel for each file being analysed
            # for k in chosen_ids:
            #     # Get all the image details eg resolution for your chosen channel
            data_edit_start = time.time()
            xres, yres, xreal, yreal, dx, dy = imagedetails(data)

            # Perform basic image processing, to align rows, flatten and set the mean value to zero
            data = editfile(data, k)
            data_edit_end = time.time()
            mol_find_start = time.time()

            # Perform basic image processing, to align rows, flatten and set the mean value to zero
            # Find all grains in the mask which are both above a height threshold
            # and bigger than the min size set in the main codegrain_mean_rad
            # 1.2 works well for DNA minicircle images
            data, mask, datafield, grains = grainfinding(data, minarea, k, 1, dx)
            # # Flattening based on masked data and subsequent grain finding
            # # Used for analysing data e.g. peptide induced bilayer degradation
            # data, mask, datafield, grains = heightthresholding.otsuthresholdgrainfinding(data, k)

            # Calculate the mean pixel area for all grains to use for renmoving small and large objects from the mask
            median_pixel_area = find_median_pixel_area(datafield, grains)
            # Remove all large objects defined as 1.2* the median grain size (in pixel area)
            mask, grains = removelargeobjects(datafield, mask, median_pixel_area, maxdeviation, dx)
            # Remove all small objects defined as less than 0.5x the median grain size (in pixel area
            mask, grains, number_of_grains = removesmallobjects(datafield, mask, median_pixel_area, mindeviation, dx)

            # if there's no grains skip this image
            if number_of_grains == 0:
                continue

            # Compute all grain statistics in in the 'values to compute' dictionary for grains in the file
            # Append data for each file (grainstats) to a list (appended_data) to obtain data in all files
            grainstatsarguments, grainstats, appended_data = grainanalysis(appended_data, filename, datafield, grains)

            # Create cropped datafields for every grain of size set in the main directory
            bbox, orig_ids, crop_ids, data, cropped_grains, cropwidth_pix = boundbox(cropwidth, datafield, grains, dx, dy, xreal, yreal, xres, yres)
            # orig_ids, crop_ids, data = splitimage(data, splitwidth, datafield, xreal, yreal, xres, yres)

            mol_find_end = time.time()

            trace_start = time.time()

            # Export the channels data and mask as numpy arrays
            npdata, npmask = exportasnparray(datafield, mask)


            try:
                channel_name = channels[k] + str(k + 1)
            except IndexError:
                channel_name = 'ZSensor'

            # bbox, orig_ids, crop_ids, cropped_grains = boundbox(cropwidth, grains, grains, dx, dy, xreal, yreal, xres, yres)
            # saving plots of individual grains/traces
            # for grain_num, data_num in enumerate(range(len(orig_ids), len(crop_ids), 1)):
            #     gwy.gwy_app_data_browser_select_data_field(data, data_num)
            #     datafield = gwy.gwy_app_data_browser_get_current(gwy.APP_DATA_FIELD)
            #
            #     np_data_array = gwyutils.data_field_data_as_array(datafield)
            #
            #     dna_traces = dnatracing.dnaTrace(np_data_array, cropped_grains[grain_num], filename, dx, cropwidth_pix*2, cropwidth_pix*2)
            #     # dna_traces.showTraces()
            #     dna_traces.saveTraceFigures(filename, channel_name+str(grain_num), minheightscale, maxheightscale, 'cropped')

            # #trace the DNA molecules - can compute stats etc as needed
            dna_traces = dnatracing.dnaTrace(npdata, grains, filename, dx, yres, xres)
            trace_end = time.time()
            # #dna_traces.showTraces()
            dna_traces.saveTraceFigures(filename, channel_name, minheightscale, maxheightscale, 'processed')
            # dna_traces.writeContourLengths(filename, channel_name)

            # Update the pandas Dataframe used to monitor stats
            try:
                tracing_stats.updateTraceStats(dna_traces)
            except NameError:
                tracing_stats = dnatracing.traceStats(dna_traces)

            # Save out cropped files as images with no scales to a subfolder
            # savecroppedfiles(path, data, filename, extension, orig_ids, crop_ids, minheightscale, maxheightscale)

            # Skeletonise data after performing an aggressive gaussian to improve skeletonisation
            # data, mask = grainthinning(data, mask, dx)

            # Export the channels data and mask as numpy arrays
            npdata, npmask = exportasnparray(datafield, mask)

            # Save data as 2 images, with and without mask
            savefiles(data, filename, extension)
            # saveunknownfiles(data, filename, extension)

            print('Image correction took %f seconds' % (data_edit_end - data_edit_start))
            print('Molecule identification took %f seconds' % (mol_find_end - mol_find_start))
            print('Tracing took %f seconds' % (trace_end - trace_start))

            # Saving stats to text and JSON files named by master path
            # saveindividualstats(filename, grainstats, k)

        # Save modified files as gwyddion files
        # savefilesasgwy(data, filename)

    tracing_stats.saveTraceStats(path)
    # Concatenate statistics form all files into one dataframe for saving and plotting statistics
    grainstats_df = getdataforallfiles(appended_data)
    # # Search dataframes and return a new dataframe of only files containing a specific string
    # grainstats_searched = searchgrainstats(grainstats_df, 'filename', '339', 'nothing')

    # Saving stats to text and JSON files named by master path
    savestats(path, grainstats_df)
