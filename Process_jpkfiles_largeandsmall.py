import os
import sys
sys.path.append("C:/Program Files (x86)/Gwyddion/bin")
import gwy
sys.path.append("C:/Program Files (x86)/Gwyddion/share/gwyddion/pygwy")
import gwyutils
import numpy as np
from scipy import stats


# If you do not have one you will need to generate a settings file - should be found at /Users/alice/.gwyddion/settings
# # a = gwy.gwy_app_settings_get_settings_filename()
# # Location of the settings file - edit to change values
# # print a

# Set the settings for each function from the saved settings file (~/.gwyddion/settings)
s = gwy.gwy_app_settings_get()
# You alter these settings as you go in the script


def traversedirectories_nosubfolders(fileend, path):
    # This function finds all the files (with the file ending set in the main script as fileend (usually .jpk))
    # in the path directory, and all subfolders

    # initialise the list
    jpkfiles = []
    # use os.walk to search folders and subfolders and append each file with the correct filetype to the list jpkfiles
    filelist = os.listdir(path)
    for i in filelist:
        # Find files ending in 'fileend'
        if i.endswith((fileend)):
            jpkfiles.append(os.path.join(path, i))
    # print the number of files found
    print 'Files found: ' + str(len(jpkfiles))
    # return a list of files including their root and the original path specified
    return jpkfiles


def getdata(filename):
    # Gets the data you want from a filename to something gwyddion can use
    # Open file and add data to browser
    data = gwy.gwy_file_load(filename, gwy.RUN_NONINTERACTIVE)
    # Add data to browser
    gwy.gwy_app_data_browser_add(data)
    return data


def choosechannels(data):
    # Obtain the data field ids for the data file (i.e. the various channels within the file)
    gwy.gwy_app_data_browser_get_data_ids(data)
    # Find trace and retrace channels with the title Height (measured),
    # if these do not exist, find channels with the title Height
    Height_ids = gwy.gwy_app_data_browser_find_data_by_title(data, 'Height (measured)*')
    if Height_ids == []:
        Height_ids = gwy.gwy_app_data_browser_find_data_by_title(data, 'Height*')
    Phase_ids = gwy.gwy_app_data_browser_find_data_by_title(data, 'Lock-In Phase*')
    # Save out chosen_ids as a list of the channel ids for Height and phase
    return Height_ids, Phase_ids


def imagedetails(data, k):
    # select the channel file of x_ids
    gwy.gwy_app_data_browser_select_data_field(data, k)
    datafield = gwy.gwy_app_data_browser_get_current(gwy.APP_DATA_FIELD)
    # x pixels
    xres = datafield.get_xres()
    # y pixels
    yres = datafield.get_yres()
    # x physical dimensions
    xreal = datafield.get_xreal()
    # y physical dimensions
    yreal = datafield.get_yreal()
    # gets xreal/xres e.g. nm/pixel
    dx = datafield.get_dx()
    # gets yreal/yres e.g. nm/pixel
    dy = datafield.get_dy
    # get the max value of the image
    max = datafield.get_max()
    # get the min value of the image
    min = datafield.get_min()
    return xres, yres, xreal, yreal, dx, dy, max, min


def EditSmallHeightFile(data, k,min, max):
    # select each channel of the file in turn
    # this is run within the for k in chosen_ids loop so k refers to the
    # index of each chosen channel to analyse (see main script below)
    gwy.gwy_app_data_browser_select_data_field(data, k)



    # If the range is large, apply a mask to the lower part of the image
    zrange = max - min
    if zrange >= 5e-10:
        # Make sure you can draw the mask
        s['/module/pixmap/draw_mask'] = True
        # Allocate your datafield
        datafield = gwy.gwy_app_data_browser_get_current(gwy.APP_DATA_FIELD)
        # Make a datafield for the mask
        mask = gwy.DataField.new_alike(datafield, False)
        # Add the mask datafield to the data container
        data['/%d/mask' % k] = mask
        # Determine the mask
        datafield.grains_mark_height(mask, 40, True)

    # Flatten the data using flatten base
    gwy.gwy_process_func_run('flatten_base', data, gwy.RUN_IMMEDIATE)
    # Align the rows with polynomial 2nd order (set in main script)
    gwy.gwy_process_func_run("align_rows", data, gwy.RUN_IMMEDIATE)
    # Remove mask
    s['/module/pixmap/draw_mask'] = False
    # Apply a 1 pixel gaussian filter
    datafield = gwy.gwy_app_data_browser_get_current(gwy.APP_DATA_FIELD)
    datafield.filter_gaussian(1)
    # Fix zero
    gwy.gwy_process_func_run('fix_zero', data, gwy.RUN_IMMEDIATE)



    # Set the colour scale:
    datafield = gwy.gwy_app_data_browser_get_current(gwy.APP_DATA_FIELD)
    # To get the mode, convert data to an array, round to 1nm and get the mode
    # Get your data as a list
    x = datafield.get_data()
    # Convert it to an array
    x_array = np.asarray(x)
    # Round it
    x_rounded = np.round_(x_array, decimals=9)
    # Get the mode
    mode = stats.mode(x_rounded)

    # Get the standard deviation and median of data
    rms = datafield.get_rms()
    # Round the rms to a nice number
    if rms >= 9.99e-07:
        rms_rounded = (round(rms / 1e-06, 2) * 1e-06)
    elif 9.99e-07 > rms >= 9.9e-08:
        rms_rounded = (round(rms / 1e-07, 1) * 1e-07)
    elif 9.9e-08 > rms >= 9e-09:
        rms_rounded = (round(rms / 1e-08, 0) * 1e-08)
    else:
        rms_rounded = (round(rms / 1e-09, 0) * 1e-09)
    if rms_rounded == 0:
        rms_rounded = 1e-10


    # Set the minimum as the median of the middle minus 2 rounded rms
    minheightscale = float(mode[0]) - 2 * rms_rounded
    # Set the maximum as the median of the middle plus 5 rounded rms
    maxheightscale = float(mode[0]) + 4 * rms_rounded



    return data, minheightscale, maxheightscale


def EditBigHeightFile(data, k, heightthresh):
    # select each channel of the file in turn
    # this is run within the for k in chosen_ids loop so k refers to the
    # index of each chosen channel to analyse (see main script below)
    gwy.gwy_app_data_browser_select_data_field(data, k)
    # Make sure you can draw the mask
    s['/module/pixmap/draw_mask'] = True
    # Allocate your datafield
    datafield = gwy.gwy_app_data_browser_get_current(gwy.APP_DATA_FIELD)
    # Make a datafield for the mask
    mask = gwy.DataField.new_alike(datafield, False)
    # Add the mask datafield to the data container
    data['/%d/mask' % k] = mask
    # Determine the mask
    datafield.grains_mark_height(mask, heightthresh, False)
    # Plane level the data
    gwy.gwy_process_func_run("level", data, gwy.RUN_IMMEDIATE)
    # Fix zero
    gwy.gwy_process_func_run('fix_zero', data, gwy.RUN_IMMEDIATE)
    # Remove mask
    s['/module/pixmap/draw_mask'] = False

    # Set the colour scale:
    datafield = gwy.gwy_app_data_browser_get_current(gwy.APP_DATA_FIELD)

    # To get the mode, convert data to an array, round to 100nm and get the mode
    # Get your data as a list
    x = datafield.get_data()
    # Convert it to an array
    x_array = np.asarray(x)
    # Round it
    x_rounded = np.round_(x_array, decimals=7)
    # Get the mode
    mode = stats.mode(x_rounded)


    # Get the standard deviation of data
    rms = datafield.get_rms()
    # Round the rms to a nice number
    if rms >= 9.99e-07:
        rms_rounded = (round(rms / 1e-06, 2) * 1e-06)
    elif 9.99e-07 > rms >= 9.9e-08:
        rms_rounded = (round(rms / 1e-07, 1) * 1e-07)
    elif 9.9e-08 > rms >= 9e-09:
        rms_rounded = (round(rms / 1e-08, 0) * 1e-08)
    else:
        rms_rounded = (round(rms / 1e-09, 0) * 1e-09)
    if rms_rounded == 0:
        rms_rounded = 1e-10

    # Set the minimum as the mode
    minheightscale = float(mode[0])
    # Set the maximum as the mode plus 5 rounded rms
    maxheightscale = float(mode[0]) + 5 * rms_rounded


    return data, minheightscale, maxheightscale

def EditSmallPhaseFile(data, k):
    # select each channel of the file in turn
    # this is run within the for k in chosen_ids loop so k refers to the
    # index of each chosen channel to analyse (see main script below)
    gwy.gwy_app_data_browser_select_data_field(data, k)
    # align the rows polynomial 2nd order (set in main script)
    gwy.gwy_process_func_run("align_rows", data, gwy.RUN_IMMEDIATE)
    # Apply a 1 pixel gaussian filter
    datafield = gwy.gwy_app_data_browser_get_current(gwy.APP_DATA_FIELD)
    datafield.filter_gaussian(1)
    # Fix zero
    gwy.gwy_process_func_run('fix_zero', data, gwy.RUN_IMMEDIATE)


    # Set the colour scale:
    datafield = gwy.gwy_app_data_browser_get_current(gwy.APP_DATA_FIELD)

    # To get the mode, convert data to an array, round to 0.1deg and get the mode
    # Get your data as a list
    x = datafield.get_data()
    # Convert it to an array
    x_array = np.asarray(x)
    # Round it
    x_rounded = np.round_(x_array, decimals=2)
    # Get the mode
    mode = stats.mode(x_rounded)

    # Get the standard deviation and median of data
    rms = datafield.get_rms()
    # Round the rms to a nice number
    if rms >= 10:
        rms_rounded = (round(rms / 10, 0) * 10)
    elif 10 > rms >= 1:
        rms_rounded = (round(rms / 0.5, 1) * 0.5)
    elif 1 > rms >= 0.1:
        rms_rounded = (round(rms / 0.1, 1) * 0.1)
    else:
        rms_rounded = (round(rms / 0.05, 1) * 0.05)
    if rms_rounded == 0:
        rms_rounded = 0.01

    # Set the minimum as the median of the middle minus 2 rounded rms
    minphasescale = float(mode[0]) - 3 * rms_rounded
    # Set the maximum as the median of the middle plus 5 rounded rms
    maxphasescale = float(mode[0]) + 5 * rms_rounded


    return data, minphasescale, maxphasescale


def EditBigPhaseFile(data, k, min, max, phasethresh):
    # select each channel of the file in turn
    # this is run within the for k in chosen_ids loop so k refers to the
    # index of each chosen channel to analyse (see main script below)
    gwy.gwy_app_data_browser_select_data_field(data, k)
    # Make sure you can draw the mask
    s['/module/pixmap/draw_mask'] = True
    # Allocate your datafield
    datafield = gwy.gwy_app_data_browser_get_current(gwy.APP_DATA_FIELD)
    # Make a datafield for the mask
    mask = gwy.DataField.new_alike(datafield, False)
    # Add the mask datafield to the data container
    data['/%d/mask' % k] = mask
    # Determine the mask
    datafield.grains_mark_height(mask, phasethresh, False)
    # Plane level the data
    gwy.gwy_process_func_run("level", data, gwy.RUN_IMMEDIATE)
    # Fix zero
    gwy.gwy_process_func_run('fix_zero', data, gwy.RUN_IMMEDIATE)
    # Remove mask
    s['/module/pixmap/draw_mask'] = False


    # Set the colour scale:
    datafield = gwy.gwy_app_data_browser_get_current(gwy.APP_DATA_FIELD)

    # To get the mode, convert data to an array, round to 0.1deg and get the mode
    # Get your data as a list
    x = datafield.get_data()
    # Convert it to an array
    x_array = np.asarray(x)
    # Round it
    x_rounded = np.round_(x_array, decimals=1)
    # Get the mode
    mode = stats.mode(x_rounded)

    # Get the standard deviation and median of data
    rms = datafield.get_rms()
    # Round the rms to a nice number
    if rms >= 10:
        rms_rounded = (round(rms / 10, 0) * 10)
    elif 10 > rms >= 1:
        rms_rounded = (round(rms / 0.5, 1) * 0.5)
    elif 1 > rms >= 0.1:
        rms_rounded = (round(rms / 0.1, 1) * 0.1)
    else:
        rms_rounded = (round(rms / 0.05, 1) * 0.05)
    if rms_rounded == 0:
        rms_rounded = 0.01

    # Set the minimum as the median of the middle minus 2 rounded rms
    minphasescale = float(mode[0])
    # Set the maximum as the median of the middle plus 5 rounded rms
    maxphasescale = float(mode[0]) + 5 * rms_rounded

    return data, minphasescale, maxphasescale


def setcolourscale(data, k, minscale, maxscale):
    gwy.gwy_app_data_browser_select_data_field(data, k)
    # Set the image display to fixed z range
    data.set_int32_by_name("/" + str(k) + "/base/range-type", int(1))
    # Set the minimum and maximum
    data.set_double_by_name("/" + str(k) + "/base/min", float(minscale))
    data.set_double_by_name("/" + str(k) + "/base/max", float(maxscale))
    return data


def savefiles(data, filename, extension):
    # Get directory path and filename (excluding extension)
    directory, filenameext = os.path.split(filename)
    filename, ext = str.split(filenameext, '.jpk')

    # Create a folder called Processed to save results into
    savedir = os.path.join(directory, 'Processed')
    # If the folder Processed doesnt exist make it here
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # Select your data
    gwy.gwy_app_data_browser_select_data_field(data, k)

    # change the colour map for all channels (k) in the image:
    palette = data.set_string_by_name("/" + str(k) + "/base/palette", "Nanoscopegradient.txt")

    # Determine the title of each channel
    title = data["/%d/data/title" % k]

    # When saving as gwyddion file, don't add channel name to filename. Add '_processed' to the end
    if "gwy" in extension:
        savename = os.path.join(savedir, filename) + '_' + str(title) + '_processed' + str(extension)
    # When saving as anything else, generate a filename to save to by removing the extension to the file, adding the
    # suffix '_processed' and an extension set in the main file
    else:
        savename = os.path.join(savedir, filename) + '_' + str(k) + '_' + str(title) + str(extension)
    # Save the file
    gwy.gwy_file_save(data, savename, gwy.RUN_NONINTERACTIVE)



# This the main script
if __name__ == '__main__':
    # Set various options here:

    # Set the file path, i.e. the directory where the files are here
    path = 'C:/Users/gdbenn/Documents/Data/Georgina data/201911/20191126/sample2'
    # Set file type to look for here
    fileend = '.jpk'

    # Set the percentage threshold used in the mask in edit files functions
    heightthresh = 40
    phasethresh = 50


    # Set extension to export files as here e.g. '.tiff'
    extension = '.tiff'
    extension2 = '.gwy'


    # Look through the current directory for files ending in .jpk and add to file list
    # using the traversedirectories function
    jpkfiles = traversedirectories_nosubfolders(fileend, path)


    # Iterate over all files found
    for i, filename in enumerate(jpkfiles):
        print 'Analysing ' + str(os.path.basename(filename))
        # Load the data for the specified filename
        data = getdata(filename)
        # Find the channels of data you wish to use within the file e.g. ZSensor or height
        Height_ids, Phase_ids = choosechannels(data)
        # Iterate over the chosen channels in your file e.g. the ZSensor channel
        # Make sure mask is turned off at first
        s['/module/pixmap/draw_mask'] = False
        for k in Height_ids:
            # get some handy image deets
            xres, yres, xreal, yreal, dx, dy, max, min = imagedetails(data, k)
            # Perform basic image processing, to align rows, flatten and set the mean value to zero
            if xreal <= 1e-06:
                # Set the align rows settings (2nd order polynomial exclude mask)
                s['/module/linematch/mode'] = 0
                s['/modules/linematch/polynomial'] = 2
                s['/module/linematch/masking'] = int(gwy.MASK_EXCLUDE)
                height_data, minheightscale, maxheightscale = EditSmallHeightFile(data, k, min, max)
            else:
                # Set level data to exclude masked region
                s['/module/level/mode'] = int(gwy.MASK_EXCLUDE)
                height_data, minheightscale, maxheightscale = EditBigHeightFile(data, k, heightthresh)
            # Set the colour scales
            height_data_colour = setcolourscale(height_data, k, minheightscale, maxheightscale)
            # Save data as image
            savefiles(height_data_colour, filename, extension)
        for k in Phase_ids:
            # get some handy image deets
            xres, yres, xreal, yreal, dx, dy, max, min = imagedetails(data, k)
            # Perform basic image processing, to align rows, flatten and set the mean value to zero
            if xreal <= 1e-06:
                # Set the align rows settings (2nd order polynomial exclude mask)
                s['/module/linematch/mode'] = 0
                s['/modules/linematch/polynomial'] = 2
                s['/module/linematch/masking'] = int(gwy.MASK_EXCLUDE)
                phase_data, minphasescale, maxphasescale = EditSmallPhaseFile(data, k)
            else:
                # Set level data to exclude masked region
                s['/module/level/mode'] = int(gwy.MASK_EXCLUDE)
                phase_data, minphasescale, maxphasescale = EditBigPhaseFile(data, k, min, max, phasethresh)
            # Set the colour scales
            phase_data_colour = setcolourscale(phase_data, k, minphasescale, maxphasescale)
            # Save data as image
            savefiles(phase_data_colour, filename, extension)
            # Save data as gwyddion file
            # If the title of the datafield contains 'retrace' save the file in gwyddion. 'retrace is chosen because it
            # is analysed after trace. If this is not the case, just change to 'not in' or visa versa.
            name = str(data["/%d/data/title" % k])
            if "retrace" not in name:
            # if "retrace" in name:
                # Save data as image
                savefiles(phase_data, filename, extension2)