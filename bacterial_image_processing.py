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

# Set the settings for each function from the saved settings file (~/.gwyddion/settings) (C:\Users\gdbenn\gwyddion\settings)
s = gwy.gwy_app_settings_get()
# You alter these settings as you go



## The following functions are to find .jpk files in directory and/or subdirectories


# This function finds all the files ending in .jpk in the path directory
def traversedirectories_nosubfolders(fileend, path):
    # initialise the list
    jpkfiles = []
    filelist = os.listdir(path)
    for i in filelist:
        # Find files ending in 'fileend'
        if i.endswith((fileend)):
            jpkfiles.append(os.path.join(path, i))
    # print the number of files found
    print 'Files found: ' + str(len(jpkfiles))
    # return a list of files including their root and the original path specified
    return jpkfiles

# This function finds all the files ending in .jpk in the path directory and its subfolders
def traversedirectories(fileend, path):
    # initialise the list
    jpkfiles = []
    # use os.walk to search folders and subfolders and append each file with the correct filetype to the list jpkfiles
    for dirpath, subdirs, files in os.walk(path):
        # Looking for files ending in fileend
        for filename in files:
                # Find files ending in 'fileend'
                if filename.endswith((fileend)):
                    jpkfiles.append(os.path.join(dirpath, filename))
        # print the number of files found
    print 'Files found: ' + str(len(jpkfiles))
    # return a list of files including their root and the original path specified
    return jpkfiles


# This function finds all files ending in .jpk in the path directory and corresponding .tiff files
# This was used for the conversion of .jpk files to .txt files as phase images were first analysed
# entirely in python to determine if trace or retrace were best. The corresponding .tiff image
# was then placed in the same folder and used here
def traversedirectories_nosubfolders_tiff(fileend, path):
    # initialise the lists
    jpkfiles = []
    tifffiles = []
    filelist = os.listdir(path)
    for i in filelist:
        # Find files ending in 'fileend'
        if i.endswith((fileend)):
            jpkfiles.append(os.path.join(path, i))
        # Find phase .tiff files corresponding to jpk files you want to use
        if i.endswith(('.tif', '.tiff')) and '_Lock' in i:
            tifffiles.append(os.path.join(path, i))
    # Remove jpk files without corresponding tiff
    todelete = []
    for i in range(len(jpkfiles)):
        nameA = jpkfiles[i].split('.jpk')
        name = nameA[0].split('/')
        if not str(name[-1]) in str(tifffiles):
           todelete.append(jpkfiles[i])
    for i in range(len(todelete)):
        jpkfiles.remove(todelete[i])


    # print the number of files found
    print 'Files found: ' + str(len(jpkfiles))
    # return a list of files including their root and the original path specified
    return jpkfiles, tifffiles


## The following functions are used to get the data and channels you want to use

def getdata(filename):
    # Gets the data you want from a filename to something gwyddion can use
    # Open file and add data to browser
    data = gwy.gwy_file_load(filename, gwy.RUN_NONINTERACTIVE)
    # Add data to browser
    gwy.gwy_app_data_browser_add(data)
    return data

# Find both trace and retrace phase and height channels. This is useful when initially checking
# which trace to use.
def choosechannels_both(data):
    # Obtain the data field ids for the data file (i.e. the various channels within the file)
    gwy.gwy_app_data_browser_get_data_ids(data)
    # Find trace AND retrace channels with the title Height (measured),
    # if these do not exist, find channels with the title Height
    Height_ids = gwy.gwy_app_data_browser_find_data_by_title(data, 'Height (measured)*')
    if Height_ids == []:
        Height_ids = gwy.gwy_app_data_browser_find_data_by_title(data, 'Height*')
    Phase_ids = gwy.gwy_app_data_browser_find_data_by_title(data, 'Lock-In Phase*')
    # Save out chosen_ids as a list of the channel ids for Height and phase

    return Height_ids, Phase_ids

# Find trace phase and height channels
def choosechannels_trace(data):
    # Obtain the data field ids for the data file (i.e. the various channels within the file)
    gwy.gwy_app_data_browser_get_data_ids(data)
    # Find trace channels with the title Height (measured),
    # if these do not exist, find channels with the title Height
    Height_ids = gwy.gwy_app_data_browser_find_data_by_title(data, 'Height (measured)')
    if Height_ids == []:
        Height_ids = gwy.gwy_app_data_browser_find_data_by_title(data, 'Height')
    Phase_ids = gwy.gwy_app_data_browser_find_data_by_title(data, 'Lock-In Phase')
    # Save out chosen_ids as a list of the channel ids for Height and phase
    return Height_ids, Phase_ids

# Find retrace phase and height channels
def choosechannels_retrace(data):
    # Obtain the data field ids for the data file (i.e. the various channels within the file)
    gwy.gwy_app_data_browser_get_data_ids(data)
    # Find retrace channels with the title Height (measured),
    # if these do not exist, find channels with the title Height
    Height_ids = gwy.gwy_app_data_browser_find_data_by_title(data, 'Height (measured) (retrace)')
    if Height_ids == []:
        Height_ids = gwy.gwy_app_data_browser_find_data_by_title(data, 'Height (retrace)')
    Phase_ids = gwy.gwy_app_data_browser_find_data_by_title(data, 'Lock-In Phase (retrace)')

    # Save out chosen_ids as a list of the channel ids for Height and phase

    return Height_ids, Phase_ids

# This function is used with traversedirectories_nosubfolders_tiff() to find correct trace or retrace
def choosechannels_traceorretrace(data, tifffiles):
    # Obtain the data field ids for the data file (i.e. the various channels within the file)
    gwy.gwy_app_data_browser_get_data_ids(data)
    if 'retrace' in tifffiles[i]:
        # Find trace channels with the title Height (measured),
        # if these do not exist, find channels with the title Height
        Height_ids = gwy.gwy_app_data_browser_find_data_by_title(data, 'Height (measured) (retrace)')
        if Height_ids == []:
            Height_ids = gwy.gwy_app_data_browser_find_data_by_title(data, 'Height (retrace)')
        Phase_ids = gwy.gwy_app_data_browser_find_data_by_title(data, 'Lock-In Phase (retrace)')
    else:
        # Find trace channels with the title Height (measured),
        # if these do not exist, find channels with the title Height
        Height_ids = gwy.gwy_app_data_browser_find_data_by_title(data, 'Height (measured)')
        if Height_ids == []:
            Height_ids = gwy.gwy_app_data_browser_find_data_by_title(data, 'Height')
        Phase_ids = gwy.gwy_app_data_browser_find_data_by_title(data, 'Lock-In Phase')


    # Save out chosen_ids as a list of the channel ids for Height and phase
    return Height_ids, Phase_ids

## These functions are to process files in python with pygwy

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

def EditSmallHeightFile(data, k, min, max, LH, UH):
    # select each channel of the file in turn
    # this is run within the "for k in chosen_ids" loop so k refers to the
    # index of each chosen channel to analyse (see main script below)
    gwy.gwy_app_data_browser_select_data_field(data, k)


    # If the range is large (more than 50 nm), apply a mask to the lower part of the image
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
    gwy.gwy_process_func_run('align_rows', data, gwy.RUN_IMMEDIATE)
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

    # Set the minimum as the mode minus LH nm and the maximum as the mode plus UH nm
    # LH and UH are set in the main script
    minheightscale = float(mode[0]) - (LH*1e-09)
    maxheightscale = float(mode[0]) + (UH*1e-09)

    return data, minheightscale, maxheightscale

def EditSmallPhaseFile(data, k, LP, UP):
    # select each channel of the file in turn
    # this is run within the "for k in chosen_ids" loop so k refers to the
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


    # Set the minimum as the mean minus LP deg and the minimum plus UP deg
    # LP and UP are set in the main script
    mean = datafield.get_avg()
    minphasescale = mean - LP
    maxphasescale = minphasescale + UP

    return data, minphasescale, maxphasescale

def setcolourscale(data, k, minscale, maxscale):
    gwy.gwy_app_data_browser_select_data_field(data, k)
    # Set the image display to fixed z range
    data.set_int32_by_name("/" + str(k) + "/base/range-type", int(1))
    # Set the minimum and maximum
    data.set_double_by_name("/" + str(k) + "/base/min", float(minscale))
    data.set_double_by_name("/" + str(k) + "/base/max", float(maxscale))
    return data


## These functions are to save files. A different function is used when converting to .txt

# For .txt files
def savefiles(data, filename, extension):
    # Get directory path and filename (excluding extension)
    directory, filenameext = os.path.split(filename)
    filename, ext = str.split(filenameext, '.jpk')

    # Create a folder called Processed to save results into
    savedir = os.path.join(directory, 'Text')
    # If the folder Processed doesn't exist make it here
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # Select your data
    gwy.gwy_app_data_browser_select_data_field(data, k)


    # Determine the title of each channel
    title = data["/%d/data/title" % k]
    title_spaceless = str.replace(title, ' ', '')

    # Get save name
    savename = os.path.join(savedir, filename) + '_' + str(title_spaceless) + str(extension)
    # Save the file
    gwy.gwy_file_save(data, savename, gwy.RUN_NONINTERACTIVE)

# For when full analysis is in python
def savefiles(data, filename, extension):
    # Get directory path and filename (excluding extension)
    directory, filenameext = os.path.split(filename)
    filename, ext = str.split(filenameext, '.jpk')

    # Create a folder called Processed to save results into
    savedir = os.path.join(directory, 'Processed')
    # If the folder Processed doesn't exist make it here
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # Select your data
    gwy.gwy_app_data_browser_select_data_field(data, k)

    # change the colour map for all channels (k) in the image:
    palette = data.set_string_by_name("/" + str(k) + "/base/palette", "Nanoscope.txt")

    # Determine the title of each channel
    title = data["/%d/data/title" % k]

    # When saving as gwyddion file, don't add channel name to filename. Add '_processed' to the end
    if "gwy" in extension:
        savename = os.path.join(savedir, filename) + '_processed' + str(extension)

    # When saving as anything else, generate a filename to save to by removing the extension to
    # the file and adding the channel name
    else:
        savename = os.path.join(savedir, filename) + '_' + str(title) + str(extension)
    # Save the file
    gwy.gwy_file_save(data, savename, gwy.RUN_NONINTERACTIVE)


## This is the main script for when full processing is done in python with pygwy. This was edited
## depending on the requirements


if __name__ == '__main__':
    # Set various options here:

    # Set the file path, i.e. the directory where the files are here
    path = ''
    # Set file type to look for here
    fileend = '.jpk'

    # Set the upper and lower scales for small height (in nm) and phase (in deg) channels
    # (if you need to change the lower and upper limits of the big files, you will have to do this in the functions)
    LH = 3
    UH = 7
    LP = 0.75
    UP = 2.75
    # Set the percentage threshold used in the mask in edit files functions
    heightthresh = 40
    phasethresh = 50


    # Set extension to export files as here e.g. '.tiff'
    extension = '.tiff'
    extension2 = '.gwy'


    # Look through the current directory (with or without subfolders) for files ending in .jpk and add to file list
    # using the traversedirectories function
    # jpkfiles = traversedirectories_nosubfolders(fileend, path)
    jpkfiles = traversedirectories(fileend, path)

    # Iterate over all files found
    for i, filename in enumerate(jpkfiles):
        print 'Analysing ' + str(os.path.basename(filename))
        # Load the data for the specified filename
        data = getdata(filename)

        # Find the channels of data you wish to use within the file e.g. ZSensor or height.
        # Also decide if you want to analyse the trace, retrace or both
        Height_ids, Phase_ids = choosechannels_both(data)
        # Height_ids, Phase_ids = choosechannels_trace(data)
        # Height_ids, Phase_ids = choosechannels_retrace(data)

        # Make sure mask is turned off at first
        s['/module/pixmap/draw_mask'] = False

        # Iterate over the chosen channels in your file

        for k in Height_ids:
            # get some image details you might need
            xres, yres, xreal, yreal, dx, dy, max, min = imagedetails(data, k)

            # Perform basic image processing, to align rows, flatten and set the mean value to zero
            # Check the image is not on  a whole cell image
            if xreal < 1e-06:
                 # Set the align rows settings (2nd order polynomial exclude mask)
                 s['/module/linematch/mode'] = 0
                 s['/modules/linematch/polynomial'] = 2
                 s['/module/linematch/masking'] = int(gwy.MASK_EXCLUDE)
                 # Process channel
                 height_data, minheightscale, maxheightscale = EditSmallHeightFile(data, k, min, max, LH, UH)
            # Set the colour scales
            height_data_colour = setcolourscale(height_data, k, minheightscale, maxheightscale)
            # Save data as image and gwyddion file
            savefiles(height_data_colour, filename, extension)
            savefiles(height_data_colour, filename, extension2)

        for k in Phase_ids:
            # get some image details you might need
            xres, yres, xreal, yreal, dx, dy, max, min = imagedetails(data, k)

            # Perform basic image processing, to align rows, flatten and set the mean value to zero
            # Check the image is not on  a whole cell image
            if xreal < 1e-06:
                 # Set the align rows settings (2nd order polynomial exclude mask)
                 s['/module/linematch/mode'] = 0
                 s['/modules/linematch/polynomial'] = 2
                 s['/module/linematch/masking'] = int(gwy.MASK_EXCLUDE)
                 # Process channel
                 phase_data, minphasescale, maxphasescale = EditSmallPhaseFile(data, k, LP, UP)
            # Set the colour scales
            phase_data_colour = setcolourscale(phase_data, k, minphasescale, maxphasescale)
            # Save data as image and gwyddion file
            savefiles(phase_data_colour, filename, extension)
            savefiles(phase_data_colour, filename, extension2)


## This is the main script for when the files are exported to imageJ for further processing

if __name__ == '__main__':
    # Set various options here:

    # Set the file path, i.e. the directory where the files are here
    path = ''

    # Set file type to look for here
    fileend = '.jpk'

    # Set extension to export files as here e.g. '.tiff'
    extension = '.txt'

    # Look through the current directory (with or without subfolders) for files ending in .jpk and add to file list
    # using the traversedirectories function
    jpkfiles = traversedirectories_nosubfolders(fileend, path)
    # jpkfiles, tifffiles = traversedirectories_nosubfolders_tiff(fileend, path)

    # Iterate over all files found
    for i, filename in enumerate(jpkfiles):
        print 'Analysing ' + str(os.path.basename(filename))
        # Load the data for the specified filename
        data = getdata(filename)

        # Find the channels of data you wish to use within the file e.g. ZSensor or height.
        # Also decide if you want to analyse the trace, retrace or both
        Height_ids, Phase_ids = choosechannels_both(data)
        # Height_ids, Phase_ids = choosechannels_trace(data)
        # Height_ids, Phase_ids = choosechannels_retrace(data)
        # Height_ids, Phase_ids = choosechannels_traceorretrace(data, tifffiles)

        for k in Phase_ids:
             # Plane level the data
             gwy.gwy_process_func_run("level", data, gwy.RUN_IMMEDIATE)
             # Set the align rows settings (1st order polynomial exclude mask)
             s['/module/linematch/mode'] = 0
             s['/modules/linematch/polynomial'] = 1
             # select each channel of the file in turn
             # this is run within the "for k in chosen_ids" loop so k refers to the
             # index of each chosen channel to analyse (see main script below)
             gwy.gwy_app_data_browser_select_data_field(data, k)
             # align the rows polynomial 1st order (set in main script)
             gwy.gwy_process_func_run("align_rows", data, gwy.RUN_IMMEDIATE)
             # save
             savefiles(data, filename, extension)

        for k in Height_ids:
            # Plane level the data
            gwy.gwy_process_func_run("level", data, gwy.RUN_IMMEDIATE)
            # Set the align rows settings (1st order polynomial exclude mask)
            s['/module/linematch/mode'] = 0
            s['/modules/linematch/polynomial'] = 1
            # select each channel of the file in turn
            # this is run within the "for k in chosen_ids" loop so k refers to the
            # index of each chosen channel to analyse (see main script below)
            gwy.gwy_app_data_browser_select_data_field(data, k)
            # align the rows polynomial 1st order (set in main script)
            gwy.gwy_process_func_run("align_rows", data, gwy.RUN_IMMEDIATE)
            # save
            savefiles(data, filename, extension)