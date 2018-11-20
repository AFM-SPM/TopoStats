#!/usr/bin/env python2

import glob, sys, time, os, gtk, gwy, gwyutils, numpy as np
sys.path.append("/usr/local/Cellar/gwyddion/2.52/share/gwyddion/pygwy")

# Set the settings for each function from the saved settings file (~/.gwyddion/settings)
s = gwy.gwy_app_settings_get()

# Define the settings for image processing functions e.g. align rows here
s['/module/linematch/method'] = 1

# Define the settings for exporting images
#Font and linewidth info
s['/module/pixmap/font'] =  "Liberation Sans"
s['/module/pixmap/line_width'] =  float(2)
s["/module/pixmap/scale_font"] = True
s['/module/pixmap/font_size'] =  float(18)
#Put a scale bar for 100 nm on the image
s['/module/pixmap/inset_draw_label'] = True
s['/module/pixmap/inset_draw_text_above'] = True
s['/module/pixmap/inset_draw_ticks'] = False
s['/module/pixmap/inset_length'] = "100 nm"
s["/module/pixmap/inset_pos"] = 3 #bottom left = 3
#Keep the original image parameters - e.g. number of pixels
s["/module/pixmap/xytype"] = 2
s["/module/pixmap/zoom"] = float(1)
s["/module/pixmap/ztype"] = 1
#Export image with the mask on
s['/module/pixmap/draw_mask'] = True


def getfiles(filetype):
        # This function finds all the files in your current directory with the filetype set in the main script
        dir=os.getcwd()
        importdirectory = dir + filetype
        flist = glob.glob(importdirectory)
        if bool(flist) == True:
            print 'Files found: ' + str(len(flist))
        else:
            filetype = '/*.*[0-9]'
            print 'No files found, trying next filetype'
            importdirectory = dir + filetype
            flist = glob.glob(importdirectory)
            print 'Files found: ' + str(len(flist))
        return flist, dir

def getallfiles(filetype):
        dir = '/Users/alice/Dropbox/UCL/DNA MiniCircles/Minicircle Data/20160215_339-2_8ng_Ni_3mM'
        filetype = filetype
        importdirectory = dir + filetype
        flist = glob.glob(importdirectory)
        if bool(flist) == True:
            print 'Files found: ' + str(len(flist))
        else:
            filetype = '/*.*[0-9]'
            print 'No files found, trying next filetype'
            importdirectory = dir + filetype
            flist = glob.glob(importdirectory)
            print 'Files found: ' + str(len(flist))
        return flist

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
        if bool(chosen_ids) == False:
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
        # level the data
        gwy.gwy_process_func_run("level", data, gwy.RUN_IMMEDIATE)
        # align the rows
        gwy.gwy_process_func_run("align_rows", data, gwy.RUN_IMMEDIATE)  # NONINTERACTIVE is only for file modules
        # Fix zero
        gwy.gwy_process_func_run('zero_mean', data, gwy.RUN_IMMEDIATE)
        #remove scars
        gwy.gwy_process_func_run('scars_remove', data, gwy.RUN_IMMEDIATE)
        # # Apply a 1.5 pixel gaussian filter
        # data_field = gwy.gwy_app_data_browser_get_current(gwy.APP_DATA_FIELD)
        # data_field.filter_gaussian(1.5)
        # # Shift contrast - equivalent to 'fix zero'
        # datafield.add(-data_field.get_min())

        # Set the image display to fized range and the colour scale for the images
        maximum_disp_value = data.set_int32_by_name("/"+str(k)+"/base/range-type", int(1))
        minimum_disp_value = data.set_double_by_name("/"+str(k)+"/base/min", float(minheightscale))
        maximum_disp_value = data.set_double_by_name("/"+str(k)+"/base/max", float(maxheightscale))
        return data, filename

def grainfinding(data, DNAarea):
        # select each channel of the file in turn
        gwy.gwy_app_data_browser_select_data_field(data, k) 
        datafield = gwy.gwy_app_data_browser_get_current(gwy.APP_DATA_FIELD)
        mask = gwy.DataField.new_alike(datafield, False)
        mask2 = gwy.DataField.new_alike(datafield, False)

        # Mask data that are above thresh*sigma from average height.
        # Sigma denotes root-mean square deviation of heights. 
        # This criterium corresponds to the usual Gaussian distribution outliers detection if thresh is 3.
        datafield.mask_outliers(mask, 1)

        ## Editing grain mask
        # Remove grains touching the edge of the mask
        mask.grains_remove_touching_border()
        # Calculate pixel width in nm
        dx = datafield.get_dx()
        # Calculate minimum feature size in pixels (integer)
        minsize = int(DNAarea/dx)
        maxsize = int(3.5*minsize)
        # Remove grains smaller than (size) in integer pixels
        mask.grains_remove_by_size(minsize)

        # Numbering grains for grain analysis
        grains = mask.number_grains()

        # Update data to show mask, comment out to remove mask
        # data['/%d/mask' % k] = mask
        return data, mask, datafield, grains

def grainanalysis(directory, filename, data, mask, datafield, grains):
        ### Setting up filenames and directories for writing data
        # Get the base of the filename i.e. the last part without directory or extension
        filename = os.path.splitext(os.path.basename(filename))[0]
        # If the folder Grain Statistics doest exist make it here
        if not os.path.exists(directory + '/GrainStatistics/'):
            os.makedirs(directory + '/GrainStatistics/')
            grain_directory = directory + '/GrainStatistics/'
        # Otherwise set the existing GrainStatistics dorectory as the directory to write to
        else:
            grain_directory = directory + '/GrainStatistics/'


        ### Calculating grain statistics using numbered grains file
        # Statistics to be computed should be specified here as a dictionary
        values_to_compute = {'grain_proj_area' : gwy.GRAIN_VALUE_PROJECTED_AREA,
                    'grain_maximum' : gwy.GRAIN_VALUE_MAXIMUM,
                    'grain_mean' : gwy.GRAIN_VALUE_MEAN, 
                    'grain_median' : gwy.GRAIN_VALUE_MEDIAN,
                    'grain_pixel_area' : gwy.GRAIN_VALUE_PIXEL_AREA,
                    'grain_half_height_area' : gwy.GRAIN_VALUE_HALF_HEIGHT_AREA, 
                    'grain_bound_len' : gwy.GRAIN_VALUE_FLAT_BOUNDARY_LENGTH,
                    'grain_min_bound_size' : gwy.GRAIN_VALUE_MINIMUM_BOUND_SIZE,
                    'grain_max_bound_size' : gwy.GRAIN_VALUE_MAXIMUM_BOUND_SIZE,
                    'grain_center_x' : gwy.GRAIN_VALUE_CENTER_X, 
                    'grain_center_y' : gwy.GRAIN_VALUE_CENTER_Y,
                    'grain_curvature1' : gwy.GRAIN_VALUE_CURVATURE1,
                    'grain_curvature2' : gwy.GRAIN_VALUE_CURVATURE2,
                    'grain_mean_radius' : gwy.GRAIN_VALUE_MEAN_RADIUS,
                    'grain_ellipse_angle' : gwy.GRAIN_VALUE_EQUIV_ELLIPSE_ANGLE,
                    'grain_ellipse_major' : gwy.GRAIN_VALUE_EQUIV_ELLIPSE_MAJOR,
                    'grain_ellipse_minor' : gwy.GRAIN_VALUE_EQUIV_ELLIPSE_MINOR,
                    }
        # Create empty dictionary for data to be saved out
        grain_data_to_save = {}


        ### Saving out the Grain Statistics  
        try:
            # Write the statistics to a file called: Grain_Statistics_filename.txt
            write_file = open(grain_directory + 'Grain_Statistics_' + filename + '.txt', 'w')
            # Write a header for the file
            print >>write_file, '#This file contains the grain statistics from file ' + filename + '\n\n'
            # Iterate over each grain statistic (key) and save out as 'grainstats'
            for key in values_to_compute.keys():
                ### here we stave the gran stats to both a dictionary and an array in that order
                ### these are basically duplicate steps - but are both included as we arent sure which to use later
                # Save grain statistics to a dictionary: grain_data_to_save
                grain_data_to_save[key] = datafield.grains_get_values(grains, values_to_compute[key])
                # Delete 0th value in all arrays - this corresponds to the background
                del grain_data_to_save[key][0]
                # Save grain statistcis to an array: grainstats
                grainstats = datafield.grains_get_values(grains, values_to_compute[key])
                # Delete 0th value in all arrays - this corresponds to the background
                del grainstats[0]
                ### Currently save only the array to a text file
                # Use numpy (np) to save out grain values as text files
                # np.savetxt('{}.txt'.format(str(key)),grainstats)
                # Write each grain statistic type to the file Grain_Statistics_filename.txt (must be within for loop)
                print >>write_file, str(key) + '\n' + str(grainstats) + '\n'

        except TypeError:
            write_file = open(grain_directory + 'Grain_Statistics_' + filename + '.txt', 'w')
            print >>write_file, '#The file ' + filename + ' contains no detectable grain statistics'
        print 'Saving grain statistics for: ' + str(filename)

        return values_to_compute, grainstats, grain_data_to_save

def grainthinning(data):
        ## script to edit image with higher gaussian filtering to obtain mask for tracing
        gwy.gwy_app_data_browser_select_data_field(data, k) 
        datafield = gwy.gwy_app_data_browser_get_current(gwy.APP_DATA_FIELD)
        mask = gwy.DataField.new_alike(datafield, False)

        # Calculate gaussian width from pixel size
        dx = datafield.get_dx()
        Gaussiansize = 2e-9/dx
        # Gaussian filter data
        datafield.filter_gaussian(Gaussiansize)

        # Thresholding data above 1 * sigma * average height.
        datafield.mask_outliers(mask, 1)
        
        ## Editing grain mask
        # Remove grains touching the edge of the mask
        mask.grains_remove_touching_border()
        # Calculate pixel width in nm
        dx = datafield.get_dx()
        # Calculate minimum feature size in pixels (integer)
        minsize = int(DNAarea/dx)
        # Remove grains smaller than (size) in integer pixels
        mask.grains_remove_by_size(minsize)

        # Numbering grains for grain analysis
        grains = mask.number_grains()

        # Thin (skeletonise) gaussian filtered grains to get traces
        mask.grains_thin()

        # Update data to show mask, comment out to remove mask
        data['/%d/mask' % k] = mask
        return data, mask

def savefiles(data, filename, extension):
        # Save the data for the channels found above i.e. ZSensor/Height, as chosen_ids
        # Data is exported to a file of extension set in the main script
        # Data is exported with the string '_processed' added to the end of its filename
        gwy.gwy_app_data_browser_select_data_field(data, k)
        # change the colour map for all channels (k) in the image:
        palette = data.set_string_by_name("/"+str(k)+"/base/palette", "Nanoscope")
        # Determine the title of each channel 
        title = data["/%d/data/title" % k]
        # Determine the filename for each file including path
        filename = os.path.splitext(filename)[0]
        # Generate a filename to save to by removing the extension to the file, adding the suffix '_processed'
        # and an extension set in the main file 
        savename = filename + '_' + str(k) +'_' + str(title) + '_processed' + str(extension)
        # Save the file
        gwy.gwy_file_save(data, savename, gwy.RUN_NONINTERACTIVE) 
        # Show the mask
        data['/%d/mask' % k] = mask
        # Add the sufix _masked to the previous filename
        savename = filename + '_' + str(k) +'_' + str(title) + '_processed_masked' + str(extension)
        # Save the data
        gwy.gwy_file_save(data, savename, gwy.RUN_NONINTERACTIVE) 

        # Print the name of the file you're saving to the command line
        print 'Saving file: ' + str((os.path.splitext(os.path.basename(savename))[0]))    



# This the main script
if __name__ == '__main__':

    #Set various options here:    
    # Set file type to run here e.g.'/*.spm*'
    filetype = '/*.spm' #default
    # filetype = '/*.*[0-9]'
    # filetype = '/*.gwy'
    # Set extension to export files as here e.g. '.tiff'
    extension = '.tiff'
    # # Set saving values for DNA 
    # minheightscale = -1e-9
    # maxheightscale = 3e-9
    # # Set values for determining sizes:
    # sequencelength = 339
    # DNAarea = sequencelength*0.34*2e-9

    # Set saving values for MAC 
    minheightscale = -30e-9
    maxheightscale = 30e-9
    # Set values for determining sizes:
    DNAarea = 500e-9

    # Call the first function, which finds the files in your current directory
    flist, dir = getfiles(filetype)
    # Call the first function, which finds the files in a set directory
    # flist = getallfiles(filetype)
    # Iterate over all files found 
    for i, filename in enumerate(flist):
        data = getdata(filename)
        chosen_ids = choosechannels(data)
        for k in chosen_ids:
            xres, yres, xreal, yreal, dx, dy = imagedetails(data)
            data, filename = editfile(data, minheightscale, maxheightscale)
            data, mask, datafield, grains = grainfinding(data, DNAarea)
            values_to_compute, grainstats, grain_data_to_save = grainanalysis(dir, filename, data, mask, datafield, grains)
            # data, mask = grainthinning(data)
            savefiles(data, filename, extension)
        gwy.gwy_app_data_browser_remove(data) #close the file once we've finished with it 
