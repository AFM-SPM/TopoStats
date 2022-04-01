def savefiles(data, filename, extension, options, k, savefile_zscalecolour, mask):
    """
    
    """

    """ Old Gwyddion function - sets plot output options"""
    # Save file scale option: 1 - ruler, 2 - inset scale bar, 0 - none
    #s["/module/pixmap/xytype"] = savefilesScale_option
    """ Needs a way to get this into the matplotlib plots """




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
    palette = data.set_string_by_name("/%s/base/palette" % k, savefile_zscalecolour)






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



