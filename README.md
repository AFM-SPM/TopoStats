# TopoStats
An AFM image analysis program to batch process data and obtain statistics from images

Image progessing is performed using the 'pygwytracing' script

The algorithm searches recursively for files within a user-defined directory. 
This search also excludes any files of the format ‘_cs’ which are cropped files exported by the Nanoscope Analysis software. 
AFM images are loaded  using gwyddion functions and topography data is automatically selected using the choosechannels function. 
The pixel size and dimensions of each image are determined using the imagedetails function, which allows all inputs to be specified in real, i.e. nanometre values, in place of pixel values. 
This is especially important for datasets with changing resolution. 

Basic image processing is performed in the function editfile which uses the functions: ‘align rows’ to remove offsets between scan lines; ‘level’ to remove sample tilt as a first order polynomial; ‘flatten base’ which uses a combination of facet and polynomial levelling with automated masking; and ‘zeromean’ which sets the mean value of the image, i.e. the background, to zero. 
A gaussian filter (sigma = 1.5) of 3.5 pixels (1-2 nm) is applied to remove pixel errors and high frequency noise. 

Single DNA molecules are identified in images using a modified extension of Gwyddion’s automated masking protocols, in which masks are used to define the positions of individual features (grains) on the imaged surface. 
The grains within a flattened AFM image are identified using the ‘mask_outliers’ function, which masks data  points with height values that deviate from the mean by more than 1sigma (with 3sigma corresponding to a standard gaussian). 
Grains which touch the edge of the image (i.e. are incomplete) are removed using the ‘grains_remove_touching_border’ function and grains which are smaller than 200 nm2 are removed using the ‘grains_remove_by_size’ function. 
Erroneous grains are removed using the removelargeobjects and removesmallobjects functions, which themselves use the function find_median_pixel_area to determine the size range of objects to remove. The ‘grains_remove_by_size’ function is then called again to remove grains which fall outside 50 % - 150 % of the median grain area determined in the previous step.  

Grain statistics are then calculated for each image using the grainanalysis function which utilises the ‘grains_get_values’ function to obtain a number of statistical properties which are saved using the saveindividualstats function as ‘.json’ and ‘.txt’ files for later use in a subdirectory ‘GrainStatistics’ in the specified path. 
In addition, each grain’s values are appended to an array [appended_data], to statistically analyse the morphologies of DNA molecules from all images for a given experiment (presumed to be within a single  directory). 
This array is converted to a pandas dataframe using the getdataforallfiles function and saved out using the savestats function as ‘.json’ and ‘.txt’ files with the name of the directory in the original path. 

Individual grains (i.e. isolated molecules) are cropped out using the function bbox, which uses the grain centre x and y positions obtained in the grainanalysis function to duplicate the original image and crop it to a predefined size (here 80 nm) around the centre of the grain. These images are then labelled with the grain ID and saved out as tiff files in a subdirectory ‘Cropped’ in the specified path.
To allow for further processing in python, there is an option to obtain the image or mask as a numpy array41, using the function exportasnparray. The processed image, and a copy with the mask overlaid are saved out using the savefiles function to a subdirectory ‘Processed’ in the specified path. 

Statistical analysis and plotting is performed using the 'statsplotting' script. 
This script uses the importfromjson function to import the JSON format file exported by pygwytracing and calculates various statistical parameters for all grain quantities, e.g. length, width and saves these out as a new JSON file using the savestats function. 
Both KDE plots and histograms are generated for any of the grain quantities using the matplotlib42 and seaborn43 libraries within the functions plotkde, plotcolumns and plothist. 


