import numpy as np
import re
from itertools import cycle
from skimage.segmentation import active_contour
import scipy.ndimage.filters
from math import sqrt,acos, degrees, atan2, pi, cos, sin
from scipy import spatial, ndimage, interpolate as interp
from scipy.ndimage import morphology as morph, measurements, fourier, interpolation as imginterp
from skimage import feature, filters, morphology, segmentation, measure, transform
import matplotlib.pyplot as plt
import os


def LoadGwydTextFileToArray(file_name, directory):
	data = np.loadtxt(directory+file_name, skiprows=4)
	return data


def GetDimensions(Header, dim_name):
	DimRE = re.compile(dim_name+'\: *(\d+\.?\d+?) *(\w+)')
	Search = DimRE.search(Header)
	if Search.group(2) == 'nm':			#Checking for nm units and correcting for this
		dim_val = (float(Search.group(1))*10)
		print 'The '+ dim_name +' is ' +str(dim_val)+' angstroms.'
	else:
		print   +' not in nm.'
		dim_val = None
	return dim_val


def GetScale(Header):
	ScaleRE = re.compile('Value units\: *(\w+)')
	Search = ScaleRE.search(Header)
	if Search.group(1) == 'm':
		print 'Scale is '+Search.group(1)
		return 10.**10
	elif Search.group(1) == 'um':
		print 'Scale is '+Search.group(1)
		return 10.**4
	elif Search.group(1) == 'nm':
		print 'Scale is '+Search.group(1)
		return 10.
	else:
		print 'Scale is assumed to be angstroms.'
		return 1.


def GetHeightWidthScale(file_name, directory):
	"""
	Assumes Gwyddion .txt fixed format.
	"""
	File = open(directory + file_name, 'r')
	Header = File.read()
	File.close()
	width = GetDimensions(Header, 'Width')
	height = GetDimensions(Header, 'Height')
	scale = GetScale(Header)
	return height, width, scale


def DefineFeature(data, width, height, sequence_length, DNA_diameter, extra_info=False):
	pixel_size = width/(data.shape[0]-1)
	print "Pixel size (A): {}".format(pixel_size)

	minicircle_length = (33.2*(sequence_length/10.5))
	print "Minicircle length (A): {}".format(minicircle_length)

	minicircle_area = minicircle_length*DNA_diameter
	print "Minicircle area (A): {}".format(minicircle_area)

	min_size = (minicircle_area/pixel_size)/10 #needs to be defined in dimensionless units - grid points
	print "Minimum feature size (A): {}".format(min_size)

	# Everything below here is for information only and may be deleted if not required
	if extra_info:
		expected_minicircle_dia=minicircle_length/pi
		print "Expected minicircle diameter (A): {}".format(expected_minicircle_dia)
	
		minicircle_px=expected_minicircle_dia*pixel_size
		print "Expected minicircle size (px): {}".format(minicircle_px) 
	
		circle_pixels = minicircle_area/(pixel_size/sqrt(2))
		print "Circle pixels: {}".format(circle_pixels)
		#pixels = circle_pixels - (0.1*circle_pixels)	#Don't know what this is for? !!

		sigma = sigma = (20/sqrt(pixel_size))/1.5  #2*pixel_size
		print "sigma: {}".format(sigma)

	return min_size, pixel_size, minicircle_length


def GetOtsuMask(data, min_size, pixel_size):
	connectivitymap = np.ones((2,2)) # Erosion shape

	# # Gaussian filter data
	# data = filters.gaussian(data,1.5)

	# Obtain otsu threshold value
	otsu_value = filters.threshold_otsu(data)
	print otsu_value
	otsu_mask = data > otsu_value

	# Remove small objects
	morphology.remove_small_objects(otsu_mask, min_size = min_size, in_place = True) 





	
	# Apply mask to data
	otsu_data = np.multiply(data, otsu_mask)

	# # Threshold data to remove anything but top 5%
	# thresholded_otsu = otsu_data > np.percentile(otsu_data,90)
	# otsu_data = np.multiply(otsu_data,thresholded_otsu)

# Erode if required to thin
# if pixel_size<30:
# 	otsu_eroded_mask = morph.binary_erosion(otsu_data, structure = connectivitymap, iterations = 1)
# elif pixel_size<10:
# 	otsu_eroded_mask = morph.binary_erosion(otsu_data, structure = connectivitymap, iterations = 2)
# else:
	otsu_eroded_mask = otsu_mask

	# #Close small holes in the mask using a binary closing operation
	# otsu_eroded_mask=morph.binary_closing(otsu_data)

	return otsu_eroded_mask, otsu_data, otsu_mask

#def GetTraces(file_name, directory, sequence_length, DNA_diameter):
#	height, width, scale = GetHeightWidthScale(file_name, directory)
#	data = LoadGwydTextFileToArray(file_name, directory)
#	data = scale*data
#
#	min_size, pixel_size = DefineFeature(data, width, height, sequence_length, DNA_diameter, extra_info=False)
#	
#	# Otsu filter data 
#	otsu_eroded_mask = GetOtsuMask(data, min_size, pixel_size)
#
#	# Skeletonise ostu filtered data (acts on mask)
#	skele_otsu = morphology.skeletonize(otsu_eroded_mask)
#
#	return skele_otsu, pixel_size
 
def Edgefinding(minicircle_length, pixel_size, otsu_data, otsu_mask):
	connectivitymap = np.ones((3,3))
	trace_length = minicircle_length # expected trace length
	print 'Expected trace length:', trace_length

	# Laplace filter otsu masked image to obtain the traces of the edges of the features
	otsu_edges = scipy.ndimage.filters.laplace(otsu_mask)

	# Perform a binary closing operation to fill in any gaps in the traces
	otsu_edges = morphology.binary_closing(otsu_edges)

	# Determine the number of features in the image and return an image with the features labelled iteratively 
	mask = otsu_edges > 0
	labeled_otsu_edges, num_features = measurements.label(otsu_edges, structure = connectivitymap)
	print 'No features:', num_features

	# Measure the lengths of the traces
	DNA_lengths = measurements.sum(mask, labeled_otsu_edges, range(0, num_features+1))
	print DNA_lengths

	# Remove small objects less 1/2 mean
	Small_features = DNA_lengths < np.mean(DNA_lengths)/2
	Removing_small_features = Small_features[labeled_otsu_edges]
	labeled_otsu_edges[Removing_small_features] = 0
	small_cleaned_otsu_edges = np.multiply((labeled_otsu_edges>0),otsu_edges)

	# Remove large objects more than 2*mean
	Large_features = DNA_lengths > np.mean(DNA_lengths)*2
	Removing_large_features = Large_features[labeled_otsu_edges]
	labeled_otsu_edges[Removing_large_features] = 0
	cleaned_otsu_edges = np.multiply((labeled_otsu_edges>0),small_cleaned_otsu_edges)


	# Relabel cleaned image and recount the number of features post cleaning
	labelled_cleaned_otsu_edges, num_features_cleaned = measurements.label(cleaned_otsu_edges, structure = connectivitymap)
	print 'No features post cleaning:', num_features_cleaned
	DNA_lengths_cleaned = measurements.sum(mask, labelled_cleaned_otsu_edges, range(0, num_features+1))
	print 'Clean DNA lengths:', DNA_lengths_cleaned

	fig = plt.figure()
	fig.add_subplot(121)   #top right
	plt.pcolor(otsu_edges, cmap = 'binary')
	plt.title('Labelled Otsu')
	cbar = plt.colorbar()
	fig.add_subplot(122)   #top left
	plt.pcolor(labelled_cleaned_otsu_edges, cmap = 'binary')
	plt.title('Cleaned')
	cbar = plt.colorbar()
	plt.savefig(file_name + 'edges.png')
	plt.show()
	plt.close()

	return otsu_edges, labelled_cleaned_otsu_edges, DNA_lengths_cleaned, num_features_cleaned




if __name__ == '__main__':
	sequence_length = 339
	DNA_diameter = 20
	directory = './'
	directory_files = os.listdir(directory)

	plt.close()

	for file_name in directory_files:
		if file_name.endswith('.txt'):
			print 'Filename:'+file_name
			# Load data
			data = LoadGwydTextFileToArray(file_name, directory)
			# Determine the size of the image and units to obtain scaling factor 
			height, width, scale = GetHeightWidthScale(file_name, directory)
			# Scale data into Angstroms
			data = scale*data

			# Determine the pixel size and minimum feature size for removal of small objects in next step
			min_size, pixel_size, minicircle_length = DefineFeature(data, width, height, sequence_length, DNA_diameter, extra_info=False)
	
			# Otsu filter data to remove background
			otsu_eroded_mask, otsu_data, otsu_mask = GetOtsuMask(data, min_size, pixel_size)

			# Laplace transform the image masked by Otsu thresholding to obtain the edge traces of the molecules
			# Return an image of the edges of each feature, an image with aggregates and small features removed which is labelled by its iterative value, 
			# the lengths of the edge features remaining after cleaning and the number of edge features remaining after cleaning
			otsu_edges, labelled_cleaned_otsu_edges, DNA_lengths_cleaned, num_features_cleaned = Edgefinding(minicircle_length,pixel_size,otsu_data,otsu_mask)

print 'Job Finished!'