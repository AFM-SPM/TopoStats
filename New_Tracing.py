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

	otsu_edges = scipy.ndimage.filters.laplace(otsu_mask)
	otsu_edges = morphology.binary_closing(otsu_edges)

	mask = otsu_edges > 0

	labeled_otsu_edges, num_features = measurements.label(otsu_edges, structure = connectivitymap)
	print 'No features:', num_features

	DNA_lengths = measurements.sum(mask, labeled_otsu_edges, range(0, num_features+1))
	print DNA_lengths

	# Remove small objects less 1/2 median
	Small_features = DNA_lengths < np.median(DNA_lengths)/2
	Removing_small_features = Small_features[labeled_otsu_edges]
	labeled_otsu_edges[Removing_small_features] = 0
	small_cleaned_otsu_edges = np.multiply((labeled_otsu_edges>0),otsu_edges)

	# Remove large objects more than 2*median
	Large_features = DNA_lengths > np.median(DNA_lengths)*7
	Removing_large_features = Large_features[labeled_otsu_edges]
	labeled_otsu_edges[Removing_large_features] = 0
	cleaned_otsu_edges = np.multiply((labeled_otsu_edges>0),small_cleaned_otsu_edges)


	fig = plt.figure()
	fig.add_subplot(121)   #top right
	plt.pcolor(otsu_edges, cmap = 'binary')
	plt.title('Labelled Otsu')
	cbar = plt.colorbar()
	fig.add_subplot(122)   #top left
	plt.pcolor(cleaned_otsu_edges, cmap = 'binary')
	plt.title('Cleaned')
	cbar = plt.colorbar()
	plt.savefig(file_name + 'edges.png')
	plt.close()

	return otsu_edges


def ImageLabelling(minicircle_length,pixel_size,skele_otsu):
	connectivitymap = np.ones((3,3)) # Erosion shape
	trace_length = minicircle_length  # expected trace length
	print 'Expected trace length:', trace_length

	# Trace all objects in image
	mask = skele_otsu > 0
	labelled_DNA, number_of_circles = measurements.label(skele_otsu, structure = connectivitymap) #labelled DNA is plot of circles with each circle assigned the smae values based on its iterative number 
	DNA_lengths = measurements.sum(mask, labelled_DNA, range(0, number_of_circles+1))

	# Remove small objects less than 1/3 standard trace length
	mask_size = DNA_lengths < trace_length/3
	remove_mask = mask_size[labelled_DNA]
	labelled_DNA[remove_mask] = 0
	temp_map = np.multiply((labelled_DNA>0), skele_otsu)

	# Remove large objects more than 1.5x standard trace length
	mask_size = DNA_lengths > trace_length*1.5
	remove_mask = mask_size[labelled_DNA]
	labelled_DNA[remove_mask] = 0
	temp_map = np.multiply((labelled_DNA>0), skele_otsu)

	# Retrace clean data to get trace lengths
	labelled_skele_otsu, number_of_circles = measurements.label(temp_map, structure = connectivitymap)
	trace_lengths = measurements.sum(mask, labelled_skele_otsu, range(1, number_of_circles + 1))	#This could be improved using range(1,number_of_circles+1) I think, would get rid of checking at 0
	print trace_lengths
	print number_of_circles

	plt.figure()
	plt.pcolor(labelled_skele_otsu, cmap = 'binary')
	cbar = plt.colorbar()
	cbar.ax.set_ylabel('Height ($\AA$)', size = 'large')
	plt.xlabel('x-axis ($\AA$)')
	plt.ylabel('y-axis ($\AA$)')
	#plt.show()
	plt.savefig(file_name + 'lso.png')
	plt.close()

	return trace_lengths, labelled_skele_otsu, number_of_circles



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

			otsu_edges = Edgefinding(minicircle_length,pixel_size,otsu_data,otsu_mask)

			# # Skeletonise otsu filtered data (acts on mask)
			# skele_otsu = morphology.skeletonize(otsu_eroded_mask)

			# # Return the trace lengths, number of traces and an image with each feature labelled by its iternative value
			# trace_lengths, labelled_skele_otsu, number_of_circles = ImageLabelling(minicircle_length,pixel_size, skele_otsu)
			# labelled_skele_otsu

			# # Make a box 80 nm by 80 nm in which to plot the traces
			# square_size = int(800 / pixel_size)	#This makes an 80nm "box" with which to extract the trace
			# if square_size % 2 == 1:
			# 	square_size += 1
			# if square_size > 512:
			# 	square_size = 512
			# print 'Square size is ' + str(square_size) + ' pixels'
			# print 'There were ' + str(number_of_circles) + ' circles detected'
		
			# # Loop through all found traces to determine coordinates
			# count = 0	
			# circle_no = 1

			# 		# while circle_no < number_of_circles:	#should this be number_of_circles + 1?????
			# 		# 	# This if statement checks if the initial tracelength is listed as having a length of zero, which is common. If this is the case the '0' length circle is skipped and the next trace is found	
			# 		# 	if trace_lengths[(circle_no):(circle_no+1)] == 0:
			# 		# 			circle_no += 1
			# 	 	# 	print circle_no

		 # 	#find correct angle
			# tree = spatial.cKDTree(labelled_skele_otsu)

			# pixel_distance = int(40./sqrt(pixel_size))
			# print pixel_size
			# print pixel_distance
			# sigma = (20./sqrt(pixel_size))/1.5
			
			# data = filters.gaussian(data, sigma) #optional filtering step

			# for i in labelled_skele_otsu:
			# 	if i[0] < 0:
			# 		i[0] = pixel_distance
			# 	elif i[0] >= len(data[0]) - pixel_distance:
			# 		i[0] = i[0] = len(data[0]) - pixel_distance
			# 	elif i[1] < 0:
			# 		i[1] = (pixel_distance + 1)
			# 	elif i[1] >= len(data) - pixel_distance:
			# 		i[1] = len(data) - pixel_distance	

			# 	height_values = None
			# 	neighbour_array = tree.query(i, k = 6)
			# 	nearest_point = data[neighbour_array[1][3]]
			# 	vector = np.subtract(nearest_point, i)
			# 	vector_angle = degrees(atan2(vector[1],vector[0]))	#Angle with respect to x-axis

			# 	if vector_angle < 0:
			# 		vector_angle += 180
		
			# 	if 67.5 > vector_angle >= 22.5:	#aka if  angle is closest to 45 degrees
			# 		perp_direction = 'negative diagonal'
			# 		#positive diagonal (change in x and y)		
			# 		#Take height values at the inverse of the positive diagonal (i.e. the negative diagonal)			
			# 		y_coords = np.arange(i[1] - pixel_distance, i[1] + pixel_distance)[::-1]
			# 		x_coords = np.arange(i[0] - pixel_distance, i[0] + pixel_distance)
					
			# 	elif 157.5 >= vector_angle >= 112.5:#aka if angle is closest to 135 degrees
			# 		perp_direction = 'positive diagonal'
			# 		y_coords = np.arange(i[1] - pixel_distance, i[1] + pixel_distance)
			# 		x_coords = np.arange(i[0] - pixel_distance, i[0] + pixel_distance)

			# 	if 112.5 > vector_angle >= 67.5: #if angle is closest to 90 degrees
			# 		perp_direction = 'horizontal'
			# 		x_coords = np.arange(i[0] - pixel_distance, i[0]+pixel_distance)
			# 		y_coords = np.full(len(x_coords), i[1])
				
			# 	elif 22.5 > vector_angle: #if angle is closest to 0 degrees
			# 		perp_direction = 'vertical'
			# 		y_coords = np.arange(i[1] - pixel_distance, i[1] + pixel_distance)
			# 		x_coords = np.full(len(y_coords), i[0])
				
			# 	elif vector_angle >= 157.5: #if angle is closest to 180 degrees
			# 		perp_direction = 'vertical'
			# 		y_coords = np.arange(i[1] - pixel_distance, i[1] + pixel_distance)
			# 		x_coords = np.full(len(y_coords), i[0])
				
			# 	perp_array = np.column_stack((x_coords, y_coords))

			# 	for j in perp_array:
			# 		height = labelled_skele_otsu[j[0], j[1]]
			# 		if height_values is None:
			# 			height_values = height
			# 		else:
			# 			height_values = np.vstack((height_values, height))
						
			# 	if perp_direction == 'negative diagonal':
			# 		int_func = interp.interp1d(perp_array[:,0], np.ndarray.flatten(height_values), kind = 'cubic')
			# 		interp_heights = int_func(np.arange(perp_array[0,0], perp_array[-1,0], 0.1))
				
			# 	elif perp_direction == 'positive diagonal':
			# 		int_func = interp.interp1d(perp_array[:,0], np.ndarray.flatten(height_values), kind = 'cubic')
			# 		interp_heights = int_func(np.arange(perp_array[0,0], perp_array[-1,0], 0.1))
				
			# 	elif perp_direction == 'vertical':
			# 		int_func = interp.interp1d(perp_array[:,1], np.ndarray.flatten(height_values), kind = 'cubic')
			# 		interp_heights = int_func(np.arange(perp_array[0,1], perp_array[-1,1], 0.1))
				
			# 	elif perp_direction == 'horizontal':
			# 		int_func = interp.interp1d(perp_array[:,0], np.ndarray.flatten(height_values), kind = 'cubic')
			# 		interp_heights = int_func(np.arange(perp_array[0,0], perp_array[-1,0], 0.1))
			# 	else:
			# 		quit('A fatal error occured in the CorrectHeightPositions function, this was likely caused by miscalculating vector angles')
				
			# 	if perp_direction == 'negative diagonal':
			# 		fine_x_coords = np.arange(perp_array[0,0], perp_array[-1,0], 0.1)
			# 		fine_y_coords = np.arange(perp_array[-1,1], perp_array[0,1], 0.1)[::-1]
			# 	elif perp_direction == 'positive diagonal':
			# 		fine_x_coords = np.arange(perp_array[0,0], perp_array[-1,0], 0.1)
			# 		fine_y_coords = np.arange(perp_array[0,1], perp_array[-1,1], 0.1)
			# 	elif perp_direction == 'vertical':
			# 		fine_y_coords = np.arange(perp_array[0,1], perp_array[-1,1], 0.1)
			# 		fine_x_coords = np.full(len(fine_y_coords), i[0], dtype = 'float')
			# 	elif perp_direction == 'horizontal':
			# 		fine_x_coords = np.arange(perp_array[0,0], perp_array[-1,0], 0.1)
			# 		fine_y_coords = np.full(len(fine_x_coords), i[1], dtype = 'float')			
				
			# 	fine_coords = np.column_stack((fine_x_coords, fine_y_coords))

			# 	# plt.figure()
			# 	# plt.pcolor(fine_coords, cmap = 'binary')
			# 	# cbar = plt.colorbar()
			# 	# cbar.ax.set_ylabel('Height ($\AA$)', size = 'large')
			# 	# plt.xlabel('x-axis ($\AA$)')
			# 	# plt.ylabel('y-axis ($\AA$)')
			# 	# plt.savefig('fine.png')
			# 	# plt.close()

			# 	sorted_array = fine_coords[np.argsort(interp_heights)]
			# 	highest_point = sorted_array[-1]

			# 	try:
			# 		fitted_coordinate_array = np.vstack((fitted_coordinate_array, highest_point))
			# 	except NameError:
			# 		fitted_coordinate_array = highest_point


print 'Job Finished!'