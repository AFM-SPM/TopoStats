"""
Code written by Joseph Beton under the supervision of Maya Topf and Bart Hoogenboom. 
Special thanks to Agnel Joseph and Alice Pyne for their contributions.
This work is an extension and adaptation of algorithms written by both Luzie Helfmann and Robert Gray.

This algorithm is written to analyse AFM images of biomolecules and was optimised using 
data from DNA imaging, where it is particularly successful. The algorithm uses output files
from its sister programme 'HeighttoXYZ.py', which is required to run this code.

The algorithm is designed to find traces for biomolecules and polymers imaged using AFM.
Broadly, the algorithm can be divided in to these key stages, with the relevant functions labelled:
	
	-Loading Data:
		LoadTextFile
		ImageUnpacking
	-Background Removal Cascade:
		CannyEdgeFinder
		SkeletonizeLoops
	-Coordinate Discovery and Trace Computation:
		ImageLabelling
		TraceDetection
		Coordinatetransform
		FindOrderedTrace
		FindNextPoint
		SplineOrderedPoints
	-Calculate Values:
		FindContourLength
		FindRadiusOfCurvature
		RadiusOfGyration
	-Saving Information
		SaveTraceCoordinates
		SaveContourLengths
		SaveMinicircleArea
		
Within the body of the code, descriptions of how each of the functions execute as well as
information on pre-written algorithms called in the code is included.

The code accesses all .txt files in a directory specified within the code as the object 'directory'. For saving files, new folders will be made within the directory containing the original files
for saving the 
"""
import numpy as np
from math import sqrt,acos, degrees, atan2, pi, cos, sin
from scipy import spatial, interpolate as interp
from scipy.ndimage import morphology as morph, measurements, fourier, interpolation as imginterp
from skimage import feature, filters, morphology, segmentation, measure, transform
import matplotlib.pyplot as plt
import os

connectivitymap = np.ones((3,3))
sequence_length = 339
DNA_diameter = 20

"""
The functions 'LoadTextFile' and 'ImageUnpacking' work in concert to generate the Numpy
arrays that represent the AFM image being analysed. Loadtxtfile uses the Numpy function
'np.loadtxtfile' to create a large array containing all x,y and z coordinates for the AFM
image, termed variable_name. Following this, within ImageUnpacking, the large array, now 
termed scandata, is segmented in to individual Numpy arrays that contain the x,y and z 
coordinates individually, known as xscan, yscan and zscan respectively. 

Following this, if the user required the images of noise removal, these are saved in to a 
folder within the directory containing the original input files. This is achieved using the
python native 'os' module, which first checks for the existence of a folder named 'AFM 
Images' within the directory, creating this folder if it does not find this. Once the folder
has been created, the original AFM image and a histogram which plots the occurrence of 
height values in the image.
"""
def LoadTextFile(file_name, directory):
	with open(directory + file_name , "r") as file:
		opened_file = np.loadtxt(file)
	return opened_file

def ImageUnpacking(scandata, directory, file_name):
	if len(np.shape(scandata))==2:
		scandata = np.reshape(scandata, (sqrt(len(scandata)),sqrt(len(scandata)),3))
	xscan = np.array([[i[j,0] for j in range(len(i))] for i in scandata])
	yscan = np.array([[i[j,1] for j in range(len(i))]for i in scandata])
	zscan = np.array([[i[j,2] for j in range(len(i))] for i in scandata])

	if not os.path.exists(directory + 'AFM Images'):
		os.makedirs(directory + 'AFM Images')
		image_direct = directory + 'AFM Images/'
	else:
		image_direct = directory + 'AFM Images/'

	if plot_display == 'y':
		plt.figure()
		plt.pcolor(xscan, yscan, zscan, cmap = 'binary')
		cbar = plt.colorbar()
		cbar.ax.set_ylabel('Height ($\AA$)', size = 'large')
		plt.xlabel('x-axis ($\AA$)')
		plt.ylabel('y-axis ($\AA$)')
		#plt.show()
		plt.savefig(image_direct + file_name + 'AFMdata.png')
		plt.close()
	"""
	if plot_display == 'y':
		plt.figure()
		n, bins, patches = plt.hist(zscan, 50, normed = 1, facecolor = 'g', alpha = 0.75)
		plt.xlabel('Height ($\AA$)')
		plt.ylabel('Frequency')
		plt.savefig(image_direct + file_name + 'Hist.png')
		plt.close()
	"""
	return xscan,yscan,zscan

"""
The CannyEdgeFinder function uses the canny algorithm from the Scikit Image library (link here http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.canny)
to define the edges of the molecules in the AFM image. The canny algorithm works on the basis of initially defining the intensity gradients in the x and y directions for the image.
The intensity gradients are summed together and the norm is found, referred to as edge strength. The edge direction is then also found using inverse tangent of intensity gradients.
Following this, a single pixel wide representation of the edge is defined by finding the local maxima, with the algorithm finishing by defining edges using a high and low thresholding.

For use in this algorithm, a wide (2nm) Gaussian filter is applied to the image in order to minimise intramolecular detection. This is done using the sigma parameter within the canny
function from the Scikit Image library. Care needs to be taken if substituting or updating the function used here as recent developments in the canny algorithm aimed at increasing 
accuracy use a weighted Gaussian filter, which is not appropriate for reducing intramolecular detection.

"""

def CannyEdgeFinder(zscan, pixel_size):
	median = np.median(zscan)
	mask = zscan > median
	sigma = (20/sqrt(pixel_size))/1.5
	gaussian_scan = filters.gaussian_filter(zscan, sigma)
	cannyzscan = feature.canny(gaussian_scan, sigma = sigma)
	return cannyzscan, gaussian_scan

"""
The SkeletonizeLoops function does both the background removal and the skeletonisation of the foreground image. 

Background removal is started by thresholding the image at the median intensity value, making the image referred to as thresh_zscan. Here, all pixels with an intensity below the median are 
set to zero. To avoid any issues with intramolecular details affecting image processing a gaussian version of the image is used for downstream image processing. 

Further background removal is achieved through 2 functions for pixel cluster removal, binary_opening and remove_small_objects. binary_opening is a function from the Scipy ndimage library,
this function removes small clusters of dark spots within an image based on the connectivitymap, defined on line 49 of this code. The reference page for morph.binary_opening can be found here
http://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.ndimage.morphology.binary_opening.html.

The remove_small_objects function, as its name suggests, removes clusters of pixels in an image depending on their size. Here, the minimum size for pixel removal is derived as an approximation
of DNA area, which is calculated within the main function. This function is important for removing small clusters of pixels that occur in the image as a result of anomalous adsorption in the
AFM image. 

After the first application of binary_opening and remove_small_objects, the canny image defined in the function CannyEdgeFinder is used to generate pixel wide seperation between the foreground
and background. This is followed by an addition iteration of binary_opening and remove_small_objects, which clears the remaining background from the image.

For images with a low pixel size, more harsh use of the binary_opening function is required. To this end, the number of iterations used each time the function is called is defined, somewhat
arbitrarily, at the start of the function based on the average number of pixels that makes up a DNA molecule.

Once a forground only image has been generated, the skeletonisation function, from the Scikit Image library, generates a pixel wide backbone of each distinct pixel cluster in the image. 
The skeletonisation algorithm uses the protocol initially described by Zhang, T.Y. and Suen, C.Y. in their 1984 report 'A Fast Parallel Algorithm for Thinning Digital Patterns'. In brief, the
algorithm assess each pixel of a binary image using a 3x3 kernel which checks for local environment based on connectivity and the number of non-zero pixels in the kernel. The algorithm divides
each iteration of pixel removal in to two sub-iterations where north and west connectivity is initially checked followed by south and east. The output is a binary image showing only the backbones
of the molecules.

"""

def SkeletonizeLoops(zscan, pixel_size, circle_pixels, gaussian_scan, cannyzscan):
	sigma = (20/sqrt(pixel_size))/1.5
	
	if circle_pixels > 3000:
		iterations = 4
	if circle_pixels > 2000:
		iterations = 3
	elif circle_pixels > 1000:
		iterations = 2
	else:
		iterations = 1

	"""Thresholding the scan at the median value, using a simple Numpy multiplication
	and removing residual noise via the	binary_opening function"""
	fresh_zscan = zscan > np.median(zscan)
	thresh_zscan = np.multiply(gaussian_scan, fresh_zscan)
	eroded_zscan = np.multiply((morph.binary_opening(thresh_zscan, structure = connectivitymap, iterations = iterations)), thresh_zscan)
	
	"""Masking the miniciricles and trying to remove as much of everything else as possible"""
	mask = eroded_zscan > 0
	morphology.remove_small_objects(mask, min_size = circle_pixels*2, in_place = True)
	canny_eroded_mask = np.multiply(np.invert(cannyzscan), mask)
	eroded_zscanmasked = np.multiply(eroded_zscan, canny_eroded_mask)	#This is not a masked (binary) image - it contains height values
	eroded_zscanclean = np.multiply(eroded_zscanmasked, morph.binary_opening(eroded_zscanmasked, iterations = iterations))

	fresh_scan_2 = eroded_zscanclean > np.median(eroded_zscan)		#This is pointless probably, could delete these 2 lines
	eroded_zscanclean = np.multiply(gaussian_scan, fresh_scan_2)	#

	guass_clean = filters.gaussian_filter(eroded_zscanclean, sigma/5)	#This is to close small 1-5 pixel holes in the DNA caused by erroneous edge detection
	inverted_canny = np.invert(feature.canny(guass_clean))
	if circle_pixels  > 2000:
		print 'Extra step of denoiseing'
		inverted_canny = morph.binary_opening(inverted_canny, iterations = iterations)
	Cannyeroded_zscan = np.multiply(inverted_canny, eroded_zscanclean)

	"""Masking for a second time in order to remove any left over noise"""
	if circle_pixels > 2000:
		mask2 = morph.binary_opening(Cannyeroded_zscan, iterations = iterations)
	else:
		mask2 = Cannyeroded_zscan>0

	morphology.remove_small_objects(mask2, min_size = circle_pixels/2, in_place = True)
	masked_image2 = gaussian_scan*mask2
	if circle_pixels > 2000:
		final_image = masked_image2*morph.binary_opening(masked_image2, iterations = iterations)
	else:
		final_image = masked_image2
	
	mask3 = filters.gaussian_filter(final_image, sigma/10) > 0

	skele_zscan = morphology.skeletonize(mask3)

	return skele_zscan, mask3, inverted_canny, eroded_zscanclean
"""

The function ImageLabelling uses the 

"""
def ImageLabelling(pixel_size, skele_zscan):
	mask = skele_zscan > 0
	labelled_DNA, number_of_circles = measurements.label(skele_zscan, structure = connectivitymap)
	DNA_lengths = measurements.sum(mask, labelled_DNA, range(number_of_circles + 1))
	
	trace_length = (33.2*(sequence_length/10.5)) / sqrt(pixel_size)
	
	mask_size = DNA_lengths < trace_length/3
	remove_mask = mask_size[labelled_DNA]
	labelled_DNA[remove_mask] = 0
	temp_map = np.multiply((labelled_DNA>0), skele_zscan)
	
	labelled_temp_map, number_of_circles = measurements.label(temp_map, structure = connectivitymap)
	trace_lengths = measurements.sum(mask, labelled_temp_map, range(number_of_circles + 1))
	mask_size = DNA_lengths > trace_length*5
	remove_mask = mask_size[labelled_temp_map]
	labelled_temp_map[remove_mask] = 0
	trace_map = np.multiply((labelled_temp_map>0), temp_map)
	
	mask = trace_map > 0
	labelled_trace_map, number_of_circles = measurements.label(trace_map, structure = connectivitymap)
	trace_lengths = measurements.sum(mask, labelled_trace_map, range(number_of_circles + 1))	#This could be improved using range(1,number_of_circles+1) I think, would get rid of checking at 0

	return trace_lengths, labelled_trace_map, number_of_circles
	
def CorrectHeightPositions(DNA_map, coordinates, pixel_size, xscan, yscan):
	#find correct angle
	tree = spatial.cKDTree(coordinates)
	pixel_distance = int(40/sqrt(pixel_size))
	
	sigma = (20/sqrt(pixel_size))/1.5
	
	DNA_map = filters.gaussian_filter(DNA_map, sigma)
	#plt.figure()
	for i in coordinates:
		if i[0] < 0:
			i[0] = pixel_distance
		elif i[0] >= len(xscan)-pixel_distance:
			i[0] = i[0] = len(xscan)-pixel_distance
		elif i[1] < 0:
			i[1] = (pixel_distance+1)
		elif i[1] >= len(yscan)-pixel_distance:
			i[1] = len(yscan) - pixel_distance
		
		height_values = None
		neighbour_array = tree.query(i, k = 6)
		nearest_point = coordinates[neighbour_array[1][3]]
		vector = np.subtract(nearest_point, i)
		vector_angle = degrees(atan2(vector[1],vector[0]))	#Angle with respect to x-axis
		
		if vector_angle < 0:
			vector_angle += 180
		
		if 67.5 > vector_angle >= 22.5:	#aka if  angle is closest to 45 degrees
			perp_direction = 'negative diaganol'
			#positive diagonal (change in x and y)		
			#Take height values at the inverse of the positive diaganol (i.e. the negative diaganol)			
			y_coords = np.arange(i[1] - pixel_distance, i[1] + pixel_distance)[::-1]
			x_coords = np.arange(i[0] - pixel_distance, i[0] + pixel_distance)
			
		elif 157.5 >= vector_angle >= 112.5:#aka if angle is closest to 135 degrees
			perp_direction = 'positive diaganol'
			y_coords = np.arange(i[1] - pixel_distance, i[1] + pixel_distance)
			x_coords = np.arange(i[0] - pixel_distance, i[0] + pixel_distance)

		if 112.5 > vector_angle >= 67.5: #if angle is closest to 90 degrees
			perp_direction = 'horizontal'
			x_coords = np.arange(i[0] - pixel_distance, i[0]+pixel_distance)
			y_coords = np.full(len(x_coords), i[1])
		
		elif 22.5 > vector_angle: #if angle is closest to 0 degrees
			perp_direction = 'vertical'
			y_coords = np.arange(i[1] - pixel_distance, i[1] + pixel_distance)
			x_coords = np.full(len(y_coords), i[0])
		
		elif vector_angle >= 157.5: #if angle is closest to 180 degrees
			perp_direction = 'vertical'
			y_coords = np.arange(i[1] - pixel_distance, i[1] + pixel_distance)
			x_coords = np.full(len(y_coords), i[0])
		
		perp_array = np.column_stack((x_coords, y_coords))
		
		#plt.plot(perp_array[:,1], perp_array[:,0])
		#plt.draw()
		for j in perp_array:
			height = DNA_map[j[0], j[1]]
			if height_values == None:
				height_values = height
			else:
				height_values = np.vstack((height_values, height))
		
		if perp_direction == 'negative diaganol':
			int_func = interp.interp1d(perp_array[:,0], np.ndarray.flatten(height_values), kind = 'cubic')
			interp_heights = int_func(np.arange(perp_array[0,0], perp_array[-1,0], 0.1))
		
		elif perp_direction == 'positive diaganol':
			int_func = interp.interp1d(perp_array[:,0], np.ndarray.flatten(height_values), kind = 'cubic')
			interp_heights = int_func(np.arange(perp_array[0,0], perp_array[-1,0], 0.1))
		
		elif perp_direction == 'vertical':
			int_func = interp.interp1d(perp_array[:,1], np.ndarray.flatten(height_values), kind = 'cubic')
			interp_heights = int_func(np.arange(perp_array[0,1], perp_array[-1,1], 0.1))
		
		elif perp_direction == 'horizontal':
			int_func = interp.interp1d(perp_array[:,0], np.ndarray.flatten(height_values), kind = 'cubic')
			interp_heights = int_func(np.arange(perp_array[0,0], perp_array[-1,0], 0.1))
		else:
			quit('A fatal error occured in the CorrectHeightPositions function, this was likely caused by miscalculating vector angles')
		
		if perp_direction == 'negative diaganol':
			fine_x_coords = np.arange(perp_array[0,0], perp_array[-1,0], 0.1)
			fine_y_coords = np.arange(perp_array[-1,1], perp_array[0,1], 0.1)[::-1]
		elif perp_direction == 'positive diaganol':
			fine_x_coords = np.arange(perp_array[0,0], perp_array[-1,0], 0.1)
			fine_y_coords = np.arange(perp_array[0,1], perp_array[-1,1], 0.1)
		elif perp_direction == 'vertical':
			fine_y_coords = np.arange(perp_array[0,1], perp_array[-1,1], 0.1)
			fine_x_coords = np.full(len(fine_y_coords), i[0], dtype = 'float')
		elif perp_direction == 'horizontal':
			fine_x_coords = np.arange(perp_array[0,0], perp_array[-1,0], 0.1)
			fine_y_coords = np.full(len(fine_x_coords), i[1], dtype = 'float')
			
		
		fine_coords = np.column_stack((fine_x_coords, fine_y_coords))
		
		sorted_array = fine_coords[np.argsort(interp_heights)]
		highest_point = sorted_array[-1]
		try:
			fitted_coordinate_array = np.vstack((fitted_coordinate_array, highest_point))
		except UnboundLocalError:
			fitted_coordinate_array = highest_point
	#plt.pcolor(xlimit, ylimit, DNA_map, cmap = 'binary')
	#plt.plot(coordinates[:,1], coordinates[:,0], '.')
	#plt.plot(fitted_coordinate_array[:,1], fitted_coordinate_array[:,0], '.')
	#plt.show()
	
	return fitted_coordinate_array

"""

"""
def SaveTraceCoordinates(ordered_points, file_counter, file_name, pixel_size):
	ordered_points = np.multiply(ordered_points, [sqrt(pixel_size),sqrt(pixel_size)])
	
	if not os.path.exists(directory + 'TraceCoordinates'):
		os.makedirs(directory + 'TraceCoordinates')
		write_coords = directory + 'TraceCoordinates/'
	else:
		write_coords = directory + 'TraceCoordinates/'
	try:
		write_file = open(write_coords + 'ordered_coordinates' + file_name + str(file_counter) + '.txt', 'w')
		#print >>write_file, '#Contour Length:' + str(contour_length)
		for i in ordered_points:
			print >>write_file, str(i[0]) + '\t' + str(i[1])
	except TypeError:
		pass

"""


"""

def SaveMinicircleArea(xmid, ymid, file_counter, splined_coords, square_size, DNA_circle, loop, file_name):
	if not os.path.exists(directory + 'TraceImages/'):
		os.makedirs(directory + 'TraceImages/')
		trace_directory = directory + 'TraceImages/'
	else:
		trace_directory = directory + 'TraceImages/'
	
	xlimit = np.arange((xmid-square_size/2),(xmid+square_size/2))
	ylimit = np.arange((ymid-square_size/2),(ymid+square_size/2))
	
	file_counter +=1
	plt.figure()
	plt.pcolor(ylimit, xlimit, DNA_circle, cmap = 'binary')
	cbar = plt.colorbar()
	cbar.ax.set_ylabel('Height ($\AA$)', size = 'large')
	if splined_coords is not None:
		plt.plot(splined_coords[:,1],splined_coords[:,0], 'r')
	#plt.plot(splined_coords[0,1], splined_coords[0,0], 'ro')
	#plt.plot(splined_coords[10,1], splined_coords[10,0], '^')
	#plt.plot(centroid[1], centroid[0], 'ro')
	#if loop:
	#	plt.title('Looped DNA Minicircle')
	#else:
	#	plt.title('Non-looped DNA Minicircle')
	plt.savefig(trace_directory + file_name +' minicircle ' + str(file_counter) + '.png', bbox_inches='tight')
	plt.close()			

"""

"""

def TraceDetection(trace_map, circle_no, xscan, yscan, zscan, pixel_size, square_size):
	found_trace = False

	single_trace  = trace_map == circle_no

	label = np.multiply(trace_map,single_trace)	#might be pointless, could just use single_trace
	trace_coords = np.nonzero(label)

	sorted_xcoords = np.sort(np.array(trace_coords[0]))
	sorted_ycoords = np.sort(np.array(trace_coords[1]))
	
	xmin, xmax = sorted_xcoords[0], sorted_xcoords[-1]
	ymin, ymax = sorted_ycoords[0], sorted_ycoords[-1]
	
	if xmax-xmin < square_size and ymax-ymin < square_size:
		found_trace = True
		if xmax < square_size:
			xmid = square_size/2
		elif xmax > (len(xscan)-square_size/2):
			xmid = len(xscan)-square_size/2
		else:
			xmid = xmax - ((xmax-xmin)/2)
		if ymax < square_size:
			ymid = square_size/2
		elif ymax > (len(yscan)-square_size/2):
			ymid = len(xscan)-square_size/2
		else:
			ymid = ymax - ((ymax-ymin)/2)
		
		trace_area = label[xmid-(square_size/2):xmid+(square_size/2), ymid-(square_size/2):ymid+(square_size/2)]
		DNA_circle = zscan[xmid-(square_size/2):xmid+(square_size/2), ymid-(square_size/2):ymid+(square_size/2)]

		xlimit = np.arange((xmid-square_size/2),(xmid+square_size/2))	#I think for plotting data, no longer in use
		ylimit = np.arange((ymid-square_size/2),(ymid+square_size/2))

	else:		#This code is now essentially not in use, can ignore
		found_trace = False
		if (xmax - xmin) > (ymax-ymin):
			square_border = xmax - xmin + 20
			if ymin + square_border > 512:
				ymin = 512 - square_border
			elif xmin + square_border > 512:
				xmin = 512 - square_border
		else:
			square_border = ymax-ymin + 20
			if xmin + square_border > 512:
				xmin = 512 - square_border
			elif ymin + square_border > 512:
				ymin = 512 - square_border
		xmid, ymid = 0,0
		trace_area = label[xmin:xmin + square_border, ymin:ymin+square_border]
		coordinates_array = None
		DNA_circle = zscan[xmid-square_size/2:xmid+square_size/2, ymid-square_size/2:ymid+square_size/2]
		xlimit = np.arange((xmid-square_size/2),(xmid+square_size/2))
		ylimit = np.arange((ymid-square_size/2),(ymid+square_size/2))

	trace_coords2d = np.column_stack((trace_coords[0], trace_coords[1]))

	expected_CL = (33.2*(sequence_length/10.5))
	
	if len(trace_coords2d) < (expected_CL/sqrt(pixel_size))/sqrt(2): #What is the purpose of this?
		found_trace = False

	#Possible edit: End this function here and call subsequent functions down in main to improve readibility and clarity.
	if found_trace == True:
		new_coordinates = CorrectHeightPositions(zscan, trace_coords2d, pixel_size, xscan, yscan)
	else:
		new_coordinates = trace_coords2d
	
	splined_coords, contour_length = FindOrderedTrace(trace_area, new_coordinates, xmid, ymid, square_size, pixel_size)
	#splined_fit_coords, contour_length = FindOrderedTrace(trace_area, new_coordinates, xmid, ymid, square_size, pixel_size)	
	"""
	print len(trace_coords2d), len(new_coordinates)
	plt.pcolor(xlimit, ylimit, DNA_circle, cmap = 'binary')
	plt.plot(splined_coords[:,1], splined_coords[:,0], 'r')
	plt.plot(splined_fit_coords[:,1], splined_fit_coords[:,0], 'b')
	cbar = plt.colorbar()
	cbar.ax.set_ylabel('Height ($\AA$)', size = 'large')
	plt.xlabel('x-axis ($\AA$)')
	plt.ylabel('y-axis ($\AA$)')
	plt.savefig('hello.png')
	plt.close()
	"""
	try:
		curvature = FindRadiusOfCurvature(splined_coords)
		centroid = RadiusOfGyration(splined_coords, curvature)
		loop = ChamferDistance(splined_coords, centroid, pixel_size)
		if np.amax(curvature) > 5:
			found_trace = False
	except:
		curvature = None
		loop = False
		centroid = None

	if contour_length > expected_CL*2:
		found_trace = False
	
	return found_trace, trace_area, DNA_circle, splined_coords, xmid,ymid, contour_length, loop, curvature

def FindNextPoint(tree, next_point, query, ordered_points, vector_angle_array, trace_coords, vector2, average_angle2, last_point):
	points_tree = spatial.cKDTree(ordered_points)
	end = False
	scale_factor = 40
	for j in range(len(query[1])):
		next_point = trace_coords[query[1][j]]
		vector1 = np.subtract(next_point, last_point)
		
		angle = atan2(vector2[1], vector2[0]) - atan2(vector1[1], vector1[0])
		if angle < 0:
			angle += 2*pi
		angle = degrees(angle)
		if angle > 180:
			angle -= 180
		points_query = points_tree.query(next_point, k = 1)

		last_vectors = np.append(vector_angle_array[-19:], angle)
		average_angle1 = np.mean(last_vectors)

		if points_query[0] == 0:
			continue
		elif len(ordered_points) < 10:
			break
		elif average_angle2 - average_angle1 == 0:
			break
		elif average_angle2 - average_angle1 > scale_factor:
			continue
		elif np.all(ordered_points[0] == next_point):
			print 'end'
			end = True
			break
		else:
			for k in range(1,5,1):
				try:
					test_point = trace_coords[query[1][j+k]]
					
					points_query = points_tree.query(test_point, k = 1)
					
					if points_query[0] == 0:
						continue
					elif query[0][j+k] > query[0][j]*sqrt(2):
						continue
						
					test_vector = np.subtract(test_point, last_point)
					test_angle = atan2(vector2[1], vector2[0]) - atan2(test_vector[1], test_vector[0])
					if test_angle < 0:
						test_angle += 2*pi
					test_angle = degrees(test_angle)
					if test_angle > 180:
						test_angle -= 180
					
					local_angles = np.mean(vector_angle_array[-20:])
					local_average_angle = np.mean(np.append(vector_angle_array[-19:], angle))
					test_average_angle = np.mean(np.append(vector_angle_array[-19:], test_angle))
					
					if local_angles - local_average_angle > local_angles - test_average_angle:
						next_point = test_point
						angle = test_angle
						break
					if local_angles - local_average_angle == local_angles - test_average_angle:
						test_vector = test_point - last_point
											
						old_angle = atan2(vector2[1], vector2[0])
						new_angle = atan2(vector1[1], vector1[0])
						test_angle = atan2(test_vector[1], test_vector[0])
						
						if old_angle - test_angle > old_angle - new_angle:
							next_point = test_point
						break
				except IndexError:
					break
			break
	average_angle2 = average_angle1
	last_point = next_point
	vector_angle_array = np.append(vector_angle_array, angle)
	ordered_points = np.vstack((ordered_points, last_point))
	
	return ordered_points, vector_angle_array, vector2, average_angle2, end, last_point
	
def FindOrderedTrace(trace_area, trace_coords2d, xmid, ymid, square_size, pixel_size):
	trace_coords = trace_coords2d

	first_point = np.array((trace_coords[0]))
	tree = spatial.cKDTree(trace_coords)
	ordered_points = np.empty((0,1))
	vector_angle_array = np.empty((0))

	query = tree.query(first_point, k=2)
	next_point = trace_coords[query[1][1]]
	last_point = next_point

	vector2 = np.subtract(next_point, last_point)
	third_point = first_point
	ordered_points = np.vstack((first_point, next_point))
	average_angle2 = 0


	contour_length = 0
	for i in range(len(trace_coords)-2):
		query = tree.query(last_point, k = 30)
		ordered_points, vector_angle_array, vector2, average_angle2, end, last_point = FindNextPoint(tree, next_point, query, ordered_points, vector_angle_array, trace_coords, vector2, average_angle2, last_point)
		if end == True:
			break
	step_size = int((sequence_length/8)/sqrt(pixel_size))
	
	splined_coords, spline_success = SplineOrderedPoints(ordered_points, step_size)
	contour_length = FindContourLength(spline_success, splined_coords, ordered_points, pixel_size)
	
	if spline_success:
		crossing = FindCrossing(splined_coords, step_size)
	if not spline_success:
		first_point = np.array((trace_coords[-1]))
		tree = spatial.cKDTree(trace_coords)
		ordered_points = np.empty((0,1))
		vector_angle_array = np.empty((0))
		loop = False
		
		query = tree.query(first_point, k=2)
		next_point = trace_coords[query[1][1]]
		last_point = next_point
	
		vector2 = np.subtract(next_point, last_point)
		third_point = first_point
		ordered_points = np.vstack((first_point, next_point))
		average_angle2 = 0
	
		contour_length = 0
	
		for i in range(len(trace_coords)-2):
			query = tree.query(last_point, k = 30)
			ordered_points, vector_angle_array, vector2, average_angle2, end, last_point = FindNextPoint(tree, next_point, query, ordered_points, vector_angle_array, trace_coords, vector2, average_angle2, last_point)
			if end == True:
				break
		splined_coords, spline_success = SplineOrderedPoints(ordered_points, step_size)
		contour_length = FindContourLength(spline_success, splined_coords, ordered_points, pixel_size)
	return splined_coords, contour_length

def CoordinateTransform(splined_coords):
	
	for i in range(len(splined_coords)):
		vector = i+1 - i
		vector_norm = np.linalg.norm(vector)

def FindCrossing(splined_coords, step_size):
	crossing = False
	
	splined_coords = splined_coords[0:len(splined_coords)-10]

	#use interpolation?

	for i in range(len(splined_coords)):
		for j in range(len(splined_coords)):
			if i == j:
				continue
			elif np.allclose(splined_coords[i],splined_coords[j], rtol = 5e-3, atol = 0.5e-4):
				crossing = True
	
	return crossing

def RadiusOfGyration(splined_coords, curvature):
	centroid = np.array((np.mean(splined_coords[:,0]),np.mean(splined_coords[:,1])))
	radius_gyration = np.empty((0,2))
	
	for i in range(len(splined_coords)):
		centroid_vector = np.linalg.norm(centroid - splined_coords[i])
		local_curvature = curvature[i]
		local_gyration = np.array((centroid_vector, local_curvature))
		radius_gyration = np.vstack((radius_gyration, local_gyration))

	x = radius_gyration[:,0]
	radius_gyration[np.argsort(x)]
	segment_size = int(len(radius_gyration[:,0])/15)
	
	return centroid

def SplineOrderedPoints(ordered_points, step_size):
	count = 0

	for i in range(step_size):
		try:
			nbr = len(ordered_points[:,0])
			x = [ordered_points[:,0][j] for j in range(i,nbr,step_size)]
			y = [ordered_points[:,1][j] for j in range(i,nbr,step_size)]
			tck,u = interp.splprep([x,y], s=0, per=1)
			out = interp.splev(np.linspace(0,1,nbr), tck)
			splined_coords = np.column_stack((out[0], out[1]))
			try:
				rolling_total = np.add(rolling_total, splined_coords)
			except UnboundLocalError:
				rolling_total = splined_coords
			spline_success = True
			count +=1
		
		except SystemError:
			print 'Could not spline coordinates'
			spline_success = False
			splined_coords = None
			continue
		except TypeError:
			print 'The trace is too short or something'
			spline_success = False
			splined_coords = None
	if spline_success:
		rolling_average = np.divide(rolling_total, [count, count])

		nbr = len(rolling_average[:,0])
		x = rolling_average[:,0]
		y = rolling_average[:,1]
		tck,u = interp.splprep([x,y], s=0, per=1)
		out = interp.splev(np.linspace(0,1,nbr), tck)

		splined_coords = np.column_stack((out[0], out[1]))
	else:
		spline_coords = None
	
	return splined_coords, spline_success

def FindContourLength(spline_success, splined_coords, ordered_points, pixel_size):
	contour_length = 0
	
	if spline_success:
		for i in splined_coords:
			vector = i+1 - i
			vector_magnitude = np.linalg.norm(vector)
			contour_length = contour_length + vector_magnitude
	else:
		for i in ordered_points:
			vector = i - i+1
			vector_magnitude = np.linalg.norm(vector)
			contour_length = contour_length + vector_magnitude 
	
	contour_length = contour_length*sqrt(pixel_size)

	return contour_length

def SaveContourLengths(contour_length_array, file_name):
	if not os.path.exists(directory + 'Contour Lengths/'):
		os.makedirs(directory + 'Contour Lengths/')
		CL_directory = directory + 'Contour Lengths/'
	else:
		CL_directory = directory + 'Contour Lengths/'
	
	try:
		write_file = open(CL_directory + 'Contour Lengths' + file_name + '.txt', 'w')
		print >>write_file, '#This file contains the contour lengths for the minicircles from file ' + file_name + '\n\n\n'
		print >>write_file, '#DNA Minicircle\tContour Length\n' 
	
		for i in range(len(contour_length_array)):
			print >>write_file, str(i) +'\t'+ str(contour_length_array[i])
	except TypeError:
		write_file = open(CL_directory + 'Contour Lengths' + file_name + '.txt', 'w')
		print >>write_file, '#The file ' + file_name + ' contains no detectable contour lengths for the minicircles'

def FindRadiusOfCurvature(splined_coords):
	x_derivatives = np.gradient(splined_coords[:,0])
	y_derivatives = np.gradient(splined_coords[:,1])

	gradients = np.array( [[x_derivatives[i], y_derivatives[i]] for i in range(x_derivatives.size)])

	vector_function = np.sqrt(x_derivatives*x_derivatives + y_derivatives*y_derivatives)

	tangent = np.array([1/vector_function] * 2).transpose() * gradients

	x_tangent = tangent[:,0]
	y_tangent = tangent[:,1]

	deriv_x_tangent = np.gradient(x_tangent)
	deriv_y_tangent = np.gradient(y_tangent)

	dT_dt = np.array([ [ deriv_x_tangent[i], deriv_y_tangent[i]] for i in range(deriv_x_tangent.size)] )

	length_dT_dt = np.sqrt(deriv_x_tangent * deriv_x_tangent + deriv_y_tangent * deriv_y_tangent)

	normal = np.array([1/length_dT_dt]*2).transpose() * dT_dt

	d2s_dt2 = np.gradient(gradients)
	d2x_dt2 = np.gradient(x_derivatives)
	d2y_dt2 = np.gradient(y_derivatives)

	curvature = np.abs(d2x_dt2 * y_derivatives - x_derivatives * d2y_dt2) / (x_derivatives * x_derivatives + y_derivatives * y_derivatives)**1.5
	
	return curvature

def SaveCurvature(directory, file_name, curvature, file_counter):
	if not os.path.exists(directory + 'Curvature'):
		os.makedirs(directory + 'Curvature')
		curve_direct = directory + 'Curvature/'
	else:
		curve_direct = directory + 'Curvature/'
	try:
		plt.figure()
		plt.title('Radius of Curvature for ordered-coordinates for circle 3')
		plt.plot(curvature)
		plt.savefig(curve_direct + file_name +'Curvature' +  str(file_counter) + '.png')
		plt.close()
	except ValueError:
		pass

def AlignImages(centroid, splined_coords, pixel_size):
	origin = np.array([0,0])
	
	alignment_vector = np.subtract(origin, centroid)
	aligned_centroid = np.add(centroid, alignment_vector)
	aligned_coords = np.add(splined_coords, alignment_vector)
	aligned_coords = np.multiply(aligned_coords, [sqrt(pixel_size), sqrt(pixel_size)])
	
	contour_length = 0
	
	sort_index = np.argsort(aligned_coords[:,0])
	
	line = np.array((aligned_coords[sort_index[0]], aligned_coords[sort_index[-1]]))
	
	distance_tree = spatial.KDTree(aligned_coords)
	query = distance_tree.query(aligned_centroid, k = len(aligned_coords))
	maximum_distance = aligned_coords[query[1][-1]]
	
	plot = np.vstack((aligned_centroid, maximum_distance))
	
	vector = aligned_centroid - maximum_distance 
	vector_angle = atan2(vector[1], vector[0])
	
	if vector_angle < 0:
		vector_angle += 2*pi
	
	theta = pi/2 - vector_angle
	
	rotated_x = np.subtract(np.multiply(aligned_coords[:,0], [cos(theta)]), np.multiply(aligned_coords[:,1], [sin(theta)]))
	rotated_y = np.add(np.multiply(aligned_coords[:,0], [sin(theta)]), np.multiply(aligned_coords[:,1], [cos(theta)]))

	rotated_aligned_coords = np.column_stack((rotated_x, rotated_y))
	
	return rotated_aligned_coords, aligned_coords

def ChamferDistance(splined_coords, centroid, pixel_size):
	rotated_aligned_coords, aligned_coords = AlignImages(centroid, splined_coords, pixel_size)
	loop = False
	open = False
	chamf_directory = '/Users/bj002/rotation1/AliceCode/ChamferCoordinates/'
	
	
	open1 = LoadTextFile('open1.txt', chamf_directory)
	open2 = LoadTextFile('open2.txt', chamf_directory)
	loop1 = LoadTextFile('loop1.txt', chamf_directory)
	loop2 = LoadTextFile('loop2.txt', chamf_directory)
	"""
	#plt.plot(rotated_aligned_coords[:,0], rotated_aligned_coords[:,1])
	plt.plot(open1[:,0], open1[:,1])
	plt.plot(open2[:,0], open2[:,1])
	plt.plot(loop1[:,0], loop1[:,1])
	plt.plot(loop2[:,0], loop2[:,1])
	plt.show()
	"""
	open1_tree = spatial.cKDTree(open1)
	open2_tree = spatial.cKDTree(open2)
	loop1_tree = spatial.cKDTree(loop1)
	loop2_tree = spatial.cKDTree(loop2)
	
	open1_query = open1_tree.query(aligned_coords, k=1)
	open2_query = open2_tree.query(aligned_coords, k=1)
	loop1_query = loop1_tree.query(aligned_coords, k=1)
	loop2_query = loop2_tree.query(aligned_coords, k=1)
	
	open1_chamf = np.mean(open1_query[0])
	open2_chamf = np.mean(open2_query[0])
	loop1_chamf = np.mean(loop1_query[0])
	loop2_chamf = np.mean(loop1_query[0])
	
	if loop1_chamf > loop2_chamf:
		loop_chamf = loop1_chamf
	else:
		loop_chamf = loop2_chamf
	
	if open1_chamf > open2_chamf:
		open_chamf = open1_chamf
	else:
		open_chamf = open2_chamf
	
	if loop_chamf < open_chamf:
		loop = True
	else:
		open = True
	
	return loop

plot_display = raw_input('Would you like to see plots of your data and the skeletonisation? y/n (this will reduce the speed of the program)\n')

def main(AFM_Data, Directory):
	circle_no, file_counter = 0,0

	image_file = LoadTextFile(AFM_Data, directory)
	
	xscan, yscan, zscan = ImageUnpacking(image_file, directory, AFM_Data)

	minicircle_length = (33.2*(sequence_length/10.5))
	minicircle_area = (33.2*(sequence_length/10.5))*DNA_diameter
	pixel_size = (xscan[0,-1]*yscan[-1,0])/len(np.ndarray.flatten(zscan))
	circle_pixels = minicircle_area/(pixel_size/sqrt(2))
	pixels = circle_pixels - (0.1*circle_pixels)	#Don't know what this is for?

	cannyzscan, gaussian_scan = CannyEdgeFinder(zscan, pixel_size)

	xlimit = len(xscan)	#possibly can be deleted
	ylimit = len(yscan)	#ditto
	found_circles = 0
	contour_length_array = None #maybe move closer to where it is used
	
	sigma = (20/sqrt(pixel_size))/1.5

	skele_zscan, clean_image, eroded_zscan, eroded_zscanclean = SkeletonizeLoops(zscan, pixel_size, circle_pixels, gaussian_scan, cannyzscan)

	if not os.path.exists(directory + 'AFM Images'):
		os.makedirs(directory + 'AFM Images')
		image_direct = directory + 'AFM Images/'
	else:
		image_direct = directory + 'AFM Images/'

	if plot_display == 'y':
		plt.figure()
		plt.pcolor(xscan,yscan,skele_zscan, cmap = 'binary')
		cbar = plt.colorbar()
		cbar.ax.set_ylabel('Height ($\AA$)', size = 'large')
		plt.xlabel('x-axis ($\AA$)')
		plt.ylabel('y-axis ($\AA$)')
		plt.savefig(image_direct + AFM_Data + 'skele.png')
		plt.close()
	if plot_display == 'y':
		plt.figure()
		plt.pcolor(xscan,yscan,filters.gaussian_filter(clean_image, sigma = sigma/10), cmap = 'binary')
		cbar = plt.colorbar()
		cbar.ax.set_ylabel('Height ($\AA$)', size = 'large')
		plt.xlabel('x-axis ($\AA$)')
		plt.ylabel('y-axis ($\AA$)')
		plt.savefig(image_direct + AFM_Data + 'final.png')
		plt.close()

	if plot_display == 'y':
		plt.figure()
		plt.pcolor(xscan,yscan,cannyzscan, cmap = 'binary')
		cbar = plt.colorbar()
		cbar.ax.set_ylabel('Height ($\AA$)', size = 'large')
		plt.xlabel('x-axis ($\AA$)')
		plt.ylabel('y-axis ($\AA$)')
		plt.savefig(image_direct + AFM_Data + 'canny.png')
		plt.close()	

	if plot_display == 'y':
		plt.figure()
		plt.pcolor(xscan,yscan,eroded_zscanclean, cmap = 'binary')
		cbar = plt.colorbar()
		cbar.ax.set_ylabel('Height ($\AA$)', size = 'large')
		plt.xlabel('x-axis ($\AA$)')
		plt.ylabel('y-axis ($\AA$)')
		plt.savefig(image_direct + AFM_Data + 'cannyeroded.png')
		plt.close()

	trace_lengths, trace_map, number_of_circles = ImageLabelling(pixel_size, skele_zscan)
	
	square_size = int(800 / sqrt(pixel_size))	#This makes an 80nm "box" with which to extract the trace
	if square_size % 2 == 1:
		square_size += 1

	if square_size > 512:
		square_size = 512

	print 'square size is ' + str(square_size)
	print 'There were approximately ' + str(number_of_circles) + ' number of circles detected'
	count = 0

	while circle_no < number_of_circles:	#should this be number_of_circles + 1?????
	###-------------------------------------------------------------------------------###
	### This if statement checks if the initial tracelength is listed as having a 	  ###
	### length of zero, which is common. If this is the case the '0' length circle is ###
	### skipped and the next trace is found.										  ###
	###-------------------------------------------------------------------------------###
		if trace_lengths[(circle_no):(circle_no+1)] == 0:
			circle_no += 1

		count +=1
		found_trace, trace_area, DNA_circle, splined_coords,xmid,ymid, contour_length, loop, curvature = TraceDetection(trace_map, circle_no, xscan, yscan, zscan, pixel_size, square_size)
		
		if found_trace == False:
			circle_no += 1
		elif found_trace == True:
			file_counter +=1
			SaveMinicircleArea(xmid, ymid, file_counter, splined_coords, square_size, DNA_circle, loop, AFM_Data)
			SaveTraceCoordinates(splined_coords, file_counter, AFM_Data, pixel_size)
			SaveCurvature(directory, AFM_Data, curvature, file_counter)
			try:
				contour_length_array = np.append(contour_length_array, contour_length)
			except NameError:
				contour_length_array = np.array((contour_length))
			circle_no += 1
			found_circles += 1

	SaveContourLengths(contour_length_array, AFM_Data)
	print 'There were',circle_no, 'total traces detected in the image, ' + str(AFM_Data) + ' with a total of', found_circles, 'standalone traces discovered.'

directory = '/Users/Alice/'
directory = directory + 'coordinates/'

directory_files = os.listdir(directory)

for i in directory_files:
	if i.endswith('.txt'):
		main(i, directory)
		
print 'Job Finished!'

