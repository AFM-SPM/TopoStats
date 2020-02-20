import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, spatial, interpolate as interp
from skimage import morphology, filters
import math

from tracingfuncs import genTracingFuncs, getSkeleton, reorderTrace

class dnaTrace(object):

    '''
    This class gets all the useful functions from the old tracing code and staples
    them together to create an object that contains the traces for each DNA molecule
    in an image and functions to calculate stats from those traces.

    The traces are stored in dictionaries labelled by their gwyddion defined grain
    number and are represented as numpy arrays.

    The object also keeps track of the skeletonised plots and other intermediates
    in case these are useful for other things in the future.
    '''

    def __init__(self, full_image_data, gwyddion_grains, afm_image_name, pixel_size,
    number_of_columns, number_of_rows, savefile = False):
        self.full_image_data = full_image_data
        self.gwyddion_grains = gwyddion_grains
        self.afm_image_name = afm_image_name
        self.pixel_size = pixel_size
        self.number_of_columns = number_of_columns
        self.number_of_rows = number_of_rows
        self.savefile = savefile

        self.gauss_image = []
        self.grains = {}
        self.dna_masks = {}
        self.skeletons = {}
        self.disordered_trace = {}
        self.ordered_traces = {}
        self.fitted_traces = {}
        self.splined_traces = {}
        self.contour_lengths = {}
        self.mol_is_circular = {}

        self.number_of_traces = 0

        self.getNumpyArraysfromGwyddion()
        self.getDisorderedTrace()
        self.purgeObviousCrap()
        #self.isMolLooped()
        self.determineLinearOrCircular()
        self.getOrderedTraces()
        #self.getFittedTraces()
        #self.getSplinedTraces()
        #self.measureContourLength()

        if self.savefile:
            self.saveTraceFigures()

    def getNumpyArraysfromGwyddion(self):

        ''' Function to get each grain as a numpy array which is stored in a
        dictionary

        Currently the grains are unnecessarily large (the full image) as I don't
        know how to handle the cropped versions

        I find using the gwyddion objects clunky and not helpful once the
        grains have been found

        There is some kind of discrepency between the ordering of arrays from
        gwyddion and how they're usually handled in np arrays meaning you need
        to be careful when indexing from gwyddion derived numpy arrays'''

        for grain_num in set(self.gwyddion_grains):
            #Skip the background
            if grain_num == 0:
                continue

            #Saves each grain as a multidim numpy array
            single_grain_1d = np.array([1 if i == grain_num else 0 for i in self.gwyddion_grains])
            self.grains[int(grain_num)] = np.reshape(single_grain_1d, (self.number_of_columns, self.number_of_rows))

        #Get a 20 A gauss filtered version of the original image - used in refining the pixel positions in getFittedTraces()
        sigma = (20/math.sqrt(self.pixel_size*1e8))
        self.gauss_image = filters.gaussian(self.full_image_data, sigma)

    def getDisorderedTrace(self):

        '''Function to make a skeleton for each of the grains in the image

        Uses my own skeletonisation function from tracingfuncs module. I will
        eventually get round to editing this function to try to reduce the branching
        and to try to better trace from looped molecules '''

        #plt.pcolor(self.full_image_data)

        for grain_num in sorted(self.grains.keys()):

            smoothed_grain = ndimage.binary_dilation(self.grains[grain_num], iterations = 1).astype(self.grains[grain_num].dtype)

            sigma = (5/math.sqrt(self.pixel_size*1e8))/2
            very_smoothed_grain = ndimage.gaussian_filter(smoothed_grain, sigma)

            dna_skeleton = getSkeleton(self.full_image_data, self.grains[grain_num], self.number_of_columns, self.number_of_rows)
            self.disordered_trace[grain_num] = dna_skeleton.output_skeleton

    def purgeObviousCrap(self):

        for dna_num in sorted(self.disordered_trace.keys()):

            if len(self.disordered_trace[dna_num]) < 30:
                self.disordered_trace.pop(dna_num, None)

    def determineLinearOrCircular(self):

        ''' Determines whether each molecule is circular or linear based on the
        local environment of each pixel from the trace

        This function is sensitive to branches from the skeleton so might need
        to implement a function to remove them'''

        for dna_num in sorted(self.disordered_trace.keys()):

            points_with_one_neighbour = 0
            fitted_trace_list = self.disordered_trace[dna_num].tolist()

            #For loop determines how many neighbours a point has - if only one it is an end
            for x,y in fitted_trace_list:

                if genTracingFuncs.countNeighbours(x, y, fitted_trace_list) == 1:
                    points_with_one_neighbour += 1
                else:
                    pass

            if points_with_one_neighbour == 2:
                self.mol_is_circular[dna_num] = False
            else:
                self.mol_is_circular[dna_num] = True

            if self.mol_is_circular[dna_num]:
                print('Found a circular DNA molecule')
            else:
                print('Found a linear DNA molecule')

    def getOrderedTraces(self):

        for dna_num in sorted(self.disordered_trace.keys()):

            if self.mol_is_circular[dna_num]: #and not self.mol_is_looped[dna_num]:
                self.ordered_traces[dna_num] = self._getOrderedCircularTrace(self.disordered_trace[dna_num])

            elif not self.mol_is_circular[dna_num]: #and not self.mol_is_looped[dna_num]:
                self.ordered_traces[dna_num] = reorderTrace.linearTrace(self.disordered_trace[dna_num].tolist())

    def getOrderedTraces(self):

        for dna_num in sorted(self.disordered_trace.keys()):

            if self.mol_is_circular[dna_num]: #and not self.mol_is_looped[dna_num]:
                self.ordered_traces[dna_num] = reorderTrace.circularTrace(self.disordered_trace[dna_num])

            elif not self.mol_is_circular[dna_num]: #and not self.mol_is_looped[dna_num]:
                self.ordered_traces[dna_num] = reorderTrace.linearTrace(self.disordered_trace[dna_num].tolist())

    def getFittedTraces(self):

        ''' Moves the coordinates from the skeletionised traces to lie on the
        highest point on the DNA molecule
        '''

        for dna_num in sorted(self.grains.keys()):

            individual_skeleton = self.disordered_trace[dna_num]
            tree = spatial.cKDTree(individual_skeleton)

            #This sets a 5 nm search in a direction perpendicular to the DNA chain
            height_search_distance = int(20/(self.pixel_size*1e7))

            for coord_num, trace_coordinate in enumerate(individual_skeleton):

                height_values = None

                #I don't have a clue what this block of code is doing
                if trace_coordinate[0] < 0:
                    trace_coordinate[0] = height_search_distance
                elif trace_coordinate[0] >= self.number_of_rows - height_search_distance:
                    trace_coordinate[0] = trace_coordinate[0] = self.number_of_rows - height_search_distance
                elif trace_coordinate[1] < 0:
                    trace_coordinate[1] = (height_search_distance+1)
                elif trace_coordinate[1] >= self.number_of_columns - height_search_distance:
                    trace_coordinate[1] = self.number_of_columns - height_search_distance

                height_values = None
                neighbour_array = tree.query(trace_coordinate, k = 6)
                nearest_point = individual_skeleton[neighbour_array[1][3]]
                vector = np.subtract(nearest_point, trace_coordinate)
                vector_angle = math.degrees(math.atan2(vector[1],vector[0]))	#Angle with respect to x-axis

                if vector_angle < 0:
                    vector_angle += 180

                if 67.5 > vector_angle >= 22.5:	#aka if  angle is closest to 45 degrees
                    perp_direction = 'negative diaganol'
        			#positive diagonal (change in x and y)
        			#Take height values at the inverse of the positive diaganol (i.e. the negative diaganol)
                    y_coords = np.arange(trace_coordinate[1] - height_search_distance, trace_coordinate[1] + height_search_distance)[::-1]
                    x_coords = np.arange(trace_coordinate[0] - height_search_distance, trace_coordinate[0] + height_search_distance)

                elif 157.5 >= vector_angle >= 112.5:#aka if angle is closest to 135 degrees
                    perp_direction = 'positive diaganol'
                    y_coords = np.arange(trace_coordinate[1] - height_search_distance, trace_coordinate[1] + height_search_distance)
                    x_coords = np.arange(trace_coordinate[0] - height_search_distance, trace_coordinate[0] + height_search_distance)

                if 112.5 > vector_angle >= 67.5: #if angle is closest to 90 degrees
                    perp_direction = 'horizontal'
                    #print(trace_coordinate[0] - height_search_distance)
                    #print(trace_coordinate[0] + height_search_distance)
                    x_coords = np.arange(trace_coordinate[0] - height_search_distance, trace_coordinate[0]+height_search_distance)
                    y_coords = np.full(len(x_coords), trace_coordinate[1])

                elif 22.5 > vector_angle: #if angle is closest to 0 degrees
                    perp_direction = 'vertical'
                    y_coords = np.arange(trace_coordinate[1] - height_search_distance, trace_coordinate[1] + height_search_distance)
                    x_coords = np.full(len(y_coords), trace_coordinate[0])

                elif vector_angle >= 157.5: #if angle is closest to 180 degrees
                    perp_direction = 'vertical'
                    y_coords = np.arange(trace_coordinate[1] - height_search_distance, trace_coordinate[1] + height_search_distance)
                    x_coords = np.full(len(y_coords), trace_coordinate[0])

                #Use the perp array to index the guassian filtered image
                perp_array = np.column_stack((x_coords, y_coords))
                height_values = self.gauss_image[perp_array[:,1],perp_array[:,0]]

                #Use interpolation to get "sub pixel" accuracy for heighest position
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
                    #print(perp_array[:,0])
                    #print(np.ndarray.flatten(height_values))
                    int_func = interp.interp1d(perp_array[:,0], np.ndarray.flatten(height_values), kind = 'cubic')
                    interp_heights = int_func(np.arange(perp_array[0,0], perp_array[-1,0], 0.1))
                else:
                    quit('A fatal error occured in the CorrectHeightPositions function, this was likely caused by miscalculating vector angles')

                #Make "fine" coordinates which have the same number of coordinates as the interpolated height values
                if perp_direction == 'negative diaganol':
                    fine_x_coords = np.arange(perp_array[0,0], perp_array[-1,0], 0.1)
                    fine_y_coords = np.arange(perp_array[-1,1], perp_array[0,1], 0.1)[::-1]
                elif perp_direction == 'positive diaganol':
                    fine_x_coords = np.arange(perp_array[0,0], perp_array[-1,0], 0.1)
                    fine_y_coords = np.arange(perp_array[0,1], perp_array[-1,1], 0.1)
                elif perp_direction == 'vertical':
                    fine_y_coords = np.arange(perp_array[0,1], perp_array[-1,1], 0.1)
                    fine_x_coords = np.full(len(fine_y_coords), trace_coordinate[0], dtype = 'float')
                elif perp_direction == 'horizontal':
                    fine_x_coords = np.arange(perp_array[0,0], perp_array[-1,0], 0.1)
                    fine_y_coords = np.full(len(fine_x_coords), trace_coordinate[1], dtype = 'float')

                #Get the coordinates relating to the highest point in the interpolated height values
                fine_coords = np.column_stack((fine_x_coords, fine_y_coords))
                sorted_array = fine_coords[np.argsort(interp_heights)]
                highest_point = sorted_array[-1]

                try:
                    fitted_coordinate_array = np.vstack((fitted_coordinate_array, highest_point))
                except UnboundLocalError:
                    fitted_coordinate_array = highest_point

            self.fitted_traces[dna_num] = fitted_coordinate_array
            del fitted_coordinate_array

    def getSplinedTraces(self):

        '''Gets a splined version of the fitted trace - useful for finding the
        radius of gyration etc

        This function actually calculates the average of several splines which
        is important for getting a good fit on the lower res data'''

        step_size = 5 #arbitary number - needs to reflect pixel size

        for dna_num in sorted(self.ordered_traces.keys()):

            #single_fitted_trace = np.unique(self.ordered_traces[dna_num], axis = 0)

            single_fitted_trace = self.ordered_traces[dna_num]

            nbr = len(single_fitted_trace[:,0])
            count = 0

            #This function makes 5 splined plots and averages them
            if self.mol_is_circular[dna_num]:
                for i in range(step_size):
                    try:
                        #nbr = len(single_fitted_trace[:,0])
                        x = [single_fitted_trace[:,0][j] for j in range(i,nbr,step_size)]
                        y = [single_fitted_trace[:,1][j] for j in range(i,nbr,step_size)]
                        tck,u = interp.splprep([x,y], s=0, per=1)
                        out = interp.splev(np.linspace(0,1,nbr), tck)
                        splined_coords = np.column_stack((out[0], out[1]))
                        #print(np.shape(out), np.shape(splined_coords))
                        try:
                            rolling_total = np.add(rolling_total, splined_coords)
                        except UnboundLocalError:
                            rolling_total = splined_coords
                        spline_success = True
                        count +=1

                    #Old code - Not a great sign that system errors are being caught
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
                        splined_coords = None

                del rolling_total
            else:
                try:
                    #nbr = len(single_fitted_trace[:,0])
                    x = [single_fitted_trace[:,0]]
                    y = [single_fitted_trace[:,1]]
                    tck,u = interp.splprep(single_fitted_trace, s=0, per=1)
                    out = interp.splev(np.linspace(0,1,nbr), tck)
                    splined_coords = np.column_stack((out[0], out[1]))
                    #print(np.shape(out), np.shape(splined_coords))
                    spline_success = True

                #Old code - Not a great sign that system errors are being caught
                except SystemError:
                    print 'Could not spline coordinates'
                    spline_success = False
                    splined_coords = None
                    continue
                except TypeError:
                    print 'The trace is too short or something'
                    spline_success = False
                    splined_coords = None

            self.splined_traces[dna_num] = splined_coords

            #del rolling_total
    def showTraces(self):

        plt.pcolor(self.full_image_data)
        plt.colorbar()
        for dna_num in sorted(self.ordered_traces.keys()):
            plt.plot(self.ordered_traces[dna_num][:,0], self.ordered_traces[dna_num][:,1])
        plt.show()
        plt.close()

    def saveTraceFigures(self):

        plt.pcolor(self.full_image_data)
        plt.colorbar()
        for dna_num in sorted(self.ordered_traces.keys()):
            disordered_trace_list = self.ordered_traces[dna_num].tolist()
            less_dense_trace = np.array([disordered_trace_list[i] for i in range(0,len(disordered_trace_list),5)])
            plt.plot(less_dense_trace[:,0], less_dense_trace[:,1])
        plt.savefig('ordered_trace.png')
        plt.close()

        plt.pcolor(self.full_image_data)
        plt.colorbar()
        for dna_num in sorted(self.disordered_trace.keys()):
            disordered_trace_list = self.disordered_trace[dna_num].tolist()
            less_dense_trace = np.array([disordered_trace_list[i] for i in range(0,len(disordered_trace_list),5)])
            plt.plot(less_dense_trace[:,0], less_dense_trace[:,1])
        plt.savefig('disordered_trace.png')
        plt.close()

        plt.pcolor(self.full_image_data)
        plt.colorbar()
        for dna_num in sorted(self.grains.keys()):
            grain_plt = np.argwhere(self.grains[dna_num] == 1)
            plt.plot(grain_plt[:,0], grain_plt[:,1], '.')
        plt.savefig('grains.png')
        plt.close()


    def findWrithe(self):
        pass

    def findRadiusOfCurvature(self):
        pass

    def measureContourLength(self):

        '''Measures the contour length for each of the splined traces taking into
        account whether the molecule is circular or linear

        Splined traces are currently complete junk so this uses the ordered traces
        for now'''

        for dna_num in sorted(self.ordered_traces.keys()):

            if self.mol_is_circular[dna_num]:
                for num, i in enumerate(self.ordered_traces[dna_num]):

                    x1 = self.ordered_traces[dna_num][num - 1, 0]
                    y1 = self.ordered_traces[dna_num][num - 1, 1]

                    x2 = self.ordered_traces[dna_num][num, 0]
                    y2 = self.ordered_traces[dna_num][num, 1]

                    try:
                        hypotenuse_array.append(math.hypot((x1 - x2), (y1 - y2)))
                    except NameError:
                        hypotenuse_array = [math.hypot((x1 - x2), (y1 - y2))]

                self.contour_lengths[dna_num] = np.sum(np.array(hypotenuse_array)) * self.pixel_size *1e9
                del hypotenuse_array

            else:
                for num, i in enumerate(self.ordered_traces[dna_num]):
                    try:
                        x1 = self.ordered_traces[dna_num][num, 0]
                        y1 = self.ordered_traces[dna_num][num, 1]

                        x2 = self.ordered_traces[dna_num][num + 1, 0]
                        y2 = self.ordered_traces[dna_num][num + 1, 1]

                        try:
                            hypotenuse_array.append(math.hypot((x1 - x2), (y1 - y2)))
                        except NameError:
                            hypotenuse_array = [math.hypot((x1 - x2), (y1 - y2))]
                    except IndexError: #IndexError happens at last point in array
                        self.contour_lengths[dna_num] = np.sum(np.array(hypotenuse_array)) * self.pixel_size * 1e9
                        del hypotenuse_array
                        break

    def writeContourLengths(self, filename):

        if not self.contour_lengths:
            self.measureContourLength()

        with open('%s_contours.txt' % filename, 'w') as writing_file:
            writing_file.write('#units: nm\n')
            for dna_num in sorted(self.contour_lengths.keys()):
                writing_file.write('%f \n' % self.contour_lengths[dna_num])
