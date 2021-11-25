import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage, spatial, interpolate as interp
from skimage import morphology, filters
import math
import warnings
import os

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
    number_of_columns, number_of_rows):
        self.full_image_data = full_image_data
        self.gwyddion_grains = gwyddion_grains
        self.afm_image_name = afm_image_name
        self.pixel_size = pixel_size
        self.number_of_columns = number_of_columns
        self.number_of_rows = number_of_rows

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
        self.num_circular = 0
        self.num_linear = 0

        #supresses scipy splining warnings
        warnings.filterwarnings('ignore')

        self.getNumpyArraysfromGwyddion()
        self.getDisorderedTrace()
        #self.isMolLooped()
        self.purgeObviousCrap()
        self.determineLinearOrCircular(self.disordered_trace)
        self.getOrderedTraces()
        self.determineLinearOrCircular(self.ordered_traces)
        self.getFittedTraces()
        self.getSplinedTraces()
        self.measureContourLength()
        self.reportBasicStats()


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

        # Get a 7 A gauss filtered version of the original image
        # used in refining the pixel positions in getFittedTraces()
        sigma = 0.7/(self.pixel_size*1e9)
        self.gauss_image = filters.gaussian(self.full_image_data, sigma)

    def getDisorderedTrace(self):

        '''Function to make a skeleton for each of the grains in the image

        Uses my own skeletonisation function from tracingfuncs module. I will
        eventually get round to editing this function to try to reduce the branching
        and to try to better trace from looped molecules '''

        for grain_num in sorted(self.grains.keys()):

            smoothed_grain = ndimage.binary_dilation(self.grains[grain_num], iterations = 1).astype(self.grains[grain_num].dtype)

            sigma = (0.01/(self.pixel_size*1e9))
            very_smoothed_grain = ndimage.gaussian_filter(smoothed_grain, sigma)

            try:
                dna_skeleton = getSkeleton(self.gauss_image, smoothed_grain, self.number_of_columns, self.number_of_rows, self.pixel_size)
                self.disordered_trace[grain_num] = dna_skeleton.output_skeleton
            except IndexError:
                # Some gwyddion grains touch image border causing IndexError
                # These grains are deleted
                self.grains.pop(grain_num)
            #skel = morphology.skeletonize(self.grains[grain_num])
            #self.skeletons[grain_num] = np.argwhere(skel == 1)

    def purgeObviousCrap(self):

        for dna_num in sorted(self.disordered_trace.keys()):

            if len(self.disordered_trace[dna_num]) < 10:
                self.disordered_trace.pop(dna_num, None)

    def determineLinearOrCircular(self, traces):

        ''' Determines whether each molecule is circular or linear based on the
        local environment of each pixel from the trace

        This function is sensitive to branches from the skeleton so might need
        to implement a function to remove them'''

        self.num_circular = 0
        self.num_linear = 0

        for dna_num in sorted(traces.keys()):

            points_with_one_neighbour = 0
            fitted_trace_list = traces[dna_num].tolist()

            #For loop determines how many neighbours a point has - if only one it is an end
            for x,y in fitted_trace_list:

                if genTracingFuncs.countNeighbours(x, y, fitted_trace_list) == 1:
                    points_with_one_neighbour += 1
                else:
                    pass

            if points_with_one_neighbour == 0:
                self.mol_is_circular[dna_num] = True
                self.num_circular += 1
            else:
                self.mol_is_circular[dna_num] = False
                self.num_linear += 1


    def getOrderedTraces(self):

        for dna_num in sorted(self.disordered_trace.keys()):

            circle_tracing = True

            if self.mol_is_circular[dna_num]:

                self.ordered_traces[dna_num], trace_completed = reorderTrace.circularTrace(self.disordered_trace[dna_num])

                if not trace_completed:
                    self.mol_is_circular[dna_num] = False
                    try:
                        self.ordered_traces[dna_num] = reorderTrace.linearTrace(self.ordered_traces[dna_num].tolist())
                    except UnboundLocalError:
                        self.mol_is_circular.pop(dna_num)
                        self.disordered_trace.pop(dna_num)
                        self.grains.pop(dna_num)
                        self.ordered_traces.pop(dna_num)

            elif not self.mol_is_circular[dna_num]:
                self.ordered_traces[dna_num] = reorderTrace.linearTrace(self.disordered_trace[dna_num].tolist())

    def reportBasicStats(self):
        #self.determineLinearOrCircular()
        print('There are %i circular and %i linear DNA molecules found in the image' % (self.num_circular, self.num_linear))


    def getFittedTraces(self):

        '''
        Creates self.fitted_traces dictonary which contains trace
        coordinates (for each identified molecule) that are adjusted to lie
        along the highest points of each traced molecule

        param:  self.ordered_traces; the unadjusted skeleton traces
        param:  self.gauss_image; gaussian filtered AFM image of the original
                molecules
        param:  index_width; 1/2th the width of the height profile indexed from
                self.gauss_image at each coordinate (e.g. 2*index_width pixels
                are indexed)

        return: no direct output but instance variable self.fitted_traces
                is populated with adjusted x,y coordinates
        '''

        for dna_num in sorted(self.ordered_traces.keys()):

            individual_skeleton = self.ordered_traces[dna_num]

            # This indexes a 3 nm height profile perpendicular to DNA backbone
            # note that this is a hard coded parameter
            index_width = int(3e-9/(self.pixel_size))
            if index_width < 2:
                index_width = 2

            for coord_num, trace_coordinate in enumerate(individual_skeleton):
                height_values = None

                # Block of code to prevent indexing outside image limits
                # e.g. indexing self.gauss_image[130, 130] for 128x128 image
                if trace_coordinate[0] < 0:
                    # prevents negative number indexing
                    # i.e. stops (trace_coordinate - index_width) < 0
                    trace_coordinate[0] = index_width
                elif trace_coordinate[0] >= (self.number_of_rows -
                                                        index_width):
                    # prevents indexing above image range causing IndexError
                    trace_coordinate[0] = (self.number_of_rows -
                                                        index_width)
                # do same for y coordinate
                elif trace_coordinate[1] < 0:
                    trace_coordinate[1] = index_width
                elif trace_coordinate[1] >= (self.number_of_columns -
                                                        index_width):
                    trace_coordinate[1] = (self.number_of_columns -
                                                        index_width)


                # calculate vector to n - 2 coordinate in trace
                if self.mol_is_circular[dna_num]:
                    nearest_point = individual_skeleton[coord_num-2]
                    vector = np.subtract(nearest_point, trace_coordinate)
                    vector_angle = math.degrees(math.atan2(vector[1],vector[0]))
                else:
                    try:
                        nearest_point = individual_skeleton[coord_num+2]
                    except IndexError:
                        nearest_point = individual_skeleton[coord_num-2]
                    vector = np.subtract(nearest_point, trace_coordinate)
                    vector_angle = math.degrees(math.atan2(vector[1],vector[0]))

                if vector_angle < 0:
                    vector_angle += 180

                # if  angle is closest to 45 degrees
                if 67.5 > vector_angle >= 22.5:
                    perp_direction = 'negative diaganol'
        			# positive diagonal (change in x and y)
        			# Take height values at the inverse of the positive diaganol
                    # (i.e. the negative diaganol)
                    y_coords = np.arange(
                                        trace_coordinate[1] - index_width,
                                        trace_coordinate[1] + index_width
                                        )[::-1]
                    x_coords = np.arange(
                                        trace_coordinate[0] - index_width,
                                        trace_coordinate[0] + index_width
                                        )

                # if angle is closest to 135 degrees
                elif 157.5 >= vector_angle >= 112.5:
                    perp_direction = 'positive diaganol'
                    y_coords = np.arange(
                                        trace_coordinate[1] - index_width,
                                        trace_coordinate[1] + index_width
                                        )
                    x_coords = np.arange(
                                        trace_coordinate[0] - index_width,
                                        trace_coordinate[0] + index_width
                                        )

                # if angle is closest to 90 degrees
                if 112.5 > vector_angle >= 67.5:
                    perp_direction = 'horizontal'
                    x_coords = np.arange(
                                        trace_coordinate[0] - index_width,
                                        trace_coordinate[0]+index_width
                                        )
                    y_coords = np.full(len(x_coords), trace_coordinate[1])

                elif 22.5 > vector_angle: # if angle is closest to 0 degrees
                    perp_direction = 'vertical'
                    y_coords = np.arange(
                                        trace_coordinate[1] - index_width,
                                        trace_coordinate[1] + index_width
                                        )
                    x_coords = np.full(len(y_coords), trace_coordinate[0])

                elif vector_angle >= 157.5: # if angle is closest to 180 degrees
                    perp_direction = 'vertical'
                    y_coords = np.arange(
                                        trace_coordinate[1] - index_width,
                                        trace_coordinate[1] + index_width
                                        )
                    x_coords = np.full(len(y_coords), trace_coordinate[0])

                # Use the perp array to index the guassian filtered image
                perp_array = np.column_stack((x_coords, y_coords))
                height_values = self.gauss_image[perp_array[:,1], perp_array[:,0]]

                '''
                # Old code that interpolated the height profile for "sub-pixel
                # accuracy" - probably slow and not necessary, can delete

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
                '''
                # Grab x,y coordinates for highest point
                #fine_coords = np.column_stack((fine_x_coords, fine_y_coords))
                sorted_array = perp_array[np.argsort(height_values)]
                highest_point = sorted_array[-1]

                try:
                    # could use np.append() here
                    fitted_coordinate_array = np.vstack((
                                                    fitted_coordinate_array,
                                                    highest_point
                                                    ))
                except UnboundLocalError:
                    fitted_coordinate_array = highest_point

            self.fitted_traces[dna_num] = fitted_coordinate_array
            del fitted_coordinate_array # cleaned up by python anyway?

    def getSplinedTraces(self):

        '''Gets a splined version of the fitted trace - useful for finding the
        radius of gyration etc

        This function actually calculates the average of several splines which
        is important for getting a good fit on the lower res data'''

        step_size = int(7e-9/(self.pixel_size)) # 3 nm step size
        interp_step = int(1e-10/self.pixel_size)

        for dna_num in sorted(self.fitted_traces.keys()):

            self.splining_success = True
            nbr = len(self.fitted_traces[dna_num][:,0])

            #Hard to believe but some traces have less than 4 coordinates in total
            if len(self.fitted_traces[dna_num][:,1]) < 4:
                self.splined_traces[dna_num] = self.fitted_traces[dna_num]
                continue

            #The degree of spline fit used is 3 so there cannot be less than 3 points in the splined trace
            while nbr/step_size < 4:
                if step_size <= 1:
                    step_size = 1
                    break
                step_size =- 1

            if self.mol_is_circular[dna_num]:

                #if nbr/step_size > 4: #the degree of spline fit is 3 so there cannot be less than 3 points in splined trace

                ev_array = np.linspace(0,1,nbr*step_size)

                for i in range(step_size):
                    x_sampled = np.array([self.fitted_traces[dna_num][:,0][j] for j in range(i, len(self.fitted_traces[dna_num][:,0]),step_size)])
                    y_sampled = np.array([self.fitted_traces[dna_num][:,1][j] for j in range(i, len(self.fitted_traces[dna_num][:,1]),step_size)])

                    try:
                        tck, u = interp.splprep([x_sampled, y_sampled], s=0, per = 2, quiet = 1, k = 3)
                        out = interp.splev(ev_array, tck)
                        splined_trace = np.column_stack((out[0], out[1]))
                    except ValueError:
                        #Value error occurs when the "trace fitting" really messes up the traces

                        x = np.array([self.ordered_traces[dna_num][:,0][j] for j in range(i, len(self.ordered_traces[dna_num][:,0]),step_size)])
                        y = np.array([self.ordered_traces[dna_num][:,1][j] for j in range(i, len(self.ordered_traces[dna_num][:,1]),step_size)])

                        try:
                            tck, u = interp.splprep([x, y], s=0, per = 2, quiet = 1)
                            out = interp.splev(np.linspace(0,1,nbr*step_size), tck)
                            splined_trace = np.column_stack((out[0], out[1]))
                        except ValueError: #sometimes even the ordered_traces are too bugged out so just delete these traces
                            self.mol_is_circular.pop(dna_num)
                            self.disordered_trace.pop(dna_num)
                            self.grains.pop(dna_num)
                            self.ordered_traces.pop(dna_num)
                            self.splining_success = False
                            try:
                                del spline_running_total
                            except UnboundLocalError: #happens if splining fails immediately
                                break
                            break

                    try:
                        spline_running_total = np.add(spline_running_total, splined_trace)
                    except NameError:
                        spline_running_total = np.array(splined_trace)

                if not self.splining_success:
                    continue

                spline_average = np.divide(spline_running_total, [step_size,step_size])
                del spline_running_total
                self.splined_traces[dna_num] = spline_average
                #else:
                #    x = self.fitted_traces[dna_num][:,0]
                #    y = self.fitted_traces[dna_num][:,1]


                #    try:
                #        tck, u = interp.splprep([x, y], s=0, per = 2, quiet = 1, k = 3)
                #        out = interp.splev(np.linspace(0,1,nbr*step_size), tck)
                #        splined_trace = np.column_stack((out[0], out[1]))
                #        self.splined_traces[dna_num] = splined_trace
                #    except ValueError: #if the trace is really messed up just delete it
                #        self.mol_is_circular.pop(dna_num)
                #        self.disordered_trace.pop(dna_num)
                #        self.grains.pop(dna_num)
                #        self.ordered_traces.pop(dna_num)
            else:
                '''
                start_x = self.fitted_traces[dna_num][0,0]
                end_x = self.fitted_traces[dna_num][-1,0]

                for i in range(step_size):
                    x_sampled = np.array([self.fitted_traces[dna_num][:,0][j] for j in range(i, len(self.fitted_traces[dna_num][:,0]),step_size)])
                    y_sampled = np.array([self.fitted_traces[dna_num][:,1][j] for j in range(i, len(self.fitted_traces[dna_num][:,1]),step_size)])



                    #interp_f = interp.interp1d(x_sampled, y_sampled, kind = 'cubic', assume_sorted = False)

                    #x_new = np.linspace(start_x,end_x,interp_step)
                    #y_new = interp_f(x_new)

                    print(y_new)

                    #tck = interp.splrep(x_sampled, y_sampled, quiet = 0)
                    #out = interp.splev(np.linspace(start_x,end_x, nbr*step_size), tck)
                    splined_trace = np.column_stack((x_new, y_new))

                    try:
                        np.add(spline_running_total, splined_trace)
                    except NameError:
                        spline_running_total = np.array(splined_trace)

                spline_average = spline_running_total
                '''
                # can't get splining of linear molecules to work yet
                self.splined_traces[dna_num] = self.ordered_traces[dna_num]

    def showTraces(self):

        plt.pcolor(self.gauss_image, vmax = -3e-9, vmin = 3e-9)
        plt.colorbar()
        for dna_num in sorted(self.disordered_trace.keys()):
            plt.plot(self.ordered_traces[dna_num][:,0], self.ordered_traces[dna_num][:,1], markersize = 1)
            plt.plot(self.fitted_traces[dna_num][:,0], self.fitted_traces[dna_num][:,1], markersize = 1)
            plt.plot(self.splined_traces[dna_num][:,0], self.splined_traces[dna_num][:,1], markersize = 1)
            #print(len(self.skeletons[dna_num]), len(self.disordered_trace[dna_num]))
            #plt.plot(self.skeletons[dna_num][:,0], self.skeletons[dna_num][:,1], 'o', markersize = 0.8)
        plt.show()
        plt.close()

    def saveTraceFigures(self, filename_with_ext, channel_name, vmaxval, vminval, directory_name = None):

        if directory_name:
            filename_with_ext = self._checkForSaveDirectory(filename_with_ext, directory_name)

        save_file = filename_with_ext[:-4]

        # vmaxval = 20e-9
        # vminval = -10e-9

        plt.pcolor(self.full_image_data, vmax = vmaxval, vmin = vminval)
        plt.colorbar()
        plt.savefig('%s_%s_originalImage.png'  % (save_file, channel_name))
        plt.close()

        plt.pcolor(self.full_image_data, vmax = vmaxval, vmin = vminval)
        plt.colorbar()
        for dna_num in sorted(self.splined_traces.keys()):
            #disordered_trace_list = self.ordered_traces[dna_num].tolist()
            #less_dense_trace = np.array([disordered_trace_list[i] for i in range(0,len(disordered_trace_list),5)])
            plt.plot(self.splined_traces[dna_num][:,0], self.splined_traces[dna_num][:,1], color = 'c')
        plt.savefig('%s_%s_splinedtrace.png'  % (save_file, channel_name))
        plt.close()

        '''
        plt.pcolor(self.full_image_data)
        plt.colorbar()
        for dna_num in sorted(self.ordered_traces.keys()):
            #disordered_trace_list = self.ordered_traces[dna_num].tolist()
            #less_dense_trace = np.array([disordered_trace_list[i] for i in range(0,len(disordered_trace_list),5)])
            plt.plot(self.ordered_traces[dna_num][:,0], self.ordered_traces[dna_num][:,1])
        plt.savefig('%s_%s_splinedtrace.png' % (save_file, channel_name))
        plt.close()
        '''

        plt.pcolor(self.full_image_data, vmax = vmaxval, vmin = vminval)
        plt.colorbar()
        for dna_num in sorted(self.disordered_trace.keys()):
            #disordered_trace_list = self.disordered_trace[dna_num].tolist()
            #less_dense_trace = np.array([disordered_trace_list[i] for i in range(0,len(disordered_trace_list),5)])
            plt.plot(self.disordered_trace[dna_num][:,0], self.disordered_trace[dna_num][:,1], 'o', markersize = 0.5, color = 'c')
        plt.savefig('%s_%s_disorderedtrace.png'  % (save_file, channel_name))
        plt.close()

        plt.pcolor(self.full_image_data, vmax = vmaxval, vmin = vminval)
        plt.colorbar()
        for dna_num in sorted(self.grains.keys()):
            grain_plt = np.argwhere(self.grains[dna_num] == 1)
            plt.plot(grain_plt[:,0], grain_plt[:,1],  'o', markersize = 2, color = 'c')
        plt.savefig('%s_%s_grains.png' % (save_file, channel_name))
        plt.close()

    def _checkForSaveDirectory(self, filename, new_directory_name):

        split_directory_path = os.path.split(filename)

        try:
            os.mkdir(os.path.join(split_directory_path[0], new_directory_name))
        except OSError:  # OSError happens if the directory already exists
            pass

        updated_filename = os.path.join(split_directory_path[0], new_directory_name, split_directory_path[1])

        return updated_filename

    def findWrithe(self):
        pass

    def findRadiusOfCurvature(self):
        pass

    def measureContourLength(self):

        '''Measures the contour length for each of the splined traces taking into
        account whether the molecule is circular or linear

        Splined traces are currently complete junk so this uses the ordered traces
        for now

        Contour length units are nm'''

        for dna_num in sorted(self.splined_traces.keys()):

            if self.mol_is_circular[dna_num]:
                for num, i in enumerate(self.splined_traces[dna_num]):

                    x1 = self.splined_traces[dna_num][num - 1, 0]
                    y1 = self.splined_traces[dna_num][num - 1, 1]

                    x2 = self.splined_traces[dna_num][num, 0]
                    y2 = self.splined_traces[dna_num][num, 1]

                    try:
                        hypotenuse_array.append(math.hypot((x1 - x2), (y1 - y2)))
                    except NameError:
                        hypotenuse_array = [math.hypot((x1 - x2), (y1 - y2))]

                self.contour_lengths[dna_num] = np.sum(np.array(hypotenuse_array)) * self.pixel_size * 1e9
                del hypotenuse_array

            else:
                for num, i in enumerate(self.splined_traces[dna_num]):
                    try:
                        x1 = self.splined_traces[dna_num][num, 0]
                        y1 = self.splined_traces[dna_num][num, 1]

                        x2 = self.splined_traces[dna_num][num + 1, 0]
                        y2 = self.splined_traces[dna_num][num + 1, 1]

                        try:
                            hypotenuse_array.append(math.hypot((x1 - x2), (y1 - y2)))
                        except NameError:
                            hypotenuse_array = [math.hypot((x1 - x2), (y1 - y2))]
                    except IndexError: #IndexError happens at last point in array
                        self.contour_lengths[dna_num] = np.sum(np.array(hypotenuse_array)) * self.pixel_size * 1e9
                        del hypotenuse_array
                        break

    def writeContourLengths(self, filename, channel_name):

        if not self.contour_lengths:
            self.measureContourLength()

        with open('%s_%s_contours.txt' % (filename,channel_name), 'w') as writing_file:
            writing_file.write('#units: nm\n')
            for dna_num in sorted(self.contour_lengths.keys()):
                writing_file.write('%f \n' % self.contour_lengths[dna_num])

class traceStats(object):

    ''' Class used to report on the stats for all the traced molecules in the
    given directory '''

    def __init__(self, trace_object):

        self.trace_object = trace_object

        self.pd_dataframe = []

        self.createTraceStatsObject()

    def createTraceStatsObject(self):

        '''Creates a pandas dataframe with the shape:

        dna_num     directory       ImageName   contourLength   Circular
        1           exp_dir         img1_name   200             True
        2           exp_dir         img2_name   210             False
        3           exp_dir2        img3_name   100             True
        '''

        data_dict = {}

        trace_directory_file = self.trace_object.afm_image_name
        trace_directory = os.path.dirname(trace_directory_file)
        img_name = os.path.basename(trace_directory_file)

        for mol_num, dna_num in enumerate(sorted(self.trace_object.ordered_traces.keys())):

            try:
                data_dict['Molecule number'].append(mol_num)
                data_dict['Image Name'].append(img_name)
                data_dict['Experiment Directory'].append(trace_directory)
                data_dict['Contour Lengths'].append(self.trace_object.contour_lengths[dna_num])
                data_dict['Circular'].append(self.trace_object.mol_is_circular[dna_num])
            except KeyError:
                data_dict['Molecule number'] = [mol_num]
                data_dict['Image Name'] = [img_name]
                data_dict['Experiment Directory'] = [trace_directory]
                data_dict['Contour Lengths'] = [self.trace_object.contour_lengths[dna_num]]
                data_dict['Circular'] = [self.trace_object.mol_is_circular[dna_num]]

        self.pd_dataframe = pd.DataFrame(data=data_dict)

    def updateTraceStats(self, new_traces):

        data_dict = {}

        trace_directory_file = new_traces.afm_image_name
        trace_directory = os.path.dirname(trace_directory_file)
        img_name = os.path.basename(trace_directory_file)


        for mol_num, dna_num in enumerate(sorted(new_traces.contour_lengths.keys())):

            try:
                data_dict['Molecule number'].append(mol_num)
                data_dict['Image Name'].append(img_name)
                data_dict['Experiment Directory'].append(trace_directory)
                data_dict['Contour Lengths'].append(new_traces.contour_lengths[dna_num])
                data_dict['Circular'].append(new_traces.mol_is_circular[dna_num])
            except KeyError:
                data_dict['Molecule number'] = [mol_num]
                data_dict['Image Name'] = [img_name]
                data_dict['Experiment Directory'] = [trace_directory]
                data_dict['Contour Lengths'] = [new_traces.contour_lengths[dna_num]]
                data_dict['Circular'] = [new_traces.mol_is_circular[dna_num]]

        pd_new_traces_dframe = pd.DataFrame(data=data_dict)

        self.pd_dataframe = self.pd_dataframe.append(pd_new_traces_dframe, ignore_index = True)


    def saveTraceStats(self, save_path):
        save_file_name = ''

        if save_path[-1] == '/':
            pass
        else:
            save_path = save_path + '/'

        for i in self.trace_object.afm_image_name.split('/')[:-1]:
            save_file_name = save_file_name + i + '/'
        print(save_file_name)

        self.pd_dataframe.to_json('%stracestats.json' % save_path)

        print('Saved trace info for all analysed images into: %stracestats.json' % save_path)
