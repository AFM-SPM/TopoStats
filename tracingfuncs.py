import numpy as np
import matplotlib.pyplot as plt
import math

class getSkeleton(object):

    '''Skeltonisation algorithm based on the paper "A Fast Parallel Algorithm for
    Thinning Digital Patterns" by Zhang et al., 1984'''

    def __init__(self, full_image_data, binary_map, number_of_columns, number_of_rows):
        self.full_image_data = full_image_data
        self.binary_map = binary_map
        self.number_of_columns = number_of_columns
        self.number_of_rows = number_of_rows

        self.p2 = 0
        self.p3 = 0
        self.p4 = 0
        self.p5 = 0
        self.p6 = 0
        self.p7 = 0
        self.p8 = 0

        self.mask_being_skeletonised = []
        self.output_skeleton = []
        self.skeleton_converged = False

        self.getDNAmolHeightStats()
        self.doSkeletonising()

    def getDNAmolHeightStats(self):
        #Placeholder for where I want to shove my height stats
        pass

    def doSkeletonising(self):

        ''' Simple while loop to check if the skeletonising is finished '''

        self.mask_being_skeletonised = self.binary_map

        while not self.skeleton_converged:
            self._doSkeletonisingIteration()

        #When skeleton converged do an additional iteration of thinning to remove hanging points
        self.finalSkeletonisationIteration()

        self.output_skeleton = np.argwhere(self.mask_being_skeletonised == 1)

    def _doSkeletonisingIteration(self):

        '''Do an iteration of skeletonisation - check for the local binary pixel
        environment and assess the local height values to decide whether to
        delete a point
        '''

        number_of_deleted_points = 0
        pixels_to_delete = []

        #Sub-iteration 1
        mask_coordinates = np.argwhere(self.mask_being_skeletonised == 1)
        for point in mask_coordinates:
            #delete_pixel = self._assessLocalEnvironmentSubit1(point)
            if self._deletePixelSubit1(point):
                pixels_to_delete.append(point)
                number_of_deleted_points += 1

        for x, y in pixels_to_delete:
            self.mask_being_skeletonised[x, y] = 0
        pixels_to_delete = []

        #Sub-iteration 2
        mask_coordinates = np.argwhere(self.mask_being_skeletonised == 1)
        for point in mask_coordinates:
            #delete_pixel = self._assessLocalEnvironmentSubit2(point)
            if self._deletePixelSubit2(point):
                pixels_to_delete.append(point)
                number_of_deleted_points += 1

        for x,y in pixels_to_delete:
            self.mask_being_skeletonised[x, y] = 0

        if number_of_deleted_points == 0:
            self.skeleton_converged = True

    def _deletePixelSubit1(self, point):

        '''Function to check whether a single point should be deleted based
        on both its local binary environment and its local height values'''

        self.p2, self.p3, self.p4, self.p5, self.p6, self.p7, self.p8, self.p9 = genTracingFuncs.getLocalPixelsBinary(self.mask_being_skeletonised, point[0],point[1])
        #self._getLocalPixelsBinary(point[0], point[1])

        if (self._binaryThinCheck_a() and
            self._binaryThinCheck_b() and
            self._binaryThinCheck_c() and
            self._binaryThinCheck_d()):
            return True
        else:
            return False

    def _deletePixelSubit2(self, point):

        '''Function to check whether a single point should be deleted based
        on both its local binary environment and its local height values'''

        self.p2, self.p3, self.p4, self.p5, self.p6, self.p7, self.p8, self.p9 = genTracingFuncs.getLocalPixelsBinary(self.mask_being_skeletonised, point[0],point[1])

        #Add in generic code here to protect high points from being deleted
        if (self._binaryThinCheck_a() and
            self._binaryThinCheck_b() and
            self._binaryThinCheck_csharp() and
            self._binaryThinCheck_dsharp()):
            return True
        else:
            return False

    def _getLocalPixelsData(self, x, y):

        '''Function to access the local pixels from the real data'''

        local_height_values = np.ones((4,4))

        local_height_values[0,0] = self.full_image_data[(point[0] - 1), (point[1] + 1)]
        local_height_values[0,1] = self.full_image_data[(point[0]), (point[1] + 1)]
        local_height_values[0,2] = self.full_image_data[(point[0] + 1), (point[1] + 1)]
        local_height_values[1,0] = self.full_image_data[(point[0] - 1), (point[1])]
        local_height_values[1,1] = self.full_image_data[(point[0]), (point[1])]
        local_height_values[1,2] = self.full_image_data[(point[0] + 1), (point[1])]
        local_height_values[2,0] = self.full_image_data[(point[0] - 1), (point[1] - 1)]
        local_height_values[2,1] = self.full_image_data[(point[0]), (point[1] - 1)]
        local_height_values[2,2] = self.full_image_data[(point[0] + 1), (point[1] - 1)]

        return local_height_values

    '''These functions are ripped from the Zhang et al. paper and do the basic
    skeletonisation steps

    I can use the information from the c,d,c' and d' tests to determine a good
    direction to search for higher height values '''

    def _binaryThinCheck_a(self):
        #Condition A protects the endpoints (which will be > 2) - add in code here to prune low height points
        if 2 <= self.p2 + self.p3 + self.p4 + self.p5 + self.p6 + self.p7 + self.p8 + self.p9 <= 6:
            return True
        else:
            return False

    def _binaryThinCheck_b(self):
        count = 0

        if [self.p2, self.p3] == [0,1]:
            count += 1
        if [self.p3, self.p4] == [0,1]:
            count += 1
        if [self.p4, self.p5] == [0,1]:
            count += 1
        if [self.p5, self.p6] == [0,1]:
            count += 1
        if [self.p6, self.p7] == [0,1]:
            count += 1
        if [self.p7, self.p8] == [0,1]:
            count += 1
        if [self.p8, self.p9] == [0,1]:
            count += 1
        if [self.p9, self.p2] == [0,1]:
            count += 1

        if count == 1:
            return True
        else:
            return False

    def _binaryThinCheck_c(self):
        if self.p2 * self.p4 * self.p6 == 0:
            return True
        else:
            return False

    def _binaryThinCheck_d(self):
        if self.p4 * self.p6 * self.p8 == 0:
            return True
        else:
            return False

    def _binaryThinCheck_csharp(self):
        if self.p2 * self.p4 * self.p8 == 0:
            return True
        else:
            return False

    def _binaryThinCheck_dsharp(self):
        if self.p2 * self.p6 * self.p8 == 0:
            return True
        else:
            return False

    def finalSkeletonisationIteration(self):

        remaining_coordinates = np.argwhere(self.mask_being_skeletonised)

        for x, y in remaining_coordinates:

            self.p2, self.p3, self.p4, self.p5, self.p6, self.p7, self.p8, self.p9 = genTracingFuncs.getLocalPixelsBinary(self.mask_being_skeletonised, x,y)

            if (self.p2 + self.p3 + self.p4 + self.p5 + self.p6 + self.p7 + self.p8 + self.p9 == 2 and
                self._binaryFinalThinCheck_a()):
                self.mask_being_skeletonised[x,y] = 0
            elif (self._binaryThinCheck_b_returncount() == 3 and
                self._binaryFinalThinCheck_b()):
                self.mask_being_skeletonised[x,y] = 0


    def _binaryFinalThinCheck_a(self):

        if self.p2 * self.p4 == 1:
            return True
        elif self.p4 * self.p6 == 1:
            return True
        elif self.p6 * self.p8 ==1:
            return True
        elif self.p8 * self.p2 == 1:
            return True

    def _binaryFinalThinCheck_b(self):

        if self.p2 * self.p4 * self.p6 == 1:
            return True
        elif self.p4 * self.p6 * self.p8 == 1:
            return True
        elif self.p6 * self.p8 * self.p2 ==1:
            return True
        elif self.p8 * self.p2 * self.p4 == 1:
            return True

    def _binaryThinCheck_b_returncount(self):
        count = 0

        if [self.p2, self.p3] == [0,1]:
            count += 1
        if [self.p3, self.p4] == [0,1]:
            count += 1
        if [self.p4, self.p5] == [0,1]:
            count += 1
        if [self.p5, self.p6] == [0,1]:
            count += 1
        if [self.p6, self.p7] == [0,1]:
            count += 1
        if [self.p7, self.p8] == [0,1]:
            count += 1
        if [self.p8, self.p9] == [0,1]:
            count += 1
        if [self.p9, self.p2] == [0,1]:
            count += 1

        return count



class reorderTrace:

    @staticmethod
    def linearTrace(trace_coordinates):

        '''My own function to order the points from a linear trace.

        This works by checking the local neighbours for a given pixel (starting
        at one of the ends). If this pixel has only one neighbour in the array
        of unordered points, this must be the next pixel in the trace -- and it
        is added to the ordered points trace and removed from the
        remaining_unordered_coords array.

        If there is more than one neighbouring pixel, a fairly simple function
        (checkVectorsCandidatePoints) finds which pixel incurs the smallest
        change in angle compared with the rest of the trace and chooses that as
        the next point.

        This process is repeated until all the points are placed in the ordered
        trace array or the other end point is reached. '''

        try:
            trace_coordinates = trace_coordinates.tolist()
        except AttributeError: #array is already a python list
            pass

        #Find one of the end points
        for i, (x, y) in enumerate(trace_coordinates):
            if genTracingFuncs.countNeighbours(x, y, trace_coordinates) == 1:
                ordered_points = [[x, y]]
                trace_coordinates.pop(i)
                break

        remaining_unordered_coords = trace_coordinates[:]

        while remaining_unordered_coords:

            if len(ordered_points) > len(trace_coordinates):
                break

            x_n, y_n = ordered_points[-1] #get the last point to be added to the array and find its neighbour

            no_of_neighbours, neighbour_array = genTracingFuncs.countandGetNeighbours(x_n, y_n, remaining_unordered_coords)

            if no_of_neighbours == 1: #if there's only one candidate - its the next point add it to array and delete from candidate points
                ordered_points.append(neighbour_array[0])
                remaining_unordered_coords.pop(remaining_unordered_coords.index(neighbour_array[0]))
                continue
            elif no_of_neighbours > 1:
                try:
                    best_next_pixel = genTracingFuncs.checkVectorsCandidatePoints(x_n, y_n, ordered_points, neighbour_array)
                except IndexError:
                    best_next_pixel = genTracingFuncs.checkVectorsCandidatePoints(x_n, y_n, ordered_points, neighbour_array, compare = False)

                ordered_points.append(best_next_pixel)
                remaining_unordered_coords.pop(remaining_unordered_coords.index(best_next_pixel))
                continue
            elif no_of_neighbours == 0:
                nn, neighbour_array_all_coords = genTracingFuncs.countandGetNeighbours(x_n, y_n, trace_coordinates)
                best_next_pixel = genTracingFuncs.checkVectorsCandidatePoints(x_n, y_n, ordered_points, neighbour_array_all_coords)
                ordered_points.append(best_next_pixel)

            #If the tracing has reached the other final point then tracing is finished
            if genTracingFuncs.countNeighbours(x_n, y_n,trace_coordinates) == 1:
                break
            #print(x_n, y_n)
        return np.array(ordered_points)

    @staticmethod
    def circularTrace(trace_coordinates):

        ''' An alternative implementation of the linear tracing algorithm but
        with some adaptations to work with circular dna molecules'''

        try:
            trace_coordinates = trace_coordinates.tolist()
        except AttributeError: #array is already a python list
            pass

        remaining_unordered_coords = trace_coordinates[:]

        #Find a sensible point to start of the end points
        for i, (x, y) in enumerate(trace_coordinates):
            if genTracingFuncs.countNeighbours(x, y, trace_coordinates) == 2:
                ordered_points = [[x, y]]
                remaining_unordered_coords.pop(i)
                break

        #Randomly choose one of the neighbouring points as the next point
        x_n = ordered_points[0][0]
        y_n = ordered_points[0][1]
        no_of_neighbours, neighbour_array = genTracingFuncs.countandGetNeighbours(x_n, y_n,remaining_unordered_coords)
        ordered_points.append(neighbour_array[0])
        remaining_unordered_coords.pop(remaining_unordered_coords.index(neighbour_array[0]))

        count = 0

        while remaining_unordered_coords:

            count +=1

            x_n, y_n = ordered_points[-1] #get the last point to be added to the array and find its neighbour

            no_of_neighbours, neighbour_array = genTracingFuncs.countandGetNeighbours(x_n, y_n, remaining_unordered_coords)

            if no_of_neighbours == 1: #if there's only one candidate - its the next point add it to array and delete from candidate points
                ordered_points.append(neighbour_array[0])
                remaining_unordered_coords.pop(remaining_unordered_coords.index(neighbour_array[0]))
                continue

            elif no_of_neighbours > 1:
                try:
                    best_next_pixel = genTracingFuncs.checkVectorsCandidatePoints(x_n, y_n, ordered_points, neighbour_array)
                except IndexError:
                    best_next_pixel = genTracingFuncs.checkVectorsCandidatePoints(x_n, y_n, ordered_points, neighbour_array, compare = False)

                ordered_points.append(best_next_pixel)
                remaining_unordered_coords.pop(remaining_unordered_coords.index(best_next_pixel))
                continue
            elif len(ordered_points) > len(trace_coordinates):
                break

            elif no_of_neighbours == 0:
                #Check if the tracing is finished
                nn, neighbour_array_all_coords = genTracingFuncs.countandGetNeighbours(x_n, y_n, trace_coordinates)
                if ordered_points[0] in neighbour_array_all_coords:
                    break
                #Maybe at a crossing with all neighbours deleted
                else:
                    best_next_pixel = genTracingFuncs.checkVectorsCandidatePoints(x_n, y_n, ordered_points, neighbour_array_all_coords, compare = False)
                    ordered_points.append(best_next_pixel)
                    #trace_coordinates.pop(trace_coordinates.index(best_next_pixel))
                    if count > 1000:
                        #print(len(ordered_points))
                        #print(len(trace_coordinates))
                        #print(ordered_points[-2])
                        #print(ordered_points[-1])
                        #np_array_1 = np.array(ordered_points)
                        #np_array_2 = np.array(remaining_unordered_coords)
                        #plt.plot(np_array_1[:,0], np_array_1[:,1])
                        #plt.plot(np_array_2[:,0], np_array_2[:,1], '.')
                        #plt.show()
                        break
                    continue

        ordered_points.append(ordered_points[0])
        return np.array(ordered_points)

    @staticmethod
    def circularTrace_old(trace_coordinates):

        ''' Reorders the coordinates of a trace from a circular DNA molecule
        (with no loops) using a polar coordinate system with reference to the
        center of mass

        I think every step of this can be vectorised for speed up

        This is vulnerable to bugs if the dna molecule folds in on itself slightly'''

        #calculate the centre of mass for the trace
        com_x = np.average(trace_coordinates[:,0])
        com_y = np.average(trace_coordinates[:,1])

        #convert to polar coordinates with respect to the centre of mass
        polar_coordinates = []
        for x1, y1 in trace_coordinates:

            x = x1 - com_x
            y = y1 - com_y

            r = math.hypot(x,y)
            theta = math.atan2(x,y)

            polar_coordinates.append([theta,r])

        sorted_polar_coordinates = sorted(polar_coordinates, key = lambda i:i[0])

        #Reconvert to x, y coordinates
        sorted_coordinates = []
        for theta, r in sorted_polar_coordinates:

            x = r*math.sin(theta)
            y = r*math.cos(theta)

            x2 = x + com_x
            y2 = y + com_y

            sorted_coordinates.append([x2,y2])

        return np.array(sorted_coordinates)

    def loopedCircularTrace():
        pass

    def loopedLinearTrace():
        pass

class genTracingFuncs:

    @staticmethod
    def getLocalPixelsBinary(binary_map, x, y):
        p2 = binary_map[x    , y + 1]
        p3 = binary_map[x + 1, y + 1]
        p4 = binary_map[x + 1, y    ]
        p5 = binary_map[x + 1, y - 1]
        p6 = binary_map[x    , y - 1]
        p7 = binary_map[x - 1, y - 1]
        p8 = binary_map[x - 1, y    ]
        p9 = binary_map[x - 1, y + 1]

        return p2,p3,p4,p5,p6,p7,p8,p9

    @staticmethod
    def countNeighbours( x, y, trace_coordinates):

        '''Counts the number of neighbouring points for a given coordinate in
        a list of points '''

        number_of_neighbours = 0
        if [x    , y + 1] in trace_coordinates:
            number_of_neighbours += 1
        if [x + 1, y + 1] in trace_coordinates:
            number_of_neighbours +=1
        if [x + 1, y    ] in trace_coordinates:
            number_of_neighbours +=1
        if [x + 1, y - 1] in trace_coordinates:
            number_of_neighbours +=1
        if [x    , y - 1] in trace_coordinates:
            number_of_neighbours +=1
        if [x - 1, y - 1] in trace_coordinates:
            number_of_neighbours +=1
        if [x - 1, y    ] in trace_coordinates:
            number_of_neighbours +=1
        if [x - 1, y + 1] in trace_coordinates:
            number_of_neighbours +=1
        return number_of_neighbours

    @staticmethod
    def getNeighbours(x, y, trace_coordinates):

        '''Returns an array containing the neighbouring points for a given
        coordinate in a list of points '''

        neighbour_array = []
        if [x    , y + 1] in trace_coordinates:
            neighbour_array.append([x    ,y + 1])
        if [x + 1, y + 1] in trace_coordinates:
            neighbour_array.append([x + 1,y + 1])
        if [x + 1, y    ] in trace_coordinates:
            neighbour_array.append([x + 1,y    ])
        if [x + 1, y - 1] in trace_coordinates:
            neighbour_array.append([x + 1, y - 1])
        if [x    , y - 1] in trace_coordinates:
            neighbour_array.append([x    , y - 1])
        if [x - 1, y - 1] in trace_coordinates:
            neighbour_array.append([x - 1, y - 1])
        if [x - 1, y    ] in trace_coordinates:
            neighbour_array.append([x - 1, y    ])
        if [x - 1, y + 1] in trace_coordinates:
            neighbour_array.append([x - 1, y + 1])
        return neighbour_array

    @staticmethod
    def countandGetNeighbours(x, y, trace_coordinates):

        '''Returns the number of neighbouring points for a coordinate and an
        array containing the those points '''

        neighbour_array = []
        number_of_neighbours = 0
        if [x    , y + 1] in trace_coordinates:
            neighbour_array.append([x    ,y + 1])
            number_of_neighbours +=1
        if [x + 1, y + 1] in trace_coordinates:
            neighbour_array.append([x + 1,y + 1])
            number_of_neighbours +=1
        if [x + 1, y    ] in trace_coordinates:
            neighbour_array.append([x + 1,y    ])
            number_of_neighbours +=1
        if [x + 1, y - 1] in trace_coordinates:
            neighbour_array.append([x + 1, y - 1])
            number_of_neighbours +=1
        if [x    , y - 1] in trace_coordinates:
            neighbour_array.append([x    , y - 1])
            number_of_neighbours +=1
        if [x - 1, y - 1] in trace_coordinates:
            neighbour_array.append([x - 1, y - 1])
            number_of_neighbours +=1
        if [x - 1, y    ] in trace_coordinates:
            neighbour_array.append([x - 1, y    ])
            number_of_neighbours +=1
        if [x - 1, y + 1] in trace_coordinates:
            neighbour_array.append([x - 1, y + 1])
            number_of_neighbours +=1
        return number_of_neighbours, neighbour_array

    @staticmethod
    def checkVectorsCandidatePoints(x, y, ordered_points, candidate_points, compare = True):

        '''Finds which neighbouring pixel incurs the smallest angular change
        with reference to a previous pixel in the ordered trace and chooses that
        as the next point '''
        if compare:
            #Calculate reference angle
            x_1 = ordered_points[-4][0]
            y_1 = ordered_points[-4][1]

            x_2 = ordered_points[-1][0]
            y_2 = ordered_points[-1][1]

            dx = x_2 - x_1
            dy = y_2 - y_1

            ref_theta = math.atan2(dx,dy)

        x_y_theta = []
        point_to_check_from = ordered_points[-2]

        for x_n, y_n in candidate_points:

            x = x_n - point_to_check_from[0]
            y = y_n - point_to_check_from[1]

            theta = math.atan2(x,y)
            if compare:
                x_y_theta.append([x_n,y_n,abs(theta-ref_theta)])
            else:
                x_y_theta.append([x_n,y_n,abs(theta)])

        ordered_x_y_theta = sorted(x_y_theta, key = lambda x:x[2])
        return [ordered_x_y_theta[0][0], ordered_x_y_theta[0][1]]
