import numpy as np
import matplotlib.pyplot as plt

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

        all_points = np.argwhere(self.mask_being_skeletonised == 1)
        #plt.plot(all_points[:,0], all_points[:,1], '.')

        while not self.skeleton_converged:
            self._doSkeletonisingIteration()

        self.output_skeleton = np.argwhere(self.mask_being_skeletonised == 1)

        #plt.plot(self.output_skeleton[:,0], self.output_skeleton[:,1], '.')
        #plt.show()

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

        self._getLocalPixelsBinary(point[0], point[1])

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

        #Does the binary pixel enviroment mean a pixel should be killed?
        local_binary_pixels = self._getLocalPixelsBinary(point[0], point[1])

        #Add in generic code here to protect high points from being deleted
        if (self._binaryThinCheck_a() and
            self._binaryThinCheck_b() and
            self._binaryThinCheck_csharp() and
            self._binaryThinCheck_dsharp()):
            return True
        else:
            return False

    def _getLocalPixelsBinary(self, x, y):

        '''Function to get the local pixels from the binary map'''

        self.p2 = self.mask_being_skeletonised[x    , y + 1]
        self.p3 = self.mask_being_skeletonised[x + 1, y + 1]
        self.p4 = self.mask_being_skeletonised[x + 1, y    ]
        self.p5 = self.mask_being_skeletonised[x + 1, y - 1]
        self.p6 = self.mask_being_skeletonised[x    , y - 1]
        self.p7 = self.mask_being_skeletonised[x - 1, y - 1]
        self.p8 = self.mask_being_skeletonised[x - 1, y    ]
        self.p9 = self.mask_being_skeletonised[x - 1, y + 1]

    def _getLocalPixelsData(self, x, y):

        '''Function to access the local pixels from the real data'''

        local_height_values = np.ones((4,4))

        #for i in range(4):
        #    for j in range(4):
        #        local_height_values[i,j] = self.full_image_data[]

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
