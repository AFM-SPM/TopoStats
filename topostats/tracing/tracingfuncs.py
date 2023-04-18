import numpy as np
import matplotlib.pyplot as plt
import math


class getSkeleton:

    """Skeltonisation algorithm based on the paper "A Fast Parallel Algorithm for
    Thinning Digital Patterns" by Zhang et al., 1984"""

    def __init__(self, image_data, binary_map, number_of_columns, number_of_rows, pixel_size):
        self.image_data = image_data
        self.binary_map = binary_map
        self.number_of_columns = number_of_columns
        self.number_of_rows = number_of_rows
        self.pixel_size = pixel_size

        self.p2 = 0
        self.p3 = 0
        self.p4 = 0
        self.p5 = 0
        self.p6 = 0
        self.p7 = 0
        self.p8 = 0

        # skeletonising variables
        self.mask_being_skeletonised = []
        self.output_skeleton = []
        self.skeleton_converged = False
        self.pruning = True

        # Height checking variables
        self.average_height = 0
        # self.cropping_dict = self._initialiseHeightFindingDict()
        self.highest_points = {}
        self.search_window = int(3 / (pixel_size * 1e9))
        # Check that the search window is bigger than 0:
        if self.search_window < 2:
            self.search_window = 3
        self.dir_search = int(0.75 / (pixel_size * 1e9))
        if self.dir_search < 3:
            self.dir_search = 3

        self.getDNAmolHeightStats()
        self.doSkeletonising()

    def getDNAmolHeightStats(self):
        # Why are axes swapped here?
        self.image_data = np.swapaxes(self.image_data, 0, 1)

        # This doesn't appear to be used within this class and its not used anywhere in dnatracing.py either.
        self.average_height = np.average(self.image_data[np.argwhere(self.binary_map == 1)])
        # print(self.average_height)

    def doSkeletonising(self):
        """Simple while loop to check if the skeletonising is finished"""

        self.mask_being_skeletonised = self.binary_map

        while not self.skeleton_converged:
            self._doSkeletonisingIteration()

        # When skeleton converged do an additional iteration of thinning to remove hanging points
        self.finalSkeletonisationIteration()

        self.pruning = True
        while self.pruning:
            self.pruneSkeleton()

        self.output_skeleton = np.argwhere(self.mask_being_skeletonised == 1)

    def _doSkeletonisingIteration(self):
        """Do an iteration of skeletonisation - check for the local binary pixel
        environment and assess the local height values to decide whether to
        delete a point
        """

        number_of_deleted_points = 0
        pixels_to_delete = []

        # Sub-iteration 1 - binary check
        mask_coordinates = np.argwhere(self.mask_being_skeletonised == 1).tolist()
        for point in mask_coordinates:
            if self._deletePixelSubit1(point):
                pixels_to_delete.append(point)

        # Check the local height values to determine if pixels should be deleted
        # pixels_to_delete = self._checkHeights(pixels_to_delete)

        for x, y in pixels_to_delete:
            number_of_deleted_points += 1
            self.mask_being_skeletonised[x, y] = 0
        pixels_to_delete = []

        # Sub-iteration 2 - binary check
        mask_coordinates = np.argwhere(self.mask_being_skeletonised == 1).tolist()
        for point in mask_coordinates:
            if self._deletePixelSubit2(point):
                pixels_to_delete.append(point)

        # Check the local height values to determine if pixels should be deleted
        # pixels_to_delete = self._checkHeights(pixels_to_delete)

        for x, y in pixels_to_delete:
            number_of_deleted_points += 1
            self.mask_being_skeletonised[x, y] = 0

        if number_of_deleted_points == 0:
            self.skeleton_converged = True

    def _deletePixelSubit1(self, point):
        """Function to check whether a single point should be deleted based
        on both its local binary environment and its local height values"""

        self.p2, self.p3, self.p4, self.p5, self.p6, self.p7, self.p8, self.p9 = genTracingFuncs.getLocalPixelsBinary(
            self.mask_being_skeletonised, point[0], point[1]
        )

        if (
            self._binaryThinCheck_a()
            and self._binaryThinCheck_b()
            and self._binaryThinCheck_c()
            and self._binaryThinCheck_d()
        ):
            return True
        else:
            return False

    def _deletePixelSubit2(self, point):
        """Function to check whether a single point should be deleted based
        on both its local binary environment and its local height values"""

        self.p2, self.p3, self.p4, self.p5, self.p6, self.p7, self.p8, self.p9 = genTracingFuncs.getLocalPixelsBinary(
            self.mask_being_skeletonised, point[0], point[1]
        )

        # Add in generic code here to protect high points from being deleted
        if (
            self._binaryThinCheck_a()
            and self._binaryThinCheck_b()
            and self._binaryThinCheck_csharp()
            and self._binaryThinCheck_dsharp()
        ):
            return True
        else:
            return False

    """These functions are ripped from the Zhang et al. paper and do the basic
    skeletonisation steps

    I can use the information from the c,d,c' and d' tests to determine a good
    direction to search for higher height values """

    def _binaryThinCheck_a(self):
        # Condition A protects the endpoints (which will be > 2) - add in code here to prune low height points
        if 2 <= self.p2 + self.p3 + self.p4 + self.p5 + self.p6 + self.p7 + self.p8 + self.p9 <= 6:
            return True
        else:
            return False

    def _binaryThinCheck_b(self):
        count = 0

        if [self.p2, self.p3] == [0, 1]:
            count += 1
        if [self.p3, self.p4] == [0, 1]:
            count += 1
        if [self.p4, self.p5] == [0, 1]:
            count += 1
        if [self.p5, self.p6] == [0, 1]:
            count += 1
        if [self.p6, self.p7] == [0, 1]:
            count += 1
        if [self.p7, self.p8] == [0, 1]:
            count += 1
        if [self.p8, self.p9] == [0, 1]:
            count += 1
        if [self.p9, self.p2] == [0, 1]:
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

    def _checkHeights(self, candidate_points):
        try:
            candidate_points = candidate_points.tolist()
        except AttributeError:
            pass

        for x, y in candidate_points:
            # if point is basically at background don't bother assessing height and just delete:
            if self.image_data[x, y] < 1e-9:
                continue

            # Check if the point has already been identified as a high point
            try:
                self.highest_points[(x, y)]
                candidate_points.pop(candidate_points.index([x, y]))
                # print(x,y)
                continue
            except KeyError:
                pass

            (
                self.p2,
                self.p3,
                self.p4,
                self.p5,
                self.p6,
                self.p7,
                self.p8,
                self.p9,
            ) = genTracingFuncs.getLocalPixelsBinary(self.mask_being_skeletonised, x, y)

            print([self.p9, self.p2, self.p3], [self.p8, 1, self.p4], [self.p7, self.p6, self.p5])

            height_points_to_check = self._checkWhichHeightPoints()
            height_points = np.around(self.cropping_dict[height_points_to_check](x, y), decimals=11)
            test_value = np.around(self.image_data[x, y], decimals=11)
            # print(height_points_to_check, [x,y], self.image_data[x,y], height_points)

            # if the candidate points is the highest local point don't delete it
            if test_value >= sorted(height_points)[-1]:
                print([self.p9, self.p2, self.p3], [self.p8, 1, self.p4], [self.p7, self.p6, self.p5])
                print(height_points_to_check, [x, y], self.image_data[x, y], height_points)
                self.highest_points[(x, y)] = height_points_to_check
                candidate_points.pop(candidate_points.index([x, y]))
                print(height_points_to_check, (x, y))
            else:
                x_n, y_n = self._identifyHighestPoint(x, y, height_points_to_check, height_points)
                self.highest_points[(x_n, y_n)] = height_points_to_check
                pass

        return candidate_points

    def _checkWhichHeightPoints(self):
        # Is the point on the left hand edge?
        # if (self.p8 == 1 and self.p4 == 0 and self.p2 == self.p6):
        if self.p7 + self.p8 + self.p9 == 3 and self.p3 + self.p4 + self.p5 == 0 and self.p2 == self.p6:
            """e.g. [1, 1, 0]
            [1, 1, 0]
            [1, 1, 0]"""
            return "horiz_left"
        # elif (self.p8 == 0 and self.p4 == 1 and self.p2 == self.p6):
        elif self.p7 + self.p8 + self.p9 == 0 and self.p3 + self.p4 + self.p5 == 3 and self.p2 == self.p6:
            """e.g. [0, 1, 1]
            [0, 1, 1]
            [0, 1, 1]"""
            return "horiz_right"
        # elif (self.p2 == 1 and self.p6 == 0 and self.p4 == self.p8):
        elif self.p9 + self.p2 + self.p3 == 3 and self.p5 + self.p6 + self.p7 == 0 and self.p4 == self.p8:
            """e.g. [1, 1, 1]
            [1, 1, 1]
            [0, 0, 0]"""
            return "vert_up"
        # elif (self.p2 == 0 and self.p6 == 1 and self.p4 == self.p8):
        elif (
            self.p9 + self.p2 + self.p3 == 0 and self.p5 + self.p6 + self.p7 == 3 and self.p4 == self.p8
        ):  # and self.p4 == self.p8):
            """e.g. [0, 0, 0]
            [1, 1, 1]
            [1, 1, 1]"""
            return "vert_down"
        elif self.p2 + self.p8 <= 1 and self.p4 + self.p5 + self.p6 >= 2:
            """e.g. [0, 0, 1]       [0, 0, 0]
            [0, 1, 1]       [0, 1, 1]
            [1, 1, 1]   or  [0, 1, 1]"""
            return "diagright_down"
        elif self.p4 + self.p6 <= 1 and self.p8 + self.p9 + self.p2 >= 2:
            """e.g. [1, 1, 1]       [1, 1, 0]
            [1, 1, 0]       [1, 1, 0]
            [1, 0, 0]   or  [0, 0, 0]"""
            return "diagright_up"
        elif self.p2 + self.p4 <= 1 and self.p8 + self.p7 + self.p6 >= 2:
            """e.g. [1, 0, 0]       [0, 0, 0]
            [1, 1, 0]       [1, 1, 0]
            [1, 1, 1]   or  [1, 1, 0]"""
            return "diagleft_down"
        elif self.p8 + self.p6 <= 1 and self.p2 + self.p3 + self.p4 >= 2:
            """e.g. [1, 1, 1]       [0, 1, 1]
            [0, 1, 1]       [0, 1, 1]
            [0, 0, 1]   or  [0, 0, 0]"""
            return "diagleft_up"
        # else:
        #    return 'save'

    def _initialiseHeightFindingDict(self):
        height_cropping_funcs = {}

        height_cropping_funcs["horiz_left"] = self._getHorizontalLeftHeights
        height_cropping_funcs["horiz_right"] = self._getHorizontalRightHeights
        height_cropping_funcs["vert_up"] = self._getVerticalUpwardHeights
        height_cropping_funcs["vert_down"] = self._getVerticalDonwardHeights
        height_cropping_funcs["diagleft_up"] = self._getDiaganolLeftUpwardHeights
        height_cropping_funcs["diagleft_down"] = self._getDiaganolLeftDownwardHeights
        height_cropping_funcs["diagright_up"] = self._getHorizontalRightHeights
        height_cropping_funcs["diagright_down"] = self._getHorizontalRightHeights
        height_cropping_funcs["save"] = self._savePoint

        return height_cropping_funcs

    def _getHorizontalLeftHeights(self, x, y):
        heights = []  # [self.image_data[x,y]]

        for i in range(-self.search_window, self.search_window):
            if i == 0:
                continue
            heights.append(self.image_data[x - i, y])
        return heights

    def _getHorizontalRightHeights(self, x, y):
        heights = []  # [self.image_data[x,y]]

        for i in range(-self.search_window, self.search_window):
            if i == 0:
                continue
            heights.append(self.image_data[x + i, y])
        return heights

    def _getVerticalUpwardHeights(self, x, y):
        heights = []  # [self.image_data[x,y]]

        for i in range(-self.search_window, self.search_window):
            if i == 0:
                continue
            heights.append(self.image_data[x, y + i])
        return heights

    def _getVerticalDonwardHeights(self, x, y):
        heights = []  # [self.image_data[x,y]]

        for i in range(-self.search_window, self.search_window):
            if i == 0:
                continue
            heights.append(self.image_data[x, y - i])
        return heights

    def _getDiaganolLeftUpwardHeights(self, x, y):
        heights = []  # [self.image_data[x,y]]

        for i in range(-self.search_window, self.search_window):
            if i == 0:
                continue
            heights.append(self.image_data[x + i, y + i])
        return heights

    def _getDiaganolLeftDownwardHeights(self, x, y):
        heights = []  # [self.image_data[x,y]]

        for i in range(-self.search_window, self.search_window):
            if i == 0:
                continue
            heights.append(self.image_data[x - i, y - i])
        return heights

    def _getDiaganolRightUpwardHeights(self, x, y):
        heights = []  # [self.image_data[x,y]]

        for i in range(-self.search_window, self.search_window):
            if i == 0:
                continue
            heights.append(self.image_data[x - i, y + i])
        return heights

    def _getDiaganolRightDownwardHeights(self, x, y):
        heights = []  # [self.image_data[x,y]]

        for i in range(-self.search_window, self.search_window):
            if i == 0:
                continue
            heights.append(self.image_data[x + i, y - i])
        return heights

    def _condemnPoint(self, x, y):
        heights = []  # [self.image_data[x,y]]

        for i in range(1, self.search_window):
            heights.append(10)
        return heights

    def _identifyHighestPoint(self, x, y, index_direction, indexed_heights):
        highest_value = 0

        offset = len(indexed_heights) / 2

        for num, height_value in enumerate(indexed_heights):
            if height_value > highest_value:
                highest_point = height_value
                index_position = (num + 1) - offset

        if index_direction == "horiz_left":
            return x - num, y
        elif index_direction == "horiz_right":
            return x + num, y
        elif index_direction == "vert_up":
            return x, y + num
        elif index_direction == "vert_down":
            return x, y - num
        elif index_direction == "diagleft_up":
            return x + num, y + num
        elif index_direction == "diagleft_down":
            return x + num, y - num
        elif index_direction == "diagright_up":
            return x - num, y + num
        elif index_direction == "diagright_down":
            return x - num, y - num

    def finalSkeletonisationIteration(self):
        """A final skeletonisation iteration that removes "hanging" pixels.
        Examples of such pixels are:

                    [0, 0, 0]               [0, 1, 0]            [0, 0, 0]
                    [0, 1, 1]               [0, 1, 1]            [0, 1, 1]
            case 1: [0, 1, 0]   or  case 2: [0, 1, 0] or case 3: [1, 1, 0]

        This is useful for the future functions that rely on local pixel environment
        to make assessments about the overall shape/structure of traces"""

        remaining_coordinates = np.argwhere(self.mask_being_skeletonised).tolist()

        for x, y in remaining_coordinates:
            (
                self.p2,
                self.p3,
                self.p4,
                self.p5,
                self.p6,
                self.p7,
                self.p8,
                self.p9,
            ) = genTracingFuncs.getLocalPixelsBinary(self.mask_being_skeletonised, x, y)

            # Checks for case 1 pixels
            if self._binaryThinCheck_b_returncount() == 2 and self._binaryFinalThinCheck_a():
                self.mask_being_skeletonised[x, y] = 0
            # Checks for case 2 pixels
            elif self._binaryThinCheck_b_returncount() == 3 and self._binaryFinalThinCheck_b():
                self.mask_being_skeletonised[x, y] = 0

    def _binaryFinalThinCheck_a(self):
        if self.p2 * self.p4 == 1:
            return True
        elif self.p4 * self.p6 == 1:
            return True
        elif self.p6 * self.p8 == 1:
            return True
        elif self.p8 * self.p2 == 1:
            return True

    def _binaryFinalThinCheck_b(self):
        if self.p2 * self.p4 * self.p6 == 1:
            return True
        elif self.p4 * self.p6 * self.p8 == 1:
            return True
        elif self.p6 * self.p8 * self.p2 == 1:
            return True
        elif self.p8 * self.p2 * self.p4 == 1:
            return True

    def _binaryThinCheck_b_returncount(self):
        count = 0

        if [self.p2, self.p3] == [0, 1]:
            count += 1
        if [self.p3, self.p4] == [0, 1]:
            count += 1
        if [self.p4, self.p5] == [0, 1]:
            count += 1
        if [self.p5, self.p6] == [0, 1]:
            count += 1
        if [self.p6, self.p7] == [0, 1]:
            count += 1
        if [self.p7, self.p8] == [0, 1]:
            count += 1
        if [self.p8, self.p9] == [0, 1]:
            count += 1
        if [self.p9, self.p2] == [0, 1]:
            count += 1

        return count

    def pruneSkeleton(self):
        """Function to remove the hanging branches from the skeletons - these
        are a persistent problem in the overall tracing process."""

        number_of_branches = 0
        coordinates = np.argwhere(self.mask_being_skeletonised == 1).tolist()

        # The branches are typically short so if a branch is longer than a quarter
        # of the total points its assumed to be part of the real data
        length_of_trace = len(coordinates)
        max_branch_length = int(length_of_trace * 0.15)

        # _deleteSquareEnds(coordinates)

        # first check to find all the end coordinates in the trace
        potential_branch_ends = self._findBranchEnds(coordinates)

        # Now check if its a branch - and if it is delete it
        for x_b, y_b in potential_branch_ends:
            branch_coordinates = [[x_b, y_b]]
            branch_continues = True
            temp_coordinates = coordinates[:]
            temp_coordinates.pop(temp_coordinates.index([x_b, y_b]))

            count = 0

            while branch_continues:
                no_of_neighbours, neighbours = genTracingFuncs.countandGetNeighbours(x_b, y_b, temp_coordinates)

                # If branch continues
                if no_of_neighbours == 1:
                    x_b, y_b = neighbours[0]
                    branch_coordinates.append([x_b, y_b])
                    temp_coordinates.pop(temp_coordinates.index([x_b, y_b]))

                # If the branch reaches the edge of the main trace
                elif no_of_neighbours > 1:
                    branch_coordinates.pop(branch_coordinates.index([x_b, y_b]))
                    branch_continues = False
                    is_branch = True
                # Weird case that happens sometimes
                elif no_of_neighbours == 0:
                    is_branch = True
                    branch_continues = False

                if len(branch_coordinates) > max_branch_length:
                    branch_continues = False
                    is_branch = False

            if is_branch:
                number_of_branches += 1
                for x, y in branch_coordinates:
                    self.mask_being_skeletonised[x, y] = 0

        remaining_coordinates = np.argwhere(self.mask_being_skeletonised)

        if number_of_branches == 0:
            self.pruning = False

    def _findBranchEnds(self, coordinates):
        potential_branch_ends = []

        # Most of the branch ends are just points with one neighbour
        for x, y in coordinates:
            if genTracingFuncs.countNeighbours(x, y, coordinates) == 1:
                potential_branch_ends.append([x, y])
        # Find the ends that are 3/4 neighbouring points
        return potential_branch_ends

    def _deleteSquareEnds(self, coordinates):
        for x, y in coordinates:
            pass


class reorderTrace:
    @staticmethod
    def linearTrace(trace_coordinates):
        """My own function to order the points from a linear trace.

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
        trace array or the other end point is reached."""

        try:
            trace_coordinates = trace_coordinates.tolist()
        except AttributeError:  # array is already a python list
            pass

        # Find one of the end points
        for i, (x, y) in enumerate(trace_coordinates):
            if genTracingFuncs.countNeighbours(x, y, trace_coordinates) == 1:
                ordered_points = [[x, y]]
                trace_coordinates.pop(i)
                break

        remaining_unordered_coords = trace_coordinates[:]

        while remaining_unordered_coords:
            if len(ordered_points) > len(trace_coordinates):
                break

            x_n, y_n = ordered_points[-1]  # get the last point to be added to the array and find its neighbour

            no_of_neighbours, neighbour_array = genTracingFuncs.countandGetNeighbours(
                x_n, y_n, remaining_unordered_coords
            )

            if (
                no_of_neighbours == 1
            ):  # if there's only one candidate - its the next point add it to array and delete from candidate points
                ordered_points.append(neighbour_array[0])
                remaining_unordered_coords.pop(remaining_unordered_coords.index(neighbour_array[0]))
                continue
            elif no_of_neighbours > 1:
                best_next_pixel = genTracingFuncs.checkVectorsCandidatePoints(x_n, y_n, ordered_points, neighbour_array)
                ordered_points.append(best_next_pixel)
                remaining_unordered_coords.pop(remaining_unordered_coords.index(best_next_pixel))
                continue
            elif no_of_neighbours == 0:
                # nn, neighbour_array_all_coords = genTracingFuncs.countandGetNeighbours(x_n, y_n, trace_coordinates)
                # best_next_pixel = genTracingFuncs.checkVectorsCandidatePoints(x_n, y_n, ordered_points, neighbour_array_all_coords)
                best_next_pixel = genTracingFuncs.findBestNextPoint(
                    x_n, y_n, ordered_points, remaining_unordered_coords
                )

                if not best_next_pixel:
                    return np.array(ordered_points)

                ordered_points.append(best_next_pixel)

            # If the tracing has reached the other end of the trace then its finished
            if genTracingFuncs.countNeighbours(x_n, y_n, trace_coordinates) == 1:
                break

        return np.array(ordered_points)

    @staticmethod
    def circularTrace(trace_coordinates):
        """An alternative implementation of the linear tracing algorithm but
        with some adaptations to work with circular dna molecules"""

        try:
            trace_coordinates = trace_coordinates.tolist()
        except AttributeError:  # array is already a python list
            pass

        remaining_unordered_coords = trace_coordinates[:]

        # Find a sensible point to start of the end points
        for i, (x, y) in enumerate(trace_coordinates):
            if genTracingFuncs.countNeighbours(x, y, trace_coordinates) == 2:
                ordered_points = [[x, y]]
                remaining_unordered_coords.pop(i)
                break

        # Randomly choose one of the neighbouring points as the next point
        x_n = ordered_points[0][0]
        y_n = ordered_points[0][1]
        no_of_neighbours, neighbour_array = genTracingFuncs.countandGetNeighbours(x_n, y_n, remaining_unordered_coords)
        ordered_points.append(neighbour_array[0])
        remaining_unordered_coords.pop(remaining_unordered_coords.index(neighbour_array[0]))

        count = 0

        while remaining_unordered_coords:
            x_n, y_n = ordered_points[-1]  # get the last point to be added to the array and find its neighbour

            no_of_neighbours, neighbour_array = genTracingFuncs.countandGetNeighbours(
                x_n, y_n, remaining_unordered_coords
            )

            if (
                no_of_neighbours == 1
            ):  # if there's only one candidate - its the next point add it to array and delete from candidate points
                ordered_points.append(neighbour_array[0])
                remaining_unordered_coords.pop(remaining_unordered_coords.index(neighbour_array[0]))
                continue

            elif no_of_neighbours > 1:
                best_next_pixel = genTracingFuncs.checkVectorsCandidatePoints(x_n, y_n, ordered_points, neighbour_array)
                ordered_points.append(best_next_pixel)
                remaining_unordered_coords.pop(remaining_unordered_coords.index(best_next_pixel))
                continue

            elif len(ordered_points) > len(trace_coordinates):
                vector_start_end = abs(
                    math.hypot(
                        ordered_points[0][0] - ordered_points[-1][0], ordered_points[0][1] - ordered_points[-1][1]
                    )
                )
                if vector_start_end > 5:  # Checks if trace has basically finished i.e. is close to where it started
                    ordered_points.pop(-1)
                    return np.array(ordered_points), False
                else:
                    break

            elif no_of_neighbours == 0:
                # Check if the tracing is finished
                nn, neighbour_array_all_coords = genTracingFuncs.countandGetNeighbours(x_n, y_n, trace_coordinates)
                if ordered_points[0] in neighbour_array_all_coords:
                    break

                    # Checks for bug that happens when tracing messes up
                if ordered_points[-1] == ordered_points[-3]:
                    ordered_points = ordered_points[:-6]
                    return np.array(ordered_points), False

                # Maybe at a crossing with all neighbours deleted - this is crucially a point where errors often occur
                else:
                    # best_next_pixel = genTracingFuncs.checkVectorsCandidatePoints(x_n, y_n, ordered_points, remaining_unordered_coords)
                    best_next_pixel = genTracingFuncs.findBestNextPoint(
                        x_n, y_n, ordered_points, remaining_unordered_coords
                    )

                    if not best_next_pixel:
                        return np.array(ordered_points), False

                    vector_to_new_point = abs(math.hypot(best_next_pixel[0] - x_n, best_next_pixel[1] - y_n))

                    if vector_to_new_point > 5:  # arbitary distinction but mostly valid probably
                        return np.array(ordered_points), False
                    else:
                        ordered_points.append(best_next_pixel)
                    if ordered_points[-1] == ordered_points[-3] and ordered_points[-3] == ordered_points[-5]:
                        ordered_points = ordered_points[:-6]
                        return np.array(ordered_points), False
                    continue

        ordered_points.append(ordered_points[0])
        return np.array(ordered_points), True

    @staticmethod
    def circularTrace_old(trace_coordinates):
        """Reorders the coordinates of a trace from a circular DNA molecule
        (with no loops) using a polar coordinate system with reference to the
        center of mass

        I think every step of this can be vectorised for speed up

        This is vulnerable to bugs if the dna molecule folds in on itself slightly"""

        # calculate the centre of mass for the trace
        com_x = np.average(trace_coordinates[:, 0])
        com_y = np.average(trace_coordinates[:, 1])

        # convert to polar coordinates with respect to the centre of mass
        polar_coordinates = []
        for x1, y1 in trace_coordinates:
            x = x1 - com_x
            y = y1 - com_y

            r = math.hypot(x, y)
            theta = math.atan2(x, y)

            polar_coordinates.append([theta, r])

        sorted_polar_coordinates = sorted(polar_coordinates, key=lambda i: i[0])

        # Reconvert to x, y coordinates
        sorted_coordinates = []
        for theta, r in sorted_polar_coordinates:
            x = r * math.sin(theta)
            y = r * math.cos(theta)

            x2 = x + com_x
            y2 = y + com_y

            sorted_coordinates.append([x2, y2])

        return np.array(sorted_coordinates)

    def loopedCircularTrace():
        pass

    def loopedLinearTrace():
        pass


class genTracingFuncs:
    @staticmethod
    def getLocalPixelsBinary(binary_map, x, y):
        p2 = binary_map[x, y + 1]
        p3 = binary_map[x + 1, y + 1]
        p4 = binary_map[x + 1, y]
        p5 = binary_map[x + 1, y - 1]
        p6 = binary_map[x, y - 1]
        p7 = binary_map[x - 1, y - 1]
        p8 = binary_map[x - 1, y]
        p9 = binary_map[x - 1, y + 1]

        return p2, p3, p4, p5, p6, p7, p8, p9

    @staticmethod
    def countNeighbours(x, y, trace_coordinates):
        """Counts the number of neighbouring points for a given coordinate in
        a list of points"""

        number_of_neighbours = 0
        if [x, y + 1] in trace_coordinates:
            number_of_neighbours += 1
        if [x + 1, y + 1] in trace_coordinates:
            number_of_neighbours += 1
        if [x + 1, y] in trace_coordinates:
            number_of_neighbours += 1
        if [x + 1, y - 1] in trace_coordinates:
            number_of_neighbours += 1
        if [x, y - 1] in trace_coordinates:
            number_of_neighbours += 1
        if [x - 1, y - 1] in trace_coordinates:
            number_of_neighbours += 1
        if [x - 1, y] in trace_coordinates:
            number_of_neighbours += 1
        if [x - 1, y + 1] in trace_coordinates:
            number_of_neighbours += 1
        return number_of_neighbours

    @staticmethod
    def getNeighbours(x, y, trace_coordinates):
        """Returns an array containing the neighbouring points for a given
        coordinate in a list of points"""

        neighbour_array = []
        if [x, y + 1] in trace_coordinates:
            neighbour_array.append([x, y + 1])
        if [x + 1, y + 1] in trace_coordinates:
            neighbour_array.append([x + 1, y + 1])
        if [x + 1, y] in trace_coordinates:
            neighbour_array.append([x + 1, y])
        if [x + 1, y - 1] in trace_coordinates:
            neighbour_array.append([x + 1, y - 1])
        if [x, y - 1] in trace_coordinates:
            neighbour_array.append([x, y - 1])
        if [x - 1, y - 1] in trace_coordinates:
            neighbour_array.append([x - 1, y - 1])
        if [x - 1, y] in trace_coordinates:
            neighbour_array.append([x - 1, y])
        if [x - 1, y + 1] in trace_coordinates:
            neighbour_array.append([x - 1, y + 1])
        return neighbour_array

    @staticmethod
    def countandGetNeighbours(x, y, trace_coordinates):
        """Returns the number of neighbouring points for a coordinate and an
        array containing the those points"""

        neighbour_array = []
        number_of_neighbours = 0
        if [x, y + 1] in trace_coordinates:
            neighbour_array.append([x, y + 1])
            number_of_neighbours += 1
        if [x + 1, y + 1] in trace_coordinates:
            neighbour_array.append([x + 1, y + 1])
            number_of_neighbours += 1
        if [x + 1, y] in trace_coordinates:
            neighbour_array.append([x + 1, y])
            number_of_neighbours += 1
        if [x + 1, y - 1] in trace_coordinates:
            neighbour_array.append([x + 1, y - 1])
            number_of_neighbours += 1
        if [x, y - 1] in trace_coordinates:
            neighbour_array.append([x, y - 1])
            number_of_neighbours += 1
        if [x - 1, y - 1] in trace_coordinates:
            neighbour_array.append([x - 1, y - 1])
            number_of_neighbours += 1
        if [x - 1, y] in trace_coordinates:
            neighbour_array.append([x - 1, y])
            number_of_neighbours += 1
        if [x - 1, y + 1] in trace_coordinates:
            neighbour_array.append([x - 1, y + 1])
            number_of_neighbours += 1
        return number_of_neighbours, neighbour_array

    @staticmethod
    def returnPointsInArray(points_array, trace_coordinates):
        for x, y in points_array:
            if [x, y] in trace_coordinates:
                try:
                    points_in_trace_coordinates.append([x, y])
                except NameError:
                    points_in_trace_coordinates = [[x, y]]
        # for x, y in points_array:
        #    print([x,y])
        #    try:
        #        trace_coordinates.index([x,y])
        #        print(trace_coordinates.index([x,y]))
        #    except ValueError:
        #        continue
        #    else:
        #        try:
        #            points_in_trace_coordinates.append([x,y])
        #        except NameError:
        #            points_in_trace_coordinates = [[x,y]]
        try:
            return points_in_trace_coordinates
        except UnboundLocalError:
            return None

    @staticmethod
    def makeGrid(x, y, size):
        for x_n in range(-size, size + 1):
            x_2 = x + x_n
            for y_n in range(-size, size + 1):
                y_2 = y + y_n
                try:
                    grid.append([x_2, y_2])
                except NameError:
                    grid = [[x_2, y_2]]
        return grid

    @staticmethod
    def findBestNextPoint(x, y, ordered_points, candidate_points):
        ordered_points = np.array(ordered_points)
        candidate_points = np.array(candidate_points)

        ordered_points = ordered_points.tolist()
        candidate_points = candidate_points.tolist()

        for i in range(1, 8):
            # build array of coordinates from which to check
            coords_to_check = genTracingFuncs.makeGrid(x, y, i)
            # check for potential points in the larger search area
            points_in_array = genTracingFuncs.returnPointsInArray(coords_to_check, candidate_points)

            # Make a decision depending on how many points are found
            if not points_in_array:
                continue
            elif len(points_in_array) == 1:
                best_next_point = points_in_array[0]
                return best_next_point
            else:
                best_next_point = genTracingFuncs.checkVectorsCandidatePoints(x, y, ordered_points, points_in_array)
                return best_next_point
        return None

    @staticmethod
    def checkVectorsCandidatePoints(x, y, ordered_points, candidate_points):
        """Finds which neighbouring pixel incurs the smallest angular change
        with reference to a previous pixel in the ordered trace, and chooses that
        as the next point"""

        x_test = ordered_points[-1][0]
        y_test = ordered_points[-1][1]

        if len(ordered_points) > 4:
            x_ref = ordered_points[-3][0]
            y_ref = ordered_points[-3][1]

            x_ref_2 = ordered_points[-2][0]
            y_ref_2 = ordered_points[-2][1]
        elif len(ordered_points) > 3:
            x_ref = ordered_points[-2][0]
            y_ref = ordered_points[-2][1]

            x_ref_2 = ordered_points[0][0]
            y_ref_2 = ordered_points[0][1]
        else:
            x_ref = ordered_points[0][0]
            y_ref = ordered_points[0][1]

            x_ref_2 = ordered_points[0][0]
            y_ref_2 = ordered_points[0][1]

        dx = x_test - x_ref
        dy = y_test - y_ref

        ref_theta = math.atan2(dx, dy)

        x_y_theta = []

        for x_n, y_n in candidate_points:
            x = x_n - x_ref_2
            y = y_n - y_ref_2

            theta = math.atan2(x, y)

            x_y_theta.append([x_n, y_n, abs(theta - ref_theta)])

        ordered_x_y_theta = sorted(x_y_theta, key=lambda x: x[2])
        return [ordered_x_y_theta[0][0], ordered_x_y_theta[0][1]]
