"""Miscellaneous tracing functions."""

from __future__ import annotations
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import math

from topostats.utils import convolve_skeleton


class reorderTrace:
    """
    Class to aid the consecutive ordering of adjacent coordinates of a pixel grid.
    """

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
            if genTracingFuncs.count_and_get_neighbours(x, y, trace_coordinates)[0] == 1:
                ordered_points = [[x, y]]
                trace_coordinates.pop(i)
                break

        remaining_unordered_coords = trace_coordinates[:]

        while remaining_unordered_coords:
            if len(ordered_points) > len(trace_coordinates):
                break

            x_n, y_n = ordered_points[-1]  # get the last point to be added to the array and find its neighbour

            no_of_neighbours, neighbour_array = genTracingFuncs.count_and_get_neighbours(
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
                # nn, neighbour_array_all_coords = genTracingFuncs.count_and_get_neighbours(x_n, y_n, trace_coordinates)
                # best_next_pixel = genTracingFuncs.checkVectorsCandidatePoints(x_n, y_n, ordered_points, neighbour_array_all_coords)
                best_next_pixel = genTracingFuncs.findBestNextPoint(
                    x_n, y_n, ordered_points, remaining_unordered_coords
                )

                if not best_next_pixel:
                    return np.array(ordered_points)

                ordered_points.append(best_next_pixel)

            # If the tracing has reached the other end of the trace then its finished
            if genTracingFuncs.count_and_get_neighbours(x_n, y_n, trace_coordinates)[0] == 1:
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
            if genTracingFuncs.count_and_get_neighbours(x, y, trace_coordinates)[0] == 2:
                ordered_points = [[x, y]]
                remaining_unordered_coords.pop(i)
                break

        # Randomly choose one of the neighbouring points as the next point
        x_n = ordered_points[0][0]
        y_n = ordered_points[0][1]
        no_of_neighbours, neighbour_array = genTracingFuncs.count_and_get_neighbours(
            x_n, y_n, remaining_unordered_coords
        )
        ordered_points.append(neighbour_array[0])
        remaining_unordered_coords.pop(remaining_unordered_coords.index(neighbour_array[0]))

        count = 0

        while remaining_unordered_coords:
            x_n, y_n = ordered_points[-1]  # get the last point to be added to the array and find its neighbour

            no_of_neighbours, neighbour_array = genTracingFuncs.count_and_get_neighbours(
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
                nn, neighbour_array_all_coords = genTracingFuncs.count_and_get_neighbours(x_n, y_n, trace_coordinates)
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

                    if vector_to_new_point > 5:  # arbitrary distinction but mostly valid probably
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
    def count_and_get_neighbours(x, y, trace_coordinates) -> tuple[int, list]:
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


def order_branch(binary_image: npt.NDArray, anchor: list):
    """
    Order a linear branch by identifying an endpoint, and looking at the local area of the point to find the next.

    Parameters
    ----------
    binary_image : npt.NDArray
        A binary image of a skeleton segment to order it's points.
    anchor : list
        A list of 2 integers representing the coordinate to order the branch from the endpoint closest to this.

    Returns
    -------
    npt.NDArray
        An array of ordered coordinates.
    """
    skel = binary_image.copy()

    if len(np.argwhere(skel == 1)) < 3:  # if < 3 coords just return them
        return np.argwhere(skel == 1)

    # get branch starts
    endpoints_highlight = convolve_skeleton(skel)
    endpoints = np.argwhere(endpoints_highlight == 2)
    if len(endpoints) != 0:  # if any endpoints, start closest to anchor
        dist_vals = abs(endpoints - anchor).sum(axis=1)
        start = endpoints[np.argmin(dist_vals)]
    else:  # will be circular so pick the first coord (is this always the case?)
        start = np.argwhere(skel == 1)[0]
    # order the points according to what is nearby
    ordered = order_branch_from_start(skel, start)

    return np.array(ordered)


def order_branch_from_start(
    nodeless: npt.NDArray, start: npt.NDArray, max_length: float | np.inf = np.inf
) -> npt.NDArray:
    """
    Order an unbranching skeleton from an end (startpoint) along a specified length.

    Parameters
    ----------
    nodeless : npt.NDArray
        A 2D array of a binary unbranching skeleton.
    start : npt.NDArray
        2x1 coordinate that must exist in 'nodeless'.
    max_length : float | np.inf, optional
        Maximum length to traverse along while ordering, by default np.inf.

    Returns
    -------
    npt.NDArray
        Ordered coordinates.
    """
    dist = 0
    # add starting point to ordered array
    ordered = []
    ordered.append(start)
    nodeless[start[0], start[1]] = 0  # remove from array

    # iterate to order the rest of the points
    current_point = ordered[-1]  # get last point
    area, _ = local_area_sum(nodeless, current_point)  # look at local area
    local_next_point = np.argwhere(
        area.reshape(
            (
                3,
                3,
            )
        )
        == 1
    ) - (1, 1)
    dist += np.sqrt(2) if abs(local_next_point).sum() > 1 else 1

    while len(local_next_point) != 0 and dist <= max_length:
        next_point = (current_point + local_next_point)[0]
        # find where to go next
        ordered.append(next_point)
        nodeless[next_point[0], next_point[1]] = 0  # set value to zero
        current_point = ordered[-1]  # get last point
        area, _ = local_area_sum(nodeless, current_point)  # look at local area
        local_next_point = np.argwhere(
            area.reshape(
                (
                    3,
                    3,
                )
            )
            == 1
        ) - (1, 1)
        dist += np.sqrt(2) if abs(local_next_point).sum() > 1 else 1

    return np.array(ordered)


def local_area_sum(binary_map: npt.NDArray, point: list | tuple | npt.NDArray) -> npt.NDArray:
    """
    Evaluate the local area around a point in a binary map.

    Parameters
    ----------
    binary_map : npt.NDArray
        A binary array of an image.
    point : Union[list, tuple, npt.NDArray]
        A single object containing 2 integers relating to a point within the binary_map.

    Returns
    -------
    npt.NDArray
        An array values of the local coordinates around the point.
    int
        A value corresponding to the number of neighbours around the point in the binary_map.
    """
    x, y = point
    local_pixels = binary_map[x - 1 : x + 2, y - 1 : y + 2].flatten()
    local_pixels[4] = 0  # ensure centre is 0
    return local_pixels, local_pixels.sum()


@staticmethod
def coord_dist(coords: npt.NDArray, pixel_to_nm_scaling: float = 1) -> npt.NDArray:
    """
    Accumulate a real distance traversing from pixel to pixel from a list of coordinates.

    Parameters
    ----------
    coords : npt.NDArray
        A Nx2 integer array corresponding to the ordered coordinates of a binary trace.
    pixel_to_nm_scaling : float
        The pixel to nanometer scaling factor.

    Returns
    -------
    npt.NDArray
        An array of length N containing thcumulative sum of the distances.
    """
    dist_list = [0]
    dist = 0
    for i in range(len(coords) - 1):
        if abs(coords[i] - coords[i + 1]).sum() == 2:
            dist += 2**0.5
        else:
            dist += 1
        dist_list.append(dist)
    return np.asarray(dist_list) * pixel_to_nm_scaling


def get_two_combinations(list_of_values) -> list:
    """
    Obtain all paired combinations of values in the list.

    Example: [1,2] -> [[1,2]], [1,2,3] -> [[1,2],[1,3],[2,3]]

    Parameters
    ----------
    list_of_values : list
        List of numbers.

    Returns
    -------
    list
        A list of combinatoric pairs of values, retaining the initial order.
    """
    combinations = []
    for i in range(len(list_of_values) - 1):
        for j in list_of_values[i + 1 :]:
            combinations.append([list_of_values[i], j])
    return combinations
