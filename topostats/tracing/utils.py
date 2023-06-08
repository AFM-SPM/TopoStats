"""Utilities for tracing"""
from typing import Callable, Union

import math
import numpy as np


class orderTrace:  # pylint: disable=too-few-public-methods
    """Class for ordering co-ordinates of a skeleton"""

    def __init__(self, coordinates: Union[np.ndarray, list]):
        """Initialise the class."""
        self.coordinates = np.array(coordinates).tolist()

    def order(self, shape: str) -> np.ndarray:
        """Order a trace of the given shape.

        Parameters
        ----------
        shape: str
            The shape of the molecule, should be either 'linear' or 'circle'.capitalize()

        Returns
        -------
        Callable"""
        return self._order(shape)

    def _order(self, shape: str) -> Callable:
        """Creator for the method of ordering to use.

        Parameters
        ----------
        shape : str
            The shape of the molecule, should be either 'linear' or 'circle'.

        Returns
        -------
        Callable
            Returns the appropriate method for ordering the coordinates.

        """
        if shape == "linear":
            return self._linear()
        if shape == "circle":
            return self._circle()
        raise ValueError(shape)



    def _linear(self):
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

        trace_coordinates = self.coordinates

        # Find one of the end points
        for i, (x, y) in enumerate(trace_coordinates):
            if orderTrace.countNeighbours(x, y, trace_coordinates) == 1:
                ordered_points = [[x, y]]
                trace_coordinates.pop(i)
                break

        remaining_unordered_coords = trace_coordinates[:]

        while remaining_unordered_coords:
            if len(ordered_points) > len(trace_coordinates):
                break

            x_n, y_n = ordered_points[-1]  # get the last point to be added to the array and find its neighbour

            no_of_neighbours, neighbour_array = orderTrace.countandGetNeighbours(
                x_n, y_n, remaining_unordered_coords
            )

            if (
                no_of_neighbours == 1
            ):  # if there's only one candidate - its the next point add it to array and delete from candidate points
                ordered_points.append(neighbour_array[0])
                remaining_unordered_coords.pop(remaining_unordered_coords.index(neighbour_array[0]))
                continue
            elif no_of_neighbours > 1:
                best_next_pixel = orderTrace.checkVectorsCandidatePoints(x_n, y_n, ordered_points, neighbour_array)
                ordered_points.append(best_next_pixel)
                remaining_unordered_coords.pop(remaining_unordered_coords.index(best_next_pixel))
                continue
            elif no_of_neighbours == 0:
                # nn, neighbour_array_all_coords = genTracingFuncs.countandGetNeighbours(x_n, y_n, trace_coordinates)
                # best_next_pixel = genTracingFuncs.checkVectorsCandidatePoints(x_n, y_n, ordered_points, neighbour_array_all_coords)
                best_next_pixel = orderTrace.findBestNextPoint(
                    x_n, y_n, ordered_points, remaining_unordered_coords
                )

                if not best_next_pixel:
                    return np.array(ordered_points)

                ordered_points.append(best_next_pixel)

            # If the tracing has reached the other end of the trace then its finished
            if orderTrace.countNeighbours(x_n, y_n, trace_coordinates) == 1:
                break

        return np.array(ordered_points)


    def _circle(self):
        """An alternative implementation of the linear tracing algorithm but
        with some adaptations to work with circular dna molecules"""

        trace_coordinates = self.coordinates

        remaining_unordered_coords = trace_coordinates[:]

        # Find a sensible point to start of the end points
        for i, (x, y) in enumerate(trace_coordinates):
            if orderTrace.countNeighbours(x, y, trace_coordinates) == 2:
                ordered_points = [[x, y]]
                remaining_unordered_coords.pop(i)
                break

        # Randomly choose one of the neighbouring points as the next point
        x_n = ordered_points[0][0]
        y_n = ordered_points[0][1]
        no_of_neighbours, neighbour_array = orderTrace.countandGetNeighbours(x_n, y_n, remaining_unordered_coords)
        ordered_points.append(neighbour_array[0])
        remaining_unordered_coords.pop(remaining_unordered_coords.index(neighbour_array[0]))

        count = 0

        while remaining_unordered_coords:
            x_n, y_n = ordered_points[-1]  # get the last point to be added to the array and find its neighbour

            no_of_neighbours, neighbour_array = orderTrace.countandGetNeighbours(
                x_n, y_n, remaining_unordered_coords
            )

            if (
                no_of_neighbours == 1
            ):  # if there's only one candidate - its the next point add it to array and delete from candidate points
                ordered_points.append(neighbour_array[0])
                remaining_unordered_coords.pop(remaining_unordered_coords.index(neighbour_array[0]))
                continue

            elif no_of_neighbours > 1:
                best_next_pixel = orderTrace.checkVectorsCandidatePoints(x_n, y_n, ordered_points, neighbour_array)
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
                nn, neighbour_array_all_coords = orderTrace.countandGetNeighbours(x_n, y_n, trace_coordinates)
                if ordered_points[0] in neighbour_array_all_coords:
                    break

                    # Checks for bug that happens when tracing messes up
                if ordered_points[-1] == ordered_points[-3]:
                    ordered_points = ordered_points[:-6]
                    return np.array(ordered_points), False

                # Maybe at a crossing with all neighbours deleted - this is crucially a point where errors often occur
                else:
                    # best_next_pixel = genTracingFuncs.checkVectorsCandidatePoints(x_n, y_n, ordered_points, remaining_unordered_coords)
                    best_next_pixel = orderTrace.findBestNextPoint(
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
        return np.array(ordered_points)


    #    def _dilate(self):

    def _find_end_point(self):
        """Find end-points of linear grains."""

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
    def findBestNextPoint(x, y, ordered_points, candidate_points):
        ordered_points = np.array(ordered_points)
        candidate_points = np.array(candidate_points)

        ordered_points = ordered_points.tolist()
        candidate_points = candidate_points.tolist()

        for i in range(1, 8):
            # build array of coordinates from which to check
            coords_to_check = orderTrace.makeGrid(x, y, i)
            # check for potential points in the larger search area
            points_in_array = orderTrace.returnPointsInArray(coords_to_check, candidate_points)

            # Make a decision depending on how many points are found
            if not points_in_array:
                continue
            elif len(points_in_array) == 1:
                best_next_point = points_in_array[0]
                return best_next_point
            else:
                best_next_point = orderTrace.checkVectorsCandidatePoints(x, y, ordered_points, points_in_array)
                return best_next_point
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

class Neighbours:  # pylint: disable=too-few-public-methods
    """Class for summarising features of neighbouring pixels."""

    def __init__(self):
        """Initialise the class."""
