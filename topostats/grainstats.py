"""Contains class for calculating the statistics of grains - 2d raster images"""
from pathlib import Path
from random import randint
from statistics import mean
from typing import Union

import numpy as np
import skimage.filters as skimage_filters
import skimage.measure as skimage_measure
import skimage.feature as skimage_feature
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import plottingfuncs


class GrainStats:
    """Class for calculating grain stats.
    """

    def __init__(self,
                 data: np.ndarray,
                 labelled_data: np.ndarray,
                 pixel_to_nanometre_scaling: float,
                 path: Union[str, Path],
                 float_format: str = '.4f'):
        """Initialise the class.

        Parameters
        ----------
        data : np.ndarray
            2D Numpy array containing the flattened afm image. Data in this 2D array is floating point.
        labelled_data: np.ndarray
            2D Numpy array containing all the grain masks in the image. Data in this 2D array is boolean.
        pixel_to_nanometre_scaling: float
            Floating point value that defines the scaling factor between nanometers and pixels.
        path : Path
            Path to the folder that will store the grain stats output images and data.
        float_format: str
            Formatter for saving floats.
        """

        self.data = data
        self.labelled_data = labelled_data
        self.pixel_to_nanometre_scaling = pixel_to_nanometre_scaling
        self.savepath = Path(path)
        self.float_format = float_format

        # Calculate grain stats.
        self.calculate_stats()

    def calculate_stats(self):
        """Calculate the stats of grains in the labelled image"""

        # Calculate region properties
        region_properties = skimage_measure.regionprops(self.labelled_data)

        # Plot labelled data
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(self.labelled_data, interpolation='nearest', cmap='afmhot')

        # Iterate over all the grains in the image
        stats_array = []
        for index, region in enumerate(region_properties):

            # Create directory for each grain's plots
            path_grain = self.savepath / ('_' + str(index))
            # Path.mkdir(path_grain, parents=True, exist_ok=True)
            path_grain.mkdir(parents=True, exist_ok=True)

            # Obtain and plot the cropped grain mask
            grain_mask = np.array(region.image)
            plottingfuncs.plot_and_save(grain_mask, path_grain / 'grainmask.png')

            # Obtain the cropped grain image
            minr, minc, maxr, maxc = region.bbox
            grain_image = self.data[minr:maxr, minc:maxc]
            plottingfuncs.plot_and_save(grain_image, path_grain / 'grain_image_raw.png')
            grain_image = np.ma.masked_array(grain_image, mask=np.invert(grain_mask), fill_value=np.nan).filled()
            plottingfuncs.plot_and_save(grain_image, path_grain / 'grain_image.png')

            # Calculate the stats
            min_height = np.nanmin(grain_image)
            max_height = np.nanmax(grain_image)
            median_height = np.nanmedian(grain_image)
            mean_height = np.nanmean(grain_image)

            volume = np.nansum(grain_image)

            area = region.area
            area_cartesian_bbox = region.area_bbox

            edges = self.calculate_edges(grain_mask)
            min_radius, max_radius, mean_radius, median_radius = self.calculate_radius_stats(edges)
            hull, hull_indices, hull_simplexes = self.convex_hull(grain_mask, edges, path_grain)
            smallest_bounding_width, smallest_bounding_length, aspect_ratio = self.calculate_aspect_ratio(
                edges, hull, hull_indices, hull_simplexes, path_grain)

            # save_format = '.4f'

            # Save the stats to csv file. Note that many of the stats are multiplied by a scaling factor to convert from pixel units to nanometres.
            stats = {
                'min_radius':
                min_radius * self.pixel_to_nanometre_scaling,
                'max_radius':
                max_radius * self.pixel_to_nanometre_scaling,
                'mean_radius':
                mean_radius * self.pixel_to_nanometre_scaling,
                'median_radius':
                median_radius * self.pixel_to_nanometre_scaling,
                'min_height':
                min_height,
                'max_height':
                max_height,
                'median_height':
                median_height,
                'mean_height':
                mean_height,
                'volume':
                volume * self.pixel_to_nanometre_scaling**2,
                'area':
                area * self.pixel_to_nanometre_scaling**2,
                'area_cartesian_bbox':
                area_cartesian_bbox * self.pixel_to_nanometre_scaling**2,
                'smallest_bounding_width':
                smallest_bounding_width * self.pixel_to_nanometre_scaling,
                'smallest_bounding_length':
                smallest_bounding_length * self.pixel_to_nanometre_scaling,
                'smallest_bounding_area':
                smallest_bounding_length * smallest_bounding_width * self.pixel_to_nanometre_scaling**2,
                'aspect_ratio':
                aspect_ratio
            }
            stats_array.append(stats)

            # Add cartesian bounding box for the grain to the labelled image
            min_row, min_col, max_row, max_col = region.bbox
            rectangle = mpl.patches.Rectangle((min_col, min_row),
                                              max_col - min_col,
                                              max_row - min_row,
                                              fill=False,
                                              edgecolor='white',
                                              linewidth=2)
            ax.add_patch(rectangle)

        # Create pandas dataframe to hold the stats and save it to a csv file.
        grainstats = pd.DataFrame(data=stats_array)
        grainstats.to_csv(self.savepath / 'grainstats.csv', float_format=self.float_format)

        # Save and close the plot
        plt.savefig(self.savepath / 'labelled_image_bboxes.png')
        plt.close()

    def calculate_radius_stats(self, edges: list):
        """Class method that calculates the statistics relating to the radius. The radius in this context is the distance from the centroid to points on the edge of the grain.

        Parameters
        ----------
        edges : List
            A 2D python list containing the coordinates of the edges of a grain.

        Returns
        -------
        min_radius : float
            The minimum radius of the grain
        max_radius : float
            The maximum radius of the grain
        mean_radius : float
            The mean radius of the grain
        median_radius : float
            The median radius of the grain
        """
        # Convert the edges to the form of a numpy array
        edges = np.array(edges)
        # Calculate the centroid of the grain
        centroid = (sum(edges[:, 0]) / len(edges), sum(edges[:, 1] / len(edges)))
        # Calculate the radii
        displacements = edges - centroid
        radii = [radius for radius in np.sqrt(displacements[:, 0]**2 + displacements[:, 1]**2)]
        # Calculate the statistics relating to the radii
        mean_radius = np.mean(radii)
        median_radius = np.median(radii)
        max_radius = np.max(radii)
        min_radius = np.min(radii)

        return min_radius, max_radius, mean_radius, median_radius

    def calculate_edges(self, grain_mask: np.ndarray):
        """Class method that takes a 2D boolean numpy array image of a grain and returns a python list continting the coordinates of the edges of the grain.

            Parameters:
                grain_mask : np.ndarray
                    A 2D numpy array image of a grain. Data in the array must be boolean.

            Returns:
                edges : list
                    A python list containing the coordinates of the edges of the grain.
        """

        # Fill any holes
        filled_grain_mask = scipy.ndimage.binary_fill_holes(grain_mask)

        # Get outer edge using canny filtering
        edges = skimage_feature.canny(filled_grain_mask, sigma=3)
        nonzero_coordinates = edges.nonzero()

        # Get vector representation of the points
        edges = []
        for vector in np.transpose(nonzero_coordinates):
            edges.append(list(vector))

        return edges

    def convex_hull(self, grain_mask: np.ndarray, edges: list, path: Path, debug: bool = False):
        """Class method that takes a grain mask and the edges of the grain and returns the grain's convex hull. I based this off of the Graham Scan algorithm and should ideally scale in time with O(nlog(n)).

        Parameters
        ----------
        grain_mask : np.ndarray
            A 2D numpy array containing the boolean grain mask for the grain.
        edges : list
            A python list contianing the coordinates of the edges of the grain.
        debug : bool
            Default false. If true, debug information will be displayed to the terminal and plots for the convex hulls and edges will be saved.

        Returns
        -------
        hull : list
            A python list containing coordinates of the points in the hull.
        hull_indices : list
            A python list containing the hull points indices inside the edges list. In other words, this provides a way to find the points from the hull inside the edges list that was passed.
        simplices : list
            A python list of tuples, each tuple representing a simplex of the convex hull. These simplices are sorted such that they follow each other in counterclockwise order.
        """

        def get_angle(point_1: tuple, point_2: tuple):
            """Function that calculates the angle in radians between two points.

                Parameters:
                    point1, point2 : tuple
                        Coordinate vectors for the two points to find the angle between.

                Returns:
                    angle : float
                        The angle in radians between the two input vectors.
            """

            angle = np.arctan2(point_1[1] - point_2[1], point_1[0] - point_2[0])
            return angle

        def get_displacement(point_2: tuple, point_1: tuple = None):
            """Function that calculates the distance squared between two points. Used for distance sorting purposes and therefore does not perform a square root in the interests of efficiency.

                Parameters:
                    point_2 : tuple
                        The point to find the squared distance to.
                    point_1 : tuple
                        Optional - defaults to the starting point defined in the graham_scan() function. The point to find the squared distance from.

                Returns:
                    distance_squared : float
                        The squared distance between the two points.
            """
            # Get the distance squared between two points. If the second point is not provided, use the starting point.
            if point_1 == None: point_1 = start_point
            delta_x = point_2[0] - point_1[1]
            delta_y = point_2[1] - point_1[1]
            distance_squared = delta_x**2 + delta_y**2
            # Don't need the sqrt since just sorting for dist
            return distance_squared

        def is_clockwise(p1: tuple, p2: tuple, p3: tuple) -> Bool:
            """Function to determine if three points make a clockwise or counter-clockwise turn.

                Parameters:
                    p1, p2, p3 : tuple
                        The points that will be used to determine a clockwise or counter-clockwise turn.
                Returns:
                    boolean
                        True if the turn is clockwise
                        False if the turn is counter-clockwise, or if there is no turn.
            """
            # Determine if three points form a clockwise or counter-clockwise turn.
            # I use the method of calculating the determinant of the following rotation matrix here. If the determinant is > 0 then the rotation is counter-clockwise.
            M = np.array(((p1[0], p1[1], 1), (p2[0], p2[1], 1), (p3[0], p3[1], 1)))
            if np.linalg.det(M) > 0:
                return False
            else:
                return True

        def sort_points(points: list):
            """Function to sort points in counter-clockwise order of angle made with the starting point.

                Parameters:
                    points : list
                        A python list of the coordinates to sort.

                Returns:
                    sorted_points : list
                        A python list of sorted points.

            """

            # Return if the list is length 1 or 0.
            if len(points) <= 1: return points
            # Lists that allow sorting of points relative to a current comparision point
            smaller, equal, larger = [], [], []
            # Get a random point in the array to calculate the pivot angle from. This sorts the points relative to this point.
            pivot_angle = get_angle(points[randint(0, len(points) - 1)], start_point)
            for point in points:
                point_angle = get_angle(point, start_point)
                # If the
                if point_angle < pivot_angle: smaller.append(point)
                elif point_angle == pivot_angle: equal.append(point)
                else: larger.append(point)
            # Recursively sort the arrays until each point is sorted
            sorted_points = sort_points(smaller) + sorted(equal, key=get_displacement) + sort_points(larger)
            # Return sorted array where equal angle points are sorted by distance
            return sorted_points

        def graham_scan(edges: list):
            """A function based on the Graham Scan algorithm that constructs a convex hull from points in 2D cartesian space. Ideally this algorithm will take O( n * log(n) ) time.

                Parameters:
                    edges : list
                        A python list of coordinates that make up the edges of the grain.

                Returns:
                hull : list
                    A python list containing coordinates of the points in the hull.
                hull_indices : list
                    A python list containing the hull points indices inside the edges list. In other words, this provides a way to find the points from the hull inside the edges list that was passed.
                simplices : list
                    A python list of tuples, each tuple representing a simplex of the convex hull. These simplices are sorted such that they follow each other in counterclockwise order.
            """
            # Global variable - it's simplest to keep this as a global due to how the get_displacement function is used in sort_points().
            global start_point

            # Find a point guaranteed to be on the hull. I find the bottom most point(s) and sort by x-position.
            min_y_index = None
            for index, point in enumerate(edges):
                if min_y_index == None or point[1] < edges[min_y_index][1]:
                    min_y_index = index
                if point[1] == edges[min_y_index][1] and point[0] < edges[min_y_index][0]:
                    min_y_index = index

            # Set the starting point for the hull. Reminder - this is a global variable.
            start_point = edges[min_y_index]
            # Sort the points
            points_sorted_by_angle = sort_points(edges)

            # Remove starting point from the list so it's not added more than once to the hull
            start_point_index = points_sorted_by_angle.index(start_point)
            del points_sorted_by_angle[start_point_index]
            # Add start point and the first point sorted by angle. Both of these points will always be on the hull.
            hull = [start_point, points_sorted_by_angle[0]]

            # Iterate through each point, checking if this point would cause a clockwise rotation if added to the hull, and if so, backtracking.
            for index, point in enumerate(points_sorted_by_angle[1:]):
                # Determine if the proposed point demands a clockwise rotation
                while is_clockwise(hull[-2], hull[-1], point) == True:
                    # Delete the failed point
                    del hull[-1]
                    if len(hull) < 2: break
                # The point does not immediately cause a clockwise rotation.
                hull.append(point)

            # Get hull indices from original points array
            hull_indices = []
            for point in hull:
                hull_indices.append(edges.index(point))

            # Create simplices from the hull points
            simplices = []
            for i in range(len(hull_indices)):
                simplices.append((hull_indices[i - 1], hull_indices[i]))

            return hull, hull_indices, simplices

        def plot(edges: list, convex_hull: list = None, file_path: Path = None):
            """A function that plots and saves the coordinates of the edges in the grain and optionally the hull. The plot is saved as the file name that is provided.

                Parameters:
                    coordinates : list
                        A list of points to be plotted.
                    convex_hull : list
                        Optional argument. A list of points that form the convex hull. Will be plotted with the coordinates if provided.
                    file_path : Path
                        Path of the file to save the plot as.

                Returns:
                    None
            """

            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            xs, ys = zip(*edges)
            ax.scatter(xs, ys)
            if convex_hull != None:
                for index in range(1, len(convex_hull) + 1):
                    # Loop on the final simplex of the hull to join the last and first points together.
                    if len(convex_hull) == index:
                        index = 0
                    point2 = convex_hull[index]
                    point1 = convex_hull[index - 1]
                    # Plot a line between the two points
                    plt.plot((point1[0], point2[0]), (point1[1], point2[1]), '#994400')
            plt.savefig(file_path)
            plt.close()

        hull, hull_indices, simplexes = graham_scan(edges)

        # Debug information
        if debug:
            plot(edges, hull, path / '_points_hull.png')
            print(f'points: {edges}')
            print(f'hull: {hull}')
            print(f'hull indexes: {hull_indices}')
            print(f'simplexes: {simplexes}')

        return hull, hull_indices, simplexes

    def calculate_aspect_ratio(self,
                               edges: list,
                               convex_hull: np.ndarray,
                               hull_indices: np.ndarray,
                               hull_simplices: np.ndarray,
                               path: Path,
                               debug: bool = False):
        """Class method that takes a list of edge points for a grain, and convex hull simplices and returns the width, length and aspect ratio of the smallest bounding rectangle for the grain.

            Parameters :
                edges : list
                    A python list of coordinates of the edge of the grain.
                hull_simplices : np.ndarray
                    A 2D numpy array of simplices that the hull is comprised of.
                path : Path
                    Path to the save folder for the grain.
                debug : bool
                    If true, various plots will be saved for diagnostic purposes.

            Returns :
                smallest_bounding_width : float
                    The width in pixels (not nanometres), of the smallest bounding rectangle for the grain.
                smallest_bounding_length : float
                    The length in pixels (not nanometres), of the smallest bounding rectangle for the grain.
                aspect_ratio : float
                    The width divided by the length of the smallest bounding rectangle for the grain. It will always be greater or equal to 1.
        """

        # Ensure the edges are in the form of a numpy array.
        edges = np.array(edges)

        # Create a variable to store the smallest area in - this is to be able to compare whilst iterating
        smallest_bounding_area = None
        smallest_bounding_rectangle = None

        # Iterate through the simplices
        for simplex_index, simplex in enumerate(hull_simplices):
            p1 = edges[simplex[0]]
            p1_index = simplex[0]
            p2 = edges[simplex[1]]
            p2_index = simplex[1]
            delta = p1 - p2
            angle = np.arctan2(delta[0], delta[1])

            # Find the centroid of the points
            centroid = (sum(edges[:, 0]) / len(edges), sum(edges[:, 1] / len(edges)))

            # Map the coordinates such that the centroid is now centered on the origin. This is needed for the matrix rotation step coming up.
            remapped_points = edges - centroid

            # Rotate the coordinates using a rotation matrix
            R = np.array(((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))))

            # For each point in the set, rotate it using the above rotation matrix.
            rotated_points = []
            for index, point in enumerate(remapped_points):
                newpoint = R @ point
                rotated_points.append(newpoint)
            rotated_points = np.array(rotated_points)

            # Find the cartesian extremities
            x_min = np.min(rotated_points[:, 0])
            x_max = np.max(rotated_points[:, 0])
            y_min = np.min(rotated_points[:, 1])
            y_max = np.max(rotated_points[:, 1])

            if debug:
                # Create plot
                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111)

                # Draw the points and the current simplex that is being tested
                plt.scatter(x=remapped_points[:, 0], y=remapped_points[:, 1])
                plt.plot(remapped_points[simplex, 0], remapped_points[simplex, 1], '#444444', linewidth=4)
                plt.scatter(x=rotated_points[:, 0], y=rotated_points[:, 1])
                plt.plot(rotated_points[simplex, 0], rotated_points[simplex, 1], 'k-', linewidth=5)
                print(rotated_points[simplex, 0], rotated_points[simplex, 1])

                # Draw the convex hulls
                for simplex in hull_simplices:
                    plt.plot(remapped_points[simplex, 0], remapped_points[simplex, 1], '#888888')
                    plt.plot(rotated_points[simplex, 0], rotated_points[simplex, 1], '#555555')

                # Draw bounding box
                plt.plot([x_min, x_min, x_max, x_max, x_min], [y_min, y_max, y_max, y_min, y_min], '#994400')

                plt.savefig(path / ('bounding_rectangle_construction_simplex_' + str(simplex_index) + '.png'))

            # Calculate the area of the proposed bounding rectangle
            bounding_area = (x_max - x_min) * (y_max - y_min)

            # If current bounding rectangle is the smallest so far
            if smallest_bounding_area == None or bounding_area < smallest_bounding_area:
                smallest_bounding_area = bounding_area
                smallest_bounding_rectangle = (x_min, x_max, y_min, y_max)
                aspect_ratio = (x_max - x_min) / (y_max - y_min)
                smallest_bounding_width = min((x_max - x_min), (y_max - y_min))
                smallest_bounding_length = max((x_max - x_min), (y_max - y_min))
                # Enforce >= 1 aspect ratio
                if aspect_ratio < 1.0:
                    aspect_ratio = 1 / aspect_ratio

        # Unrotate the bounding box vertices
        RINVERSE = R.T
        translated_rotated_bounding_rectangle_vertices = np.array(
            ([x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]))
        translated_bounding_rectangle_vertices = []
        for index, point in enumerate(translated_rotated_bounding_rectangle_vertices):
            newpoint = RINVERSE @ point
            translated_bounding_rectangle_vertices.append(newpoint)
        translated_bounding_rectangle_vertices = np.array(translated_bounding_rectangle_vertices)

        if debug:
            # Create plot
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111)
            plt.scatter(x=edges[:, 0], y=edges[:, 1])
            ax.plot(np.append(translated_rotated_bounding_rectangle_vertices[:, 0],
                              translated_rotated_bounding_rectangle_vertices[0, 0]),
                    np.append(translated_rotated_bounding_rectangle_vertices[:, 1],
                              translated_rotated_bounding_rectangle_vertices[0, 1]),
                    '#994400',
                    label='rotated')
            ax.plot(np.append(translated_bounding_rectangle_vertices[:, 0], translated_bounding_rectangle_vertices[0,
                                                                                                                   0]),
                    np.append(translated_bounding_rectangle_vertices[:, 1], translated_bounding_rectangle_vertices[0,
                                                                                                                   1]),
                    '#004499',
                    label='unrotated')
            ax.scatter(x=remapped_points[:, 0], y=remapped_points[:, 1], color='#004499', label='translated')
            ax.scatter(x=rotated_points[:, 0], y=rotated_points[:, 1], label='rotated')
            ax.legend()
            plt.savefig(path / 'hull_bounding_rectangle_extra')

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        bounding_rectangle_vertices = translated_bounding_rectangle_vertices + centroid
        ax.plot(np.append(bounding_rectangle_vertices[:, 0], bounding_rectangle_vertices[0, 0]),
                np.append(bounding_rectangle_vertices[:, 1], bounding_rectangle_vertices[0, 1]),
                '#994400',
                label='unrotated')
        ax.scatter(x=edges[:, 0], y=edges[:, 1], label='original points')
        ax.set_aspect(1)
        ax.legend()
        plt.savefig(path / 'minimum_bbox.png')
        plt.close()

        return smallest_bounding_width, smallest_bounding_length, aspect_ratio
