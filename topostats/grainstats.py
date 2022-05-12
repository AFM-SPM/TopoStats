"""Contains class for calculating the statistics of grains - 2d raster images"""
import logging
from pathlib import Path
from random import randint
from statistics import mean
from typing import Union, List, Tuple

import numpy as np
import skimage.filters as skimage_filters
import skimage.measure as skimage_measure
import skimage.feature as skimage_feature
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

from topostats.plottingfuncs import plot_and_save
from topostats.logs.logs import LOGGER_NAME

LOGGER = logging.getLogger(LOGGER_NAME)


class GrainStats:
    """Class for calculating grain stats."""

    def __init__(
        self,
        data: np.ndarray,
        labelled_data: np.ndarray,
        pixel_to_nanometre_scaling: float,
        img_name: str,
        output_dir: Union[str, Path],
        float_format: str = ".4f",
    ):
        """Initialise the class.

        Parameters
        ----------
        data : np.ndarray
            2D Numpy array containing the flattened afm image. Data in this 2D array is floating point.
        labelled_data: np.ndarray
            2D Numpy array containing all the grain masks in the image. Data in this 2D array is boolean.
        pixel_to_nanometre_scaling: float
            Floating point value that defines the scaling factor between nanometres and pixels.
        output_dir : Path
            Path to the folder that will store the grain stats output images and data.
        float_format: str
            Formatter for saving floats.
        """

        self.data = data
        self.labelled_data = labelled_data
        self.pixel_to_nanometre_scaling = pixel_to_nanometre_scaling
        self.img_name = img_name
        self.output_dir = Path(output_dir)
        self.float_format = float_format
        self.start_point = None
        # Calculate grain stats.
        # self.calculate_stats()

    def get_angle(self, point_1: tuple, point_2: tuple) -> float:
        """Function that calculates the angle in radians between two points.

        Parameters
        ----------
        point1: tuple
            Coordinate vectors for the first point to find the angle between.
        point2: tuple
            Coordinate vectors for the second point to find the angle between.

        Returns
        -------
        angle : float
            The angle in radians between the two input vectors.
        """
        #        print(f'point_1    : {point_1}')
        #        print(f'point_2    : {point_2}')
        return np.arctan2(point_1[1] - point_2[1], point_1[0] - point_2[0])

    def is_clockwise(self, p1: tuple, p2: tuple, p3: tuple) -> bool:
        """Function to determine if three points make a clockwise or counter-clockwise turn.

        Parameters
        ----------
        p1: tuple
            First point to be used to calculate turn.
        p2: tuple
            Second point to be used to calculate turn.
        p3: tuple
            Third point to be used to calculate turn.

        Returns
        -------
        boolean
            Indicator of whether turn is clockwise.
        """
        # Determine if three points form a clockwise or counter-clockwise turn.
        # I use the method of calculating the determinant of the following rotation matrix here. If the determinant
        # is > 0 then the rotation is counter-clockwise.
        M = np.array(((p1[0], p1[1], 1), (p2[0], p2[1], 1), (p3[0], p3[1], 1)))
        return False if np.linalg.det(M) > 0 else True

    def calculate_stats(self):
        """Calculate the stats of grains in the labelled image"""

        # Calculate region properties
        region_properties = skimage_measure.regionprops(self.labelled_data)

        # Plot labelled data
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(self.labelled_data, interpolation="nearest", cmap="afmhot")

        # Iterate over all the grains in the image
        stats_array = []
        for index, region in enumerate(region_properties):

            # Create directory for each grain's plots
            output_grain = self.output_dir / f"grain_{index}"
            # Path.mkdir(output_grain, parents=True, exist_ok=True)
            output_grain.mkdir(parents=True, exist_ok=True)

            # Obtain and plot the cropped grain mask
            grain_mask = np.array(region.image)
            plot_and_save(grain_mask, output_grain / "grainmask.png")
            LOGGER.info(
                f'[{self.img_name}] Grain {index} : cropped image saved : {str(output_grain / "grainmask.png")}'
            )

            # Obtain the cropped grain image
            minr, minc, maxr, maxc = region.bbox
            grain_image = self.data[minr:maxr, minc:maxc]
            plot_and_save(grain_image, output_grain / "grain_image_raw.png")
            LOGGER.info(
                f'[{self.img_name}] Grain {index} : cropped image saved : {str(output_grain / "grain_image_raw.png")}'
            )
            grain_image = np.ma.masked_array(grain_image, mask=np.invert(grain_mask), fill_value=np.nan).filled()
            plot_and_save(grain_image, output_grain / "grain_image.png")
            LOGGER.info(
                f'[{self.img_name}] Grain {index} : cropped image saved : {str(output_grain / "grain_image.png")}'
            )

            edges = self.calculate_edges(grain_mask)
            # print(f'type(edges)   :  {type(edges)}')
            # print(f'edges         :\n{edges}')
            radius_stats = self.calculate_radius_stats(edges)
            hull, hull_indices, hull_simplexes = self.convex_hull(grain_mask, edges, output_grain)
            smallest_bounding_width, smallest_bounding_length, aspect_ratio = self.calculate_aspect_ratio(
                edges, hull, hull_indices, hull_simplexes, output_grain
            )

            # save_format = '.4f'

            # Save the stats to csv file. Note that many of the stats are multiplied by a scaling factor to convert
            # from pixel units to nanometres.
            # Removed formatting, better to keep accurate until the end, including in CSV, then shorten display
            stats = {
                "radius_min": radius_stats["min"] * self.pixel_to_nanometre_scaling,
                "radius_max": radius_stats["max"] * self.pixel_to_nanometre_scaling,
                "radius_mean": radius_stats["mean"] * self.pixel_to_nanometre_scaling,
                "radius_median": radius_stats["median"] * self.pixel_to_nanometre_scaling,
                "height_min": np.nanmin(grain_image),
                "height_max": np.nanmax(grain_image),
                "height_median": np.nanmedian(grain_image),
                "height_mean": np.nanmean(grain_image),
                "volume": np.nansum(grain_image) * self.pixel_to_nanometre_scaling**2,
                "area": region.area * self.pixel_to_nanometre_scaling**2,
                "area_cartesian_bbox": region.area_bbox * self.pixel_to_nanometre_scaling**2,
                "smallest_bounding_width": smallest_bounding_width * self.pixel_to_nanometre_scaling,
                "smallest_bounding_length": smallest_bounding_length * self.pixel_to_nanometre_scaling,
                "smallest_bounding_area": smallest_bounding_length
                * smallest_bounding_width
                * self.pixel_to_nanometre_scaling**2,
                "aspect_ratio": aspect_ratio,
            }
            # print(f'[{index}] statistics :\\n{stats}')
            stats_array.append(stats)

            # Add cartesian bounding box for the grain to the labelled image
            min_row, min_col, max_row, max_col = region.bbox
            rectangle = mpl.patches.Rectangle(
                (min_col, min_row), max_col - min_col, max_row - min_row, fill=False, edgecolor="white", linewidth=2
            )
            ax.add_patch(rectangle)

        # Create pandas dataframe to hold the stats and save it to a csv file.
        grainstats = pd.DataFrame(data=stats_array)
        grainstats.to_csv(self.output_dir / "grainstats.csv", float_format=self.float_format)

        # Save and close the plot
        plt.savefig(self.output_dir / "labelled_image_bboxes.png")
        plt.close()

        return grainstats

    def calculate_edges(self, grain_mask: np.ndarray):
        """Class method that takes a 2D boolean numpy array image of a grain and returns a python list of the
        coordinates of the edges of the grain.

        Parameters
        ----------
        grain_mask : np.ndarray
            A 2D numpy array image of a grain. Data in the array must be boolean.

        Returns
        -------
        edges : list
            List containing the coordinates of the edges of the grain.
        """
        # Fill any holes
        filled_grain_mask = scipy.ndimage.binary_fill_holes(grain_mask)
        # Get outer edge using canny filtering
        edges = skimage_feature.canny(filled_grain_mask, sigma=3)
        nonzero_coordinates = edges.nonzero()
        # Get vector representation of the points
        # FIXME : Switched to list comprehension but should be unnecessary to create this as a list as we can use
        # np.stack() to combine the arrays and use that...
        # return np.stack(nonzero_coordinates, axis=1)
        # edges = []
        # for vector in np.transpose(nonzero_coordinates):
        #     edges.append(list(vector))
        # return edges
        return [list(vector) for vector in np.transpose(nonzero_coordinates)]

    def calculate_radius_stats(self, edges: list) -> Tuple:
        """Class method that calculates the statistics relating to the radius. The radius in this context
        is the distance from the centroid to points on the edge of the grain.

        Parameters
        ----------
        edges: list
            A 2D python list containing the coordinates of the edges of a grain.

        Returns
        -------
        Tuple[float]
            A tuple of the minimum, maximum, mean and median radius of the grain
        """
        # Convert the edges to the form of a numpy array
        # edges = np.array(edges)
        # Calculate the centroid of the grain
        centroid = self._calculate_centroid(edges)
        # Calculate the displacement
        displacements = self._calculate_displacement(edges, centroid)
        # Calculate the radius of each point
        radii = self._calculate_radius(displacements)
        return {"min": np.min(radii), "max": np.max(radii), "mean": np.mean(radii), "median": np.median(radii)}

    def _calculate_centroid(self, edges: np.array) -> tuple:
        """Calculate the centroid of a bounding box.

        Parameters
        ----------
        edges: list
            A 2D python list containing the co-ordinates of the edges of a grain.

        Returns
        -------
        tuple
            The co-ordinates of the centroid.
        """
        # FIXME : Remove once we have a numpy array returned by calculate_edges
        edges = np.array(edges)
        return (sum(edges[:, 0]) / len(edges), sum(edges[:, 1] / len(edges)))

    def _calculate_displacement(self, edges: np.array, centroid: tuple) -> np.array:
        """Calculate the displacement between the centroid and edges"""
        # FIXME : Remove once we have a numpy array returned by calculate_edges
        return np.array(edges) - centroid

    def _calculate_radius(self, displacements) -> np.array:
        """Calculate the radius of each point from the centroid

        Parameters
        ----------
        displacements: List[list]

        Retrurns
        --------
        np.array
        """
        return np.array([np.sqrt(radius[0] ** 2 + radius[1] ** 2) for radius in displacements])

    def convex_hull(self, grain_mask: np.ndarray, edges: list, output_dir: Path, debug: bool = False):
        """Class method that takes a grain mask and the edges of the grain and returns the grain's convex hull. Based
        off of the Graham Scan algorithm and should ideally scale in time with O(nlog(n)).

        Parameters
        ----------
        grain_mask : np.ndarray
            A 2D numpy array containing the boolean grain mask for the grain.
        edges : list
            A python list contianing the coordinates of the edges of the grain.
        output_dir : Union[str, Path]
            Directory to save output to.
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
        hull, hull_indices, simplexes = self.graham_scan(edges)

        # Debug information
        if debug:
            self.plot(edges, hull, output_dir / "_points_hull.png")
            print(f"points: {edges}")
            print(f"hull: {hull}")
            print(f"hull indexes: {hull_indices}")
            print(f"simplexes: {simplexes}")

        return hull, hull_indices, simplexes

    def get_displacement(self, point_2: tuple, point_1: tuple = None) -> float:
        """Function that calculates the distance squared between two points. Used for distance sorting purposes and
        therefore does not perform a square root in the interests of efficiency.

        Parameters
        ----------
        point_2 : tuple
            The point to find the squared distance to.
        point_1 : tuple
            Optional - defaults to the starting point defined in the graham_scan() function. The point to find the
        squared distance from.

        Returns
        -------
        distance_squared : float
            The squared distance between the two points.
        """
        # Get the distance squared between two points. If the second point is not provided, use the starting point.
        point_1 = self.start_point if point_1 == None else point_1
        delta_x = point_2[0] - point_1[0]
        delta_y = point_2[1] - point_1[1]
        # Don't need the sqrt since just sorting for dist
        return float(delta_x**2 + delta_y**2)

    def sort_points(self, points: List) -> List:
        #    def sort_points(self, points: np.array) -> List:
        """Function to sort points in counter-clockwise order of angle made with the starting point.

        Parameters
        ----------
        points: list
            A python list of the coordinates to sort.

        Returns
        -------
        sorted_points : list
            A python list of sorted points.
        """
        # Return if the list is length 1 or 0 (i.e. a single point).
        if len(points) <= 1:
            return points
        # Lists that allow sorting of points relative to a current comparision point
        smaller, equal, larger = [], [], []
        # Get a random point in the array to calculate the pivot angle from. This sorts the points relative to this point.
        pivot_angle = self.get_angle(points[randint(0, len(points) - 1)], self.start_point)
        for point in points:
            point_angle = self.get_angle(point, self.start_point)
            # If the
            if point_angle < pivot_angle:
                smaller.append(point)
            elif point_angle == pivot_angle:
                equal.append(point)
            else:
                larger.append(point)
        # Lets take a different approach and use arrays, we have a start point lets work out the angle of each point
        # relative to that and _then_ sort it.
        # pivot_angles = self.get_angle(points, self.start_point)
        # Recursively sort the arrays until each point is sorted
        sorted_points = self.sort_points(smaller) + sorted(equal, key=self.get_displacement) + self.sort_points(larger)
        # Return sorted array where equal angle points are sorted by distance
        return sorted_points

    def get_start_point(self, edges) -> int:
        """Determine the index of the bottom most point of the hull when sorted by x-position.

        Parameters
        ----------
        edges: np.array

        Returns
        -------
        """
        min_y_index = np.argmin(edges[:, 1])
        self.start_point = edges[min_y_index]

    def graham_scan(self, edges: list):
        """A function based on the Graham Scan algorithm that constructs a convex hull from points in 2D cartesian
        space. Ideally this algorithm will take O( n * log(n) ) time.

        Parameters
        ----------
        edges : list
            A python list of coordinates that make up the edges of the grain.

        Returns
        -------
        hull : list
            A list containing coordinates of the points in the hull.
        hull_indices : list
            A list containing the hull points indices inside the edges list. In other words, this provides a way to find
            the points from the hull inside the edges list that was passed.
        simplices : list
            A  list of tuples, each tuple representing a simplex of the convex hull. These simplices are sorted such
            that they follow each other in counterclockwise order.
        """
        # FIXME : Make this an isolated method
        # Find a point guaranteed to be on the hull. I find the bottom most point(s) and sort by x-position.
        min_y_index = None
        for index, point in enumerate(edges):
            if min_y_index == None or point[1] < edges[min_y_index][1]:
                min_y_index = index
            if point[1] == edges[min_y_index][1] and point[0] < edges[min_y_index][0]:
                min_y_index = index
        self.start_point = edges[min_y_index]
        # This does the same thing, but as a separate method and with Numpy Array rather than a lsit
        # self.get_start_point(edges)
        # Sort the points
        points_sorted_by_angle = self.sort_points(edges)

        # Remove starting point from the list so it's not added more than once to the hull
        start_point_index = points_sorted_by_angle.index(self.start_point)
        del points_sorted_by_angle[start_point_index]
        # Add start point and the first point sorted by angle. Both of these points will always be on the hull.
        hull = [self.start_point, points_sorted_by_angle[0]]

        # Iterate through each point, checking if this point would cause a clockwise rotation if added to the hull, and
        # if so, backtracking.
        for index, point in enumerate(points_sorted_by_angle[1:]):
            # Determine if the proposed point demands a clockwise rotation
            while self.is_clockwise(hull[-2], hull[-1], point) == True:
                # Delete the failed point
                del hull[-1]
                if len(hull) < 2:
                    break
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

    def plot(self, edges: list, convex_hull: list = None, file_path: Path = None):
        """A function that plots and saves the coordinates of the edges in the grain and optionally the hull. The
        plot is saved as the file name that is provided.

        Parameters
        ----------
        coordinates : list
            A list of points to be plotted.
        convex_hull : list
            Optional argument. A list of points that form the convex hull. Will be plotted with the coordinates if
            provided.
        file_path : Path
            Path of the file to save the plot as.
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
                plt.plot((point1[0], point2[0]), (point1[1], point2[1]), "#994400")
        plt.savefig(file_path)
        plt.close()

    def calculate_aspect_ratio(
        self,
        edges: list,
        convex_hull: np.ndarray,
        hull_indices: np.ndarray,
        hull_simplices: np.ndarray,
        path: Path,
        debug: bool = False,
    ):
        """Class method that takes a list of edge points for a grain, and convex hull simplices and returns the width,
           length and aspect ratio of the smallest bounding rectangle for the grain.

        Parameters
        ----------
        edges : list
            A python list of coordinates of the edge of the grain.
        hull_simplices : np.ndarray
            A 2D numpy array of simplices that the hull is comprised of.
        path : Path
            Path to the save folder for the grain.
        debug : bool
            If true, various plots will be saved for diagnostic purposes.

        Returns
        -------
        smallest_bounding_width : float
            The width in pixels (not nanometres), of the smallest bounding rectangle for the grain.
        smallest_bounding_length : float
            The length in pixels (not nanometres), of the smallest bounding rectangle for the grain.
        aspect_ratio : float
            The width divided by the length of the smallest bounding rectangle for the grain. It will always be greater
            or equal to 1.
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

            # Map the coordinates such that the centroid is now centered on the origin. This is needed for the
            # matrix rotation step coming up.
            remapped_points = edges - centroid

            # Rotate the coordinates using a rotation matrix
            R = np.array(((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))))

            # For each point in the set, rotate it using the above rotation matrix.
            rotated_points = []
            for index, point in enumerate(remapped_points):
                newpoint = R @ point
                # FIXME : Can probably use np.append() here to append arrays directly, something like
                # np.append(rotated_points, newpoint, axis=0)
                rotated_points.append(newpoint)
            rotated_points = np.array(rotated_points)
            # Find the cartesian extremities
            x_min = np.min(rotated_points[:, 0])
            x_max = np.max(rotated_points[:, 0])
            y_min = np.min(rotated_points[:, 1])
            y_max = np.max(rotated_points[:, 1])

            if debug:
                # Create plot
                # FIXME : Make this a private method
                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111)

                # Draw the points and the current simplex that is being tested
                plt.scatter(x=remapped_points[:, 0], y=remapped_points[:, 1])
                plt.plot(remapped_points[simplex, 0], remapped_points[simplex, 1], "#444444", linewidth=4)
                plt.scatter(x=rotated_points[:, 0], y=rotated_points[:, 1])
                plt.plot(rotated_points[simplex, 0], rotated_points[simplex, 1], "k-", linewidth=5)
                print(rotated_points[simplex, 0], rotated_points[simplex, 1])

                # Draw the convex hulls
                for simplex in hull_simplices:
                    plt.plot(remapped_points[simplex, 0], remapped_points[simplex, 1], "#888888")
                    plt.plot(rotated_points[simplex, 0], rotated_points[simplex, 1], "#555555")

                # Draw bounding box
                plt.plot([x_min, x_min, x_max, x_max, x_min], [y_min, y_max, y_max, y_min, y_min], "#994400")
                plt.savefig(path / ("bounding_rectangle_construction_simplex_" + str(simplex_index) + ".png"))

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
            ([x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max])
        )
        translated_bounding_rectangle_vertices = []
        for index, point in enumerate(translated_rotated_bounding_rectangle_vertices):
            newpoint = RINVERSE @ point
            # FIXME : As above can likely use np.append(, axis=0) here
            translated_bounding_rectangle_vertices.append(newpoint)
        translated_bounding_rectangle_vertices = np.array(translated_bounding_rectangle_vertices)

        if debug:
            # Create plot
            # FIXME : Make this a private method
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111)
            plt.scatter(x=edges[:, 0], y=edges[:, 1])
            ax.plot(
                np.append(
                    translated_rotated_bounding_rectangle_vertices[:, 0],
                    translated_rotated_bounding_rectangle_vertices[0, 0],
                ),
                np.append(
                    translated_rotated_bounding_rectangle_vertices[:, 1],
                    translated_rotated_bounding_rectangle_vertices[0, 1],
                ),
                "#994400",
                label="rotated",
            )
            ax.plot(
                np.append(translated_bounding_rectangle_vertices[:, 0], translated_bounding_rectangle_vertices[0, 0]),
                np.append(translated_bounding_rectangle_vertices[:, 1], translated_bounding_rectangle_vertices[0, 1]),
                "#004499",
                label="unrotated",
            )
            ax.scatter(x=remapped_points[:, 0], y=remapped_points[:, 1], color="#004499", label="translated")
            ax.scatter(x=rotated_points[:, 0], y=rotated_points[:, 1], label="rotated")
            ax.legend()
            plt.savefig(path / "hull_bounding_rectangle_extra")

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        bounding_rectangle_vertices = translated_bounding_rectangle_vertices + centroid
        ax.plot(
            np.append(bounding_rectangle_vertices[:, 0], bounding_rectangle_vertices[0, 0]),
            np.append(bounding_rectangle_vertices[:, 1], bounding_rectangle_vertices[0, 1]),
            "#994400",
            label="unrotated",
        )
        ax.scatter(x=edges[:, 0], y=edges[:, 1], label="original points")
        ax.set_aspect(1)
        ax.legend()
        plt.savefig(path / "minimum_bbox.png")
        plt.close()

        return smallest_bounding_width, smallest_bounding_length, aspect_ratio
