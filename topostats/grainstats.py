"""Contains class for calculating the statistics of grains - 2d raster images"""
import logging
from pathlib import Path
from random import randint
from typing import Union, List, Tuple, Dict

import numpy as np
import skimage.measure as skimage_measure
import skimage.feature as skimage_feature
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

from topostats.plottingfuncs import plot_and_save
from topostats.logs.logs import LOGGER_NAME

# pylint: disable=line-too-long
# pylint: disable=fixme
# FIXME : The calculate_stats() and calculate_aspect_ratio() raise this error when linting, could consider putting
#         variables into dictionar, see example of breaking code out to staticmethod extremes() and returning a
#         dictionary of x_min/x_max/y_min/y_max
# pylint: disable=too-many-locals
# FIXME : calculate_aspect_ratio raises this error when linting it has 65 statements, recommended not to exceed 50.
#         Could break some out to small functions such as the lines that calculate the samllest bounding rectangle
# pylint: disable=too-many-statements

LOGGER = logging.getLogger(LOGGER_NAME)


GRAIN_STATS_COLUMNS = [
    "Molecule Number",
    "centre_x",
    "centre_y",
    "radius_min",
    "radius_max",
    "radius_mean",
    "radius_median",
    "height_min",
    "height_max",
    "height_median",
    "height_mean",
    "volume",
    "area",
    "area_cartesian_bbox",
    "smallest_bounding_width",
    "smallest_bounding_length",
    "smallest_bounding_area",
    "aspect_ratio",
]


class GrainStats:
    """Class for calculating grain stats."""

    def __init__(
        self,
        data: np.ndarray,
        labelled_data: np.ndarray,
        pixel_to_nanometre_scaling: float,
        direction: str,
        base_output_dir: Union[str, Path],
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
        """

        self.data = data
        self.labelled_data = labelled_data
        self.pixel_to_nanometre_scaling = pixel_to_nanometre_scaling
        self.direction = direction
        self.base_output_dir = Path(base_output_dir)
        self.start_point = None

    @staticmethod
    def get_angle(point_1: tuple, point_2: tuple) -> float:
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

        return np.arctan2(point_1[1] - point_2[1], point_1[0] - point_2[0])

    @staticmethod
    def is_clockwise(p_1: tuple, p_2: tuple, p_3: tuple) -> bool:
        """Function to determine if three points make a clockwise or counter-clockwise turn.

        Parameters
        ----------
        p_1: tuple
            First point to be used to calculate turn.
        p_2: tuple
            Second point to be used to calculate turn.
        p_3: tuple
            Third point to be used to calculate turn.

        Returns
        -------
        boolean
            Indicator of whether turn is clockwise.
        """
        # Determine if three points form a clockwise or counter-clockwise turn.
        # I use the method of calculating the determinant of the following rotation matrix here. If the determinant
        # is > 0 then the rotation is counter-clockwise.
        rotation_matrix = np.array(((p_1[0], p_1[1], 1), (p_2[0], p_2[1], 1), (p_3[0], p_3[1], 1)))
        return not np.linalg.det(rotation_matrix) > 0

    def calculate_stats(self) -> Dict:
        """Calculate the stats of grains in the labelled image"""

        if self.labelled_data is None:
            LOGGER.info(
                f"[{self.img_name}] : No labelled regions for this image, grain statistics can not be calculated."
            )
            return {"statistics": pd.DataFrame(columns=GRAIN_STATS_COLUMNS), "plot": None}

        # Calculate region properties
        region_properties = skimage_measure.regionprops(self.labelled_data)

        # Plot labelled data
        _, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(self.labelled_data, interpolation="nearest", cmap="afmhot")

        # Iterate over all the grains in the image
        stats_array = []
        for index, region in enumerate(region_properties):

            # FIXME : Get [{self.image_name}] included in LOGGER
            LOGGER.info(f" Processing grain: {index}")
            # Create directory for each grain's plots
            output_grain = self.base_output_dir / self.direction / f"grain_{index}"
            # Path.mkdir(output_grain, parents=True, exist_ok=True)
            output_grain.mkdir(parents=True, exist_ok=True)

            # Obtain and plot the cropped grain mask
            grain_mask = np.array(region.image)
            plot_and_save(grain_mask, output_grain, "grainmask.png")
            print("saved grain mask", index)
            print("grain mask shape: ", grain_mask.shape)

            # Obtain the cropped grain image
            minr, minc, maxr, maxc = region.bbox
            grain_image = self.data[minr:maxr, minc:maxc]
            plot_and_save(grain_image, output_grain, "grain_image_raw.png")
            grain_image = np.ma.masked_array(grain_image, mask=np.invert(grain_mask), fill_value=np.nan).filled()
            plot_and_save(grain_image, output_grain, "grain_image.png")

            LOGGER.info(f"saved grain image: {index}")

            points = self.calculate_points(grain_mask)
            print("points size: " + str(len(points[0])) + " " + str(len(points[1])))
            edges = self.calculate_edges(grain_mask)
            print("edges size: " + str(len(edges[0])) + " " + str(len(edges[1])))
            radius_stats = self.calculate_radius_stats(edges, points)
            # hull, hull_indices, hull_simplexes = self.convex_hull(edges, output_grain)
            _, _, hull_simplexes = self.convex_hull(edges, output_grain)
            centroid = self._calculate_centroid(points)
            # Centroids for the grains (minc and minr added because centroid returns values local to the cropped grain images)
            centre_x = centroid[0] + minc
            centre_y = centroid[1] + minr
            (smallest_bounding_width, smallest_bounding_length, aspect_ratio,) = self.calculate_aspect_ratio(
                edges=edges,
                hull_simplices=hull_simplexes,
                path=output_grain,
            )

            # save_format = '.4f'

            # Save the stats to csv file. Note that many of the stats are multiplied by a scaling factor to convert
            # from pixel units to nanometres.
            # Removed formatting, better to keep accurate until the end, including in CSV, then shorten display
            stats = {
                "centre_x": centre_x * self.pixel_to_nanometre_scaling,
                "centre_y": centre_y * self.pixel_to_nanometre_scaling,
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

            stats_array.append(stats)

            # Add cartesian bounding box for the grain to the labelled image
            min_row, min_col, max_row, max_col = region.bbox
            rectangle = mpl.patches.Rectangle(
                (min_col, min_row),
                max_col - min_col,
                max_row - min_row,
                fill=False,
                edgecolor="white",
                linewidth=2,
            )
            ax.add_patch(rectangle)

        Path.mkdir(self.base_output_dir / self.direction, exist_ok=True, parents=True)
        grainstats = pd.DataFrame(data=stats_array)
        grainstats.index.name = "Molecule Number"
        grainstats.to_csv(self.output_dir / self.direction / "grainstats.csv")

        return {"statistics": grainstats, "plot": ax}

    # def save_plot(self, img: Axes, title: str = None, filename: str = "15-labelled_image_bboxes.png") -> None:
    #     """Save the image adding a title if specified."""
    #     title = title if title is not None else "Labelled Image with Bounding Boxes"
    #     plt.title(title)
    #     plt.savefig(self.output_dir / filename)
    #     plt.close()

    @staticmethod
    def calculate_points(grain_mask: np.ndarray):
        """Class method that takes a 2D boolean numpy array image of a grain and returns a list containing the
        co-ordinates of the points in the grain.


        Parameters
        ----------
        grain_mask : np.ndarray
            A 2D numpy array image of a grain. Data in the array must be boolean.

        Returns
        -------
        edges : list
            A python list containing the coordinates of the pixels in the grain."""
        print("calculating points : shape of grain_mask: ", grain_mask.shape)

        nonzero_coordinates = grain_mask.nonzero()
        points = []
        for point in np.transpose(nonzero_coordinates):
            points.append(list(point))

        return points

    @staticmethod
    def calculate_points(grain_mask: np.ndarray):
        """Class method that takes a 2D boolean numpy array image of a grain and returns a list containing the co-ordinates of the points in the grain.

        Parameters
        ----------
        grain_mask : np.ndarray
            A 2D numpy array image of a grain. Data in the array must be boolean.

        Returns
        -------
        points : list
            A python list containing the coordinates of the pixels in the grain."""

        nonzero_coordinates = grain_mask.nonzero()
        points = []
        for point in np.transpose(nonzero_coordinates):
            points.append(list(point))

        return points

    @staticmethod
    def calculate_edges(grain_mask: np.ndarray):
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

    def calculate_radius_stats(self, edges: list, points: list) -> Tuple:
        """Class method that calculates the statistics relating to the radius. The radius in this context
        is the distance from the centroid to points on the edge of the grain.

        Parameters
        ----------
        edges: list
            A 2D python list containing the coordinates of the edges of a grain.
        points: list
            A 2D python list containing the coordinates of the points in a grain.

        Returns
        -------
        Tuple[float]
            A tuple of the minimum, maximum, mean and median radius of the grain
        """
        # Calculate the centroid of the grain
        centroid = self._calculate_centroid(points)
        # Calculate the displacement
        displacements = self._calculate_displacement(edges, centroid)
        # Calculate the radius of each point
        radii = self._calculate_radius(displacements)
        return {
            "min": np.min(radii),
            "max": np.max(radii),
            "mean": np.mean(radii),
            "median": np.median(radii),
        }

    @staticmethod
    def _calculate_centroid(points: np.array) -> tuple:
        """Calculate the centroid of a bounding box.

        Parameters
        ----------
        points: list
            A 2D python list containing the co-ordinates of the points in a grain.

        Returns
        -------
        tuple
            The co-ordinates of the centroid.
        """
        # FIXME : Remove once we have a numpy array returned by calculate_edges
        points = np.array(points)
        return (np.mean(points[:, 0]), np.mean(points[:, 1]))

    @staticmethod
    def _calculate_displacement(edges: np.array, centroid: tuple) -> np.array:
        """Calculate the displacement between the centroid and edges"""
        # FIXME : Remove once we have a numpy array returned by calculate_edges
        return np.array(edges) - centroid

    @staticmethod
    def _calculate_radius(displacements) -> np.array:
        """Calculate the radius of each point from the centroid

        Parameters
        ----------
        displacements: List[list]

        Retrurns
        --------
        np.array
        """
        return np.array([np.sqrt(radius[0] ** 2 + radius[1] ** 2) for radius in displacements])

    def convex_hull(self, edges: list, output_dir: Path, debug: bool = False):
        """Class method that takes a grain mask and the edges of the grain and returns the grain's convex hull. Based
        off of the Graham Scan algorithm and should ideally scale in time with O(nlog(n)).

        Parameters
        ----------
        edges : list
            A python list contianing the coordinates of the edges of the grain.
        output_dir : Union[str, Path]
            Directory to save output to.
        debug : bool
            Default false. If true, debug information will be displayed to the terminal and plots for the convex hulls and edges will be saved.

        Returns
        -------
        hull : list
            Coordinates of the points in the hull.
        hull_indices : list
            The hull points indices inside the edges list. In other words, this provides a way to find the points from the hull inside the edges list that was passed.
        simplices : list
            List of tuples, each tuple representing a simplex of the convex hull. These simplices are sorted such that they follow each other in counterclockwise order.
        """
        hull, hull_indices, simplexes = self.graham_scan(edges)

        # Debug information
        if debug:
            self.plot(edges, hull, output_dir / "_points_hull.png")
            LOGGER.info(f"points: {edges}")
            LOGGER.info(f"hull: {hull}")
            LOGGER.info(f"hull indexes: {hull_indices}")
            LOGGER.info(f"simplexes: {simplexes}")

        return hull, hull_indices, simplexes

    def calculate_squared_distance(self, point_2: tuple, point_1: tuple = None) -> float:
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
        point_1 = self.start_point if point_1 is None else point_1
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
        sorted_points = (
            self.sort_points(smaller) + sorted(equal, key=self.calculate_squared_distance) + self.sort_points(larger)
        )
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
            if min_y_index is None or point[1] < edges[min_y_index][1]:
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
            while self.is_clockwise(hull[-2], hull[-1], point) is True:
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
        for index, value in enumerate(hull_indices):
            simplices.append((hull_indices[index - 1], value))

        return hull, hull_indices, simplices

    @staticmethod
    def plot(edges: list, convex_hull: list = None, file_path: Path = None):
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

        _, ax = plt.subplots(1, 1, figsize=(8, 8))
        x_s, y_s = zip(*edges)
        ax.scatter(x_s, y_s)
        if convex_hull is not None:
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

    def calculate_aspect_ratio(self, edges: list, hull_simplices: np.ndarray, path: Path, debug: bool = False) -> tuple:
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
        # FIXME : pylint complains that this is unused which looks like a false positive to me as it is used.
        #         Probably does not need initiating here though (and code runs fine when doing so)
        # smallest_bounding_rectangle = None

        # Iterate through the simplices
        for simplex_index, simplex in enumerate(hull_simplices):
            p_1 = edges[simplex[0]]
            p_2 = edges[simplex[1]]
            delta = p_1 - p_2
            angle = np.arctan2(delta[0], delta[1])

            # Find the centroid of the points
            centroid = (sum(edges[:, 0]) / len(edges), sum(edges[:, 1] / len(edges)))

            # Map the coordinates such that the centroid is now centered on the origin. This is needed for the
            # matrix rotation step coming up.
            remapped_points = edges - centroid

            # Rotate the coordinates using a rotation matrix
            rotated_coordinates = np.array(((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))))

            # For each point in the set, rotate it using the above rotation matrix.
            rotated_points = []
            for _, point in enumerate(remapped_points):
                newpoint = rotated_coordinates @ point
                # FIXME : Can probably use np.append() here to append arrays directly, something like
                # np.append(rotated_points, newpoint, axis=0) but doing so requires other areas to be modified
                rotated_points.append(newpoint)
            rotated_points = np.array(rotated_points)
            # Find the cartesian extremities
            extremes = self.find_cartesian_extremes(rotated_points)

            if debug:
                # Create plot
                # FIXME : Make this a method
                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111)

                # Draw the points and the current simplex that is being tested
                plt.scatter(x=remapped_points[:, 0], y=remapped_points[:, 1])
                plt.plot(
                    remapped_points[simplex, 0],
                    remapped_points[simplex, 1],
                    "#444444",
                    linewidth=4,
                )
                plt.scatter(x=rotated_points[:, 0], y=rotated_points[:, 1])
                plt.plot(
                    rotated_points[simplex, 0],
                    rotated_points[simplex, 1],
                    "k-",
                    linewidth=5,
                )
                LOGGER.info(rotated_points[simplex, 0], rotated_points[simplex, 1])

                # Draw the convex hulls
                for simplex in hull_simplices:
                    plt.plot(
                        remapped_points[simplex, 0],
                        remapped_points[simplex, 1],
                        "#888888",
                    )
                    plt.plot(
                        rotated_points[simplex, 0],
                        rotated_points[simplex, 1],
                        "#555555",
                    )

                # Draw bounding box
                plt.plot(
                    [
                        extremes["x_min"],
                        extremes["x_min"],
                        extremes["x_max"],
                        extremes["x_max"],
                        extremes["x_min"],
                    ],
                    [
                        extremes["y_min"],
                        extremes["y_max"],
                        extremes["y_max"],
                        extremes["y_min"],
                        extremes["y_min"],
                    ],
                    "#994400",
                )
                plt.savefig(path / ("bounding_rectangle_construction_simplex_" + str(simplex_index) + ".png"))

            # Calculate the area of the proposed bounding rectangle
            bounding_area = (extremes["x_max"] - extremes["x_min"]) * (extremes["y_max"] - extremes["y_min"])

            # If current bounding rectangle is the smallest so far
            if smallest_bounding_area is None or bounding_area < smallest_bounding_area:
                smallest_bounding_area = bounding_area
                # smallest_bounding_rectangle = (
                #     extremes["x_min"],
                #     extremes["x_max"],
                #     extremes["y_min"],
                #     extremes["y_max"],
                # )
                aspect_ratio = (extremes["x_max"] - extremes["x_min"]) / (extremes["y_max"] - extremes["y_min"])
                smallest_bounding_width = min(
                    (extremes["x_max"] - extremes["x_min"]),
                    (extremes["y_max"] - extremes["y_min"]),
                )
                smallest_bounding_length = max(
                    (extremes["x_max"] - extremes["x_min"]),
                    (extremes["y_max"] - extremes["y_min"]),
                )
                # Enforce >= 1 aspect ratio
                if aspect_ratio < 1.0:
                    aspect_ratio = 1 / aspect_ratio

        # Unrotate the bounding box vertices
        r_inverse = rotated_coordinates.T
        translated_rotated_bounding_rectangle_vertices = np.array(
            (
                [extremes["x_min"], extremes["y_min"]],
                [extremes["x_max"], extremes["y_min"]],
                [extremes["x_max"], extremes["y_max"]],
                [extremes["x_min"], extremes["y_max"]],
            )
        )
        translated_bounding_rectangle_vertices = []
        for _, point in enumerate(translated_rotated_bounding_rectangle_vertices):
            newpoint = r_inverse @ point
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
                np.append(
                    translated_bounding_rectangle_vertices[:, 0],
                    translated_bounding_rectangle_vertices[0, 0],
                ),
                np.append(
                    translated_bounding_rectangle_vertices[:, 1],
                    translated_bounding_rectangle_vertices[0, 1],
                ),
                "#004499",
                label="unrotated",
            )
            ax.scatter(
                x=remapped_points[:, 0],
                y=remapped_points[:, 1],
                color="#004499",
                label="translated",
            )
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

    @staticmethod
    def find_cartesian_extremes(rotated_points: np.ndarray) -> Dict:
        """Find the limits of x and y of rotated points.

        Parameters
        ----------
        rotated_points: np.ndarray
            2-D array of rotated points.

        Returns
        -------
        Dict
            Dictionary of the x and y min and max.__annotations__
        """
        extremes = {}
        extremes["x_min"] = np.min(rotated_points[:, 0])
        extremes["x_max"] = np.max(rotated_points[:, 0])
        extremes["y_min"] = np.min(rotated_points[:, 1])
        extremes["y_max"] = np.max(rotated_points[:, 1])
        return extremes


def get_grainstats(
    data: np.ndarray,
    labelled_data: np.ndarray,
    pixel_to_nanometre_scaling: float,
    img_name: str,
    output_dir: Union[str, Path],
) -> Dict:
    """Wrapper function to instantiate a GrainStats() class and run it with the options on a single image.

    Parameters
    ----------
    data: np.ndarray
        2D Numpy image to be processed.
    labelled_data: np.ndarray
        2D Numpy image of labelled regions of data.
    pixel_to_nanometre_scaling: float
        Scaling of pixels to nanometres
    img_name: str
        Image being processed.
    output_dir: Union[str, Path]
        Output directory.

    Returns
    -------
    Dict
        Returns a dictionary with "statsitics" the calculated statistics for the grains and "ax" a plot image
        with bounding boxes.
    """
    return GrainStats(
        data=data,
        labelled_data=labelled_data,
        pixel_to_nanometre_scaling=pixel_to_nanometre_scaling,
        direction=img_name,
        base_output_dir=output_dir,
    ).calculate_stats()
