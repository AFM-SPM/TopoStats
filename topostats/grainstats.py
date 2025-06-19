"""Contains class for calculating the statistics of grains - 2d raster images."""

import logging
from pathlib import Path
from random import randint

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.ndimage
import skimage.feature as skimage_feature
import skimage.measure as skimage_measure
import skimage.morphology as skimage_morphology

from topostats.grains import GrainCrop
from topostats.logs.logs import LOGGER_NAME
from topostats.measure import feret, height_profiles
from topostats.utils import create_empty_dataframe

# pylint: disable=too-many-lines
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-arguments
# pylint: disable=too-many-branches
# pylint: disable=line-too-long
# pylint: disable=fixme
# FIXME : The calculate_stats() and calculate_aspect_ratio() raise this error when linting, could consider putting
#         variables into dictionar, see example of breaking code out to staticmethod extremes() and returning a
#         dictionary of x_min/x_max/y_min/y_max
# pylint: disable=too-many-locals
# FIXME : calculate_aspect_ratio raises this error when linting it has 65 statements, recommended not to exceed 50.
#         Could break some out to small functions such as the lines that calculate the samllest bounding rectangle
# pylint: disable=too-many-statements
# pylint: disable=too-many-positional-arguments

LOGGER = logging.getLogger(LOGGER_NAME)


GRAIN_STATS_COLUMNS = [
    "grain_number",
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
    """
    Class for calculating grain stats.

    Parameters
    ----------
    grain_crops : dict[int, GrainCrop]
        Dictionary of GrainCrops to calculate stats for.
    direction : str
        Direction for which grains have been detected ("above" or "below").
    base_output_dir : Path
        Path to the folder that will store the grain stats output images and data.
    image_name : str
        The name of the file being processed.
    edge_detection_method : str
        Method used for detecting the edges of grain masks before calculating statistics on them.
        Do not change unless you know exactly what this is doing. Options: "binary_erosion", "canny".
    extract_height_profile : bool
        Extract the height profile.
    plot_opts : dict
        Plotting options dictionary for the cropped grains.
    metre_scaling_factor : float
        Multiplier to convert the current length scale to metres. Default: 1e-9 for the
        usual AFM length scale of nanometres.
    """

    def __init__(
        self,
        grain_crops: dict[int, GrainCrop],
        direction: str,
        base_output_dir: str | Path,
        image_name: str = None,
        edge_detection_method: str = "binary_erosion",
        extract_height_profile: bool = False,
        plot_opts: dict = None,
        metre_scaling_factor: float = 1e-9,
    ):
        """
        Initialise the class.

        Parameters
        ----------
        grain_crops : dict[int, GrainCrop]
            Dictionary of GrainCrops to calculate stats for.
        direction : str
            Direction for which grains have been detected ("above" or "below").
        base_output_dir : Path
            Path to the folder that will store the grain stats output images and data.
        image_name : str
            The name of the file being processed.
        edge_detection_method : str
            Method used for detecting the edges of grain masks before calculating statistics on them.
            Do not change unless you know exactly what this is doing. Options: "binary_erosion", "canny".
        extract_height_profile : bool
            Extract the height profile.
        plot_opts : dict
            Plotting options dictionary for the cropped grains.
        metre_scaling_factor : float
            Multiplier to convert the current length scale to metres. Default: 1e-9 for the
            usual AFM length scale of nanometres.
        """
        self.grain_crops = grain_crops
        self.direction = direction
        self.base_output_dir = Path(base_output_dir)
        self.start_point = None
        self.image_name = image_name
        self.edge_detection_method = edge_detection_method
        self.extract_height_profile = extract_height_profile
        self.plot_opts = plot_opts
        self.metre_scaling_factor = metre_scaling_factor

    @staticmethod
    def get_angle(point_1: tuple, point_2: tuple) -> float:
        """
        Calculate the angle in radians between two points.

        Parameters
        ----------
        point_1 : tuple
            Coordinate vectors for the first point to find the angle between.
        point_2 : tuple
            Coordinate vectors for the second point to find the angle between.

        Returns
        -------
        float
            The angle in radians between the two input vectors.
        """
        return np.arctan2(point_1[1] - point_2[1], point_1[0] - point_2[0])

    @staticmethod
    def is_clockwise(p_1: tuple, p_2: tuple, p_3: tuple) -> bool:
        """
        Determine if three points make a clockwise or counter-clockwise turn.

        Parameters
        ----------
        p_1 : tuple
            First point to be used to calculate turn.
        p_2 : tuple
            Second point to be used to calculate turn.
        p_3 : tuple
            Third point to be used to calculate turn.

        Returns
        -------
        boolean
            Indicator of whether turn is clockwise.
        """
        # Determine if three points form a clockwise or counter-clockwise turn.
        # I use the method of calculating the determinant of the following rotation matrix here. If the determinant
        # is > 0 then the rotation is counter-clockwise.
        rotation_matrix = np.asarray(((p_1[0], p_1[1], 1), (p_2[0], p_2[1], 1), (p_3[0], p_3[1], 1)))
        return not np.linalg.det(rotation_matrix) > 0

    def calculate_stats(self) -> tuple[pd.DataFrame, dict]:
        """
        Calculate the stats of grains in the labelled image.

        Returns
        -------
        tuple
            Consists of a pd.DataFrame containing all the grain stats that have been calculated for the labelled image
            and a list of dictionaries containing grain data to be plotted.
        """
        all_height_profiles: dict[int, npt.NDArray] = {}
        if len(self.grain_crops) == 0:
            LOGGER.warning(
                f"[{self.image_name}] : No grain crops for this image, grain statistics can not be calculated."
            )
            return pd.DataFrame(columns=GRAIN_STATS_COLUMNS), all_height_profiles

        grainstats_rows: list[dict] = []

        # Iterate over each grain
        for grain_index, grain_crop in self.grain_crops.items():
            LOGGER.debug(f"Processing grain {grain_index}")
            all_height_profiles[grain_index] = {}
            grain_crop.stats = {}
            grain_crop.height_profiles = {}
            image = grain_crop.image
            mask = grain_crop.mask
            grain_bbox = grain_crop.bbox
            grain_anchor = (grain_bbox[0], grain_bbox[1])
            pixel_to_nm_scaling = grain_crop.pixel_to_nm_scaling

            # Calculate scaling factors
            length_scaling_factor = pixel_to_nm_scaling * self.metre_scaling_factor
            area_scaling_factor = length_scaling_factor**2

            # Create directory for grain's plots
            output_grain = self.base_output_dir / self.direction / f"grain_{grain_index}"

            # Iterate over all the classes except background
            for class_index in range(1, mask.shape[2]):
                all_height_profiles[grain_index][class_index] = {}
                grain_crop.stats[class_index] = {}
                grain_crop.height_profiles[class_index] = {}
                class_mask = mask[:, :, class_index]
                labelled_class_mask = skimage_measure.label(class_mask)
                # Split the class into connected components
                class_mask_regionprops = skimage_measure.regionprops(labelled_class_mask)

                # Iterate over all the sub_grains in the class
                for subgrain_index, subgrain_region in enumerate(class_mask_regionprops):
                    # Remove all but the current subgrain from the mask
                    subgrain_only_mask = class_mask * (labelled_class_mask == subgrain_region.label)
                    # Create a masked image of the subgrain
                    subgrain_mask_image = np.ma.masked_array(
                        image, mask=np.invert(subgrain_only_mask), fill_value=np.nan
                    ).filled()
                    # Shape of the subgrain region with no padding and not necessarily square, more accurate measure of
                    # the bounding box size
                    subgrain_tight_shape = subgrain_region.image.shape
                    # Skip subgrain if too small to calculate stats for
                    if min(subgrain_tight_shape) < 5:
                        LOGGER.debug(
                            f"[{self.image_name}] : Skipping subgrain due to being too small "
                            f"(size: {subgrain_tight_shape}) to calculate stats for."
                        )

                    # Calculate all the stats
                    points = self.calculate_points(subgrain_only_mask)
                    edges = self.calculate_edges(subgrain_only_mask, edge_detection_method=self.edge_detection_method)
                    radius_stats = self.calculate_radius_stats(edges, points)
                    # hull, hull_indices, hull_simplexes = self.convex_hull(edges, output_grain)
                    _, _, hull_simplexes = self.convex_hull(edges, output_grain)
                    local_centroid = self._calculate_centroid(points)

                    # Centroids for the grains (grain anchor added because centroid returns values local to the
                    # cropped grain images)
                    centre_global_x_px = local_centroid[1] + grain_anchor[1]
                    centre_global_y_px = local_centroid[0] + grain_anchor[0]

                    centre_x_m = centre_global_x_px * length_scaling_factor
                    centre_y_m = centre_global_y_px * length_scaling_factor

                    (
                        smallest_bounding_width,
                        smallest_bounding_length,
                        aspect_ratio,
                    ) = self.calculate_aspect_ratio(
                        edges=edges,
                        hull_simplices=hull_simplexes,
                        path=output_grain,
                    )

                    # Calculate minimum and maximum feret diameters and scale the distances
                    feret_statistics = feret.min_max_feret(points)
                    feret_statistics["min_feret"] = feret_statistics["min_feret"] * length_scaling_factor
                    feret_statistics["max_feret"] = feret_statistics["max_feret"] * length_scaling_factor

                    if self.extract_height_profile:
                        _height_profiles = height_profiles.interpolate_height_profile(
                            img=image, mask=subgrain_only_mask
                        )
                        all_height_profiles[grain_index][class_index][subgrain_index] = _height_profiles
                        grain_crop.height_profiles[class_index][subgrain_index] = _height_profiles
                        LOGGER.debug(f"[{self.image_name}] : Height profiles extracted.")

                    # Save the stats to dictionary. Note that many of the stats are multiplied by a scaling factor to convert
                    # from pixel units to nanometres.
                    # Removed formatting, better to keep accurate until the end, including in CSV, then shorten display
                    stats = {
                        "grain_number": grain_index,
                        "class_number": class_index,
                        "subgrain_number": subgrain_index,
                        "centre_x": centre_x_m,
                        "centre_y": centre_y_m,
                        "radius_min": radius_stats["min"] * length_scaling_factor,
                        "radius_max": radius_stats["max"] * length_scaling_factor,
                        "radius_mean": radius_stats["mean"] * length_scaling_factor,
                        "radius_median": radius_stats["median"] * length_scaling_factor,
                        "height_min": np.nanmin(subgrain_mask_image) * self.metre_scaling_factor,
                        "height_max": np.nanmax(subgrain_mask_image) * self.metre_scaling_factor,
                        "height_median": np.nanmedian(subgrain_mask_image) * self.metre_scaling_factor,
                        "height_mean": np.nanmean(subgrain_mask_image) * self.metre_scaling_factor,
                        # [volume] = [pixel] * [pixel] * [height] = px * px * nm.
                        # To turn into m^3, multiply by pixel_to_nanometre_scaling^2 and metre_scaling_factor^3.
                        "volume": np.nansum(subgrain_mask_image)
                        * pixel_to_nm_scaling**2
                        * (self.metre_scaling_factor**3),
                        "area": subgrain_region.area * area_scaling_factor,
                        "area_cartesian_bbox": subgrain_region.area_bbox * area_scaling_factor,
                        "smallest_bounding_width": smallest_bounding_width * length_scaling_factor,
                        "smallest_bounding_length": smallest_bounding_length * length_scaling_factor,
                        "smallest_bounding_area": smallest_bounding_length
                        * smallest_bounding_width
                        * area_scaling_factor,
                        "aspect_ratio": aspect_ratio,
                        "threshold": self.direction,
                        "max_feret": feret_statistics["max_feret"],
                        "min_feret": feret_statistics["min_feret"],
                    }
                    grainstats_rows.append(stats)
                    # remove indexes from stats and save to nested stats object
                    _stats = stats.copy()
                    for x in ["grain_number", "class_number", "subgrain_number"]:
                        _stats.pop(x)
                    grain_crop.stats[class_index][subgrain_index] = _stats

        # Check if the dataframe is empty
        if len(grainstats_rows) == 0:
            grainstats_df = create_empty_dataframe()
        else:
            # Create a dataframe from the list of dictionaries
            grainstats_df = pd.DataFrame(grainstats_rows)
        # Change the index column from the arbitrary one to the grain number
        grainstats_df["image"] = self.image_name
        return (grainstats_df, all_height_profiles)

    @staticmethod
    def calculate_points(grain_mask: npt.NDArray) -> list:
        """
        Convert a 2D boolean array to a list of coordinates.

        Parameters
        ----------
        grain_mask : npt.NDArray
            A 2D numpy array image of a grain. Data in the array must be boolean.

        Returns
        -------
        list
            A python list containing the coordinates of the pixels in the grain.
        """
        nonzero_coordinates = grain_mask.nonzero()
        points = []
        for point in np.transpose(nonzero_coordinates):
            points.append(list(point))

        return points

    @staticmethod
    def calculate_edges(grain_mask: npt.NDArray, edge_detection_method: str) -> list:
        """
        Convert 2D boolean array to list of the coordinates of the edges of the grain.

        Parameters
        ----------
        grain_mask : npt.NDArray
            A 2D numpy array image of a grain. Data in the array must be boolean.
        edge_detection_method : str
            Method used for detecting the edges of grain masks before calculating statistics on them.
            Do not change unless you know exactly what this is doing. Options: "binary_erosion", "canny".

        Returns
        -------
        list
            List containing the coordinates of the edges of the grain.
        """
        # Fill any holes
        filled_grain_mask = scipy.ndimage.binary_fill_holes(grain_mask)

        if edge_detection_method == "binary_erosion":
            # Add padding (needed for erosion)
            padded = np.pad(filled_grain_mask, 1)
            # Erode by 1 pixel
            eroded = skimage_morphology.binary_erosion(padded)
            # Remove padding
            eroded = eroded[1:-1, 1:-1]

            # Edges is equal to the difference between the
            # original image and the eroded image.
            edges = filled_grain_mask.astype(int) - eroded.astype(int)
        else:
            # Get outer edge using canny filtering
            edges = skimage_feature.canny(filled_grain_mask, sigma=3)

        nonzero_coordinates = edges.nonzero()
        # Get vector representation of the points
        # FIXME : Switched to list comprehension but should be unnecessary to create this as a list as we can use
        # np.stack() to combine the arrays and use that...
        # return np.stack(nonzero_coordinates, axis=1)
        return [list(vector) for vector in np.transpose(nonzero_coordinates)]

    def calculate_radius_stats(self, edges: list, points: list) -> dict[str, float]:
        """
        Calculate the radius of grains.

        The radius in this context is the distance from the centroid to points on the edge of the grain.

        Parameters
        ----------
        edges : list
            A 2D python list containing the coordinates of the edges of a grain.
        points : list
            A 2D python list containing the coordinates of the points in a grain.

        Returns
        -------
        dict[str, float]
            A dictionary containing the minimum, maximum, mean and median radius of the grain.
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
            "mean": float(np.mean(radii)),
            "median": float(np.median(radii)),
        }

    @staticmethod
    def _calculate_centroid(points: np.array) -> tuple:
        """
        Calculate the centroid of a bounding box.

        Parameters
        ----------
        points : list
            A 2D python list containing the coordinates of the points in a grain.

        Returns
        -------
        tuple
            The coordinates of the centroid.
        """
        # FIXME : Remove once we have a numpy array returned by calculate_edges
        points = np.array(points)
        return (np.mean(points[:, 0]), np.mean(points[:, 1]))

    @staticmethod
    def _calculate_displacement(edges: npt.NDArray, centroid: tuple) -> npt.NDArray:
        """
        Calculate the displacement between the edges and centroid.

        Parameters
        ----------
        edges : npt.NDArray
            Coordinates of the edge points.
        centroid : tuple
            Coordinates of the centroid.

        Returns
        -------
        npt.NDArray
            Array of displacements.
        """
        # FIXME : Remove once we have a numpy array returned by calculate_edges
        return np.array(edges) - centroid

    @staticmethod
    def _calculate_radius(displacements: list[list]) -> npt.NDArray:
        """
        Calculate the radius of each point from the centroid.

        Parameters
        ----------
        displacements : List[list]
            A list of displacements.

        Returns
        -------
        npt.NDArray
            Array of radii of each point from the centroid.
        """
        return np.array([np.sqrt(radius[0] ** 2 + radius[1] ** 2) for radius in displacements])

    def convex_hull(self, edges: list, base_output_dir: Path, debug: bool = False) -> tuple[list, list, list]:
        """
        Calculate a grain's convex hull.

        Based off of the Graham Scan algorithm and should ideally scale in time with O(nlog(n)).

        Parameters
        ----------
        edges : list
            A python list containing the coordinates of the edges of the grain.
        base_output_dir : Path
            Directory to save output to.
        debug : bool
            Default false. If true, debug information will be displayed to the terminal and plots for the convex hulls
            and edges will be saved.

        Returns
        -------
        tuple[list, list, list]
            A hull (list) of the coordinates of each point on the hull. Hull indices providing a way to find the points
            from the hill inside the edge list that was passed. Simplices (list) of tuples each representing a simplex
            of the convex hull, these are sorted in a counter-clockwise order.
        """
        hull, hull_indices, simplexes = self.graham_scan(edges)

        # Debug information
        if debug:
            base_output_dir.mkdir(parents=True, exist_ok=True)
            self.plot(edges, hull, base_output_dir / "_points_hull.png")
            LOGGER.debug(f"points: {edges}")
            LOGGER.debug(f"hull: {hull}")
            LOGGER.debug(f"hull indexes: {hull_indices}")
            LOGGER.debug(f"simplexes: {simplexes}")

        return hull, hull_indices, simplexes

    def calculate_squared_distance(self, point_2: tuple, point_1: tuple = None) -> float:
        """
        Calculate the squared distance between two points.

        Used for distance sorting purposes and therefore does not perform a square root in the interests of efficiency.

        Parameters
        ----------
        point_2 : tuple
            The point to find the squared distance to.
        point_1 : tuple
            Optional - defaults to the starting point defined in the graham_scan() function. The point to find the
            squared distance from.

        Returns
        -------
        float
            The squared distance between the two points.
        """
        # Get the distance squared between two points. If the second point is not provided, use the starting point.
        point_1 = self.start_point if point_1 is None else point_1
        delta_x = point_2[0] - point_1[0]
        delta_y = point_2[1] - point_1[1]
        # Don't need the sqrt since just sorting for dist
        return float(delta_x**2 + delta_y**2)

    def sort_points(self, points: list) -> list:
        #    def sort_points(self, points: np.array) -> List:
        """
        Sort points in counter-clockwise order of angle made with the starting point.

        Parameters
        ----------
        points : list
            A python list of the coordinates to sort.

        Returns
        -------
        list
            Points (coordinates) sorted counter-clockwise.
        """
        # Return if the list is length 1 or 0 (i.e. a single point).
        if len(points) <= 1:
            return points
        # Lists that allow sorting of points relative to a current comparison point
        smaller, equal, larger = [], [], []
        # Get a random point in the array to calculate the pivot angle from. This sorts the points relative to this point.
        pivot_angle = self.get_angle(points[randint(0, len(points) - 1)], self.start_point)  # noqa: S311
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
        return self.sort_points(smaller) + sorted(equal, key=self.calculate_squared_distance) + self.sort_points(larger)
        # Return sorted array where equal angle points are sorted by distance

    def get_start_point(self, edges: npt.NDArray) -> None:
        """
        Determine the index of the bottom most point of the hull when sorted by x-position.

        Parameters
        ----------
        edges : npt.NDArray
            Array of coordinates.
        """
        min_y_index = np.argmin(edges[:, 1])
        self.start_point = edges[min_y_index]

    def graham_scan(self, edges: list) -> tuple[list, list, list]:
        """
        Construct the convex hull using the  Graham Scan algorithm.

        Ideally this algorithm will take O( n * log(n) ) time.

        Parameters
        ----------
        edges : list
            A python list of coordinates that make up the edges of the grain.

        Returns
        -------
        tuple[list, list, list]
            A hull (list) of the coordinates of each point on the hull. Hull indices providing a way to find the points
            from the hill inside the edge list that was passed. Simplices (list) of tuples each representing a simplex
            of the convex hull, these are sorted in a counter-clockwise order.
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
        # This does the same thing, but as a separate method and with Numpy Array rather than a list
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
        for _, point in enumerate(points_sorted_by_angle[1:]):
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
    def plot(edges: list, convex_hull: list = None, file_path: Path = None) -> None:
        """
        Plot and save the coordinates of the edges in the grain and optionally the hull.

        Parameters
        ----------
        edges : list
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

    def calculate_aspect_ratio(
        self, edges: list, hull_simplices: npt.NDArray, path: Path, debug: bool = False
    ) -> tuple:
        """
        Calculate the width, length and aspect ratio of the smallest bounding rectangle of a grain.

        Parameters
        ----------
        edges : list
            A python list of coordinates of the edge of the grain.
        hull_simplices : npt.NDArray
            A 2D numpy array of simplices that the hull is comprised of.
        path : Path
            Path to the save folder for the grain.
        debug : bool
            If true, various plots will be saved for diagnostic purposes.

        Returns
        -------
        tuple:
            The smallest_bouning_width (float) in pixels (not nanometres) of the smallest bounding rectangle for the
            grain. The smallest_bounding_length (float) in pixels (not nanometres), of the smallest bounding rectangle
            for the grain. And the aspect_ratio (float) the width divided by the length of the smallest bounding
            rectangle for the grain. It will always be greater or equal to 1.
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
                # Ensure directory is there
                path.mkdir(parents=True, exist_ok=True)

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
                LOGGER.debug(rotated_points[simplex, 0], rotated_points[simplex, 1])

                # Draw the convex hulls
                for _simplex in hull_simplices:
                    plt.plot(
                        remapped_points[_simplex, 0],
                        remapped_points[_simplex, 1],
                        "#888888",
                    )
                    plt.plot(
                        rotated_points[_simplex, 0],
                        rotated_points[_simplex, 1],
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
                smallest_bounding_width = min(
                    (extremes["x_max"] - extremes["x_min"]),
                    (extremes["y_max"] - extremes["y_min"]),
                )
                smallest_bounding_length = max(
                    (extremes["x_max"] - extremes["x_min"]),
                    (extremes["y_max"] - extremes["y_min"]),
                )

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
        plt.xlabel("Grain Length (nm)")
        plt.ylabel("Grain Width (nm)")
        # plt.savefig(path / "minimum_bbox.png")
        plt.close()

        return (
            smallest_bounding_width,  # pylint: disable=possibly-used-before-assignment
            smallest_bounding_length,  # pylint: disable=possibly-used-before-assignment
            smallest_bounding_width / smallest_bounding_length,  # pylint: disable=possibly-used-before-assignment
        )

    @staticmethod
    def find_cartesian_extremes(rotated_points: npt.NDArray) -> dict:
        """
        Find the limits of x and y of rotated points.

        Parameters
        ----------
        rotated_points : npt.NDArray
            2-D array of rotated points.

        Returns
        -------
        Dict
            Dictionary of the x and y min and max.__annotations__.
        """
        extremes = {}
        extremes["x_min"] = np.min(rotated_points[:, 0])
        extremes["x_max"] = np.max(rotated_points[:, 0])
        extremes["y_min"] = np.min(rotated_points[:, 1])
        extremes["y_max"] = np.max(rotated_points[:, 1])
        return extremes

    @staticmethod
    def get_shift(coords: npt.NDArray, shape: npt.NDArray) -> int:
        """
        Obtain the coordinate shift to reflect the cropped image box for molecules near the edges of the image.

        Parameters
        ----------
        coords : npt.NDArray
            Value representing integer coordinates which may be outside of the image.
        shape : npt.NDArray
            Array of the shape of an image.

        Returns
        -------
        np.int64
            Max value of the shift to reflect the croped region so it stays within the image.
        """
        shift = shape - coords[np.where(coords > shape)]
        shift = np.hstack((shift, -coords[np.where(coords < 0)]))
        if len(shift) == 0:
            return 0
        max_index = np.argmax(abs(shift))
        return shift[max_index]

    def get_cropped_region(self, image: npt.NDArray, length: int, centre: npt.NDArray) -> npt.NDArray:
        """
        Crop the image with respect to a given pixel length around the centre coordinates.

        Parameters
        ----------
        image : npt.NDArray
            The image array.
        length : int
            The length (in pixels) of the resultant cropped image.
        centre : npt.NDArray
            The centre of the object to crop.

        Returns
        -------
        npt.NDArray
            Cropped array of the image.
        """
        shape = image.shape
        xy1 = shape - (centre + length + 1)
        xy2 = shape - (centre - length)
        xy = np.stack((xy1, xy2))
        shiftx = self.get_shift(xy[:, 0], shape[0])
        shifty = self.get_shift(xy[:, 1], shape[1])
        return image.copy()[
            centre[0] - length - shiftx : centre[0] + length + 1 - shiftx,  # noqa: E203
            centre[1] - length - shifty : centre[1] + length + 1 - shifty,  # noqa: E203
        ]
