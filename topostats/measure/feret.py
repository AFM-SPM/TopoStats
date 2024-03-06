"""Calculate feret distances for 2-D objects.

This code comes from a gist written by @VolkerH under BSD-3 License

https://gist.github.com/VolkerH/0d07d05d5cb189b56362e8ee41882abf

During testing it was discovered that sorting points prior to derivation of upper and lower convex hulls was problematic
so this step was removed.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Generator
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import skimage.morphology
from shapely import LineString, Polygon, contains

from topostats.logs.logs import LOGGER_NAME

LOGGER = logging.getLogger(LOGGER_NAME)

# Handle warnings as exceptions (encountered when gradient of base triangle is zero)
warnings.filterwarnings("error")

# pylint: disable=fixme


def orientation(p: npt.NDArray, q: npt.NDArray, r: npt.NDArray) -> int:
    """
    Determine the orientation of three points as either clockwise, counter-clock-wise or colinear.

    Parameters
    ----------
    p : npt.NDArray
        First point (assumed to have a length of 2).
    q : npt.NDArray
        Second point (assumed to have a length of 2).
    r : npt.NDArray
        Third point (assumed to have a length of 2).

    Returns
    -------
    int:
        Returns a positive value if p-q-r are clockwise, neg if counter-clock-wise, zero if colinear.
    """
    return (q[1] - p[1]) * (r[0] - p[0]) - (q[0] - p[0]) * (r[1] - p[1])


def sort_coords(points: npt.NDArray, axis: int = 1) -> npt.NDArray:
    """
    Sort the coordinates.

    Parameters
    ----------
    points : npt.NDArray
        Array of coordinates
    axis : int
        Which axis to axis coordinates on 0 for row; 1 for columns (default).

    Returns
    -------
    npt.NDArray
        Array sorted by row then column.
    """
    if axis == 1:
        order = np.lexsort((points[:, 0], points[:, 1]))
    elif axis == 0:
        order = np.lexsort((points[:, 1], points[:, 0]))
    else:
        raise ValueError("Invalid axis provided for sorting, only 0 and 1 permitted.")
    return points[order]


def hulls(points: npt.NDArray, axis: int = 1) -> tuple[list, list]:
    """
    Graham scan to find upper and lower convex hulls of a set of 2-D points.

    Points should be sorted in asecnding order first.

    `Graham scan <https://en.wikipedia.org/wiki/Graham_scan>`

    Parameters
    ----------
    points : npt.NDArray
        2-D Array of points for the outline of an object.
    axis : int
        Which axis to sort coordinates on 0 for row; 1 for columns (default).

    Returns
    -------
    tuple[list, list]
        Tuple of two Numpy arrays of the original coordinates split into upper and lower hulls.
    """
    upper_hull: list = []
    lower_hull: list = []
    if axis:
        points = sort_coords(points, axis)
    for p in points:
        # Remove points if they are not in the correct hull
        while len(upper_hull) > 1 and orientation(upper_hull[-2], upper_hull[-1], p) <= 0:
            upper_hull.pop()
        while len(lower_hull) > 1 and orientation(lower_hull[-2], lower_hull[-1], p) >= 0:
            lower_hull.pop()
        # Append point to each hull (removed from one in next pass)
        upper_hull.append(list(p))
        lower_hull.append(list(p))

    return upper_hull, lower_hull


def all_pairs(points: npt.NDArray) -> list[tuple[list, list]]:
    """
    Given a list of 2-D points, finds all ways of sandwiching the points.

    Calculates the upper and lower convex hulls and then finds all pairwise combinations between each set of points.

    Parameters
    ----------
    points : npt.NDArray
        Numpy array of coordinates defining the outline of an object.mro

    Returns
    -------
    List[tuple[int, int]]
    """
    upper_hull, lower_hull = hulls(points)
    unique_combinations = {}
    for upper in upper_hull:
        for lower in lower_hull:
            if upper != lower:
                lowest = min(upper, lower)
                highest = max(upper, lower)
                unique_combinations[f"{str(lowest)}-{str(highest)}"] = (lowest, highest)
    return list(unique_combinations.values())


def rotating_calipers(points: npt.NDArray, axis: int = 0) -> Generator:
    """
    Given a list of 2-D points, finds all ways of sandwiching the points between two parallel lines.

    This yields the sequence of pairs of points touched by each pair of lines across all points around the hull of a
    polygon.

    `Rotating Calipers <https://en.wikipedia.org/wiki/Rotating_calipers>_`

    Parameters
    ----------
    points : npt.NDArray
        Numpy array of coordinates defining the outline of an object.
    axis : int
        Which axis to sort coordinates on, 0 for row (default); 1 for columns.

    Returns
    -------
    Generator
        Numpy array of pairs of points
    """
    upper_hull, lower_hull = hulls(points, axis)
    upper_index = 0
    lower_index = len(lower_hull) - 1
    counter = 0
    while upper_index < len(upper_hull) - 1 or lower_index > 0:
        # yield upper_hull[upper_index], lower_hull[lower_index]
        calipers = (lower_hull[lower_index], upper_hull[upper_index])
        if upper_index == len(upper_hull) - 1:
            lower_index -= 1
            base1 = lower_hull[lower_index + 1]  # original lower caliper
            base2 = lower_hull[lower_index]  # previous point on lower hull
            apex = upper_hull[upper_index]  # original upper caliper
        elif lower_index == 0:
            upper_index += 1
            base1 = upper_hull[upper_index - 1]  # original upper caliper
            base2 = upper_hull[upper_index]  # next point on upper hull
            apex = lower_hull[lower_index]  # original lower caliper
        # still points left on both lists, compare slopes of next hull edges
        # being careful to avoid ZeroDivisionError in slope calculation
        elif (
            (upper_hull[upper_index + 1][1] - upper_hull[upper_index][1])
            * (lower_hull[lower_index][0] - lower_hull[lower_index - 1][0])
        ) > (
            (lower_hull[lower_index][1] - lower_hull[lower_index - 1][1])
            * (upper_hull[upper_index + 1][0] - upper_hull[upper_index][0])
        ):
            upper_index += 1
            base1 = upper_hull[upper_index - 1]  # original upper caliper
            base2 = upper_hull[upper_index]  # next point on upper hull
            apex = lower_hull[lower_index]  # original lower caliper
        else:
            lower_index -= 1
            base1 = lower_hull[lower_index + 1]  # original lower caliper
            base2 = lower_hull[lower_index]  # previous point on lower hull
            apex = upper_hull[upper_index]  # original upper caliper
        counter += 1
        yield triangle_height(base1, base2, apex), calipers, np.asarray(
            [
                list(_min_feret_coord(np.asarray(base1), np.asarray(base2), np.asarray(apex))),
                apex,
            ]
        )


def triangle_height(base1: npt.NDArray | list, base2: npt.NDArray | list, apex: npt.NDArray | list) -> float:
    """
    Calculate the height of triangle formed by three points.

    Parameters
    ----------
    base1 : int
        Coordinate of first base point of triangle.
    base2 : int
        Coordinate of second base point of triangle.
    apex : int
        Coordinate of the apex of the triangle.

    Returns
    -------
    float
        Height of the triangle.

    Examples
    --------
    >>> min_feret([4, 0], [4, 3], [0,0])
        4.0
    """
    base1_base2 = np.asarray(base1) - np.asarray(base2)
    base1_apex = np.asarray(base1) - np.asarray(apex)
    return np.linalg.norm(np.cross(base1_base2, base1_apex)) / np.linalg.norm(base1_base2)


def _min_feret_coord(
    base1: npt.NDArray, base2: npt.NDArray, apex: npt.NDArray, round_coord: bool = False
) -> npt.NDArray:
    """
    Calculate the coordinate opposite the apex that is prependicular to the base of the triangle.

    Code courtesy of @SylviaWhittle.

    Parameters
    ----------
    base1 : npt.NDArray
        Coordinates of one point on base of triangle, these are on the same side of the hull.
    base2 : npt.NDArray
        Coordinates of second point on base of triangle, these are on the same side of the hull.
    apex : npt.NDArray
        Coordinate of the apex of the triangle, this is on the opposite hull.
    round_coord : bool
        Whether to round the point to the nearest NumPy index relative to the apex's position (i.e. either floor or
        ceiling).

    Returns
    -------
    npt.NDArray
        Coordinates of the point perpendicular to the base line that is opposite the apex, this line is the minimum
        feret distance for acute triangles (but not scalene triangles).
    """

    def angle_between(apex, b):
        return np.arccos(np.dot(apex, b) / (np.linalg.norm(apex) * np.linalg.norm(b)))

    angle_apex_base1_base2 = angle_between(apex - base1, base2 - base1)
    len_apex_base2 = np.linalg.norm(apex - base1)
    cos_apex_base1_base2 = np.cos(angle_apex_base1_base2)
    k = len_apex_base2 * cos_apex_base1_base2
    unit_base1_base2 = (base1 - base2) / np.linalg.norm(base2 - base1)
    d = base1 - k * unit_base1_base2
    if round_coord:
        # Round up/down base on position relative to apex to get an actual cell
        d[0] = np.ceil(d[0]) if d[0] > apex[0] else np.floor(d[0])
        d[1] = np.ceil(d[1]) if d[1] > apex[1] else np.floor(d[1])
        return np.asarray([int(d[0]), int(d[1])])
    return np.asarray([d[0], d[1]])


def sort_clockwise(coordinates: npt.NDArray) -> npt.NDArray:
    """Sort an array of coordinates in a clockwise order.

    Parameters
    ----------
    coordinates : npt.NDArray
        Unordered array of points. Typically a convex hull.

    Returns
    -------
    npt.NDArray
        Points ordered in a clockwise direction.

    Examples
    --------
    >>> import numpy as np
    >>> from topostats.measure import feret
    >>> unordered = np.asarray([[0, 0], [5, 5], [0, 5], [5, 0]])
    >>> feret.sort_clockwise(unordered)

        array([[0, 0],
               [0, 5],
               [5, 5],
               [5, 0]])
    """
    center_x, center_y = coordinates.mean(0)
    x, y = coordinates.T
    angles = np.arctan2(x - center_x, y - center_y)
    order = np.argsort(angles)
    return coordinates[order]


def in_polygon(line: npt.NDArray, lower_hull: npt.NDArray, upper_hull: npt.NDArray) -> bool:
    """
    Check whether a line is within or on the edge of a polygon.

    If either or both of the line points the edges of the polygon this is considered to be within, but if one of the
    points is outside of the polygon it is not contained within. Uses Shapely for most checks but it was found that if a
    given  line was identical to one of the edges of the polygon it was considered to be outside and this is not the
    desired behaviour as such lines, typically the height of triangles used when determining minimum feret distances,
    are the values we wish to retain. It is only lines with points that are completely outside of the polygon that we
    wish to exclude.

    Parameters
    ----------
    line : npt.NDArray
        Numpy array defining the coordinates of a single, linear line.
    lower_hull : npt.NDArray
        The lower convex hull.
    upper_hull : npt.NDArray
        The upper convex hull of the polygon.

    Returns
    -------
    bool
        Whether the line is contained within or is on the border of the polygon.
    """
    # Combine the upper and lower hulls
    hull = np.unique(np.concatenate([lower_hull, upper_hull], axis=0), axis=0)
    # Sort the hull and create Polygon (closes the shape for testing last edge)
    polygon = Polygon(sort_clockwise(hull))
    # Extract coordinates for comparison to line.
    x, y = polygon.exterior.coords.xy
    closed_shape = np.asarray(tuple(zip(x, y)))
    # Check whether the line (notionally the triangle height) is equivalent to one of the edges. Required as Shapely
    # returns False in such situations.
    line = LineString(sort_clockwise(line))
    length = len(closed_shape)
    edge_count = 0
    while edge_count < length - 1:
        sorted_edge = sort_clockwise(closed_shape[edge_count : edge_count + 2])
        edge = LineString(sorted_edge)
        if list(line.coords) == list(edge.coords):
            return True
        edge_count += 1
    return contains(polygon, line)


def min_max_feret(points: npt.NDArray, axis: int = 0) -> tuple[float, tuple[int, int], float, tuple[int, int]]:
    """
    Given a list of 2-D points, returns the minimum and maximum feret diameters.

    `Feret diameter <https://en.wikipedia.org/wiki/Feret_diameter>`

    Parameters
    ----------
    points : npt.NDArray
        A 2-D array of points for the outline of an object.
    axis : int
        Which axis to sort coordinates on, 0 for row (default); 1 for columns.

    Returns
    -------
    tuple
        Tuple of the minimum feret distance and its coordinates and the maximum feret distance and  its coordinates.
    """
    caliper_min_feret = list(rotating_calipers(points, axis))
    min_ferets, calipers, min_feret_coords = zip(*caliper_min_feret)
    # Calculate the squared distance between caliper pairs for max feret
    calipers = np.asarray(calipers)
    caliper1 = calipers[:, 0]
    caliper2 = calipers[:, 1]
    # Determine maximum feret (and coordinates) from all possible calipers
    squared_distance_per_pair = [
        ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2, (list(p), list(q))) for p, q in zip(caliper1, caliper2)
    ]
    max_feret_sq, max_feret_coord = max(squared_distance_per_pair)
    # Determine minimum feret (and coordinates) from all caliper triangles, but only if the min_feret_coords (y) are
    # within the polygon
    hull = hulls(points)
    triangle_min_feret = [
        [x, (list(map(list, y)))] for x, y in zip(min_ferets, min_feret_coords) if in_polygon(y, hull[0], hull[1])
    ]
    min_feret, min_feret_coord = min(triangle_min_feret)
    return min_feret, np.asarray(min_feret_coord), sqrt(max_feret_sq), np.asarray(max_feret_coord)


def get_feret_from_mask(mask_im: npt.NDArray, axis: int = 0) -> tuple[float, tuple[int, int], float, tuple[int, int]]:
    """
    Calculate the minimum and maximum feret diameter of the foreground object of a binary mask.

    The outline of the object is calculated and the pixel coordinates transformed to a list for calculation.

    Parameters
    ----------
    mask_im : npt.NDArray
        Binary Numpy array.
    axis : int
        Which axis to sort coordinates on, 0 for row (default); 1 for columns.

    Returns
    -------
    Tuple[float, Tuple[int, int], float, Tuple[int, int]]
        Returns a tuple of the minimum feret and its coordinates and the maximum feret and its coordinates.
    """
    eroded = skimage.morphology.erosion(mask_im)
    outline = mask_im ^ eroded
    boundary_points = np.argwhere(outline > 0)
    boundary_point_list = np.asarray(list(map(list, list(boundary_points))))
    return min_max_feret(boundary_point_list, axis)


def get_feret_from_labelim(label_image: npt.NDArray, labels: None | list | set = None, axis: int = 0) -> dict:
    """
    Calculate the minimum and maximum feret and coordinates of each connected component within a labelled image.

    If labels is None, all labels > 0 will be analyzed.

    Parameters
    ----------
    label_image : npt.NDArray
        Numpy array with labelled connected components (integer)
    labels : None | list
        A list of labelled objects for which to calculate
    axis : int
        Which axis to sort coordinates on, 0 for row (default); 1 for columns.

    Returns
    -------
    dict
        Dictionary with labels as keys and values are a tuple of the minimum and maximum feret distances and
    coordinates.
    """
    if labels is None:
        labels = set(np.unique(label_image)) - {0}
    results = {}
    for label in labels:
        results[label] = get_feret_from_mask(label_image == label, axis)
    return results


def plot_feret(  # pylint: disable=too-many-arguments,too-many-locals # noqa: C901
    points: npt.NDArray,
    axis: int = 0,
    plot_points: str = "k",
    plot_hulls: tuple = ("g-", "r-"),
    plot_calipers: str = "y-",
    plot_triangle_heights: str = "b:",
    plot_min_feret: str = "m--",
    plot_max_feret: str = "m--",
    filename: str | Path | None = "./feret.png",
    show: bool = False,
) -> None:
    """
    Plot upper and lower convex hulls with rotating calipers and optionally the minimum feret distances.

    Plot varying levels of details in constructing convex hulls and deriving the minimum and maximum feret.

    For format strings see the Notes section of `matplotlib.pyplot.plot
    <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html>`.

    Parameters
    ----------
    points : npt.NDArray
        Points to be plotted which form the shape of interest.
    axis : int
        Which axis to sort coordinates on, 0 for row (default); 1 for columns. (Should give the same results!).
    plot_points : str
        Format string for plotting points. If 'None' points are not plotted.
    plot_hulls : tuple
        Tuple of length 2 of format strings for plotting the convex hull, these should differe to allow distinction
        between hulls. If 'None' hulls are not plotted.
    plot_calipers : str
        Format string for plotting calipers. If 'None' calipers are not plotted.
    plot_triangle_heights : str
        Format string for plotting the triangle heights used in calulcating the minimum feret. These should cross the
    opposite edge perpendicularly. If 'None' triangle heights are not plotted.
    plot_min_feret : str
        Format string for plotting the minimum feret. If 'None' the minimum feret is not plotted.
    plot_max_feret : str
        Format string for plotting the maximum feret. If 'None' the maximum feret is not plotted.
    filename : str | Path | None
        Location to save the image to.
    show : bool
        Whether to display the image.

    Examples
    --------
    >>> from skimage import draw
    >>> from topostats.measure import feret

    >>> tiny_quadrilateral = np.asarray(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint8)

    >>> feret.plot_feret(np.argwhere(tiny_quadrilateral == 1))

    >>> holo_ellipse_angled = np.zeros((8, 10), dtype=np.uint8)
        rr, cc = draw.ellipse_perimeter(4, 5, 1, 3, orientation=np.deg2rad(30))
        holo_ellipse_angled[rr, cc] = 1

    >>> feret.plot_feret(np.argwhere(holo_ellipse_angled == 1), plot_heights = None)

    >>> another_triangle = np.asarray([[5, 4], [2, 1], [8,2]])

    >>> feret.plot_feret(another_triangle)
    """
    # Derive everything needed regardless of required, routine is only for investigating/debugging so speed not
    # critical.
    upper_hull, lower_hull = hulls(points)
    upper_hull = np.asarray(upper_hull)
    lower_hull = np.asarray(lower_hull)
    min_feret_calipers_base = list(rotating_calipers(points, axis))
    _, calipers, triangle_coords = zip(*min_feret_calipers_base)
    calipers = np.asarray(calipers)
    triangle_coords = np.asarray(triangle_coords)
    min_feret_distance, min_feret_coords, max_feret_distance, max_feret_coords = min_max_feret(points, axis)
    min_feret_coords = np.asarray(min_feret_coords)
    max_feret_coords = np.asarray(max_feret_coords)

    fig, ax = plt.subplots(1, 1)
    if plot_points is not None:
        plt.scatter(points[:, 0], points[:, 1], c=plot_points)
    if plot_hulls is not None:
        plt.plot(upper_hull[:, 0], upper_hull[:, 1], plot_hulls[0], label="Upper Hull")
        plt.scatter(upper_hull[:, 0], upper_hull[:, 1], c=plot_hulls[0][0])
        plt.plot(lower_hull[:, 0], lower_hull[:, 1], plot_hulls[1], label="Lower Hull")
        plt.scatter(lower_hull[:, 0], lower_hull[:, 1], c=plot_hulls[1][0])
    if plot_calipers is not None:
        for caliper in calipers:
            plt.plot(caliper[:, 0], caliper[:, 1], plot_calipers)
    if plot_triangle_heights is not None:
        for triangle_h in triangle_coords:
            plt.plot(triangle_h[:, 0], triangle_h[:, 1], plot_triangle_heights)
    if plot_min_feret is not None:
        # for min_feret in min_feret_coords:
        plt.plot(
            min_feret_coords[:, 0],
            min_feret_coords[:, 1],
            plot_min_feret,
            label=f"Minimum Feret ({min_feret_distance:.3f})",
        )
    if plot_max_feret is not None:
        # for max_feret in max_feret_coords:
        plt.plot(
            max_feret_coords[:, 0],
            max_feret_coords[:, 1],
            plot_max_feret,
            label=f"Maximum Feret ({max_feret_distance:.3f})",
        )
    plt.title("Upper and Lower Convex Hulls")
    plt.axis("equal")
    plt.legend()
    plt.grid(True)
    if filename is not None:
        plt.savefig(filename)
    if show:
        plt.show()

    return fig, ax
