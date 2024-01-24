"""Calculate feret distances for 2-D objects.

This code comes from a gist written by @VolkerH under BSD-3 License

https://gist.github.com/VolkerH/0d07d05d5cb189b56362e8ee41882abf

During testing it was discovered that sorting points prior to derivation of upper and lower convex hulls was problematic
so this step was removed.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from math import sqrt

import numpy as np
import numpy.typing as npt
import skimage.morphology

from topostats.logs.logs import LOGGER_NAME

LOGGER = logging.getLogger(LOGGER_NAME)

# pylint: disable=fixme


def orientation(p: npt.NDArray, q: npt.NDArray, r: npt.NDArray) -> int:
    """Determine the orientation of three points as either clockwise, counter-clock-wise or colinear.

    Parameters
    ----------
    p: npt.NDArray
        First point (assumed to have a length of 2).
    q: npt.NDArray
        Second point (assumed to have a length of 2).
    r: npt.NDArray
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
    points: npt.NDArray
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
    """Given a list of 2d points, finds all ways of sandwiching the points.

    Calculates the upper and lower convex hulls and then finds all pairwise combinations between each set of points.

    Parameters
    ----------
    points: npt.NDArray
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


# snoop.install(enabled=True)


# @snoop
def rotating_calipers(points: npt.NDArray, axis: int = 0) -> Generator:
    """Given a list of 2d points, finds all ways of sandwiching the points between two parallel lines.

    This yields the sequence of pairs of points touched by each pair of lines across all points around the hull of a
    polygon.

    `Rotating Calipers <https://en.wikipedia.org/wiki/Rotating_calipers>_`

    Parameters
    ----------
    points: npt.NDArray
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
        yield triangle_height(base1, base2, apex), calipers, [list(_mid_point(base1, base2, apex)), apex]


# @snoop
def triangle_height(
    base1: npt.NDArray | list, base2: npt.NDArray | list, apex: npt.NDArray | list
) -> tuple[float, list]:
    """Calculate the height of triangle formed by three points.

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
    tuple
        Height of the triangle.

    Examples
    --------
    >>> min_feret([4, 0], [4, 3], [0,0])
        4.0
    """
    a_b = np.asarray(base1) - np.asarray(base2)
    a_c = np.asarray(base1) - np.asarray(apex)
    return np.linalg.norm(np.cross(a_b, a_c)) / np.linalg.norm(a_b)


# TODO : Remove, ultimately redundant its not the mid-point that is required.
def _mid_point(point1: npt.NDArray | list, point2: npt.NDArray | list, apex: npt.NDArray | list) -> list:
    """Return the closest integer to the mid-point between two adjacent calliper points.

    When using triangles to calculate the minimum feret the height of the triangle formed by antipodal points and the
    next point on the convex hull is the minimum feret. In order to trake a profile the position at which the height
    touches the line between adjacent convex hulls is required. This is half the linear distance between these two
    points, but because we are dealing with arrays this needs rounding so that we have a coordinate in the `x,y` point
    of the array. Whether the floor or ceiling is taken depends on where the point is relative to the apex.

     Example 1        Example 2

       1 1              0 1
       1 0              1 1

    In both these examples the minimum feret is between the apex and the baseline formed by the points [[1, 0], [0, 1]]
    which in Example 1 is [0, 0] in Example 2 its [1, 1]. The minimum feret distance is sqrt(2) / 2  (i.e. 0.707106) and
    formed by the coordinates [[1, 1], [0.5, 0.5]] but the later point is not a valid point in a Numpy array where
    indices are integers and so we have to find the next point "outwards" for which to obtain the opposite feret
    coordinates. For Example 1 this would be the ceiling (i.e. rounding up) of both points, whilst for Example 2 it
    would be the floor (i.e. rounding down) of both points.

    Parameters
    ----------
    point1: npt.NDArray | list
        Position of the first point on the convex hull.
    point2: npt.NDArray | list
        Position of the second point on the convex hull.
    apex: npt.NDArray | list
        Position of the apex of the triangle on the convex hull.

    Returns
    -------
    list
        coordinate of the nearest point to the mid-point.
    """
    mid_x = (point1[0] + point2[0]) / 2
    mid_y = (point1[1] + point2[1]) / 2
    mid_x = np.ceil(mid_x) if mid_x > apex[0] else np.floor(mid_x)
    mid_y = np.ceil(mid_y) if mid_y > apex[1] else np.floor(mid_y)
    return [int(mid_x), int(mid_y)]


def _min_feret_coord(base1: list, base2: list, apex: list):
    """
    Calculate the coordinate opposite the apex that is prependicular to the base of the triangle.

    Code courtesy of @SylviaWhittle.
    """
    # Find the perpendicular gradient to bc
    grad_base = (base2[1] - base1[1]) / (base2[0] - base1[0])
    grad_ad = -1 / grad_base
    # Find the intercept
    intercept_ad = base1[1] - grad_ad * base1[0]
    intercept_bc = base1[1] - grad_base * base1[0]
    # Find the intersection
    x = (intercept_bc - intercept_ad) / (grad_ad - grad_base)
    y = grad_ad * x + intercept_ad
    # Round up/down base on position relative to apex
    x = np.ceil(x) if x > apex[0] else np.floor(x)
    y = np.ceil(y) if y > apex[1] else np.floor(y)
    return [int(x), int(y)]


# @snoop
def min_max_feret(points: npt.NDArray, axis: int = 0) -> tuple[float, tuple[int, int], float, tuple[int, int]]:
    """Given a list of 2-D points, returns the minimum and maximum feret diameters.

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
    # TODO : Use this instead once we are using the min_feret_coords
    # min_ferets, calipers, min_feret_coords = zip(*caliper_min_feret)
    min_ferets, calipers, _ = zip(*caliper_min_feret)
    # Calculate the squared distance between caliper pairs for max feret
    calipers = np.asarray(calipers)
    caliper1 = calipers[:, 0]
    caliper2 = calipers[:, 1]
    squared_distance_per_pair = [
        ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2, (list(p), list(q))) for p, q in zip(caliper1, caliper2)
    ]
    # TODO : replace calipers with min_feret_coords once correctly calculated
    caliper_min_feret = [[x, (list(map(list, y)))] for x, y in zip(min_ferets, calipers)]
    min_feret, min_feret_coord = min(caliper_min_feret)
    max_feret_sq, max_feret_coord = max(squared_distance_per_pair)
    return min_feret, min_feret_coord, sqrt(max_feret_sq), max_feret_coord


def get_feret_from_mask(mask_im: npt.NDArray, axis: int = 0) -> tuple[float, tuple[int, int], float, tuple[int, int]]:
    """Calculate the minimum and maximum feret diameter of the foreground object of a binary mask.

    The outline of the object is calculated and the pixel coordinates transformed to a list for calculation.

    Parameters
    ----------
    mask_im: npt.NDArray
        Binary Numpy array.
    axis: int
        Which axis to sort coordinates on, 0 for row (default); 1 for columns.

    Returns
    -------
    Tuple[float, Tuple[int, int], float, Tuple[int, int]]
        Returns a tuple of the minimum feret and its coordinates and the maximum feret and its coordinates.
    """
    eroded = skimage.morphology.erosion(mask_im)
    outline = mask_im ^ eroded
    boundary_points = np.argwhere(outline > 0)
    # convert numpy array to a list of (x,y) tuple points
    boundary_point_list = list(map(list, list(boundary_points)))
    return min_max_feret(boundary_point_list, axis)


def get_feret_from_labelim(label_image: npt.NDArray, labels: None | list | set = None, axis: int = 0) -> dict:
    """Calculate the minimum and maximum feret and coordinates of each connected component within a labelled image.

    If labels is None, all labels > 0 will be analyzed.

    Parameters
    ----------
    label_image: npt.NDArray
        Numpy array with labelled connected components (integer)
    labels: None | list
        A list of labelled objects for which to calculate
    axis: int
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
