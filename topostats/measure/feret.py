"""Calculate feret distances for 2-D objects.

This code comes from a gist written by @VolkerH under BSD-3 License

https://gist.github.com/VolkerH/0d07d05d5cb189b56362e8ee41882abf

During testing it was discovered that sorting points prior to derivation of upper and lower convex hulls was problematic
so this step was removed.
"""
from __future__ import annotations

from math import sqrt

import numpy as np
import numpy.typing as npt
import skimage.morphology


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
        Returns a positive value of p-q-r are clockwise, neg if counter-clock-wise, zero if colinear.
    """
    return (q[1] - p[1]) * (r[0] - p[0]) - (q[0] - p[0]) * (r[1] - p[1])


def hulls(points: npt.NDArray) -> tuple[list, list]:
    """Graham scan to find upper and lower convex hulls of a set of 2-D points.

    Parameters
    ----------
    points : npt.NDArray
        2-D Array of points for the outline of an object.

    Returns
    -------
    Tuple[list, list]
        Tuple of two Numpy arrays of the original coordinates split into upper and lower hulls.
    """
    upper_hull = []
    lower_hull = []
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
    List[Tuple[int, int]]
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


def rotating_calipers(points: npt.NDArray) -> list[tuple[list, list]]:
    """Given a list of 2d points, finds all ways of sandwiching the points.

    Between two parallel lines that touch one point each, and yields the sequence
    of pairs of points touched by each pair of lines.

    `Rotating Calipers <https://en.wikipedia.org/wiki/Rotating_calipers>_`
    """
    upper_hull, lower_hull = hulls(points)
    i = 0
    j = len(lower_hull) - 1
    while i < len(upper_hull) or j > 0:
        yield upper_hull[i], lower_hull[j]
        # if all the way through one side of hull, advance the other side
        if i == len(upper_hull):
            j -= 1
        elif j == 0:
            i += 1
        # still points left on both lists, compare slopes of next hull edges
        # being careful to avoid divide-by-zero in slope calculation
        elif ((upper_hull[i + 1][1] - upper_hull[i][1]) * (lower_hull[j][0] - lower_hull[j - 1][0])) > (
            (lower_hull[j][1] - lower_hull[j - 1][1]) * (upper_hull[i + 1][0] - upper_hull[i][0])
        ):
            i += 1
        else:
            j -= 1


def min_max_feret(points: npt.NDArray) -> tuple[float, tuple[int, int], float, tuple[int, int]]:
    """Given a list of 2-D points, returns the minimum and maximum feret diameters.

    Parameters
    ----------
    points: npt.NDArray
        A 2-D array of points for the outline of an object.

    Returns
    -------
    tuple
        Tuple of the minimum feret distance and its coordinates and the maximum feret distance and  its coordinates.
    """
    squared_distance_per_pair = [
        ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2, (p, q)) for p, q in rotating_calipers(points)
    ]
    min_feret_sq, min_feret_coords = min(squared_distance_per_pair)
    max_feret_sq, max_feret_coords = max(squared_distance_per_pair)
    return sqrt(min_feret_sq), min_feret_coords, sqrt(max_feret_sq), max_feret_coords


def get_feret_from_mask(mask_im: npt.NDArray) -> tuple[float, tuple[int, int], float, tuple[int, int]]:
    """Calculate the minimum and maximum feret diameter of the foreground object of a binary mask.

    The outline of the object is calculated and the pixel coordinates transformed to a list for calculation.

    Parameters
    ----------
    mask_im: npt.NDArray
        Binary Numpy array.

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
    return min_max_feret(boundary_point_list)


def get_feret_from_labelim(label_image: npt.NDArray, labels: None | list | set = None) -> dict:
    """Calculate the minimum and maximum feret and coordinates of each connected component within a labelled image.

    If labels is None, all labels > 0
    will be analyzed.

    Parameters
    ----------
    label_image: npt.NDArray
        Numpy array with labelled connected components (integer)
    labels: None | list
        A list of labelled objects for which to calculate

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
        results[label] = get_feret_from_mask(label_image == label)
    return results
