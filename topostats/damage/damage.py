"""Functions for damage detection and quantification."""

from dataclasses import dataclass
import numpy as np
import numpy.typing as npt


def calculate_defects_and_gap_lengths(
    points_distance_to_previous_nm: npt.NDArray[np.float64],
    defects_bool: npt.NDArray[np.bool_],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Calculate the lengths of defects"""
    defect_lengths = []
    gap_lengths = []

    current_gap_length = 0.0
    current_defect_length = 0.0

    for index, point in enumerate(defects_bool):

        if point:
            current_defect_length += points_distance_to_previous_nm[index]
            if current_gap_length > 0:
                # End the current gap
                gap_lengths.append(current_gap_length)
                current_gap_length = 0.0
        else:
            if current_defect_length > 0:
                defect_lengths.append(current_defect_length)
                current_defect_length = 0.0
            # Continue the gap
            current_gap_length += points_distance_to_previous_nm[index]

    # if the last defect is still open, check if it's connected to the start defect and combine if necessary
    if current_defect_length > 0:
        if defects_bool[0]:
            # Check if there are any defects in the list, if not, then the whole thing is a defect
            if len(defect_lengths) == 0:
                defect_lengths.append(current_defect_length)
            else:
                # End is connected to start, so combine the lengths into the first defect
                defect_lengths[0] += current_defect_length
        else:
            # End is not connected to start, so add as a new defect
            defect_lengths.append(current_defect_length)
    else:
        # Check if there is a gap in the list, if not, then there is just one big gap
        if len(gap_lengths) == 0:
            gap_lengths.append(current_gap_length)
        else:
            # There is a gap already in the list, so we now check if this gap is connected to the first gap
            # If the first point is also a gap, we can combine it with the last gap
            if current_gap_length > 0 and not defects_bool[0]:
                gap_lengths[0] += current_gap_length
            else:
                gap_lengths.append(current_gap_length)

    return np.array(defect_lengths), np.array(gap_lengths)


def get_defect_start_end_indexes(
    defects_bool: npt.NDArray[np.bool_],
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    """Get the start and end indexes of defects in a boolean array.

    Stars and ends are
    defined as the first and last point of a defect, this can be the same point if the defect is only one point long.

    Parameters
    ----------
    defects_bool : npt.NDArray[np.bool_]
        A boolean array where True indicates a defect and False indicates no defect.
    Returns
    -------
    tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]
        A tuple containing two arrays:
        - The indexes of the starts of defects.
        - The indexes of the ends of defects.
    """
    defect_start_indexes = []
    defect_end_indexes = []

    for index, point in enumerate(defects_bool):
        previous_point = defects_bool[index - 1]
        next_point = defects_bool[(index + 1) % len(defects_bool)]
        if point and not previous_point:
            defect_start_indexes.append(index)
        if point and not next_point:
            defect_end_indexes.append(index)

    return (
        np.array(defect_start_indexes, dtype=np.int_),
        np.array(defect_end_indexes, dtype=np.int_),
    )


@dataclass
class Defect:
    """A class to represent a defect in a point cloud.

    Attributes
    ----------
    start_index : int
        The index of the first point of the defect.
    end_index : int
        The index of the last point of the defect.
    length_nm : float
        The length of the defect in nanometers.
    """

    start_index: int
    end_index: int
    length_nm: float


@dataclass
class DefectGap:
    """A class to represent a gap between defects in a point cloud.

    Attributes
    ----------
    start_index : int
        The index of the first point of the gap.
    end_index : int
        The index of the last point of the gap.
    length_nm : float
        The length of the gap in nanometers.
    """

    start_index: int
    end_index: int
    length_nm: float


class OrderedDefectGapList:
    """A class to store defects and gaps in a list ordered by the start index of the defect or gap."""

    def __init__(self, defect_gap_list: list[Defect | DefectGap] | None = None) -> None:
        """Initialise the class

        Parameters
        ----------
        defect_gap_list : list[Defect | DefectGap] | None, optional
            An optional list of Defect or DefectGap objects to initialize the class with.
            If provided, the list will be sorted by start index. If None, an empty list is created.
        """
        if defect_gap_list is None:
            self.defect_gap_list: list[Defect | DefectGap] = []
        else:
            self.defect_gap_list = defect_gap_list.copy()
            self.sort_defect_gap_list()

    def sort_defect_gap_list(self) -> None:
        """Sort the defect and gap list by the start index of the defect or gap."""
        self.defect_gap_list.sort(key=lambda x: x.start_index)

    def add_item(self, item: Defect | DefectGap) -> None:
        """Add a defect or gap to the list."""
        self.defect_gap_list.append(item)
        self.sort_defect_gap_list()

    def __eq__(self, other: object) -> bool:
        """Check if two OrderedDefectGapList objects are equal."""
        if not isinstance(other, OrderedDefectGapList):
            return NotImplemented
        return self.defect_gap_list == other.defect_gap_list


def get_defects_and_gaps_linear(
    defects_bool: npt.NDArray[np.bool_],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Get the defects as a list of tuples of start and end indexes.

    Parameters
    ----------
    defects_bool : npt.NDArray[np.bool_]
        A boolean array where True indicates a defect and False indicates no defect.
    Returns
    -------
    """
    defects: list[tuple[int, int]] = []
    gaps: list[tuple[int, int]] = []
    in_defect = False
    in_gap = False
    current_defect_gap_start_index = 0

    for index, point in enumerate(defects_bool):

        if point:
            if in_gap:
                # End of the gap
                gaps.append((current_defect_gap_start_index, index - 1))
                in_gap = False
            if not in_defect:
                # Start new defect
                current_defect_gap_start_index = index
                in_defect = True
        else:
            if in_defect:
                # End of the defect
                defects.append((current_defect_gap_start_index, index - 1))
                in_defect = False
            if not in_gap:
                # Start new gap
                current_defect_gap_start_index = index
                in_gap = True
    # If we are still in a defect or gap at the end of the loop, we need to close it
    if in_defect:
        defects.append((current_defect_gap_start_index, len(defects_bool) - 1))
    elif in_gap:
        gaps.append((current_defect_gap_start_index, len(defects_bool) - 1))

    return (
        defects,
        gaps,
    )


def get_defects_circular(defects_bool: npt.NDArray[np.bool_]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Get the defects as a list of tuples of start and end indexes, with the array ends connected.

    Parameters
    ----------
    defects_bool : npt.NDArray[np.bool_]
        A boolean array where True indicates a defect and False indicates no defect.
    Returns
    -------
    tuple[list[tuple[int, int]], list[tuple[int, int]]]
        A tuple containing two lists:
        - The defects as a list of tuples of start and end indexes.
        - The gaps as a list of tuples of start and end indexes.
    """
    defects, gaps = get_defects_linear(defects_bool)

    # If the first and last points are both defects, combine the first and last defects
    if defects_bool[0] and defects_bool[-1]:
        assert len(defects) > 0, "There should be at least one defect if the first and last points are defects."
        if len(defects) == 1:
            # The whole array is a defect
            pass
        else:
            # Combine the first and last defects
            _, first_defect_end = defects[0]
            last_defect_start, _ = defects[-1]
            # Update the first defect to include the last defect
            defects[0] = (last_defect_start, first_defect_end)
            # Remove the last defect
            defects.pop()

    # If the first and last points are both gaps, combine the first and last gaps
    if not defects_bool[0] and not defects_bool[-1]:
        assert len(gaps) > 0, "There should be at least one gap if the first and last points are gaps."
        if len(gaps) == 1:
            # The whole array is a gap
            pass
        else:
            # Combine the first and last gaps
            _, first_gap_end = gaps[0]
            last_gap_start, _ = gaps[-1]
            # Update the first gap to include the last gap
            gaps[0] = (last_gap_start, first_gap_end)
            # Remove the last gap
            gaps.pop()

    return (
        defects,
        gaps,
    )


def calculate_distance_of_region(
    start_index: int,
    end_index: int,
    distance_to_previous_points_nm: npt.NDArray[np.float64],
    circular: bool,
) -> float:
    """Calculate the distance of a region in the point cloud.

    Note: This function cannot take a region that is the whole array, since that would imply the start and end be the
    same point, but this is assumed to be a unit region, not an array-wide region.

    Note to devs: remember that the array is the distance to the previous point. So the distance between
    points i and j does not include the distance of index i, since that's the distance to the previous point.
    We do however include half the distance to the previous point for i, since we want to support an approximation
    for unitary regions
    ie the distance for the following array with regions marked with X,
    o--o--X--o--o--X--o--o
    would be calculated as this distance, marked by |:
    o--o-|-X--o--o--X-|-o--o
    since, if you imagine an entire array filled with regions, then this would mean the total distance is equal to
    the total length of the array (makes sense), else there would be gaps in between the regions
    A unitary region looks like this:
    o--o-|-X-|-o--o
    so it does have length, however if we didn't do this, it would have 0 length!

    Parameters
    ----------
    start_index : int
        The index of the first point in the region.
    end_index : int
        The index of the last point in the region.
    distance_to_previous_points_nm : npt.NDArray[np.float64]
        An array of distances to the previous points in nanometers.
    circular : bool
        If True, the array is treated as circular, meaning that the end of the array wraps around to the start.
        If False, the array is treated as linear, meaning that the end of the array does not wrap around to the start.

    Returns
    -------
    float
        The total distance of the region in nanometers.

    Raises
    ------
    ValueError
        If the start index is greater than the end index in a linear array, since this necessitates wrapping around the
        end of the array to meet the end index.
    """

    # Get the distance from the start index to the end index
    if start_index <= end_index:
        # Normal case, no wrapping around the end of the array, just sum the distances
        distance_without_halves = np.sum(distance_to_previous_points_nm[start_index + 1 : end_index + 1])
        # Add half the distance to the start previous point and half the distance to the end next point
        # Check if at the start or end of array
        if start_index == 0:
            # At the start
            if circular:
                # If circular, then can take half the distance to the end point of the array since it wraps around
                start_half_distance = distance_to_previous_points_nm[start_index] / 2
                end_half_distance = distance_to_previous_points_nm[end_index + 1] / 2
            else:
                # If not circular, then we can't add this half distance
                start_half_distance = 0.0
                end_half_distance = distance_to_previous_points_nm[end_index + 1] / 2
        elif end_index == len(distance_to_previous_points_nm) - 1:
            # End point is at the end of the array
            if circular:
                # If circular, then can take half the distance to the start point of the array since it wraps around
                end_half_distance = distance_to_previous_points_nm[0] / 2
                start_half_distance = distance_to_previous_points_nm[start_index] / 2
            else:
                # If not circular, then we can't add this half distance
                end_half_distance = 0.0
                start_half_distance = distance_to_previous_points_nm[start_index] / 2
        else:
            # Normal case
            start_half_distance = distance_to_previous_points_nm[start_index] / 2
            end_half_distance = distance_to_previous_points_nm[end_index + 1] / 2
        return distance_without_halves + start_half_distance + end_half_distance
    else:
        if not circular:
            # This cannot happen, since if the start index is greater than the end index, then we must wrap around to
            # the start of the array to meet the end index, but cannot in a linear array.
            raise ValueError(
                f"Cannot calculate distance of region {start_index} to {end_index} in a linear array. "
                "Start index cannot be greater than end index in a linear array."
            )
        # The region wraps around the end of the array
        # Calculate the distance from the start index to the end of the array
        distance_to_end = np.sum(distance_to_previous_points_nm[start_index + 1 :])
        # Calculate the distance from the start of the array to the end index
        distance_to_start = np.sum(distance_to_previous_points_nm[: end_index + 1])
        # Here we don't need to worry about the indexes of the start and end points since the ends of the array are
        # inside the region.
        # Add the half distances to the start and end points
        start_half_distance = distance_to_previous_points_nm[start_index] / 2
        end_half_distance = distance_to_previous_points_nm[end_index + 1] / 2
        return distance_to_end + distance_to_start + start_half_distance + end_half_distance


# def calculate_defect_and_gap_lengths(
#     distance_to_previous_points_nm: npt.NDArray[np.float64],
#     defects: list[tuple[int, int]],
#     gaps: list[tuple[int, int]],
# ) -> tuple[list[tuple[int, int, float]], list[tuple[int, int, float]]]:
#     """Calculate the lengths of the defects and gaps."""
#     defect_lengths = []
#     gap_lengths = []
