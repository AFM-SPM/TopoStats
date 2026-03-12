"""Functions for damage detection and quantification."""

from collections.abc import Generator
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, computed_field

from topostats.measure.curvature import (
    calculate_discrete_angle_difference_circular,
    calculate_discrete_angle_difference_linear,
    total_turn_in_region_radians,
)
from topostats.plottingfuncs import Colormap

colormap = Colormap()
CMAP: plt.Colormap = colormap.get_cmap()
VMIN = -3
VMAX = 4
IMGPLOTARGS: dict = {"cmap": CMAP, "vmin": VMIN, "vmax": VMAX}


@dataclass
class Defect:
    """A class to represent a defect in a trace.

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
    position_along_trace_nm: float
    total_turn_radians: tuple[float, float]


@dataclass
class DefectGap:
    """A class to represent a gap between defects in a trace.

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
    position_along_trace_nm: float
    total_turn_radians: tuple[float, float]


class OrderedDefectGapList:
    """A class to store defects and gaps in a list ordered by the start index of the defect or gap."""

    def __init__(self, defect_gap_list: list[Defect | DefectGap] | None = None) -> None:
        """
        Initialise the class.

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
        """Check if two OrderedDefectGapList objects are equal.

        Floating-point lengths are compared with a small tolerance to handle precision errors.
        """
        if not isinstance(other, OrderedDefectGapList):
            return NotImplemented

        # Check if lists have the same length
        if len(self.defect_gap_list) != len(other.defect_gap_list):
            return False

        # Compare each item with floating-point tolerance
        for self_item, other_item in zip(self.defect_gap_list, other.defect_gap_list):
            # Check if items are the same type
            if not isinstance(self_item, type(other_item)):
                return False

            # Check if start and end indices are equal (these should be exact)
            if self_item.start_index != other_item.start_index or self_item.end_index != other_item.end_index:
                return False

            # Check if lengths are approximately equal (with tolerance for floating-point errors)
            if not np.isclose(self_item.length_nm, other_item.length_nm, rtol=1e-9, atol=1e-12):
                return False

            # Check if position along trace is approximately equal (with tolerance for floating-point errors)
            if not np.isclose(
                self_item.position_along_trace_nm, other_item.position_along_trace_nm, rtol=1e-9, atol=1e-12
            ):
                return False

            # Check if total turns are approximately equal (with tolerance for floating-point errors)
            if not np.isclose(
                self_item.total_turn_radians[0], other_item.total_turn_radians[0], rtol=1e-3, atol=1e-3
            ) or not np.isclose(
                self_item.total_turn_radians[1], other_item.total_turn_radians[1], rtol=1e-3, atol=1e-3
            ):
                return False

        return True


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
    tuple[list[tuple[int, int]], list[tuple[int, int]]]
        A tuple containing two lists:
        - The defects as a list of tuples of start and end indexes.
        - The gaps as a list of tuples of start and end indexes.
        Note: the end index is inclusive, so the defect or gap includes the point at the end index.
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


def get_defects_and_gaps_circular(
    defects_bool: npt.NDArray[np.bool_],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
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
    defects, gaps = get_defects_and_gaps_linear(defects_bool)

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
    """
    Calculate the distance of a region in the trace.

    Note: This function cannot take a circular region that is the whole array, since that would imply the start and
    end be the same point, but this is assumed to be a unit region, not an array-wide region.

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
            else:
                # If not circular, then we can't add this half distance
                start_half_distance = 0.0
        else:
            start_half_distance = distance_to_previous_points_nm[start_index] / 2
        if end_index == len(distance_to_previous_points_nm) - 1:
            # End point is at the end of the array
            if circular:
                # If circular, then can take half the distance to the start point of the array since it wraps around
                end_half_distance = distance_to_previous_points_nm[0] / 2
            else:
                # If not circular, then we can't add this half distance
                end_half_distance = 0.0
        else:
            end_half_distance = distance_to_previous_points_nm[end_index + 1] / 2
        return distance_without_halves + start_half_distance + end_half_distance

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


def connect_close_defects(  # noqa: C901
    defects: list[tuple[int, int]],
    gaps: list[tuple[int, int]],
    distance_to_previous_points_nm: npt.NDArray[np.float64],
    circular: bool,
    connect_close_defect_threshold_nm: float,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Connect any defects that are within a given threshold distance of each other.

    Parameters
    ----------
    defects : list[tuple[int, int]]
        A list of tuples of start and end indexes of defects.
    gaps : list[tuple[int, int]]
        A list of tuples of start and end indexes of gaps.
    distance_to_previous_points_nm : npt.NDArray[np.float64]
        An array of distances to the previous points in nanometres.
    circular : bool
        If the trace is circular.
    connect_close_defect_threshold_nm : float
        The threshold distance in nanometres for connecting close defects.

    Returns
    -------
    tuple[list[tuple[int, int]], list[tuple[int, int]]]
        A tuple containing the updated lists of defects and gaps after connecting close defects.
    """
    connected_defects: list[tuple[int, int]] = []
    connected_gaps: list[tuple[int, int]] = []

    # If there are no defects, then do nothing.
    if len(defects) == 0:
        return defects, gaps

    first_defect = defects[0]
    first_gap = gaps[0]
    gap_index = 0
    if first_gap[0] < first_defect[0]:
        # first gap comes before the first defect, add the gap
        connected_gaps.append(first_gap)
        gap_index = 1

    for defect in defects:
        if len(connected_defects) == 0:
            # add the first defect to the list of connected defects
            connected_defects.append(defect)
        else:
            previous_defect = connected_defects[-1]
            gap = gaps[gap_index]
            distance_between_defects = calculate_distance_of_region(
                previous_defect[1],
                defect[0],
                distance_to_previous_points_nm,
                circular,
            )
            if distance_between_defects <= connect_close_defect_threshold_nm:
                # Connect the defects by merging the previous defect and the current defect into a single defect
                connected_defects[-1] = (previous_defect[0], defect[1])
                # Don't add the current gap since it's now part of the defect
            else:
                # Add the current defect to the list of connected defects
                connected_defects.append(defect)
                # Add the current gap to the list of connected gaps
                connected_gaps.append(gap)
            gap_index += 1
    # Add the last gap if it hasn't been added yet
    if gap_index < len(gaps):
        connected_gaps.append(gaps[gap_index])

    # If circular and the trace ends or starts in a gap, then check if we need to connect the first and last defects
    if circular:
        # Only check for connection if there is more than one defect.
        if len(connected_defects) > 1:
            if connected_gaps[0][0] < connected_defects[0][0]:
                # The trace starts with a gap
                first_defect = connected_defects[0]
                last_defect = connected_defects[-1]
                distance_between_defects = calculate_distance_of_region(
                    last_defect[1],
                    first_defect[0],
                    distance_to_previous_points_nm,
                    circular,
                )
                if distance_between_defects <= connect_close_defect_threshold_nm:
                    # Connect the defects by merging the last defect and first defect into a single defect
                    # Update the first defect to include the last defect
                    connected_defects[0] = (last_defect[0], first_defect[1])
                    # Remove the last defect since it's now part of the first defect
                    connected_defects.pop()
                    # Remove the first gap since it's now part of the defect
                    connected_gaps.pop(0)
            elif connected_gaps[-1][1] > connected_defects[-1][1]:
                # The trace ends with a gap
                first_defect = connected_defects[0]
                last_defect = connected_defects[-1]
                distance_between_defects = calculate_distance_of_region(
                    last_defect[1],
                    first_defect[0],
                    distance_to_previous_points_nm,
                    circular,
                )
                if distance_between_defects <= connect_close_defect_threshold_nm:
                    # Connect the defects by merging the last defect and first defect into a single defect
                    # Update the first defect to include the last defect
                    connected_defects[0] = (last_defect[0], first_defect[1])
                    # Remove the last defect since it's now part of the first defect
                    connected_defects.pop()
                    # Remove the last gap since it's now part of the defect
                    connected_gaps.pop()
    return connected_defects, connected_gaps


def get_defects_and_gaps_from_bool_array(
    defects_bool: npt.NDArray[np.bool_],
    trace_points_nm: npt.NDArray[np.float64],
    circular: bool,
    distance_to_previous_points_nm: npt.NDArray[np.float64],
    connect_close_defect_threshold_nm: float | None,
) -> OrderedDefectGapList:
    """
    Get the Defects and DefectGaps from a boolean array of defects and gaps.

    Parameters
    ----------
    defects_bool : npt.NDArray[np.bool_]
        A boolean array where True indicates a defect and False indicates no defect.
    trace_points : npt.NDArray[np.float64]
        The coordinate trace points in nanometre units.
    circular : bool
        If True, the trace is treated as circular, meaning that the end of the trace wraps around to the start.
        If False, the trace is treated as linear, meaning that the end of the trace does not wrap around to the start.
    distance_to_previous_points_nm : npt.NDArray[np.float64]
        An array of distances to the previous points in nanometers. This is used to calculate the lengths of the
        defects and gaps.
    connect_close_defect_threshold_nm : float
        The threshold in nanometers for considering two defects as close.

    Returns
    -------
    OrderedDefectGapList
        An ordered list of Defect and DefectGap objects, sorted by the start index of the defect or gap.
    """
    if circular:
        defects, gaps = get_defects_and_gaps_circular(
            defects_bool=defects_bool,
        )
    else:
        defects, gaps = get_defects_and_gaps_linear(
            defects_bool=defects_bool,
        )

    if connect_close_defect_threshold_nm is not None:
        defects, gaps = connect_close_defects(
            defects=defects,
            gaps=gaps,
            distance_to_previous_points_nm=distance_to_previous_points_nm,
            circular=circular,
            connect_close_defect_threshold_nm=connect_close_defect_threshold_nm,
        )

    # Calculate the lengths of the defects and gaps
    defect_gap_list = calculate_defect_and_gap_lengths(
        distance_to_previous_points_nm,
        defects,
        gaps,
        circular,
    )

    # Calculate the total turns for the defects and gaps
    defect_gap_list_with_turns = calculate_defect_and_gap_turns(
        defects=defect_gap_list,
        trace_points=trace_points_nm,
        circular=circular,
    )

    # Create an OrderedDefectGapList and add the defects and gaps
    ordered_defect_gap_list = OrderedDefectGapList()
    for defect_or_gap in defect_gap_list_with_turns:
        type_of_region, start_index, end_index, length_nm, position_along_trace_nm, total_turn_radians = defect_or_gap
        if type_of_region == "defect":
            ordered_defect_gap_list.add_item(
                Defect(
                    start_index=start_index,
                    end_index=end_index,
                    length_nm=length_nm,
                    position_along_trace_nm=position_along_trace_nm,
                    total_turn_radians=total_turn_radians,
                )
            )
        elif type_of_region == "gap":
            ordered_defect_gap_list.add_item(
                DefectGap(
                    start_index=start_index,
                    end_index=end_index,
                    length_nm=length_nm,
                    position_along_trace_nm=position_along_trace_nm,
                    total_turn_radians=total_turn_radians,
                )
            )

    return ordered_defect_gap_list


def get_midpoint_index_of_region(
    start_index: int,
    end_index: int,
    distance_to_previous_points_nm: npt.NDArray[np.float64],
    circular: bool,
) -> int:
    """
    Get the midpoint index of a region in a trace.

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
    int
        The midpoint index of the region. If the region has an even number of points, the midpoint is rounded down.
        If the region is circular, the midpoint is calculated as if the array wraps around.
    """
    if start_index <= end_index:
        # Normal case, no wrapping needed.
        midpoint_index = (start_index + end_index) // 2
    else:
        if not circular:
            # This cannot happen, since if the start index is greater than the end index, then we must wrap around to
            # the start of the array to meet the end index, but cannot in a linear array.
            raise ValueError(
                f"Cannot calculate midpoint index of region {start_index} to {end_index} in a linear array. "
                "Start index cannot be greater than end index in a linear array."
            )
        # the region wraps around the end of the array, calculate the midpoint index as if the array wraps around
        midpoint_index = (start_index + end_index + len(distance_to_previous_points_nm)) // 2
        # wrap the index to be within the bounds of the array
        midpoint_index = midpoint_index % len(distance_to_previous_points_nm)
    return midpoint_index


def calculate_defect_and_gap_lengths(
    distance_to_previous_points_nm: npt.NDArray[np.float64],
    defects_without_lengths: list[tuple[int, int]],
    gaps_without_lengths: list[tuple[int, int]],
    circular: bool,
) -> list[tuple[str, int, int, float, float]]:
    """Calculate the lengths of the defects and gaps."""
    defects_and_gaps: list[tuple[str, int, int, float, float]] = []

    # Calculate the lengths of the defects
    for start_index, end_index in defects_without_lengths:
        length_nm = calculate_distance_of_region(
            start_index,
            end_index,
            distance_to_previous_points_nm,
            circular,
        )
        midpoint_index = get_midpoint_index_of_region(
            start_index,
            end_index,
            distance_to_previous_points_nm,
            circular,
        )
        # Calculate the position along the trace in nanometers by summing the distances to previous points up to the
        # midpoint index.
        # Note that if circular, then the distance to the previous point at the starting index, is non zero, so we need
        # to ignore this initial value. If linear, then the initial value is zero, and we can ignore it.
        # sum indexes from 1 to midpoint_index (inclusive)
        position_along_trace_nm = np.sum(distance_to_previous_points_nm[1 : midpoint_index + 1])

        defects_and_gaps.append(
            (
                "defect",
                start_index,
                end_index,
                length_nm,
                position_along_trace_nm,
            )
        )

    # Calculate the lengths of the gaps
    for start_index, end_index in gaps_without_lengths:
        length_nm = calculate_distance_of_region(
            start_index,
            end_index,
            distance_to_previous_points_nm,
            circular,
        )
        # Calculate the position along the trace in nanometers by summing the distances to previous points up to the
        # midpoint index
        midpoint_index = get_midpoint_index_of_region(
            start_index,
            end_index,
            distance_to_previous_points_nm,
            circular,
        )
        # Note that if ciruclar, then the distance to the previous point at the starting index, is non zero, so we need
        # to ignore this initial value. If linear, then the initial value is zero, and we can ignore it.
        # sum indexes from 1 to midpoint_index (inclusive)
        position_along_trace_nm = np.sum(distance_to_previous_points_nm[1 : midpoint_index + 1])

        defects_and_gaps.append(
            (
                "gap",
                start_index,
                end_index,
                length_nm,
                position_along_trace_nm,
            )
        )

    return defects_and_gaps


def calculate_defect_and_gap_turns(
    defects: list[tuple[str, int, int, float, float]],
    trace_points: npt.NDArray[np.float64],
    circular: bool,
) -> list[tuple[str, int, int, float, float, tuple[float, float]]]:
    """Calculate the total turn in radians for each defect and gap.

    Parameters
    ----------
    defects : list[tuple[str, int, int, float, float]]
        A list of tuples containing the defect or gap type, start index, end index, length in nm, and position along
        trace in nm.
    trace_points : npt.NDArray[np.number]
        The coordinate trace points in nanometre units.
    circular : bool
        If True, the trace is treated as circular, meaning that the end of the trace wraps around to the start.
        If False, the trace is treated as linear, meaning that the end of the trace does not wrap around to the start.

    Returns
    -------
    list[tuple[str, int, int, float, float, tuple[float, float]]]
        A list of tuples containing the defect or gap type, start index, end index, length in nm, position along trace
        in nm, and a tuple of total left turn and total right turn in radians.
    """
    defects_and_gaps_with_turns = []

    # Calculate the angles in radians for the trace points
    if circular:
        angles_radians = calculate_discrete_angle_difference_circular(trace_points)
    else:
        angles_radians = calculate_discrete_angle_difference_linear(trace_points)
    for defect_or_gap in defects:
        type_of_region, start_index, end_index, length_nm, position_along_trace_nm = defect_or_gap

        # Calculate the total turn in radians for the region
        total_left_turn, total_right_turn = total_turn_in_region_radians(
            angles_radians=angles_radians,
            region_inclusive=(start_index, end_index),
            circular=circular,
        )

        # Append the defect or gap with the total turn
        defects_and_gaps_with_turns.append(
            (
                type_of_region,
                start_index,
                end_index,
                length_nm,
                position_along_trace_nm,
                (total_left_turn, total_right_turn),
            )
        )

    return defects_and_gaps_with_turns


def calculate_indirect_defect_gaps(  # noqa: C901
    ordered_defect_gap_list: OrderedDefectGapList,
    circular: bool,
) -> list[float]:
    """Calculate all indirect defect gaps."""
    # If there is only one gap, return an empty list
    if len(ordered_defect_gap_list.defect_gap_list) == 0:
        return []
    if len(ordered_defect_gap_list.defect_gap_list) == 1:
        if isinstance(ordered_defect_gap_list.defect_gap_list[0], DefectGap):
            return []
        # If there is only one defect, return an empty list
        if isinstance(ordered_defect_gap_list.defect_gap_list[0], Defect):
            return []

    indirect_gaps = []
    for start_defect_gap_number, this_defect_or_gap in enumerate(ordered_defect_gap_list.defect_gap_list):
        if isinstance(this_defect_or_gap, Defect):
            # Iterate over all other defects
            for other_defect_gap_number, other_defect_or_gap in enumerate(ordered_defect_gap_list.defect_gap_list):
                if isinstance(other_defect_or_gap, Defect):
                    # Iterate over the defects and gaps between the two defects
                    if other_defect_gap_number > start_defect_gap_number:
                        # Sum all the lengths of the gaps and defects between the two defects
                        indirect_gap_length = 0.0
                        for inner_number in range(start_defect_gap_number + 1, other_defect_gap_number):
                            inner_defect_or_gap = ordered_defect_gap_list.defect_gap_list[inner_number]
                            indirect_gap_length += inner_defect_or_gap.length_nm
                        indirect_gaps.append(indirect_gap_length)
                    else:
                        if not circular:
                            # This would require wrapping around the end of the array, so this proposed gap is not
                            # valid.
                            continue
                        # The other defect is before this defect
                        # Sum all the lengths of the regions until the end of the array, then wrap around to the other
                        # defect
                        indirect_gap_length_to_end_of_list = 0.0
                        indirect_gap_length_from_start_of_list = 0.0
                        # Check if it's the last defect in the list
                        if other_defect_gap_number == len(ordered_defect_gap_list.defect_gap_list) - 1:
                            # If it is, then we skip the adding of lengths until end of the list since there are no
                            # defects/gaps past this one.
                            pass
                        else:
                            for inner_number in range(
                                start_defect_gap_number + 1, len(ordered_defect_gap_list.defect_gap_list)
                            ):
                                inner_defect_or_gap = ordered_defect_gap_list.defect_gap_list[inner_number]
                                indirect_gap_length_to_end_of_list += inner_defect_or_gap.length_nm

                        # Calculate length from start of list to other defect
                        for inner_number in range(0, other_defect_gap_number):
                            inner_defect_or_gap = ordered_defect_gap_list.defect_gap_list[inner_number]
                            indirect_gap_length_from_start_of_list += inner_defect_or_gap.length_nm

                        indirect_gaps.append(
                            indirect_gap_length_to_end_of_list + indirect_gap_length_from_start_of_list
                        )

    if not circular:
        # Add any DefectGaps that are at the ends of the list
        for _, this_defect_or_gap in enumerate(ordered_defect_gap_list.defect_gap_list):
            if isinstance(this_defect_or_gap, DefectGap):
                indirect_gap_length = this_defect_or_gap.length_nm
                # insert into start of the list
                indirect_gaps.insert(0, indirect_gap_length)
            else:
                # We have reached a Defect, so we can stop
                break
        for _, this_defect_or_gap in enumerate(reversed(ordered_defect_gap_list.defect_gap_list)):
            if isinstance(this_defect_or_gap, DefectGap):
                indirect_gap_length = this_defect_or_gap.length_nm
                indirect_gaps.append(indirect_gap_length)
            else:
                # We have reached a Defect, so we can stop
                break
    return indirect_gaps


class BaseDamageAnalysis(BaseModel):
    """Data object to hold settings for Models used in the project."""

    model_config = ConfigDict(arbitrary_types_allowed=True)


class UnanalysedMoleculeData(BaseDamageAnalysis):
    """Data object to hold unanalysed molecule data."""

    molecule_id: int
    ordered_coords_heights: npt.NDArray[np.float64]
    spline_coords_heights: npt.NDArray[np.float64]
    distances: npt.NDArray[np.float64]
    circular: bool
    spline_coords: npt.NDArray[np.float64]
    ordered_coords: npt.NDArray[np.float64]
    curvature_data: dict | None


class UnanalysedMoleculeDataCollection(BaseDamageAnalysis):
    """Data object to hold a collection of unanalysed molecule data."""

    molecules: dict[int, UnanalysedMoleculeData]

    def __getitem__(self, key: int) -> UnanalysedMoleculeData:
        """Get the molecule data for a given molecule id."""
        return self.molecules[key]

    def __len__(self) -> int:
        """Get the number of molecules in the collection."""
        return len(self.molecules)

    def __contains__(self, key: int) -> bool:
        """Check if a molecule id is in the collection."""
        return key in self.molecules

    def items(self) -> Generator[tuple[int, UnanalysedMoleculeData], None, None]:
        """Get the items of the molecule data collection, yielding tuples of molecule id and molecule data."""
        return (item for item in self.molecules.items())

    def keys(self) -> Generator[int, None, None]:
        """Get the keys of the molecule data collection."""
        return (key for key in self.molecules.keys())

    def values(self) -> Generator[UnanalysedMoleculeData, None, None]:
        """Get the values of the molecule data collection."""
        return (value for value in self.molecules.values())

    def get(self, key: int, default: UnanalysedMoleculeData | None = None) -> UnanalysedMoleculeData | None:
        """Get the molecule data for a given molecule id, or return a default value if the molecule id is not in the collection."""
        return self.molecules.get(key, default)

    def add_molecule(self, molecule: UnanalysedMoleculeData) -> None:
        """Add a molecule to the collection."""
        self.molecules[molecule.molecule_id] = molecule

    def remove_molecule(self, molecule_id: int) -> None:
        """Remove a molecule from the collection by its molecule id."""
        if molecule_id not in self.molecules:
            raise KeyError(f"molecule with id {molecule_id} not found in collection, cannot remove")
        del self.molecules[molecule_id]


class UnanalysedGrain(BaseDamageAnalysis):
    """Data object to hold unanalysed grain data."""

    global_grain_id: int | None = None
    file_grain_id: int
    filename: str
    pixel_to_nm_scaling: float
    folder: str
    percent_damage: float
    bbox: tuple[int, int, int, int]
    image: npt.NDArray[np.float64]
    aspect_ratio: float
    smallest_bounding_area: float
    total_contour_length: float
    num_crossings: int
    molecule_data_collection: UnanalysedMoleculeDataCollection
    added_left: int
    added_top: int
    padding: int
    mask: npt.NDArray[np.bool_]
    node_coords: npt.NDArray[np.float64]
    num_nodes: int

    def __str__(self) -> str:
        """Return a simplified string representation of the grain."""
        return (
            f"GrainModel(global_grain_id={self.global_grain_id}), {self.percent_damage}% "
            f"damage, from file {self.filename}."
        )

    def plot(self, mask_alpha: float = 0.3) -> None:
        """Plot the grain image with the mask overlaid."""
        plt.imshow(self.image, **IMGPLOTARGS)
        plt.imshow(self.mask[:, :, 1], alpha=mask_alpha, cmap="gray")
        plt.title(f"grain {self.global_grain_id}, {self.percent_damage}% damage")
        plt.show()


class UnanalysedGrainCollection(BaseDamageAnalysis):
    """Data object to hold a collection of unanalysed grains."""

    unanalysed_grains: dict[int, UnanalysedGrain] = Field(default_factory=dict)
    current_global_grain_id: int = 0

    # pretty print
    def __str__(self) -> str:
        """Return a simplified string representation of the grain collection."""
        grain_indexes = range(self.current_global_grain_id)
        missing_grain_indexes = [index for index in grain_indexes if index not in self.unanalysed_grains]
        return (
            f"GrainModelCollection with {len(self.unanalysed_grains)} grains, with {len(missing_grain_indexes)} "
            f"omitted grains: {missing_grain_indexes}"
        )

    def __getitem__(self, key: int) -> UnanalysedGrain:
        """Get a grain from the collection by its global id."""
        return self.unanalysed_grains[key]

    def __len__(self) -> int:
        """Get the number of grains in the collection."""
        return len(self.unanalysed_grains)

    def __contains__(self, key: int) -> bool:
        """Check if a grain with a given global id is in the collection."""
        return key in self.unanalysed_grains

    def items(self) -> Generator[tuple[int, UnanalysedGrain], None, None]:
        """Get the items of the grain collection, yielding tuples of global grain id and grain."""
        for item in self.unanalysed_grains.items():
            yield from item

    def keys(self) -> Generator[int, None, None]:
        """Get the keys of the grain collection."""
        return (key for key in self.unanalysed_grains.keys())

    def values(self) -> Generator[UnanalysedGrain, None, None]:
        """Get the values of the grain collection."""
        return (value for value in self.unanalysed_grains.values())

    def get(self, key: int, default: UnanalysedGrain | None = None) -> UnanalysedGrain | None:
        """Get a grain from the collection by its global id, returning a default value if not present."""
        return self.unanalysed_grains.get(key, default)

    def add_grain(self, unanalysed_grain: UnanalysedGrain) -> None:
        """Add a grain to the collection, assigning it a global grain id."""
        # note: a grain might already have a global grain id if it came from another collection, but we can
        # just overwrite it.
        unanalysed_grain.global_grain_id = self.current_global_grain_id
        self.unanalysed_grains[self.current_global_grain_id] = unanalysed_grain
        self.current_global_grain_id += 1

    def remove_grain(self, global_grain_id: int) -> None:
        """Remove a grain from the collection by its global id."""
        if global_grain_id not in self.unanalysed_grains:
            raise KeyError(f"grain with global id {global_grain_id} not found in collection, cannot remove")
        del self.unanalysed_grains[global_grain_id]

    def remove_grains(self, global_grain_ids: list[int] | set[int]) -> None:
        """Remove multiple grains from the collection by their global ids."""
        global_grain_ids_set = set(global_grain_ids)
        for grain_id in global_grain_ids_set:
            self.remove_grain(grain_id)


def combine_unanalysed_grain_collections(collections: list[UnanalysedGrainCollection]) -> UnanalysedGrainCollection:
    """Combine multiple UnanalysedGrainCollections into a single UnanalysedGrainCollection."""
    combined_collection = UnanalysedGrainCollection(unanalysed_grains={})
    for collection in collections:
        for grain in collection.values():
            combined_collection.add_grain(grain)
    return combined_collection


# class Defect(BaseDamageAnalysis):
#     start_index: int
#     end_index: int
#     length_nm: float
#     position_along_trace_nm: float
#     total_turn_radians: tuple[float, float]

#     def __eq__(self, other: object) -> bool:
#         if not isinstance(other, Defect):
#             raise TypeError(f"Cannot compare Defect with {type(other)}")
#         return (
#             self.start_index == other.start_index
#             and self.end_index == other.end_index
#             and np.isclose(self.length_nm, other.length_nm, rtol=1e-9, atol=1e-12)
#             and np.isclose(self.position_along_trace_nm, other.position_along_trace_nm, rtol=1e-9, atol=1e-12)
#             and np.isclose(self.total_turn_radians[0], other.total_turn_radians[0], rtol=1e-9, atol=1e-12)
#             and np.isclose(self.total_turn_radians[1], other.total_turn_radians[1], rtol=1e-9, atol=1e-12)
#         )


# class DefectGap(BaseDamageAnalysis):
#     start_index: int
#     end_index: int
#     length_nm: float
#     position_along_trace_nm: float


# class OrderedDefectsGaps(BaseDamageAnalysis):
#     defect_gap_list: list[Defect | DefectGap] = Field(default_factory=list)

#     # post-init to sort the list by start index
#     def model_post_init(self, __context: dict | None = None) -> None:
#         self.sort_defect_gap_list()

#     def sort_defect_gap_list(self) -> None:
#         self.defect_gap_list.sort(key=lambda x: x.start_index)

#     def add_item(self, item: Defect | DefectGap) -> None:
#         self.defect_gap_list.append(item)
#         self.sort_defect_gap_list()

#     def __eq__(self, other: object) -> bool:
#         if not isinstance(other, OrderedDefectsGaps):
#             raise TypeError(f"Cannot compare OrderedDefectsGaps with {type(other)}")
#         if len(self.defect_gap_list) != len(other.defect_gap_list):
#             return False

#         for item_self, item_other in zip(self.defect_gap_list, other.defect_gap_list):
#             if item_self != item_other:
#                 return False


class MoleculeDefectData(BaseDamageAnalysis):
    """Data object to hold the defect and gap data for a molecule."""

    ordered_defects_and_gaps: OrderedDefectGapList

    @computed_field
    @property
    def num_defects(self) -> int:
        """Calculate the number of defects."""
        return sum(isinstance(item, Defect) for item in self.ordered_defects_and_gaps.defect_gap_list)

    @computed_field
    @property
    def num_gaps(self) -> int:
        """Calculate the number of gaps."""
        return sum(isinstance(item, DefectGap) for item in self.ordered_defects_and_gaps.defect_gap_list)

    @computed_field
    @property
    def defect_lengths_nm(self) -> list[float]:
        """Get a list of the lengths of the defects in nanometres."""
        return [
            defect_or_gap.length_nm
            for defect_or_gap in self.ordered_defects_and_gaps.defect_gap_list
            if isinstance(defect_or_gap, Defect)
        ]


class GrainDefectData(BaseDamageAnalysis):
    """Data object to hold the defect and gap data for a grain."""

    molecule_defect_data_dict: dict[int, MoleculeDefectData] = Field(default_factory=dict)

    @computed_field
    @property
    def num_defects(self) -> int:
        """Calculate the total number of defects across all molecules."""
        return sum(molecule_defect_data.num_defects for molecule_defect_data in self.molecule_defect_data_dict.values())

    @computed_field
    @property
    def num_gaps(self) -> int:
        """Calculate the total number of gaps across all molecules."""
        return sum(molecule_defect_data.num_gaps for molecule_defect_data in self.molecule_defect_data_dict.values())

    @computed_field
    @property
    def defect_lengths_nm(self) -> list[float]:
        """Get a list of the lengths of all defects across all molecules in nanometres."""
        defect_lengths = []
        for molecule_defect_data in self.molecule_defect_data_dict.values():
            defect_lengths.extend(molecule_defect_data.defect_lengths_nm)
        return defect_lengths


class MoleculeData(UnanalysedMoleculeData):
    """Data object to hold the analysed molecule data."""

    def from_unanalysed_molecule_data(unanalysed_data: UnanalysedMoleculeData) -> "MoleculeData":
        """Create a MoleculeData object from an UnanalysedMoleculeData object."""
        return MoleculeData(
            molecule_id=unanalysed_data.molecule_id,
            ordered_coords_heights=unanalysed_data.ordered_coords_heights,
            spline_coords_heights=unanalysed_data.spline_coords_heights,
            distances=unanalysed_data.distances,
            circular=unanalysed_data.circular,
            spline_coords=unanalysed_data.spline_coords,
            ordered_coords=unanalysed_data.ordered_coords,
            curvature_data=unanalysed_data.curvature_data,
        )


class MoleculeDataCollection(UnanalysedMoleculeDataCollection):
    """Data object to hold a collection of analysed molecule data."""

    molecules: dict[int, MoleculeData]

    def from_unanalysed_molecule_data_collection(
        unanalysed_collection: UnanalysedMoleculeDataCollection,
    ) -> "MoleculeDataCollection":
        """Create a MoleculeDataCollection object from an UnanalysedMoleculeDataCollection object."""
        molecule_data_dict = {}
        for molecule_id, unanalysed_molecule_data in unanalysed_collection.molecules.items():
            molecule_data = MoleculeData.from_unanalysed_molecule_data(unanalysed_molecule_data)
            molecule_data_dict[molecule_id] = molecule_data
        return MoleculeDataCollection(molecules=molecule_data_dict)

    def __getitem__(self, key: int) -> MoleculeData:
        """Get the molecule data for a given molecule id."""
        return self.molecules[key]

    def __len__(self) -> int:
        """Get the number of molecules in the collection."""
        return len(self.molecules)

    def __contains__(self, key: int) -> bool:
        """Check if a molecule id is in the collection."""
        return key in self.molecules

    def items(self) -> Generator[tuple[int, MoleculeData], None, None]:
        """Get the items of the molecule data collection, yielding tuples of molecule id and molecule data."""
        return (item for item in self.molecules.items())

    def keys(self) -> Generator[int, None, None]:
        """Get the keys of the molecule data collection."""
        return (key for key in self.molecules.keys())

    def values(self) -> Generator[MoleculeData, None, None]:
        """Get the values of the molecule data collection."""
        return (value for value in self.molecules.values())

    def get(self, key: int, default: MoleculeData | None = None) -> MoleculeData | None:
        """Get the molecule data for a given molecule id, or return a default value if the molecule id is not in the collection."""
        return self.molecules.get(key, default)

    def add_molecule(self, molecule: MoleculeData) -> None:
        """Add a molecule to the collection."""
        self.molecules[molecule.molecule_id] = molecule

    def remove_molecule(self, molecule_id: int) -> None:
        """Remove a molecule from the collection by its molecule id."""
        if molecule_id not in self.molecules:
            raise KeyError(f"molecule with id {molecule_id} not found in collection, cannot remove")
        del self.molecules[molecule_id]


class GrainModel(UnanalysedGrain):
    """Data object to hold the analysed grain data."""

    curvature_defect_data: GrainDefectData = Field(default_factory=GrainDefectData)
    height_defect_data: GrainDefectData = Field(default_factory=GrainDefectData)
    molecule_data_collection: MoleculeDataCollection

    def from_unanalysed_grain(unanalysed_grain: UnanalysedGrain) -> "GrainModel":
        """Create a GrainModel object from an UnanalysedGrain object."""
        # Create the new molecule data collection
        molecule_data_collection = MoleculeDataCollection.from_unanalysed_molecule_data_collection(
            unanalysed_grain.molecule_data_collection
        )
        return GrainModel(
            global_grain_id=unanalysed_grain.global_grain_id,
            file_grain_id=unanalysed_grain.file_grain_id,
            filename=unanalysed_grain.filename,
            pixel_to_nm_scaling=unanalysed_grain.pixel_to_nm_scaling,
            folder=unanalysed_grain.folder,
            percent_damage=unanalysed_grain.percent_damage,
            bbox=unanalysed_grain.bbox,
            image=unanalysed_grain.image,
            aspect_ratio=unanalysed_grain.aspect_ratio,
            smallest_bounding_area=unanalysed_grain.smallest_bounding_area,
            total_contour_length=unanalysed_grain.total_contour_length,
            num_crossings=unanalysed_grain.num_crossings,
            molecule_data_collection=molecule_data_collection,
            added_left=unanalysed_grain.added_left,
            added_top=unanalysed_grain.added_top,
            padding=unanalysed_grain.padding,
            mask=unanalysed_grain.mask,
            node_coords=unanalysed_grain.node_coords,
            num_nodes=unanalysed_grain.num_nodes,
        )

    def __str__(self) -> str:
        """Return a simplified string representation of the grain."""
        return (
            f"GrainModel(global_grain_id={self.global_grain_id}), {self.percent_damage}% damage, "
            f"with {len(self.molecule_data_collection)} molecules, {self.num_crossings} crossings "
            f"from file {self.filename}."
        )

    def plot(  # noqa: C901
        self, mask_alpha: float = 0.3, linemode: str = "", curvature_defects: bool = False, height_defects: bool = False
    ) -> None:
        """Plot the grain image with the mask and molecule data overlaid."""
        plt.imshow(self.image, **IMGPLOTARGS)
        plt.imshow(self.mask[:, :, 1], alpha=mask_alpha, cmap="gray")
        if linemode == "spline":
            for _molecule_id, molecule_data in self.molecule_data_collection.items():
                spline_coords = molecule_data.spline_coords
                plt.plot(spline_coords[:, 1], spline_coords[:, 0])
        elif linemode == "curvature":
            for _molecule_id, molecule_data in self.molecule_data_collection.items():
                spline_coords = molecule_data.spline_coords
                curvature_data = molecule_data.curvature_data
                if curvature_data is not None:
                    curvature_values = curvature_data["smoothed_curvatures"]
                    # plot the curvature values as a colormap along the spline coords
                    assert len(curvature_values) == len(spline_coords), (
                        f"length of curvature values {len(curvature_values)} does not match"
                        f"length of spline coords {len(spline_coords)}"
                    )
                    curvature_norm_bounds_lower = -0.1
                    curvature_norm_bounds_upper = 0.1
                    curvature_values_clipped = np.clip(
                        curvature_values, curvature_norm_bounds_lower, curvature_norm_bounds_upper
                    )
                    curvature_values_normalised = (curvature_values_clipped - curvature_norm_bounds_lower) / (
                        curvature_norm_bounds_upper - curvature_norm_bounds_lower
                    )
                    curvature_cmap = plt.get_cmap("coolwarm")
                    for index, point in enumerate(spline_coords):
                        color = curvature_cmap(curvature_values_normalised[index])
                        if index > 0:
                            previous_point = spline_coords[index - 1]
                            plt.plot(
                                [previous_point[1], point[1]],
                                [previous_point[0], point[0]],
                                color=color,
                                linewidth=1,
                            )
        if curvature_defects:
            # plot all the curvature defects as pink dots
            for molecule_id, molecule_defect_data in self.curvature_defect_data.molecule_defect_data_dict.items():
                for item in molecule_defect_data.ordered_defects_and_gaps.defect_gap_list:
                    if isinstance(item, Defect):
                        defect_start_index = item.start_index
                        defect_end_index = item.end_index
                        spline_coords = self.molecule_data_collection[molecule_id].spline_coords
                        defect_coords = spline_coords[defect_start_index:defect_end_index]
                        plt.scatter(defect_coords[:, 1], defect_coords[:, 0], color="magenta", s=10)
        if height_defects:
            # plot all the height defects as cyan dots
            for molecule_id, molecule_defect_data in self.height_defect_data.molecule_defect_data_dict.items():
                for item in molecule_defect_data.ordered_defects_and_gaps.defect_gap_list:
                    if isinstance(item, Defect):
                        defect_start_index = item.start_index
                        defect_end_index = item.end_index
                        spline_coords = self.molecule_data_collection[molecule_id].spline_coords
                        defect_coords = spline_coords[defect_start_index:defect_end_index]
                        plt.scatter(defect_coords[:, 1], defect_coords[:, 0], color="cyan", s=10)
        plt.show()


class GrainCollection(BaseDamageAnalysis):
    """Data object to hold a collection of analysed grains."""

    grains: dict[int, GrainModel]
    current_global_grain_id: int = 0

    def __str__(self) -> str:
        """Return a simplified string representation of the grain collection."""
        grain_indexes = range(self.current_global_grain_id)
        missing_grain_indexes = [index for index in grain_indexes if index not in self.grains]
        return (
            f"GrainModelCollection with {len(self.grains)} grains, with {len(missing_grain_indexes)} "
            f"omitted grains: {missing_grain_indexes}"
        )

    def __getitem__(self, key: int) -> GrainModel:
        """Get a grain from the collection by its global id."""
        return self.grains[key]

    def __len__(self) -> int:
        """Get the number of grains in the collection."""
        return len(self.grains)

    def __contains__(self, key: int) -> bool:
        """Check if a grain with a given global id is in the collection."""
        return key in self.grains

    def items(self) -> Generator[tuple[int, GrainModel], None, None]:
        """Get the items of the grain collection, yielding tuples of global grain id and grain."""
        return (item for item in self.grains.items())

    def keys(self) -> Generator[int, None, None]:
        """Get the keys of the grain collection."""
        return (key for key in self.grains.keys())

    def values(self) -> Generator[GrainModel, None, None]:
        """Get the values of the grain collection."""
        return (value for value in self.grains.values())

    def get(self, key: int, default: GrainModel | None = None) -> GrainModel | None:
        """Get a grain from the collection by its global id, returning a default value if not present."""
        return self.grains.get(key, default)

    def add_grain(self, grain_model: GrainModel) -> None:
        """Add a grain to the collection, assigning it a global grain id."""
        # note: a grain might already have a global grain id if it came from another collection, but we can
        # just overwrite it.
        grain_model.global_grain_id = self.current_global_grain_id
        self.grains[self.current_global_grain_id] = grain_model
        self.current_global_grain_id += 1

    def remove_grain(self, global_grain_id: int) -> None:
        """Remove a grain from the collection by its global id."""
        if global_grain_id not in self.grains:
            raise KeyError(f"grain with global id {global_grain_id} not found in collection, cannot remove")
        del self.grains[global_grain_id]

    def remove_grains(self, global_grain_ids: list[int] | set[int]) -> None:
        """Remove multiple grains from the collection by their global ids."""
        global_grain_ids_set = set(global_grain_ids)
        for grain_id in global_grain_ids_set:
            self.remove_grain(grain_id)

    def combine_with_other_collection(self, other_collection: "GrainCollection") -> None:
        """Combine another GrainCollection into this GrainCollection."""
        for grain_model in other_collection.values():
            self.add_grain(grain_model)

    def from_unanalysed_grain_collection(
        unanalysed_collection: UnanalysedGrainCollection,
    ) -> "GrainCollection":
        """Create a GrainCollection object from an UnanalysedGrainCollection object."""
        grain_dict = {}
        for global_grain_id, unanalysed_grain in unanalysed_collection.unanalysed_grains.items():
            grain_model = GrainModel.from_unanalysed_grain(unanalysed_grain)
            grain_dict[global_grain_id] = grain_model
        return GrainCollection(grains=grain_dict, current_global_grain_id=unanalysed_collection.current_global_grain_id)
