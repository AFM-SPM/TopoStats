"""Functions for finding and quantifying defects in grains."""

from typing import Literal

import numpy as np
import numpy.typing as npt

from topostats.array_manipulation import calculate_distance_of_region
from topostats.classes import Defect, Gap, MoleculeDefectData, OrderedDefectGapList, TopoStats
from topostats.measure.curvature import (
    calculate_discrete_angle_difference_circular,
    calculate_discrete_angle_difference_linear,
    total_turn_in_region_radians,
)


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

    # If there are no gaps (if it's all defect), then do nothing.
    if len(gaps) == 0:
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


def remove_too_short_defects(  # noqa: C901
    defects: list[tuple[int, int]],
    gaps: list[tuple[int, int]],
    distance_to_previous_points_nm: npt.NDArray[np.float64],
    defect_minimum_length_nm: float,
    circular: bool,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Remove defects that are shorter than the minimum length.

    Defects and gaps are stored in a list of tuples.

    defects: [(start_index, end_index), (start_index, end_index), ...]
    gaps: [(start_index, end_index), (start_index, end_index), ...]

    Defects and gaps align, meaning that the first defect is followed by the first gap, etc.

    Parameters
    ----------
    defects : list[tuple[int, int]]
        A list of tuples containing the start and end indexes of defects.
    gaps : list[tuple[int, int]]
        A list of tuples containing the start and end indexes of gaps.
    distance_to_previous_points_nm : npt.NDArray[np.float64]
        An array of distances to the previous points in nanometres.
    defect_minimum_length_nm : float
        The minimum length of a defect in nanometres. Defects shorter than this length will be removed.
    circular : bool
        If True, the trace is treated as circular, meaning that the end of the trace wraps around to the start.

    Returns
    -------
    tuple[list[tuple[int, int]], list[tuple[int, int]]]
        A tuple containing two lists:
        - The defects as a list of tuples of start and end indexes.
        - The gaps as a list of tuples of start and end indexes.
    """
    # Remove defects that are shorter than the minimum length, and merge any adjacent gaps.
    filtered_defects: list[tuple[int, int]] = []
    filtered_gaps: list[tuple[int, int]] = []

    # If there are no defects, do nothing
    if len(defects) == 0:
        return defects, gaps

    # If there are no gaps, do nothing
    if len(gaps) == 0:
        return defects, gaps

    # Determine if we start with a defect or a gap.
    first_defect = defects[0]
    first_gap = gaps[0]
    gap_index = 0
    if first_gap[0] < first_defect[0]:
        # first gap comes before the first defect, add the gap
        filtered_gaps.append(first_gap)
        gap_index = 1
        # Check if there is not a second gap
        if len(gaps) == 1:
            # If there is only one gap, then have to check if the first defect is too short here
            if len(defects) == 1:
                start_index, end_index = first_defect
                length_nm = calculate_distance_of_region(
                    start_index=start_index,
                    end_index=end_index,
                    distance_to_previous_points_nm=distance_to_previous_points_nm,
                    circular=circular,
                )
                if length_nm >= defect_minimum_length_nm:
                    # print(f"keeping defect {first_defect} with length {length_nm:.2f} nm")
                    # Keep the defect
                    filtered_defects.append(first_defect)
                else:
                    # print(f"removing defect {first_defect} with length {length_nm:.2f} nm")
                    # Don't add the defect
                    # Remove the defect and merge the gap with the previous gap
                    # There is guaranteed to be a previous gap but check anyway
                    assert len(filtered_gaps) > 0, "There should be a previous gap to merge with."
                    previous_gap = filtered_gaps[-1]
                    merged_gap = (previous_gap[0], first_gap[1])
                    # Apply the merged gap to the last gap in the filtered gaps list
                    filtered_gaps[-1] = merged_gap
            else:
                raise ValueError("There is only one gap, but more than one defect. This should not happen.")

            # Nothing more needs doing, all defects and gaps have been processed
            return filtered_defects, filtered_gaps

    # Filter out defects that are too short
    for defect in defects:
        # Check if there is a corresponding gap for this defect, if not, then we are at the end of the trace
        if gap_index >= len(gaps):
            # If there is no corresponding gap, then we are at the end of the trace, so we can just add the defect if it
            # is long enough and return.
            start_index, end_index = defect
            length_nm = calculate_distance_of_region(
                start_index=start_index,
                end_index=end_index,
                distance_to_previous_points_nm=distance_to_previous_points_nm,
                circular=circular,
            )
            if length_nm >= defect_minimum_length_nm:
                # print(f"keeping defect {defect} with length {length_nm:.2f} nm")
                # Keep the defect
                filtered_defects.append(defect)
            else:
                # print(f"removing defect {defect} with length {length_nm:.2f} nm")
                # Don't add the defect
                # Remove the defect and merge the gap with the previous gap
                if len(filtered_gaps) > 0:
                    previous_gap = filtered_gaps[-1]
                    merged_gap = (previous_gap[0], previous_gap[1])
                    # Apply the merged gap to the last gap in the filtered gaps list
                    filtered_gaps[-1] = merged_gap
            break
        gap = gaps[gap_index]
        start_index, end_index = defect
        length_nm = calculate_distance_of_region(
            start_index=start_index,
            end_index=end_index,
            distance_to_previous_points_nm=distance_to_previous_points_nm,
            circular=circular,
        )
        if length_nm >= defect_minimum_length_nm:
            # print(f"keeping defect {defect} with length {length_nm:.2f} nm")
            # Keep the defect
            filtered_defects.append(defect)
            # Keep the gap that follows the defect
            filtered_gaps.append(gap)
        else:
            # print(f"removing defect {defect} with length {length_nm:.2f} nm")
            # Don't add the defect
            # Remove the defect and merge the gap with the previous gap
            if len(filtered_gaps) > 0:
                previous_gap = filtered_gaps[-1]
                merged_gap = (previous_gap[0], gap[1])
                # Apply the merged gap to the last gap in the filtered gaps list
                filtered_gaps[-1] = merged_gap
            else:
                # If there is no previous gap, we are at the start of the trace, handle this later
                pass

        gap_index += 1

    # If there are no defects, left, then the whole trace is a gap
    if len(filtered_defects) == 0:
        # Set the filtered gaps to be the whole trace
        filtered_gaps = [(0, len(distance_to_previous_points_nm) - 1)]
        return filtered_defects, filtered_gaps

    # Sort out the start and end gaps

    first_defect = filtered_defects[0]
    first_defect_first_index = first_defect[0]
    if first_defect_first_index == 0:
        # The first defect starts at the start of the trace, so we don't need to add / merge a start gap
        pass
    else:
        # If the first gap starts at the start of the trace, then we might need to merge it with the last gap if
        # the trace is circular. Else we might need to create a new gap at the start of the trace if linear.
        if len(filtered_gaps) > 0:
            first_gap = filtered_gaps[0]
            if first_gap[0] == 0:
                # First gap starts at the start of the trace, so don't need to add but might need to merge
                if circular:
                    last_gap = filtered_gaps[-1]
                    last_defect = filtered_defects[-1]
                    # Check if the trace ends in a gap
                    if last_gap[1] > last_defect[1]:
                        # The trace ends in a gap, so need to merge the first and last gaps
                        merged_gap = (last_gap[0], first_gap[1])
                        filtered_gaps[-1] = merged_gap
                    else:
                        # The circular trace ends in a defect, so can skip merging
                        pass
                else:
                    # Linear with first gap at start, so can skip merging
                    pass
            else:
                # First gap does not start at the start of the trace, so might need to create a new gap at the start
                # of the trace if linear or if the circular trace ends in a defect.
                if circular:
                    # If circular trace ends in a defect, we can create a new gap at the start of the trace
                    last_gap = filtered_gaps[-1]
                    last_defect = filtered_defects[-1]
                    if last_gap[1] > last_defect[1]:
                        # The circular trace ends in a gap, so we need to extend the last gap to the start of the first
                        # defect
                        merged_gap = (last_gap[0], first_defect[0] - 1)
                        filtered_gaps[-1] = merged_gap
                    else:
                        # The circular trace ends in a defect, so we can create a new gap at the start of the trace
                        new_gap = (0, first_gap[1])
                        filtered_gaps.insert(0, new_gap)
                else:
                    # Linear trace so just create a new gap at the start of the trace
                    new_gap = (0, first_gap[1])
                    filtered_gaps.insert(0, new_gap)
        else:
            # If the first defect starts at the start of the trace, create a gap at the end
            if filtered_defects[0][0] == 0:
                # Defect starts at start of trace
                new_gap = (first_defect[1] + 1, len(distance_to_previous_points_nm) - 1)
                filtered_gaps.append(new_gap)
            else:
                # Prepend a gap at the start of the trace
                new_gap = (0, first_defect[0] - 1)
                filtered_gaps.insert(0, new_gap)

    return filtered_defects, filtered_gaps


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
        position_along_trace_nm = float(np.sum(distance_to_previous_points_nm[1 : midpoint_index + 1]))

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
        position_along_trace_nm = float(np.sum(distance_to_previous_points_nm[1 : midpoint_index + 1]))

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


def calculate_defect_depth(
    start_index: int,
    end_index: int,
    trace_heights_nm: npt.NDArray[np.float64],
) -> float:
    """Calculate the depth of a defect."""
    if start_index <= end_index:
        defect_heights = trace_heights_nm[start_index : end_index + 1]
    else:
        defect_heights = np.concatenate((trace_heights_nm[start_index:], trace_heights_nm[: end_index + 1]))
    return float(np.min(defect_heights))


def calculate_defect_volume(
    start_index: int,
    end_index: int,
    trace_heights_nm: npt.NDArray[np.float64],
    trace_distances_to_previous_points_nm: npt.NDArray[np.float64],
) -> float:
    """Calculate the volume of a defect."""
    if start_index <= end_index:
        defect_heights = trace_heights_nm[start_index : end_index + 1]
        defect_distances_to_previous_points_nm = trace_distances_to_previous_points_nm[start_index : end_index + 1]
        defect_distances_nm = np.cumsum(defect_distances_to_previous_points_nm)
    else:
        defect_heights = np.concatenate((trace_heights_nm[start_index:], trace_heights_nm[: end_index + 1]))
        defect_distances_to_previous_points_nm = np.concatenate(
            (
                trace_distances_to_previous_points_nm[start_index:],
                trace_distances_to_previous_points_nm[: end_index + 1],
            )
        )
        defect_distances_nm = np.cumsum(defect_distances_to_previous_points_nm)
    # calculate the volume of the defect by integrating the area under the curve of the defect heights
    defect_volume = np.trapezoid(defect_heights, x=defect_distances_nm)  # type: ignore - np.trapezoid is not
    # recognised by ty.
    assert defect_volume >= 0, f"defect volume is negative: {defect_volume}"
    return float(defect_volume)


def get_defects_and_gaps_from_bool_array(
    defects_bool: npt.NDArray[np.bool_],
    trace_points_nm: npt.NDArray[np.float64],
    trace_heights_nm: npt.NDArray[np.float64],
    circular: bool,
    distance_to_previous_points_nm: npt.NDArray[np.float64],
    defect_minimum_length_nm: float,
    connect_close_defect_threshold_nm: float | None,
) -> OrderedDefectGapList:
    """
    Get the Defects and Gaps from a boolean array of defects and gaps.

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
    defect_minimum_length_nm : float
        The minimum length of a defect in nanometers. Defects shorter than this length will be removed.
    connect_close_defect_threshold_nm : float
        The threshold in nanometers for considering two defects as close.

    Returns
    -------
    OrderedGapList
        An ordered list of Defect and Gap objects, sorted by the start index of the defect or gap.
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

        defects, gaps = remove_too_short_defects(
            defects=defects,
            gaps=gaps,
            distance_to_previous_points_nm=distance_to_previous_points_nm,
            defect_minimum_length_nm=defect_minimum_length_nm,
            circular=circular,
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

    # Create an OrderedGapList and add the defects and gaps
    ordered_defect_gap_list = OrderedDefectGapList()
    for defect_or_gap in defect_gap_list_with_turns:
        type_of_region, start_index, end_index, length_nm, position_along_trace_nm, total_turn_radians = defect_or_gap
        if type_of_region == "defect":
            defect_depth_nm = calculate_defect_depth(
                start_index=start_index,
                end_index=end_index,
                trace_heights_nm=trace_heights_nm,
            )
            # convert distances to previous points to absolute distances
            defect_volume_nm3 = calculate_defect_volume(
                start_index=start_index,
                end_index=end_index,
                trace_heights_nm=trace_heights_nm,
                trace_distances_to_previous_points_nm=distance_to_previous_points_nm,
            )
            ordered_defect_gap_list.add_item(
                Defect(
                    start_index=start_index,
                    end_index=end_index,
                    length_nm=length_nm,
                    position_along_trace_nm=position_along_trace_nm,
                    total_turn_radians=total_turn_radians,
                    depth_nm=defect_depth_nm,
                    volume_nm3=defect_volume_nm3,
                )
            )
        elif type_of_region == "gap":
            ordered_defect_gap_list.add_item(
                Gap(
                    start_index=start_index,
                    end_index=end_index,
                    length_nm=length_nm,
                    position_along_trace_nm=position_along_trace_nm,
                    total_turn_radians=total_turn_radians,
                    depth_nm=np.nan,
                    volume_nm3=np.nan,
                )
            )

    return ordered_defect_gap_list


def find_curvature_defects(  # noqa: C901
    topostats_object: TopoStats,
    curvature_defect_method: Literal["iqr", "absolute", "turn_in_distance"],
    curvature_threshold_iqr_multiplier: float,
    curvature_threshold_absolute_pernm: float,
    curvature_turn_in_distance_turn_threshold_deg: float,
    connect_close_defect_threshold_nm: float | None,
    minimum_defect_length_nm: float,
) -> set[str]:
    """
    Find curvature defects for all molecules in all grains in the grain collection.

    Parameters
    ----------
    topostats_object : TopoStats
        TopoStats object to populate with defect data.
    curvature_defect_method : Literal["iqr", "absolute", "turn_in_distance"]
        The method to use for finding curvature defects. Options are "iqr" for interquartile range method or "absolute"
        for an absolute threshold in inverse nanometres.
    curvature_threshold_iqr_multiplier : float
        The multiplier for the interquartile range when using the "iqr" method for curvature defect detection.
    curvature_threshold_absolute_pernm : float
        The absolute threshold in inverse nanometres for curvature defect detection when using the "absolute"
        method.
    curvature_turn_in_distance_turn_threshold_deg : float
        The turn threshold in degrees for the "turn_in_distance" method for curvature defect detection.
    connect_close_defect_threshold_nm : float | None
        The distance in nanometres between defects below which two defects will be connected into one defect.
    minimum_defect_length_nm : float
        The minimum length in nanometres for a defect to be considered valid. Defects shorter than this length will be removed.

    Returns
    -------
    set[int]
        A set of global grain IDs for which curvature defect detection failed.
    """
    # find curvature defects
    bad_grains = set()
    if curvature_defect_method == "iqr":
        # iterate over each grain
        for _grain_id, grain in topostats_object.require_grain_crops().items():
            unique_grain_id = grain.unique_grain_id
            for molecule_id, molecule in grain.molecules.items():
                # for molecule_id, molecule_data in grain_model.molecule_data_collection.items():
                molecule_data_curvature_data = molecule.curvature_data
                trace_heights_nm = molecule.smoothed_spline_coords_heights
                assert trace_heights_nm is not None
                if molecule_data_curvature_data is None:
                    print(
                        f"no curvature data for grain {unique_grain_id} molecule {molecule_id}, skipping curvature defect detection for this molecule"
                    )
                    bad_grains.add(unique_grain_id)
                    continue
                curvatures = molecule_data_curvature_data.curvatures
                curvatures_abs = np.abs(curvatures)
                assert isinstance(
                    curvatures, np.ndarray
                ), f"expected curvatures to be a numpy array, but got {type(curvatures)}"
                # calculate the curvature thresholds
                iqr = np.percentile(curvatures_abs, 75) - np.percentile(curvatures_abs, 25)
                # the threshold is the median + factor * iqr
                curvature_threshold_iqr = curvature_threshold_iqr_multiplier * iqr + np.percentile(curvatures_abs, 50)
                curvature_defects_bool = curvatures_abs > curvature_threshold_iqr
                ordered_defect_gap_list = get_defects_and_gaps_from_bool_array(
                    defects_bool=curvature_defects_bool,
                    trace_points_nm=molecule.spline_coords_nm,
                    trace_heights_nm=trace_heights_nm,
                    distance_to_previous_points_nm=molecule.distances_between_points_nm,
                    circular=molecule.circular,
                    defect_minimum_length_nm=minimum_defect_length_nm,
                    connect_close_defect_threshold_nm=connect_close_defect_threshold_nm,
                )

                molecule.curvature_defect_data = MoleculeDefectData(ordered_defects_and_gaps=ordered_defect_gap_list)
    elif curvature_defect_method == "absolute":
        for _grain_id, grain in topostats_object.require_grain_crops().items():
            unique_grain_id = grain.unique_grain_id
            for molecule_id, molecule in grain.molecules.items():
                molecule_data_curvature_data = molecule.curvature_data
                heights_nm = molecule.smoothed_spline_coords_heights
                assert heights_nm is not None
                if molecule_data_curvature_data is None:
                    print(
                        f"no curvature data for grain {unique_grain_id} molecule {molecule_id}, skipping curvature defect detection for this molecule"
                    )
                    bad_grains.add(unique_grain_id)
                    continue
                curvatures = molecule_data_curvature_data.curvatures
                curvatures_abs = np.abs(curvatures)
                assert isinstance(
                    curvatures, np.ndarray
                ), f"expected curvatures to be a numpy array, but got {type(curvatures)}"
                curvature_threshold_absolute = curvature_threshold_absolute_pernm / grain.pixel_to_nm_scaling
                curvature_defects_bool = curvatures_abs > curvature_threshold_absolute

                ordered_defect_gap_list = get_defects_and_gaps_from_bool_array(
                    defects_bool=curvature_defects_bool,
                    trace_points_nm=molecule.spline_coords_nm,
                    trace_heights_nm=heights_nm,
                    distance_to_previous_points_nm=molecule.distances_between_points_nm,
                    circular=molecule.circular,
                    defect_minimum_length_nm=minimum_defect_length_nm,
                    connect_close_defect_threshold_nm=connect_close_defect_threshold_nm,
                )

                molecule.curvature_defect_data = MoleculeDefectData(ordered_defects_and_gaps=ordered_defect_gap_list)
    elif curvature_defect_method == "turn_in_distance":
        for _grain_id, grain in topostats_object.require_grain_crops().items():
            unique_grain_id = grain.unique_grain_id
            for molecule_id, molecule in grain.molecules.items():
                molecule_data_curvature_data = molecule.curvature_data
                heights_nm = molecule.smoothed_spline_coords_heights
                assert heights_nm is not None
                if molecule_data_curvature_data is None:
                    print(
                        f"no curvature data for grain {unique_grain_id} molecule {molecule_id}, skipping curvature defect detection for this molecule"
                    )
                    bad_grains.add(unique_grain_id)
                    continue
                turn_in_distances = molecule_data_curvature_data.turn_in_distances_deg
                if turn_in_distances is None:
                    print(
                        f"no turn in distance data for grain {unique_grain_id} molecule {molecule_id}, skipping curvature defect detection for this molecule"
                    )
                    bad_grains.add(unique_grain_id)
                    continue
                curvature_defects_bool = np.abs(turn_in_distances) > curvature_turn_in_distance_turn_threshold_deg
                ordered_defect_gap_list = get_defects_and_gaps_from_bool_array(
                    defects_bool=curvature_defects_bool,
                    trace_points_nm=molecule.spline_coords_nm,
                    trace_heights_nm=heights_nm,
                    distance_to_previous_points_nm=molecule.distances_between_points_nm,
                    circular=molecule.circular,
                    defect_minimum_length_nm=minimum_defect_length_nm,
                    connect_close_defect_threshold_nm=connect_close_defect_threshold_nm,
                )

                molecule.curvature_defect_data = MoleculeDefectData(ordered_defects_and_gaps=ordered_defect_gap_list)
    else:
        raise ValueError(f"Invalid curvature defect method: {curvature_defect_method}")

    return bad_grains


def find_height_defects(  # noqa: C901
    topostats_object: TopoStats,
    height_defect_method: Literal["iqr", "absolute", "percentage_of_median"],
    height_threshold_iqr_multiplier: float,
    height_threshold_absolute_nm: float,
    height_threshold_percentage_of_median: float,
    connect_close_defect_threshold_nm: float | None,
    minimum_defect_length_nm: float,
) -> set[str]:
    """
    Find height defects for all molecules in all grains in the grain collection.

    Parameters
    ----------
    topostats_object : TopoStats
        TopoStats object to populate with defect data.
    height_defect_method : Literal["iqr", "absolute"]
        The method to use for finding height defects. Options are "iqr" for interquartile range method or "absolute"
        for an absolute threshold in nanometres.
    height_threshold_iqr_multiplier : float
        The multiplier for the interquartile range when using the "iqr" method for height defect detection.
    height_threshold_absolute_nm : float
        The absolute threshold in nanometres for height defect detection when using the "absolute" method.
    height_threshold_percentage_of_median : float
        The percentage of the median height of each *grain* to use as a threshold for height defect detection when
        using the "percentage_of_median" method.
    connect_close_defect_threshold_nm : float | None
        The distance in nanometres between defects below which two defects will be connected into one defect.
    minimum_defect_length_nm : float
        The minimum allowed length of a defect - else removed.

    Returns
    -------
    set[str]
        A set of global grain IDs for which height defect detection failed.
    """
    bad_grains = set()
    if height_defect_method == "iqr":
        for _grain_id, grain in topostats_object.require_grain_crops().items():
            unique_grain_id = grain.unique_grain_id
            for molecule_id, molecule in grain.molecules.items():
                heights_nm = molecule.smoothed_spline_coords_heights
                assert heights_nm is not None
                spline_coords_nm = molecule.spline_coords_nm
                assert len(heights_nm) == len(spline_coords_nm), (
                    f"length of heights does not match length of spline coords "
                    f"for grain {unique_grain_id} molecule {molecule_id}, got {len(heights_nm)} heights and "
                    f"{len(spline_coords_nm)} spline coords"
                )
                iqr = np.percentile(heights_nm, 75) - np.percentile(heights_nm, 25)
                height_threshold_iqr = np.percentile(heights_nm, 50) - height_threshold_iqr_multiplier * iqr
                print(f"height threshold iqr: {height_threshold_iqr}")
                height_defects_bool = heights_nm < height_threshold_iqr

                ordered_defect_gap_list = get_defects_and_gaps_from_bool_array(
                    defects_bool=height_defects_bool,
                    trace_points_nm=spline_coords_nm,
                    trace_heights_nm=heights_nm,
                    distance_to_previous_points_nm=molecule.distances_between_points_nm,
                    circular=molecule.circular,
                    connect_close_defect_threshold_nm=connect_close_defect_threshold_nm,
                    defect_minimum_length_nm=minimum_defect_length_nm,
                )

                molecule.height_defect_data = MoleculeDefectData(ordered_defects_and_gaps=ordered_defect_gap_list)
    elif height_defect_method == "absolute":
        for _grain_id, grain in topostats_object.require_grain_crops().items():
            for _molecule_id, molecule in grain.molecules.items():
                heights_nm = molecule.smoothed_spline_coords_heights
                assert heights_nm is not None
                spline_coords_nm = molecule.spline_coords_nm
                height_defects_bool = heights_nm < height_threshold_absolute_nm

                ordered_defect_gap_list = get_defects_and_gaps_from_bool_array(
                    defects_bool=height_defects_bool,
                    trace_points_nm=spline_coords_nm,
                    trace_heights_nm=heights_nm,
                    distance_to_previous_points_nm=molecule.distances_between_points_nm,
                    circular=molecule.circular,
                    defect_minimum_length_nm=minimum_defect_length_nm,
                    connect_close_defect_threshold_nm=connect_close_defect_threshold_nm,
                )

                molecule.height_defect_data = MoleculeDefectData(ordered_defects_and_gaps=ordered_defect_gap_list)
    elif height_defect_method == "percentage_of_median":
        for _grain_id, grain in topostats_object.require_grain_crops().items():
            all_grain_heights = []
            for _molecule_id, molecule in grain.molecules.items():
                heights_nm = molecule.smoothed_spline_coords_heights
                assert heights_nm is not None
                all_grain_heights.extend(heights_nm)
            all_grain_heights = np.array(all_grain_heights)
            median_height = np.median(all_grain_heights)
            height_threshold_percentage_of_median_value = median_height * height_threshold_percentage_of_median
            for _molecule_id, molecule in grain.molecules.items():
                heights_nm = molecule.smoothed_spline_coords_heights
                assert heights_nm is not None
                spline_coords_nm = molecule.spline_coords_nm
                height_defects_bool = heights_nm < height_threshold_percentage_of_median_value

                ordered_defect_gap_list = get_defects_and_gaps_from_bool_array(
                    defects_bool=height_defects_bool,
                    trace_points_nm=spline_coords_nm,
                    trace_heights_nm=heights_nm,
                    distance_to_previous_points_nm=molecule.distances_between_points_nm,
                    circular=molecule.circular,
                    connect_close_defect_threshold_nm=connect_close_defect_threshold_nm,
                    defect_minimum_length_nm=minimum_defect_length_nm,
                )

                molecule.height_defect_data = MoleculeDefectData(ordered_defects_and_gaps=ordered_defect_gap_list)
    else:
        raise ValueError(f"Invalid height defect method: {height_defect_method}")

    return bad_grains
