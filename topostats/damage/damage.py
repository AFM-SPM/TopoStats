"""Functions for damage detection and quantification."""

from dataclasses import dataclass
from topostats.measure.curvature import total_turn_in_region_radians
import numpy as np
import numpy.typing as npt


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
            if type(self_item) != type(other_item):
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
            if not np.isclose(self_item.total_turn_radians, other_item.total_turn_radians, rtol=1e-9, atol=1e-12):
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
    """Calculate the distance of a region in the trace.

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


def get_defects_and_gaps_from_bool_array(
    defects_bool: npt.NDArray[np.bool_],
    trace_points_nm: npt.NDArray[np.number],
    circular: bool,
    distance_to_previous_points_nm: npt.NDArray[np.float64],
) -> OrderedDefectGapList:
    """Get the Defects and DefectGaps from a boolean array of defects and gaps.

    Parameters
    ----------
    defects_bool : npt.NDArray[np.bool_]
        A boolean array where True indicates a defect and False indicates no defect.
    trace_points : npt.NDArray[np.number]
        The coordinate trace points in nanometre units.
    circular : bool
        If True, the trace is treated as circular, meaning that the end of the trace wraps around to the start.
        If False, the trace is treated as linear, meaning that the end of the trace does not wrap around to the start.
    distance_to_previous_points_nm : npt.NDArray[np.float64]
        An array of distances to the previous points in nanometers. This is used to calculate the lengths of the
        defects and gaps.

    Returns
    -------
    OrderedDefectGapList
        An ordered list of Defect and DefectGap objects, sorted by the start index of the defect or gap.
    """

    if circular:
        defects, gaps = get_defects_and_gaps_circular(defects_bool=defects_bool)
    else:
        defects, gaps = get_defects_and_gaps_linear(defects_bool=defects_bool)

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
        # Note that if ciruclar, then the distance to the previous point at the starting index, is non zero, so we need
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
    trace_points: npt.NDArray[np.number],
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

    for defect_or_gap in defects:
        type_of_region, start_index, end_index, length_nm, position_along_trace_nm = defect_or_gap
        # Get the trace points for
        # the region
        if start_index <= end_index:
            trace_region = trace_points[start_index : end_index + 1]
        else:
            if not circular:
                raise ValueError(
                    f"Cannot calculate turns for region {start_index} to {end_index} in a linear array. "
                    "Start index cannot be greater than end index in a linear array."
                )
            # The region wraps around the end of the array
            trace_region = np.concatenate((trace_points[start_index:], trace_points[: end_index + 1]))
        # Calculate the total turn in radians for the region
        total_left_turn, total_right_turn = total_turn_in_region_radians(trace=trace_region)

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


def calculate_indirect_defect_gaps(
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
