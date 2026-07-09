"""Functions for processing damage data."""

from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.ndimage import distance_transform_edt

from topostats.array_manipulation import calculate_distance_of_region, distances_nm, rolling_average
from topostats.damage.classes import (
    Defect,
    Gap,
    GrainCollection,
    MoleculeData,
    MoleculeDefectData,
    OrderedDefectGapList,
)
from topostats.measure.curvature import (
    angle_diff_signed,
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


def get_defects_and_gaps_from_bool_array(
    defects_bool: npt.NDArray[np.bool_],
    trace_points_nm: npt.NDArray[np.float64],
    trace_heights_nm: npt.NDArray[np.float64],
    circular: bool,
    distance_to_previous_points_nm: npt.NDArray[np.float64],
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


def calculate_indirect_defect_gaps(  # noqa: C901
    ordered_defect_gap_list: OrderedDefectGapList,
    circular: bool,
) -> list[float]:
    """Calculate all indirect defect gaps."""
    # If there is only one gap, return an empty list
    if len(ordered_defect_gap_list.defect_gap_list) == 0:
        return []
    if len(ordered_defect_gap_list.defect_gap_list) == 1:
        if isinstance(ordered_defect_gap_list.defect_gap_list[0], Gap):
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
            if isinstance(this_defect_or_gap, Gap):
                indirect_gap_length = this_defect_or_gap.length_nm
                # insert into start of the list
                indirect_gaps.insert(0, indirect_gap_length)
            else:
                # We have reached a Defect, so we can stop
                break
        for _, this_defect_or_gap in enumerate(reversed(ordered_defect_gap_list.defect_gap_list)):
            if isinstance(this_defect_or_gap, Gap):
                indirect_gap_length = this_defect_or_gap.length_nm
                indirect_gaps.append(indirect_gap_length)
            else:
                # We have reached a Defect, so we can stop
                break
    return indirect_gaps


def find_curvature_defects(  # noqa: C901
    grain_collection: GrainCollection,
    curvature_defect_method: Literal["iqr", "absolute", "turn_in_distance"],
    curvature_threshold_iqr_multiplier: float,
    curvature_threshold_absolute_pernm: float,
    curvature_turn_in_distance_turn_threshold_deg: float,
    connect_close_defect_threshold_nm: float | None,
) -> set[int]:
    """
    Find curvature defects for all molecules in all grains in the grain collection.

    Parameters
    ----------
    grain_collection : GrainCollection
        The GrainCollection.
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

    Returns
    -------
    set[int]
        A set of global grain IDs for which curvature defect detection failed.
    """
    # find curvature defects
    bad_grains = set()
    if curvature_defect_method == "iqr":
        # iterate over each grain
        for global_grain_id, grain_model in grain_collection.items():
            for molecule_id, molecule_data in grain_model.molecule_data_collection.items():
                molecule_data_curvature_data = molecule_data.curvature_data
                trace_heights_nm = molecule_data.smoothed_spline_coords_heights
                assert trace_heights_nm is not None
                if molecule_data_curvature_data is None:
                    print(
                        f"no curvature data for grain {global_grain_id} molecule {molecule_id}, skipping curvature defect detection for this molecule"
                    )
                    bad_grains.add(global_grain_id)
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
                    trace_points_nm=molecule_data.spline_coords_nm,
                    trace_heights_nm=trace_heights_nm,
                    distance_to_previous_points_nm=distances_nm(
                        molecule_data.spline_coords_nm, circular=molecule_data.circular
                    ),
                    circular=molecule_data.circular,
                    connect_close_defect_threshold_nm=connect_close_defect_threshold_nm,
                )

                molecule_data.curvature_defect_data = MoleculeDefectData(
                    ordered_defects_and_gaps=ordered_defect_gap_list
                )
    elif curvature_defect_method == "absolute":
        for global_grain_id, grain_model in grain_collection.items():
            for molecule_id, molecule_data in grain_model.molecule_data_collection.items():
                molecule_data_curvature_data = molecule_data.curvature_data
                heights_nm = molecule_data.smoothed_spline_coords_heights
                assert heights_nm is not None
                if molecule_data_curvature_data is None:
                    print(
                        f"no curvature data for grain {global_grain_id} molecule {molecule_id}, skipping curvature defect detection for this molecule"
                    )
                    bad_grains.add(global_grain_id)
                    continue
                curvatures = molecule_data_curvature_data.curvatures
                curvatures_abs = np.abs(curvatures)
                assert isinstance(
                    curvatures, np.ndarray
                ), f"expected curvatures to be a numpy array, but got {type(curvatures)}"
                curvature_threshold_absolute = curvature_threshold_absolute_pernm / grain_model.pixel_to_nm_scaling
                curvature_defects_bool = curvatures_abs > curvature_threshold_absolute

                ordered_defect_gap_list = get_defects_and_gaps_from_bool_array(
                    defects_bool=curvature_defects_bool,
                    trace_points_nm=molecule_data.spline_coords_nm,
                    trace_heights_nm=heights_nm,
                    distance_to_previous_points_nm=distances_nm(
                        molecule_data.spline_coords_nm, circular=molecule_data.circular
                    ),
                    circular=molecule_data.circular,
                    connect_close_defect_threshold_nm=connect_close_defect_threshold_nm,
                )

                molecule_data.curvature_defect_data = MoleculeDefectData(
                    ordered_defects_and_gaps=ordered_defect_gap_list
                )
    elif curvature_defect_method == "turn_in_distance":
        for global_grain_id, grain_model in grain_collection.items():
            for molecule_id, molecule_data in grain_model.molecule_data_collection.items():
                molecule_data_curvature_data = molecule_data.curvature_data
                heights_nm = molecule_data.smoothed_spline_coords_heights
                assert heights_nm is not None
                if molecule_data_curvature_data is None:
                    print(
                        f"no curvature data for grain {global_grain_id} molecule {molecule_id}, skipping curvature defect detection for this molecule"
                    )
                    bad_grains.add(global_grain_id)
                    continue
                turn_in_distances = molecule_data_curvature_data.turn_in_distances_deg
                if turn_in_distances is None:
                    print(
                        f"no turn in distance data for grain {global_grain_id} molecule {molecule_id}, skipping curvature defect detection for this molecule"
                    )
                    bad_grains.add(global_grain_id)
                    continue
                curvature_defects_bool = np.abs(turn_in_distances) > curvature_turn_in_distance_turn_threshold_deg
                ordered_defect_gap_list = get_defects_and_gaps_from_bool_array(
                    defects_bool=curvature_defects_bool,
                    trace_points_nm=molecule_data.spline_coords_nm,
                    trace_heights_nm=heights_nm,
                    distance_to_previous_points_nm=distances_nm(
                        molecule_data.spline_coords_nm, circular=molecule_data.circular
                    ),
                    circular=molecule_data.circular,
                    connect_close_defect_threshold_nm=connect_close_defect_threshold_nm,
                )

                molecule_data.curvature_defect_data = MoleculeDefectData(
                    ordered_defects_and_gaps=ordered_defect_gap_list
                )
    else:
        raise ValueError(f"Invalid curvature defect method: {curvature_defect_method}")

    return bad_grains


def find_height_defects(  # noqa: C901
    grain_collection: GrainCollection,
    height_defect_method: Literal["iqr", "absolute", "percentage_of_median"],
    height_threshold_iqr_multiplier: float,
    height_threshold_absolute_nm: float,
    height_threshold_percentage_of_median: float,
    connect_close_defect_threshold_nm: float | None,
    find_height_defects_close_to_curvature_defects: bool,
    curvature_defect_height_defect_search_range_nm: float,
) -> set[int]:
    """
    Find height defects for all molecules in all grains in the grain collection.

    Parameters
    ----------
    grain_collection : GrainCollection
        The GrainCollection.
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

    Returns
    -------
    set[int]
        A set of global grain IDs for which height defect detection failed.
    """
    bad_grains = set()
    if height_defect_method == "iqr":
        for global_grain_id, grain_model in grain_collection.items():
            for molecule_id, molecule_data in grain_model.molecule_data_collection.items():
                heights_nm = molecule_data.smoothed_spline_coords_heights
                assert heights_nm is not None
                spline_coords_nm = molecule_data.spline_coords_nm
                assert len(heights_nm) == len(spline_coords_nm), (
                    f"length of heights does not match length of spline coords "
                    f"for grain {global_grain_id} molecule {molecule_id}, got {len(heights_nm)} heights and "
                    f"{len(spline_coords_nm)} spline coords"
                )
                iqr = np.percentile(heights_nm, 75) - np.percentile(heights_nm, 25)
                height_threshold_iqr = np.percentile(heights_nm, 50) - height_threshold_iqr_multiplier * iqr
                height_defects_bool = heights_nm < height_threshold_iqr

                if find_height_defects_close_to_curvature_defects:
                    height_defects_near_curvature_defects = find_height_defects_near_curvature_defects(
                        molecule_data=molecule_data,
                        grain_image=grain_model.image,
                        grain_mask=grain_model.mask,
                        height_threshold=float(height_threshold_iqr),
                        curvature_defect_height_defect_search_range_nm=curvature_defect_height_defect_search_range_nm,
                        pixel_to_nm_scaling=grain_model.pixel_to_nm_scaling,
                    )
                    height_defects_bool = np.logical_or(height_defects_bool, height_defects_near_curvature_defects)

                ordered_defect_gap_list = get_defects_and_gaps_from_bool_array(
                    defects_bool=height_defects_bool,
                    trace_points_nm=spline_coords_nm,
                    trace_heights_nm=heights_nm,
                    distance_to_previous_points_nm=distances_nm(
                        coords_nm=spline_coords_nm, circular=molecule_data.circular
                    ),
                    circular=molecule_data.circular,
                    connect_close_defect_threshold_nm=connect_close_defect_threshold_nm,
                )

                molecule_data.height_defect_data = MoleculeDefectData(ordered_defects_and_gaps=ordered_defect_gap_list)
    elif height_defect_method == "absolute":
        for _global_grain_id, grain_model in grain_collection.items():
            for _molecule_id, molecule_data in grain_model.molecule_data_collection.items():
                heights_nm = molecule_data.smoothed_spline_coords_heights
                assert heights_nm is not None
                spline_coords_nm = molecule_data.spline_coords_nm
                height_defects_bool = heights_nm < height_threshold_absolute_nm

                if find_height_defects_close_to_curvature_defects:
                    height_defects_near_curvature_defects = find_height_defects_near_curvature_defects(
                        molecule_data=molecule_data,
                        grain_image=grain_model.image,
                        grain_mask=grain_model.mask,
                        height_threshold=height_threshold_absolute_nm,
                        curvature_defect_height_defect_search_range_nm=curvature_defect_height_defect_search_range_nm,
                        pixel_to_nm_scaling=grain_model.pixel_to_nm_scaling,
                    )
                    height_defects_bool = np.logical_or(height_defects_bool, height_defects_near_curvature_defects)

                ordered_defect_gap_list = get_defects_and_gaps_from_bool_array(
                    defects_bool=height_defects_bool,
                    trace_points_nm=spline_coords_nm,
                    trace_heights_nm=heights_nm,
                    distance_to_previous_points_nm=distances_nm(
                        coords_nm=spline_coords_nm, circular=molecule_data.circular
                    ),
                    circular=molecule_data.circular,
                    connect_close_defect_threshold_nm=connect_close_defect_threshold_nm,
                )

                molecule_data.height_defect_data = MoleculeDefectData(ordered_defects_and_gaps=ordered_defect_gap_list)
    elif height_defect_method == "percentage_of_median":
        for _global_grain_id, grain_model in grain_collection.items():
            all_grain_heights = []
            for _molecule_id, molecule_data in grain_model.molecule_data_collection.items():
                heights_nm = molecule_data.smoothed_spline_coords_heights
                assert heights_nm is not None
                all_grain_heights.extend(heights_nm)
            all_grain_heights = np.array(all_grain_heights)
            median_height = np.median(all_grain_heights)
            height_threshold_percentage_of_median_value = median_height * height_threshold_percentage_of_median
            for _molecule_id, molecule_data in grain_model.molecule_data_collection.items():
                heights_nm = molecule_data.smoothed_spline_coords_heights
                assert heights_nm is not None
                spline_coords_nm = molecule_data.spline_coords_nm
                height_defects_bool = heights_nm < height_threshold_percentage_of_median_value

                if find_height_defects_close_to_curvature_defects:
                    height_defects_near_curvature_defects = find_height_defects_near_curvature_defects(
                        molecule_data=molecule_data,
                        grain_image=grain_model.image,
                        grain_mask=grain_model.mask,
                        height_threshold=height_threshold_percentage_of_median_value,
                        curvature_defect_height_defect_search_range_nm=curvature_defect_height_defect_search_range_nm,
                        pixel_to_nm_scaling=grain_model.pixel_to_nm_scaling,
                    )
                    if np.any(height_defects_near_curvature_defects):
                        print(
                            f"found height defect near curvature defect for file {grain_model.filename} grain {grain_model.global_grain_id}"
                        )
                    height_defects_bool = np.logical_or(height_defects_bool, height_defects_near_curvature_defects)

                ordered_defect_gap_list = get_defects_and_gaps_from_bool_array(
                    defects_bool=height_defects_bool,
                    trace_points_nm=spline_coords_nm,
                    trace_heights_nm=heights_nm,
                    distance_to_previous_points_nm=distances_nm(
                        coords_nm=spline_coords_nm, circular=molecule_data.circular
                    ),
                    circular=molecule_data.circular,
                    connect_close_defect_threshold_nm=connect_close_defect_threshold_nm,
                )

                molecule_data.height_defect_data = MoleculeDefectData(ordered_defects_and_gaps=ordered_defect_gap_list)
    else:
        raise ValueError(f"Invalid height defect method: {height_defect_method}")

    return bad_grains


def find_height_defects_near_curvature_defects(
    molecule_data: MoleculeData,
    grain_image: npt.NDArray[np.float64],
    grain_mask: npt.NDArray[np.bool_],
    height_threshold: float,
    curvature_defect_height_defect_search_range_nm: float,
    pixel_to_nm_scaling: float,
) -> npt.NDArray[np.bool_]:
    """Find height defects near any curvature defects."""
    spline_coords_px = molecule_data.spline_coords_px
    height_defects_near_curvature_defects = np.zeros(len(spline_coords_px)).astype(np.bool_)
    heights_nm = molecule_data.smoothed_spline_coords_heights
    assert heights_nm is not None
    curvature_defect_data = molecule_data.curvature_defect_data
    assert curvature_defect_data is not None
    curvature_defects_and_gaps = curvature_defect_data.ordered_defects_and_gaps
    for defect_or_gap in curvature_defects_and_gaps.defect_gap_list:
        if isinstance(defect_or_gap, Defect):
            defect = defect_or_gap
            defect_position_index = int((defect.start_index + defect.end_index) / 2)
            spline_coords_px = molecule_data.spline_coords_px
            defect_position_px = spline_coords_px[defect_position_index]

            # search for local height defect in the mask
            # get the mask of points within that range
            distance_mask = np.ones(grain_mask.shape)
            distance_mask[int(defect_position_px[0]), int(defect_position_px[1])] = 0
            distance_mask = (
                distance_transform_edt(distance_mask)
                < curvature_defect_height_defect_search_range_nm / pixel_to_nm_scaling
            )
            masked_image = np.ma.masked_array(data=grain_image, mask=~distance_mask)
            if np.any(masked_image < height_threshold):
                # add the index as a defect
                height_defects_near_curvature_defects[defect_position_index] = True

    return height_defects_near_curvature_defects


def find_defects_in_height_and_curvature(
    grain_collection: GrainCollection,
    height_defect_method: Literal["iqr", "absolute"],
    height_threshold_iqr_multiplier: float,
    height_threshold_absolute_nm: float,
    height_threshold_percentage_of_median: float,
    curvature_defect_method: Literal["iqr", "absolute", "turn_in_distance"],
    curvature_threshold_iqr_multiplier: float,
    curvature_threshold_absolute_pernm: float,
    curvature_turn_in_distance_turn_threshold_deg: float,
    connect_close_defect_threshold_nm: float | None,
    find_height_defects_close_to_curvature_defects: bool,
    curvature_defect_height_defect_search_range_nm: float,
) -> set[int]:
    """
    Find defects in height and curvature for all molecules in all grains in the grain collection.

    Parameters
    ----------
    grain_collection : GrainCollection
        The GrainCollection.
    height_defect_method : Literal["iqr", "absolute"]
        The method to use for finding height defects. Options are "iqr" for interquartile range method or "absolute"
        for an absolute threshold in nanometres.
    height_threshold_iqr_multiplier : float
        The multiplier for the interquartile range when using the "iqr" method for height defect detection.
    height_threshold_absolute_nm : float
        The absolute threshold in nanometres for height defect detection when using the "absolute" method
    height_threshold_percentage_of_median : float
        The percentage of the median height of each *grain* to use as a threshold for height defect detection
        when using the "percentage_of_median" method.
    curvature_defect_method : Literal["iqr", "absolute"]
        The method to use for finding curvature defects. Options are "iqr" for interquartile range method or "absolute"
        for an absolute threshold in inverse nanometres.
    curvature_threshold_iqr_multiplier : float
        The multiplier for the interquartile range when using the "iqr" method for curvature defect detection.
    curvature_threshold_absolute_pernm : float
        The absolute threshold in inverse nanometres for curvature defect detection when using the "absolute" method.
    turn_in_distance_turn_threshold_deg : float
        The threshold in degrees for turn in distance defect detection when using the "turn_in_distance" method for
        curvature defect detection. If the turn turns more than this value within the window length (defined
        in the call to the function that calculates the turn in distance), then it is considered a defect.
    connect_close_defect_threshold_nm : float | None
        The distance in nanometres between defects below which two defects will be connected into one defect
    find_height_defects_close_to_curvature_defects : bool
        Whether to search for height defects in the proximity of curvature defects, away from the trace.
    curvature_defect_height_defect_search_range_nm : bool
        Distance in nanometres within which to search for height defects from a curvature defect (away from the trace).

    Returns
    -------
    set[int]
        A set of global grain IDs for which height or curvature defect detection failed.
    """
    bad_grains = set()

    # find curvature defects
    additional_bad_grains = find_curvature_defects(
        grain_collection=grain_collection,
        curvature_defect_method=curvature_defect_method,
        curvature_threshold_iqr_multiplier=curvature_threshold_iqr_multiplier,
        curvature_threshold_absolute_pernm=curvature_threshold_absolute_pernm,
        curvature_turn_in_distance_turn_threshold_deg=curvature_turn_in_distance_turn_threshold_deg,
        connect_close_defect_threshold_nm=connect_close_defect_threshold_nm,
    )
    bad_grains.update(additional_bad_grains)

    additional_bad_grains = find_height_defects(
        grain_collection=grain_collection,
        height_defect_method=height_defect_method,
        height_threshold_iqr_multiplier=height_threshold_iqr_multiplier,
        height_threshold_absolute_nm=height_threshold_absolute_nm,
        height_threshold_percentage_of_median=height_threshold_percentage_of_median,
        connect_close_defect_threshold_nm=connect_close_defect_threshold_nm,
        find_height_defects_close_to_curvature_defects=find_height_defects_close_to_curvature_defects,
        curvature_defect_height_defect_search_range_nm=curvature_defect_height_defect_search_range_nm,
    )
    bad_grains.update(additional_bad_grains)

    grain_collection.remove_grains(bad_grains)
    print(f"removed grains with ids {bad_grains}, {len(grain_collection)} grains remain")

    return bad_grains


def calculate_turn_in_distance_for_trace_deg(
    trace_coords_nm: npt.NDArray[np.float64],
    turn_in_distance_window_length_nm: float,
    circular: bool,
    turn_in_distance_window_end_sampling_points: int,
) -> npt.NDArray[np.float64]:
    """Calculate the degrees turned over a given window length for each point in a trace."""
    assert turn_in_distance_window_end_sampling_points >= 2, "need at least 2 points to calculate a vector"
    turn_in_distances_deg = np.zeros(len(trace_coords_nm))
    distances_to_previous_points_nm = distances_nm(trace_coords_nm, circular)

    # for each point, construct a symmetric window around it with the given window length
    for index in range(len(trace_coords_nm)):
        try:
            left_index, right_index = construct_symmetric_window_of_length(
                distances_to_previous_points=distances_to_previous_points_nm,
                midpoint_index=index,
                window_length=turn_in_distance_window_length_nm,
                circular=circular,
            )
        # check for the error where the edge of the window meets the end of the trace in linear traces
        # if this happens, set the value to nan and continue
        except ValueError as e:
            if "required right window bound exceeds the end of the linear trace" in str(
                e
            ) or "required left window bound exceeds the start of the linear trace" in str(e):
                turn_in_distances_deg[index] = np.nan
                continue
            raise e

        # To calculate vectors, we will need to grab another point on either side. Grab the inner points
        # so, for the left, get the next point, and for the right, get the previous point.
        # will need to check for windows with points that have too few points to calculate the vectors
        window_num_points = (
            right_index - left_index + 1
            if right_index >= left_index
            else left_index + (len(trace_coords_nm) - right_index) + 1
        )
        if window_num_points < turn_in_distance_window_end_sampling_points * 2:
            raise ValueError(
                f"window of {window_num_points} points contains too few points to sample the required number of points"
                f" {turn_in_distance_window_end_sampling_points} from each side to calculate the turn in distance"
            )

        # get the indices of the points to sample, wrapping where needed
        window_left_points_indices = []
        window_right_points_indices = []
        for sample_index in range(turn_in_distance_window_end_sampling_points):
            left_sample_index = left_index + sample_index
            if left_sample_index >= len(trace_coords_nm):
                if circular:
                    left_sample_index = left_sample_index % len(trace_coords_nm)
                else:
                    raise ValueError(
                        "required left sample point index exceeds the end of the linear trace, cannot calculate turn in distance"
                    )
            window_left_points_indices.append(left_sample_index)
            right_sample_index = right_index - sample_index
            if right_sample_index < 0:
                if circular:
                    right_sample_index = right_sample_index % len(trace_coords_nm)
                else:
                    raise ValueError(
                        "required right sample point index exceeds the start of the linear trace, cannot calculate turn in distance"
                    )
            window_right_points_indices.append(right_sample_index)

        # calculate the vectors by taking the differences between each sequential point pair and then averaging them
        left_vectors = np.diff(trace_coords_nm[window_left_points_indices], axis=0)
        # flip the right vectors since we are going in the opposite direction and we want the vectors to point
        # in the same direction for angle calculation
        window_right_points_indices_flipped = list(reversed(window_right_points_indices))
        right_vectors = np.diff(trace_coords_nm[window_right_points_indices_flipped], axis=0)
        left_vector = np.mean(left_vectors, axis=0)
        right_vector = np.mean(right_vectors, axis=0)
        # calculate angle difference between the two vectors
        turn_rad = angle_diff_signed(v1=left_vector, v2=right_vector)
        turn_deg = np.degrees(turn_rad)
        turn_in_distances_deg[index] = turn_deg

    return turn_in_distances_deg


def construct_symmetric_window_of_length(
    distances_to_previous_points: npt.NDArray[np.float64],
    midpoint_index: int,
    window_length: float,
    circular: bool,
) -> tuple[int, int]:
    """Construct a symmetric window around a midpoint index where the len of the window is defined by a given length."""
    left_index = 0
    right_index = 0
    cumulative_distance_left = 0.0
    cumulative_distance_right = 0.0
    half_window_length = window_length / 2
    # can get the reversed direction distances to previous points by just rolling the array by 1 in the negative
    # direction
    reversed_distances_to_previous_points = np.roll(distances_to_previous_points, -1)

    # guard for the length of the window being longer than the total length of the trace
    if window_length >= np.sum(distances_to_previous_points):
        raise ValueError("window length exceeds total length of the trace, cannot construct window")

    # Find the right index by moving forwards until the cumulative distance exceeds the window length, wrapping around
    # if necessary and circular
    for index in range(1, len(distances_to_previous_points) + 1):
        prospective_right_index = midpoint_index + index
        if prospective_right_index >= len(distances_to_previous_points):
            if circular:
                prospective_right_index = prospective_right_index % len(distances_to_previous_points)
            else:
                raise ValueError("required right window bound exceeds the end of the linear trace")
        cumulative_distance_right += distances_to_previous_points[prospective_right_index]
        if cumulative_distance_right > half_window_length:
            right_index = prospective_right_index
            break
    # Find the left index by moving backwards until the cumulative distance exceeds the window length, wrapping around
    # if necessary and circular
    for index in range(1, len(distances_to_previous_points) + 1):
        prospective_left_index = midpoint_index - index
        if prospective_left_index < 0:
            if circular:
                prospective_left_index = prospective_left_index % len(distances_to_previous_points)
            else:
                raise ValueError("required left window bound exceeds the start of the linear trace")
        cumulative_distance_left += reversed_distances_to_previous_points[prospective_left_index]
        if cumulative_distance_left > half_window_length:
            left_index = prospective_left_index
            break
    return left_index, right_index


def calculate_turn_in_distance(
    grain_collection: GrainCollection,
    turn_in_distance_window_length_nm: float,
    turn_in_distance_window_end_sampling_points: int,
) -> set[int]:
    """Calculate the turn in distance for each molecule in the grain collection."""
    bad_grains = set()
    for _global_grain_id, grain_model in grain_collection.items():
        for _molecule_id, molecule_data in grain_model.molecule_data_collection.items():
            try:
                turn_in_distances = calculate_turn_in_distance_for_trace_deg(
                    trace_coords_nm=molecule_data.spline_coords_nm,
                    turn_in_distance_window_length_nm=turn_in_distance_window_length_nm,
                    circular=molecule_data.circular,
                    turn_in_distance_window_end_sampling_points=turn_in_distance_window_end_sampling_points,
                )
            except ValueError as e:
                if "window length exceeds total length of the trace" in str(e):
                    turn_in_distances = np.full(len(molecule_data.spline_coords_nm), np.nan)
                else:
                    raise e
            molecule_data.curvature_data.turn_in_distances_deg = turn_in_distances
            molecule_data.curvature_data.turn_in_distance_window_length_nm = turn_in_distance_window_length_nm
            molecule_data.curvature_data.turn_in_distance_window_end_sampling_points = (
                turn_in_distance_window_end_sampling_points
            )
    return bad_grains


def smooth_height_traces(
    grain_collection: GrainCollection,
    window_length_nm: float,
) -> set[int]:
    """Smooth the height traces for each molecule in the grain collection using a rolling average window."""
    bad_grains = set()
    for _global_grain_id, grain_model in grain_collection.items():
        for _molecule_id, molecule_data in grain_model.molecule_data_collection.items():
            try:
                heights_nm = molecule_data.spline_coords_heights
                point_spacing_nm = molecule_data.point_spacing_nm
                window_length_points = int(window_length_nm / point_spacing_nm)
                if window_length_points < 1:
                    raise ValueError(
                        f"window length of {window_length_nm} nm is too small for the point spacing of"
                        f"{point_spacing_nm} nm, resulting in a window length of {window_length_points} points, which is"
                        f" less than 1"
                    )
                smoothed_heights_nm = rolling_average(
                    data=heights_nm,
                    window_size_points=window_length_points,
                    circular=molecule_data.circular,
                )
            except ValueError as e:
                if "window length exceeds total length of the trace" in str(e):
                    smoothed_heights_nm = heights_nm
                else:
                    raise e
            molecule_data.smoothed_spline_coords_heights = smoothed_heights_nm
    return bad_grains


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
    defect_volume = np.trapezoid(defect_heights, x=defect_distances_nm)
    assert defect_volume >= 0, f"defect volume is negative: {defect_volume}"
    return float(defect_volume)
