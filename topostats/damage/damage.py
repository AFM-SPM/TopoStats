"""Functions for damage detection and quantification."""

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
    """Get the start and end indexes of defects in a boolean array."""
    defect_start_indexes = []
    defect_end_indexes = []

    for index, point in enumerate(defects_bool):
        previous_point = defects_bool[index - 1] if index > 0 else False
        if point and not previous_point:
            defect_start_indexes.append(index)
        if not point and previous_point:
            defect_end_indexes.append(index)


def calculate_indirect_defect_gap_lengths(
    points_distance_to_previous_nm: npt.NDArray[np.float64],
    defects_bool: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float64]:
    """Calculate the lengths of indirect defects and gaps."""

    # Need not just distance to next defect, but the distance to any defect start.

    # Need defect start points
    # Need defect end points
    # Then calculate the distance from each defect end point to every defect start point

    # Find defect start points

    defect_start_points, defect_end_points = get_defect_start_end_indexes(defects_bool)
