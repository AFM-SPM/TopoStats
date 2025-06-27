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
    length_nm: float | None = None


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
    length_nm: float | None = None


def get_defects_linear(
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
            if not in_defect:
                # Start new defect
                current_defect_gap_start_index = index
                in_defect = True
            if in_gap:
                # End of the gap
                gaps.append((current_defect_gap_start_index, index - 1))
                in_gap = False
        else:
            if not in_gap:
                # Start new gap
                current_defect_gap_start_index = index
                in_gap = True
            if in_defect:
                # End of the defect
                defects.append((current_defect_gap_start_index, index - 1))
                in_defect = False
    # If we are still in a defect or gap at the end of the loop, we need to close it
    if in_defect:
        defects.append((current_defect_gap_start_index, len(defects_bool) - 1))
    elif in_gap:
        gaps.append((current_defect_gap_start_index, len(defects_bool) - 1))

    return (
        defects,
        gaps,
    )


def calculate_indirect_defect_gap_lengths(
    defect_start_end_indexes: tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]],
    cumulative_distance_nm: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Calculate the lengths of the gaps between defects for all defects."""
    defect_start_indexes, defect_end_indexes = defect_start_end_indexes

    if len(defect_start_indexes) == 0:
        return np.array([], dtype=np.float64)

    # For each defect end, calculate the distance to the every defect start
    gap_lengths = []
    for defect_end_index in defect_end_indexes:
        for defect_start_index in defect_start_indexes:
            # Calculate the distance between the end index and the start index using the cumulative distance
            if defect_start_index > defect_end_index:
                # Defect start is later than the end, therefore we can simply subtract the cumulative distances
                gap_length = cumulative_distance_nm[defect_start_index] - cumulative_distance_nm[defect_end_index]
                gap_lengths.append(gap_length)
            else:
                # The defect start is looped around to the beginning of the array, so we need to add the distance from
                # the end of the array to the start of the array
                gap_length = (
                    cumulative_distance_nm[-1]
                    - cumulative_distance_nm[defect_end_index]
                    + cumulative_distance_nm[defect_start_index]
                )
                gap_lengths.append(gap_length)
    return np.array(gap_lengths, dtype=np.float64)
