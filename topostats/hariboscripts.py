"""Scripts for the hariborings project"""

import numpy as np


def flip_if_anticlockwise(trace: np.ndarray):
    """Flip the trace if it is anticlockwise. Ensures that the trace is always clockwise.

    If the trace is clockwise, do nothing.If the trace is a straight line, do nothing.

    Parameters
    ----------
    trace : np.ndarray
        The trace to be checked and flipped if necessary.

    Returns
    -------
    np.ndarray
        The trace, flipped if necessary.
    """
    # Check if the trace is clockwise or anticlockwise by summing the cross products of the vectors
    # If the sum is positive, the trace is clockwise
    # If the sum is negative, the trace is anticlockwise
    # If the sum is 0, the trace is a straight line
    cross_sum = 0
    for i in range(len(trace) - 1):
        cross_sum += np.cross(trace[i], trace[i + 1])
    if cross_sum > 0:
        # print("clockwise")
        # Reverse the trace
        trace = np.flip(trace, axis=0)
    elif cross_sum < 0:
        # print("anticlockwise")
        pass

    return trace


def calculate_real_distance_between_points_in_array(
    array: np.ndarray, indexes_to_calculate_distance_between: np.ndarray, p_to_nm: float
):
    """Calculate the real distance between points in an array.

    Parameters
    ----------
    array : np.ndarray
        The array of points.
    indexes_to_calculate_distance_between : np.ndarray
        The indexes of the points in the array to calculate the distance between.
    p_to_nm : float
        The pixel to nanometre conversion factor.

    Returns
    -------
    np.ndarray
        The distances between the points in the array.
    """
    # Calculate the distances between each defect along the trace
    to_find = np.copy(indexes_to_calculate_distance_between)
    original = to_find[0]
    current_index = original
    current_position = array[current_index]
    previous_position = current_position
    distances = []
    distance = 0
    while len(to_find) > 0:
        # Update old position
        previous_position = current_position
        # Increment the current position along the trace
        current_index += 1
        if current_index >= len(array):
            current_index -= len(array)
        current_position = array[current_index]

        # Increment the distance
        distance += np.linalg.norm(current_position - previous_position) * p_to_nm

        # Check if the current index is in the to_find list
        if current_index in to_find:
            # Get rid of the current index from the to_find list
            to_find = to_find[to_find != current_index]
            # Store the distance
            distances.append(distance)
            # Reset the distance
            distance = 0

    return distances
