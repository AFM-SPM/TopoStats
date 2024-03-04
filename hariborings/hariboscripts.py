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
