"""Scripts for array manipulation."""

import numpy as np
import numpy.typing as npt


def distances_nm(coords_nm: npt.NDArray[np.float64], circular: bool) -> npt.NDArray[np.float64]:
    """
    Calculate the distances between consecutive coordinates in nanometres.

    Parameters
    ----------
    coords_nm : npt.NDArray[np.float64]
        An array of shape (N, 2) containing the x and y coordinates in nanometres.
    circular : bool
        Whether the trace is circular (i.e., whether the last point connects back to the first point).

    Returns
    -------
    npt.NDArray[np.float64]
        An array of shape (N,) containing the distances between consecutive coordinates in nanometres.
    """
    deltas = np.diff(coords_nm, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    # prepend a 0 to the distances to align with the original coordinates
    distances = np.concatenate(([0], segment_lengths))
    if circular:
        # if the trace is circular, we also need to account for the distance from the last point back to the first point
        last_to_first_distance = np.linalg.norm(coords_nm[0] - coords_nm[-1])
        distances[-1] += last_to_first_distance
    return distances


def cumulative_distances_nm(coords_nm: npt.NDArray[np.float64], circular: bool) -> npt.NDArray[np.float64]:
    """
    Calculate the cumulative distances along the trace in nanometres.

    Parameters
    ----------
    coords_nm : npt.NDArray[np.float64]
        An array of shape (N, 2) containing the x and y coordinates in nanometres.
    circular : bool
        Whether the trace is circular (i.e., whether the last point connects back to the first point).

    Returns
    -------
    npt.NDArray[np.float64]
        An array of shape (N,) containing the cumulative distances along the trace in nanometres.
    """
    distances = distances_nm(coords_nm, circular=circular)
    return np.cumsum(distances)
