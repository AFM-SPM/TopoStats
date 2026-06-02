"""Functions for manipulating numpy arrays."""

import logging

import numpy as np
import numpy.typing as npt

from topostats.logs.logs import LOGGER_NAME

LOGGER = logging.getLogger(LOGGER_NAME)


def re_crop_grain_image_and_mask_to_set_size_nm(
    filename: str,
    grain_number: int,
    grain_bbox: tuple[int, int, int, int],
    pixel_to_nm_scaling: float,
    full_image: npt.NDArray[np.float32],
    full_mask_tensor: npt.NDArray[np.bool_],
    target_size_nm: float,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """
    Re-crop a grain image and mask to be a target size in nanometres.

    Parameters
    ----------
    filename : str
        The name of the file being processed, used for logging.
    grain_number : int
        The number of the grain being processed, used for logging.
    grain_bbox : tuple[int, int, int, int]
        The bounding box of the grain in the form (min_row, min_col, max_row (exclusive), max_col (exclusive)).
    pixel_to_nm_scaling : float
        Pixel to nanometre scaling factor.
    full_image : npt.NDArray[np.float32]
        The full image from which to crop the grain image.
    full_mask_tensor : npt.NDArray[np.bool_]
        The full mask tensor from which to crop the mask.
    target_size_nm : float
        The target size in nanometres to crop the grain image and mask to.

    Returns
    -------
    tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]
        The cropped grain image and mask, both as numpy arrays.

    Raises
    ------
    ValueError
        If the target size in nanometres is larger than the full image or mask dimensions.
    """
    # Re-slice the image to get a larger or smaller crop depending on the grain size.
    target_size_px = int(target_size_nm / pixel_to_nm_scaling)

    # To create a bbox that is the right size, we can create a small bbox and then pad it.
    # - find the centroid of the grain bbox
    # - determine if the target bbox is going to be odd or even centred.
    #   - If odd centred, we can pad the centre pixel(1x1) bbox by target_size_nm // 2 in each direction
    #   - If even centred, we can pad the centre pixels(2x2) bbox by target_size_nm // 2 - 1 in each direction.

    if target_size_px % 2 == 0:
        # Even proposed size, so take the centre 2x2 pixels and pad by half the size minus 1
        # Get centre 2x2 pixel bbox of grain crop
        grain_crop_bbox_centre = (
            (grain_bbox[0] + grain_bbox[2]) // 2 - 1,
            (grain_bbox[1] + grain_bbox[3]) // 2 - 1,
            (grain_bbox[0] + grain_bbox[2]) // 2 + 1,
            (grain_bbox[1] + grain_bbox[3]) // 2 + 1,
        )
        to_pad = target_size_px // 2 - 1
    else:
        # Odd proposed size, so take the centre 1x1 pixel and pad by half the size
        # Get centre 1x1 pixel bbox of grain crop
        grain_crop_bbox_centre = (
            (grain_bbox[0] + grain_bbox[2]) // 2,
            (grain_bbox[1] + grain_bbox[3]) // 2,
            (grain_bbox[0] + grain_bbox[2]) // 2 + 1,
            (grain_bbox[1] + grain_bbox[3]) // 2 + 1,
        )
        to_pad = target_size_px // 2

    # Pad the bbox to the desired size
    try:
        grain_crop_bbox_resized = pad_bounding_box_dynamically_at_limits(
            bbox=grain_crop_bbox_centre,
            limits=(0, 0, full_image.shape[0], full_image.shape[1]),
            padding=to_pad,
        )
    except ValueError as e:
        if "Proposed size" in str(e):
            raise ValueError(
                f"[{filename}] : Grain {grain_number} crop cannot be re-cropped at size {target_size_nm} nm "
                f"({target_size_px} px) "
            ) from e
        # If the error is not about the proposed size, re-raise it
        raise e

    # Crop the image and mask to the new bbox
    crop_image = full_image[
        grain_crop_bbox_resized[0] : grain_crop_bbox_resized[2],
        grain_crop_bbox_resized[1] : grain_crop_bbox_resized[3],
    ]
    crop_mask = full_mask_tensor[
        grain_crop_bbox_resized[0] : grain_crop_bbox_resized[2],
        grain_crop_bbox_resized[1] : grain_crop_bbox_resized[3],
        :,
    ]

    return crop_image, crop_mask


def pad_bounding_box_dynamically_at_limits(
    bbox: tuple[int, int, int, int],
    limits: tuple[int, int, int, int],
    padding: int,
) -> tuple[int, int, int, int]:
    """
    Pad a bounding box within limits. If the padding would exceed the limits bounds, pad in the other direction.

    Parameters
    ----------
    bbox : tuple[int, int, int, int]
        The bounding box to pad.
    limits : tuple[int, int, int, int]
        The region to limit the bounding box to in the form (min_row, min_col, max_row, max_col).
    padding : int
        The padding to apply to the bounding box.

    Returns
    -------
    tuple[int, int, int, int]
        The new bounding box indices.
    """
    # check that the padded size is smaller than the limits
    bbox_height = bbox[2] - bbox[0]
    bbox_width = bbox[3] - bbox[1]
    proposed_height = bbox_height + 2 * padding
    proposed_width = bbox_width + 2 * padding
    limits_height = limits[2] - limits[0]
    limits_width = limits[3] - limits[1]
    if proposed_height > limits_height or proposed_width > limits_width:
        raise ValueError(
            f"Proposed size {proposed_height}x{proposed_width} px = ({bbox_width}x{bbox_height}) + "
            f"({2*padding}x{2*padding}) px is larger than limits size "
            f"({limits_height}x{limits_width}) px. Cannot pad bounding box beyond limits."
        )
    pad_up_amount = padding
    pad_down_amount = padding
    pad_left_amount = padding
    pad_right_amount = padding
    # try padding up, check if hit the top of the limits
    if bbox[0] - padding < limits[0]:
        # if so, restrict up padding to the limits and add the remaining padding to the down padding
        pad_up_amount = bbox[0] - limits[0]
        # Can safely assume can increase down padding since we checked earlier that the proposed size is smaller than
        # limits
        pad_down_amount += padding - pad_up_amount
    # try padding down, check if hit the bottom of the limits
    elif bbox[2] + padding > limits[2]:
        # if so, restrict down padding to the limits and add the remaining padding to the up padding
        pad_down_amount = limits[2] - bbox[2]
        # Can safely assume can increase up padding since we checked earlier that the proposed size is smaller than
        # limits
        pad_up_amount += padding - pad_down_amount
    # try padding left, check if hit the left of the limits
    if bbox[1] - padding < limits[1]:
        # if so, restrict left padding to the limits and add the remaining padding to the right padding
        pad_left_amount = bbox[1] - limits[1]
        # Can safely assume can increase right padding since we checked earlier that the proposed size is smaller than
        # limits
        pad_right_amount += padding - pad_left_amount
    # try padding right, check if hit the right of the limits
    elif bbox[3] + padding > limits[3]:
        # if so, restrict right padding to the limits and add the remaining padding to the left padding
        pad_right_amount = limits[3] - bbox[3]
        # Can safely assume can increase left padding since we checked earlier that the proposed size is smaller than
        # limits
        pad_left_amount += padding - pad_right_amount
    # Return the new bounding box indices
    return (
        bbox[0] - pad_up_amount,
        bbox[1] - pad_left_amount,
        bbox[2] + pad_down_amount,
        bbox[3] + pad_right_amount,
    )


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
        return float(distance_without_halves + start_half_distance + end_half_distance)

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


def rolling_average(data: npt.NDArray[np.float64], window_size_points: int, circular: bool) -> npt.NDArray[np.float64]:
    """Calculate the rolling average of a 1D array with a specified window size and circular / linear padding."""
    half_window_size = window_size_points // 2
    # pad the data
    # for circular, pad by wrapping
    to_pad_right = half_window_size
    if window_size_points % 2 == 0:
        to_pad_left = half_window_size - 1
    else:
        to_pad_left = half_window_size
    if circular:
        padded_data = np.pad(data, (to_pad_left, to_pad_right), mode="wrap")
    # for linear, pad with the edge values
    else:
        padded_data = np.pad(data, (to_pad_left, to_pad_right), mode="edge")
    # convolve with a uniform kernel
    kernel = np.ones(window_size_points) / window_size_points
    convolved = np.convolve(padded_data, kernel, mode="valid")
    assert len(convolved) == len(data), f"Expected convolved length to be {len(data)}, but got {len(convolved)}"
    return np.array(convolved).astype(np.float64)
