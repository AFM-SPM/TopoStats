"""Functions for manipulating numpy arrays."""

import logging

import numpy as np
import numpy.typing as npt

from topostats.logs.logs import LOGGER_NAME

LOGGER = logging.getLogger(LOGGER_NAME)


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
    proposed_size = (
        bbox[2] - bbox[0] + 2 * padding,
        bbox[3] - bbox[1] + 2 * padding,
    )
    limits_size = (
        limits[2] - limits[0],
        limits[3] - limits[1],
    )
    if proposed_size[0] > limits_size[0] or proposed_size[1] > limits_size[1]:
        raise ValueError(
            f"Proposed size {proposed_size} px = {bbox} px + ({padding},{padding}) px is larger than limits size"
            f"{limits_size} px. Cannot pad bounding box beyond limits."
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
