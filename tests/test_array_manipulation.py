"""Test array manipulation functions."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from topostats.array_manipulation import (
    pad_bounding_box_dynamically_at_limits,
    re_crop_grain_image_and_mask_to_set_size_nm,
)


@pytest.mark.parametrize(
    ("bbox", "limits", "padding", "expected_bbox"),
    [
        pytest.param(
            (3, 4, 5, 6),
            (0, 0, 10, 10),
            0,
            (3, 4, 5, 6),
            id="no padding, within limits",
        ),
        pytest.param(
            (3, 4, 5, 6),
            (0, 0, 10, 10),
            2,
            (1, 2, 7, 8),
            id="padding within limits",
        ),
        pytest.param(
            (3, 4, 5, 6),
            (0, 0, 12, 12),
            4,
            (0, 0, 10, 10),
            id="top (min row) padding exceeds limits, expand down to compensate",
        ),
        pytest.param(
            (4, 3, 6, 5),
            (0, 0, 12, 12),
            4,
            (0, 0, 10, 10),
            id="left (min col) padding exceeds limits, expand right to compensate",
        ),
        pytest.param(
            (5, 4, 8, 7),
            (0, 0, 10, 10),
            3,
            (1, 1, 10, 10),
            id="bottom (max row) padding exceeds limits, expand up to compensate",
        ),
        pytest.param(
            (4, 5, 7, 8),
            (0, 0, 10, 10),
            3,
            (1, 1, 10, 10),
            id="right (max col) padding exceeds limits, expand left to compensate",
        ),
    ],
)
def test_pad_bounding_box_dynamically_at_limits(
    bbox: tuple[int, int, int, int],
    limits: tuple[int, int, int, int],
    padding: int,
    expected_bbox: tuple[int, int, int, int],
) -> None:
    """Test the pad_bounding_box_cutting_off_at_image_bounds method with dynamic limits."""
    result = pad_bounding_box_dynamically_at_limits(bbox=bbox, limits=limits, padding=padding)
    assert result == expected_bbox
