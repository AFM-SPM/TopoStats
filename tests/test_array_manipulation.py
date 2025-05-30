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
    """Test the pad_bounding_box_dynamically_at_limits method."""
    result = pad_bounding_box_dynamically_at_limits(bbox=bbox, limits=limits, padding=padding)
    assert result == expected_bbox


@pytest.mark.parametrize(
    (
        "grain_bbox",
        "pixel_to_nm_scaling",
        "full_image",
        "full_mask_tensor",
        "target_size_nm",
        "expected_grain_crop",
        "expected_grain_tensor",
    ),
    [
        pytest.param(
            (0, 2, 5, 6),
            1.0,
            np.array(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 2, 2, 1, 1, 1, 1],
                    [1, 1, 1, 1, 2, 2, 1, 1, 1, 1],
                    [1, 1, 1, 1, 2, 2, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ]
            ).astype(np.float32),
            np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                            [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                            [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ).astype(np.bool_),
            8.0,
            np.array(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 2, 2, 1, 1, 1],
                    [1, 1, 1, 2, 2, 1, 1, 1],
                    [1, 1, 1, 2, 2, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                ]
            ),
            np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 0, 0, 1, 1, 1],
                            [1, 1, 1, 0, 0, 1, 1, 1],
                            [1, 1, 1, 0, 0, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 0, 0, 0],
                            [0, 0, 0, 1, 1, 0, 0, 0],
                            [0, 0, 0, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ),
            id="grain crop 8nm, even centred, restricted at top",
        ),
        pytest.param(
            (0, 2, 5, 6),
            1.0,
            np.array(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 2, 2, 1, 1, 1, 1],
                    [1, 1, 1, 1, 2, 2, 1, 1, 1, 1],
                    [1, 1, 1, 1, 2, 2, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ]
            ).astype(np.float32),
            np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                            [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                            [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ).astype(np.bool_),
            7.0,
            np.array(
                [
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 2, 2, 1, 1, 1],
                    [1, 1, 2, 2, 1, 1, 1],
                    [1, 1, 2, 2, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                ]
            ),
            np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 0, 0, 1, 1, 1],
                            [1, 1, 0, 0, 1, 1, 1],
                            [1, 1, 0, 0, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0, 0, 0],
                            [0, 0, 1, 1, 0, 0, 0],
                            [0, 0, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ),
            id="grain crop 7nm, odd centred, restricted at top",
        ),
    ],
)
def test_re_crop_grain_image_and_mask_to_set_size_nm(
    grain_bbox: tuple[int, int, int, int],
    pixel_to_nm_scaling: float,
    full_image: npt.NDArray[np.float32],
    full_mask_tensor: npt.NDArray[np.bool_],
    target_size_nm: float,
    expected_grain_crop: npt.NDArray[np.float32],
    expected_grain_tensor: npt.NDArray[np.bool_],
) -> None:
    """Test the re_crop_grain_image_and_mask_to_set_size_nm method."""
    grain_crop, grain_tensor = re_crop_grain_image_and_mask_to_set_size_nm(
        filename="test_image",
        grain_number=1,
        grain_bbox=grain_bbox,
        pixel_to_nm_scaling=pixel_to_nm_scaling,
        full_image=full_image,
        full_mask_tensor=full_mask_tensor,
        target_size_nm=target_size_nm,
    )

    np.testing.assert_array_equal(grain_crop, expected_grain_crop)
    np.testing.assert_array_equal(grain_tensor, expected_grain_tensor)
