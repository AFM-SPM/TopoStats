"""Additional tests of dnaTracing methods."""
from pathlib import Path

import numpy as np
import pytest

from topostats.tracing.dnatracing import dnaTrace, crop_array

# This is required because of the inheritance used throughout
# pylint: disable=redefined-outer-name
BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"
PIXEL_SIZE = 0.4940029296875

LINEAR_IMAGE = np.load(RESOURCES / "dnatracing_image_linear.npy")
LINEAR_MASK = np.load(RESOURCES / "dnatracing_mask_linear.npy")
CIRCULAR_IMAGE = np.load(RESOURCES / "dnatracing_image_circular.npy")
CIRCULAR_MASK = np.load(RESOURCES / "dnatracing_mask_circular.npy")
MIN_SKELETON_SIZE = 10


@pytest.fixture()
def dnatrace() -> dnaTrace:
    """DnaTrace object for use in tests."""
    return dnaTrace(
        image=np.asarray([[1]]),
        grain=None,
        filename="test.spm",
        pixel_to_nm_scaling=PIXEL_SIZE,
        min_skeleton_size=MIN_SKELETON_SIZE,
    )


GRAINS = {}
GRAINS["vertical"] = np.asarray(
    [
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ]
)
GRAINS["horizontal"] = np.asarray(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
)
GRAINS["diagonal1"] = np.asarray(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
)
GRAINS["diagonal2"] = np.asarray(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
)
GRAINS["diagonal3"] = np.asarray(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
)
GRAINS["circle"] = np.asarray(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
)
GRAINS["blob"] = np.asarray(
    [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ]
)
GRAINS["cross"] = np.asarray(
    [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]
)
GRAINS["single_L"] = np.asarray(
    [
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
    ]
)
GRAINS["double_L"] = np.asarray(
    [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ]
)
GRAINS["diagonal_end_single_L"] = np.asarray(
    [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ]
)
GRAINS["diagonal_end_straight"] = np.asarray(
    [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ]
)
GRAINS["figure8"] = np.asarray(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
)
GRAINS["three_ends"] = np.asarray(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
)
GRAINS["six_ends"] = np.asarray(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
)


@pytest.mark.parametrize(
    ("grain", "mol_is_circular"),
    [
        (GRAINS["vertical"], False),
        (GRAINS["horizontal"], False),
        (GRAINS["diagonal1"], True),  # This is wrong, this IS a linear molecule
        (GRAINS["diagonal2"], False),
        (GRAINS["diagonal3"], False),
        (GRAINS["circle"], True),
        (GRAINS["blob"], True),
        (GRAINS["cross"], False),
        (GRAINS["single_L"], False),
        (GRAINS["double_L"], True),  # This is wrong, this IS a linear molecule
        (GRAINS["diagonal_end_single_L"], False),
        (GRAINS["diagonal_end_straight"], False),
        (GRAINS["figure8"], True),
        (GRAINS["three_ends"], False),
        (GRAINS["six_ends"], False),
    ],
)
def test_linear_or_circular(dnatrace, grain: np.ndarray, mol_is_circular: bool) -> None:
    """Test the linear_or_circular method with a range of different structures."""
    linear_coordinates = np.argwhere(grain == 1)
    dnatrace.linear_or_circular(linear_coordinates)
    assert dnatrace.mol_is_circular == mol_is_circular


TEST_LABELLED = np.asarray(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0, 0, 2, 2, 2, 2, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0],
        [0, 3, 3, 3, 3, 3, 3, 0, 0, 2, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 0, 3, 0, 0, 2, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 0, 3, 0, 0, 2, 2, 2, 2, 2, 0],
        [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 3, 0, 4, 4, 4, 4, 4, 4, 0],
        [0, 0, 0, 0, 0, 0, 3, 0, 4, 4, 4, 4, 4, 4, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 5, 5, 5, 5, 6, 6, 6, 6, 0, 0, 6, 0, 0, 0],
        [0, 5, 5, 0, 0, 6, 0, 0, 6, 0, 0, 6, 0, 0, 0],
        [0, 5, 5, 5, 5, 6, 0, 0, 6, 6, 6, 6, 6, 6, 0],
        [0, 0, 0, 5, 5, 6, 6, 6, 6, 0, 0, 6, 0, 0, 0],
        [0, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
)


@pytest.mark.parametrize(
    ("bounding_box", "target", "pad_width"),
    [
        (
            (1, 1, 2, 7),
            np.asarray(
                [
                    [1, 1, 1, 1, 1, 1],
                ]
            ),
            0,
        ),
        (
            (1, 1, 2, 7),
            np.asarray(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
            1,
        ),
        (
            (1, 1, 2, 7),
            np.asarray(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 3, 3, 3, 3, 3, 3, 0, 0],
                ]
            ),
            2,
        ),
        (
            (1, 9, 6, 14),
            np.asarray(
                [
                    [2, 2, 2, 2, 2],
                    [2, 0, 0, 0, 2],
                    [2, 0, 0, 0, 2],
                    [2, 0, 0, 0, 2],
                    [2, 2, 2, 2, 2],
                ]
            ),
            0,
        ),
        (
            (3, 1, 9, 7),
            np.asarray(
                [
                    [3, 3, 3, 3, 3, 3],
                    [0, 0, 0, 0, 0, 3],
                    [0, 0, 0, 0, 0, 3],
                    [0, 0, 0, 0, 0, 3],
                    [0, 0, 0, 0, 0, 3],
                    [0, 0, 0, 0, 0, 3],
                ]
            ),
            0,
        ),
        (
            (7, 8, 9, 14),
            np.asarray(
                [
                    [4, 4, 4, 4, 4, 4],
                    [4, 4, 4, 4, 4, 4],
                ]
            ),
            0,
        ),
        (
            (10, 1, 15, 5),
            np.asarray(
                [
                    [5, 5, 5, 5],
                    [5, 5, 0, 0],
                    [5, 5, 5, 5],
                    [0, 0, 5, 5],
                    [5, 5, 5, 5],
                ]
            ),
            0,
        ),
        (
            (10, 5, 14, 14),
            np.asarray(
                [
                    [6, 6, 6, 6, 0, 0, 6, 0, 0],
                    [6, 0, 0, 6, 0, 0, 6, 0, 0],
                    [6, 0, 0, 6, 6, 6, 6, 6, 6],
                    [6, 6, 6, 6, 0, 0, 6, 0, 0],
                ]
            ),
            0,
        ),
        (
            (10, 5, 14, 14),
            np.asarray(
                [
                    [0, 0, 0, 3, 0, 4, 4, 4, 4, 4, 4, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [5, 5, 6, 6, 6, 6, 0, 0, 6, 0, 0, 0],
                    [0, 0, 6, 0, 0, 6, 0, 0, 6, 0, 0, 0],
                    [5, 5, 6, 0, 0, 6, 6, 6, 6, 6, 6, 0],
                    [5, 5, 6, 6, 6, 6, 0, 0, 6, 0, 0, 0],
                    [5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
            2,
        ),
    ],
)
def test_crop_array(bounding_box: tuple, target: np.array, pad_width: int) -> None:
    """Test the cropping of images."""
    cropped = crop_array(TEST_LABELLED, bounding_box, pad_width)
    print(f"cropped :\n{cropped}")
    np.testing.assert_array_equal(cropped, target)
