"""Additional tests of dnaTracing methods"""
from pathlib import Path

import numpy as np
import pytest
import skimage.measure as skimage_measure

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


@pytest.fixture
def dnatrace() -> dnaTrace:
    """Instantiated object of class dnaTrace for use in tests."""
    _dnatrace = dnaTrace(
        full_image_data=np.asarray([[1]]),
        grains=None,
        filename="test.spm",
        pixel_size=PIXEL_SIZE,
        min_skeleton_size=MIN_SKELETON_SIZE,
    )
    return _dnatrace


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
    "grain, num_linear, num_circular",
    [
        (GRAINS["vertical"], 1, 0),
        (GRAINS["horizontal"], 1, 0),
        (GRAINS["diagonal1"], 0, 1),  # This is wrong, this IS a linear molecule
        (GRAINS["diagonal2"], 1, 0),
        (GRAINS["diagonal3"], 1, 0),
        (GRAINS["circle"], 0, 1),
        (GRAINS["blob"], 0, 1),
        (GRAINS["cross"], 1, 0),
        (GRAINS["single_L"], 1, 0),
        (GRAINS["double_L"], 0, 1),  # This is wrong, this IS a linear molecule
        (GRAINS["diagonal_end_single_L"], 1, 0),
        (GRAINS["diagonal_end_straight"], 1, 0),
        (GRAINS["figure8"], 0, 1),
        (GRAINS["three_ends"], 1, 0),
        (GRAINS["six_ends"], 1, 0),
    ],
)
def test_linear_or_circular(dnatrace, grain: np.ndarray, num_linear: int, num_circular: int) -> None:
    """Test the linear_or_circular method with a range of different structures."""
    linear_coordinates = {1: np.argwhere(grain == 1)}
    dnatrace.linear_or_circular(linear_coordinates)
    assert dnatrace.num_linear == num_linear
    assert dnatrace.num_circular == num_circular


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
    "bounding_box,target",
    [
        (
            (1, 1, 2, 7),
            np.asarray(
                [
                    [1, 1, 1, 1, 1, 1],
                ]
            ),
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
        ),
        (
            (7, 8, 9, 14),
            np.asarray(
                [
                    [4, 4, 4, 4, 4, 4],
                    [4, 4, 4, 4, 4, 4],
                ]
            ),
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
        ),
    ],
)
def test_crop_array(bounding_box: tuple, target: np.array) -> None:
    """Test the cropping of images."""
    check = skimage_measure.regionprops(TEST_LABELLED)
    for x in check:
        print(x.bbox)
    cropped = crop_array(TEST_LABELLED, bounding_box)
    np.testing.assert_array_equal(cropped, target)


def test_tracedna():
    """Test tracedna function."""
