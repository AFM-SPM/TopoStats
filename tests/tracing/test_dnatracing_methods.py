"""Additional tests of dnaTracing methods."""
from pathlib import Path

import numpy as np
import pytest

from topostats.tracing.dnatracing import dnaTrace, crop_array, round_splined_traces
from topostats.tracing.tracingfuncs import reorderTrace

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
    """dnaTrace object for use in tests."""  # noqa: D403
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


@pytest.mark.parametrize(
    ("trace_image", "step_size_m", "mol_is_circular", "smoothness", "expected_spline_image"),
    # Note that some images here have gaps in them. This is NOT an issue with the splining, but
    # a natural result of rounding the floating point coordinates to integers. We only do this
    # to provide a visualisation for easy test comparison and does not affect results.
    [
        # Test circular with no smoothing. The shape remains unchanged. The gap is just a result
        # of rounding the floating point coordinates to integers for this neat visualisation
        # and can be ignored.
        (
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 1, 1, 1, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
            1.0,
            True,
            0.0,
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 0, 1, 1, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 1, 1, 1, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
        ),
        # Test circular with smoothing. The shape is smoothed out.
        (
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 1, 1, 1, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
            1.0,
            True,
            10.0,
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 1, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
        ),
        # Another circular with no smoothing. Relatively unchanged
        (
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                    [0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
            10.0,
            True,
            0.0,
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                    [0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
        ),
        # Another circular with smoothing. The shape is smoothed out.
        (
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                    [0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
            10.0,
            True,
            5.0,
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
                    [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
        ),
        # Simple line with smoothing unchanged because too simple.
        (
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
            5.0,
            False,
            5.0,
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
        ),
        # Another line with smoothing unchanged because too simple.
        (
            np.array(
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
            ),
            4.0,
            False,
            5.0,
            np.array(
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
            ),
        ),
        # Complex line without smoothing, unchanged.
        (
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
            5.0,
            False,
            0.0,
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
        ),
        # Complex line with smoothing, smoothed out.
        (
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
            5.0,
            False,
            5.0,
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
        ),
    ],
)
def test_get_splined_traces(
    dnatrace: dnaTrace,
    trace_image: np.ndarray,
    step_size_m: float,
    mol_is_circular: bool,
    smoothness: float,
    expected_spline_image: np.ndarray,
) -> None:
    """Test the get_splined_traces() function of dnatracing.py."""
    # For development visualisations - keep this in for future use
    # plt.imsave('./fitted_trace.png', trace_image)

    # Obtain the coords of the pixels from our test image
    trace_coords = np.argwhere(trace_image == 1)

    # Get an ordered trace from our test trace images
    if mol_is_circular:
        fitted_trace, _whether_trace_completed = reorderTrace.circularTrace(trace_coords)
    else:
        fitted_trace = reorderTrace.linearTrace(trace_coords)

    # Set dnatrace smoothness accordingly
    if mol_is_circular:
        dnatrace.spline_circular_smoothness = smoothness
    else:
        dnatrace.spline_linear_smoothness = smoothness

    # Generate splined trace
    dnatrace.fitted_trace = fitted_trace
    dnatrace.step_size_m = step_size_m
    # Fixed pixel to nm scaling since changing this is redundant due to the internal effect being linked to
    # the step_size_m divided by this value, so changing both doesn't make sense.
    dnatrace.pixel_to_nm_scaling = 1.0
    dnatrace.mol_is_circular = mol_is_circular
    dnatrace.n_grain = 1

    # Spline the traces
    dnatrace.get_splined_traces()

    # Extract the splined trace
    splined_trace = dnatrace.splined_trace

    # This is just for easier human-readable tests. Turn the splined coords into a visualisation.
    splined_image = np.zeros_like(trace_image)
    splined_image[np.round(splined_trace[:, 0]).astype(int), np.round(splined_trace[:, 1]).astype(int)] = 1

    # For development visualisations - keep this in for future use
    # plt.imsave(f'./test_splined_image_{splined_image.shape}.png', splined_image)

    np.testing.assert_array_equal(splined_image, expected_spline_image)


def test_round_splined_traces():
    """Test the round splined traces function of dnatracing.py."""
    splined_traces = [np.array([[1.2, 2.3], [3.4, 4.5]]), None, np.array([[5.6, 6.7], [7.8, 8.9]])]
    expected_result = np.array([[[1, 2], [3, 4]], [[6, 7], [8, 9]]])
    result = round_splined_traces(splined_traces)
    np.testing.assert_array_equal(result, expected_result)
