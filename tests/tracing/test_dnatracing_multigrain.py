"""Tests for tracing images with multiple (2) grains."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from topostats.tracing.dnatracing import trace_image, prep_arrays, trace_mask

# This is required because of the inheritance used throughout
# pylint: disable=redefined-outer-name
BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"
PIXEL_SIZE = 0.4940029296875
MIN_SKELETON_SIZE = 10
PAD_WIDTH = 1

LINEAR_IMAGE = np.load(RESOURCES / "dnatracing_image_linear.npy")
LINEAR_MASK = np.load(RESOURCES / "dnatracing_mask_linear.npy")
CIRCULAR_IMAGE = np.load(RESOURCES / "dnatracing_image_circular.npy")
CIRCULAR_MASK = np.load(RESOURCES / "dnatracing_mask_circular.npy")
PADDED_LINEAR_IMAGE = np.pad(LINEAR_IMAGE, ((7, 6), (4, 4)))
PADDED_LINEAR_MASK = np.pad(LINEAR_MASK, ((7, 6), (4, 4)))
CIRCULAR_MASK = np.where(CIRCULAR_MASK == 1, 2, CIRCULAR_MASK)
MULTIGRAIN_IMAGE = np.concatenate((PADDED_LINEAR_IMAGE, CIRCULAR_IMAGE), axis=0)
MULTIGRAIN_MASK = np.concatenate((PADDED_LINEAR_MASK, CIRCULAR_MASK), axis=0)


RNG = np.random.default_rng(seed=1000)
SMALL_ARRAY_SIZE = (10, 10)
SMALL_ARRAY = np.asarray(RNG.random(SMALL_ARRAY_SIZE))
SMALL_MASK = np.asarray(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 2, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
        [0, 3, 3, 3, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
)


@pytest.mark.parametrize(
    ("pad_width", "target_image", "target_mask"),
    [
        (
            0,
            [
                np.asarray(
                    [
                        [0.24837266, 0.18985741, 0.98399558],
                        [0.15393237, 0.69908928, 0.44724145],
                        [0.34320907, 0.8119946, 0.148494],
                    ]
                ),
                np.asarray([[0.321028], [0.808018], [0.856378], [0.835525]]),
                np.asarray([[0.658189, 0.516731, 0.823857], [0.431365, 0.076602, 0.098614]]),
            ],
            [
                np.asarray([[1, 1, 1], [1, 0, 0], [1, 0, 0]]),
                np.asarray([[1], [1], [1], [1]]),
                np.asarray([[0, 0, 1], [1, 1, 1]]),
            ],
        ),
        (
            1,
            [
                np.asarray(
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.52138574, 0.60384185, 0.4709418, 0.20324794, 0.52875903, 0.0],
                        [0.0, 0.80537222, 0.24837266, 0.18985741, 0.98399558, 0.66999717, 0.0],
                        [0.0, 0.97476378, 0.15393237, 0.69908928, 0.44724145, 0.01751321, 0.0],
                        [0.0, 0.13645032, 0.34320907, 0.8119946, 0.148494, 0.05932569, 0.0],
                        [0.0, 0.55868699, 0.00288863, 0.29775757, 0.05379911, 0.56766875, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ),
                np.asarray(
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.20391323, 0.62506469, 0.65260432, 0.0],
                        [0.0, 0.38123661, 0.32102791, 0.94254467, 0.0],
                        [0.0, 0.42015645, 0.80801771, 0.00950759, 0.0],
                        [0.0, 0.72427372, 0.85637809, 0.54431565, 0.0],
                        [0.0, 0.29894051, 0.83552541, 0.18450649, 0.0],
                        [0.0, 0.39284504, 0.45345328, 0.27428462, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ),
                np.asarray(
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.60442473, 0.73460399, 0.98840812, 0.89224091, 0.51964178, 0.0],
                        [0.0, 0.23502814, 0.65818858, 0.51673102, 0.82385723, 0.18965801, 0.0],
                        [0.0, 0.62741705, 0.43136525, 0.07660225, 0.09861362, 0.06647744, 0.0],
                        [0.0, 0.69874299, 0.88569365, 0.93542321, 0.19316749, 0.95909555, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ),
            ],
            [
                np.asarray(
                    [
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 1, 1, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                    ]
                ),
                np.asarray(
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                np.asarray(
                    [
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 1, 1, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                    ]
                ),
            ],
        ),
    ],
)
def test_prep_arrays(pad_width: int, target_image: np.ndarray, target_mask: np.ndarray) -> None:
    """Tests the image and masks are correctly prepared to lists."""
    images, masks = prep_arrays(image=SMALL_ARRAY, labelled_grains_mask=SMALL_MASK, pad_width=pad_width)
    grain = 0
    for image, mask in zip(images, masks):  # noqa: PT011
        np.testing.assert_array_almost_equal(image, target_image[grain])
        np.testing.assert_array_equal(mask, target_mask[grain])
        grain += 1


def test_image_trace_unequal_arrays() -> None:
    """Test arrays that are unequal throw a ValueError."""
    irregular_mask = np.zeros((MULTIGRAIN_IMAGE.shape[0] + 4, MULTIGRAIN_IMAGE.shape[1] + 5))
    with pytest.raises(ValueError):  # noqa: PT011
        trace_image(
            image=MULTIGRAIN_IMAGE,
            grains_mask=irregular_mask,
            filename="dummy",
            pixel_to_nm_scaling=PIXEL_SIZE,
            min_skeleton_size=MIN_SKELETON_SIZE,
            skeletonisation_method="topostats",
            pad_width=PAD_WIDTH,
            cores=1,
        )


TARGET_ARRAY = np.asarray(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
)


@pytest.mark.parametrize(
    ("grain_anchors", "ordered_traces", "image_shape", "expected", "pad_width"),
    [
        # pad_width = 0
        (
            [[0, 0], [0, 9], [7, 0], [5, 4], [10, 7]],
            [
                np.asarray([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6]]),  # Horizontal grain
                np.asarray([[1, 1], [2, 1], [3, 1], [4, 1]]),  # Vertical grain
                np.asarray([[1, 1], [2, 2], [3, 3], [4, 4]]),  # Diagnoal grain
                np.asarray([[1, 1], [1, 2], [1, 3], [1, 4], [2, 4], [3, 4]]),  # L-shaped grain grain
                np.asarray(  # Small square
                    [
                        [1, 1],
                        [1, 2],
                        [1, 3],
                        [1, 4],
                        [2, 1],
                        [2, 4],
                        [3, 1],
                        [3, 4],
                        [4, 1],
                        [4, 2],
                        [4, 3],
                        [4, 4],
                    ]
                ),
            ],
            (16, 13),
            TARGET_ARRAY,
            0,
        ),
        # pad_width = 1
        (
            [[0, 0], [0, 9], [7, 0], [4, 3], [9, 6]],
            [
                np.asarray([[2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7]]),  # Horizontal grain
                np.asarray([[2, 2], [3, 2], [4, 2], [5, 2]]),  # Vertical grain
                np.asarray([[2, 2], [3, 3], [4, 4], [5, 5]]),  # Diagnoal grain
                np.asarray([[3, 3], [3, 4], [3, 5], [3, 6], [4, 6], [5, 6]]),  # L-shaped grain grain
                np.asarray(  # Small square
                    [
                        [3, 3],
                        [3, 4],
                        [3, 5],
                        [3, 6],
                        [4, 3],
                        [4, 6],
                        [5, 3],
                        [5, 6],
                        [6, 3],
                        [6, 4],
                        [6, 5],
                        [6, 6],
                    ]
                ),
            ],
            (16, 13),
            TARGET_ARRAY,
            1,
        ),
        # pad_width = 2
        (
            [[0, 0], [0, 9], [7, 0], [4, 3], [9, 6]],
            [
                np.asarray([[3, 3], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8]]),  # Horizontal grain
                np.asarray([[3, 3], [4, 3], [5, 3], [6, 3]]),  # Vertical grain
                np.asarray([[3, 3], [4, 4], [5, 5], [6, 6]]),  # Diagnoal grain
                np.asarray([[4, 4], [4, 5], [4, 6], [4, 7], [5, 7], [6, 7]]),  # L-shaped grain grain
                np.asarray(  # Small square
                    [
                        [4, 4],
                        [4, 5],
                        [4, 6],
                        [4, 7],
                        [5, 4],
                        [5, 7],
                        [6, 4],
                        [6, 7],
                        [7, 4],
                        [7, 5],
                        [7, 6],
                        [7, 7],
                    ]
                ),
            ],
            (16, 13),
            TARGET_ARRAY,
            2,
        ),
    ],
)
def test_trace_mask(
    grain_anchors: list, ordered_traces: list, image_shape: tuple, pad_width: int, expected: np.ndarray
) -> None:
    """Test the trace_mask."""
    image = trace_mask(grain_anchors, ordered_traces, image_shape, pad_width)
    np.testing.assert_array_equal(image, expected)


@pytest.mark.parametrize(
    ("image", "skeletonisation_method", "cores", "statistics", "ordered_trace_start", "ordered_trace_end"),
    [
        (
            "multigrain_topostats",
            "topostats",
            1,
            pd.DataFrame(
                {
                    "molecule_number": [0, 1],
                    "image": ["multigrain_topostats", "multigrain_topostats"],
                    "contour_length": [5.684734982126663e-08, 7.574136072208753e-08],
                    "circular": [False, True],
                    "end_to_end_distance": [3.120049919984285e-08, 0.000000e00],
                }
            ),
            [np.asarray([6, 25]), np.asarray([31, 32])],
            [np.asarray([65, 47]), np.asarray([31, 31])],
        ),
        (
            "multigrain_zhang",
            "zhang",
            1,
            pd.DataFrame(
                {
                    "molecule_number": [0, 1],
                    "image": ["multigrain_zhang", "multigrain_zhang"],
                    "contour_length": [6.194694383968301e-08, 8.187508931608563e-08],
                    "circular": [False, False],
                    "end_to_end_distance": [2.257869018994927e-08, 1.2389530445725336e-08],
                }
            ),
            [np.asarray([5, 28]), np.asarray([16, 66])],
            [np.asarray([61, 32]), np.asarray([33, 54])],
        ),
        (
            "multigrain_lee",
            "lee",
            1,
            pd.DataFrame(
                {
                    "molecule_number": [0, 1],
                    "image": ["multigrain_lee", "multigrain_lee"],
                    "contour_length": [5.6550320018177204e-08, 8.062559919860786e-08],
                    "circular": [False, False],
                    "end_to_end_distance": [3.13837693459974e-08, 6.7191662793734405e-09],
                }
            ),
            [np.asarray([4, 23]), np.asarray([18, 65])],
            [np.asarray([65, 47]), np.asarray([34, 54])],
        ),
        (
            "multigrain_thin",
            "thin",
            1,
            pd.DataFrame(
                {
                    "molecule_number": [0, 1],
                    "image": ["multigrain_thin", "multigrain_thin"],
                    "contour_length": [5.4926652806911664e-08, 3.6512544238919696e-08],
                    "circular": [False, False],
                    "end_to_end_distance": [4.367667613976452e-08, 3.440332307376993e-08],
                }
            ),
            [np.asarray([5, 23]), np.asarray([10, 58])],
            [np.asarray([71, 78]), np.asarray([83, 30])],
        ),
    ],
)
def test_trace_image(
    image: str, skeletonisation_method: str, cores: int, statistics, ordered_trace_start: list, ordered_trace_end: list
) -> None:
    """Tests the processing of an image using trace_image() function.

    NB - This test isn't complete, there is only limited testing of the results["ordered_traces"].
         The results["image_trace"] that are not tested either, these are large arrays and constructing them in the test
         is cumbersome.
         Initial attempts at using SMALL_ARRAY/SMALL_MASK were unsuccessful as they were not traced because the grains
         are < min_skeleton_size, adjusting this to 1 didn't help they still weren't skeletonised.
    """
    results = trace_image(
        image=MULTIGRAIN_IMAGE,
        grains_mask=MULTIGRAIN_MASK,
        filename=image,
        pixel_to_nm_scaling=PIXEL_SIZE,
        min_skeleton_size=MIN_SKELETON_SIZE,
        skeletonisation_method=skeletonisation_method,
        pad_width=PAD_WIDTH,
        cores=cores,
    )
    statistics.set_index(["molecule_number"], inplace=True)
    pd.testing.assert_frame_equal(results["statistics"], statistics)
    for ordered_trace, start, end in zip(results["ordered_traces"], ordered_trace_start, ordered_trace_end):
        np.testing.assert_array_equal(ordered_trace[1], start)
        np.testing.assert_array_equal(ordered_trace[-1], end)
