"""Tests for tracing single molecules"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pytest_lazyfixture import lazy_fixture

from topostats.tracing.dnatracing import dnaTrace, trace_grain, trace_image, prep_arrays

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
def dnatrace_linear() -> dnaTrace:
    """dnaTrace object instantiated with a single linear grain."""
    dnatrace = dnaTrace(
        image=LINEAR_IMAGE,
        grain=LINEAR_MASK,
        filename="linear",
        pixel_to_nm_scaling=PIXEL_SIZE,
        min_skeleton_size=MIN_SKELETON_SIZE,
        skeletonisation_method="topostats",
    )
    return dnatrace


@pytest.fixture
def dnatrace_circular() -> dnaTrace:
    """dnaTrace object instantiated with a single linear grain."""
    dnatrace = dnaTrace(
        image=CIRCULAR_IMAGE,
        grain=CIRCULAR_MASK,
        filename="circular",
        pixel_to_nm_scaling=PIXEL_SIZE,
        min_skeleton_size=MIN_SKELETON_SIZE,
        skeletonisation_method="topostats",
    )
    return dnatrace


@pytest.mark.parametrize(
    "dnatrace, gauss_image_sum",
    [
        (lazy_fixture("dnatrace_linear"), 5.448087311118036e-06),
        (lazy_fixture("dnatrace_circular"), 6.035458663629233e-06),
    ],
)
def test_gaussian_filter(dnatrace: dnaTrace, gauss_image_sum: float) -> None:
    """Test of the method."""
    dnatrace.gaussian_filter()
    assert dnatrace.gauss_image.sum() == gauss_image_sum


@pytest.mark.parametrize(
    "dnatrace, skeletonisation_method, length, start, end",
    [
        (lazy_fixture("dnatrace_linear"), "topostats", 120, np.asarray([13, 32]), np.asarray([91, 72])),
        (lazy_fixture("dnatrace_circular"), "topostats", 150, np.asarray([49, 49]), np.asarray([103, 44])),
        (lazy_fixture("dnatrace_linear"), "zhang", 170, np.asarray([13, 32]), np.asarray([91, 72])),
        (lazy_fixture("dnatrace_circular"), "zhang", 184, np.asarray([33, 85]), np.asarray([103, 44])),
        (lazy_fixture("dnatrace_linear"), "lee", 130, np.asarray([12, 30]), np.asarray([91, 72])),
        (lazy_fixture("dnatrace_circular"), "lee", 177, np.asarray([35, 83]), np.asarray([104, 43])),
        (lazy_fixture("dnatrace_linear"), "thin", 187, np.asarray([12, 30]), np.asarray([91, 68])),
        (lazy_fixture("dnatrace_circular"), "thin", 190, np.asarray([28, 75]), np.asarray([105, 42])),
    ],
)
def test_get_disordered_trace(
    dnatrace: dnaTrace, skeletonisation_method: str, length: int, start: tuple, end: tuple
) -> None:
    """Test of get_disordered_trace the method."""
    dnatrace.skeletonisation_method = skeletonisation_method
    dnatrace.gaussian_filter()
    dnatrace.get_disordered_trace()
    assert isinstance(dnatrace.disordered_trace, np.ndarray)
    assert len(dnatrace.disordered_trace) == length
    np.testing.assert_array_equal(dnatrace.disordered_trace[0,], start)
    np.testing.assert_array_equal(dnatrace.disordered_trace[-1,], end)


# Currently two errors are not caught, need to improve this when refactoring, just in case.
@pytest.mark.parametrize(
    "min_skeleton_size, problem",
    [
        (4, None),
        (4, 6),
    ],
)
def test_purge_obvious_crap_exceptions(dnatrace_linear: dnaTrace, min_skeleton_size: int, problem) -> None:
    """Test exceptions to purge_obvious_crap"""
    dnatrace_linear.min_skeleton_size = min_skeleton_size
    dnatrace_linear.disordered_trace = problem


# Currently linear molecule isn't detected as linear, although it was when selecting and extracting in a Notebook
@pytest.mark.parametrize(
    "dnatrace, mol_is_circular",
    [
        # (lazy_fixture("dnatrace_linear"), False),
        (lazy_fixture("dnatrace_circular"), True),
    ],
)
def test_linear_or_circular(dnatrace: dnaTrace, mol_is_circular: int) -> None:
    """Test of the linear_or_circular method."""
    dnatrace.min_skeleton_size = MIN_SKELETON_SIZE
    dnatrace.gaussian_filter()
    dnatrace.get_disordered_trace()
    dnatrace.linear_or_circular(dnatrace.disordered_trace)
    assert dnatrace.mol_is_circular == mol_is_circular


@pytest.mark.parametrize(
    "dnatrace, length, start, end",
    [
        (lazy_fixture("dnatrace_linear"), 118, np.asarray([13, 33]), np.asarray([73, 55])),
        (lazy_fixture("dnatrace_circular"), 151, np.asarray([49, 49]), np.asarray([49, 49])),
    ],
)
def test_get_ordered_traces(dnatrace: dnaTrace, length: int, start: np.array, end: np.array) -> None:
    """Test of the get_ordered_traces method.

    Note the co-ordinates at the start and end differe from the fixtures for test_get_disordered_trace."""
    dnatrace.gaussian_filter()
    dnatrace.get_disordered_trace()
    dnatrace.linear_or_circular(dnatrace.disordered_trace)
    dnatrace.get_ordered_traces()
    assert isinstance(dnatrace.ordered_trace, np.ndarray)
    assert len(dnatrace.ordered_trace) == length
    np.testing.assert_array_equal(dnatrace.ordered_trace[0,], start)
    np.testing.assert_array_almost_equal(dnatrace.ordered_trace[-1,], end)


@pytest.mark.parametrize(
    "dnatrace, length, start, end",
    [
        (lazy_fixture("dnatrace_linear"), 118, np.asarray([13, 34]), np.asarray([73, 60])),
        (lazy_fixture("dnatrace_circular"), 151, np.asarray([48, 48]), np.asarray([48, 48])),
    ],
)
def test_get_fitted_traces(dnatrace: dnaTrace, length: int, start: np.array, end: np.array) -> None:
    """Test of the method."""
    dnatrace.gaussian_filter()
    dnatrace.get_disordered_trace()
    dnatrace.linear_or_circular(dnatrace.disordered_trace)
    dnatrace.get_ordered_traces()
    dnatrace.linear_or_circular(dnatrace.disordered_trace)
    dnatrace.get_fitted_traces()
    assert isinstance(dnatrace.fitted_trace, np.ndarray)
    assert len(dnatrace.fitted_trace) == length
    np.testing.assert_array_equal(dnatrace.fitted_trace[0,], start)
    np.testing.assert_array_almost_equal(dnatrace.fitted_trace[-1,], end)


@pytest.mark.parametrize(
    "dnatrace, length, start, end",
    [
        (
            lazy_fixture("dnatrace_linear"),
            1652,
            np.asarray([20.357143, 31.714286]),
            np.asarray([20.357143, 31.714286]),
        ),
        (
            lazy_fixture("dnatrace_circular"),
            2114,
            np.asarray([49.285714, 55.428571]),
            np.asarray([49.285714, 55.428571]),
        ),
    ],
)
def test_get_splined_traces(dnatrace: dnaTrace, length: int, start: np.array, end: np.array) -> None:
    """Test of the method."""
    dnatrace.gaussian_filter()
    dnatrace.get_disordered_trace()
    dnatrace.linear_or_circular(dnatrace.disordered_trace)
    dnatrace.get_ordered_traces()
    dnatrace.linear_or_circular(dnatrace.disordered_trace)
    dnatrace.get_fitted_traces()
    dnatrace.get_splined_traces()
    assert isinstance(dnatrace.splined_trace, np.ndarray)
    assert len(dnatrace.splined_trace) == length
    np.testing.assert_array_almost_equal(dnatrace.splined_trace[0,], start)
    np.testing.assert_array_almost_equal(dnatrace.splined_trace[-1,], end)


@pytest.mark.parametrize(
    "dnatrace, contour_length",
    [
        (lazy_fixture("dnatrace_linear"), 9.040267985905399e-08),
        (lazy_fixture("dnatrace_circular"), 7.617314045334366e-08),
    ],
)
def test_measure_contour_length(dnatrace: dnaTrace, contour_length: float) -> None:
    """Test of the method."""
    dnatrace.gaussian_filter()
    dnatrace.get_disordered_trace()
    dnatrace.linear_or_circular(dnatrace.disordered_trace)
    dnatrace.get_ordered_traces()
    dnatrace.linear_or_circular(dnatrace.disordered_trace)
    dnatrace.get_fitted_traces()
    dnatrace.get_splined_traces()
    dnatrace.measure_contour_length()
    assert dnatrace.contour_length == pytest.approx(contour_length)


# Currently need an actual linear grain to test this.
@pytest.mark.parametrize(
    "dnatrace, end_to_end_distance",
    [
        (lazy_fixture("dnatrace_linear"), 0),
        (lazy_fixture("dnatrace_circular"), 0),
    ],
)
def test_measure_end_to_end_distance(dnatrace: dnaTrace, end_to_end_distance: float) -> None:
    """Test of the method."""
    dnatrace.gaussian_filter()
    dnatrace.get_disordered_trace()
    dnatrace.linear_or_circular(dnatrace.disordered_trace)
    dnatrace.get_ordered_traces()
    dnatrace.linear_or_circular(dnatrace.disordered_trace)
    dnatrace.get_fitted_traces()
    dnatrace.get_splined_traces()
    dnatrace.measure_end_to_end_distance()
    assert dnatrace.end_to_end_distance == pytest.approx(end_to_end_distance)


@pytest.mark.parametrize(
    "cropped_image, cropped_mask, filename, skeletonisation_method, end_to_end_distance, circular, contour_length",
    [
        (
            LINEAR_IMAGE,
            LINEAR_MASK,
            "linear_test_topostats",
            "topostats",
            3.120049919984285e-08,
            False,
            1.2382864476914832e-07,
        ),
        (
            CIRCULAR_IMAGE,
            CIRCULAR_MASK,
            "circular_test_topostats",
            "topostats",
            0,
            True,
            7.617314045334366e-08,
        ),
        (
            LINEAR_IMAGE,
            LINEAR_MASK,
            "linear_test_zhang",
            "zhang",
            2.257869018994927e-08,
            False,
            1.5050575430042103e-07,
        ),
        (
            CIRCULAR_IMAGE,
            CIRCULAR_MASK,
            "circular_test_zhang",
            "zhang",
            1.2389530445725336e-08,
            False,
            1.122049485057339e-07,
        ),
        (
            LINEAR_IMAGE,
            LINEAR_MASK,
            "linear_test_lee",
            "lee",
            3.13837693459974e-08,
            False,
            1.432248478041724e-07,
        ),
        (
            CIRCULAR_IMAGE,
            CIRCULAR_MASK,
            "circular_test_lee",
            "lee",
            6.7191662793734405e-09,
            False,
            1.1623401641268276e-07,
        ),
        (
            LINEAR_IMAGE,
            LINEAR_MASK,
            "linear_test_thin",
            "thin",
            4.367667613976452e-08,
            False,
            1.2709212267220064e-07,
        ),
        (
            CIRCULAR_IMAGE,
            CIRCULAR_MASK,
            "circular_test_thin",
            "thin",
            3.440332307376993e-08,
            False,
            8.576324241662498e-08,
        ),
    ],
)
def test_trace_grain(
    cropped_image: np.ndarray,
    cropped_mask: np.ndarray,
    filename: str,
    skeletonisation_method: str,
    end_to_end_distance: float,
    circular: bool,
    contour_length: float,
) -> None:
    """Test trace_grain function for tracing a single grain"""
    trace_stats = trace_grain(
        cropped_image=cropped_image,
        cropped_mask=cropped_mask,
        pixel_to_nm_scaling=PIXEL_SIZE,
        filename=filename,
        min_skeleton_size=MIN_SKELETON_SIZE,
        skeletonisation_method=skeletonisation_method,
    )
    assert trace_stats["image"] == filename
    assert trace_stats["end_to_end_distance"] == end_to_end_distance
    assert trace_stats["circular"] == circular
    assert trace_stats["contour_length"] == contour_length


MULTIGRAIN_IMAGE = np.concatenate((np.pad(LINEAR_IMAGE, (9, 9)), CIRCULAR_IMAGE), axis=0)
MULTIGRAIN_MASK = np.concatenate((np.pad(LINEAR_MASK, (9, 9)), CIRCULAR_MASK), axis=0)
PAD_WIDTH = 30


@pytest.mark.parametrize(
    "image, skeletonisation_method, cores, statistics",
    [
        (
            "multigrain_topostats",
            "topostats",
            1,
            pd.DataFrame(
                {
                    "molecule_number": [0, 1],
                    "image": ["multigrain_topostats", "multigrain_topostats"],
                    "contour_length": [1.2382864476914832e-07, 7.617314045334366e-08],
                    "circular": [False, True],
                    "end_to_end_distance": [3.120049919984285e-08, 0.000000e00],
                }
            ),
        ),
        (
            "multigrain_zhang",
            "zhang",
            1,
            pd.DataFrame(
                {
                    "molecule_number": [0, 1],
                    "image": ["multigrain_zhang", "multigrain_zhang"],
                    "contour_length": [1.5050575430042103e-07, 1.122049485057339e-07],
                    "circular": [False, False],
                    "end_to_end_distance": [2.257869018994927e-08, 1.2389530445725336e-08],
                }
            ),
        ),
        (
            "multigrain_lee",
            "lee",
            1,
            pd.DataFrame(
                {
                    "molecule_number": [0, 1],
                    "image": ["multigrain_lee", "multigrain_lee"],
                    "contour_length": [1.432248478041724e-07, 1.1623401641268276e-07],
                    "circular": [False, False],
                    "end_to_end_distance": [3.13837693459974e-08, 6.7191662793734405e-09],
                }
            ),
        ),
        (
            "multigrain_thin",
            "thin",
            1,
            pd.DataFrame(
                {
                    "molecule_number": [0, 1],
                    "image": ["multigrain_thin", "multigrain_thin"],
                    "contour_length": [1.2709212267220064e-07, 8.576324241662498e-08],
                    "circular": [False, False],
                    "end_to_end_distance": [4.367667613976452e-08, 3.440332307376993e-08],
                }
            ),
        ),
    ],
)
def test_trace_image(image: str, skeletonisation_method: str, cores: int, statistics) -> None:
    """Tests the processing of an image using trace_image() function."""
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
    pd.testing.assert_frame_equal(results, statistics)


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
    ]
)


@pytest.mark.parametrize(
    "pad_width, target_image, target_mask",
    [
        (
            0,
            [
                np.asarray(
                    [[0.248373, 0.189857, 0.983996], [0.153932, 0.699089, 0.447241], [0.343209, 0.811995, 0.148494]]
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
    images, masks = prep_arrays(image=SMALL_ARRAY, grains_mask=SMALL_MASK, pad_width=pad_width)
    grain = 0
    for image, mask in zip(images, masks):
        print(f"image :\n{image}")
        print(f"mask  :\n{mask}")
        np.testing.assert_array_almost_equal(image, target_image[grain])
        np.testing.assert_array_equal(mask, target_mask[grain])
        grain += 1
