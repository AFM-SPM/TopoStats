"""Tests for tracing single molecules"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pytest_lazyfixture import lazy_fixture

from topostats.tracing.dnatracing import dnaTrace, trace_grain, trace_image

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
        full_image_data=LINEAR_IMAGE,
        grains=LINEAR_MASK,
        filename="linear",
        pixel_size=PIXEL_SIZE,
        min_skeleton_size=MIN_SKELETON_SIZE,
        skeletonisation_method="topostats",
    )
    return dnatrace


@pytest.fixture
def dnatrace_circular() -> dnaTrace:
    """dnaTrace object instantiated with a single linear grain."""
    dnatrace = dnaTrace(
        full_image_data=CIRCULAR_IMAGE,
        grains=CIRCULAR_MASK,
        filename="circular",
        pixel_size=PIXEL_SIZE,
        min_skeleton_size=MIN_SKELETON_SIZE,
        skeletonisation_method="topostats",
    )
    return dnatrace


@pytest.mark.parametrize(
    "dnatrace, grain_number, shape, total_image",
    [
        (lazy_fixture("dnatrace_linear"), 3, (109, 100), 2443),
        (lazy_fixture("dnatrace_circular"), 9, (132, 118), 2722),
    ],
)
def test_get_numpy_arrays(dnatrace: dnaTrace, grain_number: int, shape: tuple, total_image: float) -> None:
    """Test of the get_numpy_arrays method and implicitly _get_grain_array method."""
    dnatrace.get_numpy_arrays()
    assert isinstance(dnatrace.grains[grain_number], np.ndarray)
    assert dnatrace.grains[grain_number].shape == shape
    assert np.sum(dnatrace.grains[grain_number]) == total_image


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
    "dnatrace, grain_number, skeletonisation_method, length, start, end",
    [
        (lazy_fixture("dnatrace_linear"), 3, "topostats", 120, np.asarray([13, 32]), np.asarray([91, 72])),
        (lazy_fixture("dnatrace_circular"), 9, "topostats", 150, np.asarray([49, 49]), np.asarray([103, 44])),
        (lazy_fixture("dnatrace_linear"), 3, "zhang", 170, np.asarray([13, 32]), np.asarray([91, 72])),
        (lazy_fixture("dnatrace_circular"), 9, "zhang", 184, np.asarray([33, 85]), np.asarray([103, 44])),
        (lazy_fixture("dnatrace_linear"), 3, "lee", 130, np.asarray([12, 30]), np.asarray([91, 72])),
        (lazy_fixture("dnatrace_circular"), 9, "lee", 177, np.asarray([35, 83]), np.asarray([104, 43])),
        (lazy_fixture("dnatrace_linear"), 3, "thin", 187, np.asarray([12, 30]), np.asarray([91, 68])),
        (lazy_fixture("dnatrace_circular"), 9, "thin", 190, np.asarray([28, 75]), np.asarray([105, 42])),
    ],
)
def test_get_disordered_trace(
    dnatrace: dnaTrace, grain_number: int, skeletonisation_method: str, length: int, start: tuple, end: tuple
) -> None:
    """Test of get_disordered_trace the method."""
    dnatrace.skeletonisation_method = skeletonisation_method
    dnatrace.get_numpy_arrays()
    dnatrace.gaussian_filter()
    dnatrace.get_disordered_trace()
    assert isinstance(dnatrace.disordered_trace[grain_number], np.ndarray)
    assert len(dnatrace.disordered_trace[grain_number]) == length
    np.testing.assert_array_equal(dnatrace.disordered_trace[grain_number][0,], start)
    np.testing.assert_array_equal(dnatrace.disordered_trace[grain_number][-1,], end)


@pytest.mark.parametrize(
    "dna_num, min_skeleton_size, disordered_trace, expected_len",
    [
        ([1], 4, [["a", "b", "c"]], 0),
        ([2], 4, [[1, 2, 3, 4]], 1),
        ([3], 4, [[1, 2, 3, 4, 5]], 1),
        ([4], 4, [[0]], 0),
        ([5, 6], 4, [[0], [1, 2, 3, 4, 5]], 1),
        ([7, 8, 9], 4, [[0], [1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], 2),
    ],
)
def test_purge_obvious_crap(
    dnatrace_linear: dnaTrace, dna_num: int, min_skeleton_size: int, disordered_trace: list, expected_len: int
) -> None:
    """Test of the purge_obvious_crap method."""
    dnatrace_linear.min_skeleton_size = min_skeleton_size
    for x, i in enumerate(dna_num):
        dnatrace_linear.disordered_trace[i] = disordered_trace[x]
    dnatrace_linear.purge_obvious_crap()
    assert len(dnatrace_linear.disordered_trace) == expected_len


# Currently two errors are not caught, need to improve this when refactoring, just in case.
@pytest.mark.parametrize(
    "dna_num, min_skeleton_size, problem, exception",
    [
        (1, 4, None, TypeError),
        # (2, 4, {"a": 1, "b": 2}, TypeError),
        (3, 4, 6, TypeError),
        # (4, 4, "abcde", AttributeError),
    ],
)
def test_purge_obvious_crap_exceptions(
    dnatrace_linear: dnaTrace, dna_num: list, min_skeleton_size: int, problem, exception
) -> None:
    """Test exceptions to purge_obvious_crap"""
    dnatrace_linear.min_skeleton_size = min_skeleton_size
    dnatrace_linear.disordered_trace[dna_num] = problem
    with pytest.raises(exception):
        dnatrace_linear.purge_obvious_crap()


# Currently linear molecule isn't detected as linear, although it was when selecting and extracting in a Notebook
@pytest.mark.parametrize(
    "dnatrace, num_circular, num_linear",
    [
        # (lazy_fixture("dnatrace_linear"), 0, 1),
        (lazy_fixture("dnatrace_circular"), 1, 0),
    ],
)
def test_linear_or_circular(dnatrace: dnaTrace, num_circular: int, num_linear: int) -> None:
    """Test of the linear_or_circular method."""
    dnatrace.min_skeleton_size = MIN_SKELETON_SIZE
    dnatrace.get_numpy_arrays()
    dnatrace.gaussian_filter()
    dnatrace.get_disordered_trace()
    dnatrace.purge_obvious_crap()
    dnatrace.linear_or_circular(dnatrace.disordered_trace)
    assert dnatrace.num_linear == num_linear
    assert dnatrace.num_circular == num_circular


@pytest.mark.parametrize(
    "dnatrace, grain_number, length, start, end",
    [
        (lazy_fixture("dnatrace_linear"), 3, 118, np.asarray([13, 33]), np.asarray([73, 55])),
        (lazy_fixture("dnatrace_circular"), 9, 151, np.asarray([49, 49]), np.asarray([49, 49])),
    ],
)
def test_get_ordered_traces(dnatrace: dnaTrace, grain_number: int, length: int, start: np.array, end: np.array) -> None:
    """Test of the get_ordered_traces method.

    Note the co-ordinates at the start and end differe from the fixtures for test_get_disordered_trace."""
    dnatrace.get_numpy_arrays()
    dnatrace.gaussian_filter()
    dnatrace.get_disordered_trace()
    dnatrace.purge_obvious_crap()
    dnatrace.linear_or_circular(dnatrace.disordered_trace)
    dnatrace.get_ordered_traces()
    assert isinstance(dnatrace.ordered_traces[grain_number], np.ndarray)
    assert len(dnatrace.ordered_traces[grain_number]) == length
    np.testing.assert_array_equal(dnatrace.ordered_traces[grain_number][0,], start)
    np.testing.assert_array_almost_equal(dnatrace.ordered_traces[grain_number][-1,], end)


@pytest.mark.parametrize(
    "dnatrace, grain_number, length, start, end",
    [
        (lazy_fixture("dnatrace_linear"), 3, 118, np.asarray([13, 34]), np.asarray([73, 49])),
        (lazy_fixture("dnatrace_circular"), 9, 151, np.asarray([48, 48]), np.asarray([48, 48])),
    ],
)
def test_get_fitted_traces(dnatrace: dnaTrace, grain_number: int, length: int, start: np.array, end: np.array) -> None:
    """Test of the method."""
    dnatrace.get_numpy_arrays()
    dnatrace.gaussian_filter()
    dnatrace.get_disordered_trace()
    dnatrace.purge_obvious_crap()
    dnatrace.linear_or_circular(dnatrace.disordered_trace)
    dnatrace.get_ordered_traces()
    dnatrace.linear_or_circular(dnatrace.disordered_trace)
    dnatrace.get_fitted_traces()
    assert isinstance(dnatrace.fitted_traces[grain_number], np.ndarray)
    assert len(dnatrace.fitted_traces[grain_number]) == length
    np.testing.assert_array_equal(dnatrace.fitted_traces[grain_number][0,], start)
    np.testing.assert_array_almost_equal(dnatrace.fitted_traces[grain_number][-1,], end)


@pytest.mark.parametrize(
    "dnatrace, grain_number, length, start, end",
    [
        (lazy_fixture("dnatrace_linear"), 3, 1652, np.asarray([19.64285714, 30.5]), np.asarray([19.64285714, 30.5])),
        (lazy_fixture("dnatrace_circular"), 9, 2114, np.asarray([45.5, 54.64285714]), np.asarray([45.5, 54.64285714])),
    ],
)
def test_get_splined_traces(dnatrace: dnaTrace, grain_number: int, length: int, start: np.array, end: np.array) -> None:
    """Test of the method."""
    dnatrace.get_numpy_arrays()
    dnatrace.gaussian_filter()
    dnatrace.get_disordered_trace()
    dnatrace.purge_obvious_crap()
    dnatrace.linear_or_circular(dnatrace.disordered_trace)
    dnatrace.get_ordered_traces()
    dnatrace.linear_or_circular(dnatrace.disordered_trace)
    dnatrace.get_fitted_traces()
    dnatrace.get_splined_traces()
    assert isinstance(dnatrace.splined_traces[grain_number], np.ndarray)
    assert len(dnatrace.splined_traces[grain_number]) == length
    np.testing.assert_array_almost_equal(dnatrace.splined_traces[grain_number][0,], start)
    np.testing.assert_array_almost_equal(dnatrace.splined_traces[grain_number][-1,], end)


@pytest.mark.parametrize(
    "dnatrace, grain_number, contour_length",
    [
        (lazy_fixture("dnatrace_linear"), 3, 8.654848972955206e-08),
        (lazy_fixture("dnatrace_circular"), 9, 7.279700456226708e-08),
    ],
)
def test_measure_contour_length(dnatrace: dnaTrace, grain_number: int, contour_length: float) -> None:
    """Test of the method."""
    dnatrace.get_numpy_arrays()
    dnatrace.gaussian_filter()
    dnatrace.get_disordered_trace()
    dnatrace.purge_obvious_crap()
    dnatrace.linear_or_circular(dnatrace.disordered_trace)
    dnatrace.get_ordered_traces()
    dnatrace.linear_or_circular(dnatrace.disordered_trace)
    dnatrace.get_fitted_traces()
    dnatrace.get_splined_traces()
    dnatrace.measure_contour_length()
    assert dnatrace.contour_lengths[grain_number] == pytest.approx(contour_length)


# Currently need an actual linear grain to test this.
@pytest.mark.parametrize(
    "dnatrace, grain_number, end_to_end_distance",
    [
        (lazy_fixture("dnatrace_linear"), 3, 0),
        (lazy_fixture("dnatrace_circular"), 9, 0),
    ],
)
def test_measure_end_to_end_distance(dnatrace: dnaTrace, grain_number: int, end_to_end_distance: float) -> None:
    """Test of the method."""
    dnatrace.get_numpy_arrays()
    dnatrace.gaussian_filter()
    dnatrace.get_disordered_trace()
    dnatrace.purge_obvious_crap()
    dnatrace.linear_or_circular(dnatrace.disordered_trace)
    dnatrace.get_ordered_traces()
    dnatrace.linear_or_circular(dnatrace.disordered_trace)
    dnatrace.get_fitted_traces()
    dnatrace.get_splined_traces()
    dnatrace.measure_end_to_end_distance()
    assert dnatrace.end_to_end_distance[grain_number] == pytest.approx(end_to_end_distance)


@pytest.mark.parametrize(
    "cropped_image, cropped_mask, filename, skeletonisation_method, end_to_end_distance, circular, contour_length",
    [
        (
            LINEAR_IMAGE,
            LINEAR_MASK,
            "linear_test_topostats",
            "topostats",
            2.7708236360734238e-08,
            False,
            1.7005914942906357e-07,
        ),
        (
            CIRCULAR_IMAGE,
            CIRCULAR_MASK,
            "circular_test_topostats",
            "topostats",
            0,
            True,
            7.279700456226708e-08,
        ),
        (
            LINEAR_IMAGE,
            LINEAR_MASK,
            "linear_test_zhang",
            "zhang",
            2.474456680567355e-08,
            False,
            1.9290667640444226e-07,
        ),
        (
            CIRCULAR_IMAGE,
            CIRCULAR_MASK,
            "circular_test_zhang",
            "zhang",
            1.438558245500607e-08,
            False,
            2.1058770448077433e-07,
        ),
        (
            LINEAR_IMAGE,
            LINEAR_MASK,
            "linear_test_lee",
            "lee",
            2.9213079505690535e-08,
            False,
            1.6412088863885653e-07,
        ),
        (
            CIRCULAR_IMAGE,
            CIRCULAR_MASK,
            "circular_test_lee",
            "lee",
            1.4223510240918788e-08,
            False,
            2.4838250943353e-07,
        ),
        (
            LINEAR_IMAGE,
            LINEAR_MASK,
            "linear_test_thin",
            "thin",
            4.43751038313367e-08,
            False,
            1.3152374762504836e-07,
        ),
        (
            CIRCULAR_IMAGE,
            CIRCULAR_MASK,
            "circular_test_thin",
            "thin",
            4.229426789739925e-08,
            False,
            1.1532177230480711e-07,
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
    assert trace_stats["filename"] == filename
    assert trace_stats["end_to_end_distance"] == end_to_end_distance
    assert trace_stats["circular"] == circular
    assert trace_stats["contour_length"] == contour_length


MULTIGRAIN_IMAGE = np.concatenate((np.pad(LINEAR_IMAGE, (9, 9)), CIRCULAR_IMAGE), axis=0)
MULTIGRAIN_MASK = np.concatenate((np.pad(LINEAR_MASK, (9, 9)), CIRCULAR_MASK), axis=0)
PAD_WIDTH = 20


@pytest.mark.parametrize(
    "filename, skeletonisation_method, cores, statistics",
    [
        (
            "multigrain_topostats",
            "topostats",
            1,
            pd.DataFrame(
                {
                    "molecule_number": [0, 1],
                    "filename": ["multigrain_topostats", "multigrain_topostats"],
                    "end_to_end_distance": [2.7708236360734238e-08, 0.000000e00],
                    "circular": [False, True],
                    "contour_length": [1.7005914942906357e-07, 7.279700456226708e-08],
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
                    "filename": ["multigrain_zhang", "multigrain_zhang"],
                    "end_to_end_distance": [2.474456680567355e-08, 1.438558245500607e-08],
                    "circular": [False, False],
                    "contour_length": [1.9290667640444226e-07, 2.1058770448077433e-07],
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
                    "filename": ["multigrain_lee", "multigrain_lee"],
                    "end_to_end_distance": [2.9213079505690535e-08, 1.4223510240918788e-08],
                    "circular": [False, False],
                    "contour_length": [1.6412088863885653e-07, 2.4838250943353e-07],
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
                    "filename": ["multigrain_thin", "multigrain_thin"],
                    "end_to_end_distance": [4.43751038313367e-08, 4.229426789739925e-08],
                    "circular": [False, False],
                    "contour_length": [1.3152374762504836e-07, 1.1532177230480711e-07],
                }
            ),
        ),
    ],
)
def test_trace_image(filename: str, skeletonisation_method: str, cores: int, statistics) -> None:
    """Tests the processing of an image using trace_image() function."""
    results = trace_image(
        image=MULTIGRAIN_IMAGE,
        grains_mask=MULTIGRAIN_MASK,
        filename=filename,
        pixel_to_nm_scaling=PIXEL_SIZE,
        min_skeleton_size=MIN_SKELETON_SIZE,
        skeletonisation_method=skeletonisation_method,
        pad_width=PAD_WIDTH,
        cores=cores,
    )
    statistics.set_index(["molecule_number"], inplace=True)
    pd.testing.assert_frame_equal(results, statistics)
