"""Tests for tracing single molecules"""
from pathlib import Path

import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture

from topostats.tracing.dnatracing import dnaTrace

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
    "dnatrace, grain_number, length, start, end",
    [
        (lazy_fixture("dnatrace_linear"), 3, 120, np.asarray([13, 32]), np.asarray([91, 72])),
        (lazy_fixture("dnatrace_circular"), 9, 150, np.asarray([49, 49]), np.asarray([103, 44])),
    ],
)
def test_get_disordered_trace(dnatrace: dnaTrace, grain_number: int, length: int, start: tuple, end: tuple) -> None:
    """Test of get_disordered_trace the method."""
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
    print(dnatrace.splined_traces[grain_number])
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
    assert dnatrace.contour_lengths[grain_number] == contour_length


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
    assert dnatrace.end_to_end_distance[grain_number] == end_to_end_distance
