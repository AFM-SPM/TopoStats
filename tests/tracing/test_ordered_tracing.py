"""Tests for the ordered_tracing module."""

import numpy as np
import numpy.typing as npt
import pytest

from topostats.tracing import ordered_tracing


@pytest.mark.parametrize(
    ("coordinates", "shape", "target"),
    [
        pytest.param([[1, 1]], (3, 3), np.asarray([[0, 0, 0], [0, 1, 0], [0, 0, 0]]), id="single point in 3x3 array"),
        pytest.param(
            [[0, 0], [1, 1], [2, 2]],
            (3, 3),
            np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            id="diagonal line across 3x3 array (top left to bottom right)",
        ),
    ],
)
def test_ordered_trace_mask(coordinates: list, shape: tuple, target: npt.NDArray) -> None:
    """Test of ordered_trace_mask()."""
    np.testing.assert_array_equal(ordered_tracing.ordered_trace_mask(coordinates, shape), target)


@pytest.mark.parametrize(
    ("coordinates", "shape", "target"),
    [
        pytest.param(
            [[1, 1]],
            (3, 3),
            np.asarray([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
            id="list of coordinates with single point in 3x3 array",
        ),
        pytest.param(
            [[0, 0], [1, 1], [2, 2]],
            (3, 3),
            np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            id="list of coordinates with diagonal line across 3x3 array (top left to bottom right)",
        ),
        pytest.param(
            np.asarray([[1, 1]]),
            (3, 3),
            np.asarray([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
            id="Numpy array of coordinates with single point in 3x3 array",
        ),
        pytest.param(
            np.asarray([[0, 0], [1, 1], [2, 2]]),
            (3, 3),
            np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            id="Numpy array of coordinates with diagonal line across 3x3 array (top left to bottom right)",
        ),
    ],
)
def test_trace_mask(coordinates: list, shape: tuple, target: npt.NDArray) -> None:
    """Test of ordered_trace_mask()."""
    np.testing.assert_array_equal(ordered_tracing.trace_mask(coordinates, shape), target)


@pytest.mark.parametrize(
    ("coordinates", "shape"),
    [
        pytest.param(
            ((1, 1)),
            (3, 3),
            id="tuple of coordinates with single point in 3x3 array",
        ),
        pytest.param(
            ([0, 0], [1, 1], [2, 2]),
            (3, 3),
            id="tuple of coordinates with diagonal line across 3x3 array (top left to bottom right)",
        ),
    ],
)
def test_trace_mask_type_error(coordinates: tuple, shape: tuple) -> None:
    """Assert type error is raised by trace_mask() when passed neither list or Numpy array."""
    with pytest.raises(TypeError):
        ordered_tracing.trace_mask(coordinates, shape)
