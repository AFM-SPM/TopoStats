"""Tests for the geometry module."""

import numpy as np
import pytest
from numpy.typing import NDArray

from topostats.measure.geometry import (
    bounding_box_cartesian_points_float,
    bounding_box_cartesian_points_integer,
    do_points_in_arrays_touch,
)


def test_bounding_box_cartesian_points_float_raises_value_error():
    """Test the bounding_box_cartesian_points function raises a ValueError."""
    with pytest.raises(ValueError, match="Input array must be Nx2."):
        bounding_box_cartesian_points_float(np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]))


def test_bounding_box_cartesian_points_integer_raises_value_error():
    """Test the bounding_box_cartesian_points function raises a ValueError."""
    with pytest.raises(ValueError, match="Input array must be Nx2."):
        bounding_box_cartesian_points_float(np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]))


@pytest.mark.parametrize(
    ("points", "expected_bbox"),
    [
        pytest.param(np.array([[0, 0], [1, 1], [2, 2]]), (0, 0, 2, 2), id="diagonal line"),
        pytest.param(np.array([[0, 0], [1, 1], [2, 0]]), (0, 0, 2, 1), id="triangle"),
        pytest.param(np.array([[-5, -5], [-10, -10], [-3, -3]]), (-10, -10, -3, -3), id="negative values"),
        pytest.param(np.array([[-1, -1], [1, 1], [2, 0], [-2, 0]]), (-2, -1, 2, 1), id="negative and positive values"),
        pytest.param(np.array([[0.1, 0.1], [1.1, 1.1], [2.1, 2.1]]), (0.1, 0.1, 2.1, 2.1), id="diagonal line, floats"),
    ],
)
def test_bounding_box_cartesian_points_float(
    points: NDArray[np.number], expected_bbox: tuple[np.float64, np.float64, np.float64, np.float64]
):
    """Test the bounding_box_cartesian_points function."""
    assert bounding_box_cartesian_points_float(points) == expected_bbox


@pytest.mark.parametrize(
    ("points", "expected_bbox"),
    [
        pytest.param(np.array([[0, 0], [1, 1], [2, 2]]), (0, 0, 2, 2), id="diagonal line"),
        pytest.param(np.array([[0, 0], [1, 1], [2, 0]]), (0, 0, 2, 1), id="triangle"),
        pytest.param(np.array([[-5, -5], [-10, -10], [-3, -3]]), (-10, -10, -3, -3), id="negative values"),
        pytest.param(np.array([[-1, -1], [1, 1], [2, 0], [-2, 0]]), (-2, -1, 2, 1), id="negative and positive values"),
        pytest.param(np.array([[0.1, 0.1], [1.1, 1.1], [2.1, 2.1]]), (0, 0, 2, 2), id="diagonal line, floats"),
    ],
)
def test_bounding_box_cartesian_points_integer(
    points: NDArray[np.number], expected_bbox: tuple[np.float64, np.float64, np.float64, np.float64]
):
    """Test the bounding_box_cartesian_points function."""
    assert bounding_box_cartesian_points_integer(points) == expected_bbox


def test_do_points_in_arrays_touch_raises_value_error():
    """Test the do_points_in_arrays_touch function raises a ValueError."""
    with pytest.raises(ValueError, match="Input arrays must be Nx2 and Mx2."):
        do_points_in_arrays_touch(np.array([[0, 0], [1, 1], [2, 2]]), np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]))


@pytest.mark.parametrize(
    ("array_1", "array_2", "expected_result_touching", "expected_point_1_touching", "expected_point_2_touching"),
    [
        pytest.param(
            np.array([[0, 0], [1, 1], [2, 2]]),
            np.array([[4, 4], [5, 5], [6, 6]]),
            False,
            None,
            None,
            id="no touching points",
        ),
        pytest.param(
            np.array([[0, 0], [1, 1], [2, 2]]),
            np.array([[2, 3], [3, 4], [5, 5]]),
            True,
            np.array([2, 2]),
            np.array([2, 3]),
            id="touching points, non_diag",
        ),
        pytest.param(
            np.array([[0, 0], [1, 1], [2, 2]]),
            np.array([[3, 3], [4, 4], [5, 5]]),
            True,
            np.array([2, 2]),
            np.array([3, 3]),
            id="touching points, diag",
        ),
    ],
)
def test_do_points_in_arrays_touch(
    array_1: NDArray[np.number],
    array_2: NDArray[np.number],
    expected_result_touching: bool,
    expected_point_1_touching: NDArray[np.number],
    expected_point_2_touching: NDArray[np.number],
):
    """Test the do_points_in_arrays_touch function."""
    result_touching, point_touching_1, point_touching_2 = do_points_in_arrays_touch(array_1, array_2)

    assert result_touching == expected_result_touching
    np.testing.assert_array_equal(point_touching_1, expected_point_1_touching)
    np.testing.assert_array_equal(point_touching_2, expected_point_2_touching)
