"""Tests for the geometry module."""

import numpy as np
import pytest
from numpy.typing import NDArray

from topostats.measure.geometry import bounding_box_cartesian_points


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
def test_bounding_box_cartesian_points(
    points: NDArray[np.number], expected_bbox: tuple[np.float64, np.float64, np.float64, np.float64]
):
    """Test the bounding_box_cartesian_points function."""
    assert bounding_box_cartesian_points(points) == expected_bbox
