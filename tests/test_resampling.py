"""Tests of the resampling module."""

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import pytest
from topostats.tracing.resampling import interpolate_between_two_points, resample_points_regular_interval


@pytest.mark.parametrize(
    ("point1", "point2", "distance", "expected_point"),
    [
        pytest.param(
            np.array([0.0, 0.0]),
            np.array([10.0, 0.0]),
            5.0,
            np.array([5.0, 0.0]),
            id="horizontal_interpolation",
        ),
        pytest.param(
            np.array([0.0, 0.0]),
            np.array([0.0, 10.0]),
            5.0,
            np.array([0.0, 5.0]),
            id="vertical_interpolation",
        ),
        pytest.param(
            np.array([0.0, 0.0]),
            np.array([10.0, 10.0]),
            5.0,
            np.array([3.535534, 3.535534]),
            id="diagonal_interpolation",
        ),
        pytest.param(
            np.array([-5.0, 0.0]),
            np.array([5.0, 0.0]),
            5.0,
            np.array([0.0, 0.0]),
            id="negative_interpolation",
        ),
    ],
)
def test_interpolate_between_two_points(
    point1: npt.NDArray[np.float32],
    point2: npt.NDArray[np.float32],
    distance: np.float32,
    expected_point: npt.NDArray[np.float32],
) -> None:
    """Test the interpolation between two points."""
    interpolated_point = interpolate_between_two_points(point1, point2, distance)
    assert isinstance(interpolated_point, np.ndarray)
    assert interpolated_point.shape == (2,)
    np.testing.assert_allclose(interpolated_point, expected_point, atol=1e-6)
