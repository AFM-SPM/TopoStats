"""Tests of the resampling module."""

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import pytest
from topostats.tracing.resampling import interpolate_between_two_points_distance, resample_points_regular_interval


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
def test_interpolate_between_two_points_distance(
    point1: npt.NDArray[np.float32],
    point2: npt.NDArray[np.float32],
    distance: np.float32,
    expected_point: npt.NDArray[np.float32],
) -> None:
    """Test the interpolation between two points."""
    interpolated_point = interpolate_between_two_points_distance(point1, point2, distance)
    assert isinstance(interpolated_point, np.ndarray)
    assert interpolated_point.shape == (2,)
    np.testing.assert_allclose(interpolated_point, expected_point, atol=1e-6)


def test_resample_points_regular_interval() -> None:
    """Test the resampling of points at regular intervals."""

    points = np.array(
        [
            [11.5, 10.5],
            [11.8, 10.8],
            [12.0, 11.0],
            [12.5, 11.5],
            [13.5, 12.8],
            [14.0, 13.0],
            [14.2, 13.1],
            [14.7, 13.9],
            [15.0, 14.0],
            [15.8, 14.2],
            [16.2, 14.5],
            [16.5, 13.0],
            [17.0, 12.5],
            [16.0, 12.3],
            [15.0, 11.7],
            [14.0, 11.5],
            [13.0, 10.5],
        ]
    )

    # distance between each point
    distances = np.linalg.norm(points[1:] - points[:-1], axis=1)
    print(f"max distance: {np.max(distances)}")

    plt.plot(points[:, 0], points[:, 1], "ro-", label="Original Points")

    interval = 1.0

    resampled_points = resample_points_regular_interval(points, interval, circular=True)

    plt.plot(resampled_points[:, 0], resampled_points[:, 1], "bo-", label="Resampled Points")
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Resampling Points at Regular Intervals")
    plt.show()

    resampled_distances = np.linalg.norm(resampled_points[1:] - resampled_points[:-1], axis=1)
    print("resampled distances:")
    print(resampled_distances)
    print(f"max resampled distance: {np.max(resampled_distances)}")

    raise ValueError