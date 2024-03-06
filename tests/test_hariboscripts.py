"""Test the hariboscripts module."""

import numpy as np
import pytest

from topostats.hariboscripts import flip_if_anticlockwise


@pytest.mark.parametrize(
    ("trace", "expected"),
    [
        pytest.param(
            np.array([[5, 5], [6, 8], [8, 10], [12, 6], [10, 4], [7, 4]]),
            np.array([[7, 4], [10, 4], [12, 6], [8, 10], [6, 8], [5, 5]]),
            id="clockwise convex",
        ),
        pytest.param(
            np.array([[7, 4], [10, 4], [12, 6], [8, 10], [6, 8], [5, 5]]),
            np.array([[5, 5], [6, 8], [8, 10], [12, 6], [10, 4], [7, 4]]),
            id="anticlockwise convex",
        ),
        pytest.param(
            np.array([[8, 6], [6, 8], [8, 10], [12, 6], [10, 4], [7, 4]]),
            np.array([[7, 4], [10, 4], [12, 6], [8, 10], [6, 8], [8, 6]]),
            id="clockwise concave",
        ),
        pytest.param(
            np.array([[7, 4], [10, 4], [12, 6], [8, 10], [6, 8], [8, 6, 5]]),
            np.array([[8, 6, 5], [6, 8], [8, 10], [12, 6], [10, 4], [7, 4]]),
            id="anticlockwise concave",
        ),
    ],
)
def test_flip_if_anticlockwise(trace: np.ndarray, expected: np.ndarray):
    """Test the flip_if_anticlockwise function."""

    assert np.array_equal(flip_if_anticlockwise(trace), expected)
