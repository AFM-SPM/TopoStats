"""Tests for rougness.py functions."""
import numpy as np
import pytest
from topostats.roughness import roughness_rms


@pytest.mark.parametrize(
    "test_image, expected",
    [
        (
            np.array(
                [
                    [0, 0],
                    [0, 0],
                ]
            ),
            0.0,
        ),
        (
            np.array(
                [
                    [1, 1],
                    [1, 1],
                ]
            ),
            1.0,
        ),
        (
            np.array(
                [
                    [-1, -1],
                    [-1, -1],
                ]
            ),
            1.0,
        ),
        (
            np.array(
                [
                    [1, 2],
                    [3, 4],
                ]
            ),
            2.7386127875258306,
        ),
    ],
)
def test_roughness_rms(test_image, expected):
    """Test the rms roughness calculation."""
    roughness = roughness_rms(test_image)
    np.testing.assert_almost_equal(roughness, expected)
