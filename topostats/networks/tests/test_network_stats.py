"""Tests for network_stats.py"""

from topostats.networks import network_stats

import pytest
import numpy as np


@pytest.mark.parametrize(
    "point, expected",
    [
        (np.array([0, 0]), -2.8284271247461903),
        (np.array([5, 5]), -1.4142135623730951),
        (np.array([6, 4]), 0.0),
    ],
)
def test_signed_distance_to_outline(point: np.ndarray, expected: float):
    """Test for signed_distance_to_outline()"""

    outline_mask = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    distance_result = network_stats.signed_distance_to_outline(outline_mask=outline_mask, point=point)

    assert distance_result == expected
