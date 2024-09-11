"""Test for curvature measurements."""

import numpy as np
import numpy.typing as npt
import pytest

from topostats.measure.curvature import angle_diff_signed


@pytest.mark.parametrize(
    ("v1", "v2", "expected_angle"),
    [
        pytest.param(np.array([0, 0]), np.array([0, 0]), 0, id="zero vectors"),
        pytest.param(np.array([1, 0]), np.array([1, 0]), 0, id="same vectors"),
        pytest.param(np.array([1, 0]), np.array([0, 1]), np.pi / 2, id="up & right 90 deg"),
        pytest.param(np.array([0, 1]), np.array([1, 0]), -np.pi / 2, id="right & up -90 deg"),
        pytest.param(np.array([2, 0]), np.array([0, 2]), np.pi / 2, id="up & right 90 deg, non-normalized"),
        pytest.param(np.array([-1, -1]), np.array([0, -1]), np.pi / 4, id="down & left -45 deg"),
        pytest.param(np.array([-1, -1]), np.array([1, 1]), np.pi, id="down-left & up-right 180 deg"),
    ],
)
def test_angle_diff_signed(v1: npt.NDArray[np.number], v2: npt.NDArray[np.number], expected_angle: float) -> None:
    """Test the signed angle difference calculation."""
    assert angle_diff_signed(v1, v2) == expected_angle
