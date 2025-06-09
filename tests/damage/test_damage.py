"""Test damage functions"""

import pytest

import numpy as np
import numpy.typing as npt
from topostats.damage.damage import calculate_defects_and_gap_lengths


@pytest.mark.parametrize(
    ("points_distance_to_previous_nm", "defects_bool", "expected_defect_lengths", "expected_gap_lengths"),
    [
        pytest.param(
            np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            np.array([]),
            np.array([10]),
            id="no defects, all gap",
        )
    ],
)
def test_calculate_defects_and_gap_lengths(
    defects_bool: npt.NDArray[np.bool_],
    points_distance_to_previous_nm: npt.NDArray[np.float64],
    expected_defect_lengths: npt.NDArray[np.float64],
    expected_gap_lengths: npt.NDArray[np.float64],
) -> None:
    """Test the function"""
    defect_lengths, gap_lengths = calculate_defects_and_gap_lengths(
        points_distance_to_previous_nm=points_distance_to_previous_nm,
        defects_bool=defects_bool,
    )
    assert np.array_equal(defect_lengths, expected_defect_lengths)
    assert np.array_equal(gap_lengths, expected_gap_lengths)
