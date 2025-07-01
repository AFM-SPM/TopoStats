"""Test damage functions"""

import pytest

import numpy as np
import numpy.typing as npt
from topostats.damage.damage import (
    calculate_distance_of_region,
    calculate_indirect_defect_gaps,
    OrderedDefectGapList,
    Defect,
    DefectGap,
)


@pytest.mark.parametrize(
    ("start_index", "end_index", "distance_to_previous_points_nm", "circular", "expected_distance"),
    [
        pytest.param(
            5,
            5,
            np.array([0.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]),
            False,
            1.55,
            id="unit region in middle, linear",
        ),
        pytest.param(
            5,
            6,
            np.array([0.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]),
            False,
            3.2,
            id="region in middle, linear",
        ),
        pytest.param(
            5,
            5,
            np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]),
            True,
            1.55,
            id="unit region in middle, circular",
        ),
        pytest.param(
            5,
            6,
            np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]),
            True,
            3.2,
            id="region in middle, circular",
        ),
        pytest.param(
            0,
            1,
            np.array([0.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]),
            False,
            1.7,
            id="region at start, linear",
        ),
        pytest.param(
            8,
            9,
            np.array([0.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]),
            False,
            2.8,
            id="region at end, linear",
        ),
        pytest.param(
            0,
            1,
            np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]),
            True,
            2.2,
            id="region at start, circular",
        ),
        pytest.param(
            8,
            9,
            np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]),
            True,
            3.3,
            id="region at end, circular",
        ),
        pytest.param(
            8,
            1,
            np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]),
            True,
            5.5,
            id="region spanning end, circular",
        ),
    ],
)
def test_calculate_distance_of_region(
    start_index: int,
    end_index: int,
    distance_to_previous_points_nm: npt.NDArray[np.float64],
    circular: bool,
    expected_distance: float,
) -> None:
    """Test the calculate_distance_of_region function."""
    distance = calculate_distance_of_region(
        start_index=start_index,
        end_index=end_index,
        distance_to_previous_points_nm=distance_to_previous_points_nm,
        circular=circular,
    )
    assert distance == pytest.approx(expected_distance)


def test_calculate_distance_of_region_linear_array_region_spanning_end() -> None:
    """Test that an exception is raised for the invalid case of linear array with end-spanning region."""

    with pytest.raises(ValueError):
        calculate_distance_of_region(
            start_index=8,
            end_index=1,
            distance_to_previous_points_nm=np.array([0.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]),
            circular=False,
        )


@pytest.mark.parametrize(
    ("ordered_defect_gap_list", "circular", "expected_indirect_gaps"),
    [
        pytest.param(
            OrderedDefectGapList(
                defect_gap_list=[],
            ),
            True,
            [],
            id="No defects / gaps, circular",
        ),
        pytest.param(
            OrderedDefectGapList(
                defect_gap_list=[
                    DefectGap(0, 4, 4),
                    Defect(5, 6, 1),
                    DefectGap(7, 8, 2),
                ]
            ),
            False,
            [4.0, 2.0],
            id="one defect in middle, linear",
        ),
        pytest.param(
            OrderedDefectGapList(
                defect_gap_list=[
                    Defect(5, 6, 1),
                    DefectGap(7, 8, 6),
                ],
            ),
            True,
            [6.0],
            id="one defect in middle, circular",
        ),
    ],
)
def test_calculate_indirect_defect_gaps(
    ordered_defect_gap_list: OrderedDefectGapList, circular: bool, expected_indirect_gaps: list[float]
) -> None:
    """Test the calculate_indirect_defect_gaps function."""
    indirect_gaps = calculate_indirect_defect_gaps(
        ordered_defect_gap_list=ordered_defect_gap_list,
        circular=circular,
    )
    assert indirect_gaps == expected_indirect_gaps
