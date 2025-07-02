"""Test damage functions"""

import pytest

import numpy as np
import numpy.typing as npt
from topostats.damage.damage import (
    get_defects_and_gaps_linear,
    get_defects_and_gaps_circular,
    calculate_distance_of_region,
    calculate_indirect_defect_gaps,
    get_defects_and_gaps_from_bool_array,
    OrderedDefectGapList,
    Defect,
    DefectGap,
)


@pytest.mark.parametrize(
    ("defects_bool", "expected_defect_and_gap_list"),
    [
        pytest.param(
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.bool_),
            ([], [(0, 9)]),
            id="no defects, all gap",
        ),
        pytest.param(
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.bool_),
            ([(0, 9)], []),
            id="all defects, no gap",
        ),
        pytest.param(
            np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.bool_),
            ([(4, 4)], [(0, 3), (5, 9)]),
            id="one unit defect in middle",
        ),
        pytest.param(
            np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0], dtype=np.bool_),
            ([(2, 3), (6, 7)], [(0, 1), (4, 5), (8, 9)]),
            id="two defects in middle",
        ),
        pytest.param(
            np.array([1, 1, 1, 0, 0, 0, 0, 1, 1, 1], dtype=np.bool_),
            ([(0, 2), (7, 9)], [(3, 6)]),
            id="two defects, at start and end",
        ),
    ],
)
def test_get_defects_and_gaps_linear(
    defects_bool: npt.NDArray[np.bool_],
    expected_defect_and_gap_list: tuple[list[tuple[int, int]], list[tuple[int, int]]],
) -> None:
    """Test the get_defects_and_gaps_linear function."""
    defect_and_gap_list = get_defects_and_gaps_linear(defects_bool=defects_bool)
    assert defect_and_gap_list == expected_defect_and_gap_list


@pytest.mark.parametrize(
    ("defects_bool", "expected_defect_and_gap_list"),
    [
        pytest.param(
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.bool_),
            ([], [(0, 9)]),
            id="no defects, all gap",
        ),
        pytest.param(
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.bool_),
            ([(0, 9)], []),
            id="all defects, no gap",
        ),
        pytest.param(
            np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.bool_),
            ([(4, 4)], [(5, 3)]),
            id="one unit defect in middle",
        ),
        pytest.param(
            np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0], dtype=np.bool_),
            ([(2, 3), (6, 7)], [(8, 1), (4, 5)]),
            id="two defects in middle",
        ),
        pytest.param(
            np.array([1, 1, 1, 0, 0, 0, 0, 1, 1, 1], dtype=np.bool_),
            ([(7, 2)], [(3, 6)]),
            id="two defects, at start and end",
        ),
    ],
)
def test_get_defects_and_gaps_circular(
    defects_bool: npt.NDArray[np.bool_],
    expected_defect_and_gap_list: tuple[list[tuple[int, int]], list[tuple[int, int]]],
) -> None:
    """Test the get_defects_and_gaps_circular function."""
    defect_and_gap_list = get_defects_and_gaps_circular(defects_bool=defects_bool)
    assert defect_and_gap_list == expected_defect_and_gap_list


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
            0,
            9,
            np.array([0.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]),
            False,
            13.5,
            id="region spanning all, linear",
        ),
        # Note: we cannot represent a region spanning all in circular, since that would have the same index
        # for start and end and would be treated as a unit region.
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
    ("defects_bool", "circular", "distance_to_previous_points_nm", "expected_ordered_defect_gap_list"),
    [
        pytest.param(
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.bool_),
            False,
            np.array([0.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]),
            OrderedDefectGapList(defect_gap_list=[DefectGap(0, 9, 13.5)]),
            id="no defects, all gap, linear",
        ),
        # Note this is a weird case where there isn't a start or end to the gap but we create one at the bounds
        pytest.param(
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.bool_),
            True,
            np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]),
            OrderedDefectGapList(defect_gap_list=[DefectGap(0, 9, 14.5)]),
            id="no defects, all gap, circular",
        ),
        pytest.param(
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.bool_),
            False,
            np.array([0.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]),
            OrderedDefectGapList(defect_gap_list=[Defect(0, 9, 13.5)]),
            id="all defects, no gap, linear",
        ),
        pytest.param(
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.bool_),
            True,
            np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]),
            OrderedDefectGapList(defect_gap_list=[Defect(0, 9, 14.5)]),
            id="all defects, no gap, circular",
        ),
        pytest.param(
            np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.bool_),
            False,
            np.array([0.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]),
            OrderedDefectGapList(
                defect_gap_list=[
                    DefectGap(0, 3, 4.3),
                    Defect(4, 4, 1.45),
                    DefectGap(5, 9, 7.75),
                ]
            ),
            id="one unit defect in middle, linear",
        ),
        pytest.param(
            np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.bool_),
            True,
            np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]),
            OrderedDefectGapList(
                defect_gap_list=[
                    DefectGap(5, 3, 13.05),
                    Defect(4, 4, 1.45),
                ]
            ),
            id="one unit defect in middle circular",
        ),
        pytest.param(
            np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.bool_),
            False,
            np.array([0.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]),
            OrderedDefectGapList(
                defect_gap_list=[
                    Defect(0, 0, 0.55),
                    DefectGap(
                        1,
                        8,
                        12,
                    ),
                    Defect(9, 9, 0.95),
                ]
            ),
            id="two unit defects, at ends, linear",
        ),
        pytest.param(
            np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.bool_),
            True,
            np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]),
            OrderedDefectGapList(defect_gap_list=[DefectGap(1, 8, 12), Defect(9, 0, 2.5)]),
            id="two unit defects at ends circular",
        ),
    ],
)
def test_get_defects_and_gaps_from_bool_array(
    defects_bool: npt.NDArray[np.bool_],
    circular: bool,
    distance_to_previous_points_nm: npt.NDArray[np.float64],
    expected_ordered_defect_gap_list: OrderedDefectGapList,
) -> None:
    """Test the get_defects_and_gaps_from_bool_array function."""
    ordered_defect_gap_list = get_defects_and_gaps_from_bool_array(
        defects_bool=defects_bool,
        circular=circular,
        distance_to_previous_points_nm=distance_to_previous_points_nm,
    )
    assert ordered_defect_gap_list == expected_ordered_defect_gap_list


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
                    DefectGap(0, 9, 10),
                ],
            ),
            True,
            [],
            id="No defects, all gap, circular",
        ),
        pytest.param(
            OrderedDefectGapList(
                defect_gap_list=[
                    DefectGap(0, 9, 10),
                ],
            ),
            False,
            [],
            id="No defects, all gap, linear",
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
        pytest.param(
            OrderedDefectGapList(
                defect_gap_list=[
                    Defect(0, 0, 1),
                    DefectGap(1, 9, 8),
                ],
            ),
            False,
            [8.0],
            id="one defect at start, linear",
        ),
        pytest.param(
            OrderedDefectGapList(
                defect_gap_list=[
                    Defect(0, 0, 1),
                    DefectGap(1, 9, 8),
                ],
            ),
            True,
            [8.0],
            id="one defect at start, circular",
        ),
        pytest.param(
            OrderedDefectGapList(
                defect_gap_list=[
                    DefectGap(0, 8, 8),
                    Defect(9, 9, 1),
                ],
            ),
            False,
            [8.0],
            id="one defect at end, linear",
        ),
        pytest.param(
            OrderedDefectGapList(
                defect_gap_list=[
                    DefectGap(0, 8, 8),
                    Defect(9, 9, 1),
                ],
            ),
            True,
            [8.0],
            id="one defect at end, circular",
        ),
        pytest.param(
            OrderedDefectGapList(
                defect_gap_list=[
                    DefectGap(0, 1, 1),
                    Defect(2, 3, 1),
                    DefectGap(4, 5, 1),
                    Defect(6, 7, 2),
                    DefectGap(8, 9, 1),
                ]
            ),
            False,
            [
                1.0,
                1.0,
                1.0,
            ],
            id="two defects in middle, linear",
        ),
        pytest.param(
            OrderedDefectGapList(
                defect_gap_list=[Defect(2, 3, 1), DefectGap(4, 5, 1), Defect(6, 7, 1), DefectGap(8, 1, 2)]
            ),
            True,
            [4.0, 1.0, 2.0, 4.0],
            id="two defects in middle, circular",
        ),
        pytest.param(
            OrderedDefectGapList(
                defect_gap_list=[
                    DefectGap(0, 1, 1),
                    Defect(2, 3, 1),
                    DefectGap(4, 5, 1),
                    Defect(6, 7, 1),
                    DefectGap(8, 9, 1),
                    Defect(10, 11, 1),
                    DefectGap(12, 13, 1),
                ]
            ),
            False,
            [
                1.0,  # start gap (linear)
                1.0,
                3.0,
                1.0,
                1.0,  # end gap (linear)
            ],
            id="three defects, in the middle, linear",
        ),
        pytest.param(
            OrderedDefectGapList(
                defect_gap_list=[
                    DefectGap(12, 1, 1),
                    Defect(2, 3, 1),
                    DefectGap(4, 5, 1),
                    Defect(6, 7, 1),
                    DefectGap(8, 9, 1),
                    Defect(10, 11, 1),
                ]
            ),
            True,
            [
                5.0,
                1.0,
                3.0,
                3.0,
                5.0,
                1.0,
                1.0,
                3.0,
                5.0,
            ],
            id="three defects, in the middle, circular",
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
