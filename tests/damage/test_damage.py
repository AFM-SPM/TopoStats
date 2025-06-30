"""Test damage functions"""

import pytest

import numpy as np
import numpy.typing as npt
from topostats.damage.damage import (
    calculate_defects_and_gap_lengths,
    get_defect_start_end_indexes,
    calculate_indirect_defect_gap_lengths,
    get_defects_linear,
    get_defects_circular,
    calculate_distance_of_region,
)


@pytest.mark.parametrize(
    ("points_distance_to_previous_nm", "defects_bool", "expected_defect_lengths", "expected_gap_lengths"),
    [
        pytest.param(
            np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            np.array([]),
            np.array([10]),
            id="no defects, all gap",
        ),
        pytest.param(
            np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            np.array([10]),
            np.array([]),
            id="no gaps, all defect",
        ),
        pytest.param(
            np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0]),
            np.array([2]),
            np.array([8]),
            id="one defect in the middle",
        ),
        pytest.param(
            np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0]),
            np.array([4]),
            np.array([6]),
            id="one defect at the start",
        ),
        pytest.param(
            np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1]),
            np.array([4]),
            np.array([6]),
            id="one defect at the end",
        ),
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
    np.testing.assert_array_equal(defect_lengths, expected_defect_lengths)
    np.testing.assert_array_equal(gap_lengths, expected_gap_lengths)


@pytest.mark.parametrize(
    ("defects_bool", "expected_start_indexes", "expected_end_indexes"),
    [
        pytest.param(
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            np.array([]),
            np.array([]),
            id="no defects",
        ),
        pytest.param(
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            np.array([]),
            np.array([]),
            id="all defects",
        ),
        pytest.param(
            np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
            np.array([4]),
            np.array([4]),
            id="one short defect in the middle",
        ),
        pytest.param(
            np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0]),
            np.array([0]),
            np.array([3]),
            id="one defect at the start",
        ),
        pytest.param(
            np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1]),
            np.array([6]),
            np.array([9]),
            id="one defect at the end",
        ),
        pytest.param(
            np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
            np.array([1, 3, 5, 7, 9]),
            np.array([1, 3, 5, 7, 9]),
            id="multiple short defects",
        ),
    ],
)
def test_get_defect_start_end_indexes(
    defects_bool: npt.NDArray[np.bool_],
    expected_start_indexes: npt.NDArray[np.int_],
    expected_end_indexes: npt.NDArray[np.int_],
) -> None:
    """Test the get_defect_start_end_indexes function."""
    start_indexes, end_indexes = get_defect_start_end_indexes(defects_bool=defects_bool)
    np.testing.assert_array_equal(start_indexes, expected_start_indexes)
    np.testing.assert_array_equal(end_indexes, expected_end_indexes)


@pytest.mark.parametrize(
    ("defect_start_indexes", "defect_end_indexes", "cumulative_distance_nm", "expected_indirect_defect_gap_lengths"),
    [
        pytest.param(
            np.array([]),
            np.array([]),
            np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
            np.array([]),
            id="no defects",
        ),
        pytest.param(
            np.array([4]),
            np.array([4]),
            np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
            np.array([9.0]),
            id="one unit defect in the middle",
        ),
        pytest.param(
            np.array([2, 6]),
            np.array([3, 7]),
            np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
            np.array([2.0, 4.0]),
            id="two defects in the middle",
        ),
    ],
)
def test_calculate_indirect_defect_gap_lengths(
    defect_start_indexes: npt.NDArray[np.int_],
    defect_end_indexes: npt.NDArray[np.int_],
    cumulative_distance_nm: npt.NDArray[np.float64],
    expected_indirect_defect_gap_lengths: npt.NDArray[np.float64],
) -> None:
    """Test the calculate_indirect_defect_gap_lengths function."""
    gap_lengths = calculate_indirect_defect_gap_lengths(
        defect_start_end_indexes=(defect_start_indexes, defect_end_indexes),
        cumulative_distance_nm=cumulative_distance_nm,
    )
    np.testing.assert_array_equal(gap_lengths, expected_indirect_defect_gap_lengths)


@pytest.mark.parametrize(
    ("defects_bool", "expected_defects", "expected_gaps"),
    [
        pytest.param(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), [], [(0, 9)], id="all gap"),
        pytest.param(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), [(0, 9)], [], id="all defect"),
        pytest.param(np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]), [(3, 3)], [(0, 2), (4, 9)], id="unit defect in middle"),
        pytest.param(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]), [(0, 0)], [(1, 9)], id="unit defect at start"),
        pytest.param(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]), [(9, 9)], [(0, 8)], id="unit defect at end"),
        pytest.param(
            np.array([1, 1, 1, 1, 1, 0, 1, 1, 1, 1]),
            [(0, 4), (6, 9)],
            [(5, 5)],
            id="unit gap in middle",
        ),
        pytest.param(
            np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
            [(1, 1), (3, 3), (5, 5), (7, 7), (9, 9)],
            [(0, 0), (2, 2), (4, 4), (6, 6), (8, 8)],
            id="alternating unit defects and gaps",
        ),
    ],
)
def test_get_defects_linear(
    defects_bool: npt.NDArray[np.bool_],
    expected_defects: list[tuple[int, int]],
    expected_gaps: list[tuple[int, int]],
) -> None:
    """Test the get_defects_linear function."""
    defects, gaps = get_defects_linear(defects_bool=defects_bool)
    assert defects == expected_defects
    assert gaps == expected_gaps


@pytest.mark.parametrize(
    ("defects_bool", "expected_defects", "expected_gaps"),
    [
        pytest.param(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), [], [(0, 9)], id="all gap"),
        pytest.param(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), [(0, 9)], [], id="all defect"),
        pytest.param(np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]), [(3, 3)], [(4, 2)], id="unit defect in middle"),
        pytest.param(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]), [(0, 0)], [(1, 9)], id="unit defect at start"),
        pytest.param(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]), [(9, 9)], [(0, 8)], id="unit defect at end"),
        pytest.param(
            np.array([1, 1, 1, 1, 1, 0, 1, 1, 1, 1]),
            [(6, 4)],
            [(5, 5)],
            id="unit gap in middle",
        ),
        pytest.param(
            np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
            [(1, 1), (3, 3), (5, 5), (7, 7), (9, 9)],
            [(0, 0), (2, 2), (4, 4), (6, 6), (8, 8)],
            id="alternating unit defects and gaps",
        ),
    ],
)
def test_get_defects_circular(
    defects_bool: npt.NDArray[np.bool_],
    expected_defects: list[tuple[int, int]],
    expected_gaps: list[tuple[int, int]],
) -> None:
    """Test the get_defects_circular function."""
    defects, gaps = get_defects_circular(defects_bool=defects_bool)
    assert defects == expected_defects
    assert gaps == expected_gaps


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
