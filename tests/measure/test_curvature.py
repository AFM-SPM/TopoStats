"""Test for curvature measurements."""

import numpy as np
import numpy.typing as npt
import pytest

from topostats.measure.curvature import (
    angle_diff_signed,
    calculate_distances_between_defects_circular,
    calculate_number_of_defects,
    calculate_trace_distances_to_last_points_circular,
    discrete_angle_difference_per_nm_circular,
    find_curvature_defects_simple_threshold,
)


@pytest.mark.parametrize(
    ("v1", "v2", "expected_angle"),
    [
        pytest.param(np.array([0, 0]), np.array([0, 0]), 0, id="zero vectors"),
        pytest.param(np.array([1, 0]), np.array([1, 0]), 0, id="same vectors"),
        pytest.param(np.array([1, 0]), np.array([0, 1]), np.pi / 2, id="up & right 90 deg"),
        pytest.param(np.array([0, 1]), np.array([1, 0]), -np.pi / 2, id="right & up -90 deg"),
        pytest.param(np.array([-1, 0]), np.array([0, 1]), -np.pi / 2, id="down & right 90 deg"),
        pytest.param(np.array([0, -1]), np.array([-1, 0]), -np.pi / 2, id="left & down -90 deg"),
        pytest.param(np.array([1, 0]), np.array([0, -1]), -np.pi / 2, id="up & left -90 deg"),
        pytest.param(np.array([0, 1]), np.array([-1, 0]), np.pi / 2, id="left & up 90 deg"),
        pytest.param(np.array([2, 0]), np.array([0, 2]), np.pi / 2, id="up & right 90 deg, non-normalized"),
        pytest.param(np.array([-1, -1]), np.array([0, -1]), np.pi / 4, id="down & left -45 deg"),
        pytest.param(np.array([-1, -1]), np.array([1, 1]), np.pi, id="down-left & up-right 180 deg"),
    ],
)
def test_angle_diff_signed(v1: npt.NDArray[np.number], v2: npt.NDArray[np.number], expected_angle: float) -> None:
    """Test the signed angle difference calculation."""
    assert angle_diff_signed(v1, v2) == expected_angle


@pytest.mark.parametrize(
    ("trace_nm", "expected_angle_difference_per_nm"),
    [
        pytest.param(
            np.array(
                [
                    [-1, -1],
                    [-1, 1],
                    [1, 1],
                    [1, -1],
                ]
            ),
            np.array(
                [
                    -np.pi / 4,
                    -np.pi / 4,
                    -np.pi / 4,
                    -np.pi / 4,
                ]
            ),
            id="square counter-clockwise",
        ),
        pytest.param(
            np.array(
                [
                    [-1, -1],
                    [1, -1],
                    [1, 1],
                    [-1, 1],
                ]
            ),
            np.array(
                [
                    np.pi / 4,
                    np.pi / 4,
                    np.pi / 4,
                    np.pi / 4,
                ]
            ),
            id="square clockwise",
        ),
        pytest.param(
            np.array(
                [
                    [-2, -2],
                    [2, -2],
                    [2, 2],
                    [-2, 2],
                ]
            ),
            np.array(
                [
                    np.pi / 8,
                    np.pi / 8,
                    np.pi / 8,
                    np.pi / 8,
                ]
            ),
            id="2 wide square clockwise",
        ),
    ],
)
def test_discrete_angle_difference_per_nm_circular(
    trace_nm: npt.NDArray[np.number], expected_angle_difference_per_nm: npt.NDArray[np.number]
) -> None:
    """Test the discrete angle difference per nm calculation."""
    # Calculate the angle difference per nm
    angle_difference_per_nm = discrete_angle_difference_per_nm_circular(
        trace_nm=trace_nm,
    )

    np.testing.assert_array_equal(angle_difference_per_nm, expected_angle_difference_per_nm)


@pytest.mark.parametrize(
    ("curvature_angle_per_nm", "defect_threshold", "expected_defects"),
    [
        pytest.param(
            np.array(
                [0.8, 0.9, 1.0, 1.1, 1.1, 1.0, 0.9, 0.6, 0.2, 0.1, -0.1, -0.4, -0.8, -1.0, -1.2, -1.0, -0.8, -0.5]
            ),
            1.0,
            np.array([0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]).astype(np.bool_),
        )
    ],
)
def test_find_curvature_defects_simple_threshold(
    curvature_angle_per_nm: npt.NDArray[np.number], defect_threshold: float, expected_defects: npt.NDArray[np.number]
) -> None:
    """Test the simple curvature defect detection."""
    # Find curvature defects
    defects = find_curvature_defects_simple_threshold(
        curvature_angle_per_nm=curvature_angle_per_nm,
        defect_threshold=defect_threshold,
    )

    np.testing.assert_array_equal(defects, expected_defects)


@pytest.mark.parametrize(
    ("trace_nm", "expected_trace_distances_to_last_points"),
    [
        pytest.param(
            np.array(
                [
                    [0, 0],
                    [1, 0],
                    [1, 1],
                    [0, 1],
                ]
            ),
            np.array([1, 1, 1, 1]),
            id="square",
        ),
        pytest.param(
            np.array(
                [
                    [0, 0],
                    [1, 1],
                    [1, 2],
                    [0, 2],
                    [-1, 2],
                    [-1, -1],
                ]
            ),
            np.array(
                [
                    np.sqrt(2),
                    np.sqrt(2),
                    1,
                    1,
                    1,
                    3,
                ]
            ),
            id="square with last point",
        ),
    ],
)
def test_calculate_trace_distances_to_last_points_circular(
    trace_nm: npt.NDArray[np.number], expected_trace_distances_to_last_points: npt.NDArray[np.number]
) -> None:
    """Test the calculation of distances between points in a trace."""
    # Calculate distances between points
    trace_distances_to_last_points = calculate_trace_distances_to_last_points_circular(
        trace_nm=trace_nm,
    )

    np.testing.assert_array_equal(trace_distances_to_last_points, expected_trace_distances_to_last_points)


@pytest.mark.parametrize(
    ("curvature_defects", "trace_distances_to_last_points", "expected_distances_between_defects"),
    [
        pytest.param(
            np.array(
                [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            ),
            np.array([1.1, 1.3, 1.2, 1.1, 1.3, 1.1, 1.2, 1.4, 1.5, 1.3]),
            np.array([3.6, 5.2]).astype(np.float32),
            id="gap at start and end",
        ),
        pytest.param(
            np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 1]),
            np.array([1.1, 1.3, 1.2, 1.1, 1.3, 1.1, 1.2, 1.4, 1.5, 1.3]),
            np.array([3.6, 2.4]).astype(np.float32),
            id="no gap at end, only at start",
        ),
        pytest.param(
            np.array([1, 1, 1, 1, 0, 0, 1, 1, 0, 0]),
            np.array([1.1, 1.3, 1.2, 1.1, 1.3, 1.1, 1.2, 1.4, 1.5, 1.3]),
            np.array([3.6, 2.8]).astype(np.float32),
            id="no gap at start, only at end",
        ),
        pytest.param(
            np.array([1, 1, 1, 1, 0, 0, 1, 1, 1, 1]),
            np.array([1.1, 1.3, 1.2, 1.1, 1.3, 1.1, 1.2, 1.4, 1.5, 1.3]),
            np.array([3.6]).astype(np.float32),
            id="no gap at start or at end",
        ),
        pytest.param(
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            np.array([1.1, 1.3, 1.2, 1.1, 1.3, 1.1, 1.2, 1.4, 1.5, 1.3]),
            np.array([]).astype(np.float32),
            id="no defects",
        ),
        pytest.param(
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            np.array([1.1, 1.3, 1.2, 1.1, 1.3, 1.1, 1.2, 1.4, 1.5, 1.3]),
            np.array([]).astype(np.float32),
            id="all defects, no gaps",
        ),
    ],
)
def test_calculate_distances_between_defects_circular(
    curvature_defects: npt.NDArray[np.bool_],
    trace_distances_to_last_points: npt.NDArray[np.number],
    expected_distances_between_defects: npt.NDArray[np.float32],
) -> None:
    """Test the calculation of distances between defects."""
    # Calculate distances between defects
    distances_between_defects = calculate_distances_between_defects_circular(
        curvature_defects=curvature_defects,
        trace_distances_to_last_points=trace_distances_to_last_points,
    )

    np.testing.assert_array_equal(distances_between_defects, expected_distances_between_defects)


@pytest.mark.parametrize(
    ("curvature_defects", "circular", "expected_number_of_defects"),
    [
        pytest.param(np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0]), False, 2, id="linear, gap at start and end"),
        pytest.param(np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 1]), False, 2, id="linear, gap at start"),
        pytest.param(np.array([1, 1, 1, 1, 0, 0, 1, 1, 0, 0]), False, 2, id="linear, gap at end"),
        pytest.param(np.array([1, 1, 1, 1, 0, 0, 1, 1, 1, 1]), False, 2, id="linear, no gaps at ends"),
        pytest.param(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), False, 0, id="linear, no defects"),
        pytest.param(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), False, 1, id="linear, all defects"),
        pytest.param(np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0]), True, 2, id="circular, gap at start and end"),
        pytest.param(np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 1]), True, 2, id="circular, gap at start"),
        pytest.param(np.array([1, 1, 1, 1, 0, 0, 1, 1, 0, 0]), True, 2, id="circular, gap at end"),
        pytest.param(np.array([1, 1, 1, 1, 0, 0, 1, 1, 1, 1]), True, 1, id="circular, no gaps at ends"),
        pytest.param(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), True, 0, id="circular, no defects"),
        pytest.param(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), True, 1, id="circular, all defects"),
    ],
)
def test_calculate_number_of_defects(
    curvature_defects: npt.NDArray[np.bool_], circular: bool, expected_number_of_defects: int
) -> None:
    """Test the calculation of the number of defects."""
    # Calculate the number of defects
    number_of_defects = calculate_number_of_defects(curvature_defects=curvature_defects, circular=circular)

    assert number_of_defects == expected_number_of_defects
