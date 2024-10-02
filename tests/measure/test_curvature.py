"""Test for curvature measurements."""

import numpy as np
import numpy.typing as npt
import pytest

from topostats.measure.curvature import (
    angle_diff_signed,
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
        pytest.param(np.array([-1, 0]), np.array([0, 1]), np.pi / 2, id="down & right 90 deg"),
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
