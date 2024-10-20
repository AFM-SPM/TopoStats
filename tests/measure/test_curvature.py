"""Test for curvature measurements."""

import numpy as np
import numpy.typing as npt
import pytest

from topostats.io import dict_almost_equal
from topostats.measure.curvature import (
    angle_diff_signed,
    calculate_curvature_stats_image,
    discrete_angle_difference_per_nm_circular,
    discrete_angle_difference_per_nm_linear,
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
                    0.0,
                    -np.pi / 4,
                    -np.pi / 4,
                    0.0,
                ]
            ),
            id="square counter-clockwise",
        ),
    ],
)
def test_discrete_angle_difference_per_nm_linear(
    trace_nm: npt.NDArray[np.number], expected_angle_difference_per_nm: npt.NDArray[np.number]
) -> None:
    """Test the discrete angle difference per nm calculation."""
    # Calculate the angle difference per nm
    angle_difference_per_nm = discrete_angle_difference_per_nm_linear(
        trace_nm=trace_nm,
    )

    np.testing.assert_array_equal(angle_difference_per_nm, expected_angle_difference_per_nm)


# @pytest.mark.parametrize(
#     ("curvature_angle_per_nm", "defect_threshold", "expected_defects"),
#     [
#         pytest.param(
#             np.array(
#                 [0.8, 0.9, 1.0, 1.1, 1.1, 1.0, 0.9, 0.6, 0.2, 0.1, -0.1, -0.4, -0.8, -1.0, -1.2, -1.0, -0.8, -0.5]
#             ),
#             1.0,
#             np.array([0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]).astype(np.bool_),
#         )
#     ],
# )
# def test_find_curvature_defects_simple_threshold(
#     curvature_angle_per_nm: npt.NDArray[np.number], defect_threshold: float, expected_defects: npt.NDArray[np.number]
# ) -> None:
#     """Test the simple curvature defect detection."""
#     # Find curvature defects
#     defects = find_curvature_defects_simple_threshold(
#         curvature_angle_per_nm=curvature_angle_per_nm,
#         defect_threshold=defect_threshold,
#     )

#     np.testing.assert_array_equal(defects, expected_defects)


# @pytest.mark.parametrize(
#     ("trace_nm", "expected_trace_distances_to_last_points"),
#     [
#         pytest.param(
#             np.array(
#                 [
#                     [0, 0],
#                     [1, 0],
#                     [1, 1],
#                     [0, 1],
#                 ]
#             ),
#             np.array([1, 1, 1, 1]),
#             id="square",
#         ),
#         pytest.param(
#             np.array(
#                 [
#                     [0, 0],
#                     [1, 1],
#                     [1, 2],
#                     [0, 2],
#                     [-1, 2],
#                     [-1, -1],
#                 ]
#             ),
#             np.array(
#                 [
#                     np.sqrt(2),
#                     np.sqrt(2),
#                     1,
#                     1,
#                     1,
#                     3,
#                 ]
#             ),
#             id="square with last point",
#         ),
#     ],
# )
# def test_calculate_trace_distances_to_last_points_circular(
#     trace_nm: npt.NDArray[np.number], expected_trace_distances_to_last_points: npt.NDArray[np.number]
# ) -> None:
#     """Test the calculation of distances between points in a trace."""
#     # Calculate distances between points
#     trace_distances_to_last_points = calculate_trace_distances_to_last_points_circular(
#         trace_nm=trace_nm,
#     )

#     np.testing.assert_array_equal(trace_distances_to_last_points, expected_trace_distances_to_last_points)


# @pytest.mark.parametrize(
#     ("curvature_defects", "trace_distances_to_last_points", "expected_distances_between_defects"),
#     [
#         pytest.param(
#             np.array(
#                 [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
#             ),
#             np.array([1.1, 1.3, 1.2, 1.1, 1.3, 1.1, 1.2, 1.4, 1.5, 1.3]),
#             np.array([3.6, 5.2]).astype(np.float32),
#             id="gap at start and end",
#         ),
#         pytest.param(
#             np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 1]),
#             np.array([1.1, 1.3, 1.2, 1.1, 1.3, 1.1, 1.2, 1.4, 1.5, 1.3]),
#             np.array([3.6, 2.4]).astype(np.float32),
#             id="no gap at end, only at start",
#         ),
#         pytest.param(
#             np.array([1, 1, 1, 1, 0, 0, 1, 1, 0, 0]),
#             np.array([1.1, 1.3, 1.2, 1.1, 1.3, 1.1, 1.2, 1.4, 1.5, 1.3]),
#             np.array([3.6, 2.8]).astype(np.float32),
#             id="no gap at start, only at end",
#         ),
#         pytest.param(
#             np.array([1, 1, 1, 1, 0, 0, 1, 1, 1, 1]),
#             np.array([1.1, 1.3, 1.2, 1.1, 1.3, 1.1, 1.2, 1.4, 1.5, 1.3]),
#             np.array([3.6]).astype(np.float32),
#             id="no gap at start or at end",
#         ),
#         pytest.param(
#             np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
#             np.array([1.1, 1.3, 1.2, 1.1, 1.3, 1.1, 1.2, 1.4, 1.5, 1.3]),
#             np.array([]).astype(np.float32),
#             id="no defects",
#         ),
#         pytest.param(
#             np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
#             np.array([1.1, 1.3, 1.2, 1.1, 1.3, 1.1, 1.2, 1.4, 1.5, 1.3]),
#             np.array([]).astype(np.float32),
#             id="all defects, no gaps",
#         ),
#     ],
# )
# def test_calculate_distances_between_defects_circular(
#     curvature_defects: npt.NDArray[np.bool_],
#     trace_distances_to_last_points: npt.NDArray[np.number],
#     expected_distances_between_defects: npt.NDArray[np.float32],
# ) -> None:
#     """Test the calculation of distances between defects."""
#     # Calculate distances between defects
#     distances_between_defects = calculate_distances_between_defects_circular(
#         curvature_defects=curvature_defects,
#         trace_distances_to_last_points=trace_distances_to_last_points,
#     )

#     np.testing.assert_array_equal(distances_between_defects, expected_distances_between_defects)


# @pytest.mark.parametrize(
#     ("curvature_defects", "circular", "expected_number_of_defects"),
#     [
#         pytest.param(np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0]), False, 2, id="linear, gap at start and end"),
#         pytest.param(np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 1]), False, 2, id="linear, gap at start"),
#         pytest.param(np.array([1, 1, 1, 1, 0, 0, 1, 1, 0, 0]), False, 2, id="linear, gap at end"),
#         pytest.param(np.array([1, 1, 1, 1, 0, 0, 1, 1, 1, 1]), False, 2, id="linear, no gaps at ends"),
#         pytest.param(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), False, 0, id="linear, no defects"),
#         pytest.param(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), False, 1, id="linear, all defects"),
#         pytest.param(np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0]), True, 2, id="circular, gap at start and end"),
#         pytest.param(np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 1]), True, 2, id="circular, gap at start"),
#         pytest.param(np.array([1, 1, 1, 1, 0, 0, 1, 1, 0, 0]), True, 2, id="circular, gap at end"),
#         pytest.param(np.array([1, 1, 1, 1, 0, 0, 1, 1, 1, 1]), True, 1, id="circular, no gaps at ends"),
#         pytest.param(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), True, 0, id="circular, no defects"),
#         pytest.param(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), True, 1, id="circular, all defects"),
#     ],
# )
# def test_calculate_number_of_defects(
#     curvature_defects: npt.NDArray[np.bool_], circular: bool, expected_number_of_defects: int
# ) -> None:
#     """Test the calculation of the number of defects."""
#     # Calculate the number of defects
#     number_of_defects = calculate_number_of_defects(curvature_defects=curvature_defects, circular=circular)

#     assert number_of_defects == expected_number_of_defects


def test_calculate_curvature_stats_image() -> None:
    """Test the calculation of curvature statistics for an image."""
    all_grain_smoothed_data: dict = {
        "grain_0": {
            "mol_0": {
                "spline_coords": np.array(
                    [
                        np.array([7.0, 12.0]),
                        np.array([4.375, 14.75]),
                        np.array([3.77777778, 16.11111111]),
                        np.array([3.55555556, 17.0]),
                        np.array([3.44444444, 18.0]),
                        np.array([3.6, 19.5]),
                        np.array([3.55555556, 20.0]),
                        np.array([3.88888889, 21.0]),
                        np.array([4.33333333, 21.88888889]),
                        np.array([4.88888889, 22.66666667]),
                        np.array([5.55555556, 23.22222222]),
                        np.array([6.33333333, 23.55555556]),
                        np.array([7.0, 23.66666667]),
                        np.array([7.66666667, 23.55555556]),
                        np.array([8.33333333, 23.22222222]),
                        np.array([8.88888889, 22.66666667]),
                        np.array([9.22222222, 21.88888889]),
                        np.array([9.44444444, 21.0]),
                        np.array([9.55555556, 20.0]),
                        np.array([9.5, 18.5]),
                        np.array([9.4, 17.5]),
                        np.array([9.0, 13.0]),
                    ]
                ),
                "tracing_stats": {"end_to_end_distance": 4.418496527461196e-09},
            },
            "mol_1": {
                "spline_coords": np.array([np.array([12.0, 5.0]), np.array([11.0, 6.5]), np.array([9.0, 11.0])]),
                "tracing_stats": {"end_to_end_distance": 1.3255489582383588e-08},
            },
        },
        "grain_1": {
            "mol_0": {
                "spline_coords": np.array(
                    [
                        np.array([3.8, 10.5]),
                        np.array([3.88888889, 11.0]),
                        np.array([4.33333333, 11.88888889]),
                        np.array([4.88888889, 12.77777778]),
                        np.array([5.55555556, 13.55555556]),
                        np.array([6.33333333, 14.22222222]),
                        np.array([7.11111111, 14.88888889]),
                        np.array([8.0, 15.33333333]),
                        np.array([8.88888889, 15.55555556]),
                        np.array([9.66666667, 15.55555556]),
                        np.array([10.33333333, 15.33333333]),
                        np.array([10.88888889, 15.0]),
                        np.array([11.3, 14.0]),
                        np.array([11.55555556, 13.77777778]),
                        np.array([11.66666667, 13.0]),
                        np.array([11.55555556, 12.0]),
                        np.array([11.1, 10.5]),
                        np.array([11.0, 10.0]),
                        np.array([10.55555556, 9.0]),
                        np.array([10.0, 8.0]),
                        np.array([9.33333333, 7.11111111]),
                        np.array([8.55555556, 6.33333333]),
                        np.array([7.77777778, 5.66666667]),
                        np.array([6.88888889, 5.22222222]),
                        np.array([5.8, 5.0]),
                        np.array([5.1, 5.0]),
                        np.array([4.5, 5.2]),
                        np.array([3.90909091, 6.0]),
                        np.array([3.63636364, 6.63636364]),
                        np.array([3.45454545, 7.36363636]),
                        np.array([3.36363636, 8.18181818]),
                        np.array([3.45454545, 9.09090909]),
                        np.array([3.72727273, 10.0]),
                    ]
                ),
                "tracing_stats": {"end_to_end_distance": 0.0},
            }
        },
        "grain_2": {
            "mol_0": {
                "spline_coords": np.array(
                    [
                        np.array([4.1, 14.2]),
                        np.array([4.6, 14.9]),
                        np.array([5.2, 15.4]),
                        np.array([5.44444444, 15.77777778]),
                        np.array([6.22222222, 16.0]),
                        np.array([7.11111111, 16.11111111]),
                        np.array([8.0, 16.0]),
                        np.array([9.0, 15.77777778]),
                        np.array([10.0, 15.44444444]),
                        np.array([11.0, 15.11111111]),
                        np.array([12.0, 14.77777778]),
                        np.array([13.0, 14.33333333]),
                        np.array([14.0, 14.0]),
                        np.array([15.5, 13.6]),
                        np.array([16.0, 13.44444444]),
                        np.array([17.0, 13.11111111]),
                        np.array([17.88888889, 12.66666667]),
                        np.array([18.66666667, 12.11111111]),
                        np.array([19.22222222, 11.44444444]),
                        np.array([19.55555556, 10.77777778]),
                        np.array([19.66666667, 10.22222222]),
                        np.array([19.55555556, 9.66666667]),
                        np.array([19.22222222, 9.0]),
                        np.array([18.66666667, 8.44444444]),
                        np.array([17.88888889, 7.88888889]),
                        np.array([17.0, 7.44444444]),
                        np.array([16.0, 7.0]),
                        np.array([15.0, 6.55555556]),
                        np.array([14.0, 6.0]),
                        np.array([13.0, 5.44444444]),
                        np.array([12.0, 4.88888889]),
                        np.array([11.0, 4.55555556]),
                        np.array([10.0, 4.33333333]),
                        np.array([9.0, 4.33333333]),
                        np.array([8.5, 4.125]),
                        np.array([7.5, 4.375]),
                        np.array([6.22222222, 5.33333333]),
                        np.array([5.2, 6.4]),
                        np.array([4.6, 7.2]),
                        np.array([4.1, 8.1]),
                        np.array([3.7, 9.0]),
                        np.array([3.4, 9.9]),
                        np.array([3.3, 10.8]),
                        np.array([3.45454545, 12.18181818]),
                        np.array([3.4, 12.6]),
                        np.array([4.0, 13.72727273]),
                    ]
                ),
                "tracing_stats": {"end_to_end_distance": 0.0},
            },
        },
    }

    pixel_to_nm_scaling = 1.97601171875

    # Calculate curvature statistics
    curvature_stats = calculate_curvature_stats_image(
        all_grain_smoothed_data=all_grain_smoothed_data,
        pixel_to_nm_scaling=pixel_to_nm_scaling,
    )

    expected_curvature_stats = {
        "grain_0": {
            "mol_0": np.array(
                np.array(
                    [
                        0.0,
                        -0.04641295445891346,
                        -0.05737042572658705,
                        -0.07418974572278295,
                        -0.10763231579161422,
                        0.0644281215538495,
                        -0.41375728579730847,
                        -0.0681247810770507,
                        -0.0797452755749914,
                        -0.13544164334822273,
                        -0.16902701481008917,
                        -0.14337886216360415,
                        -0.24731885502240258,
                        -0.22350894703189503,
                        -0.21845703786136855,
                        -0.24509244888484372,
                        -0.09563637408173611,
                        -0.0741897457227818,
                        -0.07427795183552616,
                        -0.02112187607031149,
                        0.005545572545890469,
                        0.0,
                    ]
                )
            ),
            "mol_1": np.array([0.0, -0.047659657577634705, 0.0]),
        },
        "grain_1": {
            "mol_0": np.array(
                [
                    -0.031548436637750935,
                    -0.28670527720504535,
                    -0.04835159152970009,
                    -0.07243146847934859,
                    -0.07585352468538549,
                    6.581653379523217e-16,
                    -0.12102408032118833,
                    -0.11135125379380306,
                    -0.13530903525628518,
                    -0.20935062292340145,
                    -0.15747445330244694,
                    -0.5001687333420446,
                    0.21764978626566406,
                    -1.0656865477373554,
                    -0.1626757122877418,
                    -0.09264490302583304,
                    0.03146067506365124,
                    -0.2191691700797999,
                    -0.04110008384448684,
                    -0.06034242003101275,
                    -0.06462884199114696,
                    -0.0353217486122902,
                    -0.12102408032118725,
                    -0.13358472112413752,
                    -0.09167413816744112,
                    -0.23261180324822428,
                    -0.4903878980314438,
                    -0.11769118417381,
                    -0.1168889016554551,
                    -0.09067635588340113,
                    -0.1292971477831398,
                    -0.10623417124400564,
                    0.0783884734450021,
                ]
            )
        },
        "grain_2": {
            "mol_0": np.array(
                [
                    -0.4312843442610738,
                    -0.15049071483135906,
                    0.19552283261340256,
                    -0.8077398967552533,
                    -0.09631184447752437,
                    -0.1405042767447404,
                    -0.05328098628632549,
                    -0.05092425891404267,
                    -1.0393853232897832e-15,
                    1.8122615893257776e-15,
                    -0.04631706297445449,
                    0.04461453799302988,
                    0.029357235112230987,
                    -0.013370947930336401,
                    -0.019456065744626674,
                    -0.06812478107705004,
                    -0.07974527557499123,
                    -0.1354416433482232,
                    -0.1347629417161631,
                    -0.18077555170372614,
                    -0.3526417230371116,
                    -0.23782597097460745,
                    -0.2184570378613685,
                    -0.1063758617787666,
                    -0.08291518933993429,
                    -0.023130578028827148,
                    -2.053701629033582e-16,
                    0.04110008384448672,
                    0.0,
                    0.0,
                    -0.08199509019381802,
                    -0.04948948400140509,
                    -0.10802658374983763,
                    0.1997918918970285,
                    -0.5977259314866527,
                    -0.19565853690757773,
                    -0.051699533761804445,
                    -0.04131848565197119,
                    -0.06902924871682137,
                    -0.04368507975208147,
                    -0.04957170888114285,
                    -0.1126065743987667,
                    -0.1240874500803583,
                    0.0877454484745392,
                    -0.7425881616022431,
                    0.1112217593954098,
                ]
            )
        },
    }

    assert dict_almost_equal(curvature_stats, expected_curvature_stats, abs_tol=1e-5)
