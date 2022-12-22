"""Tests of the dnacurvature module."""
import pytest

import math
import numpy as np

from topostats.tracing.dnacurvature import Curvature


# Test curvature calculated on circles
def circle_coordinates(points=4, radius=1) -> np.ndarray:
    """A circle for testing curvature class and methods."""
    coordinates = np.zeros([points, 2])
    for i in np.arange(points):
        theta = 2 * math.pi / points * i
        x = -math.cos(theta) * radius
        y = math.sin(theta) * radius
        coordinates[i][0] = x
        coordinates[i][1] = y
    return coordinates


@pytest.mark.parametrize(
    "points, radius, edge_order, expected_variance, expected_first_derivative, expected_second_derivative, expected_local_curvature",
    [
        # 4-point circle with varying radius and edge_order
        (
            4,
            1,
            1,
            0,
            np.asarray(
                [
                    [-1.224647e-16, 1.000000e00],
                    [1.000000e00, 6.123234e-17],
                    [1.224647e-16, -1.000000e00],
                    [-1.000000e00, -6.123234e-17],
                ]
            ),
            np.asarray(
                [
                    [1.0000000e00, 6.1232340e-17],
                    [1.2246468e-16, -1.0000000e00],
                    [-1.0000000e00, -6.1232340e-17],
                    [-1.2246468e-16, 1.0000000e00],
                ]
            ),
            np.asarray(
                [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ]
            ),
        ),
        (
            4,
            1,
            2,
            0,
            np.asarray(
                [
                    [-6.123234e-17, 5.000000e-01],
                    [5.000000e-01, 3.061617e-17],
                    [6.123234e-17, -5.000000e-01],
                    [-5.000000e-01, -3.061617e-17],
                ]
            ),
            np.asarray(
                [
                    [2.5000000e-01, 1.5308085e-17],
                    [3.0616170e-17, -2.5000000e-01],
                    [-2.5000000e-01, -1.5308085e-17],
                    [-3.0616170e-17, 2.5000000e-01],
                ]
            ),
            np.asarray([1.0, 1.0, 1.0, 1.0]),
        ),
        (
            4,
            10,
            1,
            0,
            np.asarray(
                [
                    [-1.224647e-15, 1.000000e01],
                    [1.000000e01, 6.123234e-16],
                    [1.224647e-15, -1.000000e01],
                    [-1.000000e01, -6.123234e-16],
                ]
            ),
            np.asarray(
                [
                    [1.000000e01, 6.123234e-16],
                    [1.224647e-15, -1.000000e01],
                    [-1.000000e01, -6.123234e-16],
                    [-1.224647e-15, 1.000000e01],
                ]
            ),
            np.asarray([0.1, 0.1, 0.1, 0.1]),
        ),
        (
            4,
            10,
            2,
            0,
            np.asarray(
                [
                    [-6.123234e-16, 5.000000e00],
                    [5.000000e00, 3.061617e-16],
                    [6.123234e-16, -5.000000e00],
                    [-5.000000e00, -3.061617e-16],
                ]
            ),
            np.asarray(
                [
                    [2.500000e00, 1.530808e-16],
                    [3.061617e-16, -2.500000e00],
                    [-2.500000e00, -1.530808e-16],
                    [-3.061617e-16, 2.500000e00],
                ]
            ),
            np.asarray([0.1, 0.1, 0.1, 0.1]),
        ),
        # 10-point circle with varying radius and edge_order
        (
            10,
            1,
            1,
            0,
            np.asarray(
                [
                    [-5.55111512e-17, 5.87785252e-01],
                    [3.45491503e-01, 4.75528258e-01],
                    [5.59016994e-01, 1.81635632e-01],
                    [5.59016994e-01, -1.81635632e-01],
                    [3.45491503e-01, -4.75528258e-01],
                    [1.11022302e-16, -5.87785252e-01],
                    [-3.45491503e-01, -4.75528258e-01],
                    [-5.59016994e-01, -1.81635632e-01],
                    [-5.59016994e-01, 1.81635632e-01],
                    [-3.45491503e-01, 4.75528258e-01],
                ]
            ),
            np.asarray(
                [
                    [3.45491503e-01, -2.77555756e-17],
                    [2.79508497e-01, -2.03074810e-01],
                    [1.06762746e-01, -3.28581945e-01],
                    [-1.06762746e-01, -3.28581945e-01],
                    [-2.79508497e-01, -2.03074810e-01],
                    [-3.45491503e-01, -2.77555756e-17],
                    [-2.79508497e-01, 2.03074810e-01],
                    [-1.06762746e-01, 3.28581945e-01],
                    [1.06762746e-01, 3.28581945e-01],
                    [2.79508497e-01, 2.03074810e-01],
                ]
            ),
            np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        ),
        (
            10,
            1,
            2,
            0,
            np.asarray(
                [
                    [-2.77555756e-17, 2.93892626e-01],
                    [1.72745751e-01, 2.37764129e-01],
                    [2.79508497e-01, 9.08178160e-02],
                    [2.79508497e-01, -9.08178160e-02],
                    [1.72745751e-01, -2.37764129e-01],
                    [5.55111512e-17, -2.93892626e-01],
                    [-1.72745751e-01, -2.37764129e-01],
                    [-2.79508497e-01, -9.08178160e-02],
                    [-2.79508497e-01, 9.08178160e-02],
                    [-1.72745751e-01, 2.37764129e-01],
                ]
            ),
            np.asarray(
                [
                    [8.63728757e-02, -6.93889390e-18],
                    [6.98771243e-02, -5.07687025e-02],
                    [2.66906864e-02, -8.21454863e-02],
                    [-2.66906864e-02, -8.21454863e-02],
                    [-6.98771243e-02, -5.07687025e-02],
                    [-8.63728757e-02, -6.93889390e-18],
                    [-6.98771243e-02, 5.07687025e-02],
                    [-2.66906864e-02, 8.21454863e-02],
                    [2.66906864e-02, 8.21454863e-02],
                    [6.98771243e-02, 5.07687025e-02],
                ]
            ),
            np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        ),
        (
            10,
            10,
            1,
            0,
            np.asarray(
                [
                    [-8.88178420e-16, 5.87785252e00],
                    [3.45491503e00, 4.75528258e00],
                    [5.59016994e00, 1.81635632e00],
                    [5.59016994e00, -1.81635632e00],
                    [3.45491503e00, -4.75528258e00],
                    [1.77635684e-15, -5.87785252e00],
                    [-3.45491503e00, -4.75528258e00],
                    [-5.59016994e00, -1.81635632e00],
                    [-5.59016994e00, 1.81635632e00],
                    [-3.45491503e00, 4.75528258e00],
                ]
            ),
            np.asarray(
                [
                    [3.45491503e00, -4.44089210e-16],
                    [2.79508497e00, -2.03074810e00],
                    [1.06762746e00, -3.28581945e00],
                    [-1.06762746e00, -3.28581945e00],
                    [-2.79508497e00, -2.03074810e00],
                    [-3.45491503e00, -4.44089210e-16],
                    [-2.79508497e00, 2.03074810e00],
                    [-1.06762746e00, 3.28581945e00],
                    [1.06762746e00, 3.28581945e00],
                    [2.79508497e00, 2.03074810e00],
                ]
            ),
            np.asarray([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
        ),
        (
            10,
            10,
            2,
            0,
            np.asarray(
                [
                    [-4.44089210e-16, 2.93892626e00],
                    [1.72745751e00, 2.37764129e00],
                    [2.79508497e00, 9.08178160e-01],
                    [2.79508497e00, -9.08178160e-01],
                    [1.72745751e00, -2.37764129e00],
                    [8.88178420e-16, -2.93892626e00],
                    [-1.72745751e00, -2.37764129e00],
                    [-2.79508497e00, -9.08178160e-01],
                    [-2.79508497e00, 9.08178160e-01],
                    [-1.72745751e00, 2.37764129e00],
                ]
            ),
            np.asarray(
                [
                    [8.63728757e-01, -1.11022302e-16],
                    [6.98771243e-01, -5.07687025e-01],
                    [2.66906864e-01, -8.21454863e-01],
                    [-2.66906864e-01, -8.21454863e-01],
                    [-6.98771243e-01, -5.07687025e-01],
                    [-8.63728757e-01, -1.11022302e-16],
                    [-6.98771243e-01, 5.07687025e-01],
                    [-2.66906864e-01, 8.21454863e-01],
                    [2.66906864e-01, 8.21454863e-01],
                    [6.98771243e-01, 5.07687025e-01],
                ]
            ),
            np.asarray([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
        ),
    ],
)
def test_curvature_circle(
    points,
    radius,
    edge_order,
    expected_variance,
    expected_first_derivative,
    expected_second_derivative,
    expected_local_curvature,
) -> None:
    """Test first and second derivatives and local curvature of a circle.

    The variance of local curvature is also tested, for a circle it should be zero."""
    circle_points = circle_coordinates(points, radius)
    curvature = Curvature(molecule_coordinates=circle_points, circular=True)
    curvature.calculate_derivatives(edge_order=edge_order)
    curvature._calculate_local_curvature()
    np.testing.assert_array_almost_equal(curvature.first_derivative, expected_first_derivative)
    np.testing.assert_array_almost_equal(curvature.second_derivative, expected_second_derivative)
    np.testing.assert_array_almost_equal(curvature.local_curvature, expected_local_curvature)
    np.testing.assert_almost_equal(np.var(curvature.local_curvature), expected_variance)


@pytest.mark.parametrize(
    "points, radius, shift, edge_order, expected_variance, expected_first_derivative, expected_second_derivative, expected_local_curvature",
    [
        (
            10,
            10,
            5,  # Shift array by 3 points
            2,
            0,
            np.asarray(
                [
                    [-4.44089210e-16, 2.93892626e00],
                    [1.72745751e00, 2.37764129e00],
                    [2.79508497e00, 9.08178160e-01],
                    [2.79508497e00, -9.08178160e-01],
                    [1.72745751e00, -2.37764129e00],
                    [8.88178420e-16, -2.93892626e00],
                    [-1.72745751e00, -2.37764129e00],
                    [-2.79508497e00, -9.08178160e-01],
                    [-2.79508497e00, 9.08178160e-01],
                    [-1.72745751e00, 2.37764129e00],
                ]
            ),
            np.asarray(
                [
                    [8.63728757e-01, -1.11022302e-16],
                    [6.98771243e-01, -5.07687025e-01],
                    [2.66906864e-01, -8.21454863e-01],
                    [-2.66906864e-01, -8.21454863e-01],
                    [-6.98771243e-01, -5.07687025e-01],
                    [-8.63728757e-01, -1.11022302e-16],
                    [-6.98771243e-01, 5.07687025e-01],
                    [-2.66906864e-01, 8.21454863e-01],
                    [2.66906864e-01, 8.21454863e-01],
                    [6.98771243e-01, 5.07687025e-01],
                ]
            ),
            np.asarray([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
        ),
        (
            10,
            10,
            5,  # Shift array by 5 points
            2,
            0,
            np.asarray(
                [
                    [-4.44089210e-16, 2.93892626e00],
                    [1.72745751e00, 2.37764129e00],
                    [2.79508497e00, 9.08178160e-01],
                    [2.79508497e00, -9.08178160e-01],
                    [1.72745751e00, -2.37764129e00],
                    [8.88178420e-16, -2.93892626e00],
                    [-1.72745751e00, -2.37764129e00],
                    [-2.79508497e00, -9.08178160e-01],
                    [-2.79508497e00, 9.08178160e-01],
                    [-1.72745751e00, 2.37764129e00],
                ]
            ),
            np.asarray(
                [
                    [8.63728757e-01, -1.11022302e-16],
                    [6.98771243e-01, -5.07687025e-01],
                    [2.66906864e-01, -8.21454863e-01],
                    [-2.66906864e-01, -8.21454863e-01],
                    [-6.98771243e-01, -5.07687025e-01],
                    [-8.63728757e-01, -1.11022302e-16],
                    [-6.98771243e-01, 5.07687025e-01],
                    [-2.66906864e-01, 8.21454863e-01],
                    [2.66906864e-01, 8.21454863e-01],
                    [6.98771243e-01, 5.07687025e-01],
                ]
            ),
            np.asarray([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
        ),
        (
            10,
            10,
            7,  # Shift array by 7 points
            2,
            0,
            np.asarray(
                [
                    [-4.44089210e-16, 2.93892626e00],
                    [1.72745751e00, 2.37764129e00],
                    [2.79508497e00, 9.08178160e-01],
                    [2.79508497e00, -9.08178160e-01],
                    [1.72745751e00, -2.37764129e00],
                    [8.88178420e-16, -2.93892626e00],
                    [-1.72745751e00, -2.37764129e00],
                    [-2.79508497e00, -9.08178160e-01],
                    [-2.79508497e00, 9.08178160e-01],
                    [-1.72745751e00, 2.37764129e00],
                ]
            ),
            np.asarray(
                [
                    [8.63728757e-01, -1.11022302e-16],
                    [6.98771243e-01, -5.07687025e-01],
                    [2.66906864e-01, -8.21454863e-01],
                    [-2.66906864e-01, -8.21454863e-01],
                    [-6.98771243e-01, -5.07687025e-01],
                    [-8.63728757e-01, -1.11022302e-16],
                    [-6.98771243e-01, 5.07687025e-01],
                    [-2.66906864e-01, 8.21454863e-01],
                    [2.66906864e-01, 8.21454863e-01],
                    [6.98771243e-01, 5.07687025e-01],
                ]
            ),
            np.asarray([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
        ),
    ],
)
def test_curvature_circle_starting_point(
    points,
    radius,
    shift,
    edge_order,
    expected_variance,
    expected_first_derivative,
    expected_second_derivative,
    expected_local_curvature,
) -> None:
    """Test that the starting point for measuring local curvature of a circle gives the same first and second derivative
    and the same overall local curvature."""
    # Curvature for original circle
    circle_points = circle_coordinates(points, radius)
    curvature = Curvature(molecule_coordinates=circle_points, circular=True)
    curvature.calculate_derivatives(edge_order=edge_order)
    curvature._calculate_local_curvature()

    # Curvature for shifted starting point
    circle_points_rolled = np.roll(circle_points, shift=shift, axis=0)
    curvature_rolled = Curvature(molecule_coordinates=circle_points_rolled, circular=True)
    curvature_rolled.calculate_derivatives(edge_order=edge_order)
    curvature_rolled._calculate_local_curvature()

    # NB - We have to roll the resulting first and second derivatives back
    np.testing.assert_array_almost_equal(
        np.roll(curvature_rolled.first_derivative, shift=(points - shift), axis=0), expected_first_derivative
    )
    np.testing.assert_array_almost_equal(
        np.roll(curvature_rolled.second_derivative, shift=(points - shift), axis=0), expected_second_derivative
    )
    # No need to roll the local curvature back because this is a circle it should be the same everywhere
    np.testing.assert_array_almost_equal(curvature_rolled.local_curvature, expected_local_curvature)
    np.testing.assert_almost_equal(np.var(curvature_rolled.local_curvature), expected_variance)
    np.testing.assert_array_almost_equal(curvature.local_curvature, curvature_rolled.local_curvature)


# Test curvature calculated on ellipse'
def ellipse_coordinates(
    points: int = 4, major: float = 5.0, minor: float = 1.0, displacement: float = 0.0
) -> np.ndarray:
    """An ellipse for testing curvature class and methods."""
    coordinates = np.zeros([points, 2])
    for i in range(0, points):
        theta = 2 * math.pi / points * i
        x = -math.cos(theta) * major
        y = math.sin(theta) * minor
        coordinates[i][0] = x
        coordinates[i][1] = y
    return coordinates


@pytest.mark.parametrize(
    "points, major, minor, displacement, edge_order, expected_first_derivative, expected_second_derivative, expected_local_curvature",
    [
        (
            10,
            5.0,
            1.0,
            0.0,
            1,
            np.asarray(
                [
                    [-4.44089210e-16, 5.87785252e-01],
                    [1.72745751e00, 4.75528258e-01],
                    [2.79508497e00, 1.81635632e-01],
                    [2.79508497e00, -1.81635632e-01],
                    [1.72745751e00, -4.75528258e-01],
                    [8.88178420e-16, -5.87785252e-01],
                    [-1.72745751e00, -4.75528258e-01],
                    [-2.79508497e00, -1.81635632e-01],
                    [-2.79508497e00, 1.81635632e-01],
                    [-1.72745751e00, 4.75528258e-01],
                ]
            ),
            np.asarray(
                [
                    [1.72745751e00, -2.77555756e-17],
                    [1.39754249e00, -2.03074810e-01],
                    [5.33813729e-01, -3.28581945e-01],
                    [-5.33813729e-01, -3.28581945e-01],
                    [-1.39754249e00, -2.03074810e-01],
                    [-1.72745751e00, -2.77555756e-17],
                    [-1.39754249e00, 2.03074810e-01],
                    [-5.33813729e-01, 3.28581945e-01],
                    [5.33813729e-01, 3.28581945e-01],
                    [1.39754249e00, 2.03074810e-01],
                ]
            ),
            np.asarray(
                [5.0, 0.1765308, 0.04620573, 0.04620573, 0.1765308, 5.0, 0.1765308, 0.04620573, 0.04620573, 0.1765308]
            ),
        ),
        (
            10,
            5.0,
            1.0,
            0.0,
            2,
            np.asarray(
                [
                    [-2.22044605e-16, 2.93892626e-01],
                    [8.63728757e-01, 2.37764129e-01],
                    [1.39754249e00, 9.08178160e-02],
                    [1.39754249e00, -9.08178160e-02],
                    [8.63728757e-01, -2.37764129e-01],
                    [4.44089210e-16, -2.93892626e-01],
                    [-8.63728757e-01, -2.37764129e-01],
                    [-1.39754249e00, -9.08178160e-02],
                    [-1.39754249e00, 9.08178160e-02],
                    [-8.63728757e-01, 2.37764129e-01],
                ]
            ),
            np.asarray(
                [
                    [4.31864379e-01, -6.93889390e-18],
                    [3.49385621e-01, -5.07687025e-02],
                    [1.33453432e-01, -8.21454863e-02],
                    [-1.33453432e-01, -8.21454863e-02],
                    [-3.49385621e-01, -5.07687025e-02],
                    [-4.31864379e-01, -6.93889390e-18],
                    [-3.49385621e-01, 5.07687025e-02],
                    [-1.33453432e-01, 8.21454863e-02],
                    [1.33453432e-01, 8.21454863e-02],
                    [3.49385621e-01, 5.07687025e-02],
                ]
            ),
            np.asarray(
                [5.0, 0.1765308, 0.04620573, 0.04620573, 0.1765308, 5.0, 0.1765308, 0.04620573, 0.04620573, 0.1765308]
            ),
        ),
        (
            10,
            20.0,
            6.0,
            0.0,
            1,
            np.asarray(
                [
                    [-1.77635684e-15, 3.52671151e00],
                    [6.90983006e00, 2.85316955e00],
                    [1.11803399e01, 1.08981379e00],
                    [1.11803399e01, -1.08981379e00],
                    [6.90983006e00, -2.85316955e00],
                    [3.55271368e-15, -3.52671151e00],
                    [-6.90983006e00, -2.85316955e00],
                    [-1.11803399e01, -1.08981379e00],
                    [-1.11803399e01, 1.08981379e00],
                    [-6.90983006e00, 2.85316955e00],
                ]
            ),
            np.asarray(
                [
                    [6.90983006e00, -2.22044605e-16],
                    [5.59016994e00, -1.21844886e00],
                    [2.13525492e00, -1.97149167e00],
                    [-2.13525492e00, -1.97149167e00],
                    [-5.59016994e00, -1.21844886e00],
                    [-6.90983006e00, -2.22044605e-16],
                    [-5.59016994e00, 1.21844886e00],
                    [-2.13525492e00, 1.97149167e00],
                    [2.13525492e00, 1.97149167e00],
                    [5.59016994e00, 1.21844886e00],
                ]
            ),
            np.asarray(
                [
                    0.55555556,
                    0.05832825,
                    0.01719142,
                    0.01719142,
                    0.05832825,
                    0.55555556,
                    0.05832825,
                    0.01719142,
                    0.01719142,
                    0.05832825,
                ]
            ),
        ),
    ],
)
def test_curvature_ellipse(
    points,
    major,
    minor,
    displacement,
    edge_order,
    expected_first_derivative,
    expected_second_derivative,
    expected_local_curvature,
) -> None:
    ellipse_points = ellipse_coordinates(points, major, minor, displacement)
    curvature = Curvature(molecule_coordinates=ellipse_points, circular=True)
    curvature.calculate_derivatives(edge_order=edge_order)
    curvature._calculate_local_curvature()
    np.testing.assert_array_almost_equal(curvature.first_derivative, expected_first_derivative)
    np.testing.assert_array_almost_equal(curvature.second_derivative, expected_second_derivative)
    np.testing.assert_array_almost_equal(curvature.local_curvature, expected_local_curvature)


@pytest.mark.parametrize(
    "points, major, minor, displacement, shift, edge_order, expected_first_derivative, expected_second_derivative, expected_local_curvature",
    [
        (
            10,
            5.0,
            1.0,
            0.0,
            3,  # Shift array by 3 points
            1,
            np.asarray(
                [
                    [-4.44089210e-16, 5.87785252e-01],
                    [1.72745751e00, 4.75528258e-01],
                    [2.79508497e00, 1.81635632e-01],
                    [2.79508497e00, -1.81635632e-01],
                    [1.72745751e00, -4.75528258e-01],
                    [8.88178420e-16, -5.87785252e-01],
                    [-1.72745751e00, -4.75528258e-01],
                    [-2.79508497e00, -1.81635632e-01],
                    [-2.79508497e00, 1.81635632e-01],
                    [-1.72745751e00, 4.75528258e-01],
                ]
            ),
            np.asarray(
                [
                    [1.72745751e00, -2.77555756e-17],
                    [1.39754249e00, -2.03074810e-01],
                    [5.33813729e-01, -3.28581945e-01],
                    [-5.33813729e-01, -3.28581945e-01],
                    [-1.39754249e00, -2.03074810e-01],
                    [-1.72745751e00, -2.77555756e-17],
                    [-1.39754249e00, 2.03074810e-01],
                    [-5.33813729e-01, 3.28581945e-01],
                    [5.33813729e-01, 3.28581945e-01],
                    [1.39754249e00, 2.03074810e-01],
                ]
            ),
            np.asarray(
                [5.0, 0.1765308, 0.04620573, 0.04620573, 0.1765308, 5.0, 0.1765308, 0.04620573, 0.04620573, 0.1765308]
            ),
        ),
        (
            10,
            5.0,
            1.0,
            0.0,
            3,  # Shift array by 5 points
            1,
            np.asarray(
                [
                    [-4.44089210e-16, 5.87785252e-01],
                    [1.72745751e00, 4.75528258e-01],
                    [2.79508497e00, 1.81635632e-01],
                    [2.79508497e00, -1.81635632e-01],
                    [1.72745751e00, -4.75528258e-01],
                    [8.88178420e-16, -5.87785252e-01],
                    [-1.72745751e00, -4.75528258e-01],
                    [-2.79508497e00, -1.81635632e-01],
                    [-2.79508497e00, 1.81635632e-01],
                    [-1.72745751e00, 4.75528258e-01],
                ]
            ),
            np.asarray(
                [
                    [1.72745751e00, -2.77555756e-17],
                    [1.39754249e00, -2.03074810e-01],
                    [5.33813729e-01, -3.28581945e-01],
                    [-5.33813729e-01, -3.28581945e-01],
                    [-1.39754249e00, -2.03074810e-01],
                    [-1.72745751e00, -2.77555756e-17],
                    [-1.39754249e00, 2.03074810e-01],
                    [-5.33813729e-01, 3.28581945e-01],
                    [5.33813729e-01, 3.28581945e-01],
                    [1.39754249e00, 2.03074810e-01],
                ]
            ),
            np.asarray(
                [5.0, 0.1765308, 0.04620573, 0.04620573, 0.1765308, 5.0, 0.1765308, 0.04620573, 0.04620573, 0.1765308]
            ),
        ),
        (
            10,
            5.0,
            1.0,
            0.0,
            3,  # Shift array by 7 points
            1,
            np.asarray(
                [
                    [-4.44089210e-16, 5.87785252e-01],
                    [1.72745751e00, 4.75528258e-01],
                    [2.79508497e00, 1.81635632e-01],
                    [2.79508497e00, -1.81635632e-01],
                    [1.72745751e00, -4.75528258e-01],
                    [8.88178420e-16, -5.87785252e-01],
                    [-1.72745751e00, -4.75528258e-01],
                    [-2.79508497e00, -1.81635632e-01],
                    [-2.79508497e00, 1.81635632e-01],
                    [-1.72745751e00, 4.75528258e-01],
                ]
            ),
            np.asarray(
                [
                    [1.72745751e00, -2.77555756e-17],
                    [1.39754249e00, -2.03074810e-01],
                    [5.33813729e-01, -3.28581945e-01],
                    [-5.33813729e-01, -3.28581945e-01],
                    [-1.39754249e00, -2.03074810e-01],
                    [-1.72745751e00, -2.77555756e-17],
                    [-1.39754249e00, 2.03074810e-01],
                    [-5.33813729e-01, 3.28581945e-01],
                    [5.33813729e-01, 3.28581945e-01],
                    [1.39754249e00, 2.03074810e-01],
                ]
            ),
            np.asarray(
                [5.0, 0.1765308, 0.04620573, 0.04620573, 0.1765308, 5.0, 0.1765308, 0.04620573, 0.04620573, 0.1765308]
            ),
        ),
    ],
)
def test_curvature_ellipse_starting_point(
    points,
    major,
    minor,
    shift,
    displacement,
    edge_order,
    expected_first_derivative,
    expected_second_derivative,
    expected_local_curvature,
) -> None:
    ellipse_points = ellipse_coordinates(points, major, minor, displacement)
    curvature = Curvature(molecule_coordinates=ellipse_points, circular=True)
    curvature.calculate_derivatives(edge_order=edge_order)
    curvature._calculate_local_curvature()

    ellipse_points_rolled = np.roll(ellipse_points, shift=shift, axis=0)
    curvature_rolled = Curvature(molecule_coordinates=ellipse_points_rolled, circular=True)
    curvature_rolled.calculate_derivatives(edge_order=edge_order)
    curvature_rolled._calculate_local_curvature()

    # NB - We have to roll the first, second AND curvature array back
    np.testing.assert_array_almost_equal(
        np.roll(curvature_rolled.first_derivative, shift=(points - shift), axis=0), expected_first_derivative
    )
    np.testing.assert_array_almost_equal(
        np.roll(curvature_rolled.second_derivative, shift=(points - shift), axis=0), expected_second_derivative
    )
    np.testing.assert_array_almost_equal(
        np.roll(curvature_rolled.local_curvature, shift=(points - shift), axis=0), expected_local_curvature
    )
    # To check we have the same curvature as original (and not just expected) compare directly, shifting the rolled
    # curvature back.
    np.testing.assert_array_almost_equal(
        curvature.local_curvature, np.roll(curvature_rolled.local_curvature, shift=(points - shift), axis=0)
    )


# # Test curvature calculated on parabola
def parabola_coordinates(points: int = 10, order: int = 2) -> np.ndarray:
    """A parabola for testing curvature class and methods.

    Parameters
    ----------
    points: int
        Number of points to generate.
    order: int
        Power to raise y by.
    """
    x = np.linspace(-2, 2, num=points)
    y = x**order
    return np.column_stack((x, y))


@pytest.mark.parametrize(
    "points, order, edge_order, expected_first_derivative, expected_second_derivative, expected_local_curvature",
    [
        (
            10,
            2,
            1,
            np.asarray(
                [
                    [0.44444444, -1.58024691],
                    [0.44444444, -1.38271605],
                    [0.44444444, -0.98765432],
                    [0.44444444, -0.59259259],
                    [0.44444444, -0.19753086],
                    [0.44444444, 0.19753086],
                    [0.44444444, 0.59259259],
                    [0.44444444, 0.98765432],
                    [0.44444444, 1.38271605],
                    [0.44444444, 1.58024691],
                ]
            ),
            np.asarray(
                [
                    [0.00000000e00, 1.97530864e-01],
                    [0.00000000e00, 2.96296296e-01],
                    [0.00000000e00, 3.95061728e-01],
                    [5.55111512e-17, 3.95061728e-01],
                    [0.00000000e00, 3.95061728e-01],
                    [-1.66533454e-16, 3.95061728e-01],
                    [0.00000000e00, 3.95061728e-01],
                    [2.22044605e-16, 3.95061728e-01],
                    [1.11022302e-16, 2.96296296e-01],
                    [0.00000000e00, 1.97530864e-01],
                ]
            ),
            np.asarray(
                [
                    -0.01984651,
                    -0.04298279,
                    -0.13821014,
                    -0.432,
                    -1.52615949,
                    -1.52615949,
                    -0.432,
                    -0.13821014,
                    -0.04298279,
                    -0.01984651,
                ]
            ),
        ),
        (
            10,
            3,
            1,
            np.asarray(
                [
                    [0.44444444, 4.23593964],
                    [0.44444444, 3.31412894],
                    [0.44444444, 1.73388203],
                    [0.44444444, 0.68038409],
                    [0.44444444, 0.15363512],
                    [0.44444444, 0.15363512],
                    [0.44444444, 0.68038409],
                    [0.44444444, 1.73388203],
                    [0.44444444, 3.31412894],
                    [0.44444444, 4.23593964],
                ]
            ),
            np.asarray(
                [
                    [0.00000000e00, -9.21810700e-01],
                    [0.00000000e00, -1.25102881e00],
                    [0.00000000e00, -1.31687243e00],
                    [5.55111512e-17, -7.90123457e-01],
                    [0.00000000e00, -2.63374486e-01],
                    [-1.66533454e-16, 2.63374486e-01],
                    [0.00000000e00, 7.90123457e-01],
                    [2.22044605e-16, 1.31687243e00],
                    [1.11022302e-16, 1.25102881e00],
                    [0.00000000e00, 9.21810700e-01],
                ]
            ),
            np.asarray(
                [
                    0.00530246,
                    0.01487185,
                    0.10205805,
                    0.65425823,
                    1.12565704,
                    -1.12565704,
                    -0.65425823,
                    -0.10205805,
                    -0.01487185,
                    -0.00530246,
                ]
            ),
        ),
        (
            10,
            4,
            2,
            np.asarray(
                [
                    [0.22222222, -5.0723975],
                    [0.22222222, -3.61896052],
                    [0.22222222, -1.41441853],
                    [0.22222222, -0.38042981],
                    [0.22222222, -0.04877305],
                    [0.22222222, 0.04877305],
                    [0.22222222, 0.38042981],
                    [0.22222222, 1.41441853],
                    [0.22222222, 3.61896052],
                    [0.22222222, 5.0723975],
                ]
            ),
            np.asarray(
                [
                    [0.00000000e00, 7.26718488e-01],
                    [0.00000000e00, 9.14494742e-01],
                    [0.00000000e00, 8.09632678e-01],
                    [1.38777878e-17, 3.41411370e-01],
                    [0.00000000e00, 1.07300716e-01],
                    [-4.16333634e-17, 1.07300716e-01],
                    [0.00000000e00, 3.41411370e-01],
                    [5.55111512e-17, 8.09632678e-01],
                    [2.77555756e-17, 9.14494742e-01],
                    [0.00000000e00, 7.26718488e-01],
                ],
            ),
            np.asarray(
                [
                    -1.23385671e-03,
                    -4.26349220e-03,
                    -6.12994586e-02,
                    -8.87145971e-01,
                    -2.02478769e00,
                    -2.02478769e00,
                    -8.87145971e-01,
                    -6.12994586e-02,
                    -4.26349220e-03,
                    -1.23385671e-03,
                ]
            ),
        ),
    ],
)
def test_curvature_parabola(
    points,
    order,
    edge_order,
    expected_first_derivative,
    expected_second_derivative,
    expected_local_curvature,
) -> None:
    parabola_points = parabola_coordinates(points, order)
    curvature = Curvature(molecule_coordinates=parabola_points, circular=False)
    curvature.calculate_derivatives(edge_order=edge_order)
    curvature._calculate_local_curvature()
    np.testing.assert_array_almost_equal(curvature.first_derivative, expected_first_derivative)
    np.testing.assert_array_almost_equal(curvature.second_derivative, expected_second_derivative)
    np.testing.assert_array_almost_equal(curvature.local_curvature, expected_local_curvature)
