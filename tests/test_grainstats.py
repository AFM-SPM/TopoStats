"""Testing of grainstats class"""
import numpy as np

# pylint: disable=protected-access

POINT1 = (0, 0)
POINT2 = (1, 0)
POINT3 = (1, 1)
POINT4 = (0, 1)
EDGES = np.array([POINT1, POINT2, POINT3, POINT4])


def test_get_angle(grainstats) -> None:
    """Test calculation of angle."""
    angle = grainstats.get_angle(POINT1, POINT3)
    target = np.arctan2(POINT1[1] - POINT3[1], POINT1[0] - POINT3[0])

    assert isinstance(angle, float)
    assert angle == target


def test_is_clockwise_clockwise(grainstats) -> None:
    """Test calculation of whether three points make a clockwise turn"""
    clockwise = grainstats.is_clockwise(POINT3, POINT2, POINT1)

    assert isinstance(clockwise, bool)
    assert clockwise


def test_is_clockwise_anti_clockwise(grainstats) -> None:
    """Test calculation of whether three points make a clockwise turn"""
    clockwise = grainstats.is_clockwise(POINT1, POINT2, POINT3)

    assert isinstance(clockwise, bool)
    assert not clockwise


def test_get_displacement(grainstats) -> None:
    """Test the calculation of displacement between two points."""
    displacement_1_2 = grainstats.get_displacement(POINT2, POINT1)
    displacement_1_3 = grainstats.get_displacement(POINT3, POINT1)
    displacement_2_3 = grainstats.get_displacement(POINT3, POINT2)

    target_1_2 = (POINT2[0] - POINT1[0]) ** 2 + (POINT2[1] - POINT1[1]) ** 2
    target_1_3 = (POINT3[0] - POINT1[0]) ** 2 + (POINT3[1] - POINT1[1]) ** 2
    target_2_3 = (POINT2[0] - POINT3[0]) ** 2 + (POINT2[1] - POINT3[1]) ** 2

    assert isinstance(displacement_1_2, float)
    assert displacement_1_2 == target_1_2
    assert isinstance(displacement_1_3, float)
    assert displacement_1_3 == target_1_3
    assert isinstance(displacement_2_3, float)
    assert displacement_2_3 == target_2_3


def test_calculate_edges(grainstats) -> None:
    """Test calculation of edges."""
    grain_mask = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    edges = grainstats.calculate_edges(grain_mask)
    target = np.array(
        [
            [1, 1],
            [1, 2],
            [1, 3],
            [1, 4],
            [1, 5],
            [1, 6],
            [2, 1],
            [2, 6],
            [3, 1],
            [3, 6],
            [4, 1],
            [4, 6],
            [5, 1],
            [5, 6],
            [6, 1],
            [6, 2],
            [6, 3],
            [6, 4],
            [6, 5],
            [6, 6],
        ]
    )
    assert isinstance(edges, list)
    assert len(edges) == 20
    assert len(edges[0]) == 2
    assert len(edges[-1]) == 2
    # assert isinstance(edges, np.ndarray)
    # assert edges.shape == (20, 2)
    np.testing.assert_array_equal(edges, target)


def test_calculate_centroid(grainstats) -> None:
    """Test calculation of centroid."""
    centroid = grainstats._calculate_centroid(EDGES)
    target = (0.5, 0.5)

    assert isinstance(centroid, tuple)
    assert centroid == target


def test_calculate_displacement(grainstats) -> None:
    """Test calculation of displacement of points from centroid."""
    centroid = grainstats._calculate_centroid(EDGES)
    displacement = grainstats._calculate_displacement(EDGES, centroid)

    target = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])

    assert isinstance(displacement, np.ndarray)
    np.testing.assert_array_equal(displacement, target)


def test_calculate_radius(grainstats) -> None:
    """Calculate the radius of each point from the centroid."""
    centroid = grainstats._calculate_centroid(EDGES)
    displacement = grainstats._calculate_displacement(EDGES, centroid)
    radii = grainstats._calculate_radius(displacement)

    target = np.array([0.7071067811865476, 0.7071067811865476, 0.7071067811865476, 0.7071067811865476])

    assert isinstance(radii, np.ndarray)
    np.testing.assert_array_equal(radii, target)
