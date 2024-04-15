"""Tests for feret functions."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from skimage import draw

from topostats.measure import feret

# pylint: disable=protected-access
# pylint: disable=too-many-lines
# pylint: disable=fixme

POINT1 = (0, 0)
POINT2 = (1, 0)
POINT3 = (1, 1)
POINT4 = (2, 0)
POINT5 = (0, 1)
POINT6 = (0, 2)

tiny_circle = np.zeros((3, 3), dtype=np.uint8)
rr, cc = draw.circle_perimeter(1, 1, 1)
tiny_circle[rr, cc] = 1

small_circle = np.zeros((5, 5), dtype=np.uint8)
rr, cc = draw.circle_perimeter(2, 2, 2)
small_circle[rr, cc] = 1

tiny_quadrilateral = np.asarray(
    [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ],
    dtype=np.uint8,
)

tiny_square = np.asarray([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]], dtype=np.uint8)

tiny_triangle = np.asarray([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]], dtype=np.uint8)

arbitrary_triangle = np.array([[1.18727719, 2.96140198], [4.14262995, 3.17983179], [6.92472119, 0.64147496]])

tiny_rectangle = np.asarray([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]], dtype=np.uint8)

tiny_ellipse = np.asarray(
    [
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ],
    dtype=np.uint8,
)

holo_circle = np.zeros((7, 7), dtype=np.uint8)
rr, cc = draw.circle_perimeter(3, 3, 2)
holo_circle[rr, cc] = 1

holo_ellipse_vertical = np.zeros((11, 9), dtype=np.uint8)
rr, cc = draw.ellipse_perimeter(5, 4, 4, 3)
holo_ellipse_vertical[rr, cc] = 1

holo_ellipse_horizontal = np.zeros((9, 11), dtype=np.uint8)
rr, cc = draw.ellipse_perimeter(4, 5, 3, 4)
holo_ellipse_horizontal[rr, cc] = 1

holo_ellipse_angled = np.zeros((8, 10), dtype=np.uint8)
rr, cc = draw.ellipse_perimeter(4, 5, 1, 3, orientation=np.deg2rad(30))
holo_ellipse_angled[rr, cc] = 1

curved_line = np.zeros((10, 10), dtype=np.uint8)
rr, cc = draw.bezier_curve(1, 5, 5, -2, 8, 8, 2)
curved_line[rr, cc] = 1

filled_circle = np.zeros((9, 9), dtype=np.uint8)
rr, cc = draw.disk((4, 4), 4)
filled_circle[rr, cc] = 1

filled_ellipse_vertical = np.zeros((9, 7), dtype=np.uint8)
rr, cc = draw.ellipse(4, 3, 4, 3)
filled_ellipse_vertical[rr, cc] = 1

filled_ellipse_horizontal = np.zeros((7, 9), dtype=np.uint8)
rr, cc = draw.ellipse(3, 4, 3, 4)
filled_ellipse_horizontal[rr, cc] = 1

filled_ellipse_angled = np.zeros((9, 11), dtype=np.uint8)
rr, cc = draw.ellipse(4, 5, 3, 5, rotation=np.deg2rad(30))
filled_ellipse_angled[rr, cc] = 1


@pytest.mark.parametrize(
    ("point1", "point2", "point3", "target"),
    [
        pytest.param(POINT3, POINT2, POINT1, 1, id="clockwise"),
        pytest.param(POINT1, POINT2, POINT3, -1, id="anti-clockwise"),
        pytest.param(POINT1, POINT2, POINT4, 0, id="vertical-line"),
        pytest.param(POINT1, POINT5, POINT6, 0, id="horizontal-line"),
    ],
)
def test_orientation(point1: tuple, point2: tuple, point3: tuple, target: int) -> None:
    """Test calculation of orientation of three points."""
    assert feret.orientation(point1, point2, point3) == target


@pytest.mark.parametrize(
    ("shape", "axis", "target"),
    [
        pytest.param(tiny_circle, 1, [[1, 0], [0, 1], [2, 1], [1, 2]], id="tiny circle sorted on axis 1"),
        pytest.param(tiny_circle, 0, [[0, 1], [1, 0], [1, 2], [2, 1]], id="tiny circle sorted on axis 0"),
        pytest.param(tiny_square, 1, [[1, 1], [2, 1], [1, 2], [2, 2]], id="tiny square sorted on axis 1"),
        pytest.param(tiny_square, 0, [[1, 1], [1, 2], [2, 1], [2, 2]], id="tiny square sorted on axis 0"),
        pytest.param(tiny_quadrilateral, 1, [[2, 1], [1, 2], [5, 2], [2, 4]], id="tiny quadrilateral sorted on axis 1"),
        pytest.param(tiny_quadrilateral, 0, [[1, 2], [2, 1], [2, 4], [5, 2]], id="tiny quadrilateral sorted on axis 0"),
    ],
)
def test_sort_coords(shape: npt.NDArray, axis: int, target: npt.NDArray) -> None:
    """Test sorting of coordinates."""
    sorted_coords = feret.sort_coords(np.argwhere(shape == 1), axis)
    np.testing.assert_array_equal(sorted_coords, target)


@pytest.mark.parametrize(
    ("coords", "axis", "target"),
    [
        pytest.param(
            arbitrary_triangle,
            0,
            [[1.187277, 2.961402], [4.14263, 3.179832], [6.924721, 0.641475]],
            id="arbitrary triangle sorted on axis 0",
        ),
        pytest.param(
            arbitrary_triangle,
            1,
            [[6.924721, 0.641475], [1.187277, 2.961402], [4.14263, 3.179832]],
            id="arbitrary triangle sorted on axis 1",
        ),
    ],
)
def test_sort_coords_arbitrary(coords: npt.NDArray, axis: int, target: npt.NDArray) -> None:
    """Test sorting of coordinates."""
    sorted_coords = feret.sort_coords(coords, axis)
    np.testing.assert_array_almost_equal(sorted_coords, target)


@pytest.mark.parametrize(
    ("shape", "axis"),
    [
        pytest.param(tiny_triangle, 2, id="integer not 0 or 1"),
        pytest.param(tiny_triangle, "row", id="string"),
    ],
)
def test_sort_coords_invalid_axis(shape: npt.NDArray, axis: int | str) -> None:
    """Test ValueError raised when axis is not 0 or 1."""
    with pytest.raises(ValueError):  # noqa: PT011
        feret.sort_coords(np.argwhere(shape == 1), axis)


@pytest.mark.parametrize(
    ("shape", "axis", "upper_target", "lower_target"),
    [
        pytest.param(
            tiny_circle, 0, [[0, 1], [1, 2], [2, 1]], [[0, 1], [1, 0], [2, 1]], id="tiny circle sorted on axis 0"
        ),
        pytest.param(tiny_circle, 1, [[1, 0], [0, 1], [1, 2]], [[1, 0], [2, 1], [1, 2]], id="tiny circle on axis 1"),
        pytest.param(
            tiny_square, 0, [[1, 1], [1, 2], [2, 2]], [[1, 1], [2, 1], [2, 2]], id="tiny square sorted on axis 0"
        ),
        pytest.param(
            tiny_square, 1, [[1, 1], [1, 2], [2, 2]], [[1, 1], [2, 1], [2, 2]], id="tiny square sorted on axis 1"
        ),
        pytest.param(tiny_triangle, 0, [[1, 1], [1, 2], [2, 1]], [[1, 1], [2, 1]], id="tiny triangle sorted on axis 0"),
        pytest.param(tiny_triangle, 1, [[1, 1], [1, 2]], [[1, 1], [2, 1], [1, 2]], id="tiny triangle sorted on axis 1"),
        pytest.param(
            tiny_rectangle, 0, [[1, 1], [1, 2], [3, 2]], [[1, 1], [3, 1], [3, 2]], id="tiny rectangle sorted on axis 0"
        ),
        pytest.param(
            tiny_rectangle, 1, [[1, 1], [1, 2], [3, 2]], [[1, 1], [3, 1], [3, 2]], id="tiny rectangle sorted on axis 1"
        ),
        pytest.param(
            tiny_ellipse,
            0,
            [[1, 2], [2, 3], [4, 3], [5, 2]],
            [[1, 2], [2, 1], [4, 1], [5, 2]],
            id="tiny ellipse sorted on axis 0",
        ),
        pytest.param(
            tiny_ellipse,
            1,
            [[2, 1], [1, 2], [2, 3], [4, 3]],
            [[2, 1], [4, 1], [5, 2], [4, 3]],
            id="tiny ellipse sorted on axis 1",
        ),
        pytest.param(
            tiny_quadrilateral,
            0,
            [[1, 2], [2, 4], [5, 2]],
            [[1, 2], [2, 1], [5, 2]],
            id="tiny quadrialteral sorted on axis 0",
        ),
        pytest.param(
            tiny_quadrilateral,
            1,
            [[2, 1], [1, 2], [2, 4]],
            [[2, 1], [5, 2], [2, 4]],
            id="tiny quadrialteral sorted on axis 1",
        ),
        pytest.param(
            small_circle,
            0,
            [[0, 1], [0, 3], [1, 4], [3, 4], [4, 3]],
            [[0, 1], [1, 0], [3, 0], [4, 1], [4, 3]],
            id="small circle sorted on axis 0",
        ),
        pytest.param(
            small_circle,
            1,
            [[1, 0], [0, 1], [0, 3], [1, 4], [3, 4]],
            [[1, 0], [3, 0], [4, 1], [4, 3], [3, 4]],
            id="small circle sorted on axis 1",
        ),
        pytest.param(
            holo_circle,
            0,
            [[1, 2], [1, 4], [2, 5], [4, 5], [5, 4]],
            [[1, 2], [2, 1], [4, 1], [5, 2], [5, 4]],
            id="holo circle sorted on axis 0",
        ),
        pytest.param(
            holo_circle,
            1,
            [[2, 1], [1, 2], [1, 4], [2, 5], [4, 5]],
            [[2, 1], [4, 1], [5, 2], [5, 4], [4, 5]],
            id="holo circle sorted on axis 1",
        ),
        pytest.param(
            holo_ellipse_horizontal,
            0,
            [[1, 3], [1, 7], [3, 9], [5, 9], [7, 7]],
            [[1, 3], [3, 1], [5, 1], [7, 3], [7, 7]],
            id="holo ellipse horizontal sorted on axis 0",
        ),
        pytest.param(
            holo_ellipse_horizontal,
            1,
            [[3, 1], [1, 3], [1, 7], [3, 9], [5, 9]],
            [[3, 1], [5, 1], [7, 3], [7, 7], [5, 9]],
            id="holo ellipse horizontal sorted on axis 1",
        ),
        pytest.param(
            holo_ellipse_vertical,
            0,
            [[1, 3], [1, 5], [3, 7], [7, 7], [9, 5]],
            [[1, 3], [3, 1], [7, 1], [9, 3], [9, 5]],
            id="holo ellipse vertical sorted on axis 0",
        ),
        pytest.param(
            holo_ellipse_vertical,
            1,
            [[3, 1], [1, 3], [1, 5], [3, 7], [7, 7]],
            [[3, 1], [7, 1], [9, 3], [9, 5], [7, 7]],
            id="holo ellipse vertical sorted on axis 1",
        ),
        pytest.param(
            holo_ellipse_angled,
            0,
            [[1, 2], [1, 4], [5, 8], [6, 7]],
            [[1, 2], [2, 1], [6, 5], [6, 7]],
            id="holo ellipse angled sorted on axis 0",
        ),
        pytest.param(
            holo_ellipse_angled,
            1,
            [[2, 1], [1, 2], [1, 4], [5, 8]],
            [[2, 1], [6, 5], [6, 7], [5, 8]],
            id="holo ellipse angled sorted on axis 1",
        ),
        pytest.param(
            curved_line,
            0,
            [[1, 5], [8, 8]],
            [[1, 5], [2, 3], [4, 1], [5, 1], [6, 2], [7, 4], [8, 7], [8, 8]],
            id="curved line sorted on axis 0",
        ),
        pytest.param(
            curved_line,
            1,
            [[4, 1], [2, 3], [1, 5], [8, 8]],
            [[4, 1], [5, 1], [6, 2], [7, 4], [8, 7], [8, 8]],
            id="curved line sorted on axis 1",
        ),
    ],
)
def test_hulls(shape: npt.NDArray, axis: bool, upper_target: list, lower_target: list) -> None:
    """Test construction of upper and lower hulls."""
    upper, lower = feret.hulls(np.argwhere(shape == 1), axis)
    np.testing.assert_array_equal(upper, upper_target)
    np.testing.assert_array_equal(lower, lower_target)


@pytest.mark.parametrize(
    ("coords", "axis", "upper_target", "lower_target"),
    [
        pytest.param(
            arbitrary_triangle,
            0,
            [[1.187277, 2.961402], [4.14263, 3.179832], [6.924721, 0.641475]],
            [[1.187277, 2.961402], [6.924721, 0.641475]],
            id="arbitrary triangle sorted on axis 0",
        ),
    ],
)
def test_hulls_arbitrary(coords: npt.NDArray, axis: bool, upper_target: list, lower_target: list) -> None:
    """Test construction of upper and lower hulls."""
    upper, lower = feret.hulls(coords, axis)
    np.testing.assert_array_almost_equal(upper, upper_target)
    np.testing.assert_array_almost_equal(lower, lower_target)


@pytest.mark.parametrize(
    ("shape", "points_target"),
    [
        pytest.param(
            tiny_circle,
            [
                ([1, 0], [2, 1]),
                ([1, 0], [1, 2]),
                ([0, 1], [1, 0]),
                ([0, 1], [2, 1]),
                ([0, 1], [1, 2]),
                ([1, 2], [2, 1]),
            ],
            id="tiny circle",
        ),
        pytest.param(
            tiny_square,
            [
                ([1, 1], [2, 1]),
                ([1, 1], [2, 2]),
                ([1, 1], [1, 2]),
                ([1, 2], [2, 1]),
                ([1, 2], [2, 2]),
                ([2, 1], [2, 2]),
            ],
            id="tiny square",
        ),
        pytest.param(
            tiny_triangle,
            [
                ([1, 1], [2, 1]),
                ([1, 1], [1, 2]),
                ([1, 2], [2, 1]),
            ],
            id="tiny triangle",
        ),
        pytest.param(
            tiny_rectangle,
            [
                ([1, 1], [3, 1]),
                ([1, 1], [3, 2]),
                ([1, 1], [1, 2]),
                ([1, 2], [3, 1]),
                ([1, 2], [3, 2]),
                ([3, 1], [3, 2]),
            ],
            id="tiny rectangle",
        ),
        pytest.param(
            tiny_ellipse,
            [
                ([2, 1], [4, 1]),
                ([2, 1], [5, 2]),
                ([2, 1], [4, 3]),
                ([1, 2], [2, 1]),
                ([1, 2], [4, 1]),
                ([1, 2], [5, 2]),
                ([1, 2], [4, 3]),
                ([2, 1], [2, 3]),
                ([2, 3], [4, 1]),
                ([2, 3], [5, 2]),
                ([2, 3], [4, 3]),
                ([4, 1], [4, 3]),
                ([4, 3], [5, 2]),
            ],
            id="tiny ellipse",
        ),
        pytest.param(
            small_circle,
            [
                ([1, 0], [3, 0]),
                ([1, 0], [4, 1]),
                ([1, 0], [4, 3]),
                ([1, 0], [3, 4]),
                ([0, 1], [1, 0]),
                ([0, 1], [3, 0]),
                ([0, 1], [4, 1]),
                ([0, 1], [4, 3]),
                ([0, 1], [3, 4]),
                ([0, 3], [1, 0]),
                ([0, 3], [3, 0]),
                ([0, 3], [4, 1]),
                ([0, 3], [4, 3]),
                ([0, 3], [3, 4]),
                ([1, 0], [1, 4]),
                ([1, 4], [3, 0]),
                ([1, 4], [4, 1]),
                ([1, 4], [4, 3]),
                ([1, 4], [3, 4]),
                ([3, 0], [3, 4]),
                ([3, 4], [4, 1]),
                ([3, 4], [4, 3]),
            ],
            id="small circle",
        ),
        pytest.param(
            holo_circle,
            [
                ([2, 1], [4, 1]),
                ([2, 1], [5, 2]),
                ([2, 1], [5, 4]),
                ([2, 1], [4, 5]),
                ([1, 2], [2, 1]),
                ([1, 2], [4, 1]),
                ([1, 2], [5, 2]),
                ([1, 2], [5, 4]),
                ([1, 2], [4, 5]),
                ([1, 4], [2, 1]),
                ([1, 4], [4, 1]),
                ([1, 4], [5, 2]),
                ([1, 4], [5, 4]),
                ([1, 4], [4, 5]),
                ([2, 1], [2, 5]),
                ([2, 5], [4, 1]),
                ([2, 5], [5, 2]),
                ([2, 5], [5, 4]),
                ([2, 5], [4, 5]),
                ([4, 1], [4, 5]),
                ([4, 5], [5, 2]),
                ([4, 5], [5, 4]),
            ],
            id="holo circle",
        ),
        pytest.param(
            holo_ellipse_horizontal,
            [
                ([3, 1], [5, 1]),
                ([3, 1], [7, 3]),
                ([3, 1], [7, 7]),
                ([3, 1], [5, 9]),
                ([1, 3], [3, 1]),
                ([1, 3], [5, 1]),
                ([1, 3], [7, 3]),
                ([1, 3], [7, 7]),
                ([1, 3], [5, 9]),
                ([1, 7], [3, 1]),
                ([1, 7], [5, 1]),
                ([1, 7], [7, 3]),
                ([1, 7], [7, 7]),
                ([1, 7], [5, 9]),
                ([3, 1], [3, 9]),
                ([3, 9], [5, 1]),
                ([3, 9], [7, 3]),
                ([3, 9], [7, 7]),
                ([3, 9], [5, 9]),
                ([5, 1], [5, 9]),
                ([5, 9], [7, 3]),
                ([5, 9], [7, 7]),
            ],
            id="holo ellipse horizontal",
        ),
        pytest.param(
            holo_ellipse_vertical,
            [
                ([3, 1], [7, 1]),
                ([3, 1], [9, 3]),
                ([3, 1], [9, 5]),
                ([3, 1], [7, 7]),
                ([1, 3], [3, 1]),
                ([1, 3], [7, 1]),
                ([1, 3], [9, 3]),
                ([1, 3], [9, 5]),
                ([1, 3], [7, 7]),
                ([1, 5], [3, 1]),
                ([1, 5], [7, 1]),
                ([1, 5], [9, 3]),
                ([1, 5], [9, 5]),
                ([1, 5], [7, 7]),
                ([3, 1], [3, 7]),
                ([3, 7], [7, 1]),
                ([3, 7], [9, 3]),
                ([3, 7], [9, 5]),
                ([3, 7], [7, 7]),
                ([7, 1], [7, 7]),
                ([7, 7], [9, 3]),
                ([7, 7], [9, 5]),
            ],
            id="holo ellipse vertical",
        ),
        pytest.param(
            holo_ellipse_angled,
            [
                ([2, 1], [6, 5]),
                ([2, 1], [6, 7]),
                ([2, 1], [5, 8]),
                ([1, 2], [2, 1]),
                ([1, 2], [6, 5]),
                ([1, 2], [6, 7]),
                ([1, 2], [5, 8]),
                ([1, 4], [2, 1]),
                ([1, 4], [6, 5]),
                ([1, 4], [6, 7]),
                ([1, 4], [5, 8]),
                ([5, 8], [6, 5]),
                ([5, 8], [6, 7]),
            ],
            id="holo ellipse angled",
        ),
        pytest.param(
            curved_line,
            [
                ([4, 1], [5, 1]),
                ([4, 1], [6, 2]),
                ([4, 1], [7, 4]),
                ([4, 1], [8, 7]),
                ([4, 1], [8, 8]),
                ([2, 3], [4, 1]),
                ([2, 3], [5, 1]),
                ([2, 3], [6, 2]),
                ([2, 3], [7, 4]),
                ([2, 3], [8, 7]),
                ([2, 3], [8, 8]),
                ([1, 5], [4, 1]),
                ([1, 5], [5, 1]),
                ([1, 5], [6, 2]),
                ([1, 5], [7, 4]),
                ([1, 5], [8, 7]),
                ([1, 5], [8, 8]),
                ([5, 1], [8, 8]),
                ([6, 2], [8, 8]),
                ([7, 4], [8, 8]),
                ([8, 7], [8, 8]),
            ],
            id="curved line",
        ),
    ],
)
def test_all_pairs(shape: npt.NDArray, points_target: list) -> None:
    """Test calculation of all pairs."""
    points = feret.all_pairs(np.argwhere(shape == 1))
    np.testing.assert_array_equal(list(points), points_target)


@pytest.mark.parametrize(
    ("base1", "base2", "apex", "target_height"),
    [
        pytest.param([0, 0], [1, 0], [0, 1], 1.0, id="tiny triangle (vertical is base)"),
        pytest.param([0, 0], [0, 1], [1, 0], 1.0, id="tiny triangle (horizontal is base)"),
        pytest.param([0, 1], [1, 0], [0, 0], 0.7071067811865475, id="tiny triangle (hypotenuse is base)"),
        pytest.param([4, 0], [4, 3], [0, 0], 4.0, id="3, 4, 5 triangle"),
        pytest.param([4, 0], [4, 6], [0, 3], 4.0, id="equilateral triangle"),
        pytest.param(
            np.asarray([4, 0]), np.asarray([4, 6]), np.asarray([0, 3]), 4.0, id="equilateral triangle (numpy arrays)"
        ),
        pytest.param([4, 3], [4, 6], [0, 0], 4.0, id="offset"),
        pytest.param([4, 3], [3, 6], [0, 0], 4.743416490252569, id="offset"),
        pytest.param(
            [1.187277, 2.961402],
            [4.14263, 3.179832],
            [6.924721, 0.641475],
            2.736517033606448,
            id="arbitrary triangle, height 1",
        ),
        pytest.param(
            [1.187277, 2.961402],
            [6.924721, 0.641475],
            [4.14263, 3.179832],
            1.3103558947980252,
            id="arbitrary triangle height 2",
        ),
        pytest.param(
            [6.924721, 0.641475],
            [4.14263, 3.179832],
            [1.187277, 2.961402],
            2.1532876859324226,
            id="arbitrary triangle height 3",
        ),
    ],
)
def test_triangle_heights(
    base1: npt.NDArray | list, base2: npt.NDArray | list, apex: npt.NDArray | list, target_height: float
) -> None:
    """Test calculation of minimum feret (height of triangle)."""
    height = feret.triangle_height(base1, base2, apex)
    np.testing.assert_almost_equal(height, target_height)


@pytest.mark.parametrize(
    ("base1", "base2", "apex", "round_coord", "opposite_target"),
    [
        pytest.param([1, 0], [0, 1], [0, 0], True, np.asarray([1, 1]), id="tiny triangle (apex top left, rounding)"),
        pytest.param([1, 0], [0, 1], [1, 1], True, np.asarray([0, 0]), id="tiny triangle (apex top right, rounding)"),
        pytest.param([0, 0], [1, 1], [1, 0], True, np.asarray([0, 1]), id="tiny triangle (apex bottom left, rounding)"),
        pytest.param(
            [0, 0], [1, 1], [0, 1], True, np.asarray([1, 0]), id="tiny triangle (apex bottom right, rounding)"
        ),
        pytest.param([1, 2], [2, 1], [1, 1], True, np.asarray([2, 2]), id="tiny triangle (from tests, rounding)"),
        pytest.param([2, 1], [8, 2], [5, 4], True, np.asarray([6, 1]), id="another triangle (rounding)"),
        pytest.param(
            [2, 1],
            [8, 2],
            [5, 4],
            False,
            np.asarray([5.405405405405405, 1.5675675675675649]),
            id="another triangle (no rounding)",
        ),
        pytest.param(
            [1, 0],
            [1, 1],
            [0, 0],
            False,
            np.asarray([1, 0]),
            id="tiny triangle with base gradient zero, apex gradient 1 (no rounding)",
        ),
        pytest.param(
            [1, 0],
            [1, 1],
            [0, 0],
            True,
            np.asarray([1, 1]),
            id="tiny triangle with base gradient zero, apex gradient 1 (rounding)",
        ),
        pytest.param(
            [1, 0],
            [1, 2],
            [0, 1],
            False,
            np.asarray([1, 1]),
            id="tiny triangle with base gradient zero, apex gradient 0.5 (rounding)",
        ),
        pytest.param(
            [1.187277, 2.961402],
            [4.14263, 3.179832],
            [6.924721, 0.641475],
            False,
            [6.723015, 3.370548],
            id="arbitrary triangle, height 1",
        ),
        pytest.param(
            [1.187277, 2.961402],
            [6.924721, 0.641475],
            [4.14263, 3.179832],
            False,
            [3.651425, 1.965027],
            id="arbitrary triangle height 2",
        ),
        pytest.param(
            [6.924721, 0.641475],
            [4.14263, 3.179832],
            [1.187277, 2.961402],
            False,
            [2.638607, 4.55209],
            id="arbitrary triangle height 3",
        ),
    ],
)
def test_min_feret_coord(
    base1: npt.NDArray | list,
    base2: npt.NDArray | list,
    apex: npt.NDArray | list,
    round_coord: bool,
    opposite_target: float,
) -> None:
    """Test calculation of mid_point of the triangle formed by rotating caliper and next point on convex hull."""
    opposite = feret._min_feret_coord(np.asarray(base1), np.asarray(base2), np.asarray(apex), round_coord)
    np.testing.assert_array_almost_equal(opposite, opposite_target)


@pytest.mark.parametrize(
    ("shape", "axis", "calipers_target", "min_ferets_target", "min_feret_coords_target"),
    [
        pytest.param(
            tiny_circle,
            0,
            (([2, 1], [0, 1]), ([1, 0], [0, 1]), ([1, 0], [1, 2]), ([0, 1], [1, 2])),
            (1.414213562373095, 1.414213562373095, 1.414213562373095, 1.414213562373095),
            ([[1.0, 0.0], [0, 1]], [[-0.0, 1.0], [1, 0]], [[0.0, 1.0], [1, 2]], [[1.0, 2.0], [0, 1]]),
            id="tiny circle sorted by axis 0",
        ),
        pytest.param(
            tiny_quadrilateral,
            0,
            (([5, 2], [1, 2]), ([5, 2], [2, 4]), ([2, 1], [2, 4]), ([2, 1], [5, 2])),
            (3.5777087639996634, 2.846049894151541, 2.4961508830135313, 2.82842712474619),
            (
                [[1.8, 3.6], [5, 2]],
                [[2.9, 1.3000000000000007], [2, 4]],
                [[3.3846153846153846, 3.0769230769230766], [2, 1]],
                [[3.0, 0.0], [5, 2]],
            ),
            id="tiny quadrilateral sorted by axis 0",
        ),
        pytest.param(
            tiny_square,
            0,
            (([2, 2], [1, 1]), ([2, 1], [1, 1]), ([2, 1], [1, 2]), ([1, 1], [1, 2])),
            (1.0, 1.0, 1.0, 1.0),
            (
                [[2.0, 1.0], [1, 1]],
                [[1.0, 1.0], [2, 1]],
                [[1.0, 1.0], [1, 2]],
                [[1.0, 2.0], [1, 1]],
            ),
            id="tiny square sorted by axis 0",
        ),
        pytest.param(
            tiny_triangle,
            0,
            (([2, 1], [1, 1]), ([2, 1], [1, 2]), ([1, 1], [1, 2])),
            (1.0, 1.0, 0.7071067811865475),
            ([[1.0, 1.0], [2, 1]], [[1.0, 1.0], [1, 2]], [[1.5, 1.5], [1, 1]]),
            id="tiny triangle sorted by axis 0",
        ),
        pytest.param(
            tiny_rectangle,
            0,
            (([3, 2], [1, 1]), ([3, 1], [1, 1]), ([3, 1], [1, 2]), ([1, 1], [1, 2])),
            (2.0, 2.0, 1.0, 1.0),
            (
                [[3.0, 1.0], [1, 1]],
                [[1.0, 1.0], [3, 1]],
                [[1.0, 1.0], [1, 2]],
                [[1.0, 2.0], [1, 1]],
            ),
            id="tiny rectangle sorted by axis 0",
        ),
        pytest.param(
            tiny_ellipse,
            0,
            (
                ([5, 2], [1, 2]),
                ([4, 1], [1, 2]),
                ([4, 1], [2, 3]),
                ([2, 1], [2, 3]),
                ([2, 1], [4, 3]),
                ([1, 2], [4, 3]),
            ),
            (2.82842712474619, 2.82842712474619, 2.0, 2.0, 2.82842712474619, 2.82842712474619),
            (
                [[3.0, 0.0], [1, 2]],
                [[2.0, 3.0], [4, 1]],
                [[2.0, 1.0], [2, 3]],
                [[2.0, 3.0], [2, 1]],
                [[2.0, 1.0], [4, 3]],
                [[3.0, 4.0], [1, 2]],
            ),
            id="tiny ellipse sorted by axis 0",
        ),
        pytest.param(
            small_circle,
            0,
            (
                ([4, 3], [0, 1]),
                ([4, 1], [0, 1]),
                ([4, 1], [0, 3]),
                ([3, 0], [0, 3]),
                ([3, 0], [1, 4]),
                ([1, 0], [1, 4]),
                ([1, 0], [3, 4]),
                ([0, 1], [3, 4]),
            ),
            (4.0, 4.0, 4.242640687119285, 4.242640687119285, 4.0, 4.0, 4.242640687119285, 4.242640687119285),
            (
                [[4.0, 1.0], [0, 1]],
                [[0.0, 1.0], [4, 1]],
                [[3.0, 0.0], [0, 3]],
                [[0.0, 3.0], [3, 0]],
                [[1.0, 0.0], [1, 4]],
                [[1.0, 4.0], [1, 0]],
                [[0.0, 1.0], [3, 4]],
                [[3.0, 4.0], [0, 1]],
            ),
            id="small circle sorted by axis 0",
        ),
        pytest.param(
            holo_circle,
            0,
            (
                ([5, 4], [1, 2]),
                ([5, 2], [1, 2]),
                ([5, 2], [1, 4]),
                ([4, 1], [1, 4]),
                ([4, 1], [2, 5]),
                ([2, 1], [2, 5]),
                ([2, 1], [4, 5]),
                ([1, 2], [4, 5]),
            ),
            (4.0, 4.0, 4.242640687119285, 4.242640687119285, 4.0, 4.0, 4.242640687119285, 4.242640687119285),
            (
                [[5.0, 2.0], [1, 2]],
                [[1.0, 2.0], [5, 2]],
                [[4.0, 1.0], [1, 4]],
                [[1.0, 4.0], [4, 1]],
                [[2.0, 1.0], [2, 5]],
                [[2.0, 5.0], [2, 1]],
                [[1.0, 2.0], [4, 5]],
                [[4.0, 5.0], [1, 2]],
            ),
            id="holo circle sorted by axis 0",
        ),
        pytest.param(
            holo_ellipse_horizontal,
            0,
            (
                ([7, 7], [1, 3]),
                ([7, 3], [1, 3]),
                ([7, 3], [1, 7]),
                ([5, 1], [1, 7]),
                ([5, 1], [3, 9]),
                ([3, 1], [3, 9]),
                ([3, 1], [5, 9]),
                ([1, 3], [5, 9]),
            ),
            (6.0, 6.0, 7.071067811865475, 7.071067811865475, 8.0, 8.0, 7.071067811865475, 7.071067811865475),
            (
                [[7.0, 3.0], [1, 3]],
                [[1.0, 3.0], [7, 3]],
                [[6.0, 2.0], [1, 7]],
                [[0.0, 6.0], [5, 1]],
                [[3.0, 1.0], [3, 9]],
                [[3.0, 9.0], [3, 1]],
                [[0.0, 4.0], [5, 9]],
                [[6.0, 8.0], [1, 3]],
            ),
            id="holo ellipse horizontal sorted by axis 0",
        ),
        pytest.param(
            holo_ellipse_vertical,
            0,
            (
                ([9, 5], [1, 3]),
                ([9, 3], [1, 3]),
                ([9, 3], [1, 5]),
                ([7, 1], [1, 5]),
                ([7, 1], [3, 7]),
                ([3, 1], [3, 7]),
                ([3, 1], [7, 7]),
                ([1, 3], [7, 7]),
            ),
            (8.0, 8.0, 7.071067811865475, 7.071067811865475, 6.0, 6.0, 7.071067811865475, 7.071067811865475),
            (
                [[9.0, 3.0], [1, 3]],
                [[1.0, 3.0], [9, 3]],
                [[6.0, 0.0], [1, 5]],
                [[2.0, 6.0], [7, 1]],
                [[3.0, 1.0], [3, 7]],
                [[3.0, 7.0], [3, 1]],
                [[2.0, 2.0], [7, 7]],
                [[6.0, 8.0], [1, 3]],
            ),
            id="holo ellipse vertical sorted by axis 0",
        ),
        pytest.param(
            holo_ellipse_angled,
            0,
            (
                ([6, 7], [1, 2]),
                ([6, 5], [1, 2]),
                ([6, 5], [1, 4]),
                ([2, 1], [1, 4]),
                ([2, 1], [5, 8]),
                ([1, 2], [5, 8]),
            ),
            (5.0, 5.0, 2.82842712474619, 2.82842712474619, 7.071067811865475, 7.071067811865475),
            (
                [[6.0, 2.0], [1, 2]],
                [[1.0, 5.0], [6, 5]],
                [[3.0, 2.0], [1, 4]],
                [[0.0, 3.0], [2, 1]],
                [[0.0, 3.0], [5, 8]],
                [[6.0, 7.0], [1, 2]],
            ),
            id="holo ellipse angled sorted by axis 0",
        ),
        pytest.param(
            curved_line,
            0,
            (
                ([8, 8], [1, 5]),
                ([8, 7], [1, 5]),
                ([7, 4], [1, 5]),
                ([6, 2], [1, 5]),
                ([5, 1], [1, 5]),
                ([5, 1], [8, 8]),
                ([4, 1], [8, 8]),
                ([2, 3], [8, 8]),
            ),
            (
                7.0,
                6.00832755431992,
                5.813776741499453,
                5.65685424949238,
                5.252257314388902,
                7.0,
                7.7781745930520225,
                7.602631123499284,
            ),
            (
                [[8.0, 5.0], [1, 5]],
                [[6.7, 3.1], [1, 5]],
                [[6.2, 2.4], [1, 5]],
                [[5.0, 1.0], [1, 5]],
                [[2.9310344827586214, 5.827586206896551], [5, 1]],
                [[8.0, 1.0], [8, 8]],
                [[2.5, 2.5], [8, 8]],
                [[1.2, 4.6], [8, 8]],
            ),
            id="curved line sorted by axis 0",
        ),
    ],
)
def test_rotating_calipers(
    shape: npt.NDArray, axis: int, calipers_target: tuple, min_ferets_target: tuple, min_feret_coords_target: tuple
) -> None:
    """Test calculation of rotating caliper pairs."""
    caliper_min_feret = feret.rotating_calipers(np.argwhere(shape == 1), axis)
    min_ferets, calipers, min_feret_coords = zip(*caliper_min_feret)
    np.testing.assert_array_almost_equal(calipers, calipers_target)
    np.testing.assert_array_almost_equal(min_ferets, min_ferets_target)
    np.testing.assert_array_almost_equal(min_feret_coords, min_feret_coords_target)


@pytest.mark.parametrize(
    ("coords", "axis", "calipers_target", "min_ferets_target", "min_feret_coords_target"),
    [
        pytest.param(
            arbitrary_triangle,
            0,
            (
                ([6.92472119, 0.64147496], [1.18727719, 2.96140198]),
                ([6.92472119, 0.64147496], [4.14262995, 3.17983179]),
                ([1.18727719, 2.96140198], [4.14262995, 3.17983179]),
            ),
            (2.7365167317626233, 1.3103556366489382, 2.153287228471869),
            (
                [[6.7230157, 3.37054783], [6.92472119, 0.64147496]],
                [[3.65142552, 1.96502724], [4.14262995, 3.17983179]],
                [[2.63860725, 4.55208955], [1.18727719, 2.96140198]],
            ),
            id="arbitrary triangle sorted by axis 0",
        ),
    ],
)
def test_rotating_calipers_arbitrary(
    coords: npt.NDArray, axis: int, calipers_target: tuple, min_ferets_target: tuple, min_feret_coords_target: tuple
) -> None:
    """Test calculation of rotating caliper pairs."""
    caliper_min_feret = feret.rotating_calipers(coords, axis)
    min_ferets, calipers, min_feret_coords = zip(*caliper_min_feret)
    np.testing.assert_array_almost_equal(calipers, calipers_target)
    np.testing.assert_array_almost_equal(min_ferets, min_ferets_target)
    np.testing.assert_array_almost_equal(min_feret_coords, min_feret_coords_target)


@pytest.mark.parametrize(
    ("base1", "base2", "apex", "target_angle"),
    [
        pytest.param(np.asarray([1, 1]), np.asarray([1, 0]), np.asarray([0, 0]), 45.0, id="45-degree"),
        pytest.param(np.asarray([1, 3]), np.asarray([1, 0]), np.asarray([0, 0]), 18.434948822922017, id="18.43-degree"),
        pytest.param(np.asarray([1, 4]), np.asarray([1, 0]), np.asarray([0, 0]), 14.036243467926484, id="-degree"),
    ],
)
def test_angle_between(base1: npt.NDArray, base2: npt.NDArray, apex: npt.NDArray, target_angle: float) -> None:
    """Test calculation of angle between base and apex."""
    angle = feret._angle_between(apex - base1, base2 - base1)
    assert np.rad2deg(angle) == pytest.approx(target_angle)


@pytest.mark.parametrize(
    ("coordinates", "target"),
    [
        pytest.param(
            np.asarray([[0, 0], [0, 5], [5, 0], [5, 5]]),
            ([[0, 0], [0, 5], [5, 5], [5, 0]]),
            id="Simple square Top Left > Top Right > Bottom Left > Bottom Right",
        ),
        pytest.param(
            np.asarray([[1, 1], [1, 0], [0, 0]]),
            ([0, 0], [1, 1], [1, 0]),
            id="Simple Triangle Bottom Right > Bottom Left > Apex",
        ),
        pytest.param(
            np.argwhere(holo_ellipse_angled == 1),
            ([2, 1], [1, 2], [1, 4], [5, 8], [6, 7], [6, 5]),
            id="Angled ellipse.",
        ),
        pytest.param(
            np.argwhere(holo_ellipse_horizontal == 1),
            ([3, 1], [1, 3], [1, 7], [3, 9], [5, 9], [7, 7], [7, 3], [5, 1]),
            id="Horizontal ellipse.",
        ),
        pytest.param(
            np.argwhere(holo_ellipse_vertical == 1),
            ([3, 1], [1, 3], [1, 5], [3, 7], [7, 7], [9, 5], [9, 3], [7, 1]),
            id="Vertical ellipse.",
        ),
        pytest.param(
            np.argwhere(curved_line == 1),
            ([5, 1], [4, 1], [2, 3], [1, 5], [8, 8], [8, 7], [7, 4], [6, 2]),
            id="Curved line.",
        ),
        pytest.param(
            arbitrary_triangle,
            ([[1.18727719, 2.96140198], [4.14262995, 3.17983179], [6.92472119, 0.64147496]]),
            id="Arbitrary triangle",
        ),
    ],
)
def test_sort_clockwise(coordinates: npt.NDArray, target: npt.NDArray) -> None:
    """Test sorting of coordinates in a clockwise direction."""
    upper_hull, lower_hull = feret.hulls(coordinates)
    hull = np.unique(np.concatenate([lower_hull, upper_hull], axis=0), axis=0)
    np.testing.assert_array_equal(feret.sort_clockwise(hull), target)


@pytest.mark.parametrize(
    (
        "shape",
        "axis",
        "min_feret_distance_target",
        "min_feret_coord_target",
        "max_feret_distance_target",
        "max_feret_coord_target",
    ),
    [
        pytest.param(
            tiny_circle,
            0,
            1.4142135623730951,
            ([0, 1], [1, 0]),
            2.0,
            ([2, 1], [0, 1]),
            id="tiny circle sorted on axis 0",
        ),
        pytest.param(
            tiny_circle,
            1,
            1.4142135623730951,
            ([0, 1], [1, 0]),
            2.0,
            ([2, 1], [0, 1]),
            id="tiny circle sorted on axis 1",
        ),
        pytest.param(
            tiny_square,
            0,
            1.0,
            ([1, 1], [1, 2]),
            1.4142135623730951,
            ([2, 2], [1, 1]),
            id="tiny square sorted on axis 0",
        ),
        pytest.param(
            tiny_quadrilateral,
            0,
            2.4961508830135313,
            ([3.384615384615385, 3.0769230769230766], [2, 1]),
            4.0,
            ([5, 2], [1, 2]),
            id="tiny quadrilateral sorted on axis 0",
        ),
        pytest.param(
            tiny_quadrilateral,
            1,
            2.4961508830135313,
            ([3.384615384615385, 3.0769230769230766], [2, 1]),
            4.0,
            ([5, 2], [1, 2]),
            id="tiny quadrilateral sorted on axis 1",
        ),
        pytest.param(
            tiny_triangle,
            0,
            0.7071067811865475,
            ([1.5, 1.5], [1, 1]),
            1.4142135623730951,
            ([2, 1], [1, 2]),
            id="tiny triangle sorted on axis 0",
        ),
        pytest.param(
            tiny_rectangle,
            0,
            1.0,
            ([1, 1], [1, 2]),
            2.23606797749979,
            ([3, 2], [1, 1]),
            id="tiny rectangle sorted on axis 0",
        ),
        pytest.param(tiny_ellipse, 0, 2.0, ([2, 1], [2, 3]), 4.0, ([5, 2], [1, 2]), id="tiny ellipse sorted on axis 0"),
        pytest.param(
            small_circle,
            1,
            4.0,
            ([0, 1], [4, 1]),
            4.47213595499958,
            ([4, 3], [0, 1]),
            id="small circle sorted on axis 0",
        ),
        pytest.param(
            holo_circle,
            0,
            4.0,
            ([1, 2], [5, 2]),
            4.47213595499958,
            ([5, 4], [1, 2]),
            id="holo circle sorted on axis 0",
        ),
        pytest.param(
            holo_ellipse_horizontal,
            0,
            6.0,
            ([1, 3], [7, 3]),
            8.246211251235321,
            ([5, 1], [3, 9]),
            id="holo ellipse horizontal on axis 0",
        ),
        pytest.param(
            holo_ellipse_vertical,
            0,
            6.0,
            ([3, 1], [3, 7]),
            8.246211251235321,
            ([9, 5], [1, 3]),
            id="holo ellipse vertical on axis 0",
        ),
        pytest.param(
            holo_ellipse_angled,
            0,
            2.82842712474619,
            ([0, 3], [2, 1]),
            7.615773105863909,
            ([2, 1], [5, 8]),
            id="holo ellipse angled on axis 0",
        ),
        pytest.param(
            curved_line,
            0,
            5.252257314388902,
            ([2.93103448275862, 5.827586206896552], [5, 1]),
            8.06225774829855,
            ([4, 1], [8, 8]),
            id="curved line sorted on axis 0",
        ),
        pytest.param(
            curved_line,
            1,
            5.252257314388902,
            ([2.93103448275862, 5.827586206896552], [5, 1]),
            8.06225774829855,
            ([8, 8], [4, 1]),
            id="curved line sorted on axis 1",
        ),
    ],
)
def test_min_max_feret(
    shape: npt.NDArray,
    axis: int,
    min_feret_distance_target: float,
    min_feret_coord_target: list,
    max_feret_distance_target: float,
    max_feret_coord_target: list,
) -> None:
    """Test calculation of min/max feret."""
    feret_statistics = feret.min_max_feret(np.argwhere(shape == 1), axis)
    np.testing.assert_approx_equal(feret_statistics["min_feret"], min_feret_distance_target)
    np.testing.assert_array_almost_equal(feret_statistics["min_feret_coords"], min_feret_coord_target)
    np.testing.assert_approx_equal(feret_statistics["max_feret"], max_feret_distance_target)
    np.testing.assert_array_almost_equal(feret_statistics["max_feret_coords"], max_feret_coord_target)


@pytest.mark.parametrize(
    (
        "shape",
        "axis",
        "min_feret_distance_target",
        "min_feret_coord_target",
        "max_feret_distance_target",
        "max_feret_coord_target",
    ),
    [
        pytest.param(
            filled_circle,
            0,
            6.0,
            ([1.0, 2.0], [7.0, 2.0]),
            7.211102550927978,
            ([7, 6], [1, 2]),
            id="filled circle sorted on axis 0",
        ),
        pytest.param(
            filled_ellipse_horizontal,
            0,
            4.0,
            ([1.0, 2.0], [5.0, 2.0]),
            6.324555320336759,
            ([4, 1], [2, 7]),
            id="filled ellipse horizontal sorted on axis 0",
        ),
        pytest.param(
            filled_ellipse_vertical,
            0,
            4.0,
            ([2.0, 1.0], [2.0, 5.0]),
            6.324555320336759,
            ([7, 4], [1, 2]),
            id="filled ellipse vertical sorted on axis 0",
        ),
        pytest.param(
            filled_ellipse_angled,
            0,
            5.366563145999495,
            ([1.2, 4.6], [6.0, 7.0]),
            8.94427190999916,
            ([6, 1], [2, 9]),
            id="filled ellipse angled sorted on axis 0",
        ),
    ],
)
def test_get_feret_from_mask(
    shape: npt.NDArray,
    axis: int,
    min_feret_distance_target: float,
    min_feret_coord_target: list,
    max_feret_distance_target: float,
    max_feret_coord_target: list,
) -> None:
    """Test calculation of min/max feret for a single masked object."""
    feret_statistics = feret.get_feret_from_mask(shape, axis)
    np.testing.assert_approx_equal(feret_statistics["min_feret"], min_feret_distance_target)
    np.testing.assert_array_almost_equal(feret_statistics["min_feret_coords"], min_feret_coord_target)
    np.testing.assert_approx_equal(feret_statistics["max_feret"], max_feret_distance_target)
    np.testing.assert_array_almost_equal(feret_statistics["max_feret_coords"], max_feret_coord_target)


# Concatenate images to have two labeled objects within them
holo_ellipse_angled2 = holo_ellipse_angled.copy()
holo_ellipse_angled2[holo_ellipse_angled2 == 1] = 2
# Need to pad the holo_circle
holo_image = np.concatenate((np.pad(holo_circle, pad_width=((0, 1), (0, 3))), holo_ellipse_angled2))
filled_ellipse_angled2 = filled_ellipse_angled.copy()
filled_ellipse_angled2[filled_ellipse_angled2 == 1] = 2
filled_image = np.concatenate((np.pad(filled_circle, pad_width=((0, 0), (0, 2))), filled_ellipse_angled2))


@pytest.mark.parametrize(
    ("shape", "axis", "target"),
    [
        pytest.param(
            holo_image,
            0,
            {
                1: {
                    "min_feret": 4.0,
                    "min_feret_coords": ([1.0, 2.0], [5.0, 2.0]),
                    "max_feret": 4.47213595499958,
                    "max_feret_coords": ([5, 4], [1, 2]),
                },
                2: {
                    "min_feret": 2.82842712474619,
                    "min_feret_coords": ([8.0, 3.0], [10.0, 1.0]),
                    "max_feret": 7.615773105863909,
                    "max_feret_coords": ([10, 1], [13, 8]),
                },
            },
            id="holo image",
        ),
        pytest.param(
            filled_image,
            0,
            {
                1: {
                    "min_feret": 6.0,
                    "min_feret_coords": ([1.0, 2.0], [7.0, 2.0]),
                    "max_feret": 7.211102550927978,
                    "max_feret_coords": ([7, 6], [1, 2]),
                },
                2: {
                    "min_feret": 5.366563145999495,
                    "min_feret_coords": ([10.2, 4.6], [15, 7]),
                    "max_feret": 8.94427190999916,
                    "max_feret_coords": ([15, 1], [11, 9]),
                },
            },
            id="filled image",
        ),
    ],
)
def test_get_feret_from_labelim(shape: npt.NDArray, axis: int, target: dict) -> None:
    """Test calculation of min/max feret for a labelled image with multiple objects."""
    min_max_feret_size_coord = feret.get_feret_from_labelim(shape, axis=axis)
    for key, feret_statistics in min_max_feret_size_coord.items():
        np.testing.assert_equal(feret_statistics["min_feret"], target[key]["min_feret"])
        np.testing.assert_array_almost_equal(feret_statistics["min_feret_coords"], target[key]["min_feret_coords"])
        np.testing.assert_equal(feret_statistics["max_feret"], target[key]["max_feret"])
        np.testing.assert_array_almost_equal(feret_statistics["max_feret_coords"], target[key]["max_feret_coords"])


@pytest.mark.parametrize(
    (
        "shape",
        "axis",
        "plot_points",
        "plot_hulls",
        "plot_calipers",
        "plot_triangle_heights",
        "plot_min_feret",
        "plot_max_feret",
    ),
    [
        pytest.param(tiny_quadrilateral, 0, "k", ("g-", "r-"), "y-", "b:", "m--", "m--", id="Plot everything"),
        pytest.param(tiny_quadrilateral, 0, None, ("g-", "r-"), "y-", "b:", "m--", "m--", id="Exclude points"),
        pytest.param(tiny_quadrilateral, 0, "k", None, "y-", "b:", "m--", "m--", id="Exclude hull"),
        pytest.param(tiny_quadrilateral, 0, "k", ("g-", "r-"), None, "b:", "m--", "m--", id="Exclude calipers"),
        pytest.param(tiny_quadrilateral, 0, "k", ("g-", "r-"), "y-", None, "m--", "m--", id="Exclude triangle heights"),
        pytest.param(tiny_quadrilateral, 0, "k", ("g-", "r-"), "y-", "b:", None, "m--", id="Exclude min feret"),
        pytest.param(tiny_quadrilateral, 0, "k", ("g-", "r-"), "y-", "b:", "m--", None, id="Exclude max feret"),
    ],
)
@pytest.mark.mpl_image_compare(baseline_dir="../resources/img/feret")
def test_plot_feret(  # pylint: disable=too-many-arguments
    shape: npt.NDArray,
    axis: int,
    plot_points: str | None,
    plot_hulls: tuple | None,
    plot_calipers: str | None,
    plot_triangle_heights: str | None,
    plot_min_feret: str | None,
    plot_max_feret: str | None,
) -> None:
    """Tests the plotting function used for investigating whether feret distances are correct."""
    fig, _ = feret.plot_feret(
        np.argwhere(shape == 1),
        axis,
        plot_points,
        plot_hulls,
        plot_calipers,
        plot_triangle_heights,
        plot_min_feret,
        plot_max_feret,
    )
    return fig
