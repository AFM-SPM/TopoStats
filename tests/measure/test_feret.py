"""Tests for feret functions."""

import numpy as np
import numpy.typing as npt
import pytest
from skimage import draw

from topostats.measure import feret

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

tiny_square = np.asarray([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]], dtype=np.uint8)

tiny_triangle = np.asarray([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]], dtype=np.uint8)

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
    ("shape", "upper_target", "lower_target"),
    [
        pytest.param(tiny_circle, [[1, 0], [0, 1], [1, 2]], [[1, 0], [2, 1], [1, 2]], id="tiny circle"),
        pytest.param(tiny_square, [[1, 1], [1, 2], [2, 2]], [[1, 1], [2, 1], [2, 2]], id="tiny square"),
        pytest.param(tiny_triangle, [[1, 1], [1, 2]], [[1, 1], [2, 1], [1, 2]], id="tiny triangle"),
        pytest.param(tiny_rectangle, [[1, 1], [1, 2], [3, 2]], [[1, 1], [3, 1], [3, 2]], id="tiny rectangle"),
        pytest.param(
            tiny_ellipse, [[2, 1], [1, 2], [2, 3], [4, 3]], [[2, 1], [4, 1], [5, 2], [4, 3]], id="tiny ellipse"
        ),
        pytest.param(
            small_circle,
            [[1, 0], [0, 1], [0, 3], [1, 4], [3, 4]],
            [[1, 0], [3, 0], [4, 1], [4, 3], [3, 4]],
            id="small circle",
        ),
        pytest.param(
            holo_circle,
            [[2, 1], [1, 2], [1, 4], [2, 5], [4, 5]],
            [[2, 1], [4, 1], [5, 2], [5, 4], [4, 5]],
            id="hol circle",
        ),
        pytest.param(
            holo_ellipse_horizontal,
            [[3, 1], [1, 3], [1, 7], [3, 9], [5, 9]],
            [[3, 1], [5, 1], [7, 3], [7, 7], [5, 9]],
            id="holo ellipse horizontal",
        ),
        pytest.param(
            holo_ellipse_vertical,
            [[3, 1], [1, 3], [1, 5], [3, 7], [7, 7]],
            [[3, 1], [7, 1], [9, 3], [9, 5], [7, 7]],
            id="holo ellipse vertical",
        ),
        pytest.param(
            holo_ellipse_angled,
            [[2, 1], [1, 2], [1, 4], [5, 8]],
            [[2, 1], [6, 5], [6, 7], [5, 8]],
            id="holo ellipse angled",
        ),
        pytest.param(
            curved_line,
            [[4, 1], [2, 3], [1, 5], [8, 8]],
            [[4, 1], [5, 1], [6, 2], [7, 4], [8, 7], [8, 8]],
            id="curved line",
        ),
    ],
)
def test_hulls(shape: npt.NDArray, upper_target: list, lower_target: list) -> None:
    """Test construction of upper and lower hulls."""
    upper, lower = feret.hulls(np.argwhere(shape == 1))
    np.testing.assert_array_equal(upper, upper_target)
    np.testing.assert_array_equal(lower, lower_target)


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
    ("shape", "points_target"),
    [
        # pytest.param(
        #     tiny_circle,
        #     [([0, 1], [2, 1]), ([0, 1], [1, 0]), ([1, 2], [1, 0]), ([1, 2], [0, 1]), ([2, 1], [0, 1])],
        #     id="tiny circle",
        # ),
        # pytest.param(
        #     tiny_square,
        #     [([1, 1], [2, 2]), ([1, 1], [2, 1]), ([1, 2], [2, 1]), ([1, 2], [1, 1]), ([2, 2], [1, 1])],
        #     id="tiny square",
        # ),
        # pytest.param(
        #     tiny_triangle,
        #     [([1, 1], [2, 1]), ([1, 2], [2, 1]), ([1, 2], [1, 1]), ([2, 1], [1, 1])],
        #     id="tiny triangle",
        # ),
        # pytest.param(
        #     tiny_rectangle,
        #     [([1, 1], [3, 2]), ([1, 1], [3, 1]), ([1, 2], [3, 1]), ([1, 2], [1, 1]), ([3, 2], [1, 1])],
        #     id="tiny rectangle",
        # ),
        # pytest.param(
        #     tiny_ellipse,
        #     [
        #         ([1, 2], [5, 2]),
        #         ([1, 2], [4, 1]),
        #         ([2, 3], [4, 1]),
        #         ([2, 3], [2, 1]),
        #         ([4, 3], [2, 1]),
        #         ([4, 3], [1, 2]),
        #         ([5, 2], [1, 2]),
        #     ],
        #     id="tiny ellipse",
        # ),
        # pytest.param(
        #     small_circle,
        #     [
        #         ([0, 1], [4, 3]),
        #         ([0, 1], [4, 1]),
        #         ([0, 3], [4, 1]),
        #         ([0, 3], [3, 0]),
        #         ([1, 4], [3, 0]),
        #         ([1, 4], [1, 0]),
        #         ([3, 4], [1, 0]),
        #         ([3, 4], [0, 1]),
        #         ([4, 3], [0, 1]),
        #     ],
        #     id="small circle",
        # ),
        # pytest.param(
        #     holo_circle,
        #     [
        #         ([1, 2], [5, 4]),
        #         ([1, 2], [5, 2]),
        #         ([1, 4], [5, 2]),
        #         ([1, 4], [4, 1]),
        #         ([2, 5], [4, 1]),
        #         ([2, 5], [2, 1]),
        #         ([4, 5], [2, 1]),
        #         ([4, 5], [1, 2]),
        #         ([5, 4], [1, 2]),
        #     ],
        #     id="holo circle",
        # ),
        # pytest.param(
        #     holo_ellipse_horizontal,
        #     [
        #         ([1, 3], [7, 7]),
        #         ([1, 3], [7, 3]),
        #         ([1, 7], [7, 3]),
        #         ([1, 7], [5, 1]),
        #         ([3, 9], [5, 1]),
        #         ([3, 9], [3, 1]),
        #         ([5, 9], [3, 1]),
        #         ([5, 9], [1, 3]),
        #         ([7, 7], [1, 3]),
        #     ],
        #     id="holo ellipse horizontal",
        # ),
        # pytest.param(
        #     holo_ellipse_vertical,
        #     [
        #         ([1, 3], [9, 5]),
        #         ([1, 3], [9, 3]),
        #         ([1, 5], [9, 3]),
        #         ([1, 5], [7, 1]),
        #         ([3, 7], [7, 1]),
        #         ([3, 7], [3, 1]),
        #         ([7, 7], [3, 1]),
        #         ([7, 7], [1, 3]),
        #         ([9, 5], [1, 3]),
        #     ],
        #     id="holo ellipse vertical",
        # ),
        # pytest.param(
        #     holo_ellipse_angled,
        #     [
        #         ([1, 2], [6, 7]),
        #         ([1, 2], [6, 5]),
        #         ([1, 4], [6, 5]),
        #         ([1, 4], [2, 1]),
        #         ([5, 8], [2, 1]),
        #         ([5, 8], [1, 2]),
        #         ([6, 7], [1, 2]),
        #     ],
        #     id="holo ellipse angled",
        # ),
        pytest.param(
            curved_line,
            [
                ([1, 5], [8, 8]),
                ([1, 5], [8, 7]),
                ([1, 5], [7, 4]),
                ([1, 5], [6, 2]),
                ([1, 5], [5, 1]),
                ([1, 5], [4, 1]),
                ([1, 5], [2, 3]),
                ([1, 5], [1, 5]),
                ([8, 8], [1, 5]),
            ],
            id="curved line",
        ),
    ],
)
def test_rotating_calipers(shape: npt.NDArray, points_target: list) -> None:
    """Test calculation of rotating caliper pairs."""
    points = feret.rotating_calipers(np.argwhere(shape == 1))
    np.testing.assert_array_equal(list(points), points_target)


@pytest.mark.parametrize(
    (
        "shape",
        "min_feret_distance_target",
        "min_feret_coord_target",
        "max_feret_distance_target",
        "max_feret_coord_target",
    ),
    [
        pytest.param(tiny_circle, 1.4142135623730951, ([0, 1], [1, 0]), 2, ([2, 1], [0, 1]), id="tiny circle"),
        pytest.param(tiny_square, 1.0, ([1, 1], [2, 1]), 1.4142135623730951, ([2, 2], [1, 1]), id="tiny square"),
        pytest.param(tiny_triangle, 1.0, ([1, 1], [2, 1]), 1.4142135623730951, ([1, 2], [2, 1]), id="tiny triangle"),
        pytest.param(tiny_rectangle, 1.0, ([1, 2], [1, 1]), 2.23606797749979, ([3, 2], [1, 1]), id="tiny rectangle"),
        pytest.param(tiny_ellipse, 2.0, ([2, 3], [2, 1]), 4.0, ([5, 2], [1, 2]), id="tiny ellipse"),
        pytest.param(small_circle, 4.0, ([0, 1], [4, 1]), 4.47213595499958, ([4, 3], [0, 1]), id="small circle"),
        pytest.param(holo_circle, 4.0, ([1, 2], [5, 2]), 4.47213595499958, ([5, 4], [1, 2]), id="holo circle"),
        pytest.param(
            holo_ellipse_horizontal,
            6.0,
            ([1, 3], [7, 3]),
            8.246211251235321,
            ([5, 9], [3, 1]),
            id="holo ellipse horizontal",
        ),
        pytest.param(
            holo_ellipse_vertical,
            6.0,
            ([3, 7], [3, 1]),
            8.246211251235321,
            ([9, 5], [1, 3]),
            id="holo ellipse vertical",
        ),
        pytest.param(
            holo_ellipse_angled,
            3.1622776601683795,
            ([1, 4], [2, 1]),
            7.615773105863909,
            ([5, 8], [2, 1]),
            id="holo ellipse angled",
        ),
        # (curved_line, 5.656854249492381, ([1, 5], [5, 1]), 8.06225774829855, ([8, 8], [4, 1]), id="curved line"),
    ],
)
def test_min_max_feret(
    shape: npt.NDArray,
    min_feret_distance_target: float,
    min_feret_coord_target: list,
    max_feret_distance_target: float,
    max_feret_coord_target: list,
) -> None:
    """Test calculation of min/max feret."""
    min_feret_distance, min_feret_coord, max_feret_distance, max_feret_coord = feret.min_max_feret(
        np.argwhere(shape == 1)
    )
    assert min_feret_distance == min_feret_distance_target
    assert min_feret_coord == min_feret_coord_target
    assert max_feret_distance == max_feret_distance_target
    assert max_feret_coord == max_feret_coord_target


@pytest.mark.parametrize(
    (
        "shape",
        "min_feret_distance_target",
        "min_feret_coord_target",
        "max_feret_distance_target",
        "max_feret_coord_target",
    ),
    [
        pytest.param(tiny_circle, 1.4142135623730951, ([0, 1], [1, 0]), 2, ([2, 1], [0, 1]), id="tiny circle"),
        pytest.param(tiny_square, 1.0, ([1, 1], [2, 1]), 1.4142135623730951, ([2, 2], [1, 1]), id="tiny square"),
        pytest.param(tiny_triangle, 1.0, ([1, 1], [2, 1]), 1.4142135623730951, ([1, 2], [2, 1]), id="tiny triangle"),
        pytest.param(tiny_rectangle, 1.0, ([1, 2], [1, 1]), 2.23606797749979, ([3, 2], [1, 1]), id="tiny rectangle"),
        pytest.param(tiny_ellipse, 2.0, ([2, 3], [2, 1]), 4.0, ([5, 2], [1, 2]), id="tiny ellipse"),
        pytest.param(small_circle, 4.0, ([0, 1], [4, 1]), 4.47213595499958, ([4, 3], [0, 1]), id="small circle"),
        pytest.param(holo_circle, 4.0, ([1, 2], [5, 2]), 4.47213595499958, ([5, 4], [1, 2]), id="holo circle"),
        pytest.param(
            holo_ellipse_horizontal,
            6.0,
            ([1, 3], [7, 3]),
            8.246211251235321,
            ([5, 9], [3, 1]),
            id="holo ellipse horizontal",
        ),
        pytest.param(
            holo_ellipse_vertical,
            6.0,
            ([3, 7], [3, 1]),
            8.246211251235321,
            ([9, 5], [1, 3]),
            id="holo ellipse vertical",
        ),
        pytest.param(
            holo_ellipse_angled,
            3.1622776601683795,
            ([1, 4], [2, 1]),
            7.615773105863909,
            ([5, 8], [2, 1]),
            id="holo ellipse angled",
        ),
        # (curved_line, 5.656854249492381, ([1, 5], [5, 1]), 8.06225774829855, ([8, 8], [4, 1]), id="curved line"),
        pytest.param(filled_circle, 6.0, ([1, 2], [7, 2]), 7.211102550927978, ([7, 6], [1, 2]), id="filled circle"),
        pytest.param(
            filled_ellipse_horizontal,
            4.0,
            ([1, 2], [5, 2]),
            6.324555320336759,
            ([4, 7], [2, 1]),
            id="filled ellipse horizontal",
        ),
        pytest.param(
            filled_ellipse_vertical,
            4.0,
            ([2, 5], [2, 1]),
            6.324555320336759,
            ([7, 4], [1, 2]),
            id="filled ellipse vertical",
        ),
        pytest.param(
            filled_ellipse_angled,
            5.385164807134504,
            ([6, 7], [1, 5]),
            8.94427190999916,
            ([2, 9], [6, 1]),
            id="filled ellipse angled",
        ),
    ],
)
def test_get_feret_from_mask(
    shape: npt.NDArray,
    min_feret_distance_target: float,
    min_feret_coord_target: list,
    max_feret_distance_target: float,
    max_feret_coord_target: list,
) -> None:
    """Test calculation of min/max feret for a single masked object."""
    min_feret_distance, min_feret_coord, max_feret_distance, max_feret_coord = feret.get_feret_from_mask(shape)
    assert min_feret_distance == min_feret_distance_target
    assert min_feret_coord == min_feret_coord_target
    assert max_feret_distance == max_feret_distance_target
    assert max_feret_coord == max_feret_coord_target


# Concatenate images to have two labeled objects within them
holo_ellipse_angled2 = holo_ellipse_angled.copy()
holo_ellipse_angled2[holo_ellipse_angled2 == 1] = 2
# Need to pad the holo_circle
holo_image = np.concatenate((np.pad(holo_circle, pad_width=((0, 1), (0, 3))), holo_ellipse_angled2))
filled_ellipse_angled2 = filled_ellipse_angled.copy()
filled_ellipse_angled2[filled_ellipse_angled2 == 1] = 2
filled_image = np.concatenate((np.pad(filled_circle, pad_width=((0, 0), (0, 2))), filled_ellipse_angled2))


@pytest.mark.parametrize(
    ("shape", "target"),
    [
        pytest.param(
            holo_image,
            {
                1: (4.0, ([1, 2], [5, 2]), 4.47213595499958, ([5, 4], [1, 2])),
                2: (3.1622776601683795, ([9, 4], [10, 1]), 7.615773105863909, ([13, 8], [10, 1])),
            },
            id="holo image",
        ),
        pytest.param(
            filled_image,
            {
                1: (6.0, ([1, 2], [7, 2]), 7.211102550927978, ([7, 6], [1, 2])),
                2: (5.385164807134504, ([15, 7], [10, 5]), 8.94427190999916, ([11, 9], [15, 1])),
            },
            id="filled image",
        ),
    ],
)
def test_get_feret_from_labelim(shape: npt.NDArray, target) -> None:
    """Test calculation of min/max feret for a labelled image with multiuple objects."""
    min_max_feret_size_coord = feret.get_feret_from_labelim(shape)
    assert min_max_feret_size_coord == target
