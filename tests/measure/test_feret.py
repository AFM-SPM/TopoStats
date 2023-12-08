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

holo_circle = np.zeros((14, 14), dtype=np.uint8)
rr, cc = draw.circle_perimeter(6, 6, 5)
holo_circle[rr, cc] = 1

holo_ellipse_vertical = np.zeros((16, 10), dtype=np.uint8)
rr, cc = draw.ellipse_perimeter(8, 5, 3, 6, orientation=np.deg2rad(90))
holo_ellipse_vertical[rr, cc] = 1

holo_ellipse_horizontal = np.zeros((10, 16), dtype=np.uint8)
rr, cc = draw.ellipse_perimeter(5, 8, 6, 3, orientation=np.deg2rad(90))
holo_ellipse_horizontal[rr, cc] = 1

holo_ellipse_angled = np.zeros((12, 14), dtype=np.uint8)
rr, cc = draw.ellipse_perimeter(6, 7, 3, 5, orientation=np.deg2rad(30))
holo_ellipse_angled[rr, cc] = 1

curved_line = np.zeros((10, 10), dtype=np.uint8)
rr, cc = draw.bezier_curve(1, 5, 5, -2, 8, 8, 2)
curved_line[rr, cc] = 1

filled_circle = np.zeros((14, 14), dtype=np.uint8)
rr, cc = draw.disk((6, 6), 6)
filled_circle[rr, cc] = 1

filled_ellipse_vertical = np.zeros((16, 10), dtype=np.uint8)
rr, cc = draw.ellipse(8, 5, 3, 6, rotation=np.deg2rad(90))
filled_ellipse_vertical[rr, cc] = 1

filled_ellipse_horizontal = np.zeros((10, 16), dtype=np.uint8)
rr, cc = draw.ellipse(5, 8, 6, 3, rotation=np.deg2rad(90))
filled_ellipse_horizontal[rr, cc] = 1

filled_ellipse_angled = np.zeros((12, 14), dtype=np.uint8)
rr, cc = draw.ellipse(6, 7, 3, 5, rotation=np.deg2rad(30))
filled_ellipse_angled[rr, cc] = 1


@pytest.mark.parametrize(
    ("point1", "point2", "point3", "target"),
    [
        (POINT3, POINT2, POINT1, 1),  # Clockwise
        (POINT1, POINT2, POINT3, -1),  # Anti-clockwise
        (POINT1, POINT2, POINT4, 0),  # Vertical Line
        (POINT1, POINT5, POINT6, 0),  # Horizontal Line
    ],
)
def test_orientation(point1: tuple, point2: tuple, point3: tuple, target: int) -> None:
    """Test calculation of orientation of three points."""
    assert feret.orientation(point1, point2, point3) == target


@pytest.mark.parametrize(
    ("shape", "upper_target", "lower_target"),
    [
        (tiny_circle, [[0, 1], [1, 2], [2, 1]], [[0, 1], [1, 0], [2, 1]]),
        (small_circle, [[0, 1], [0, 3], [1, 4], [3, 4], [4, 3]], [[0, 1], [1, 0], [3, 0], [4, 1], [4, 3]]),
        (
            holo_circle,
            [[1, 4], [1, 8], [4, 11], [8, 11], [11, 8]],
            [[1, 4], [4, 1], [8, 1], [11, 4], [11, 8]],
        ),
        (
            holo_ellipse_horizontal,
            [[1, 5], [1, 11], [2, 13], [3, 14], [5, 14], [7, 12], [8, 10]],
            [[1, 5], [2, 3], [4, 1], [6, 1], [7, 2], [8, 4], [8, 10]],
        ),
        (
            holo_ellipse_vertical,
            [[1, 3], [1, 5], [3, 7], [5, 8], [11, 8], [13, 7], [14, 6]],
            [[1, 3], [2, 2], [4, 1], [10, 1], [12, 2], [14, 4], [14, 6]],
        ),
        (
            holo_ellipse_angled,
            [[1, 3], [1, 7], [2, 9], [4, 11], [6, 12], [8, 12], [10, 10]],
            [[1, 3], [3, 1], [5, 1], [7, 2], [9, 4], [10, 6], [10, 10]],
        ),
        (
            curved_line,
            [[1, 5], [8, 8]],
            [[1, 5], [2, 3], [4, 1], [5, 1], [6, 2], [7, 4], [8, 7], [8, 8]],
        ),
    ],
)
def test_hulls(shape: npt.NDArray, upper_target: list, lower_target: list) -> None:
    """Test construction of upper and lower hulls."""
    print(f"{holo_ellipse_angled=}")
    print(f"{np.argwhere(shape == 1)=}")
    upper, lower = feret.hulls(np.argwhere(shape == 1))
    np.testing.assert_array_equal(upper, upper_target)
    np.testing.assert_array_equal(lower, lower_target)


@pytest.mark.parametrize(
    ("shape", "points_target"),
    [
        (
            tiny_circle,
            [
                ([0, 1], [1, 0]),
                ([0, 1], [2, 1]),
                ([0, 1], [1, 2]),
                ([1, 0], [1, 2]),
                ([1, 2], [2, 1]),
                ([1, 0], [2, 1]),
            ],
        ),
        (
            small_circle,
            [
                ([0, 1], [1, 0]),
                ([0, 1], [3, 0]),
                ([0, 1], [4, 1]),
                ([0, 1], [4, 3]),
                ([0, 1], [0, 3]),
                ([0, 3], [1, 0]),
                ([0, 3], [3, 0]),
                ([0, 3], [4, 1]),
                ([0, 3], [4, 3]),
                ([0, 1], [1, 4]),
                ([1, 0], [1, 4]),
                ([1, 4], [3, 0]),
                ([1, 4], [4, 1]),
                ([1, 4], [4, 3]),
                ([0, 1], [3, 4]),
                ([1, 0], [3, 4]),
                ([3, 0], [3, 4]),
                ([3, 4], [4, 1]),
                ([3, 4], [4, 3]),
                ([1, 0], [4, 3]),
                ([3, 0], [4, 3]),
                ([4, 1], [4, 3]),
            ],
        ),
        (
            holo_circle,
            [
                ([1, 4], [4, 1]),
                ([1, 4], [8, 1]),
                ([1, 4], [11, 4]),
                ([1, 4], [11, 8]),
                ([1, 4], [1, 8]),
                ([1, 8], [4, 1]),
                ([1, 8], [8, 1]),
                ([1, 8], [11, 4]),
                ([1, 8], [11, 8]),
                ([1, 4], [4, 11]),
                ([4, 1], [4, 11]),
                ([4, 11], [8, 1]),
                ([4, 11], [11, 4]),
                ([4, 11], [11, 8]),
                ([1, 4], [8, 11]),
                ([4, 1], [8, 11]),
                ([8, 1], [8, 11]),
                ([8, 11], [11, 4]),
                ([8, 11], [11, 8]),
                ([4, 1], [11, 8]),
                ([8, 1], [11, 8]),
                ([11, 4], [11, 8]),
            ],
        ),
        (
            holo_ellipse_horizontal,
            [
                ([1, 5], [2, 3]),
                ([1, 5], [4, 1]),
                ([1, 5], [6, 1]),
                ([1, 5], [7, 2]),
                ([1, 5], [8, 4]),
                ([1, 5], [8, 10]),
                ([1, 5], [1, 11]),
                ([1, 11], [2, 3]),
                ([1, 11], [4, 1]),
                ([1, 11], [6, 1]),
                ([1, 11], [7, 2]),
                ([1, 11], [8, 4]),
                ([1, 11], [8, 10]),
                ([1, 5], [2, 13]),
                ([2, 3], [2, 13]),
                ([2, 13], [4, 1]),
                ([2, 13], [6, 1]),
                ([2, 13], [7, 2]),
                ([2, 13], [8, 4]),
                ([2, 13], [8, 10]),
                ([1, 5], [3, 14]),
                ([2, 3], [3, 14]),
                ([3, 14], [4, 1]),
                ([3, 14], [6, 1]),
                ([3, 14], [7, 2]),
                ([3, 14], [8, 4]),
                ([3, 14], [8, 10]),
                ([1, 5], [5, 14]),
                ([2, 3], [5, 14]),
                ([4, 1], [5, 14]),
                ([5, 14], [6, 1]),
                ([5, 14], [7, 2]),
                ([5, 14], [8, 4]),
                ([5, 14], [8, 10]),
                ([1, 5], [7, 12]),
                ([2, 3], [7, 12]),
                ([4, 1], [7, 12]),
                ([6, 1], [7, 12]),
                ([7, 2], [7, 12]),
                ([7, 12], [8, 4]),
                ([7, 12], [8, 10]),
                ([2, 3], [8, 10]),
                ([4, 1], [8, 10]),
                ([6, 1], [8, 10]),
                ([7, 2], [8, 10]),
                ([8, 4], [8, 10]),
            ],
        ),
        (
            holo_ellipse_vertical,
            [
                ([1, 3], [2, 2]),
                ([1, 3], [4, 1]),
                ([1, 3], [10, 1]),
                ([1, 3], [12, 2]),
                ([1, 3], [14, 4]),
                ([1, 3], [14, 6]),
                ([1, 3], [1, 5]),
                ([1, 5], [2, 2]),
                ([1, 5], [4, 1]),
                ([1, 5], [10, 1]),
                ([1, 5], [12, 2]),
                ([1, 5], [14, 4]),
                ([1, 5], [14, 6]),
                ([1, 3], [3, 7]),
                ([2, 2], [3, 7]),
                ([3, 7], [4, 1]),
                ([3, 7], [10, 1]),
                ([3, 7], [12, 2]),
                ([3, 7], [14, 4]),
                ([3, 7], [14, 6]),
                ([1, 3], [5, 8]),
                ([2, 2], [5, 8]),
                ([4, 1], [5, 8]),
                ([5, 8], [10, 1]),
                ([5, 8], [12, 2]),
                ([5, 8], [14, 4]),
                ([5, 8], [14, 6]),
                ([1, 3], [11, 8]),
                ([2, 2], [11, 8]),
                ([4, 1], [11, 8]),
                ([10, 1], [11, 8]),
                ([11, 8], [12, 2]),
                ([11, 8], [14, 4]),
                ([11, 8], [14, 6]),
                ([1, 3], [13, 7]),
                ([2, 2], [13, 7]),
                ([4, 1], [13, 7]),
                ([10, 1], [13, 7]),
                ([12, 2], [13, 7]),
                ([13, 7], [14, 4]),
                ([13, 7], [14, 6]),
                ([2, 2], [14, 6]),
                ([4, 1], [14, 6]),
                ([10, 1], [14, 6]),
                ([12, 2], [14, 6]),
                ([14, 4], [14, 6]),
            ],
        ),
        (
            holo_ellipse_angled,
            [
                ([1, 3], [3, 1]),
                ([1, 3], [5, 1]),
                ([1, 3], [7, 2]),
                ([1, 3], [9, 4]),
                ([1, 3], [10, 6]),
                ([1, 3], [10, 10]),
                ([1, 3], [1, 7]),
                ([1, 7], [3, 1]),
                ([1, 7], [5, 1]),
                ([1, 7], [7, 2]),
                ([1, 7], [9, 4]),
                ([1, 7], [10, 6]),
                ([1, 7], [10, 10]),
                ([1, 3], [2, 9]),
                ([2, 9], [3, 1]),
                ([2, 9], [5, 1]),
                ([2, 9], [7, 2]),
                ([2, 9], [9, 4]),
                ([2, 9], [10, 6]),
                ([2, 9], [10, 10]),
                ([1, 3], [4, 11]),
                ([3, 1], [4, 11]),
                ([4, 11], [5, 1]),
                ([4, 11], [7, 2]),
                ([4, 11], [9, 4]),
                ([4, 11], [10, 6]),
                ([4, 11], [10, 10]),
                ([1, 3], [6, 12]),
                ([3, 1], [6, 12]),
                ([5, 1], [6, 12]),
                ([6, 12], [7, 2]),
                ([6, 12], [9, 4]),
                ([6, 12], [10, 6]),
                ([6, 12], [10, 10]),
                ([1, 3], [8, 12]),
                ([3, 1], [8, 12]),
                ([5, 1], [8, 12]),
                ([7, 2], [8, 12]),
                ([8, 12], [9, 4]),
                ([8, 12], [10, 6]),
                ([8, 12], [10, 10]),
                ([3, 1], [10, 10]),
                ([5, 1], [10, 10]),
                ([7, 2], [10, 10]),
                ([9, 4], [10, 10]),
                ([10, 6], [10, 10]),
            ],
        ),
        (
            curved_line,
            [
                ([1, 5], [2, 3]),
                ([1, 5], [4, 1]),
                ([1, 5], [5, 1]),
                ([1, 5], [6, 2]),
                ([1, 5], [7, 4]),
                ([1, 5], [8, 7]),
                ([1, 5], [8, 8]),
                ([2, 3], [8, 8]),
                ([4, 1], [8, 8]),
                ([5, 1], [8, 8]),
                ([6, 2], [8, 8]),
                ([7, 4], [8, 8]),
                ([8, 7], [8, 8]),
            ],
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
        (tiny_circle, [([0, 1], [2, 1]), ([0, 1], [1, 0]), ([1, 2], [1, 0]), ([1, 2], [0, 1]), ([2, 1], [0, 1])]),
        (
            small_circle,
            [
                ([0, 1], [4, 3]),
                ([0, 1], [4, 1]),
                ([0, 3], [4, 1]),
                ([0, 3], [3, 0]),
                ([1, 4], [3, 0]),
                ([1, 4], [1, 0]),
                ([3, 4], [1, 0]),
                ([3, 4], [0, 1]),
                ([4, 3], [0, 1]),
            ],
        ),
        (
            holo_circle,
            [
                ([1, 4], [11, 8]),
                ([1, 4], [11, 4]),
                ([1, 8], [11, 4]),
                ([1, 8], [8, 1]),
                ([4, 11], [8, 1]),
                ([4, 11], [4, 1]),
                ([8, 11], [4, 1]),
                ([8, 11], [1, 4]),
                ([11, 8], [1, 4]),
            ],
        ),
        (
            holo_ellipse_horizontal,
            [
                ([1, 5], [8, 10]),
                ([1, 5], [8, 4]),
                ([1, 11], [8, 4]),
                ([1, 11], [7, 2]),
                ([2, 13], [7, 2]),
                ([2, 13], [6, 1]),
                ([3, 14], [6, 1]),
                ([3, 14], [4, 1]),
                ([5, 14], [4, 1]),
                ([5, 14], [2, 3]),
                ([7, 12], [2, 3]),
                ([7, 12], [1, 5]),
                ([8, 10], [1, 5]),
            ],
        ),
        (
            holo_ellipse_vertical,
            [
                ([1, 3], [14, 6]),
                ([1, 3], [14, 4]),
                ([1, 5], [14, 4]),
                ([1, 5], [12, 2]),
                ([3, 7], [12, 2]),
                ([3, 7], [10, 1]),
                ([5, 8], [10, 1]),
                ([5, 8], [4, 1]),
                ([11, 8], [4, 1]),
                ([11, 8], [2, 2]),
                ([13, 7], [2, 2]),
                ([13, 7], [1, 3]),
                ([14, 6], [1, 3]),
            ],
        ),
        (
            holo_ellipse_angled,
            [
                ([1, 3], [10, 10]),
                ([1, 3], [10, 6]),
                ([1, 7], [10, 6]),
                ([1, 7], [9, 4]),
                ([2, 9], [9, 4]),
                ([2, 9], [7, 2]),
                ([4, 11], [7, 2]),
                ([4, 11], [5, 1]),
                ([6, 12], [5, 1]),
                ([6, 12], [3, 1]),
                ([8, 12], [3, 1]),
                ([8, 12], [1, 3]),
                ([10, 10], [1, 3]),
            ],
        ),
        # (
        #     curved_line,
        #     [],
        # ),
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
        (tiny_circle, 1.4142135623730951, ([0, 1], [1, 0]), 2, ([2, 1], [0, 1])),
        (small_circle, 4.0, ([0, 1], [4, 1]), 4.47213595499958, ([4, 3], [0, 1])),
        (holo_circle, 9.899494936611665, ([1, 8], [8, 1]), 10.770329614269007, ([11, 8], [1, 4])),
        (holo_ellipse_horizontal, 7.0710678118654755, ([1, 5], [8, 4]), 13.341664064126334, ([3, 14], [6, 1])),
        (holo_ellipse_vertical, 7.0710678118654755, ([5, 8], [4, 1]), 13.341664064126334, ([14, 6], [1, 3])),
        (holo_ellipse_angled, 8.54400374531753, ([1, 7], [9, 4]), 12.083045973594572, ([8, 12], [3, 1])),
        # (curved_line, 5.656854249492381, ([1, 5], [5, 1]), 8.06225774829855, ([8, 8], [4, 1])),
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
        (tiny_circle, 1.4142135623730951, ([0, 1], [1, 0]), 2, ([2, 1], [0, 1])),
        (small_circle, 4.0, ([0, 1], [4, 1]), 4.47213595499958, ([4, 3], [0, 1])),
        (holo_circle, 9.899494936611665, ([1, 8], [8, 1]), 10.770329614269007, ([11, 8], [1, 4])),
        (holo_ellipse_horizontal, 7.0710678118654755, ([1, 5], [8, 4]), 13.341664064126334, ([3, 14], [6, 1])),
        (holo_ellipse_vertical, 7.0710678118654755, ([5, 8], [4, 1]), 13.341664064126334, ([14, 6], [1, 3])),
        (holo_ellipse_angled, 8.54400374531753, ([1, 7], [9, 4]), 12.083045973594572, ([8, 12], [3, 1])),
        # (curved_line, 5.656854249492381, ([1, 5], [5, 1]), 8.06225774829855, ([8, 8], [4, 1])),
        (filled_circle, 10.0, ([1, 3], [11, 3]), 11.661903789690601, ([11, 9], [1, 3])),
        (filled_ellipse_horizontal, 4.0, ([3, 4], [7, 4]), 10.198039027185569, ([6, 13], [4, 3])),
        (filled_ellipse_vertical, 4.0, ([4, 7], [4, 3]), 10.198039027185569, ([13, 6], [3, 4])),
        (filled_ellipse_angled, 5.385164807134504, ([8, 9], [3, 7]), 8.94427190999916, ([4, 11], [8, 3])),
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
holo_image = np.concatenate((holo_circle, holo_ellipse_angled2))
filled_ellipse_angled2 = filled_ellipse_angled.copy()
filled_ellipse_angled2[filled_ellipse_angled2 == 1] = 2
filled_image = np.concatenate((filled_circle, filled_ellipse_angled2))


@pytest.mark.parametrize(
    ("shape", "target"),
    [
        (
            holo_image,
            {
                1: (9.899494936611665, ([1, 8], [8, 1]), 10.770329614269007, ([11, 8], [1, 4])),
                2: (8.54400374531753, ([15, 7], [23, 4]), 12.083045973594572, ([22, 12], [17, 1])),
            },
        ),
        (
            filled_image,
            {
                1: (10.0, ([1, 3], [11, 3]), 11.661903789690601, ([11, 9], [1, 3])),
                2: (5.385164807134504, ([22, 9], [17, 7]), 8.94427190999916, ([18, 11], [22, 3])),
            },
        ),
    ],
)
def test_get_feret_from_labelim(shape: npt.NDArray, target) -> None:
    """Test calculation of min/max feret for a labelled image with multiuple objects."""
    min_max_feret_size_coord = feret.get_feret_from_labelim(shape)
    assert min_max_feret_size_coord == target
