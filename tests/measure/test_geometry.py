"""Tests for the geometry module."""

import networkx
import numpy as np
import numpy.typing as npt
import pytest
from pytest_lazyfixture import lazy_fixture

# pylint: disable=too-many-arguments
from topostats.measure.geometry import (
    bounding_box_cartesian_points_float,
    bounding_box_cartesian_points_integer,
    calculate_shortest_branch_distances,
    connect_best_matches,
    do_points_in_arrays_touch,
)


def test_bounding_box_cartesian_points_float_raises_value_error() -> None:
    """Test the bounding_box_cartesian_points function raises a ValueError."""
    with pytest.raises(ValueError, match="Input array must be Nx2."):
        bounding_box_cartesian_points_float(np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]))


def test_bounding_box_cartesian_points_integer_raises_value_error() -> None:
    """Test the bounding_box_cartesian_points function raises a ValueError."""
    with pytest.raises(ValueError, match="Input array must be Nx2."):
        bounding_box_cartesian_points_float(np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]))


@pytest.mark.parametrize(
    ("points", "expected_bbox"),
    [
        pytest.param(np.array([[0, 0], [1, 1], [2, 2]]), (0, 0, 2, 2), id="diagonal line"),
        pytest.param(np.array([[0, 0], [1, 1], [2, 0]]), (0, 0, 2, 1), id="triangle"),
        pytest.param(np.array([[-5, -5], [-10, -10], [-3, -3]]), (-10, -10, -3, -3), id="negative values"),
        pytest.param(np.array([[-1, -1], [1, 1], [2, 0], [-2, 0]]), (-2, -1, 2, 1), id="negative and positive values"),
        pytest.param(np.array([[0.1, 0.1], [1.1, 1.1], [2.1, 2.1]]), (0.1, 0.1, 2.1, 2.1), id="diagonal line, floats"),
    ],
)
def test_bounding_box_cartesian_points_float(
    points: npt.NDArray[np.number], expected_bbox: tuple[np.float64, np.float64, np.float64, np.float64]
) -> None:
    """Test the bounding_box_cartesian_points function."""
    assert bounding_box_cartesian_points_float(points) == expected_bbox


@pytest.mark.parametrize(
    ("points", "expected_bbox"),
    [
        pytest.param(np.array([[0, 0], [1, 1], [2, 2]]), (0, 0, 2, 2), id="diagonal line"),
        pytest.param(np.array([[0, 0], [1, 1], [2, 0]]), (0, 0, 2, 1), id="triangle"),
        pytest.param(np.array([[-5, -5], [-10, -10], [-3, -3]]), (-10, -10, -3, -3), id="negative values"),
        pytest.param(np.array([[-1, -1], [1, 1], [2, 0], [-2, 0]]), (-2, -1, 2, 1), id="negative and positive values"),
        pytest.param(np.array([[0.1, 0.1], [1.1, 1.1], [2.1, 2.1]]), (0, 0, 2, 2), id="diagonal line, floats"),
    ],
)
def test_bounding_box_cartesian_points_integer(
    points: npt.NDArray[np.number], expected_bbox: tuple[np.float64, np.float64, np.float64, np.float64]
) -> None:
    """Test the bounding_box_cartesian_points function."""
    assert bounding_box_cartesian_points_integer(points) == expected_bbox


def test_do_points_in_arrays_touch_raises_value_error() -> None:
    """Test the do_points_in_arrays_touch function raises a ValueError."""
    with pytest.raises(ValueError, match="Input arrays must be Nx2 and Mx2."):
        do_points_in_arrays_touch(np.array([[0, 0], [1, 1], [2, 2]]), np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]))


@pytest.mark.parametrize(
    ("array_1", "array_2", "expected_result_touching", "expected_point_1_touching", "expected_point_2_touching"),
    [
        pytest.param(
            np.array([[0, 0], [1, 1], [2, 2]]),
            np.array([[4, 4], [5, 5], [6, 6]]),
            False,
            None,
            None,
            id="no touching points",
        ),
        pytest.param(
            np.array([[0, 0], [1, 1], [2, 2]]),
            np.array([[2, 3], [3, 4], [5, 5]]),
            True,
            np.array([2, 2]),
            np.array([2, 3]),
            id="touching points, non_diag",
        ),
        pytest.param(
            np.array([[0, 0], [1, 1], [2, 2]]),
            np.array([[3, 3], [4, 4], [5, 5]]),
            True,
            np.array([2, 2]),
            np.array([3, 3]),
            id="touching points, diag",
        ),
    ],
)
def test_do_points_in_arrays_touch(
    array_1: npt.NDArray[np.number],
    array_2: npt.NDArray[np.number],
    expected_result_touching: bool,
    expected_point_1_touching: npt.NDArray[np.number],
    expected_point_2_touching: npt.NDArray[np.number],
) -> None:
    """Test the do_points_in_arrays_touch function."""
    result_touching, point_touching_1, point_touching_2 = do_points_in_arrays_touch(array_1, array_2)

    assert result_touching == expected_result_touching
    np.testing.assert_array_equal(point_touching_1, expected_point_1_touching)
    np.testing.assert_array_equal(point_touching_2, expected_point_2_touching)


@pytest.mark.parametrize(
    (
        "network_array_representation",
        "whole_skeleton_graph",
        "match_indexes",
        "shortest_distances_between_nodes",
        "shortest_distances_branch_indexes",
        "emanating_branch_starts_by_node",
        "extend_distance",
        "expected_network_array_representation",
    ),
    [
        pytest.param(
            lazy_fixture("network_array_representation_figure_8"),
            lazy_fixture("whole_skeleton_graph_figure_8"),
            np.array([[1, 0]]),
            np.array([[0.0, 6.0], [6.0, 0.0]]),
            np.array([[[0, 0], [1, 1]], [[1, 1], [0, 0]]]),
            {
                0: [np.array([6, 1]), np.array([7, 3]), np.array([8, 1])],
                1: [np.array([6, 11]), np.array([7, 9]), np.array([8, 11])],
            },
            8.0,
            lazy_fixture("expected_network_array_representation_figure_8"),
            id="figure 8",
        )
    ],
)
def test_connect_best_matches(
    network_array_representation: npt.NDArray[np.int32],
    whole_skeleton_graph: networkx.classes.graph.Graph,
    match_indexes: npt.NDArray[np.int32],
    shortest_distances_between_nodes: npt.NDArray[np.number],
    shortest_distances_branch_indexes: npt.NDArray[np.int32],
    emanating_branch_starts_by_node: npt.NDArray[np.int32],
    extend_distance: float,
    expected_network_array_representation: npt.NDArray[np.int32],
) -> None:
    """Test the connect_best_matches function."""
    result = connect_best_matches(
        network_array_representation,
        whole_skeleton_graph,
        match_indexes,
        shortest_distances_between_nodes,
        shortest_distances_branch_indexes,
        emanating_branch_starts_by_node,
        extend_distance,
    )

    np.testing.assert_array_equal(result, expected_network_array_representation)


@pytest.mark.parametrize(
    (
        "nodes_with_branches_starting_coords",
        "whole_skeleton_graph",
        "expected_shortest_node_distances",
        "expected_shortest_distances_branch_indexes",
        "expected_shortest_distances_branch_coordinates",
    ),
    [
        pytest.param(
            {
                0: [np.array([6, 1]), np.array([7, 3]), np.array([8, 1])],
                1: [np.array([6, 11]), np.array([7, 9]), np.array([8, 11])],
            },
            lazy_fixture("whole_skeleton_graph_figure_8"),
            np.array([[0, 6.0], [6.0, 0.0]]),
            np.array([[[0, 0], [1, 1]], [[1, 1], [0, 0]]]),
            np.array([[[[0, 0], [0, 0]], [[7, 3], [7, 9]]], [[[7, 9], [7, 3]], [[0, 0], [0, 0]]]]),
            id="figure 8",
        )
    ],
)
def test_calculate_shortest_branch_distances(
    nodes_with_branches_starting_coords: dict[int, npt.NDArray[np.number]],
    whole_skeleton_graph: networkx.classes.graph.Graph,
    expected_shortest_node_distances: dict[int, float],
    expected_shortest_distances_branch_indexes: dict[int, npt.NDArray[np.int32]],
    expected_shortest_distances_branch_coordinates: dict[int, npt.NDArray[np.number]],
) -> None:
    """Test the calculate_shortest_branch_distances function."""
    shortest_node_distances, shortest_distances_branch_indexes, shortest_distances_branch_coordinates = (
        calculate_shortest_branch_distances(nodes_with_branches_starting_coords, whole_skeleton_graph)
    )

    np.testing.assert_array_equal(shortest_node_distances, expected_shortest_node_distances)
    np.testing.assert_array_equal(shortest_distances_branch_indexes, expected_shortest_distances_branch_indexes)
    print(shortest_distances_branch_coordinates)
    np.testing.assert_array_equal(shortest_distances_branch_coordinates, expected_shortest_distances_branch_coordinates)
