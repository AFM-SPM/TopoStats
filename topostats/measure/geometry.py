"""Functions for measuring geometric properties of grains."""

import networkx
import numpy as np
from numpy.typing import NDArray


def bounding_box_cartesian_points_float(
    points: NDArray[np.number],
) -> tuple[np.float64, np.float64, np.float64, np.float64]:
    """
    Calculate the bounding box from a set of points.

    Parameters
    ----------
    points : NDArray[np.number]
        Nx2 numpy array of points.

    Returns
    -------
    Tuple[np.float64, np.float64, np.float64, np.float64]
        Tuple of (min_x, min_y, max_x, max_y).

    Raises
    ------
    ValueError
        If the input array is not Nx2.
    """
    if points.shape[1] != 2:
        raise ValueError("Input array must be Nx2.")
    x_coords, y_coords = points[:, 0].astype(np.float64), points[:, 1].astype(np.float64)
    return (np.min(x_coords), np.min(y_coords), np.max(x_coords), np.max(y_coords))


def bounding_box_cartesian_points_integer(points: NDArray[np.number]) -> tuple[np.int32, np.int32, np.int32, np.int32]:
    """
    Calculate the bounding box from a set of points.

    Parameters
    ----------
    points : NDArray[np.number]
        Nx2 numpy array of points.

    Returns
    -------
    Tuple[np.int32, np.int32, np.int32, np.int32]
        Tuple of (min_x, min_y, max_x, max_y).

    Raises
    ------
    ValueError
        If the input array is not Nx2.
    """
    if points.shape[1] != 2:
        raise ValueError("Input array must be Nx2.")
    x_coords, y_coords = points[:, 0].astype(np.int32), points[:, 1].astype(np.int32)
    return (np.min(x_coords), np.min(y_coords), np.max(x_coords), np.max(y_coords))


def do_points_in_arrays_touch(
    points1: NDArray[np.number], points2: NDArray[np.number]
) -> tuple[bool, NDArray[np.number] | None, NDArray[np.number] | None]:
    """
    Check if any points in two arrays are touching.

    Parameters
    ----------
    points1 : NDArray[np.number]
        Nx2 numpy array of points.
    points2 : NDArray[np.number]
        Mx2 numpy array of points.

    Returns
    -------
    tuple[bool, NDArray[np.number] | None, NDArray[np.number] | None]
        True if any points in the two arrays are touching, False otherwise, followed by the two points that touch.

    Raises
    ------
    ValueError
        If the input arrays are not Nx2 and Mx2.
    """
    if points1.shape[1] != 2 or points2.shape[1] != 2:
        raise ValueError("Input arrays must be Nx2 and Mx2.")

    for point1 in points1:
        for point2 in points2:
            diff = np.abs(point1 - point2)
            if np.all(diff <= 1):
                return (True, point1, point2)
    return (False, None, None)


def calculate_shortest_branch_distances(
    nodes_with_branches: dict[int, NDArray[np.number]], whole_skeleton_graph: networkx.classes.graph.Graph
):
    """Calculate the shortest distances between branches emanating from nodes."""
    num_nodes = len(nodes_with_branches)
    shortest_node_distances = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    shortest_distances_branch_indexes = np.zeros((num_nodes, num_nodes), dtype=np.int32)
    shortest_distances_coordinates = np.empty((num_nodes, num_nodes, 2), dtype=object)

    for node_index_i, (node1, branches1) in enumerate(nodes_with_branches.items()):
        for node_index_j, (node2, branches2) in enumerate(nodes_with_branches.items()):
            # Don't compare the same node to itself
            if node_index_i == node_index_j:
                continue
            shortest_distance = None
            shortest_distance_branch_indexes = None
            # Iteratively compare all branches from node1 to all branches from node2 to find the shortest distance between any two branches
            for branch1 in branches1:
                for branch2 in branches2:
                    shortest_path_length_between_branch_1_and_2 = networkx.shortest_path_length(
                        whole_skeleton_graph, tuple(branch1), tuple(branch2)
                    )
                    if shortest_distance is None or shortest_path_length_between_branch_1_and_2 < shortest_distance:
                        shortest_distance = shortest_path_length_between_branch_1_and_2
                        shortest_distance_branch_indexes = (node1, node2)

            shortest_node_distances[node_index_i, node_index_j] = shortest_distance
            shortest_distances_branch_indexes[node_index_i, node_index_j] = shortest_distance_branch_indexes
            shortest_distances_coordinates[node_index_i, node_index_j] = (
                nodes_with_branches[shortest_distance_branch_indexes[0]][0],
                nodes_with_branches[shortest_distance_branch_indexes[1]][0],
            )

    return shortest_node_distances, shortest_distances_branch_indexes, shortest_distances_coordinates


def connect_best_matches(
    network_array_representation: NDArray[np.int32],
    whole_skeleton_graph: networkx.classes.graph.Graph,
    match_indexes: NDArray[np.int32],
    shortest_distances_between_nodes: NDArray[np.float64],
    shortest_distances_branch_indexes: NDArray[np.int32],
    emanating_branch_starts_by_node: dict[int, NDArray[np.int32]],
    extend_distance: float = -1,
):
    """Connect the branches between node pairs that have been deemed to be best matches.

    Parameters
    ----------
    network_array_representation : NDArray[np.int32]
        2D numpy array representing the network using integers to represent branches, nodes etc.
    whole_skeleton_graph : networkx.classes.graph.Graph
        Networkx graph representing the whole network.
    match_indexes : NDArray[np.int32]
        Nx2 numpy array of indexes of the best matching nodes.
    shortest_distances_between_nodes : NDArray[np.float64]
        NxN numpy array of shortest distances between every node pair.
    shortest_distances_branch_indexes : NDArray[np.int32]
        NxNx2 numpy array of indexes of the branches to connect between the best matching nodes.

    Returns
    -------
    NDArray[np.int32]
        2D numpy array representing the network using integers to represent branches, nodes etc.
    """
    for node_pair_index in match_indexes:
        # Fetch the shortest distance between two nodes
        shortest_distance = shortest_distances_between_nodes[node_pair_index[0], node_pair_index[1]]
        if shortest_distance <= extend_distance or extend_distance == -1:
            # Fetch the indexes of the branches to connect defined by the closest connecting branchees of the given nodes
            indexes_of_branches_to_connect = shortest_distances_branch_indexes[node_pair_index[0], node_pair_index[1]]
            node_numbers = list(emanating_branch_starts_by_node.keys())
            # Grab the coordinate of the starting branch position of the branch to connect
            source = tuple(
                emanating_branch_starts_by_node[node_numbers[node_pair_index[0]]][indexes_of_branches_to_connect[0]]
            )
            # Grab the coordinate of the branch position to connect to
            target = tuple(
                emanating_branch_starts_by_node[node_numbers[node_pair_index[1]]][indexes_of_branches_to_connect[1]]
            )
            # Get the path between the two branches using networkx
            path = np.array(networkx.shortest_path(whole_skeleton_graph, source, target))
            # Set all the coordinates in the path to 3 in the network array representation
            network_array_representation[path[:, 0], path[:, 1]] = 3

    return network_array_representation
