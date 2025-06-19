"""Functions for measuring geometric properties of grains."""

import math

import networkx
import numpy as np
import numpy.typing as npt


def bounding_box_cartesian_points_float(
    points: npt.NDArray[np.number],
) -> tuple[np.float64, np.float64, np.float64, np.float64]:
    """
    Calculate the bounding box from a set of points.

    Parameters
    ----------
    points : npt.NDArray[np.number]
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


def bounding_box_cartesian_points_integer(
    points: npt.NDArray[np.number],
) -> tuple[np.int32, np.int32, np.int32, np.int32]:
    """
    Calculate the bounding box from a set of points.

    Parameters
    ----------
    points : npt.NDArray[np.number]
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
    points1: npt.NDArray[np.int32], points2: npt.NDArray[np.int32]
) -> tuple[bool, npt.NDArray[np.int32] | None, npt.NDArray[np.int32] | None]:
    """
    Check if any points in two arrays are touching.

    Parameters
    ----------
    points1 : npt.NDArray[np.int32]
        Nx2 numpy array of points.
    points2 : npt.NDArray[np.int32]
        Mx2 numpy array of points.

    Returns
    -------
    tuple[bool, npt.NDArray[np.int32] | None, npt.NDArray[np.int32] | None]
        True if any points in the two arrays are touching, False otherwise, followed by the first touching point pair
        that was found. If no points are touching, the second and third elements of the tuple will be None.

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


# pylint: disable=too-many-locals
def calculate_shortest_branch_distances(
    nodes_with_branch_starting_coords: dict[int, list[npt.NDArray[np.int32]]],
    whole_skeleton_graph: networkx.classes.graph.Graph,
) -> tuple[npt.NDArray[np.number], npt.NDArray[np.int32], npt.NDArray[np.number]]:
    """
    Calculate the shortest distances between branches emanating from nodes.

    Parameters
    ----------
    nodes_with_branch_starting_coords : dict[int, list[npt.NDArray[np.int32]]]
        Dictionary where the key is the node number and the value is an Nx2 numpy array of the starting coordinates
        of its branches.
    whole_skeleton_graph : networkx.classes.graph.Graph
        Networkx graph representing the whole network.

    Returns
    -------
    Tuple[npt.NDArray[np.number], npt.NDArray[np.int32], npt.NDArray[np.int32]]
        - NxN numpy array of shortest distances between every node pair. Indexes of this array represent the nodes.
        Eg for a 3x3 matrix, there are 3 nodes being compared with each other.
        This matrix is diagonally symmetric and the diagonal values are 0 since a node is always 0 distance from itself.
        - NxNx2 numpy array of indexes of the best branches to connect between each node pair.
        Eg for node 1 and 3, the closest branches might be indexes 2 and 4, so the value at [1, 3] would be [2, 4].
        - NxNx2x2 numpy array of the coordinates of the branches to connect between each node pair.
        Eg for node 1 and 3, the closest branches might be at coordinates [2, 3] and [4, 5], so the
        value at [1, 3] would be [[2, 3], [4, 5]].
    """
    num_nodes = len(nodes_with_branch_starting_coords)
    shortest_node_distances = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    # For storing the indexes of the branches that are the best candidate between two nodes.
    # Eg: [[[0, 0], [1, 2]], [[1, 2], [0, 0]]] means that node 0's branch 0 connects with node 1's branch 2.
    # Note that this matrix is symmetric about the diagonal as we double-iterate between all nodes.
    shortest_distances_branch_indexes = np.zeros((num_nodes, num_nodes, 2), dtype=np.int32)
    shortest_distances_branch_coordinates = np.zeros((num_nodes, num_nodes, 2, 2), dtype=object)

    # Iterate over the nodes twice to compare each combination of nodes. This double counts, so will create a symmetric
    # matrix about the diagonal.
    for node_index_i, (_node_i, node_branches_starts_coords_i) in enumerate(nodes_with_branch_starting_coords.items()):
        for node_index_j, (_node_j, node_branches_starts_coords_j) in enumerate(
            nodes_with_branch_starting_coords.items()
        ):
            # Don't compare the same node to itself
            if node_index_i == node_index_j:
                continue
            # Store the shortest distance as we iterate.
            shortest_distance = None
            # For storing the pair of branch indexes that are the best candidate between the two nodes.
            # Eg: (3, 2) means that node i's branch 3 connects with node j's branch 2.
            shortest_distance_branch_indexes: tuple[int, int] | None = None
            # Iteratively compare all branches from node1 to all branches from node2
            # to find the shortest distance between any two branches
            for branch_index_i, branch_start_i in enumerate(node_branches_starts_coords_i):
                for branch_index_j, branch_start_j in enumerate(node_branches_starts_coords_j):
                    shortest_path_length_between_branch_i_and_j = networkx.shortest_path_length(
                        whole_skeleton_graph, tuple(branch_start_i), tuple(branch_start_j)
                    )
                    if shortest_distance is None or shortest_path_length_between_branch_i_and_j < shortest_distance:
                        shortest_distance = shortest_path_length_between_branch_i_and_j
                        # Store the indexes of the branches that are the shortest distance apart for node i and node j
                        shortest_distance_branch_indexes = (branch_index_i, branch_index_j)

            # Store the shortest distance between the two nodes
            shortest_node_distances[node_index_i, node_index_j] = shortest_distance
            # Store the indexes of the branches that are the shortest distance apart for node i and node j.
            # Note this may be None as the nodes may not be connected?
            shortest_distances_branch_indexes[node_index_i, node_index_j] = shortest_distance_branch_indexes
            # Ensure that the nodes are connected before storing the coordinates of the branches starting coords
            if shortest_distance_branch_indexes is not None:
                # Add the coordinates of the branch pairs for each node-node combination. So for example, for
                # node 0 and node 1, branches starting [6, 1] and [6, 11]:
                # np.array([ [ [[0, 0][0, 0]] [[6 1][6 11]] ] [ [[6 11][6 1]] [[0, 0][0, 0]] ] ])
                # Where the square [0 0][0 0]s are for node 0 / node 0 and node 1 / node 1.
                # And [6 1][6 11] is for node 0 / node 1, indicating that branches start at [6, 1] and [6, 11].
                shortest_distances_branch_coordinates[node_index_i, node_index_j] = (
                    node_branches_starts_coords_i[shortest_distance_branch_indexes[0]],
                    node_branches_starts_coords_j[shortest_distance_branch_indexes[1]],
                )
            else:
                shortest_distances_branch_coordinates[node_index_i, node_index_j] = (None, None)

    return shortest_node_distances, shortest_distances_branch_indexes, shortest_distances_branch_coordinates


# pylint: disable=too-many-positional-arguments
def connect_best_matches(
    network_array_representation: npt.NDArray[np.int32],
    whole_skeleton_graph: networkx.classes.graph.Graph,
    match_indexes: npt.NDArray[np.int32],
    shortest_distances_between_nodes: npt.NDArray[np.number],
    shortest_distances_branch_indexes: npt.NDArray[np.int32],
    emanating_branch_starts_by_node: dict[int, list[npt.NDArray[np.int32]]],
    extend_distance: float = -1,
) -> npt.NDArray[np.int32]:
    """
    Connect the branches between node pairs that have been deemed to be best matches.

    Parameters
    ----------
    network_array_representation : npt.NDArray[np.int32]
        2D numpy array representing the network using integers to represent branches, nodes etc.
    whole_skeleton_graph : networkx.classes.graph.Graph
        Networkx graph representing the whole network.
    match_indexes : npt.NDArray[np.int32]
        Nx2 numpy array of indexes of the best matching nodes.
        Eg: np.array([[1, 0], [2, 3]]) means that the best matching nodes are node 1 and node 0, and node 2 and node 3.
    shortest_distances_between_nodes : npt.NDArray[np.number]
        NxN numpy array of shortest distances between every node pair.
        Index positions indicate which node it's referring to, so index 2, 3 will be the shortest distance
        between nodes 2 and 3.
        Values on the diagonal will be 0 because the shortest distance between a node and itself is 0.
        Eg: np.array([[0.0, 6.0], [6.0, 0.0]]) means that the shortest distance between node 0 and node 1 is 6.0.
    shortest_distances_branch_indexes : npt.NDArray[np.int32]
        NxNx2 numpy array of indexes of the branches to connect between the best matching nodes.
        Not entirely sure what it does so won't attempt to explain more to avoid confusion.
    emanating_branch_starts_by_node : dict[int, list[npt.NDArray[np.int32]]]
        Dictionary where the key is the node number and the value is an Nx2 numpy array of the starting coordinates
        of the branches emanating from that node. Rather self-explanatory.
        Eg:
        ```python
        {
            0: [np.array([6, 1]), np.array([7, 3]), np.array([8, 1])],
            1: [np.array([6, 11]), np.array([7, 9]), np.array([8, 11])],
        },
        ```.
    extend_distance : float
        The distance to extend the branches to connect. If the shortest distance between two nodes is less than or equal
        to this distance, the branches will be connected. If -1, the branches will be connected regardless of distance.

    Returns
    -------
    npt.NDArray[np.int32]
        2D numpy array representing the network using integers to represent branches, nodes etc.
    """
    for node_pair_index in match_indexes:
        # Fetch the shortest distance between two nodes
        shortest_distance = shortest_distances_between_nodes[node_pair_index[0], node_pair_index[1]]
        if shortest_distance <= extend_distance or extend_distance == -1:
            # Fetch the indexes of the branches to connect defined by the closest connecting
            # branchees of the given nodes
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


# pylint: disable=too-many-locals
def find_branches_for_nodes(
    network_array_representation: npt.NDArray[np.int32],
    labelled_nodes: npt.NDArray[np.int32],
    labelled_branches: npt.NDArray[np.int32],
) -> dict[int, list[npt.NDArray[np.int32]]]:
    """
    Locate branch starting positions for each node in a network.

    Parameters
    ----------
    network_array_representation : npt.NDArray[np.int32]
        2D numpy array representing the network using integers to represent branches, nodes etc.
    labelled_nodes : npt.NDArray[np.int32]
        2D numpy array representing the network using integers to represent nodes.
    labelled_branches : npt.NDArray[np.int32]
        2D numpy array representing the network using integers to represent branches.

    Returns
    -------
    dict[int, list[npt.NDArray[np.int32]]]
        Dictionary where the key is the node number and the value is an Nx2 numpy array of the starting coordinates
        of the branches emanating from that node.
    """
    # Dictionary to store emanating branches for each labelled node
    emanating_branch_starts_by_node = {}

    # Iterate over all the nodes in the labelled nodes image
    for node_num in range(1, labelled_nodes.max() + 1):
        num_branches = 0
        # makes lil box around node with 1 overflow
        bounding_box = bounding_box_cartesian_points_integer(np.argwhere(labelled_nodes == node_num))
        crop_left = bounding_box[0] - 1
        crop_right = bounding_box[2] + 2
        crop_top = bounding_box[1] - 1
        crop_bottom = bounding_box[3] + 2
        cropped_matrix = network_array_representation[crop_left:crop_right, crop_top:crop_bottom]
        # get coords of nodes and branches in box
        node_coords = np.argwhere(cropped_matrix == 3)
        branch_coords = np.argwhere(cropped_matrix == 1)
        # iterate through node coords to see which are within 8 dirs
        for node_coord in node_coords:
            for branch_coord in branch_coords:
                distance = math.dist(node_coord, branch_coord)
                if distance <= math.sqrt(2):
                    num_branches = num_branches + 1

        # All nodes with even branches are considered to be complete as they have one
        # strand going in for each coming out. This assumes that no strands naturally terminate at nodes.

        # find the branch start point of odd branched nodes
        if num_branches % 2 == 1:
            emanating_branches: list[npt.NDArray[np.int32]] = (
                []
            )  # List to store emanating branches for the current label
            for branch in range(1, labelled_branches.max() + 1):
                # technically using labelled_branches when there's an end loop will only cause one
                #   of the end loop coords to be captured. This shopuldn't matter as the other
                #   label after the crossing should be closer to another node.
                # The touching_point_1 and touching_point_2 can be None since the function returns None for both
                # if no points touch.
                touching, touching_point_1, _touching_point_2 = do_points_in_arrays_touch(
                    np.argwhere(labelled_branches == branch),
                    np.argwhere(labelled_nodes == node_num),
                )
                if touching:
                    assert touching_point_1 is not None
                    # Above required for mypy to ensure that there are no Nones
                    # in the list, to prevent the return type being npt.NDArray[np.int32 | None]
                    emanating_branches.append(touching_point_1)

            assert len(emanating_branches) > 0, f"No branches found for node {node_num}"
            emanating_branch_starts_by_node[node_num - 1] = (
                emanating_branches  # Store emanating branches for this label
            )

    return emanating_branch_starts_by_node
