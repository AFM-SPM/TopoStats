"""Functions for measuring geometric properties of grains."""

from __future__ import annotations

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


# Test cases

node_no = 0
pairs = np.array([1, 3]), np.array([2, 0])
# ordered_branches =
matched_branches = {
    # Node pair 0
    0: {
        "ordered_coords": np.array([[279, 351], [279, 352], [280, 353], [281, 352]]),
        "heights": np.array([3.70838688e-09, 3.77688620e-09, 3.85533492e-09, 3.82206153e-09]),
        "distances": np.array([-2.23606798, -1.41421356, 0.0, 1.41421356]),
        "fwhm2": (
            0.48609207943034094,
            [0, 0.48609207943034094, 3.843898223226858e-09],
            [2, 0.0, 3.855334917702758e-09],
        ),
    },
    # Node pair 1
    1: {
        "ordered_coords": np.array([[278, 353], [279, 352], [280, 353], [281, 354]]),
        "heights": np.array([3.76817435e-09, 3.77688620e-09, 3.85533492e-09, 3.84389822e-09]),
        "distances": np.array([-2.0, -1.41421356, 0.0, 1.41421356]),
        "fwhm2": (1.4142135623730951, [0, 1.4142135623730951, 3.843898223226858e-09], [2, 0.0, 3.855334917702758e-09]),
    },
}
node_no = 1
matched_branches = {
    # Node pair 1
    0: {
        "ordered_coords": np.array(
            [[313, 234], [312, 235], [312, 236], [312, 237], [312, 238], [312, 239], [312, 240], [311, 241]]
        ),
        "heights": np.array(
            [
                3.63410835e-09,
                3.78225104e-09,
                3.83124345e-09,
                3.84546139e-09,
                3.83745040e-09,
                3.81844585e-09,
                3.79119805e-09,
                3.77335965e-09,
            ]
        ),
        "distances": np.array([-3.16227766, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.12310563]),
        "fwhm2": (
            6.192864473563146,
            [-2.0697588479454967, 4.123105625617649, 3.773359646805024e-09],
            [3, 0.0, 3.845461393270907e-09],
        ),
    },
    1: {
        "ordered_coords": np.array(
            [[310, 234], [311, 235], [312, 236], [312, 237], [312, 238], [312, 239], [313, 240], [314, 241]]
        ),
        "heights": np.array(
            [
                3.68470013e-09,
                3.78198234e-09,
                3.83124345e-09,
                3.84546139e-09,
                3.83745040e-09,
                3.81844585e-09,
                3.75514644e-09,
                3.63121390e-09,
            ]
        ),
        "distances": np.array([-3.60555128, -2.23606798, -1.0, 0.0, 1.0, 2.0, 3.16227766, 4.47213595]),
        "fwhm2": (
            6.190404878612629,
            [-2.8504730390078383, 3.3399318396047906, 3.738337645629056e-09],
            [3, 0.0, 3.845461393270907e-09],
        ),
    },
}
node_no = 2
matched_branches = {
    # Node pair 2
    0: {
        "ordered_coords": np.array(
            [
                [400, 435],
                [400, 436],
                [401, 437],
                [402, 438],
                [403, 438],
                [404, 439],
                [405, 439],
                [406, 439],
                [407, 438],
                [408, 437],
                [409, 436],
            ]
        ),
        "heights": np.array(
            [
                3.10881654e-09,
                3.20104991e-09,
                3.32195139e-09,
                3.39756843e-09,
                3.40635237e-09,
                3.44510007e-09,
                3.45618980e-09,
                3.46052370e-09,
                3.47123389e-09,
                3.45614207e-09,
                3.40259807e-09,
            ]
        ),
        "distances": np.array(
            [
                -7.61577311,
                -7.28010989,
                -6.08276253,
                -5.0,
                -4.0,
                -3.16227766,
                -2.23606798,
                -1.41421356,
                0.0,
                1.41421356,
                2.82842712,
            ]
        ),
        "fwhm2": (
            7.2558316483840475,
            [-4.427404523637857, 2.8284271247461903, 3.4025980724929314e-09],
            [8, 0.0, 3.471233888229924e-09],
        ),
    },
    1: {
        "ordered_coords": np.array(
            [
                [399, 437],
                [400, 436],
                [401, 437],
                [402, 438],
                [403, 438],
                [404, 439],
                [405, 439],
                [406, 439],
                [407, 438],
                [408, 437],
                [409, 438],
            ]
        ),
        "heights": np.array(
            [
                3.17860763e-09,
                3.20104991e-09,
                3.32195139e-09,
                3.39756843e-09,
                3.40635237e-09,
                3.44510007e-09,
                3.45618980e-09,
                3.46052370e-09,
                3.47123389e-09,
                3.45614207e-09,
                3.41577219e-09,
            ]
        ),
        "distances": np.array(
            [
                -8.06225775,
                -7.28010989,
                -6.08276253,
                -5.0,
                -4.0,
                -3.16227766,
                -2.23606798,
                -1.41421356,
                0.0,
                1.41421356,
                2.0,
            ]
        ),
        "fwhm2": (
            5.7963441650834095,
            [-3.7963441650834064, 2.0000000000000027, 3.415772191480199e-09],
            [8, 0.0, 3.471233888229924e-09],
        ),
    },
}
node_no = 3
matched_branches = {
    # Node pair 3
    0: {
        "ordered_coords": np.array([[451, 222], [450, 223], [451, 224], [452, 225], [453, 224]]),
        "heights": np.array([3.57558624e-09, 3.58253193e-09, 3.67405939e-09, 3.65632948e-09, 3.64216711e-09]),
        "distances": np.array([-2.0, -1.41421356, 0.0, 1.41421356, 2.0]),
        "fwhm2": (2.0000000000000053, [0, 2.0000000000000053, 3.642167106242417e-09], [2, 0.0, 3.6740593863660243e-09]),
    },
    1: {
        "ordered_coords": np.array([[449, 223], [450, 223], [451, 224], [452, 225], [453, 226]]),
        "heights": np.array([3.49343420e-09, 3.58253193e-09, 3.67405939e-09, 3.65632948e-09, 3.57037692e-09]),
        "distances": np.array([-2.23606798, -1.41421356, 0.0, 1.41421356, 2.82842712]),
        "fwhm2": (2.608446976971279, [0, 2.608446976971279, 3.5837467914050826e-09], [2, 0.0, 3.6740593863660243e-09]),
    },
}
node_no = 4
matched_branches = {
    # Node pair 4
    0: {
        "ordered_coords": np.array([[555, 195], [556, 194], [557, 193], [558, 194], [558, 195]]),
        "heights": np.array([[555, 195], [556, 194], [557, 193], [558, 194], [558, 195]]),
        "distances": np.array([-3.16227766, -2.0, -1.41421356, 0.0, 1.0]),
        "fwhm2": (0, [0, 0, 2.9737831933660626e-09], [3, 0.0, 2.9737831933660626e-09]),
    },
    1: {
        "ordered_coords": np.array([[555, 192], [556, 193], [557, 193], [558, 193], [559, 192]]),
        "heights": np.array([2.84683056e-09, 2.91621364e-09, 2.94411806e-09, 2.95509886e-09, 2.89934181e-09]),
        "distances": np.array([-4.60555128, -3.23606798, -2.41421356, 0.0, 1.23606798]),
        "fwhm2": (
            4.7371413484873175,
            [-3.5370513137311965, 1.2000900347561207, 2.900964712955829e-09],
            [3, 0.0, 2.9550988629992053e-09],
        ),
    },
}

# def join_matching_branches_through_node(pairs: npt.NDArray[np.int32]):
#     """
#     Join branches that are matched through a node.

#     Parameters
#     ----------
#     pairs: npt.NDArray[np.int32]
#         Nx2 numpy array of pairs of branches that are matched through a node.

#     Returns
#     -------
#     matched_branches: dict[int, dict[str, npt.NDArray[np.number]]]
#         Dictionary where the key is the index of the pair and the value is a dictionary containing the following
#         keys:
#         - "ordered_coords" : npt.NDArray[np.int32].
#         - "heights" : npt.NDArray[np.number]. Heights of the branches.
#         - "distances" :
#     """

#     matched_branches = {}
#     masked_imgae = {}
#     branch_img = np.zeros_like(self.skeleton)  # initialising paired branch img
#     avg_img = np.zeros_like(self.skeleton)
#     for i, (branch_1, branch_2) in enumerate(pairs):
#         matched_branches[i] = {}
#         masked_image[i] = {}
#         # find close ends by rearranging branch coords
#         branch_1_coords, branch_2_coords = self.order_branches(
#             ordered_branches[branch_1], ordered_branches[branch_2]
#         )  # Sylvia: CLEAN OF SELF.
#         # Get graphical shortest path between branch ends on the skeleton
#         crossing = nx.shortest_path(
#             self.reduced_skel_graph,
#             source=tuple(branch_1_coords[-1]),
#             target=tuple(branch_2_coords[0]),
#             weight="weight",
#         )
#         crossing = np.asarray(crossing[1:-1])  # remove start and end points & turn into array
#         # Branch coords and crossing
#         if crossing.shape == (0,):
#             branch_coords = np.vstack([branch_1_coords, branch_2_coords])
#         else:
#             branch_coords = np.vstack([branch_1_coords, crossing, branch_2_coords])
#         # make images of single branch joined and multiple branches joined
#         single_branch_img = np.zeros_like(self.skeleton)
#         single_branch_img[branch_coords[:, 0], branch_coords[:, 1]] = 1
#         single_branch_coords = self.order_branch(single_branch_img, [0, 0])  # Sylvia: CLEAN OF SELF.
#         # calc image-wide coords
#         matched_branches[i]["ordered_coords"] = single_branch_coords
#         # get heights and trace distance of branch
#         try:
#             assert average_trace_advised
#             distances, heights, mask, _ = self.average_height_trace(
#                 self.image, single_branch_img, single_branch_coords, [x, y]
#             )  # hess_area Sylvia: CLEAN OF SELF.
#             masked_image[i]["avg_mask"] = mask
#         except (
#             AssertionError,
#             IndexError,
#         ) as e:  # Assertion - avg trace not advised, Index - wiggy branches
#             LOGGER.info(f"[{self.filename}] : avg trace failed with {e}, single trace only.")
#             average_trace_advised = False
#             distances = self.coord_dist_rad(single_branch_coords, [x, y])  # Sylvia: CLEAN OF SELF.
#             # distances = self.coord_dist(single_branch_coords)
#             zero_dist = distances[
#                 np.argmin(np.sqrt((single_branch_coords[:, 0] - x) ** 2 + (single_branch_coords[:, 1] - y) ** 2))
#             ]
#             heights = self.image[single_branch_coords[:, 0], single_branch_coords[:, 1]]  # self.hess
#             distances = distances - zero_dist
#             distances, heights = self.average_uniques(distances, heights)  # needs to be paired with coord_dist_rad
#         matched_branches[i]["heights"] = heights
#         matched_branches[i]["distances"] = distances  # * self.px_2_nm
#         # identify over/under
#         matched_branches[i]["fwhm2"] = self.fwhm2(heights, distances)
