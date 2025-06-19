"""Perform Crossing Region Processing and Analysis."""

import logging
from itertools import combinations
from typing import TypedDict

import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.ndimage import binary_dilation
from scipy.signal import argrelextrema
from skimage.morphology import label

from topostats.logs.logs import LOGGER_NAME
from topostats.measure.geometry import (
    calculate_shortest_branch_distances,
    connect_best_matches,
    find_branches_for_nodes,
)
from topostats.tracing.pruning import prune_skeleton
from topostats.tracing.skeletonize import getSkeleton
from topostats.tracing.tracingfuncs import order_branch, order_branch_from_start
from topostats.utils import ResolutionError, convolve_skeleton

LOGGER = logging.getLogger(LOGGER_NAME)

# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-branches
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-lines
# pylint: disable=too-many-locals
# pylint: disable=too-many-nested-blocks
# pylint: disable=too-many-public-methods
# pylint: disable=too-many-statements


class MatchedBranch(TypedDict):
    """
    Dictionary containing the matched branches.

    matched_branches: dict[int, dict[str, npt.NDArray[np.number]]]
        Dictionary where the key is the index of the pair and the value is a dictionary containing the following
        keys:
        - "ordered_coords" : npt.NDArray[np.int32]. The ordered coordinates of the branch.
        - "heights" : npt.NDArray[np.number]. Heights of the branch coordinates.
        - "distances" : npt.NDArray[np.number]. Distances of the branch coordinates.
        - "fwhm" : npt.NDArray[np.number]. Full width half maximum of the branch.
        - "angles" : np.float64. The initial direction angle of the branch, added in later steps.
    """

    ordered_coords: npt.NDArray[np.int32]
    heights: npt.NDArray[np.number]
    distances: npt.NDArray[np.number]
    fwhm: dict[str, np.float64 | tuple[np.float64]]
    angles: np.float64 | None


class NodeDict(TypedDict):
    """Dictionary containing the node information."""

    error: bool
    pixel_to_nm_scaling: np.float64
    branch_stats: dict[int, MatchedBranch] | None
    node_coords: npt.NDArray[np.int32] | None
    confidence: np.float64 | None


class ImageDict(TypedDict):
    """Dictionary containing the image information."""

    nodes: dict[str, dict[str, npt.NDArray[np.int32]]]
    grain: dict[str, npt.NDArray[np.int32] | dict[str, npt.NDArray[np.int32]]]


class nodeStats:
    """
    Class containing methods to find and analyse the nodes/crossings within a grain.

    Parameters
    ----------
    filename : str
        The name of the file being processed. For logging purposes.
    image : npt.npt.NDArray
        The array of pixels.
    mask : npt.npt.NDArray
        The binary segmentation mask.
    smoothed_mask : npt.NDArray
        A smoothed version of the bianary segmentation mask.
    skeleton : npt.NDArray
        A binary single-pixel wide mask of objects in the 'image'.
    pixel_to_nm_scaling : np.float32
        The pixel to nm scaling factor.
    n_grain : int
        The grain number.
    node_joining_length : float
        The length over which to join skeletal intersections to be counted as one crossing.
    node_joining_length : float
        The distance over which to join nearby odd-branched nodes.
    node_extend_dist : float
        The distance under which to join odd-branched node regions.
    branch_pairing_length : float
        The length from the crossing point to pair and trace, obtaining FWHM's.
    pair_odd_branches : bool
        Whether to try and pair odd-branched nodes.
    """

    def __init__(
        self,
        filename: str,
        image: npt.NDArray,
        mask: npt.NDArray,
        smoothed_mask: npt.NDArray,
        skeleton: npt.NDArray,
        pixel_to_nm_scaling: np.float64,
        n_grain: int,
        node_joining_length: float,
        node_extend_dist: float,
        branch_pairing_length: float,
        pair_odd_branches: bool,
    ) -> None:
        """
        Initialise the nodeStats class.

        Parameters
        ----------
        filename : str
            The name of the file being processed. For logging purposes.
        image : npt.NDArray
            The array of pixels.
        mask : npt.NDArray
            The binary segmentation mask.
        smoothed_mask : npt.NDArray
            A smoothed version of the bianary segmentation mask.
        skeleton : npt.NDArray
            A binary single-pixel wide mask of objects in the 'image'.
        pixel_to_nm_scaling : float
            The pixel to nm scaling factor.
        n_grain : int
            The grain number.
        node_joining_length : float
            The length over which to join skeletal intersections to be counted as one crossing.
        node_joining_length : float
            The distance over which to join nearby odd-branched nodes.
        node_extend_dist : float
            The distance under which to join odd-branched node regions.
        branch_pairing_length : float
            The length from the crossing point to pair and trace, obtaining FWHM's.
        pair_odd_branches : bool
            Whether to try and pair odd-branched nodes.
        """
        self.filename = filename
        self.image = image
        self.mask = mask
        self.smoothed_mask = smoothed_mask  # only used to average traces
        self.skeleton = skeleton
        self.pixel_to_nm_scaling = pixel_to_nm_scaling
        self.n_grain = n_grain
        self.node_joining_length = node_joining_length
        self.node_extend_dist = node_extend_dist / self.pixel_to_nm_scaling
        self.branch_pairing_length = branch_pairing_length
        self.pair_odd_branches = pair_odd_branches

        self.conv_skelly = np.zeros_like(self.skeleton)
        self.connected_nodes = np.zeros_like(self.skeleton)
        self.all_connected_nodes = np.zeros_like(self.skeleton)
        self.whole_skel_graph: nx.classes.graph.Graph | None = None
        self.node_centre_mask = np.zeros_like(self.skeleton)

        self.metrics = {
            "num_crossings": np.int64(0),
            "avg_crossing_confidence": None,
            "min_crossing_confidence": None,
        }

        self.node_dicts: dict[str, NodeDict] = {}
        self.image_dict: ImageDict = {
            "nodes": {},
            "grain": {
                "grain_image": self.image,
                "grain_mask": self.mask,
                "grain_skeleton": self.skeleton,
            },
        }

        self.full_dict = {}
        self.mol_coords = {}
        self.visuals = {}
        self.all_visuals_img = None

    def get_node_stats(self) -> tuple[dict, dict]:
        """
        Run the workflow to obtain the node statistics.

        .. code-block:: RST

            node_dict key structure:  <grain_number>
                                        └-> <node_number>
                                            |-> 'error'
                                            └-> 'node_coords'
                                            └-> 'branch_stats'
                                                └-> <branch_number>
                                                    |-> 'ordered_coords'
                                                    |-> 'heights'
                                                    |-> 'gaussian_fit'
                                                    |-> 'fwhm'
                                                    └-> 'angles'

            image_dict key structure:  'nodes'
                                            <node_number>
                                                |-> 'node_area_skeleton'
                                                |-> 'node_branch_mask'
                                                └-> 'node_avg_mask
                                        'grain'
                                            |-> 'grain_image'
                                            |-> 'grain_mask'
                                            └-> 'grain_skeleton'

        Returns
        -------
        tuple[dict, dict]
            Dictionaries of the node_information and images.
        """
        LOGGER.debug(f"Node Stats - Processing Grain: {self.n_grain}")
        self.conv_skelly = convolve_skeleton(self.skeleton)
        if len(self.conv_skelly[self.conv_skelly == 3]) != 0:  # check if any nodes
            LOGGER.debug(f"[{self.filename}] : Nodestats - {self.n_grain} contains crossings.")
            # convolve to see crossing and end points
            # self.conv_skelly = self.tidy_branches(self.conv_skelly, self.image)
            # reset skeleton var as tidy branches may have modified it
            self.skeleton = np.where(self.conv_skelly != 0, 1, 0)
            self.image_dict["grain"]["grain_skeleton"] = self.skeleton
            # get graph of skeleton
            self.whole_skel_graph = self.skeleton_image_to_graph(self.skeleton)
            # connect the close nodes
            LOGGER.debug(f"[{self.filename}] : Nodestats - {self.n_grain} connecting close nodes.")
            self.connected_nodes = self.connect_close_nodes(self.conv_skelly, node_width=self.node_joining_length)
            # connect the odd-branch nodes
            self.connected_nodes = self.connect_extended_nodes_nearest(
                self.connected_nodes, node_extend_dist=self.node_extend_dist
            )
            # obtain a mask of node centers and their count
            self.node_centre_mask = self.highlight_node_centres(self.connected_nodes)
            # Begin the hefty crossing analysis
            LOGGER.debug(f"[{self.filename}] : Nodestats - {self.n_grain} analysing found crossings.")
            self.analyse_nodes(max_branch_length=self.branch_pairing_length)
            self.compile_metrics()
        else:
            LOGGER.debug(f"[{self.filename}] : Nodestats - {self.n_grain} has no crossings.")
        return self.node_dicts, self.image_dict
        # self.all_visuals_img = dnaTrace.concat_images_in_dict(self.image.shape, self.visuals)

    @staticmethod
    def skeleton_image_to_graph(skeleton: npt.NDArray) -> nx.classes.graph.Graph:
        """
        Convert a skeletonised mask into a Graph representation.

        Graphs conserve the coordinates via the node label.

        Parameters
        ----------
        skeleton : npt.NDArray
            A binary single-pixel wide mask, or result from conv_skelly().

        Returns
        -------
        nx.classes.graph.Graph
            A networkX graph connecting the pixels in the skeleton to their neighbours.
        """
        skeImPos = np.argwhere(skeleton).T
        g = nx.Graph()
        neigh = np.array([[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]])

        for idx in range(skeImPos[0].shape[0]):
            for neighIdx in range(neigh.shape[0]):
                curNeighPos = skeImPos[:, idx] + neigh[neighIdx]
                if np.any(curNeighPos < 0) or np.any(curNeighPos >= skeleton.shape):
                    continue
                if skeleton[curNeighPos[0], curNeighPos[1]] > 0:
                    idx_coord = skeImPos[0, idx], skeImPos[1, idx]
                    curNeigh_coord = curNeighPos[0], curNeighPos[1]
                    # assign lower weight to nodes if not a binary image
                    if skeleton[idx_coord] == 3 and skeleton[curNeigh_coord] == 3:
                        weight = 0
                    else:
                        weight = 1
                    g.add_edge(idx_coord, curNeigh_coord, weight=weight)
        g.graph["physicalPos"] = skeImPos.T
        return g

    @staticmethod
    def graph_to_skeleton_image(g: nx.Graph, im_shape: tuple[int]) -> npt.NDArray:
        """
        Convert the skeleton graph back to a binary image.

        Parameters
        ----------
        g : nx.Graph
            Graph with coordinates as node labels.
        im_shape : tuple[int]
            The shape of the image to dump.

        Returns
        -------
        npt.NDArray
            Skeleton binary image from the graph representation.
        """
        im = np.zeros(im_shape)
        for node in g:
            im[node] = 1

        return im

    def tidy_branches(self, connect_node_mask: npt.NDArray, image: npt.NDArray) -> npt.NDArray:
        """
        Wrangle distant connected nodes back towards the main cluster.

        Works by filling and reskeletonising soely the node areas.

        Parameters
        ----------
        connect_node_mask : npt.NDArray
            The connected node mask - a skeleton where node regions = 3, endpoints = 2, and skeleton = 1.
        image : npt.NDArray
            The intensity image.

        Returns
        -------
        npt.NDArray
            The wrangled connected_node_mask.
        """
        new_skeleton = np.where(connect_node_mask != 0, 1, 0)
        labeled_nodes = label(np.where(connect_node_mask == 3, 1, 0))
        for node_num in range(1, labeled_nodes.max() + 1):
            solo_node = np.where(labeled_nodes == node_num, 1, 0)
            coords = np.argwhere(solo_node == 1)
            node_centre = coords.mean(axis=0).astype(np.int32)
            node_wid = coords[:, 0].max() - coords[:, 0].min() + 2  # +2 so always 2 by default
            node_len = coords[:, 1].max() - coords[:, 1].min() + 2  # +2 so always 2 by default
            overflow = int(10 / self.pixel_to_nm_scaling) if int(10 / self.pixel_to_nm_scaling) != 0 else 1
            # grain mask fill
            new_skeleton[
                node_centre[0] - node_wid // 2 - overflow : node_centre[0] + node_wid // 2 + overflow,
                node_centre[1] - node_len // 2 - overflow : node_centre[1] + node_len // 2 + overflow,
            ] = self.mask[
                node_centre[0] - node_wid // 2 - overflow : node_centre[0] + node_wid // 2 + overflow,
                node_centre[1] - node_len // 2 - overflow : node_centre[1] + node_len // 2 + overflow,
            ]
        # remove any artifacts of the grain caught in the overflow areas
        new_skeleton = self.keep_biggest_object(new_skeleton)
        # Re-skeletonise
        new_skeleton = getSkeleton(image, new_skeleton, method="topostats", height_bias=0.6).get_skeleton()
        # new_skeleton = pruneSkeleton(image, new_skeleton).prune_skeleton(
        #     {"method": "topostats", "max_length": -1}
        # )
        new_skeleton = prune_skeleton(
            image, new_skeleton, self.pixel_to_nm_scaling, **{"method": "topostats", "max_length": -1}
        )
        # cleanup around nibs
        new_skeleton = getSkeleton(image, new_skeleton, method="zhang").get_skeleton()
        # might also need to remove segments that have squares connected

        return convolve_skeleton(new_skeleton)

    @staticmethod
    def keep_biggest_object(mask: npt.NDArray) -> npt.NDArray:
        """
        Retain the largest object in a binary mask.

        Parameters
        ----------
        mask : npt.NDArray
            Binary mask.

        Returns
        -------
        npt.NDArray
            A binary mask with only one object.
        """
        labelled_mask = label(mask)
        idxs, counts = np.unique(mask, return_counts=True)
        try:
            max_idx = idxs[np.argmax(counts[1:]) + 1]
            return np.where(labelled_mask == max_idx, 1, 0)
        except ValueError as e:
            LOGGER.debug(f"{e}: mask is empty.")
            return mask

    def connect_close_nodes(self, conv_skelly: npt.NDArray, node_width: float = 2.85) -> npt.NDArray:
        """
        Connect nodes within the 'node_width' boundary distance.

        This labels them as part of the same node.

        Parameters
        ----------
        conv_skelly : npt.NDArray
            A labeled skeleton image with skeleton = 1, endpoints = 2, crossing points =3.
        node_width : float
            The width of the dna in the grain, used to connect close nodes.

        Returns
        -------
        np.ndarray
            The skeleton (label=1) with close nodes connected (label=3).
        """
        self.connected_nodes = conv_skelly.copy()
        nodeless = conv_skelly.copy()
        nodeless[(nodeless == 3) | (nodeless == 2)] = 0  # remove node & termini points
        nodeless_labels = label(nodeless)
        for i in range(1, nodeless_labels.max() + 1):
            if nodeless[nodeless_labels == i].size < (node_width / self.pixel_to_nm_scaling):
                # maybe also need to select based on height? and also ensure small branches classified
                self.connected_nodes[nodeless_labels == i] = 3

        return self.connected_nodes

    def highlight_node_centres(self, mask: npt.NDArray) -> npt.NDArray:
        """
        Calculate the node centres based on height and re-plot on the mask.

        Parameters
        ----------
        mask : npt.NDArray
            2D array with background = 0, skeleton = 1, endpoints = 2, node_centres = 3.

        Returns
        -------
        npt.NDArray
            2D array with the highest node coordinate for each node labeled as 3.
        """
        small_node_mask = mask.copy()
        small_node_mask[mask == 3] = 1  # remap nodes to skeleton
        big_nodes = mask.copy()
        big_nodes = np.where(mask == 3, 1, 0)  # remove non-nodes & set nodes to 1
        big_node_mask = label(big_nodes)

        for i in np.delete(np.unique(big_node_mask), 0):  # get node indices
            centre = np.unravel_index((self.image * (big_node_mask == i).astype(int)).argmax(), self.image.shape)
            small_node_mask[centre] = 3

        return small_node_mask

    def connect_extended_nodes_nearest(
        self, connected_nodes: npt.NDArray, node_extend_dist: float = -1
    ) -> npt.NDArray[np.int32]:
        """
        Extend the odd branched nodes to other odd branched nodes within the 'extend_dist' threshold.

        Parameters
        ----------
        connected_nodes : npt.NDArray
            A 2D array representing the network with background = 0, skeleton = 1, endpoints = 2,
            node_centres = 3.
        node_extend_dist : int | float, optional
            The distance over which to connect odd-branched nodes, by default -1 for no-limit.

        Returns
        -------
        npt.NDArray[np.int32]
            Connected nodes array with odd-branched nodes connected.
        """
        just_nodes = np.where(connected_nodes == 3, 1, 0)  # remove branches & termini points
        labelled_nodes = label(just_nodes)

        just_branches = np.where(connected_nodes == 1, 1, 0)  # remove node & termini points
        just_branches[connected_nodes == 1] = labelled_nodes.max() + 1
        labelled_branches = label(just_branches)

        nodes_with_branch_starting_coords = find_branches_for_nodes(
            network_array_representation=connected_nodes,
            labelled_nodes=labelled_nodes,
            labelled_branches=labelled_branches,
        )

        # If there is only one node, then there is no need to connect the nodes since there is nothing to
        # connect it to. Return the original connected_nodes instead.
        if len(nodes_with_branch_starting_coords) <= 1:
            self.connected_nodes = connected_nodes
            return self.connected_nodes

        assert self.whole_skel_graph is not None, "Whole skeleton graph is not defined."  # for type safety
        shortest_node_dists, shortest_dists_branch_idxs, _shortest_dist_coords = calculate_shortest_branch_distances(
            nodes_with_branch_starting_coords=nodes_with_branch_starting_coords,
            whole_skeleton_graph=self.whole_skel_graph,
        )

        # Matches is an Nx2 numpy array of indexes of the best matching nodes.
        # Eg: np.array([[1, 0], [2, 3]]) means that the best matching nodes are
        # node 1 and node 0, and node 2 and node 3.
        matches: npt.NDArray[np.int32] = self.best_matches(shortest_node_dists, max_weight_matching=False)

        # Connect the nodes by their best matches, using the shortest distances between their branch starts.
        connected_nodes = connect_best_matches(
            network_array_representation=connected_nodes,
            whole_skeleton_graph=self.whole_skel_graph,
            match_indexes=matches,
            shortest_distances_between_nodes=shortest_node_dists,
            shortest_distances_branch_indexes=shortest_dists_branch_idxs,
            emanating_branch_starts_by_node=nodes_with_branch_starting_coords,
            extend_distance=node_extend_dist,
        )

        self.connected_nodes = connected_nodes
        return self.connected_nodes

    @staticmethod
    def find_branch_starts(reduced_node_image: npt.NDArray) -> npt.NDArray:
        """
        Find the coordinates where the branches connect to the node region through binary dilation of the node.

        Parameters
        ----------
        reduced_node_image : npt.NDArray
            A 2D numpy array containing a single node region (=3) and its connected branches (=1).

        Returns
        -------
        npt.NDArray
            Coordinate array of pixels next to crossing points (=3 in input).
        """
        node = np.where(reduced_node_image == 3, 1, 0)
        nodeless = np.where(reduced_node_image == 1, 1, 0)
        thick_node = binary_dilation(node, structure=np.ones((3, 3)))

        return np.argwhere(thick_node * nodeless == 1)

    # pylint: disable=too-many-locals
    def analyse_nodes(self, max_branch_length: float = 20) -> None:
        """
        Obtain the main analyses for the nodes of a single molecule along the 'max_branch_length'(nm) from the node.

        Parameters
        ----------
        max_branch_length : float
            The side length of the box around the node to analyse (in nm).
        """
        # Get coordinates of nodes
        # This is a numpy array of coords, shape Nx2
        assert self.node_centre_mask is not None, "Node centre mask is not defined."
        node_coords: npt.NDArray[np.int32] = np.argwhere(self.node_centre_mask.copy() == 3)

        # Check whether average trace resides inside the grain mask
        # Checks if we dilate the skeleton once or twice, then all the pixels should fit in the grain mask
        dilate = binary_dilation(self.skeleton, iterations=2)
        # This flag determines whether to use average of 3 traces in calculation of FWHM
        average_trace_advised = dilate[self.smoothed_mask == 1].sum() == dilate.sum()
        LOGGER.debug(f"[{self.filename}] : Branch height traces will be averaged: {average_trace_advised}")

        # Iterate over the nodes and analyse the branches
        matched_branches = None
        branch_image = None
        avg_image = np.zeros_like(self.image)
        real_node_count = 0
        for node_no, (node_x, node_y) in enumerate(node_coords):
            unmatched_branches = {}
            error = False

            # Get branches relevant to the node
            max_length_px = max_branch_length / (self.pixel_to_nm_scaling * 1)
            reduced_node_area: npt.NDArray[np.int32] = nodeStats.only_centre_branches(
                self.connected_nodes, np.array([node_x, node_y])
            )
            # Reduced skel graph is a networkx graph of the reduced node area.
            reduced_skel_graph: nx.classes.graph.Graph = nodeStats.skeleton_image_to_graph(reduced_node_area)

            # Binarise the reduced node area
            branch_mask = reduced_node_area.copy()
            branch_mask[branch_mask == 3] = 0
            branch_mask[branch_mask == 2] = 1
            node_coords = np.argwhere(reduced_node_area == 3)

            # Find the starting coordinates of any branches connected to the node
            branch_start_coords = self.find_branch_starts(reduced_node_area)

            # Stop processing if nib (node has 2 branches)
            if branch_start_coords.shape[0] <= 2:
                LOGGER.debug(
                    f"node {node_no} has only two branches - skipped & nodes removed.{len(node_coords)}"
                    "pixels in nib node."
                )
            else:
                try:
                    real_node_count += 1
                    LOGGER.debug(f"Node: {real_node_count}")

                    # Analyse the node branches
                    (
                        pairs,
                        matched_branches,
                        ordered_branches,
                        masked_image,
                        branch_under_over_order,
                        confidence,
                        singlet_branch_vectors,
                    ) = nodeStats.analyse_node_branches(
                        p_to_nm=self.pixel_to_nm_scaling,
                        reduced_node_area=reduced_node_area,
                        branch_start_coords=branch_start_coords,
                        max_length_px=max_length_px,
                        reduced_skeleton_graph=reduced_skel_graph,
                        image=self.image,
                        average_trace_advised=average_trace_advised,
                        node_coord=(node_x, node_y),
                        pair_odd_branches=self.pair_odd_branches,
                        filename=self.filename,
                        resolution_threshold=np.float64(1000 / 512),
                    )

                    # Add the analysed branches to the labelled image
                    branch_image, avg_image = nodeStats.add_branches_to_labelled_image(
                        branch_under_over_order=branch_under_over_order,
                        matched_branches=matched_branches,
                        masked_image=masked_image,
                        branch_start_coords=branch_start_coords,
                        ordered_branches=ordered_branches,
                        pairs=pairs,
                        average_trace_advised=average_trace_advised,
                        image_shape=(self.image.shape[0], self.image.shape[1]),
                    )

                    # Calculate crossing angles of unpaired branches and add to stats dict
                    nodestats_calc_singlet_angles_result = nodeStats.calc_angles(np.asarray(singlet_branch_vectors))
                    angles_between_singlet_branch_vectors: npt.NDArray[np.float64] = (
                        nodestats_calc_singlet_angles_result[0]
                    )

                    for branch_index, angle in enumerate(angles_between_singlet_branch_vectors):
                        unmatched_branches[branch_index] = {"angles": angle}

                    # Get the vector of each branch based on ordered_coords. Ordered_coords is only the first N nm
                    # of the branch so this is just a general vibe on what direction a branch is going.
                    if len(branch_start_coords) % 2 == 0 or self.pair_odd_branches:
                        vectors: list[npt.NDArray[np.float64]] = []
                        for _, values in matched_branches.items():
                            vectors.append(nodeStats.get_vector(values["ordered_coords"], np.array([node_x, node_y])))
                        # Calculate angles between the vectors
                        nodestats_calc_angles_result = nodeStats.calc_angles(np.asarray(vectors))
                        angles_between_vectors_along_branch: npt.NDArray[np.float64] = nodestats_calc_angles_result[0]
                        for branch_index, angle in enumerate(angles_between_vectors_along_branch):
                            if len(branch_start_coords) % 2 == 0 or self.pair_odd_branches:
                                matched_branches[branch_index]["angles"] = angle
                    else:
                        self.image_dict["grain"]["grain_skeleton"][node_coords[:, 0], node_coords[:, 1]] = 0

                    # Eg: length 2 array: [array([ nan, 79.00]), array([79.00, 0.0])]
                    # angles_between_vectors_along_branch

                except ResolutionError:
                    LOGGER.debug(f"Node stats skipped as resolution too low: {self.pixel_to_nm_scaling}nm per pixel")
                    error = True

                self.node_dicts[f"node_{real_node_count}"] = {
                    "error": error,
                    "pixel_to_nm_scaling": self.pixel_to_nm_scaling,
                    "branch_stats": matched_branches,
                    "unmatched_branch_stats": unmatched_branches,
                    "node_coords": node_coords,
                    "confidence": confidence,
                }

                assert reduced_node_area is not None, "Reduced node area is not defined."
                assert branch_image is not None, "Branch image is not defined."
                assert avg_image is not None, "Average image is not defined."
                node_images_dict: dict[str, npt.NDArray[np.int32]] = {
                    "node_area_skeleton": reduced_node_area,
                    "node_branch_mask": branch_image,
                    "node_avg_mask": avg_image,
                }
                self.image_dict["nodes"][f"node_{real_node_count}"] = node_images_dict

            self.all_connected_nodes[self.connected_nodes != 0] = self.connected_nodes[self.connected_nodes != 0]

    # pylint: disable=too-many-arguments
    @staticmethod
    def add_branches_to_labelled_image(
        branch_under_over_order: npt.NDArray[np.int32],
        matched_branches: dict[int, MatchedBranch],
        masked_image: dict[int, dict[str, npt.NDArray[np.bool_]]],
        branch_start_coords: npt.NDArray[np.int32],
        ordered_branches: list[npt.NDArray[np.int32]],
        pairs: npt.NDArray[np.int32],
        average_trace_advised: bool,
        image_shape: tuple[int, int],
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        """
        Add branches to a labelled image.

        Parameters
        ----------
        branch_under_over_order : npt.NDArray[np.int32]
            The order of the branches.
        matched_branches : dict[int, dict[str, MatchedBranch]]
            Dictionary where the key is the index of the pair and the value is a dictionary containing the following
            keys:
            - "ordered_coords" : npt.NDArray[np.int32].
            - "heights" : npt.NDArray[np.number]. Heights of the branches.
            - "distances" :
            - "fwhm" : npt.NDArray[np.number]. Full width half maximum of the branches.
        masked_image : dict[int, dict[str, npt.NDArray[np.bool_]]]
            Dictionary where the key is the index of the pair and the value is a dictionary containing the following
            keys:
            - "avg_mask" : npt.NDArray[np.bool_]. Average mask of the branches.
        branch_start_coords : npt.NDArray[np.int32]
            An Nx2 numpy array of the coordinates of the branches connected to the node.
        ordered_branches : list[npt.NDArray[np.int32]]
            List of numpy arrays of ordered branch coordinates.
        pairs : npt.NDArray[np.int32]
            Nx2 numpy array of pairs of branches that are matched through a node.
        average_trace_advised : bool
            Flag to determine whether to use the average trace.
        image_shape : tuple[int]
            The shape of the image, to create a mask from.

        Returns
        -------
        tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]
            The branch image and the average image.
        """
        branch_image: npt.NDArray[np.int32] = np.zeros(image_shape).astype(np.int32)
        avg_image: npt.NDArray[np.int32] = np.zeros(image_shape).astype(np.int32)

        for i, branch_index in enumerate(branch_under_over_order):
            branch_coords = matched_branches[branch_index]["ordered_coords"]

            # Add the matched branch to the image, starting at index 1
            branch_image[branch_coords[:, 0], branch_coords[:, 1]] = i + 1
            if average_trace_advised:
                # For type safety, check if avg_image is None and skip if so.
                # This is because the type hinting does not allow for None in the array.
                avg_image[masked_image[branch_index]["avg_mask"] != 0] = i + 1

        # Determine branches that were not able to be paired
        unpaired_branches = np.delete(np.arange(0, branch_start_coords.shape[0]), pairs.flatten())
        LOGGER.debug(f"Unpaired branches: {unpaired_branches}")
        # Ensure that unpaired branches start at index I where I is the number of paired branches.
        branch_label = branch_image.max()
        # Add the unpaired branches back to the branch image
        for i in unpaired_branches:
            branch_label += 1
            branch_image[ordered_branches[i][:, 0], ordered_branches[i][:, 1]] = branch_label

        return branch_image, avg_image

    @staticmethod
    def analyse_node_branches(
        p_to_nm: np.float64,
        reduced_node_area: npt.NDArray[np.int32],
        branch_start_coords: npt.NDArray[np.int32],
        max_length_px: np.float64,
        reduced_skeleton_graph: nx.classes.graph.Graph,
        image: npt.NDArray[np.number],
        average_trace_advised: bool,
        node_coord: tuple[np.int32, np.int32],
        pair_odd_branches: bool,
        filename: str,
        resolution_threshold: np.float64,
    ) -> tuple[
        npt.NDArray[np.int32],
        dict[int, MatchedBranch],
        list[npt.NDArray[np.int32]],
        dict[int, dict[str, npt.NDArray[np.bool_]]],
        npt.NDArray[np.int32],
        np.float64 | None,
    ]:
        """
        Analyse the branches of a single node.

        Parameters
        ----------
        p_to_nm : np.float64
            The pixel to nm scaling factor.
        reduced_node_area : npt.NDArray[np.int32]
            An NxM numpy array of the node in question and the branches connected to it.
            Node is marked by 3, and branches by 1.
        branch_start_coords : npt.NDArray[np.int32]
            An Nx2 numpy array of the coordinates of the branches connected to the node.
        max_length_px : np.int32
            The maximum length in pixels to traverse along while ordering.
        reduced_skeleton_graph : nx.classes.graph.Graph
            The graph representation of the reduced node area.
        image : npt.NDArray[np.number]
            The full image of the grain.
        average_trace_advised : bool
            Flag to determine whether to use the average trace.
        node_coord : tuple[np.int32, np.int32]
            The node coordinates.
        pair_odd_branches : bool
            Whether to try and pair odd-branched nodes.
        filename : str
            The filename of the image.
        resolution_threshold : np.float64
            The resolution threshold below which to warn the user that the node is difficult to analyse.

        Returns
        -------
        pairs: npt.NDArray[np.int32]
            Nx2 numpy array of pairs of branches that are matched through a node.
        matched_branches: dict[int, MatchedBranch]]
            Dictionary where the key is the index of the pair and the value is a dictionary containing the following
            keys:
            - "ordered_coords" : npt.NDArray[np.int32].
            - "heights" : npt.NDArray[np.number]. Heights of the branches.
            - "distances" : npt.NDArray[np.number]. The accumulating distance along the branch.
            - "fwhm" : npt.NDArray[np.number]. Full width half maximum of the branches.
            - "angles" : np.float64. The angle of the branch, added in later steps.
        ordered_branches: list[npt.NDArray[np.int32]]
            List of numpy arrays of ordered branch coordinates.
        masked_image: dict[int, dict[str, npt.NDArray[np.bool_]]]
            Dictionary where the key is the index of the pair and the value is a dictionary containing the following
            keys:
            - "avg_mask" : npt.NDArray[np.bool_]. Average mask of the branches.
        branch_under_over_order: npt.NDArray[np.int32]
            The order of the branches based on the FWHM.
        confidence: np.float64 | None
            The confidence of the crossing. Optional.
        """
        if not p_to_nm <= resolution_threshold:
            LOGGER.debug(f"Resolution {p_to_nm} is below suggested {resolution_threshold}, node difficult to analyse.")

        # Pixel-wise order the branches coming from the node and calculate the starting vector for each branch
        ordered_branches, singlet_branch_vectors = nodeStats.get_ordered_branches_and_vectors(
            reduced_node_area, branch_start_coords, max_length_px
        )

        # Pair the singlet branch vectors based on their suitability using vector orientation.
        if len(branch_start_coords) % 2 == 0 or pair_odd_branches:
            pairs = nodeStats.pair_vectors(np.asarray(singlet_branch_vectors))
        else:
            pairs = np.array([], dtype=np.int32)

        # Match the branches up
        matched_branches, masked_image = nodeStats.join_matching_branches_through_node(
            pairs,
            ordered_branches,
            reduced_skeleton_graph,
            image,
            average_trace_advised,
            node_coord,
            filename,
        )

        # Redo the FWHMs after the processing for more accurate determination of under/overs.
        hms = []
        for _, values in matched_branches.items():
            hms.append(values["fwhm"]["half_maxs"][2])
        for _, values in matched_branches.items():
            values["fwhm"] = nodeStats.calculate_fwhm(values["heights"], values["distances"], hm=max(hms))

        # Get the confidence of the crossing
        crossing_fwhms = []
        for _, values in matched_branches.items():
            crossing_fwhms.append(values["fwhm"]["fwhm"])
        if len(crossing_fwhms) <= 1:
            confidence = None
        else:
            crossing_fwhm_combinations = list(combinations(crossing_fwhms, 2))
            confidence = np.float64(nodeStats.cross_confidence(crossing_fwhm_combinations))

        # Order the branch indexes based on the FWHM of the branches.
        branch_under_over_order = np.array(list(matched_branches.keys()))[np.argsort(np.array(crossing_fwhms))]

        return (
            pairs,
            matched_branches,
            ordered_branches,
            masked_image,
            branch_under_over_order,
            confidence,
            singlet_branch_vectors,
        )

    @staticmethod
    def join_matching_branches_through_node(
        pairs: npt.NDArray[np.int32],
        ordered_branches: list[npt.NDArray[np.int32]],
        reduced_skeleton_graph: nx.classes.graph.Graph,
        image: npt.NDArray[np.number],
        average_trace_advised: bool,
        node_coords: tuple[np.int32, np.int32],
        filename: str,
    ) -> tuple[dict[int, MatchedBranch], dict[int, dict[str, npt.NDArray[np.bool_]]]]:
        """
        Join branches that are matched through a node.

        Parameters
        ----------
        pairs : npt.NDArray[np.int32]
            Nx2 numpy array of pairs of branches that are matched through a node.
        ordered_branches : list[npt.NDArray[np.int32]]
            List of numpy arrays of ordered branch coordinates.
        reduced_skeleton_graph : nx.classes.graph.Graph
            Graph representation of the skeleton.
        image : npt.NDArray[np.number]
            The full image of the grain.
        average_trace_advised : bool
            Flag to determine whether to use the average trace.
        node_coords : tuple[np.int32, np.int32]
            The node coordinates.
        filename : str
            The filename of the image.

        Returns
        -------
        matched_branches: dict[int, dict[str, npt.NDArray[np.number]]]
            Dictionary where the key is the index of the pair and the value is a dictionary containing the following
            keys:
            - "ordered_coords" : npt.NDArray[np.int32].
            - "heights" : npt.NDArray[np.number]. Heights of the branches.
            - "distances" :
            - "fwhm" : npt.NDArray[np.number]. Full width half maximum of the branches.
        masked_image: dict[int, dict[str, npt.NDArray[np.bool_]]]
            Dictionary where the key is the index of the pair and the value is a dictionary containing the following
            keys:
            - "avg_mask" : npt.NDArray[np.bool_]. Average mask of the branches.
        """
        matched_branches: dict[int, MatchedBranch] = {}
        masked_image: dict[int, dict[str, npt.NDArray[np.bool_]]] = (
            {}
        )  # Masked image is a dictionary of pairs of branches
        for i, (branch_1, branch_2) in enumerate(pairs):
            matched_branches[i] = MatchedBranch(
                ordered_coords=np.array([], dtype=np.int32),
                heights=np.array([], dtype=np.float64),
                distances=np.array([], dtype=np.float64),
                fwhm={},
                angles=None,
            )
            masked_image[i] = {}
            # find close ends by rearranging branch coords
            branch_1_coords, branch_2_coords = nodeStats.order_branches(
                ordered_branches[branch_1], ordered_branches[branch_2]
            )
            # Get graphical shortest path between branch ends on the skeleton
            crossing = nx.shortest_path(
                reduced_skeleton_graph,
                source=tuple(branch_1_coords[-1]),
                target=tuple(branch_2_coords[0]),
                weight="weight",
            )
            crossing = np.asarray(crossing[1:-1])  # remove start and end points & turn into array
            # Branch coords and crossing
            if crossing.shape == (0,):
                branch_coords = np.vstack([branch_1_coords, branch_2_coords])
            else:
                branch_coords = np.vstack([branch_1_coords, crossing, branch_2_coords])
            # make images of single branch joined and multiple branches joined
            single_branch_img: npt.NDArray[np.bool_] = np.zeros_like(image).astype(bool)
            single_branch_img[branch_coords[:, 0], branch_coords[:, 1]] = True
            single_branch_coords = order_branch(single_branch_img.astype(bool), [0, 0])
            # calc image-wide coords
            matched_branches[i]["ordered_coords"] = single_branch_coords
            # get heights and trace distance of branch
            try:
                assert average_trace_advised
                distances, heights, mask, _ = nodeStats.average_height_trace(
                    image, single_branch_img, single_branch_coords, [node_coords[0], node_coords[1]]
                )
                masked_image[i]["avg_mask"] = mask
            except (
                AssertionError,
                IndexError,
            ) as e:  # Assertion - avg trace not advised, Index - wiggy branches
                LOGGER.debug(f"[{filename}] : avg trace failed with {e}, single trace only.")
                average_trace_advised = False
                distances = nodeStats.coord_dist_rad(single_branch_coords, np.array([node_coords[0], node_coords[1]]))
                # distances = self.coord_dist(single_branch_coords)
                zero_dist = distances[
                    np.argmin(
                        np.sqrt(
                            (single_branch_coords[:, 0] - node_coords[0]) ** 2
                            + (single_branch_coords[:, 1] - node_coords[1]) ** 2
                        )
                    )
                ]
                heights = image[single_branch_coords[:, 0], single_branch_coords[:, 1]]  # self.hess
                distances = distances - zero_dist
                distances, heights = nodeStats.average_uniques(
                    distances, heights
                )  # needs to be paired with coord_dist_rad
            matched_branches[i]["heights"] = heights
            matched_branches[i]["distances"] = distances
            # identify over/under
            matched_branches[i]["fwhm"] = nodeStats.calculate_fwhm(heights, distances)

        return matched_branches, masked_image

    @staticmethod
    def get_ordered_branches_and_vectors(
        reduced_node_area: npt.NDArray[np.int32],
        branch_start_coords: npt.NDArray[np.int32],
        max_length_px: np.float64,
    ) -> tuple[list[npt.NDArray[np.int32]], list[npt.NDArray[np.int32]]]:
        """
        Get ordered branches and vectors for a node.

        Branches are ordered so they are no longer just a disordered set of coordinates, and vectors are calculated to
        represent the general direction tendency of the branch, this allows for alignment matching later on.

        Parameters
        ----------
        reduced_node_area : npt.NDArray[np.int32]
            An NxM numpy array of the node in question and the branches connected to it.
            Node is marked by 3, and branches by 1.
        branch_start_coords : npt.NDArray[np.int32]
            An Px2 numpy array of coordinates representing the start of branches where P is the number of branches.
        max_length_px : np.int32
            The maximum length in pixels to traverse along while ordering.

        Returns
        -------
        tuple[list[npt.NDArray[np.int32]], list[npt.NDArray[np.int32]]]
            A tuple containing a list of ordered branches and a list of vectors.
        """
        ordered_branches = []
        vectors = []
        nodeless = np.where(reduced_node_area == 1, 1, 0)
        for branch_start_coord in branch_start_coords:
            # Order the branch coordinates so they're no longer just a disordered set of coordinates
            ordered_branch = order_branch_from_start(nodeless.copy(), branch_start_coord, max_length=max_length_px)
            ordered_branches.append(ordered_branch)

            # Calculate vector to represent the general direction tendency of the branch (for alignment matching)
            vector = nodeStats.get_vector(ordered_branch, branch_start_coord)
            vectors.append(vector)

        return ordered_branches, vectors

    @staticmethod
    def cross_confidence(pair_combinations: list) -> float:
        """
        Obtain the average confidence of the combinations using a reciprical function.

        Parameters
        ----------
        pair_combinations : list
            List of length 2 combinations of FWHM values.

        Returns
        -------
        float
            The average crossing confidence.
        """
        c = 0
        for pair in pair_combinations:
            c += nodeStats.recip(pair)
        return c / len(pair_combinations)

    @staticmethod
    def recip(vals: list) -> float:
        """
        Compute 1 - (max / min) of the two values provided.

        Parameters
        ----------
        vals : list
            List of 2 values.

        Returns
        -------
        float
            Result of applying the 1-(min / max) function to the two values.
        """
        try:
            if min(vals) == 0:  # means fwhm variation hasn't worked
                return 0
            return 1 - min(vals) / max(vals)
        except ZeroDivisionError:
            return 0

    @staticmethod
    def get_vector(coords: npt.NDArray, origin: npt.NDArray) -> npt.NDArray:
        """
        Calculate the normalised vector of the coordinate means in a branch.

        Parameters
        ----------
        coords : npt.NDArray
            2xN array of x, y coordinates.
        origin : npt.NDArray
            2x1 array of an x, y coordinate.

        Returns
        -------
        npt.NDArray
            Normalised vector from origin to the mean coordinate.
        """
        vector = coords.mean(axis=0) - origin
        norm = np.sqrt(vector @ vector)
        return vector if norm == 0 else vector / norm  # normalise vector so length=1

    @staticmethod
    def calc_angles(vectors: npt.NDArray) -> npt.NDArray[np.float64]:
        """
        Calculate the angles between vectors in an array.

        Uses the formula:

        .. code-block:: RST

            cos(theta) = |a|•|b|/|a||b|

        Parameters
        ----------
        vectors : npt.NDArray
            Array of 2x1 vectors.

        Returns
        -------
        npt.NDArray
            An array of the cosine of the angles between the vectors.
        """
        dot = vectors @ vectors.T
        norm = np.diag(dot) ** 0.5
        cos_angles = dot / (norm.reshape(-1, 1) @ norm.reshape(1, -1))
        np.fill_diagonal(cos_angles, 1)  # ensures vector_x • vector_x angles are 0
        return abs(np.arccos(cos_angles) / np.pi * 180)  # angles in degrees

    @staticmethod
    def pair_vectors(vectors: npt.NDArray) -> npt.NDArray[np.int32]:
        """
        Take a list of vectors and pairs them based on the angle between them.

        Parameters
        ----------
        vectors : npt.NDArray
            Array of 2x1 vectors to be paired.

        Returns
        -------
        npt.NDArray
            An array of the matching pair indices.
        """
        # calculate cosine of angle
        angles = nodeStats.calc_angles(vectors)
        # match angles
        return nodeStats.best_matches(angles)

    @staticmethod
    def best_matches(arr: npt.NDArray, max_weight_matching: bool = True) -> npt.NDArray:
        """
        Turn a matrix into a graph and calculates the best matching index pairs.

        Parameters
        ----------
        arr : npt.NDArray
            Transpose symmetric MxM array where the value of index i, j represents a weight between i and j.
        max_weight_matching : bool
            Whether to obtain best matching pairs via maximum weight, or minimum weight matching.

        Returns
        -------
        npt.NDArray
            Array of pairs of indexes.
        """
        if max_weight_matching:
            G = nodeStats.create_weighted_graph(arr)
            matching = np.array(list(nx.max_weight_matching(G, maxcardinality=True)))
        else:
            np.fill_diagonal(arr, arr.max() + 1)
            G = nodeStats.create_weighted_graph(arr)
            matching = np.array(list(nx.min_weight_matching(G)))
        return matching

    @staticmethod
    def create_weighted_graph(matrix: npt.NDArray) -> nx.Graph:
        """
        Create a bipartite graph connecting i <-> j from a square matrix of weights matrix[i, j].

        Parameters
        ----------
        matrix : npt.NDArray
            Square array of weights between rows and columns.

        Returns
        -------
        nx.Graph
            Bipatrite graph with edge weight i->j matching matrix[i,j].
        """
        n = len(matrix)
        G = nx.Graph()
        for i in range(n):
            for j in range(i + 1, n):
                G.add_edge(i, j, weight=matrix[i, j])
        return G

    @staticmethod
    def pair_angles(angles: npt.NDArray) -> list:
        """
        Pair angles that are 180 degrees to each other and removes them before selecting the next pair.

        Parameters
        ----------
        angles : npt.NDArray
             Square array (i,j) of angles between i and j.

        Returns
        -------
        list
             A list of paired indexes in a list.
        """
        angles_cp = angles.copy()
        pairs = []
        for _ in range(int(angles.shape[0] / 2)):
            pair = np.unravel_index(np.argmax(angles_cp), angles.shape)
            pairs.append(pair)  # add to list
            angles_cp[[pair]] = 0  # set rows 0 to avoid picking again
            angles_cp[:, [pair]] = 0  # set cols 0 to avoid picking again

        return np.asarray(pairs)

    @staticmethod
    def gaussian(x: npt.NDArray, h: float, mean: float, sigma: float):
        """
        Apply the gaussian function.

        Parameters
        ----------
        x : npt.NDArray
            X values to be passed into the gaussian.
        h : float
            The peak height of the gaussian.
        mean : float
            The mean of the x values.
        sigma : float
            The standard deviation of the image.

        Returns
        -------
        npt.NDArray
            The y-values of the gaussian performed on the x values.
        """
        return h * np.exp(-((x - mean) ** 2) / (2 * sigma**2))

    @staticmethod
    def interpolate_between_yvalue(x: npt.NDArray, y: npt.NDArray, yvalue: float) -> float:
        """
        Calculate the x value between the two points either side of yvalue in y.

        Parameters
        ----------
        x : npt.NDArray
            An array of length y.
        y : npt.NDArray
            An array of length x.
        yvalue : float
            A value within the bounds of the y array.

        Returns
        -------
        float
            The linearly interpolated x value between the arrays.
        """
        for i in range(len(y) - 1):
            if y[i] <= yvalue <= y[i + 1] or y[i + 1] <= yvalue <= y[i]:  # if points cross through the hm value
                return nodeStats.lin_interp([x[i], y[i]], [x[i + 1], y[i + 1]], yvalue=yvalue)
        return 0

    @staticmethod
    def calculate_fwhm(
        heights: npt.NDArray, distances: npt.NDArray, hm: float | None = None
    ) -> dict[str, np.float64 | list[np.float64 | float | None]]:
        """
        Calculate the FWHM value.

        First identifyies the HM then finding the closest values in the distances array and using
        linear interpolation to calculate the FWHM.

        Parameters
        ----------
        heights : npt.NDArray
            Array of heights.
        distances : npt.NDArray
            Array of distances.
        hm : Union[None, float], optional
            The halfmax value to match (if wanting the same HM between curves), by default None.

        Returns
        -------
        tuple[float, list, list]
            The FWHM value, [distance at hm for 1st half of trace, distance at hm for 2nd half of trace,
            HM value], [index of the highest point, distance at highest point, height at highest point].
        """
        centre_fraction = int(len(heights) * 0.2)  # in case zone approaches another node, look around centre for max
        if centre_fraction == 0:
            high_idx = np.argmax(heights)
        else:
            high_idx = np.argmax(heights[centre_fraction:-centre_fraction]) + centre_fraction
        # get array halves to find first points that cross hm
        arr1 = heights[:high_idx][::-1]
        dist1 = distances[:high_idx][::-1]
        arr2 = heights[high_idx:]
        dist2 = distances[high_idx:]
        if hm is None:
            # Get half max
            hm = (heights.max() - heights.min()) / 2 + heights.min()
            # half max value -> try to make it the same as other crossing branch?
            # increase make hm = lowest of peak if it doesn’t hit one side
            if np.min(arr1) > hm:
                arr1_local_min = argrelextrema(arr1, np.less)[-1]  # closest to end
                try:
                    hm = arr1[arr1_local_min][0]
                except IndexError:  # index error when no local minima
                    hm = np.min(arr1)
            elif np.min(arr2) > hm:
                arr2_local_min = argrelextrema(arr2, np.less)[0]  # closest to start
                try:
                    hm = arr2[arr2_local_min][0]
                except IndexError:  # index error when no local minima
                    hm = np.min(arr2)
        arr1_hm = nodeStats.interpolate_between_yvalue(x=dist1, y=arr1, yvalue=hm)
        arr2_hm = nodeStats.interpolate_between_yvalue(x=dist2, y=arr2, yvalue=hm)
        fwhm = np.float64(abs(arr2_hm - arr1_hm))
        return {
            "fwhm": fwhm,
            "half_maxs": [arr1_hm, arr2_hm, hm],
            "peaks": [high_idx, distances[high_idx], heights[high_idx]],
        }

    @staticmethod
    def lin_interp(point_1: list, point_2: list, xvalue: float | None = None, yvalue: float | None = None) -> float:
        """
        Linear interp 2 points by finding line equation and subbing.

        Parameters
        ----------
        point_1 : list
            List of an x and y coordinate.
        point_2 : list
            List of an x and y coordinate.
        xvalue : Union[float, None], optional
            Value at which to interpolate to get a y coordinate, by default None.
        yvalue : Union[float, None], optional
            Value at which to interpolate to get an x coordinate, by default None.

        Returns
        -------
        float
            Value of x or y linear interpolation.
        """
        m = (point_1[1] - point_2[1]) / (point_1[0] - point_2[0])
        c = point_1[1] - (m * point_1[0])
        if xvalue is not None:
            return m * xvalue + c  # interp_y
        if yvalue is not None:
            return (yvalue - c) / m  # interp_x
        raise ValueError

    @staticmethod
    def order_branches(branch1: npt.NDArray, branch2: npt.NDArray) -> tuple:
        """
        Order the two ordered arrays based on the closest endpoint coordinates.

        Parameters
        ----------
        branch1 : npt.NDArray
            An Nx2 array describing coordinates.
        branch2 : npt.NDArray
            An Nx2 array describing coordinates.

        Returns
        -------
        tuple
            An tuple with the each coordinate array ordered to follow on from one-another.
        """
        endpoints1 = np.asarray([branch1[0], branch1[-1]])
        endpoints2 = np.asarray([branch2[0], branch2[-1]])
        sum1 = abs(endpoints1 - endpoints2).sum(axis=1)
        sum2 = abs(endpoints1[::-1] - endpoints2).sum(axis=1)
        if sum1.min() < sum2.min():
            if np.argmin(sum1) == 0:
                return branch1[::-1], branch2
            return branch1, branch2[::-1]
        if np.argmin(sum2) == 0:
            return branch1, branch2
        return branch1[::-1], branch2[::-1]

    @staticmethod
    def binary_line(start: npt.NDArray, end: npt.NDArray) -> npt.NDArray:
        """
        Create a binary path following the straight line between 2 points.

        Parameters
        ----------
        start : npt.NDArray
            A coordinate.
        end : npt.NDArray
            Another coordinate.

        Returns
        -------
        npt.NDArray
            An Nx2 coordinate array that the line passes through.
        """
        arr = []
        m_swap = False
        x_swap = False
        slope = (end - start)[1] / (end - start)[0]

        if abs(slope) > 1:  # swap x and y if slope will cause skips
            start, end = start[::-1], end[::-1]
            slope = 1 / slope
            m_swap = True

        if start[0] > end[0]:  # swap x coords if coords wrong way around
            start, end = end, start
            x_swap = True

        # code assumes slope < 1 hence swap
        x_start, y_start = start
        x_end, _ = end
        for x in range(x_start, x_end + 1):
            y_true = slope * (x - x_start) + y_start
            y_pixel = np.round(y_true)
            arr.append([x, y_pixel])

        if m_swap:  # if swapped due to slope, return
            arr = np.asarray(arr)[:, [1, 0]].reshape(-1, 2).astype(int)
            if x_swap:
                return arr[::-1]
            return arr
        arr = np.asarray(arr).reshape(-1, 2).astype(int)
        if x_swap:
            return arr[::-1]
        return arr

    @staticmethod
    def coord_dist_rad(coords: npt.NDArray, centre: npt.NDArray, pixel_to_nm_scaling: float = 1) -> npt.NDArray:
        """
        Calculate the distance from the centre coordinate to a point along the ordered coordinates.

        This differs to traversal along the coordinates taken. This also averages any common distance
        values and makes those in the trace before the node index negative.

        Parameters
        ----------
        coords : npt.NDArray
            Nx2 array of branch coordinates.
        centre : npt.NDArray
            A 1x2 array of the centre coordinates to identify a 0 point for the node.
        pixel_to_nm_scaling : float, optional
            The pixel to nanometer scaling factor to provide real units, by default 1.

        Returns
        -------
        npt.NDArray
            A Nx1 array of the distance from the node centre.
        """
        diff_coords = coords - centre
        if np.all(coords == centre, axis=1).sum() == 0:  # if centre not in coords, reassign centre
            diff_dists = np.sqrt(diff_coords[:, 0] ** 2 + diff_coords[:, 1] ** 2)
            centre = coords[np.argmin(diff_dists)]
        cross_idx = np.argwhere(np.all(coords == centre, axis=1))
        rad_dist = np.sqrt(diff_coords[:, 0] ** 2 + diff_coords[:, 1] ** 2)
        rad_dist[0 : cross_idx[0][0]] *= -1
        return rad_dist * pixel_to_nm_scaling

    @staticmethod
    def above_below_value_idx(array: npt.NDArray, value: float) -> list:
        """
        Identify indices of the array neighbouring the specified value.

        Parameters
        ----------
        array : npt.NDArray
            Array of values.
        value : float
            Value to identify indices between.

        Returns
        -------
        list
            List of the lower index and higher index around the value.

        Raises
        ------
        IndexError
            When the value is in the array.
        """
        idx1 = abs(array - value).argmin()
        try:
            if array[idx1] < value < array[idx1 + 1]:
                idx2 = idx1 + 1
            elif array[idx1 - 1] < value < array[idx1]:
                idx2 = idx1 - 1
            else:
                raise IndexError  # this will be if the number is the same
            indices = [idx1, idx2]
            indices.sort()
            return indices
        except IndexError:
            return None

    @staticmethod
    def average_height_trace(  # noqa: C901
        img: npt.NDArray, branch_mask: npt.NDArray, branch_coords: npt.NDArray, centre=(0, 0)
    ) -> tuple:
        """
        Average two side-by-side ordered skeleton distance and height traces.

        Dilate the original branch to create two additional side-by-side branches
        in order to get a more accurate average of the height traces. This function produces
        the common distances between these 3 branches, and their averaged heights.

        Parameters
        ----------
        img : npt.NDArray
            An array of numbers pertaining to an image.
        branch_mask : npt.NDArray
            A binary array of the branch, must share the same dimensions as the image.
        branch_coords : npt.NDArray
            Ordered coordinates of the branch mask.
        centre : Union[float, None]
            The coordinates to centre the branch around.

        Returns
        -------
        tuple
            A tuple of the averaged heights from the linetrace and their corresponding distances
            from the crossing.
        """
        # get heights and dists of the original (middle) branch
        branch_dist = nodeStats.coord_dist_rad(branch_coords, centre)
        # branch_dist = self.coord_dist(branch_coords)
        branch_heights = img[branch_coords[:, 0], branch_coords[:, 1]]
        branch_dist, branch_heights = nodeStats.average_uniques(
            branch_dist, branch_heights
        )  # needs to be paired with coord_dist_rad
        dist_zero_point = branch_dist[
            np.argmin(np.sqrt((branch_coords[:, 0] - centre[0]) ** 2 + (branch_coords[:, 1] - centre[1]) ** 2))
        ]
        branch_dist_norm = branch_dist - dist_zero_point  # - 0  # branch_dist[branch_heights.argmax()]

        # want to get a 3 pixel line trace, one on each side of orig
        dilate = binary_dilation(branch_mask, iterations=1)
        dilate = nodeStats.fill_holes(dilate)
        dilate_minus = np.where(dilate != branch_mask, 1, 0)
        dilate2 = binary_dilation(dilate, iterations=1)
        dilate2[(dilate == 1) | (branch_mask == 1)] = 0
        labels = label(dilate2)
        # Cleanup stages - re-entering, early terminating, closer traces
        #   if parallel trace out and back in zone, can get > 2 labels
        labels = nodeStats._remove_re_entering_branches(labels, remaining_branches=2)
        #   if parallel trace doesn't exit window, can get 1 label
        #       occurs when skeleton has poor connections (extra branches which cut corners)
        if labels.max() == 1:
            conv = convolve_skeleton(branch_mask)
            endpoints = np.argwhere(conv == 2)
            for endpoint in endpoints:  # may be >1 endpoint
                para_trace_coords = np.argwhere(labels == 1)
                abs_diff = np.absolute(para_trace_coords - endpoint).sum(axis=1)
                min_idxs = np.where(abs_diff == abs_diff.min())
                trace_coords_remove = para_trace_coords[min_idxs]
                labels[trace_coords_remove[:, 0], trace_coords_remove[:, 1]] = 0
            labels = label(labels)
        #   reduce binary dilation distance
        parallel = np.zeros_like(branch_mask).astype(np.int32)
        for i in range(1, labels.max() + 1):
            single = labels.copy()
            single[single != i] = 0
            single[single == i] = 1
            sing_dil = binary_dilation(single)
            parallel[(sing_dil == dilate_minus) & (sing_dil == 1)] = i
        labels = parallel.copy()

        binary = labels.copy()
        binary[binary != 0] = 1
        binary += branch_mask

        # get and order coords, then get heights and distances relitive to node centre / highest point
        heights = []
        distances = []
        for i in np.unique(labels)[1:]:
            trace_img = np.where(labels == i, 1, 0)
            trace_img = getSkeleton(img, trace_img, method="zhang").get_skeleton()
            trace = order_branch(trace_img, branch_coords[0])
            height_trace = img[trace[:, 0], trace[:, 1]]
            dist = nodeStats.coord_dist_rad(trace, centre)  # self.coord_dist(trace)
            dist, height_trace = nodeStats.average_uniques(dist, height_trace)  # needs to be paired with coord_dist_rad
            heights.append(height_trace)
            distances.append(
                dist - dist_zero_point  # - 0
            )  # branch_dist[branch_heights.argmax()]) #dist[central_heights.argmax()])
        # Make like coord system using original branch
        avg1 = []
        avg2 = []
        for mid_dist in branch_dist_norm:
            for i, (distance, height) in enumerate(zip(distances, heights)):
                # check if distance already in traces array
                if (mid_dist == distance).any():
                    idx = np.where(mid_dist == distance)
                    if i == 0:
                        avg1.append([mid_dist, height[idx][0]])
                    else:
                        avg2.append([mid_dist, height[idx][0]])
                # if not, linearly interpolate the mid-branch value
                else:
                    # get index after and before the mid branches' x coord
                    xidxs = nodeStats.above_below_value_idx(distance, mid_dist)
                    if xidxs is None:
                        pass  # if indexes outside of range, pass
                    else:
                        point1 = [distance[xidxs[0]], height[xidxs[0]]]
                        point2 = [distance[xidxs[1]], height[xidxs[1]]]
                        y = nodeStats.lin_interp(point1, point2, xvalue=mid_dist)
                        if i == 0:
                            avg1.append([mid_dist, y])
                        else:
                            avg2.append([mid_dist, y])
        avg1 = np.asarray(avg1)
        avg2 = np.asarray(avg2)
        # ensure arrays are same length to average
        temp_x = branch_dist_norm[np.isin(branch_dist_norm, avg1[:, 0])]
        common_dists = avg2[:, 0][np.isin(avg2[:, 0], temp_x)]

        common_avg_branch_heights = branch_heights[np.isin(branch_dist_norm, common_dists)]
        common_avg1_heights = avg1[:, 1][np.isin(avg1[:, 0], common_dists)]
        common_avg2_heights = avg2[:, 1][np.isin(avg2[:, 0], common_dists)]

        average_heights = (common_avg_branch_heights + common_avg1_heights + common_avg2_heights) / 3
        return (
            common_dists,
            average_heights,
            binary,
            [[heights[0], branch_heights, heights[1]], [distances[0], branch_dist_norm, distances[1]]],
        )

    @staticmethod
    def fill_holes(mask: npt.NDArray) -> npt.NDArray:
        """
        Fill all holes within a binary mask.

        Parameters
        ----------
        mask : npt.NDArray
            Binary array of object.

        Returns
        -------
        npt.NDArray
            Binary array of object with any interior holes filled in.
        """
        inv_mask = np.where(mask != 0, 0, 1)
        lbl_inv = label(inv_mask, connectivity=1)
        idxs, counts = np.unique(lbl_inv, return_counts=True)
        max_idx = idxs[np.argmax(counts)]
        return np.where(lbl_inv != max_idx, 1, 0)

    @staticmethod
    def _remove_re_entering_branches(mask: npt.NDArray, remaining_branches: int = 1) -> npt.NDArray:
        """
        Remove smallest branches which branches exit and re-enter the viewing area.

        Contninues until only <remaining_branches> remain.

        Parameters
        ----------
        mask : npt.NDArray
            Skeletonised binary mask of an object.
        remaining_branches : int, optional
            Number of objects (branches) to keep, by default 1.

        Returns
        -------
        npt.NDArray
            Mask with only a single skeletonised branch.
        """
        rtn_image = mask.copy()
        binary_image = mask.copy()
        binary_image[binary_image != 0] = 1
        labels = label(binary_image)

        if labels.max() > remaining_branches:
            lens = [labels[labels == i].size for i in range(1, labels.max() + 1)]
            while len(lens) > remaining_branches:
                smallest_idx = min(enumerate(lens), key=lambda x: x[1])[0]
                rtn_image[labels == smallest_idx + 1] = 0
                lens.remove(min(lens))

        return rtn_image

    @staticmethod
    def only_centre_branches(node_image: npt.NDArray, node_coordinate: npt.NDArray) -> npt.NDArray[np.int32]:
        """
        Remove all branches not connected to the current node.

        Parameters
        ----------
        node_image : npt.NDArray
            An image of the skeletonised area surrounding the node where
            the background = 0, skeleton = 1, termini = 2, nodes = 3.
        node_coordinate : npt.NDArray
            2x1 coordinate describing the position of a node.

        Returns
        -------
        npt.NDArray[np.int32]
            The initial node image but only with skeletal branches
            connected to the middle node.
        """
        node_image_cp = node_image.copy()

        # get node-only image
        nodes = node_image_cp.copy()
        nodes[nodes != 3] = 0
        labeled_nodes = label(nodes)

        # find which cluster is closest to the centre
        node_coords = np.argwhere(nodes == 3)
        min_coords = node_coords[abs(node_coords - node_coordinate).sum(axis=1).argmin()]
        centre_idx = labeled_nodes[min_coords[0], min_coords[1]]

        # get nodeless image
        nodeless = node_image_cp.copy()
        nodeless = np.where(
            (node_image == 1) | (node_image == 2), 1, 0
        )  # if termini, need this in the labeled branches too
        nodeless[labeled_nodes == centre_idx] = 1  # return centre node
        labeled_nodeless = label(nodeless)

        # apply to return image
        for i in range(1, labeled_nodeless.max() + 1):
            if (node_image_cp[labeled_nodeless == i] == 3).any():
                node_image_cp[labeled_nodeless != i] = 0
                break

        # remove small area around other nodes
        labeled_nodes[labeled_nodes == centre_idx] = 0
        non_central_node_coords = np.argwhere(labeled_nodes != 0)
        for coord in non_central_node_coords:
            for j, coord_val in enumerate(coord):
                if coord_val - 1 < 0:
                    coord[j] = 1
                if coord_val + 2 > node_image_cp.shape[j]:
                    coord[j] = node_image_cp.shape[j] - 2
            node_image_cp[coord[0] - 1 : coord[0] + 2, coord[1] - 1 : coord[1] + 2] = 0

        return node_image_cp

    @staticmethod
    def average_uniques(arr1: npt.NDArray, arr2: npt.NDArray) -> tuple:
        """
        Obtain the unique values of both arrays, and the average of common values.

        Parameters
        ----------
        arr1 : npt.NDArray
            An array.
        arr2 : npt.NDArray
            An array.

        Returns
        -------
        tuple
            The unique values of both arrays, and the averaged common values.
        """
        arr1_uniq, index = np.unique(arr1, return_index=True)
        arr2_new = np.zeros_like(arr1_uniq).astype(np.float64)
        for i, val in enumerate(arr1[index]):
            mean = arr2[arr1 == val].mean()
            arr2_new[i] += mean

        return arr1[index], arr2_new

    @staticmethod
    def average_crossing_confs(node_dict) -> None | float:
        """
        Return the average crossing confidence of all crossings in the molecule.

        Parameters
        ----------
        node_dict : dict
            A dictionary containing node statistics and information.

        Returns
        -------
        Union[None, float]
            The value of minimum confidence or none if not possible.
        """
        sum_conf = 0
        valid_confs = 0
        for _, (_, values) in enumerate(node_dict.items()):
            confidence = values["confidence"]
            if confidence is not None:
                sum_conf += confidence
                valid_confs += 1
        try:
            return sum_conf / valid_confs
        except ZeroDivisionError:
            return None

    @staticmethod
    def minimum_crossing_confs(node_dict: dict) -> None | float:
        """
        Return the minimum crossing confidence of all crossings in the molecule.

        Parameters
        ----------
        node_dict : dict
            A dictionary containing node statistics and information.

        Returns
        -------
        Union[None, float]
            The value of minimum confidence or none if not possible.
        """
        confidences = []
        valid_confs = 0
        for _, (_, values) in enumerate(node_dict.items()):
            confidence = values["confidence"]
            if confidence is not None:
                confidences.append(confidence)
                valid_confs += 1
        try:
            return min(confidences)
        except ValueError:
            return None

    def compile_metrics(self) -> None:
        """Add the number of crossings, average and minimum crossing confidence to the metrics dictionary."""
        self.metrics["num_crossings"] = np.int64((self.node_centre_mask == 3).sum())
        self.metrics["avg_crossing_confidence"] = np.float64(nodeStats.average_crossing_confs(self.node_dicts))
        self.metrics["min_crossing_confidence"] = np.float64(nodeStats.minimum_crossing_confs(self.node_dicts))


def nodestats_image(
    image: npt.NDArray,
    disordered_tracing_direction_data: dict,
    filename: str,
    pixel_to_nm_scaling: float,
    node_joining_length: float,
    node_extend_dist: float,
    branch_pairing_length: float,
    pair_odd_branches: float,
) -> tuple:
    """
    Initialise the nodeStats class.

    Parameters
    ----------
    image : npt.NDArray
        The array of pixels.
    disordered_tracing_direction_data : dict
        The images and bbox coordinates of the pruned skeletons.
    filename : str
        The name of the file being processed. For logging purposes.
    pixel_to_nm_scaling : float
        The pixel to nm scaling factor.
    node_joining_length : float
        The length over which to join skeletal intersections to be counted as one crossing.
    node_joining_length : float
        The distance over which to join nearby odd-branched nodes.
    node_extend_dist : float
        The distance under which to join odd-branched node regions.
    branch_pairing_length : float
        The length from the crossing point to pair and trace, obtaining FWHM's.
    pair_odd_branches : bool
        Whether to try and pair odd-branched nodes.

    Returns
    -------
    tuple[dict, pd.DataFrame, dict, dict]
        The nodestats statistics for each crossing, crossing statistics to be added to the grain statistics,
        an image dictionary of nodestats steps for the entire image, and single grain images.
    """
    n_grains = len(disordered_tracing_direction_data)
    img_base = np.zeros_like(image)
    nodestats_data = {}

    # Images for diagnostics edited during processing
    all_images = {
        "convolved_skeletons": img_base.copy(),
        "node_centres": img_base.copy(),
        "connected_nodes": img_base.copy(),
    }
    nodestats_branch_images = {}
    grainstats_additions = {}

    LOGGER.info(f"[{filename}] : Calculating NodeStats statistics for {n_grains} grains...")

    for n_grain, disordered_tracing_grain_data in disordered_tracing_direction_data.items():
        nodestats = None  # reset the nodestats variable
        try:
            nodestats = nodeStats(
                image=disordered_tracing_grain_data["original_image"],
                mask=disordered_tracing_grain_data["original_grain"],
                smoothed_mask=disordered_tracing_grain_data["smoothed_grain"],
                skeleton=disordered_tracing_grain_data["pruned_skeleton"],
                pixel_to_nm_scaling=pixel_to_nm_scaling,
                filename=filename,
                n_grain=n_grain,
                node_joining_length=node_joining_length,
                node_extend_dist=node_extend_dist,
                branch_pairing_length=branch_pairing_length,
                pair_odd_branches=pair_odd_branches,
            )
            nodestats_dict, node_image_dict = nodestats.get_node_stats()
            LOGGER.debug(f"[{filename}] : Nodestats processed {n_grain} of {n_grains}")

            # compile images
            nodestats_images = {
                "convolved_skeletons": nodestats.conv_skelly,
                "node_centres": nodestats.node_centre_mask,
                "connected_nodes": nodestats.connected_nodes,
            }
            nodestats_branch_images[n_grain] = node_image_dict

            # compile metrics
            grainstats_additions[n_grain] = {
                "image": filename,
                "grain_number": int(n_grain.split("_")[-1]),
            }
            grainstats_additions[n_grain].update(nodestats.metrics)
            if nodestats_dict:  # if the grain's nodestats dict is not empty
                nodestats_data[n_grain] = nodestats_dict

            # remap the cropped images back onto the original
            for image_name, full_image in all_images.items():
                crop = nodestats_images[image_name]
                bbox = disordered_tracing_grain_data["bbox"]
                full_image[bbox[0] : bbox[2], bbox[1] : bbox[3]] += crop

        except Exception as e:  # pylint: disable=broad-exception-caught
            LOGGER.error(
                f"[{filename}] : Nodestats for {n_grain} failed. Consider raising an issue on GitHub. Error: ",
                exc_info=e,
            )
            nodestats_data[n_grain] = {}

        # turn the grainstats additions into a dataframe, # might need to do something for when everything is empty
        grainstats_additions_df = pd.DataFrame.from_dict(grainstats_additions, orient="index")

    return nodestats_data, grainstats_additions_df, all_images, nodestats_branch_images
