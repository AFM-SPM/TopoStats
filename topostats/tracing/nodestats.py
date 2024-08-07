"""Perform Crossing Region Processing and Analysis."""

from __future__ import annotations

import logging

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
from topostats.utils import ResolutionError, convolve_skeleton, coords_2_img

LOGGER = logging.getLogger(LOGGER_NAME)


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
    px_2_nm : float
        The pixel to nm scaling factor.
    n_grain : int
        The grain number.
    node_joining_length : float
        The length over which to join skeletal intersections to be counted as one crossing.
    node_joining_length : float
        The distance over which to join nearby odd-branched nodes.
    branch_pairing_length : float
        The length from the crossing point to pair and trace, obtaining FWHM's.
    """

    def __init__(
        self,
        filename: str,
        image: npt.NDArray,
        mask: npt.NDArray,
        smoothed_mask: npt.NDArray,
        skeleton: npt.NDArray,
        px_2_nm: float,
        n_grain: int,
        node_joining_length: float,
        node_extend_dist: float,
        branch_pairing_length: float,
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
        px_2_nm : float
            The pixel to nm scaling factor.
        n_grain : int
            The grain number.
        node_joining_length : float
            The length over which to join skeletal intersections to be counted as one crossing.
        node_joining_length : float
            The distance over which to join nearby odd-branched nodes.
        branch_pairing_length : float
            The length from the crossing point to pair and trace, obtaining FWHM's.
        """
        self.filename = filename
        self.image = image
        self.mask = mask
        self.smoothed_mask = smoothed_mask  # only used to average traces
        self.skeleton = skeleton
        self.px_2_nm = px_2_nm * 1e-9
        self.n_grain = n_grain
        self.node_joining_length = node_joining_length
        self.node_extend_dist = node_extend_dist / self.px_2_nm
        self.branch_pairing_length = branch_pairing_length

        self.conv_skelly = np.zeros_like(self.skeleton)
        self.connected_nodes = np.zeros_like(self.skeleton)
        self.all_connected_nodes = np.zeros_like(self.skeleton)
        self.whole_skel_graph = None
        self.node_centre_mask = np.zeros_like(self.skeleton)

        self.metrics = {
            "num_crossings": 0,
            "avg_crossing_confidence": None,
            "min_crossing_confidence": None,
        }

        self.node_dict = {}
        self.image_dict = {
            "nodes": {},
            "grain": {
                "grain_image": self.image,
                "grain_mask": self.mask,
            },
        }

        self.full_dict = {}
        self.mol_coords = {}
        self.visuals = {}
        self.all_visuals_img = None

    def get_node_stats(self) -> tuple:
        """
        Run the workflow to obtain the node statistics.

        Returns
        -------
        dict
            Key structure:  <grain_number>
                            |-> <node_number>
                                |-> 'error'
                                |-> 'branch_stats'
                                    |-> <branch_number>
                                        |-> 'ordered_coords'
                                        |-> 'heights'
                                        |-> 'gaussian_fit'
                                        |-> 'fwhm'
                                        |-> 'angles'
                                |-> 'node_coords'
        dict
            Key structure:  'nodes'
                                <node_number>
                                    |-> 'node_area_skeleton'
                                    |-> 'node_branch_mask'
                                    |-> 'node_avg_mask
                            'grain'
                                |-> 'grain_image'
                                |-> 'grain_mask'
        """
        LOGGER.info(f"Node Stats - Processing Grain: {self.n_grain}")
        self.conv_skelly = convolve_skeleton(self.skeleton)
        if len(self.conv_skelly[self.conv_skelly == 3]) != 0:  # check if any nodes
            LOGGER.info(f"[{self.filename}] : Nodestats - {self.n_grain} contains crossings.")
            # convolve to see crossing and end points
            self.conv_skelly = self.tidy_branches(self.conv_skelly, self.image)
            # reset skeleton var as tidy branches may have modified it
            self.skeleton = np.where(self.conv_skelly != 0, 1, 0)
            # get graph of skeleton
            self.whole_skel_graph = self.skeleton_image_to_graph(self.skeleton)
            # connect the close nodes
            LOGGER.info(f"[{self.filename}] : Nodestats - {self.n_grain} connecting close nodes.")
            self.connected_nodes = self.connect_close_nodes(self.conv_skelly, node_width=self.node_joining_length)
            # connect the odd-branch nodes
            self.connected_nodes = self.connect_extended_nodes_nearest(
                self.connected_nodes, node_extend_dist=self.node_extend_dist
            )
            # obtain a mask of node centers and their count
            self.node_centre_mask = self.highlight_node_centres(self.connected_nodes)
            print("CENT: ", self.node_centre_mask)
            # Begin the hefty crossing analysis
            LOGGER.info(f"[{self.filename}] : Nodestats - {self.n_grain} analysing found crossings.")
            self.analyse_nodes(max_branch_length=self.branch_pairing_length)
            self.compile_metrics()
        else:
            LOGGER.info(f"[{self.filename}] : Nodestats - {self.n_grain} has no crossings.")
        return self.node_dict, self.image_dict
        # self.all_visuals_img = dnaTrace.concat_images_in_dict(self.image.shape, self.visuals)

    @staticmethod
    def skeleton_image_to_graph(skeleton: npt.NDArray) -> nx.Graph:
        """
        Convert a skeletonised mask into a Graph representation.

        Graphs conserve the coordinates via the node label.

        Parameters
        ----------
        skeleton : npt.NDArray
            A binary single-pixel wide mask, or result from conv_skelly().

        Returns
        -------
        nx.Graph
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

    # TODO: Maybe move to skeletonisation
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
            overflow = int(10e-9 / self.px_2_nm) if int(10e-9 / self.px_2_nm) != 0 else 1
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
        new_skeleton = prune_skeleton(image, new_skeleton, **{"method": "topostats", "max_length": -1})
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
            LOGGER.info(f"{e}: mask is empty.")
            return mask

    def connect_close_nodes(self, conv_skelly: npt.NDArray, node_width: float = 2.85e-9) -> npt.NDArray:
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
            if nodeless[nodeless_labels == i].size < (node_width / self.px_2_nm):
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
        connected_nodes : npt.NDArra
            A 2D array representing the network with background = 0, skeleton = 1, endpoints = 2,
            node_centres = 3.
        node_extend_dist : int | float, optional
            The distance over which to connect odd-branched nodes, by default -1 for no-limit.

        Returns
        -------
        npt.NDArra[np.int32]
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
        thicc_node = binary_dilation(node, structure=np.ones((3, 3)))

        return np.argwhere(thicc_node * nodeless == 1)

    def analyse_nodes(self, max_branch_length: float = 20e-9) -> None:
        """
        Obtain the main analyses for the nodes of a single molecule along the 'max_branch_length' (nm) from the node.

        Parameters
        ----------
        max_branch_length : float
            The side length of the box around the node to analyse (in nm).
        """
        # get coordinates of nodes
        xy_arr = np.argwhere(self.node_centre_mask.copy() == 3)

        # check whether average trace resides inside the grain mask
        dilate = binary_dilation(self.skeleton, iterations=2)
        average_trace_advised = dilate[self.smoothed_mask == 1].sum() == dilate.sum()
        LOGGER.info(f"[{self.filename}] : Branch height traces will be averaged: {average_trace_advised}")

        # iterate over the nodes to find areas
        matched_branches = None
        branch_img = None
        avg_img = None

        real_node_count = 0
        for node_no, (x, y) in enumerate(xy_arr):  # get centres
            # get area around node
            max_length_px = max_branch_length / self.px_2_nm

            # reduce the skeleton area
            reduced_node_area = self._only_centre_branches(self.connected_nodes, (x, y))
            self.reduced_skel_graph = self.skeleton_image_to_graph(reduced_node_area)
            branch_mask = reduced_node_area.copy()

            branch_mask[branch_mask == 3] = 0
            branch_mask[branch_mask == 2] = 1
            node_coords = np.argwhere(reduced_node_area == 3)

            error = False  # to see if node too complex or region too small

            branch_start_coords = self.find_branch_starts(reduced_node_area)

            # stop processing if nib (node has 2 branches)
            if branch_start_coords.shape[0] <= 2:
                LOGGER.info(f"node {node_no} has only two branches - skipped & nodes removed")
                # sometimes removal of nibs can cause problems when re-indexing nodes
                LOGGER.info(f"{len(node_coords)} pixels in nib node")
                # TODO: node coords might be missaligned
                # self.node_centre_mask[node_coords[:, 0], node_coords[:, 0]] = 1  # remove these from node_centre_mask
                # self.connected_nodes[node_coords[:, 0], node_coords[:, 1]] = 1  # remove these from connected_nodes
            else:
                try:
                    # check whether resolution good enough to trace
                    res = self.px_2_nm <= 1000 / 512
                    if not res:
                        print(f"Resolution {res} is below suggested {1000 / 512}, node difficult to analyse.")
                    # raise ResolutionError
                    # elif x - length < 0 or y - length < 0 or
                    #   x + length > self.image.shape[0] or
                    #   y + length > self.image.shape[1]:
                    # LOGGER.info(f"Node lies too close to image boundary, increase 'pad_with' value.")
                    # raise ResolutionError

                    real_node_count += 1
                    print(f"Real node: {real_node_count}")
                    ordered_branches = []
                    vectors = []
                    nodeless = np.where(reduced_node_area == 1, 1, 0)
                    for branch_start_coord in branch_start_coords:
                        # order branch
                        ordered = self.order_branch_from_start(
                            nodeless.copy(), branch_start_coord, max_length=max_length_px
                        )
                        # identify vector
                        vector = self.get_vector(ordered, branch_start_coord)  # [x, y]
                        # add to list
                        vectors.append(vector)
                        ordered_branches.append(ordered)
                    if node_no == 0:
                        self.test2 = vectors
                    # pair vectors
                    # print(f"NODE {real_node_count}, vectors:\n {vectors}")
                    pairs = self.pair_vectors(np.asarray(vectors))

                    # join matching branches through node
                    matched_branches = {}
                    masked_image = {}
                    branch_img = np.zeros_like(self.skeleton)  # initialising paired branch img
                    avg_img = np.zeros_like(self.skeleton)
                    for i, (branch_1, branch_2) in enumerate(pairs):
                        matched_branches[i] = {}
                        masked_image[i] = {}
                        # find close ends by rearranging branch coords
                        branch_1_coords, branch_2_coords = self.order_branches(
                            ordered_branches[branch_1], ordered_branches[branch_2]
                        )
                        # Get graphical shortest path between branch ends on the skeleton
                        crossing = nx.shortest_path(
                            self.reduced_skel_graph,
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
                        single_branch_img = np.zeros_like(self.skeleton)
                        single_branch_img[branch_coords[:, 0], branch_coords[:, 1]] = 1
                        single_branch_coords = self.order_branch(single_branch_img, [0, 0])
                        # calc image-wide coords
                        matched_branches[i]["ordered_coords"] = single_branch_coords
                        # get heights and trace distance of branch
                        try:
                            assert average_trace_advised
                            distances, heights, mask, _ = self.average_height_trace(
                                self.image, single_branch_img, single_branch_coords, [x, y]
                            )  # hess_area
                            masked_image[i]["avg_mask"] = mask
                        except (
                            AssertionError,
                            IndexError,
                        ) as e:  # Assertion - avg trace not advised, Index - wiggy branches
                            LOGGER.info(f"[{self.filename}] : avg trace failed with {e}, single trace only.")
                            average_trace_advised = False
                            distances = self.coord_dist_rad(single_branch_coords, [x, y])
                            # distances = self.coord_dist(single_branch_coords)
                            zero_dist = distances[
                                np.argmin(
                                    np.sqrt(
                                        (single_branch_coords[:, 0] - x) ** 2 + (single_branch_coords[:, 1] - y) ** 2
                                    )
                                )
                            ]
                            heights = self.image[single_branch_coords[:, 0], single_branch_coords[:, 1]]  # self.hess
                            distances = distances - zero_dist
                            distances, heights = self.average_uniques(
                                distances, heights
                            )  # needs to be paired with coord_dist_rad
                        matched_branches[i]["heights"] = heights
                        matched_branches[i]["distances"] = distances  # * self.px_2_nm
                        # identify over/under
                        matched_branches[i]["fwhm"] = self.fwhm(heights, distances)

                    # redo fwhms after to get better baselines + same hm matching
                    hms = []
                    for _, values in matched_branches.items():  # get hms
                        hms.append(values["fwhm"]["half_maxs"][2])
                    for branch_idx, values in matched_branches.items():  # use same highest hm
                        fwhm = self.fwhm(values["heights"], values["distances"], hm=max(hms))
                        matched_branches[branch_idx]["fwhm"] = fwhm

                    # get confidences
                    crossing_quants = []
                    for _, values in matched_branches.items():
                        crossing_quants.append(values["fwhm"]["fwhm"])
                    if len(crossing_quants) == 1:  # from 3 eminnating branches
                        conf = None
                    else:
                        combs = self.get_two_combinations(crossing_quants)
                        conf = self.cross_confidence(combs)

                    # add paired and unpaired branches to image plot
                    fwhms = []
                    for _, values in matched_branches.items():
                        fwhms.append(values["fwhm"]["fwhm"])
                    branch_idx_order = np.array(list(matched_branches.keys()))[np.argsort(np.array(fwhms))]
                    # branch_idx_order = np.arange(0,len(matched_branches))
                    # uncomment to unorder (will not unorder the height traces)

                    for i, branch_idx in enumerate(branch_idx_order):
                        branch_coords = matched_branches[branch_idx]["ordered_coords"]
                        branch_img[branch_coords[:, 0], branch_coords[:, 1]] = i + 1  # add to branch img
                        if average_trace_advised:  # add avg traces
                            avg_img[masked_image[branch_idx]["avg_mask"] != 0] = i + 1
                        else:
                            avg_img = None

                    unpaired_branches = np.delete(np.arange(0, branch_start_coords.shape[0]), pairs.flatten())
                    LOGGER.info(f"Unpaired branches: {unpaired_branches}")
                    branch_label = branch_img.max()
                    for i in unpaired_branches:  # carries on from loop variable i
                        branch_label += 1
                        branch_img[ordered_branches[i][:, 0], ordered_branches[i][:, 1]] = branch_label

                    # calc crossing angle
                    # get full branch vectors
                    vectors = []
                    for _, values in matched_branches.items():
                        vectors.append(self.get_vector(values["ordered_coords"], [x, y]))
                    # calc angles to first vector i.e. first should always be 0
                    angles = self.calc_angles(np.asarray(vectors))[0]
                    for i, angle in enumerate(angles):
                        matched_branches[i]["angles"] = angle

                except ResolutionError:
                    LOGGER.info(f"Node stats skipped as resolution too low: {self.px_2_nm}nm per pixel")
                    error = True

                print("Error: ", error)
                self.node_dict[f"node_{real_node_count}"] = {
                    "error": error,
                    "px_2_nm": self.px_2_nm,
                    "branch_stats": matched_branches,
                    "node_coords": node_coords,
                    "confidence": conf,
                }

                self.image_dict["nodes"][f"node_{real_node_count}"] = {
                    "node_area_skeleton": reduced_node_area,
                    "node_branch_mask": branch_img,
                    "node_avg_mask": avg_img,
                }
            self.all_connected_nodes[self.connected_nodes != 0] = self.connected_nodes[self.connected_nodes != 0]

    @staticmethod
    def get_two_combinations(fwhm_list) -> list:
        """
        Obtain all paired combinations of values in the list.

        Example: [1,2] -> [[1,2]], [1,2,3] -> [[1,2],[1,3],[2,3]]

        Parameters
        ----------
        fwhm_list : list
            List of FWHMs from crossing analysis.

        Returns
        -------
        list
            A list of pairs of 'fwhm_list' values.
        """
        combs = []
        for i in range(len(fwhm_list) - 1):
            for j in fwhm_list[i + 1 :]:
                combs.append([fwhm_list[i], j])
        return combs

    def cross_confidence(self, combs: list) -> float:
        """
        Obtain the average confidence of the combinations using a reciprical function.

        Parameters
        ----------
        combs : list
            List of combinations of FWHM values.

        Returns
        -------
        float
            The average crossing confidence.
        """
        c = 0
        for comb in combs:
            c += self.recip(comb)
        return c / len(combs)

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

    def order_branch(self, binary_image: npt.NDArray, anchor: list):
        """
        Order a linear branch by identifying an endpoint, and looking at the local area of the point to find the next.

        Parameters
        ----------
        binary_image : npt.NDArray
            A binary image of a skeleton segment to order it's points.
        anchor : list
            A list of 2 integers representing the coordinate to order the branch from the endpoint closest to this.

        Returns
        -------
        npt.NDArray
            An array of ordered coordinates.
        """
        skel = binary_image.copy()

        if len(np.argwhere(skel == 1)) < 3:  # if < 3 coords just return them
            return np.argwhere(skel == 1)

        # get branch starts
        endpoints_highlight = convolve_skeleton(skel)
        endpoints = np.argwhere(endpoints_highlight == 2)
        if len(endpoints) != 0:  # if any endpoints, start closest to anchor
            dist_vals = abs(endpoints - anchor).sum(axis=1)
            start = endpoints[np.argmin(dist_vals)]
        else:  # will be circular so pick the first coord (is this always the case?)
            start = np.argwhere(skel == 1)[0]
        # order the points according to what is nearby
        ordered = self.order_branch_from_start(skel, start)

        return np.array(ordered)

    def order_branch_from_start(
        self, nodeless: npt.NDArray, start: npt.NDArray, max_length: float | np.inf = np.inf
    ) -> npt.NDArray:
        """
        Order an unbranching skeleton from an end (startpoint) along a specified length.

        Parameters
        ----------
        nodeless : npt.NDArray
            A 2D array of a binary unbranching skeleton.
        start : npt.NDArray
            2x1 coordinate that must exist in 'nodeless'.
        max_length : float | np.inf, optional
            Maximum length to traverse along while ordering, by default np.inf.

        Returns
        -------
        npt.NDArray
            Ordered coordinates.
        """
        dist = 0
        # add starting point to ordered array
        ordered = []
        ordered.append(start)
        nodeless[start[0], start[1]] = 0  # remove from array

        # iterate to order the rest of the points
        current_point = ordered[-1]  # get last point
        area, _ = self.local_area_sum(nodeless, current_point)  # look at local area
        local_next_point = np.argwhere(
            area.reshape(
                (
                    3,
                    3,
                )
            )
            == 1
        ) - (1, 1)
        dist += np.sqrt(2) if abs(local_next_point).sum() > 1 else 1

        while len(local_next_point) != 0 and dist <= max_length:
            next_point = (current_point + local_next_point)[0]
            # find where to go next
            ordered.append(next_point)
            nodeless[next_point[0], next_point[1]] = 0  # set value to zero
            current_point = ordered[-1]  # get last point
            area, _ = self.local_area_sum(nodeless, current_point)  # look at local area
            local_next_point = np.argwhere(
                area.reshape(
                    (
                        3,
                        3,
                    )
                )
                == 1
            ) - (1, 1)
            dist += np.sqrt(2) if abs(local_next_point).sum() > 1 else 1

        return np.array(ordered)

    @staticmethod
    def local_area_sum(binary_map: npt.NDArray, point: list | tuple | npt.NDArray) -> npt.NDArray:
        """
        Evaluate the local area around a point in a binary map.

        Parameters
        ----------
        binary_map : npt.NDArray
            A binary array of an image.
        point : Union[list, tuple, npt.NDArray]
            A single object containing 2 integers relating to a point within the binary_map.

        Returns
        -------
        npt.NDArray
            An array values of the local coordinates around the point.
        int
            A value corresponding to the number of neighbours around the point in the binary_map.
        """
        x, y = point
        local_pixels = binary_map[x - 1 : x + 2, y - 1 : y + 2].flatten()
        local_pixels[4] = 0  # ensure centre is 0
        return local_pixels, local_pixels.sum()

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
    def calc_angles(vectors: npt.NDArray) -> npt.NDArray:
        """
        Calculate the angles between vectors in an array.

        Uses the formula: cos(theta) = |a|•|b|/|a||b|

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
        return abs(np.arccos(cos_angles) / np.pi * 180)  # angles in degrees

    def pair_vectors(self, vectors: npt.NDArray) -> npt.NDArray:
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
        angles = self.calc_angles(vectors)
        # find highest values
        np.fill_diagonal(angles, 0)  # ensures not paired with itself
        # match angles
        return self.best_matches(angles)

    def best_matches(self, arr: npt.NDArray, max_weight_matching: bool = True) -> npt.NDArray:
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
            G = self.create_weighted_graph(arr)
            matching = np.array(list(nx.max_weight_matching(G, maxcardinality=True)))
        else:
            np.fill_diagonal(arr, arr.max() + 1)
            G = self.create_weighted_graph(arr)
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

    def interpolate_between_yvalue(self, x: npt.NDArray, y: npt.NDArray, yvalue: float) -> float:
        """Calculate the x value between the two points either side of yvalue in y.

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
            if y[i] <= yvalue <= y[i + 1]:  # if points cross through the hm value
                return self.lin_interp([x[i], y[i]], [x[i + 1], y[i + 1]], yvalue=yvalue)
        return 0

    def fwhm(self, heights: npt.NDArray, distances: npt.NDArray, hm: float | None = None) -> tuple:
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
            # increase make hm = lowest of peak if it doesn't hit one side
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

        arr1_hm = self.interpolate_between_yvalue(x=dist1, y=arr1, yvalue=hm)
        arr2_hm = self.interpolate_between_yvalue(x=dist2, y=arr2, yvalue=hm)

        fwhm = abs(arr2_hm - arr1_hm)

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
    def coord_dist(coords: npt.NDArray, px_2_nm: float = 1) -> npt.NDArray:
        """
        Accumulate a real distance traversing from pixel to pixel from a list of coordinates.

        Parameters
        ----------
        coords : npt.NDArray
            A Nx2 integer array corresponding to the ordered coordinates of a binary trace.
        px_2_nm : float
            The pixel to nanometer scaling factor.

        Returns
        -------
        npt.NDArray
            An array of length N containing thcumulative sum of the distances.
        """
        dist_list = [0]
        dist = 0
        for i in range(len(coords) - 1):
            if abs(coords[i] - coords[i + 1]).sum() == 2:
                dist += 2**0.5
            else:
                dist += 1
            dist_list.append(dist)
        return np.asarray(dist_list) * px_2_nm

    @staticmethod
    def coord_dist_rad(coords: npt.NDArray, centre: npt.NDArray, px_2_nm: float = 1) -> npt.NDArray:
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
        px_2_nm : float, optional
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
        return rad_dist * px_2_nm

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

    def average_height_trace(
        self, img: npt.NDArray, branch_mask: npt.NDArray, branch_coords: npt.NDArray, centre=(0, 0)
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
        branch_dist = self.coord_dist_rad(branch_coords, centre)
        # branch_dist = self.coord_dist(branch_coords)
        branch_heights = img[branch_coords[:, 0], branch_coords[:, 1]]
        branch_dist, branch_heights = self.average_uniques(
            branch_dist, branch_heights
        )  # needs to be paired with coord_dist_rad
        dist_zero_point = branch_dist[
            np.argmin(np.sqrt((branch_coords[:, 0] - centre[0]) ** 2 + (branch_coords[:, 1] - centre[1]) ** 2))
        ]
        branch_dist_norm = branch_dist - dist_zero_point  # - 0  # branch_dist[branch_heights.argmax()]

        # want to get a 3 pixel line trace, one on each side of orig
        dilate = binary_dilation(branch_mask, iterations=1)
        dilate = self.fill_holes(dilate)
        dilate_minus = np.where(dilate != branch_mask, 1, 0)
        dilate2 = binary_dilation(dilate, iterations=1)
        dilate2[(dilate == 1) | (branch_mask == 1)] = 0
        labels = label(dilate2)
        # Cleanup stages - re-entering, early terminating, closer traces
        #   if parallel trace out and back in zone, can get > 2 labels
        labels = self._remove_re_entering_branches(labels, remaining_branches=2)
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
            trace = self.order_branch(trace_img, branch_coords[0])
            height_trace = img[trace[:, 0], trace[:, 1]]
            dist = self.coord_dist_rad(trace, centre)  # self.coord_dist(trace)
            dist, height_trace = self.average_uniques(dist, height_trace)  # needs to be paired with coord_dist_rad
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
                    xidxs = self.above_below_value_idx(distance, mid_dist)
                    if xidxs is None:
                        pass  # if indexes outside of range, pass
                    else:
                        point1 = [distance[xidxs[0]], height[xidxs[0]]]
                        point2 = [distance[xidxs[1]], height[xidxs[1]]]
                        y = self.lin_interp(point1, point2, xvalue=mid_dist)
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
    def _only_centre_branches(node_image: npt.NDArray, node_coordinate: npt.NDArray):
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
        npt.NDArray
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

    def compile_trace(self) -> tuple:
        """
        Pipeline to obtain the trace and crossing trace image.

        This function uses the branches and FWHM's identified in the node_stats dictionary to create a
        continuous trace of the molecule.

        Returns
        -------
        tuple[list, npt.NDArray]
            A list of each complete path's ordered coordinates, and labeled crosing image array.
        """
        LOGGER.info(f"[{self.filename}] : Compiling the trace.")

        # iterate through the dict to get branch coords, heights and fwhms
        node_coords = []
        crossing_coords = []
        crossing_heights = []
        crossing_distances = []
        fwhms = []
        for _, stats in self.node_dict.items():
            temp_nodes = []
            temp_coords = []
            temp__heights = []
            temp_distances = []
            temp_fwhms = []
            for _, branch_stats in stats["branch_stats"].items():
                temp_coords.append(branch_stats["ordered_coords"])
                temp__heights.append(branch_stats["heights"])
                temp_distances.append(branch_stats["distances"])
                temp_fwhms.append(branch_stats["fwhm"][0])
                temp_nodes.append(stats["node_coords"])
            node_coords.append(temp_nodes)
            crossing_coords.append(temp_coords)
            crossing_heights.append(temp__heights)
            crossing_distances.append(temp_distances)
            fwhms.append(temp_fwhms)

        # Get the image minus the crossing regions
        minus = self.skeleton.copy()
        for crossings in crossing_coords:
            for crossing in crossings:
                minus[crossing[:, 0], crossing[:, 1]] = 0
        minus = label(minus)

        # Get both image
        both = minus.copy()
        for node_num, crossings in enumerate(crossing_coords):
            for crossing_num, crossing in enumerate(crossings):
                both[crossing[:, 0], crossing[:, 1]] = node_num + crossing_num + minus.max()

        # setup z array
        z = []
        # order minus segments
        ordered = []
        for i in range(1, minus.max() + 1):
            arr = np.where(minus, minus == i, 0)
            ordered.append(self.order_branch(arr, [0, 0]))  # orientated later
            z.append(0)

        # add crossing coords to ordered segment list
        for i, node_crossing_coords in enumerate(crossing_coords):
            z_idx = np.argsort(fwhms[i])
            z_idx[z_idx == 0] = -1
            for j, single_cross in enumerate(node_crossing_coords):
                # check current single cross has no duplicate coords with ordered, except crossing points
                uncommon_single_cross = np.array(single_cross).copy()
                for coords in ordered:
                    uncommon_single_cross = self.remove_common_values(
                        uncommon_single_cross, np.array(coords), retain=node_coords[i][j]
                    )
                if len(uncommon_single_cross) > 0:
                    ordered.append(uncommon_single_cross)
                z.append(z_idx[j])

        # get an image of each ordered segment
        cross_add = np.zeros_like(self.image)
        for i, coords in enumerate(ordered):
            single_cross_img = coords_2_img(np.array(coords), cross_add)
            cross_add[single_cross_img != 0] = i + 1

        coord_trace = self.trace(ordered, cross_add)

        # visual over under img
        visual = self.get_visual_img(coord_trace, fwhms, crossing_coords)
        self.image_dict["grain"]["grain_visual_crossings"] = visual

        return coord_trace, visual

    @staticmethod
    def remove_common_values(arr1: npt.NDArray, arr2: npt.NDArray, retain: list = ()) -> np.array:
        """
        Remove common values between two coordinate arrays while retaining specified coordinates.

        Parameters
        ----------
        arr1 : npt.NDArray
            Coordinate array 1.
        arr2 : npt.NDArray
            Coordinate array 2.
        retain : list, optional
            List of possible coordinates to keep, by default ().

        Returns
        -------
        np.array
            Unique array values and retained coordinates.
        """
        # Convert the arrays to sets for faster common value lookup
        set_arr2 = {tuple(row) for row in arr2}
        set_retain = {tuple(row) for row in retain}
        # Create a new filtered list while maintaining the order of the first array
        filtered_arr1 = []
        for coord in arr1:
            tup_coord = tuple(coord)
            if tup_coord not in set_arr2 or tup_coord in set_retain:
                filtered_arr1.append(coord)

        return np.asarray(filtered_arr1)

    def trace(self, ordered_segment_coords: list, both_img: npt.NDArray) -> list:
        """
        Obtain an ordered trace of each complete path.

        Here a 'complete path' means following and removing connected segments until
        there are no more segments to follow.

        Parameters
        ----------
        ordered_segment_coords : list
            Ordered coordinates of each labeled segment in 'both_img'.
        both_img : npt.NDArray
            A skeletonised labeled image of each path segment.

        Returns
        -------
        list
            Ordered trace coordinates of each complete path.
        """
        LOGGER.info(f"[{self.filename}] Getting coordinate trace")

        mol_coords = []
        remaining = both_img.copy().astype(np.int32)
        endpoints = np.unique(remaining[convolve_skeleton(remaining) == 2])  # unique in case of whole mol

        while remaining.max() != 0:
            # select endpoint to start if there is one
            endpoints = [i for i in endpoints if i in np.unique(remaining)]  # remove if removed from remaining
            if endpoints:
                coord_idx = endpoints[0] - 1
            else:  # if no endpoints, just a loop
                coord_idx = np.unique(remaining)[1] - 1  # avoid choosing 0
            coord_trace = np.empty((0, 2)).astype(np.int32)
            while coord_idx > -1:  # either cycled through all or hits terminus -> all will be just background
                remaining[remaining == coord_idx + 1] = 0
                trace_segment = self.get_trace_segment(remaining, ordered_segment_coords, coord_idx)
                if len(coord_trace) > 0:  # can only order when there's a reference point / segment
                    trace_segment = self.remove_duplicates(
                        trace_segment, prev_segment
                    )  # remove overlaps in trace (may be more efficient to do it on the previous segment)
                    trace_segment = self.order_from_end(coord_trace[-1], trace_segment)
                prev_segment = trace_segment.copy()  # update previous segment
                coord_trace = np.append(coord_trace, trace_segment.astype(np.int32), axis=0)
                x, y = coord_trace[-1]
                coord_idx = remaining[x - 1 : x + 2, y - 1 : y + 2].max() - 1  # should only be one value
            mol_coords.append(coord_trace)

        return mol_coords

    @staticmethod
    def get_trace_segment(remaining_img: npt.NDArray, ordered_segment_coords: list, coord_idx: int) -> npt.NDArray:
        """
        Return an ordered segment at the end of the current one.

        Check the branch of given index to see if it contains an endpoint. If it does,
        the segment coordinates will be returned starting from the endpoint.

        Parameters
        ----------
        remaining_img : npt.NDArray
            A 2D array representing an image composed of connected segments of different integers.
        ordered_segment_coords : list
            A list of 2xN coordinates representing each segment.
        coord_idx : int
            The index of the current segment to look at. There is an index mismatch between the
            remaining_img and ordered_segment_coords by -1.

        Returns
        -------
        npt.NDArray
            2xN array of coordinates representing a skeletonised ordered trace segment.
        """
        start_xy = ordered_segment_coords[coord_idx][0]
        start_max = remaining_img[start_xy[0] - 1 : start_xy[0] + 2, start_xy[1] - 1 : start_xy[1] + 2].max() - 1
        if start_max == -1:
            return ordered_segment_coords[coord_idx]  # start is endpoint
        return ordered_segment_coords[coord_idx][::-1]  # end is endpoint

    @staticmethod
    def remove_duplicates(current_segment: npt.NDArray, prev_segment: npt.NDArray) -> npt.NDArray:
        """
        Remove overlapping coordinates present in both arrays.

        Parameters
        ----------
        current_segment : npt.NDArray
            2xN coordinate array.
        prev_segment : npt.NDArray
            2xN coordinate array.

        Returns
        -------
        npt.NDArray
            2xN coordinate array without the previous segment coordinates.
        """
        # Convert arrays to tuples
        curr_segment_tuples = [tuple(row) for row in current_segment]
        prev_segment_tuples = [tuple(row) for row in prev_segment]
        # Find unique rows
        unique_rows = list(set(curr_segment_tuples) - set(prev_segment_tuples))
        # Remove duplicate rows from array1
        return np.array([row for row in curr_segment_tuples if tuple(row) in unique_rows])

    @staticmethod
    def order_from_end(last_segment_coord: npt.NDArray, current_segment: npt.NDArray) -> npt.NDArray:
        """
        Order the current segment to follow from the end of the previous one.

        Parameters
        ----------
        last_segment_coord : npt.NDArray
            X and Y coordinates of the end of the last segment.
        current_segment : npt.NDArray
            A 2xN array of coordinates of the current segment to order.

        Returns
        -------
        npt.NDArray
            The current segment orientated to follow on from the last.
        """
        start_xy = current_segment[0]
        dist = np.sum((start_xy - last_segment_coord) ** 2) ** 0.5
        if dist <= np.sqrt(2):
            return current_segment
        return current_segment[::-1]

    @staticmethod
    def get_trace_idxs(fwhms: list) -> tuple:
        """
        Split underpassing and overpassing indices.

        Parameters
        ----------
        fwhms : list
            List of arrays of FWHM values for each crossing point.

        Returns
        -------
        tuple
            All the under, and over indices of the for each node FWHMs in the provided FWHM list.
        """
        # node fwhms can be a list of different lengths so cannot use np arrays
        under_idxs = []
        over_idxs = []
        for node_fwhms in fwhms:
            order = np.argsort(node_fwhms)
            under_idxs.append(order[0])
            over_idxs.append(order[-1])
        return under_idxs, over_idxs

    def get_visual_img(self, coord_trace: list, fwhms: list, crossing_coords: list) -> npt.NDArray:
        """
        Obtain a labeled image according to the main trace (=1), under (=2), over (=3).

        Parameters
        ----------
        coord_trace : list
            Ordered coordinate trace of each molecule.
        fwhms : list
            List of FWHMs for each crossing in the trace.
        crossing_coords : list
            The crossing coordinates of each branch crossing.

        Returns
        -------
        npt.NDArray
            2D crossing order labeled image.
        """
        # put down traces
        img = np.zeros_like(self.skeleton)
        for mol_no, coords in enumerate(coord_trace):
            temp_img = np.zeros_like(img)
            temp_img[coords[:, 0], coords[:, 1]] = 1
            temp_img = binary_dilation(temp_img)
            img[temp_img != 0] = 1  # mol_no + 1

        lower_idxs, upper_idxs = self.get_trace_idxs(fwhms)

        if False:  # len(coord_trace) > 1:
            # plot separate mols
            for type_idxs in [lower_idxs, upper_idxs]:
                for node_crossing_coords, type_idx in zip(crossing_coords, type_idxs):
                    temp_img = np.zeros_like(img)
                    cross_coords = node_crossing_coords[type_idx]
                    # decide which val
                    matching_coords = np.array([])
                    for trace in coord_trace:
                        c = 0
                        # get overlaps between segment coords and crossing under coords
                        for cross_coord in cross_coords:
                            c += ((trace == cross_coord).sum(axis=1) == 2).sum()
                        matching_coords = np.append(matching_coords, c)
                    val = matching_coords.argmax() + 1
                    temp_img[cross_coords[:, 0], cross_coords[:, 1]] = 1
                    temp_img = binary_dilation(temp_img)
                    img[temp_img != 0] = val

        else:
            # plots over/unders
            for i, type_idxs in enumerate([lower_idxs, upper_idxs]):
                for crossing, type_idx in zip(crossing_coords, type_idxs):
                    temp_img = np.zeros_like(img)
                    cross_coords = crossing[type_idx]
                    # decide which val
                    matching_coords = np.array([])
                    c = 0
                    # get overlaps between segment coords and crossing under coords
                    for cross_coord in cross_coords:
                        c += ((coord_trace[0] == cross_coord).sum(axis=1) == 2).sum()
                    matching_coords = np.append(matching_coords, c)
                    val = matching_coords.argmax() + 1
                    temp_img[cross_coords[:, 0], cross_coords[:, 1]] = 1
                    temp_img = binary_dilation(temp_img)
                    img[temp_img != 0] = i + 2

        return img

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
        for i, (_, values) in enumerate(node_dict.items()):
            conf = values["confidence"]
            if conf is not None:
                sum_conf += conf
                valid_confs += 1
            try:
                return sum_conf / (i + 1)
            except ZeroDivisionError:
                return None
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
        confs = []
        valid_confs = 0
        for _, (_, values) in enumerate(node_dict.items()):
            conf = values["confidence"]
            if conf is not None:
                confs.append(conf)
                valid_confs += 1
        try:
            return min(confs)
        except ValueError:
            return None

    def check_node_errorless(self) -> bool:
        """
        Check if an error has occurred while processing the node dictionary.

        Returns
        -------
        bool
            Whether the error is present.
        """
        for _, vals in self.node_dict.items():
            if vals["error"]:
                return False
        return True

    def compile_metrics(self) -> None:
        """
        Add the number of crossings, average and minimum crossing confidence to the metrics dictionary.

        Returns
        -------
        None
        """
        self.metrics["num_crossings"] = (self.node_centre_mask == 3).sum()
        self.metrics["avg_crossing_confidence"] = nodeStats.average_crossing_confs(self.node_dict)
        self.metrics["min_crossing_confidence"] = nodeStats.minimum_crossing_confs(self.node_dict)


def nodestats_image(
    image: npt.NPArray,
    disordered_tracing_direction_data: dict,
    filename: str,
    pixel_to_nm_scaling: float,
    node_joining_length: float,
    node_extend_dist: float,
    branch_pairing_length: float,
    pad_width: int,
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
    branch_pairing_length : float
        The length from the crossing point to pair and trace, obtaining FWHM's.
    Pad width : int
        The number of edge pixels to pad the image by.

    Returns
    -------
    tuple[dict, pd.DataFrame, dict, dict]
        The nodestats statistics for each crossing, crossing statistics to be added to the grain statistics, an image dictionary of nodestats steps for the entire image, and single grain images.
    """
    n_grains = len(disordered_tracing_direction_data)
    img_base = np.zeros_like(image)
    nodestats_data = {}

    # want to get each cropped image, use some anchor coords to match them onto the image,
    #   and compile all the grain images onto a single image
    all_images = {
        "convolved_skeletons": img_base.copy(),
        "node_centres": img_base.copy(),
        "connected_nodes": img_base.copy(),
    }
    nodestats_branch_images = {}
    grainstats_additions = {}

    LOGGER.info(f"[{filename}] : Calculating NodeStats statistics for {n_grains} grains.")

    for n_grain, disordered_tracing_grain_data in disordered_tracing_direction_data.items():
        nodestats = None  # reset the nodestats variable
        # try:
        nodestats = nodeStats(
            image=disordered_tracing_grain_data["original_image"],
            mask=disordered_tracing_grain_data["original_grain"],
            smoothed_mask=disordered_tracing_grain_data["smoothed_grain"],
            skeleton=disordered_tracing_grain_data["pruned_skeleton"],
            px_2_nm=pixel_to_nm_scaling,
            filename=filename,
            n_grain=n_grain,
            node_joining_length=node_joining_length,
            node_extend_dist=node_extend_dist,
            branch_pairing_length=branch_pairing_length,
        )
        nodestats_dict, node_image_dict = nodestats.get_node_stats()
        LOGGER.info(f"[{filename}] : Nodestats processed {n_grain} of {n_grains}")

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
        nodestats_data[n_grain] = nodestats_dict

        # remap the cropped images back onto the original
        for image_name, full_image in all_images.items():
            crop = nodestats_images[image_name]
            bbox = disordered_tracing_grain_data["bbox"]
            full_image[bbox[0] : bbox[2], bbox[1] : bbox[3]] += crop[pad_width:-pad_width, pad_width:-pad_width]
        """
        except Exception as e:
            LOGGER.error(f"[{filename}] : Disordered tracing for {n_grain} failed with - {e}")
            nodestats_data[n_grain] = {}
        """
        # turn the grainstats additions into a dataframe
        grainstats_additions_df = pd.DataFrame.from_dict(grainstats_additions, orient="index")

    return nodestats_data, grainstats_additions_df, all_images, nodestats_branch_images
