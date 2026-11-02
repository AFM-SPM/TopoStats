"""Order single pixel skeletons with or without NodeStats Statistics."""

import logging
from collections.abc import Generator
from itertools import combinations, product
from typing import TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel
from skimage.graph import route_through_array
from skimage.measure import label
from skimage.morphology import dilation
from topoly import jones, translate_code

from topostats.classes import GrainCrop, Molecule, Node, OrderedTrace, TopoStats
from topostats.logs.logs import LOGGER_NAME
from topostats.tracing.tracingfuncs import coord_dist, genTracingFuncs, order_branch, reorderTrace
from topostats.utils import convolve_skeleton, coords_2_img

LOGGER = logging.getLogger(LOGGER_NAME)

# pylint: disable=possibly-used-before-assignment


class OrderedTraceNodestats:  # pylint: disable=too-many-instance-attributes
    """
    Order single pixel thick skeleton coordinates via NodeStats results.

    Parameters
    ----------
    grain_crop : GrainCrop
            Grain crop post nodestats.
    """

    def __init__(self, grain_crop: GrainCrop) -> None:
        """
        Initialise the OrderedTraceNodestats class.

        Parameters
        ----------
        grain_crop : GrainCrop
            Grain crop post nodestats.
        """
        self.grain_crop: GrainCrop = grain_crop
        self.image: npt.NDArray = grain_crop.image
        if grain_crop.nodes is None:
            raise AttributeError(f"Node statistics do not exist for a grain within {grain_crop.filename}.")
        self.nodestats_dict: dict[str, Node] = grain_crop.nodes
        self.filename: str = grain_crop.filename
        self.pruned_skeleton: npt.NDArray = grain_crop.skeleton
        # ns-rse 2026-01-12 Above works but in doing so during NodeStats the `grain_crop.skeleton` was updated, have
        # tried using convolved skeleton as per below which is what `grain_crop.skeleton` is set to but without success,
        # insufficient time to fix this.
        # self.pruned_skeleton: npt.NDArray = np.where(grain_crop.convolved_skeleton != 0, 1, 0)
        self.grain_tracing_stats = {
            "num_mols": 0,
            "circular": None,
        }
        self.mol_tracing_stats = {"circular": None, "topology": None, "topology_flip": None, "processing": "nodestats"}

        self.images = {
            "all_molecules": np.zeros_like(grain_crop.image),
            "ordered_traces": np.zeros_like(grain_crop.image),
            "over_under": np.zeros_like(grain_crop.image),
            "trace_segments": np.zeros_like(grain_crop.image),
        }

        self.profiles = {}

        self.img_idx_to_node = {}

        self.ordered_coordinates = []

    # pylint: disable=too-many-locals
    # pylint: disable=too-many-branches
    def compile_trace(self, reverse_min_conf_crossing: bool = False) -> tuple[list, npt.NDArray]:  # noqa: C901
        """
        Obtain the trace and diagnostic crossing trace and molecule trace images.

        This function uses the branches and full-width half-maximums (FWHMs) identified in the node_stats dictionary
        to create a continuous trace of the molecule.

        Parameters
        ----------
        reverse_min_conf_crossing : bool
            Whether to reverse the stacking order of the lowest confidence crossing in the trace.

        Returns
        -------
        tuple[list, npt.NDArray]
            A list of each complete path's ordered coordinates, and labeled crossing image array.
        """
        node_coords = [
            [node.node_coords for branch_stats in node.branch_stats.values() if branch_stats.fwhm]
            for node in self.nodestats_dict.values()
        ]
        debug_node_coords_image = np.zeros_like(self.pruned_skeleton)
        pruned_skeleton = self.pruned_skeleton
        for i, coords in enumerate(node_coords):
            for j, single_cross in enumerate(coords):
                debug_node_coords_image[single_cross[:, 0], single_cross[:, 1]] = i + 1
        print("debug")
        node_coords = [lst for lst in node_coords if lst]
        crossing_coords = [
            [branch_stats.ordered_coords for branch_stats in stats.branch_stats.values() if branch_stats.fwhm]
            for stats in self.nodestats_dict.values()
        ]
        crossing_coords = [lst for lst in crossing_coords if lst]
        fwhms = [
            [branch_stats.fwhm for branch_stats in stats.branch_stats.values() if branch_stats.fwhm]
            for stats in self.nodestats_dict.values()
        ]
        fwhms = [lst for lst in fwhms if lst]
        confidences = [stats.confidence for stats in self.nodestats_dict.values()]
        debug_crossing_coords_image = np.zeros_like(self.pruned_skeleton)
        for i, coords in enumerate(crossing_coords):
            for j, single_cross in enumerate(coords):
                debug_crossing_coords_image[single_cross[:, 0], single_cross[:, 1]] = i + 1
        print("debug_crossing_coords_image")

        # obtain the index of the underlying branch
        try:
            low_conf_idx = np.nanargmin(np.array(confidences, dtype=float))
        except ValueError:  # when no crossings or only 3-branch crossings
            low_conf_idx = None

        # ns-rse 2024-10-09 - I suspect these loops could be vectorised in some manner to speed things up but don't have
        # time to investigate and solve right now.
        # Get the image minus the crossing regions
        nodes = np.zeros_like(self.pruned_skeleton)
        for node_no in node_coords:  # this stops unpaired branches from interacting with the pairs
            nodes[node_no[0][:, 0], node_no[0][:, 1]] = 1
        minus = np.where(dilation(dilation(nodes)) == self.pruned_skeleton, 0, self.pruned_skeleton)
        # remove crossings from skeleton
        for crossings in crossing_coords:
            for crossing in crossings:
                minus[crossing[:, 0], crossing[:, 1]] = 0

        # debug
        debug_minus_image = np.zeros_like(self.pruned_skeleton)
        for i, coords in enumerate(node_coords):
            for j, single_cross in enumerate(coords):
                debug_minus_image[single_cross[:, 0], single_cross[:, 1]] = i + 1
        print("debug_minus_image")

        # debug nodes
        debug_nodes_image = np.zeros_like(self.pruned_skeleton)
        for i, coords in enumerate(node_coords):
            for j, single_cross in enumerate(coords):
                debug_nodes_image[single_cross[:, 0], single_cross[:, 1]] = i + 1
        print("debug_nodes_image")

        # Label connected regions of an integer array using scikit-image, if there are no connected regions there is
        # nothing to order and so the return values are set to None
        minus = label(minus)
        if minus.sum() > 0:
            # order minus segments
            z = []
            ordered = []
            # Iterate over the non-crossing segments and add them to a list
            for non_cross_segment_idx in range(1, minus.max() + 1):
                arr = np.where(minus, minus == non_cross_segment_idx, 0)
                non_crossing_segment_image = np.zeros_like(self.pruned_skeleton)
                non_crossing_segment_image[arr != 0] = 1
                print("debug non crossing segment")
                ordered.append(order_branch(arr, [0, 0]))  # orientated later
                # z is pseudo height of the trace.
                z.append(0)
                # map the segment index to the node index and coordinates
                self.img_idx_to_node[non_cross_segment_idx] = {}
            # add crossing coords to ordered segment list
            # uneven count is the index of the next segment to be added to the ordered list.
            # no idea why it is called this.
            uneven_count = non_cross_segment_idx + 1
            # iterate over the crossings and add them to the ordered list
            for node_num, node_crossing_coords in enumerate(crossing_coords):
                # z index is obtained from the highest fwhm being the top strand and vice versa
                z_idx = np.argsort(fwhms[node_num])
                z_idx[z_idx == 0] = -1  # set the lowest to -1
                # reverse_min_conf_crossing is a flag to reverse the order of the lowest confidence crossing?
                if reverse_min_conf_crossing and low_conf_idx == node_num:
                    # if so, reverse the order of the z index of the crossing. unsure why this happens
                    z_idx = z_idx[::-1]
                    fwhms[node_num] = fwhms[node_num][::-1]
                # iterate over the crossing coordinates and add them to the ordered list
                for node_cross_idx, single_cross in enumerate(node_crossing_coords):
                    debug_single_cross_image = np.zeros_like(self.pruned_skeleton)
                    assert single_cross is not None
                    debug_single_cross_image[single_cross[:, 0], single_cross[:, 1]] = 1
                    print("debug_single_cross_image")

                    # check current single cross has no duplicate coords with ordered, except crossing points
                    uncommon_single_cross = np.array(single_cross).copy()
                    for coords in ordered:
                        uncommon_single_cross = self.remove_common_values(
                            uncommon_single_cross, np.array(coords), retain=node_coords[node_num][node_cross_idx]
                        )

                    debug_uncommon_single_cross_image = np.zeros_like(self.pruned_skeleton)
                    debug_uncommon_single_cross_image[uncommon_single_cross[:, 0], uncommon_single_cross[:, 1]] = 1
                    print("debug_uncommon_single_cross_image")

                    # if there are any uncommon coordinates, add them to the ordered list
                    if len(uncommon_single_cross) > 0:
                        print(
                            f"adding uncommon_single_cross to ordered list for node {node_num} crossing {node_cross_idx} {uncommon_single_cross}"
                        )
                        ordered.append(uncommon_single_cross)
                    z.append(z_idx[node_cross_idx])
                    self.img_idx_to_node[uneven_count + node_cross_idx] = {
                        "node_idx": node_num,
                        "coords": single_cross,
                        "z_idx": z_idx[node_cross_idx],
                    }
                uneven_count += len(node_crossing_coords)

            # get an image of each ordered segment
            cross_add = np.zeros_like(self.image)
            for i, coords in enumerate(ordered):
                single_cross_img = coords_2_img(np.array(coords), cross_add)
                cross_add[single_cross_img != 0] = i + 1
            # trace the ordered segments
            coord_trace, simple_trace = self.trace(ordered, cross_add, z, n=100)
            # obtain topology from the simple trace
            topology = self.get_topology(simple_trace)
            if reverse_min_conf_crossing and low_conf_idx is None:  # when there's nothing to reverse
                topology = [None for _ in enumerate(topology)]
        # If there are no connected regions we set parameters to None
        else:
            coord_trace = None
            topology = None
            cross_add = None
            crossing_coords = None
            fwhms = None
        return coord_trace, topology, cross_add, crossing_coords, fwhms

    def compile_images(
        self, coord_trace: list | None, cross_add: npt.NDArray | None, crossing_coords: list | None, fwhms: list | None
    ) -> None:
        """
        Obtain all the diagnostic images based on the produced traces, and values.

        Crossing coords and fwhms are used as arguments as reversing the minimum confidence can modify these.

        If compiling the image failed, i.e. all parameters are ``None`` then all images are set to ``None`` too.

        Parameters
        ----------
        coord_trace : list
            List of N molecule objects containing 2xM arrays of X, Y coordinates.
        cross_add : npt.NDArray
            A labelled array with segments of the ordered trace.
        crossing_coords : list
            A list of I nodes objects containing 2xJ arrays of X, Y coordinates for each crossing branch.
        fwhms : list
            A list of I nodes objects containing FWHM values for each crossing branch.
        """
        # If compiling failed (i.e. all of these are None) we can not compile an image
        if any([coord_trace, cross_add, crossing_coords, fwhms]):
            # visual over under img
            self.images["trace_segments"] = cross_add
            try:
                self.images["over_under"] = self.get_over_under_img(coord_trace, fwhms, crossing_coords)
                self.images["all_molecules"] = self.get_mols_img(coord_trace, fwhms, crossing_coords)
            except IndexError:
                pass
            self.images["ordered_traces"] = ordered_trace_mask(coord_trace, self.image.shape)
        else:
            self.images = None

    @staticmethod
    def remove_common_values(
        ordered_array: npt.NDArray, common_value_check_array: npt.NDArray, retain: list = ()
    ) -> np.array:
        """
        Remove common values in common_value_check_array from ordered_array while retaining specified coordinates.

        Parameters
        ----------
        ordered_array : npt.NDArray
            Coordinate array to remove / retain values from. Will retain its order.
        common_value_check_array : npt.NDArray
            Coordinate array containing any common values to be removed from ordered_array.
        retain : list, optional
            List of possible coordinates to keep, by default ().

        Returns
        -------
        np.array
            Unique ordered_array values and retained coordinates. Retains the order of ordered_array.
        """
        # Convert the arrays to sets for faster common value lookup
        set_arr2 = {tuple(row) for row in common_value_check_array}
        set_retain = {tuple(row) for row in retain}
        # Create a new filtered list while maintaining the order of the first array
        filtered_arr1 = []
        for coord in ordered_array:
            tup_coord = tuple(coord)
            if tup_coord not in set_arr2 or tup_coord in set_retain:
                filtered_arr1.append(coord)

        return np.asarray(filtered_arr1)

    def get_topology(self, nxyz: npt.NDArray) -> list:
        """
        Obtain a topological classification from ordered XYZ coordinates.

        Parameters
        ----------
        nxyz : npt.NDArray
            A 4xN array of the order index (n), x, y and pseudo z coordinates.

        Returns
        -------
        list
            Topology(s) of the provided traced coordinates.
        """
        # Topoly doesn't work when 2 mols don't actually cross
        topology = []
        lin_idxs = []
        nxyz_cp = nxyz.copy()
        # remove linear mols as are just reidmiester moves
        for i, mol_trace in enumerate(nxyz):
            if mol_trace[-1][0] != 0:  # mol is not circular
                topology.append("linear")
                lin_idxs.append(i)
        # remove from list in reverse order so no conflicts
        lin_idxs.sort(reverse=True)
        for i in lin_idxs:
            del nxyz_cp[i]
        # classify topology for non-reidmeister moves
        if len(nxyz_cp) != 0:
            try:
                pd_code = translate_code(
                    nxyz_cp, output_type="pdcode"
                )  # pd code helps prevents freezing and spawning multiple processes
                LOGGER.debug(f"{self.filename} : PD Code is: {pd_code}")
                top_class = jones(pd_code)
            except (IndexError, KeyError):
                LOGGER.debug(f"{self.filename} : PD Code could not be obtained from trace coordinates.")
                top_class = "N/A"

            # don't separate catenanes / overlaps - used for distribution comparison
            for _ in range(len(nxyz_cp)):
                topology.append(top_class)

        return topology

    def trace(self, ordered_segment_coords: list, both_img: npt.NDArray, zs: npt.NDArray, n: int = 100) -> list:
        # pylint: disable=too-many-locals
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
        zs : npt.NDArray
            Array of pseudo heights of the traces. -1 is lowest, 0 is skeleton, then ascending integers for
            levels of overs.
        n : int
            The number of points to use for the simplified traces.

        Returns
        -------
        list
            Ordered trace coordinates of each complete path.
        """
        mol_coords = []
        simple_coords = []
        remaining = both_img.copy().astype(np.int32)
        endpoints = np.unique(remaining[convolve_skeleton(remaining.astype(bool)) == 2])  # unique in case of whole mol
        prev_segment = None
        # n points p seg is the number of points to use for simplified traces?
        # it's:
        # - 2 if number of segments is < 2? Am I sure about this?
        # - (n - 2 * remaining.max()) // remaining.max() otherwise
        n_points_p_seg = (
            2 if ((n - 2 * remaining.max()) // remaining.max()) < 2 else (n - 2 * remaining.max()) // remaining.max()
        )

        while remaining.max() != 0:
            # select endpoint to start if there is one
            endpoints = [i for i in endpoints if i in np.unique(remaining)]  # remove if removed from remaining
            if endpoints:
                coord_idx = endpoints.pop(0) - 1
            else:  # if no endpoints, just a loop
                coord_idx = np.unique(remaining)[1] - 1  # avoid choosing 0
            coord_trace = np.empty((0, 3)).astype(np.int32)
            simple_trace = np.empty((0, 3)).astype(np.int32)

            while coord_idx > -1:  # either cycled through all or hits terminus -> all will be just background
                remaining[remaining == coord_idx + 1] = 0
                trace_segment = self.get_trace_segment(remaining, ordered_segment_coords, coord_idx)
                full_trace_segment = trace_segment.copy()
                if len(coord_trace) > 0:  # can only order when there's a reference point / segment
                    trace_segment = self.remove_common_values(
                        trace_segment, prev_segment
                    )  # remove overlaps in trace (may be more efficient to do it on the previous segment)
                    trace_segment, flipped = self.order_from_end(coord_trace[-1, :2], trace_segment)
                    full_trace_segment = full_trace_segment[::-1] if flipped else full_trace_segment
                trace_segment_image = np.zeros_like(remaining)
                trace_segment_image[trace_segment[:, 0], trace_segment[:, 1]] = 1
                # get vector if crossing
                if self.img_idx_to_node[coord_idx + 1]:
                    segment_vector = full_trace_segment[-1] - full_trace_segment.mean(
                        axis=0
                    )  # from start to mean coord
                    segment_vector /= np.sqrt(segment_vector @ segment_vector)  # normalise
                    self.img_idx_to_node[coord_idx + 1]["vector"] = segment_vector
                prev_segment = trace_segment.copy()  # update previous segment
                trace_segment_z = np.column_stack(
                    (trace_segment, np.ones((len(trace_segment), 1)) * zs[coord_idx])
                ).astype(
                    np.int32
                )  # add z's
                coord_trace = np.append(coord_trace, trace_segment_z.astype(np.int32), axis=0)

                # obtain a reduced coord version of the traces for Topoly
                simple_trace_temp = self.reduce_rows(
                    trace_segment.astype(np.int32), n=n_points_p_seg
                )  # reducing rows here ensures no segments are skipped
                simple_trace_temp_z = np.column_stack(
                    (simple_trace_temp, np.ones((len(simple_trace_temp), 1)) * zs[coord_idx])
                ).astype(
                    np.int32
                )  # add z's
                simple_trace = np.append(simple_trace, simple_trace_temp_z, axis=0)

                x, y = coord_trace[-1, :2]
                coord_idx = remaining[x - 1 : x + 2, y - 1 : y + 2].max() - 1  # should only be one value
            mol_coords.append(coord_trace)

            # Issue in 0_5 where wrong nxyz[0] selected, and == nxyz[-1] so always duplicated
            nxyz = np.column_stack((np.arange(0, len(simple_trace)), simple_trace))
            end_to_end_dist_squared = (nxyz[0][1] - nxyz[-1][1]) ** 2 + (nxyz[0][2] - nxyz[-1][2]) ** 2
            if len(nxyz) > 2 and end_to_end_dist_squared <= 2:  # pylint: disable=chained-comparison
                # single coord traces mean nxyz[0]==[1] so cause issues when duplicating for topoly
                nxyz = np.append(nxyz, nxyz[0][np.newaxis, :], axis=0)
            simple_coords.append(nxyz)

        # convert into lists for Topoly
        simple_coords = [[list(row) for row in mol] for mol in simple_coords]

        return mol_coords, simple_coords

    @staticmethod
    def reduce_rows(array: npt.NDArray, n: int = 300) -> npt.NDArray:
        """
        Reduce the number of rows in the array to `n`, keeping the first and last indexes.

        Parameters
        ----------
        array : npt.NDArray
            An array to reduce the number of rows in.
        n : int, optional
            The number of indexes in the array to keep, by default 300.

        Returns
        -------
        npt.NDArray
            The `array` reduced to only `n` + 2 elements, or if shorter, the same array.
        """
        # removes reduces the number of rows (but keeping the first and last ones)
        if array.shape[0] < n or array.shape[0] < 4:
            return array

        idxs_to_keep = np.unique(np.linspace(0, array[1:-1].shape[0] - 1, n).astype(np.int32))
        new_array = array[1:-1][idxs_to_keep]
        new_array = np.append(array[0][np.newaxis, :], new_array, axis=0)
        return np.append(new_array, array[-1][np.newaxis, :], axis=0)

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
        bool
            Whether the order has been flipped.
        """
        start_xy = current_segment[0]
        dist = np.sum((start_xy - last_segment_coord) ** 2) ** 0.5
        if dist <= np.sqrt(2):
            return current_segment, False
        return current_segment[::-1], True

    def get_over_under_img(self, coord_trace: list, fwhms: list, crossing_coords: list) -> npt.NDArray:
        """
        Obtain a labelled image according to the main trace (=1), under (=2), over (=3).

        Parameters
        ----------
        coord_trace : list
            Ordered coordinate trace of each molecule.
        fwhms : list
            List of full-width half-maximums (FWHMs) for each crossing in the trace.
        crossing_coords : list
            The crossing coordinates of each branch crossing.

        Returns
        -------
        npt.NDArray
            2D crossing order labelled image.
        """
        # put down traces
        img = np.zeros_like(self.pruned_skeleton)
        for coords in coord_trace:
            temp_img = np.zeros_like(img)
            temp_img[coords[:, 0], coords[:, 1]] = 1
            img[temp_img != 0] = 1

        # place over/under strands onto image array
        lower_idxs, upper_idxs = self.get_trace_idxs(fwhms)
        for i, type_idxs in enumerate([lower_idxs, upper_idxs]):
            for crossing, type_idx in zip(crossing_coords, type_idxs):
                temp_img = np.zeros_like(img)
                cross_coords = crossing[type_idx]
                temp_img[cross_coords[:, 0], cross_coords[:, 1]] = 1
                img[temp_img != 0] = i + 2

        return img

    # pylint: disable=too-many-locals
    def get_mols_img(self, coord_trace: list, fwhms: list, crossing_coords: list) -> npt.NDArray:
        # pylint: disable=too-many-locals
        """
        Obtain a labelled image according to each molecule traced N=3 -> n=1,2,3.

        Parameters
        ----------
        coord_trace : list
            Ordered coordinate trace of each molecule.
        fwhms : list
            List of full-width half-maximums (FWHMs) for each crossing in the trace.
        crossing_coords : list
            The crossing coordinates of each branch crossing.

        Returns
        -------
        npt.NDArray
            2D individual 'molecule' labelled image.
        """
        img = np.zeros_like(self.pruned_skeleton)
        for mol_no, coords in enumerate(coord_trace):
            temp_img = np.zeros_like(img)
            temp_img[coords[:, 0], coords[:, 1]] = 1
            img[temp_img != 0] = mol_no + 1
        lower_idxs, upper_idxs = self.get_trace_idxs(fwhms)

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
                        c += ((trace[:, :2] == cross_coord).sum(axis=1) == 2).sum()
                    matching_coords = np.append(matching_coords, c)
                val = matching_coords.argmax() + 1
                temp_img[cross_coords[:, 0], cross_coords[:, 1]] = 1
                img[temp_img != 0] = val

        return img

    @staticmethod
    def get_trace_idxs(fwhms: list) -> tuple[list, list]:
        """
        Split under-passing and over-passing indices.

        Parameters
        ----------
        fwhms : list
            List of arrays of full-width half-maximum (FWHM) values for each crossing point.

        Returns
        -------
        tuple[list, list]
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

    def check_node_errorless(self) -> bool:
        """
        Check if an error has occurred while processing the node dictionary.

        Returns
        -------
        bool
            Whether the error is present.
        """
        for vals in self.nodestats_dict.values():
            # if vals["error"]:
            if vals.error:
                return False
        return True

    def identify_writhes(self) -> str | dict:
        """
        Identify the writhe topology at each crossing in the image.

        Returns
        -------
        str | dict
            A string of the whole grain writhe sign, and a dictionary linking each node to it's sign.
        """
        # compile all vectors for each node and their z_idx
        #   - want for each node, ordered vectors according to z_idx
        writhe_string = ""
        node_to_writhe = {}
        idx2node_df = pd.DataFrame.from_dict(self.img_idx_to_node, orient="index")
        if idx2node_df.empty:  # for when no crossovers but still crossings (i.e. unpaired 3-way)
            return "", {}

        for node_num, node_df in idx2node_df.groupby("node_idx"):
            vector_series = node_df.sort_values(by=["z_idx"], ascending=False)["vector"]
            vectors = list(vector_series)
            # get pairs
            vector_combinations = list(combinations(vectors, 2))
            # calculate the writhe
            temp_writhes = ""
            for vector_pair in vector_combinations:  # if > 2 crossing branches
                temp_writhes += self.writhe_direction(vector_pair[0], vector_pair[1])
            if len(temp_writhes) > 1:
                temp_writhes = f"({temp_writhes})"
            node_to_writhe[node_num] = temp_writhes
            writhe_string += temp_writhes

        return writhe_string, node_to_writhe

    @staticmethod
    def writhe_direction(first_vector: npt.NDArray, second_vector: npt.NDArray) -> str:
        """
        Use the cross product of crossing vectors to determine the writhe sign.

        Parameters
        ----------
        first_vector : npt.NDArray
            An x,y component vector of the overlying strand.
        second_vector : npt.NDArray
            An x,y component vector of the underlying strand.

        Returns
        -------
        str
            '+', '-' or '0' for positive, negative, or no writhe.
        """
        cross = np.cross(first_vector, second_vector)
        if cross < 0:
            return "-"
        if cross > 0:
            return "+"
        return "0"

    def run_nodestats_tracing(self) -> dict[str, npt.NDArray]:
        """
        Run the nodestats tracing pipeline.

        Returns
        -------
        tuple[list, dict, dict]
            A list of each molecules ordered trace coordinates, the ordered_tracing stats, and the images.
        """
        pruned_skeleton = self.pruned_skeleton
        ordered_traces, topology, cross_add, crossing_coords, fwhms = self.compile_trace(
            reverse_min_conf_crossing=False
        )
        debug_trace_image = np.zeros_like(self.image)
        for i, coords in enumerate(ordered_traces):
            debug_trace_image[coords[:, 0], coords[:, 1]] = i + 1
        print("aaa")
        self.compile_images(ordered_traces, cross_add, crossing_coords, fwhms)
        self.grain_tracing_stats["num_mols"] = len(ordered_traces) if ordered_traces is not None else None

        writhe_string, node_to_writhes = self.identify_writhes()
        self.grain_tracing_stats["writhe_string"] = writhe_string
        for node_num, node_writhes in node_to_writhes.items():  # should self update as the dicts are linked
            self.nodestats_dict[node_num].writhe = node_writhes
        # ns-rse 2026-01-06 - This appears to be used to determine the "topology flip" as it is the second call to
        # compile_trace() but this time has reverse_min_conf_crossing=True and _only_ uses the returned topology. For
        # each molecule this is stored along with the original
        topology_flip = self.compile_trace(reverse_min_conf_crossing=True)[1]

        molecule_data = {} if ordered_traces is not None else None
        if ordered_traces is not None:
            for i, mol_trace in enumerate(ordered_traces):
                if len(mol_trace) > 3:  # if > 4 coords to trace
                    self.mol_tracing_stats["circular"] = linear_or_circular(mol_trace[:, :2])
                    self.mol_tracing_stats["topology"] = topology[i]
                    self.mol_tracing_stats["topology_flip"] = topology_flip[i]
                    molecule_data[i] = Molecule(
                        circular=linear_or_circular(mol_trace[:, :2]),
                        topology=topology[i],
                        topology_flip=topology_flip[i],
                        ordered_coords=mol_trace[:, :2],
                        heights=self.image[mol_trace[:, 0], mol_trace[:, 1]],
                        distances=coord_dist(mol_trace[:, :2]),
                    )
        # debug again
        debug_trace_image_post_molecule_creation = np.zeros_like(self.image)
        assert molecule_data is not None
        for molecule_index, molecule in molecule_data.items():
            molecule_ordered_coords = molecule.ordered_coords
            debug_trace_image_post_molecule_creation[molecule_ordered_coords[:, 0], molecule_ordered_coords[:, 1]] = (
                molecule_index + 1
            )
        print("bbb")
        # Add attributes to self.grain_crop
        self.grain_crop.ordered_trace.molecule_data = molecule_data
        for class_number, stats in self.grain_crop.stats.items():
            for subgrain_index, _ in stats.items():
                self.grain_crop.stats[class_number][subgrain_index]["num_mols"] = (
                    len(molecule_data) if molecule_data is not None else 0
                )
                self.grain_crop.stats[class_number][subgrain_index]["writhe_string"] = writhe_string

        return self.images


class OrderedTraceTopostats:
    """
    Order single pixel thick skeleton coordinates via TopoStats.

    Parameters
    ----------
    grain_crop : GrainCrop
        Grain crop to perform ordered tracing on.
    """

    def __init__(
        self,
        grain_crop: GrainCrop,
    ) -> None:
        """
        Initialise the OrderedTraceTopostats class.

        Parameters
        ----------
        grain_crop : GrainCrop
            Grain crop to perform ordered tracing on.
        """
        self.grain_crop = grain_crop
        self.image = grain_crop.image
        self.pruned_skeleton = grain_crop.disordered_trace.images["pruned_skeleton"]
        self.grain_tracing_stats = {
            "num_mols": 1,
            "circular": None,
        }
        self.mol_tracing_stats = {"circular": None, "topology": None, "topology_flip": None, "processing": "topostats"}

        self.images = {
            "all_molecules": np.zeros_like(grain_crop.image),
            "ordered_traces": np.zeros_like(grain_crop.image),
            "over_under": np.zeros_like(grain_crop.image),
            "trace_segments": np.zeros_like(grain_crop.image),
        }

    @staticmethod
    def get_ordered_traces(disordered_trace_coords: npt.NDArray, mol_is_circular: bool) -> list:
        """
        Obtain ordered traces from disordered traces.

        Parameters
        ----------
        disordered_trace_coords : npt.NDArray
            A Nx2 array of coordinates to order.
        mol_is_circular : bool
            A flag of whether the molecule has at least one coordinate with only one neighbour.

        Returns
        -------
        list
            A list of each molecules ordered trace coordinates.
        """
        if mol_is_circular:
            ordered_trace, trace_completed = reorderTrace.circularTrace(disordered_trace_coords)

            if not trace_completed:
                mol_is_circular = False
                try:
                    ordered_trace = reorderTrace.linearTrace(ordered_trace)
                except UnboundLocalError:
                    pass

        elif not mol_is_circular:
            ordered_trace = reorderTrace.linearTrace(disordered_trace_coords)

        return [ordered_trace]

    def run_topostats_tracing(self) -> tuple[list, dict, dict]:
        """
        Run the topostats tracing pipeline.

        Returns
        -------
        tuple[list, dict, dict]
            A list of each molecules ordered trace coordinates, the ordered_traicing stats, and the images.
        """
        disordered_trace_coords = np.argwhere(self.pruned_skeleton == 1)
        self.mol_tracing_stats["circular"] = linear_or_circular(disordered_trace_coords)
        self.mol_tracing_stats["topology"] = "0_1" if self.mol_tracing_stats["circular"] else "linear"

        ordered_trace = self.get_ordered_traces(disordered_trace_coords, self.mol_tracing_stats["circular"])

        self.images["ordered_traces"] = ordered_trace_mask(ordered_trace, self.image.shape)

        molecule_data = {}
        self.grain_crop.ordered_trace.tracing_stats = {}
        for molecule_number, mol_trace in enumerate(ordered_trace):
            molecule_data[molecule_number] = Molecule(
                circular=linear_or_circular(mol_trace[:, :2]),
                ordered_coords=mol_trace[:, :2],
                heights=self.image[mol_trace[:, 0], mol_trace[:, 1]],
                distances=coord_dist(mol_trace[:, :2]),
            )
        # Add attributes to self.grain_crop
        self.grain_crop.ordered_trace.molecule_data = molecule_data
        for class_number, stats in self.grain_crop.stats.items():
            for subgrain_index, _ in stats.items():
                self.grain_crop.stats[class_number][subgrain_index]["num_mols"] = len(molecule_data)
                self.grain_crop.stats[class_number][subgrain_index]["writhe_string"] = "NA"

        return self.images


class Endpoint(BaseModel):
    """Class for endpoint data structures."""

    id: str
    branch_id: int
    coords: tuple[int, int]
    node_index: int

    @classmethod
    def validate_branch_id(cls, v):
        """Validate the branch_id."""
        if not isinstance(v, int) or v < 2:
            raise ValueError("branch_id must be an integer greater than 1")
        return v

    # string repr
    def __str__(self) -> str:
        """Return a string representation of the endpoint."""
        return f"Endpoint(id={self.id}, branch_id={self.branch_id}, coords={self.coords}, node_index={self.node_index})"


T = TypeVar("T")


def get_all_pairing_lists_for_items(items: list[T]) -> Generator[list[tuple[T, T]], None, None]:
    """
    Generate all possible pairing lists for a given list of items.

    Parameters
    ----------
    items : list[T]
        A list of items to be paired.

    Yields
    ------
    list[tuple[T, T]]
        A list of tuples, where each tuple contains a pair of items.
    """
    # Base case: no items left
    if not items:
        yield []
        return

    # Choose the current item to pair with the rest
    first = items[0]
    rest = items[1:]

    # Pair this item with each of the remaining items
    for i, partner in enumerate(rest):
        pair = (first, partner)
        remaining_items = rest[:i] + rest[i + 1 :]

        # Recursively pair the remaining elements that we don't pair with the current item
        for sub_pairing in get_all_pairing_lists_for_items(remaining_items):
            yield [pair] + sub_pairing


def get_permutations_of_options_with_costs(
    dict_of_items_with_options_and_costs: dict[int, list[tuple[int, float]]],
) -> list[tuple[dict[int, tuple[int, float]], float]]:
    """
    Get costs of all permutations of options for a dictionary of items with varying options.

    Given a dictionary of items with their representative options and costs, return all permutations of the options
    with their associated total costs.

    Parameters
    ----------
    dict_of_items_with_options_and_costs : dict[int, list[tuple[int, float]]]
        A dictionary where keys are item identifiers and values are lists of tuples, each containing an option and its
        cost.


    Returns
    -------
    list[tuple[dict[int, tuple[int, float]], float]]
        A list of tuples, where each tuple contains a dictionary mapping item identifiers to their selected option and
        the total cost of that permutation.
    """
    # Unpack the items (ID, options) to compute all combinations
    keys, option_lists = zip(*dict_of_items_with_options_and_costs.items())
    all_permutations = []
    for combination in product(*option_lists):
        permutation_dict = dict(zip(keys, combination))
        total_cost = sum(cost for _, cost in combination)
        all_permutations.append((permutation_dict, total_cost))
    return all_permutations


def create_single_trace(  # noqa: C901
    vertex_suggestions_with_costs: dict[int, list[tuple[Endpoint, Endpoint, float]]],
    all_endpoints: list[Endpoint],
) -> tuple[list[Endpoint], float]:
    """
    Create a single Eulerian trace from a 2k-regular graph, using suggestions for which branches to pair.

    Returns
    -------
    tuple[list[Endpoint], float]
        A list of Endpoint to Endpoint connections representing a single Eulerian trace, and the confidence cost
        associated with the whole trace.
    """
    print(f"vertex_suggestions_with_costs: {vertex_suggestions_with_costs}")
    all_endpoint_ids: set[str] = set()
    for suggestions in vertex_suggestions_with_costs.values():
        for in_endpoint, out_endpoint, _confidence in suggestions:
            all_endpoint_ids.add(in_endpoint.id)
            all_endpoint_ids.add(out_endpoint.id)
    print(f"all_endpoint_ids: {all_endpoint_ids}")
    # For each vertex, get the permutations of the suggestions with associated costs
    # Dict of vertex_index -> list of pairing options with the total cost of that pairing option
    vertex_permutations_with_costs: dict[int, list[tuple[list[tuple[str, str]], float]]] = {}
    # Iterate over the vertexes and their suggestions to create pairing options and calculate costs
    for vertex_index, suggestions in vertex_suggestions_with_costs.items():
        all_endpoint_indexes: set[str] = set()
        for in_endpoint, out_endpoint, _confidence in suggestions:
            all_endpoint_indexes.add(in_endpoint.id)
            all_endpoint_indexes.add(out_endpoint.id)
        # Get all permutations of the possible connections for this vertex
        all_pairing_options = list(get_all_pairing_lists_for_items(list(all_endpoint_indexes)))
        print(
            f"all pairing options for vertex {vertex_index} | all endpoints: {all_endpoint_indexes}: {all_pairing_options}"
        )
        # Associate costs with these pairings
        all_pairing_options_with_costs: list[tuple[list[tuple[str, str]], float]] = (
            []
        )  # list of (pairing_option, cost) tuples
        for pairing_option in all_pairing_options:
            cost = 0
            for in_endpoint, out_endpoint, confidence in suggestions:
                in_endpoint_id = in_endpoint.id
                out_endpoint_id = out_endpoint.id
                if (in_endpoint_id, out_endpoint_id) not in pairing_option and (
                    out_endpoint_id,
                    in_endpoint_id,
                ) not in pairing_option:
                    cost += confidence  # Add the confidence score to the cost if this suggestion is not used
            all_pairing_options_with_costs.append((pairing_option, cost))
            print(f"| pairing option: {pairing_option} | cost: {cost}")
        vertex_permutations_with_costs[vertex_index] = all_pairing_options_with_costs

    for vertex_index, pairing_options_with_costs in vertex_permutations_with_costs.items():
        print(f"vertex {vertex_index}")
        for pairing_option, cost in pairing_options_with_costs:
            print(f"| pairing option: {pairing_option} | cost: {cost}")

    # So here we have a list of possible pairing options for each vertex with their costs.

    # Now want to permute these globally across all vertices

    print("==== global permutations ====")

    # Create a simple mapping between vertex index and the pairing options with costs for that vertex - just
    # for simplicity with handling the itertools product later on
    vertex_pairing_options_with_costs: dict[int, list[tuple[int, float]]] = (
        {}
    )  # vertex_index -> list of (pairing_option_index, cost) tuples
    for vertex_index, pairing_options_with_costs in vertex_permutations_with_costs.items():
        vertex_pairing_options_with_costs[vertex_index] = []
        # For each pairing option with cost, add that
        for pairing_option_index, (_pairing_option, cost) in enumerate(pairing_options_with_costs):
            vertex_pairing_options_with_costs[vertex_index].append((pairing_option_index, cost))

    # print for debugging
    # for vertex_index, pairing_options_with_costs in vertex_pairing_options_with_costs.items():
    #     print(f"vertex {vertex_index}")
    #     for pairing_option_index, cost in pairing_options_with_costs:
    #         print(f"| pairing option index: {pairing_option_index} | cost: {cost}")

    # Find the combination of pairing options across all vertices and their costs
    # list of ({vertex_index: (pairing_option_index, cost)}, total_cost) tuples
    global_all_vertex_pairing_options_with_costs = get_permutations_of_options_with_costs(
        vertex_pairing_options_with_costs
    )

    # Sort the global pairing options by their total cost in ascending order
    global_all_vertex_pairing_options_with_costs.sort(key=lambda x: x[1])

    # print for debugging
    # for global_pairing_option, total_cost in global_all_vertex_pairing_options_with_costs:
    #     print(f"global pairing option: {global_pairing_option} | total cost: {total_cost}")
    #     for vertex_index, (pairing_option_index, cost) in global_pairing_option.items():
    #         print(f"| vertex {vertex_index} | pairing option index: {pairing_option_index} | cost: {cost}")

    print("=== global pairing options with costs reconstruction ===")

    # Iterate through the sorted global pairing options and find the first one that results in a single Eulerian trace

    found_single_cycle = False
    for global_pairing_option, total_cost in global_all_vertex_pairing_options_with_costs:
        print(f"global pairing option: {global_pairing_option} | total cost: {total_cost}")
        # global pairing option is a dict of vertex_index -> (pairing_option_index, cost)
        # for vertex_index, (pairing_option_index, cost) in global_pairing_option.items():
        #     # Get the pairing option for this vertex
        #     pairing_option = vertex_permutations_with_costs[vertex_index][pairing_option_index]
        #     print(f"pairing option: {pairing_option} | cost: {cost}")
        # Create cycles
        cycles: list[list[str]] = []
        current_cycle: list[str] = []
        # Start at an endpoint, any endpoint - grab the first one from the endpoint set
        first_endpoint_id = next(iter(all_endpoint_ids))
        # Now follow the first endpoint through the global pairing option to create a cycle
        visited_endpoint_ids: set[str] = set()
        current_branch_start_endpoint_id = first_endpoint_id
        print("finding cycles:")

        while True:
            # Check if we have already visited this endpoint, if so we have completed a cycle

            if current_branch_start_endpoint_id in visited_endpoint_ids:
                print(f"  completed a cycle with endpoint {current_branch_start_endpoint_id}")
                # completed a cycle - don't add the current endpoint to the cycle since it is already in there
                cycles.append(current_cycle)
                unvisited_endpoint_ids = all_endpoint_ids - visited_endpoint_ids
                print(f"  unvisited_endpoint_ids: {unvisited_endpoint_ids}")
                # if all endpoints have been visited, break
                if len(unvisited_endpoint_ids) == 0:
                    print("  all endpoints have been visited, breaking")
                    break
                # Start a new cycle with an unvisited endpoint
                current_branch_start_endpoint_id = unvisited_endpoint_ids.pop()
                current_cycle = []
                print(f"  starting new cycle with endpoint {current_branch_start_endpoint_id}")

            print(f"  current branch_start_endpoint_id: {current_branch_start_endpoint_id}")
            # Add the start endpoint
            visited_endpoint_ids.add(current_branch_start_endpoint_id)
            current_cycle.append(current_branch_start_endpoint_id)
            # Get the corresponding endpoint, ie 5_0 if current is 5_1 or vice versa
            if current_branch_start_endpoint_id.endswith("_0"):
                current_branch_end_endpoint_id = current_branch_start_endpoint_id.replace("_0", "_1")
            elif current_branch_start_endpoint_id.endswith("_1"):
                current_branch_end_endpoint_id = current_branch_start_endpoint_id.replace("_1", "_0")
            else:
                raise ValueError(f"Invalid endpoint id: {current_branch_start_endpoint_id}")
            print(f"  current branch_end_endpoint_id: {current_branch_end_endpoint_id}")

            # Add the end endpoint
            visited_endpoint_ids.add(current_branch_end_endpoint_id)
            current_cycle.append(current_branch_end_endpoint_id)

            print(f"  current cycle: {current_cycle}")

            # Find out the next start endpoint to follow
            # Get the node index for this endpoint
            current_endpoint = next(
                endpoint for endpoint in all_endpoints if endpoint.id == current_branch_end_endpoint_id
            )
            current_node_index = current_endpoint.node_index
            print(f"  current_node_index: {current_node_index}")
            # Get the pairing option for this node
            node_pairing_option_index, _cost = global_pairing_option[current_node_index]
            pairing_option = vertex_permutations_with_costs[current_node_index][node_pairing_option_index][
                0
            ]  # [0] to get the pairing option from the (pairing_option, cost) tuple
            print(f" pairing_option: {pairing_option} | node_pairing_option_index: {node_pairing_option_index}")

            # Get the corresponding endpoint for this endpoint in the pairing option
            next_branch_start_endpoint_id = None
            for endpoint_id_0, endpoint_id_1 in pairing_option:
                print(
                    f"  checking pair: ({endpoint_id_0}, {endpoint_id_1}) for current_branch_end_endpoint_id: {current_branch_end_endpoint_id}"
                )
                if endpoint_id_0 == current_branch_end_endpoint_id:
                    # Contains, and therefore choose the other
                    next_branch_start_endpoint_id = endpoint_id_1
                    break
                if endpoint_id_1 == current_branch_end_endpoint_id:
                    # Contains, and therefore choose the other
                    next_branch_start_endpoint_id = endpoint_id_0
                    break
            else:
                raise ValueError(
                    f"Pairing option {pairing_option} does not contain endpoint "
                    f"{current_branch_end_endpoint_id} - this should not happen, as the pairing option should "
                    "contain all endpoints for this node"
                )
            assert (
                next_branch_start_endpoint_id is not None
            ), f"Could not find a pair for endpoint {current_branch_end_endpoint_id}"
            print(f"  next_branch_start_endpoint_id: {next_branch_start_endpoint_id}")

            current_branch_start_endpoint_id = next_branch_start_endpoint_id

        print("----")
        if len(cycles) == 1:
            print(
                f"Found a single cycle with global pairing option: {global_pairing_option} | total cost: {total_cost}"
            )
            found_single_cycle = True
            break

        print(
            f"Found {len(cycles)} cycles with global pairing option: {global_pairing_option} | total cost: {total_cost}"
        )
        # Try the next one

    assert (
        found_single_cycle
    ), "Could not find a single cycle with any of the global pairing options - this should never happen"
    print(f"cycles: {cycles}")

    # Construct a list of Endpoint objects for the trace, rather than just indexes
    trace: list[Endpoint] = []
    single_cycle = cycles[0]
    # Iterate over the cycle and create a list of Endpoint objects for the trace
    for endpoint_id in single_cycle:
        endpoint = next(endpoint for endpoint in all_endpoints if endpoint.id == endpoint_id)
        trace.append(endpoint)

    return trace, total_cost


def ordered_tracing_new(grain_crop: GrainCrop) -> None:  # noqa: C901
    """Run new ordered tracing."""
    nodes = grain_crop.nodes
    skeleton = grain_crop.skeleton

    # Construct a branch image with node pixels
    grain_branches_image_with_nodes = skeleton
    for node_index, node in nodes.items():
        node_coords = node.node_coords
        assert node_coords is not None, f"Node {node_index} does not have node_coords"
        for node_coord in node_coords:
            grain_branches_image_with_nodes[node_coord[0], node_coord[1]] = -1  # set node pixels to -1
    # Construct a labelled branch image where node pixels are -1 and branch pixels are branch labels n: int > 1
    grain_branches_image_without_nodes = np.where(
        grain_branches_image_with_nodes == -1, 0, grain_branches_image_with_nodes
    )
    labelled_grain_branches_image = label(grain_branches_image_without_nodes.astype(bool))
    labelled_grain_branches_image_with_nodes = np.where(
        grain_branches_image_with_nodes == -1, -1, labelled_grain_branches_image
    )

    # If there is only one branch - just trace it and return
    if np.max(labelled_grain_branches_image) == 1:
        print("only one branch, tracing it")
        # One branch - pick a start point and trace until the end
        branch_image = np.where(labelled_grain_branches_image_with_nodes == 1, 1, 0)
        # Get a random coord from the branch image
        branch_coords = np.argwhere(branch_image == 1)
        current_coord = branch_coords[0]
        neighbour_coords = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                neighbour_coord = (current_coord[0] + dy, current_coord[1] + dx)
                if (
                    neighbour_coord[0] < 0
                    or neighbour_coord[0] >= branch_image.shape[0]
                    or neighbour_coord[1] < 0
                    or neighbour_coord[1] >= branch_image.shape[1]
                ):
                    continue
                if branch_image[neighbour_coord[0], neighbour_coord[1]] == 1:
                    neighbour_coords.append(neighbour_coord)
        assert (
            len(neighbour_coords) == 2
        ), f"Branch 1 does not have exactly 2 neightbours initially, this should not happen: {neighbour_coords}"
        # Just pick the first one, doesn't matter which.
        next_coord = neighbour_coords[0]
        # set the pixel to 0 so we don't revisit it
        branch_image[next_coord[0], next_coord[1]] = 0
        trace: list[tuple[int, int]] = [tuple(current_coord), tuple(next_coord)]
        while True:
            current_coord = next_coord
            neighbour_coords = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    neighbour_coord = (current_coord[0] + dy, current_coord[1] + dx)
                    if (
                        neighbour_coord[0] < 0
                        or neighbour_coord[0] >= branch_image.shape[0]
                        or neighbour_coord[1] < 0
                        or neighbour_coord[1] >= branch_image.shape[1]
                    ):
                        continue
                    if branch_image[neighbour_coord[0], neighbour_coord[1]] == 1:
                        neighbour_coords.append(neighbour_coord)
            # Remove the previous coord from the list of neighbours
            neighbour_coords = [coord for coord in neighbour_coords if coord != trace[-2]]
            if len(neighbour_coords) == 0:
                break
            assert (
                len(neighbour_coords) == 1
            ), f"Branch 1 does not have exactly 1 neighbour, this should not happen: {neighbour_coords}"
            next_coord = neighbour_coords[0]
            # set the pixel to 0 so we don't revisit it
            branch_image[next_coord[0], next_coord[1]] = 0
            trace.append(tuple(next_coord))
    else:
        print("multiple branches")
        # There are multiple branches
        # Get the endpoints
        print("| finding endpoints")
        endpoints: list[Endpoint] = []
        for branch_id in np.unique(labelled_grain_branches_image_with_nodes):
            if branch_id == 0 or branch_id == -1:  # Skip background and node pixels
                continue
            # Find the endpoints of the branch by convolving the branch
            branch_mask = np.where(labelled_grain_branches_image_with_nodes == branch_id, 1, 0)
            convolved_branch = convolve_skeleton(branch_mask.astype(bool))
            endpoints_coords = np.argwhere(convolved_branch == 2)
            assert (
                len(endpoints_coords) == 2
            ), f"Branch {branch_id} does not have exactly 2 endpoints, this should not happen: {endpoints_coords}"
            # Find which node the endpoints are touching
            for endpoint_index, endpoint_coord in enumerate(endpoints_coords):
                node_indices: list[int] = []
                endpoint_id = f"{branch_id}_{endpoint_index}"
                # Get the nodes connected to this endpoint
                for node_index, node in nodes.items():
                    node_confidence = node.confidence
                    assert node_confidence is not None, f"Node {node_index} does not have a confidence value"
                    assert node.node_coords is not None, f"Node {node_index} does not have node_coords"
                    endpoint_neighbour_pixels = [
                        (endpoint_coord[0] + dy, endpoint_coord[1] + dx) for dy in [-1, 0, 1] for dx in [-1, 0, 1]
                    ]
                    if any(
                        endpoint_neighbour_pixel in node.node_coords
                        for endpoint_neighbour_pixel in endpoint_neighbour_pixels
                    ):
                        node_indices.append(node_index)
                assert (
                    len(node_indices) == 1
                ), f"endpoint {endpoint_id} of branch {branch_id} is not connected to exactly 1 node: {node_indices}"
                endpoints.append(
                    Endpoint(
                        id=endpoint_id,
                        branch_id=branch_id,
                        coords=(endpoint_coord[0], endpoint_coord[1]),
                        node_index=node_indices[0],
                    )
                )

        # Create suggestions for each node, dictating which branches to pair together
        print(f"| creating pairing suggestions")
        # node_index -> [(endpoint1, endpoint2, confidence), ...]
        endpoint_connection_suggestions: dict[int, list[tuple[Endpoint, Endpoint, float]]] = {}
        for node_index, node in nodes.items():
            node_pairing_suggestions: list[tuple[Endpoint, Endpoint, float]] = []
            node_confidence = node.confidence
            assert node_confidence is not None, f"Node {node_index} does not have a confidence value"
            node_branch_stats = node.branch_stats
            assert node_branch_stats is not None, f"Node {node_index} does not have branch_stats"
            for node_branch_pair_data in node_branch_stats.values():
                branch_pair_coords = node_branch_pair_data.ordered_coords
                node_coords = node.node_coords
                node_image = np.zeros_like(labelled_grain_branches_image_with_nodes)
                for node_coord in node_coords:
                    node_image[node_coord[0], node_coord[1]] = 1
                assert branch_pair_coords is not None, f"Node {node_index} branch pair does not have ordered_coords"
                # Get the indexes of the branches in the pair
                endpoint_pair: list[Endpoint] = []
                for branch_pair_coord in branch_pair_coords:
                    # Skip if the coord corresponds to a node pixel
                    if labelled_grain_branches_image_with_nodes[branch_pair_coord[0], branch_pair_coord[1]] == -1:
                        continue
                    # Check if an endpoint
                    for endpoint in endpoints:
                        if endpoint.coords == (branch_pair_coord[0], branch_pair_coord[1]):
                            endpoint_pair.append(endpoint)
                            break
                assert len(endpoint_pair) == 2, (
                    f"Branch pair coordinates {branch_pair_coords} do not"
                    f"correspond to exactly 2 endpoints: {endpoint_pair}"
                )
                node_pairing_suggestions.append(
                    (endpoint_pair[0], endpoint_pair[1], float(np.round(node_confidence, 3)))
                )
            endpoint_connection_suggestions[node_index] = node_pairing_suggestions

        # Now create a single trace from the endpoints and the suggestions
        print("| creating a single cycle from the endpoints and suggestions")
        endpoint_connections, total_cost = create_single_trace(
            vertex_suggestions_with_costs=endpoint_connection_suggestions,
            all_endpoints=endpoints,
        )

        trace: list[tuple[int, int]] = []
        initial_endpoint = endpoint_connections[0]
        branch_start_endpoint = endpoint_connections.pop(0)
        final_endpoint = False
        print("| tracing through the endpoints and nodes to make a full trace")
        while True:
            # Add the current endpoint to the trace
            trace.append((branch_start_endpoint.coords[0], branch_start_endpoint.coords[1]))
            branch_end_endpoint = endpoint_connections.pop(0)
            # Check that the two endpoints are of the same branch
            assert (
                branch_start_endpoint.branch_id == branch_end_endpoint.branch_id
            ), f"Endpoints {branch_start_endpoint} and {branch_end_endpoint} are not of the same branch"

            current_branch_id = branch_start_endpoint.branch_id

            # Trace the branch from current endpoint to next endpoint
            branch_mask = np.where(labelled_grain_branches_image_with_nodes == current_branch_id, 1, 0)
            branch_start_endpoint_coords = branch_start_endpoint.coords
            branch_end_endpoint_coords = branch_end_endpoint.coords
            # Get the coordinates of the branch pixels
            # Order the branch coords by pathfinding from start to end endpoint
            branch_pathfinding_weights = np.invert(
                branch_mask.astype(bool)
            )  # Invert the branch mask to create weights for pathfinding
            branch_pathfinding_weights = np.where(
                branch_pathfinding_weights == 0, 1, 1000000
            )  # Set weights: 1 for branch pixels, high for others
            branch_route, _cost = route_through_array(
                branch_pathfinding_weights, branch_start_endpoint_coords, branch_end_endpoint_coords
            )
            # Check if the branch starts or ends with the current endpoint, if it ends with it, flip it
            branch_coords = np.array(branch_route)
            branch_end_coords = branch_coords[-1]
            if (
                branch_end_coords[0] == branch_start_endpoint_coords[0]
                and branch_end_coords[1] == branch_start_endpoint_coords[1]
            ):
                branch_coords = np.flip(branch_coords, axis=0)

            # Add the branch coordinates to the trace
            for branch_coord in branch_coords:
                trace.append((branch_coord[0], branch_coord[1]))

            # Add the branch end endpoint to the trace
            trace.append((branch_end_endpoint_coords[0], branch_end_endpoint_coords[1]))

            # Get the node index for the next endpoint
            connected_node_index = branch_end_endpoint.node_index

            # Get the next endpoint to connect to
            if len(endpoint_connections) == 0:
                # This is the final endpoint to traverse, so we get the initial endpoint instead
                next_branch_start_endpoint = initial_endpoint
                final_endpoint = True
            else:
                # This is not the final endpoint to traverse, so get the next endpoint to connect to from the
                # list of endpoint connections
                next_branch_start_endpoint = endpoint_connections.pop(0)

            # Trace through the node to the next endpoint
            node_mask = np.zeros_like(labelled_grain_branches_image_with_nodes)
            node = nodes[connected_node_index]
            assert node.node_coords is not None, "Node coordinates are None"
            for node_coord in node.node_coords:
                node_mask[node_coord[0], node_coord[1]] = 1

            # Trace the node from the current endpoint to the next endpoint
            pathfinding_weights = np.invert(
                node_mask.astype(bool)
            )  # Invert the node mask to create weights for pathfinding
            pathfinding_weights = np.where(
                pathfinding_weights == 0, 1, 1000000
            )  # Set weights: 1 for node pixels, high for others
            route, _cost = route_through_array(
                pathfinding_weights, branch_end_endpoint_coords, next_branch_start_endpoint.coords
            )

            # Add the route coordinates to the trace
            for route_coord in route:
                trace.append((route_coord[0], route_coord[1]))

            if final_endpoint:
                # We are done, having traversed through a node to reach the initial endpoint again
                break

            # Set the next branch start endpoint as the current branch start endpoint for the next iteration
            branch_start_endpoint = next_branch_start_endpoint

    # Convert the trace to a numpy array
    trace_array = np.array(trace)
    trace_heights = grain_crop.image[trace_array[:, 0], trace_array[:, 1]]

    # Construct the definitionally single molecule for this grain
    molecule = Molecule(
        circular=True,
        topology_flip="to be implemented",
        topology="to be implemented",
        ordered_coords=trace_array,
        heights=trace_heights,
        distances=coord_dist(trace_array),
    )

    # Add to grain crop molecule data
    grain_crop.ordered_trace.molecule_data = {0: molecule}
    for class_number, stats in grain_crop.stats.items():
        for subgrain_index, _ in stats.items():
            grain_crop.stats[class_number][subgrain_index]["num_mols"] = 1
            grain_crop.stats[class_number][subgrain_index]["writhe_string"] = "not calculated"


def linear_or_circular(traces) -> bool:
    """
    Determine whether the molecule is circular or linear via >1 points in the local start area.

    This function is sensitive to branches from the skeleton because it is based on whether any given point has zero
    neighbours or not so the traces should be pruned.

    Parameters
    ----------
    traces : npt.NDArray
        The array of coordinates to be assessed.

    Returns
    -------
    bool
        Whether a molecule is linear or not (True if linear, False otherwise).
    """
    points_with_one_neighbour = 0
    fitted_trace_list = traces.tolist()

    # For loop determines how many neighbours a point has - if only one it is an end
    for x, y in fitted_trace_list:
        if genTracingFuncs.count_and_get_neighbours(x, y, fitted_trace_list)[0] == 1:
            points_with_one_neighbour += 1
        else:
            pass

    if points_with_one_neighbour == 0:
        return True
    return False


def ordered_trace_mask(ordered_coordinates: npt.NDArray, shape: tuple) -> npt.NDArray:
    """
    Obtain a mask of the trace coordinates with each trace pixel.

    Parameters
    ----------
    ordered_coordinates : npt.NDArray
        Ordered array of coordinates.

    shape : tuple
        The shape of the array bounding the coordinates.

    Returns
    -------
    npt.NDArray
        NxM image with each pixel in the ordered trace labeled in ascending order.
    """
    ordered_mask = np.zeros(shape)
    if isinstance(ordered_coordinates, list):
        for mol_coords in ordered_coordinates:
            ordered_mask[mol_coords[:, 0], mol_coords[:, 1]] = np.arange(len(mol_coords)) + 1

    return ordered_mask


def ordered_tracing_image(
    topostats_object: TopoStats,
    ordering_method: str,
) -> None:
    # pylint: disable=too-many-locals
    """
    Run ordered tracing for an entire image of >=1 grains.

    Parameters
    ----------
    topostats_object : TopoStats
        TopoStats object to have ordered tracing performed on.
    ordering_method : str
        The method to order the trace coordinates - "topostats" or "nodestats".
    """
    config = topostats_object.config.copy()
    ordering_method = config["ordered_tracing"]["ordering_method"] if ordering_method is None else ordering_method
    ordered_trace_full_images = {
        "ordered_traces": np.zeros_like(topostats_object.image),
        "all_molecules": np.zeros_like(topostats_object.image),
        "over_under": np.zeros_like(topostats_object.image),
        "trace_segments": np.zeros_like(topostats_object.image),
    }
    LOGGER.info(
        f"[{topostats_object.filename}] : Calculating Ordered Traces and Statistics for "
        f"{len(topostats_object.grain_crops)} grains..."
    )
    for grain_no, grain_crop in topostats_object.grain_crops.items():
        try:
            grain_crop.ordered_trace = OrderedTrace(
                molecule_data=None,
                tracing_stats=None,
                grain_molstats=None,
                molecules=None,
                writhe=None,
                pixel_to_nm_scaling=None,
                images=None,
                error=None,
            )
            debug_skeleton = grain_crop.skeleton
            debug_image = grain_crop.image
            # check if want to perform tracing based on node statistics
            assert topostats_object.config is not None
            if topostats_object.config["ordered_tracing"]["ordering_method"] == "nodestats":
                if grain_crop.nodes is not None and len(grain_crop.nodes) > 0:
                    LOGGER.info(
                        f"[{topostats_object.filename}] : Grain {grain_no + 1} present in NodeStats. Tracing via Nodestats."
                    )
                    nodestats_tracing = OrderedTraceNodestats(grain_crop=grain_crop)

                    if nodestats_tracing.check_node_errorless():
                        grain_crop.ordered_trace.images = nodestats_tracing.run_nodestats_tracing()
                        LOGGER.info(f"[{topostats_object.filename}] : Grain {grain_no + 1} ordered via NodeStats.")
                    else:
                        LOGGER.info(f"Nodestats dict has an error for grain : ({grain_no + 1}")
                # if not doing nodestats ordering, do original TS ordering
                elif grain_crop.disordered_trace is not None:
                    LOGGER.info(
                        f"[{topostats_object.filename}] : Grain {grain_no + 1} not in NodeStats. "
                        "Attempting to trace normally."
                    )
                    topostats_tracing = OrderedTraceTopostats(grain_crop=grain_crop)
                    grain_crop.ordered_trace.images = topostats_tracing.run_topostats_tracing()
                    LOGGER.info(f"[{topostats_object.filename}] : Grain {grain_no + 1} ordered via TopoStats.")
                else:
                    LOGGER.info(
                        f"[{topostats_object.filename}] : Grain {grain_no + 1} does not have a disordered trace "
                        "skipping orderering."
                    )
                # remap the cropped images back onto the original
                for image_name, full_image in ordered_trace_full_images.items():
                    if grain_crop.ordered_trace.images is not None:
                        crop = grain_crop.ordered_trace.images[image_name]
                        full_image[
                            grain_crop.bbox[0] : grain_crop.bbox[2], grain_crop.bbox[1] : grain_crop.bbox[3]
                        ] += crop
                    # Add the ordered_trace_full_image to topostats_object.full_image_plots
                    if topostats_object.full_image_plots is None:
                        topostats_object.full_image_plots = ordered_trace_full_images
                    elif isinstance(topostats_object.full_image_plots, dict):
                        topostats_object.full_image_plots = {
                            **topostats_object.full_image_plots,
                            **ordered_trace_full_images,
                        }
            elif topostats_object.config["ordered_tracing"]["ordering_method"] == "new":
                print(f"grain {grain_no + 1} ordered tracing new : before")
                ordered_tracing_new(grain_crop=grain_crop)
                print(f"grain {grain_no + 1} ordered tracing new : done")

                assert grain_crop.ordered_trace is not None
                assert grain_crop.ordered_trace.molecule_data is not None
                ordered_trace_coords = grain_crop.ordered_trace.molecule_data[0].ordered_coords
                assert ordered_trace_coords is not None, "Ordered trace coordinates are None"

                ordered_trace_full_images["ordered_traces"][
                    grain_crop.bbox[0] : grain_crop.bbox[2], grain_crop.bbox[1] : grain_crop.bbox[3]
                ][ordered_trace_coords[:, 0], ordered_trace_coords[:, 1]] = (np.arange(len(ordered_trace_coords)) + 1)
                ordered_trace_full_images["all_molecules"] = ordered_trace_full_images["ordered_traces"]
                ordered_trace_full_images["over_under"] = np.zeros_like(topostats_object.image)
                ordered_trace_full_images["trace_segments"] = np.zeros_like(topostats_object.image)
                # Add the ordered_trace_full_image to topostats_object.full_image_plots
                if topostats_object.full_image_plots is None:
                    topostats_object.full_image_plots = ordered_trace_full_images
                elif isinstance(topostats_object.full_image_plots, dict):
                    topostats_object.full_image_plots = {
                        **topostats_object.full_image_plots,
                        **ordered_trace_full_images,
                    }
                LOGGER.info(f"[{topostats_object.filename}] : Grain {grain_no + 1} ordered via new ordered tracing.")
            else:
                raise ValueError(f"Invalid ordering method: {ordering_method}.")
                if grain_crop.disordered_trace is not None:
                    LOGGER.info(f"[{topostats_object.filename}] : Grain {grain_no + 1} ordered via TopoStats.")
                    topostats_tracing = OrderedTraceTopostats(grain_crop=grain_crop)
                    grain_crop.ordered_trace.images = topostats_tracing.run_topostats_tracing()
                else:
                    LOGGER.info(
                        f"[{topostats_object.filename}] : Grain {grain_no + 1} does not have a disordered trace "
                        "skipping orderering."
                    )
        except Exception as e:  # pylint: disable=broad-exception-caught
            LOGGER.error(
                f"[{topostats_object.filename}] : Ordered tracing for grain {grain_no + 1} failed. "
                "Consider raising an issue on GitHub. Error: ",
                exc_info=e,
            )
            raise e
