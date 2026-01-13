"""Order single pixel skeletons with or without NodeStats Statistics."""

import logging
from itertools import combinations

import numpy as np
import numpy.typing as npt
import pandas as pd
from skimage.morphology import dilation, label
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
        minus = label(minus)
        # order minus segments
        z = []
        ordered = []
        for non_cross_segment_idx in range(1, minus.max() + 1):
            arr = np.where(minus, minus == non_cross_segment_idx, 0)
            ordered.append(order_branch(arr, [0, 0]))  # orientated later
            z.append(0)
            self.img_idx_to_node[non_cross_segment_idx] = {}

        # add crossing coords to ordered segment list
        uneven_count = non_cross_segment_idx + 1
        for node_num, node_crossing_coords in enumerate(crossing_coords):
            z_idx = np.argsort(fwhms[node_num])
            z_idx[z_idx == 0] = -1
            if reverse_min_conf_crossing and low_conf_idx == node_num:
                z_idx = z_idx[::-1]
                fwhms[node_num] = fwhms[node_num][::-1]
            for node_cross_idx, single_cross in enumerate(node_crossing_coords):
                # check current single cross has no duplicate coords with ordered, except crossing points
                uncommon_single_cross = np.array(single_cross).copy()
                for coords in ordered:
                    uncommon_single_cross = self.remove_common_values(
                        uncommon_single_cross, np.array(coords), retain=node_coords[node_num][node_cross_idx]
                    )
                if len(uncommon_single_cross) > 0:
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
        coord_trace, simple_trace = self.trace(ordered, cross_add, z, n=100)
        # obtain topology from the simple trace
        topology = self.get_topology(simple_trace)
        if reverse_min_conf_crossing and low_conf_idx is None:  # when there's nothing to reverse
            topology = [None for _ in enumerate(topology)]

        return coord_trace, topology, cross_add, crossing_coords, fwhms

    def compile_images(self, coord_trace: list, cross_add: npt.NDArray, crossing_coords: list, fwhms: list) -> None:
        """
        Obtain all the diagnostic images based on the produced traces, and values.

        Crossing coords and fwhms are used as arguments as reversing the minimum confidence can modify these.

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
        # visual over under img
        self.images["trace_segments"] = cross_add
        try:
            self.images["over_under"] = self.get_over_under_img(coord_trace, fwhms, crossing_coords)
            self.images["all_molecules"] = self.get_mols_img(coord_trace, fwhms, crossing_coords)
        except IndexError:
            pass
        self.images["ordered_traces"] = ordered_trace_mask(coord_trace, self.image.shape)

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
        ordered_traces, topology, cross_add, crossing_coords, fwhms = self.compile_trace(
            reverse_min_conf_crossing=False
        )
        self.compile_images(ordered_traces, cross_add, crossing_coords, fwhms)
        self.grain_tracing_stats["num_mols"] = len(ordered_traces)

        writhe_string, node_to_writhes = self.identify_writhes()
        self.grain_tracing_stats["writhe_string"] = writhe_string
        for node_num, node_writhes in node_to_writhes.items():  # should self update as the dicts are linked
            self.nodestats_dict[node_num].writhe = node_writhes
        # ns-rse 2026-01-06 - This appears to be used to determine the "topology flip" as it is the second call to
        # compile_trace() but this time has reverse_min_conf_crossing=True and _only_ uses the returned topology. For
        # each molecule this is stored along with the original
        topology_flip = self.compile_trace(reverse_min_conf_crossing=True)[1]

        molecule_data = {}
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
        # Add attributes to self.grain_crop
        self.grain_crop.ordered_trace.molecule_data = molecule_data
        for class_number, stats in self.grain_crop.stats.items():
            for subgrain_index, _ in stats.items():
                self.grain_crop.stats[class_number][subgrain_index]["num_mols"] = len(molecule_data)
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
            # check if want to perform tracing based on node statistics
            if (
                len(grain_crop.nodes) > 0
                and topostats_object.config["ordered_tracing"]["ordering_method"] == "nodestats"
            ):
                LOGGER.info(
                    f"[{topostats_object.filename}] : Grain {grain_no + 1} present in NodeStats. Tracing via Nodestats."
                )
                nodestats_tracing = OrderedTraceNodestats(grain_crop=grain_crop)

                if nodestats_tracing.check_node_errorless():
                    grain_crop.ordered_trace.images = nodestats_tracing.run_nodestats_tracing()
                    LOGGER.debug(f"[{topostats_object.filename}] : Grain {grain_no + 1} ordered via NodeStats.")
                else:
                    LOGGER.debug(f"Nodestats dict has an error for grain : ({grain_no + 1}")
            # if not doing nodestats ordering, do original TS ordering
            elif grain_crop.disordered_trace is not None:
                LOGGER.info(f"[{topostats_object.filename}] : Grain {grain_no + 1} not in NodeStats. Tracing normally.")
                topostats_tracing = OrderedTraceTopostats(grain_crop=grain_crop)
                grain_crop.ordered_trace.images = topostats_tracing.run_topostats_tracing()
                LOGGER.debug(f"[{topostats_object.filename}] : Grain {grain_no + 1} ordered via TopoStats.")
            else:
                LOGGER.debug(
                    f"[{topostats_object.filename}] : Grain {grain_no + 1} does not have a disordered trace "
                    "skipping orderering."
                )
            # remap the cropped images back onto the original
            for image_name, full_image in ordered_trace_full_images.items():
                crop = grain_crop.ordered_trace.images[image_name]
                full_image[grain_crop.bbox[0] : grain_crop.bbox[2], grain_crop.bbox[1] : grain_crop.bbox[3]] += crop
            # Add the ordered_trace_full_image to topostats_object.full_image_plots
            if topostats_object.full_image_plots is None:
                topostats_object.full_image_plots = ordered_trace_full_images
            elif isinstance(topostats_object.full_image_plots, dict):
                topostats_object.full_image_plots = {**topostats_object.full_image_plots, **ordered_trace_full_images}
        except Exception as e:  # pylint: disable=broad-exception-caught
            LOGGER.error(
                f"[{topostats_object.filename}] : Ordered tracing for {grain_no} failed. "
                "Consider raising an issue on GitHub. Error: ",
                exc_info=e,
            )
