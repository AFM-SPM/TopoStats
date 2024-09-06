"""Order single pixel skeletons with or without NodeStats Statistics."""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt
import pandas as pd
from skimage.morphology import label

from topostats.logs.logs import LOGGER_NAME
from topostats.tracing.tracingfuncs import coord_dist, genTracingFuncs, order_branch, reorderTrace
from topostats.utils import convolve_skeleton, coords_2_img

LOGGER = logging.getLogger(LOGGER_NAME)


class OrderedTraceNodestats:
    """
    Order single pixel thick skeleton coordinates via NodeStats results.

    Parameters
    ----------
    image : npt.NDArray
        A cropped image array.
    nodestats_dict : dict
        The nodestats results for a specific grain.
    skeleton : npt.NDArray
        The pruned skeleton mask array.
    filename : str
        The image filename (for logging purposes).
    """

    def __init__(
        self,
        image: npt.NDArray,
        nodestats_dict: dict,
        skeleton: npt.NDArray,
        filename: str,
    ) -> None:
        """
        Initialise the OrderedTraceNodestats class.

        Parameters
        ----------
        image : npt.NDArray
            A cropped image array.
        nodestats_dict : dict
            The nodestats results for a specific grain.
        skeleton : npt.NDArray
            The pruned skeleton mask array.
        filename : str
            The image filename (for logging purposes).
        """
        self.image = image
        self.nodestats_dict = nodestats_dict
        self.filename = filename
        self.skeleton = skeleton

        self.grain_tracing_stats = {
            "num_mols": 0,
            "circular": None,
        }
        self.mol_tracing_stats = {"circular": None}

        self.ordered_traces = None

        self.images = {
            "over_under": np.zeros_like(image),
            "all_molecules": np.zeros_like(image),
            "ordered_traces": np.zeros_like(image),
            "trace_segments": np.zeros_like(image),
        }

        self.profiles = {}

        self.ordered_coordinates = []

    def compile_trace(self) -> tuple[list, npt.NDArray]:
        """
        Obtain the trace and diagnostic crossing trace and molecule trace images.

        This function uses the branches and full-width half-maximums (FWHMs) identified in the node_stats dictionary
        to create a continuous trace of the molecule.

        Returns
        -------
        tuple[list, npt.NDArray]
            A list of each complete path's ordered coordinates, and labeled crosing image array.
        """
        LOGGER.info(f"[{self.filename}] : Compiling the trace.")

        # iterate through the dict to get branch coords, heights and fwhms
        node_coords = [
            [stats["node_coords"] for branch_stats in stats["branch_stats"].values() if branch_stats["fwhm"]["fwhm"]]
            for stats in self.nodestats_dict.values()
        ]
        node_coords = [lst for lst in node_coords if lst]

        crossing_coords = [
            [
                branch_stats["ordered_coords"]
                for branch_stats in stats["branch_stats"].values()
                if branch_stats["fwhm"]["fwhm"]
            ]
            for stats in self.nodestats_dict.values()
        ]
        crossing_coords = [lst for lst in crossing_coords if lst]

        fwhms = [
            [
                branch_stats["fwhm"]["fwhm"]
                for branch_stats in stats["branch_stats"].values()
                if branch_stats["fwhm"]["fwhm"]
            ]
            for stats in self.nodestats_dict.values()
        ]
        fwhms = [lst for lst in fwhms if lst]

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

        # order minus segments
        ordered = []
        for i in range(1, minus.max() + 1):
            arr = np.where(minus, minus == i, 0)
            ordered.append(order_branch(arr, [0, 0]))  # orientated later

        # add crossing coords to ordered segment list
        for i, node_crossing_coords in enumerate(crossing_coords):
            for j, single_cross in enumerate(node_crossing_coords):
                # check current single cross has no duplicate coords with ordered, except crossing points
                uncommon_single_cross = np.array(single_cross).copy()
                for coords in ordered:
                    uncommon_single_cross = self.remove_common_values(
                        uncommon_single_cross, np.array(coords), retain=node_coords[i][j]
                    )
                if len(uncommon_single_cross) > 0:
                    ordered.append(uncommon_single_cross)

        # get an image of each ordered segment
        cross_add = np.zeros_like(self.image)
        for i, coords in enumerate(ordered):
            single_cross_img = coords_2_img(np.array(coords), cross_add)
            cross_add[single_cross_img != 0] = i + 1

        coord_trace = self.trace(ordered, cross_add)

        # visual over under img
        self.images["trace_segments"] = cross_add
        try:
            self.images["over_under"] = self.get_over_under_img(coord_trace, fwhms, crossing_coords)
            self.images["all_molecules"] = self.get_mols_img(coord_trace, fwhms, crossing_coords)
        except IndexError:
            pass
        self.images["ordered_traces"] = ordered_trace_mask(coord_trace, self.image.shape)

        return coord_trace, self.images

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
        mol_coords = []
        remaining = both_img.copy().astype(np.int32)
        endpoints = np.unique(remaining[convolve_skeleton(remaining.astype(bool)) == 2])  # unique in case of whole mol
        prev_segment = None

        while remaining.max() != 0:
            # select endpoint to start if there is one
            endpoints = [i for i in endpoints if i in np.unique(remaining)]  # remove if removed from remaining
            if endpoints:
                coord_idx = endpoints.pop(0) - 1
            else:  # if no endpoints, just a loop
                coord_idx = np.unique(remaining)[1] - 1  # avoid choosing 0
            coord_trace = np.empty((0, 2)).astype(np.int32)
            while coord_idx > -1:  # either cycled through all or hits terminus -> all will be just background
                remaining[remaining == coord_idx + 1] = 0
                trace_segment = self.get_trace_segment(remaining, ordered_segment_coords, coord_idx)
                if len(coord_trace) > 0:  # can only order when there's a reference point / segment
                    trace_segment = self.remove_common_values(
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

    def get_over_under_img(self, coord_trace: list, fwhms: list, crossing_coords: list) -> npt.NDArray:
        """
        Obtain a labeled image according to the main trace (=1), under (=2), over (=3).

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
            2D crossing order labeled image.
        """
        # put down traces
        img = np.zeros_like(self.skeleton)
        for coords in coord_trace:
            temp_img = np.zeros_like(img)
            temp_img[coords[:, 0], coords[:, 1]] = 1
            # temp_img = binary_dilation(temp_img)
            img[temp_img != 0] = 1  # mol_no + 1
        lower_idxs, upper_idxs = self.get_trace_idxs(fwhms)

        # place over/unders onto image array
        for i, type_idxs in enumerate([lower_idxs, upper_idxs]):
            for crossing, type_idx in zip(crossing_coords, type_idxs):
                temp_img = np.zeros_like(img)
                cross_coords = crossing[type_idx]
                temp_img[cross_coords[:, 0], cross_coords[:, 1]] = 1
                # temp_img = binary_dilation(temp_img)
                img[temp_img != 0] = i + 2

        return img

    def get_mols_img(self, coord_trace: list, fwhms: list, crossing_coords: list) -> npt.NDArray:
        """
        Obtain a labeled image according to each molecule traced N=3 -> n=1,2,3.

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
            2D individual 'molecule' labeled image.
        """
        img = np.zeros_like(self.skeleton)
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
                        c += ((trace == cross_coord).sum(axis=1) == 2).sum()
                    matching_coords = np.append(matching_coords, c)
                val = matching_coords.argmax() + 1
                temp_img[cross_coords[:, 0], cross_coords[:, 1]] = 1
                img[temp_img != 0] = val

        return img

    @staticmethod
    def get_trace_idxs(fwhms: list) -> tuple[list, list]:
        """
        Split underpassing and overpassing indices.

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
            if vals["error"]:
                return False
        return True

    def run_nodestats_tracing(self) -> tuple[list, dict, dict]:
        """
        Run the nodestats tracing pipeline.

        Returns
        -------
        tuple[list, dict, dict]
            A list of each molecules ordered trace coordinates, the ordered_traicing stats, and the images.
        """
        self.ordered_traces, self.images = self.compile_trace()
        self.grain_tracing_stats["num_mols"] = len(self.ordered_traces)

        ordered_trace_data = {}
        for i, mol_trace in enumerate(self.ordered_traces):
            if len(mol_trace) > 3:  # if > 4 coords to trace
                self.mol_tracing_stats["circular"] = linear_or_circular(mol_trace)
                ordered_trace_data[f"mol_{i}"] = {
                    "ordered_coords": mol_trace,
                    "heights": self.image[mol_trace[:, 0], mol_trace[:, 1]],
                    "distances": coord_dist(mol_trace[0]),
                    "mol_stats": self.mol_tracing_stats,
                }

        return ordered_trace_data, self.grain_tracing_stats, self.images


class OrderedTraceTopostats:
    """
    Order single pixel thick skeleton coordinates via TopoStats.

    Parameters
    ----------
    image : npt.NDArray
        A cropped image array.
    skeleton : npt.NDArray
        The pruned skeleton mask array.
    """

    def __init__(
        self,
        image,
        skeleton,
    ) -> None:
        """
        Initialise the OrderedTraceTopostats class.

        Parameters
        ----------
        image : npt.NDArray
            A cropped image array.
        skeleton : npt.NDArray
            The pruned skeleton mask array.
        """
        self.image = image
        self.skeleton = skeleton
        self.grain_tracing_stats = {
            "num_mols": 1,
            "circular": None,
        }
        self.mol_tracing_stats = {"circular": None}

        self.images = {
            "ordered_traces": np.zeros_like(image),
            "all_molecules": skeleton.copy(),
            "over_under": skeleton.copy(),
            "trace_segments": skeleton.copy(),
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
        disordered_trace_coords = np.argwhere(self.skeleton == 1)

        self.mol_tracing_stats["circular"] = linear_or_circular(disordered_trace_coords)

        ordered_trace = self.get_ordered_traces(disordered_trace_coords, self.mol_tracing_stats["circular"])

        self.images["ordered_traces"] = ordered_trace_mask(ordered_trace, self.image.shape)

        ordered_trace_data = {}
        for i, mol_trace in enumerate(ordered_trace):
            ordered_trace_data[f"mol_{i}"] = {
                "ordered_coords": mol_trace,
                "heights": self.image[ordered_trace[0][:, 0], ordered_trace[0][:, 1]],
                "distances": coord_dist(ordered_trace[0]),
                "mol_stats": self.mol_tracing_stats,
            }

        return ordered_trace_data, self.grain_tracing_stats, self.images


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
            ordered_mask[mol_coords[:, 0], mol_coords[:, 1]] = np.arange(len(mol_coords))

    return ordered_mask


def ordered_tracing_image(
    image: npt.NDArray,
    disordered_tracing_direction_data: dict,
    nodestats_direction_data: dict,
    filename: str,
    pixel_to_nm_scaling: float,
    ordering_method: str,
    pad_width: int,
) -> tuple[dict, pd.DataFrame, dict]:
    """
    Run ordered tracing for an entire image of >=1 grains.

    Parameters
    ----------
    image : npt.NDArray
        Whole FOV image.
    disordered_tracing_direction_data : dict
        Dictionary result from the disordered traces. Fields used are "original_image" and "pruned_skeleton".
    nodestats_direction_data : dict
        Dictionary result from the nodestats analysis.
    filename : str
        Image filename (for logging purposes).
    pixel_to_nm_scaling : float
        _description_.
    ordering_method : str
        The method to order the trace coordinates - "topostats" or "nodestats".
    pad_width : int
        Width to pad the images by.

    Returns
    -------
    tuple[dict, pd.DataFrame, dict]
        Results containing the ordered_trace_data (coordinates), any grain-level metrics to be added to the grains
        dataframe, and the diagnostic images.
    """
    ordered_trace_full_images = {
        "ordered_traces": np.zeros_like(image),
        "all_molecules": np.zeros_like(image),
        "over_under": np.zeros_like(image),
        "trace_segments": np.zeros_like(image),
    }
    grainstats_additions = {}
    all_traces_data = {}

    # iterate through disordered_tracing_dict
    for grain_no, disordered_trace_data in disordered_tracing_direction_data.items():
        # try:
        # check if want to do nodestats tracing or not
        if grain_no in list(nodestats_direction_data["stats"].keys()) and ordering_method == "nodestats":
            LOGGER.info(f"[{filename}] : Grain {grain_no} present in NodeStats. Tracing via Nodestats.")
            nodestats_tracing = OrderedTraceNodestats(
                image=nodestats_direction_data["images"][grain_no]["grain"]["grain_image"],
                filename=filename,
                nodestats_dict=nodestats_direction_data["stats"][grain_no],
                skeleton=nodestats_direction_data["images"][grain_no]["grain"]["grain_skeleton"],
            )
            if nodestats_tracing.check_node_errorless():
                ordered_traces_data, tracing_stats, images = nodestats_tracing.run_nodestats_tracing()
                LOGGER.info(f"[{filename}] : Grain {grain_no} ordered via NodeStats.")
            else:
                LOGGER.warning(f"Nodestats dict has an error ({nodestats_direction_data['stats'][grain_no]['error']}")
        # if not doing nodestats ordering, do original TS ordering
        else:
            LOGGER.info(f"[{filename}] : {grain_no} not in NodeStats. Tracing normally.")
            topostats_tracing = OrderedTraceTopostats(
                image=disordered_trace_data["original_image"],
                skeleton=disordered_trace_data["pruned_skeleton"],
            )
            ordered_traces_data, tracing_stats, images = topostats_tracing.run_topostats_tracing()
            LOGGER.info(f"[{filename}] : Grain {grain_no} ordered via TopoStats.")

        # compile traces
        all_traces_data[grain_no] = ordered_traces_data
        for mol_no, _ in ordered_traces_data.items():
            all_traces_data[grain_no][mol_no].update({"bbox": disordered_trace_data["bbox"]})
        # compile metrics
        grainstats_additions[grain_no] = {
            "image": filename,
            "grain_number": int(grain_no.split("_")[-1]),
        }
        tracing_stats.pop("circular")
        grainstats_additions[grain_no].update(tracing_stats)

        # remap the cropped images back onto the original
        for image_name, full_image in ordered_trace_full_images.items():
            crop = images[image_name]
            bbox = disordered_trace_data["bbox"]
            full_image[bbox[0] : bbox[2], bbox[1] : bbox[3]] += crop[pad_width:-pad_width, pad_width:-pad_width]
        """
        except Exception as e:
            LOGGER.error(f"[{filename}] : Ordered tracing for {grain_no} failed with - {e}")
            all_traces_data[grain_no] = {}
        """
    grainstats_additions_df = pd.DataFrame.from_dict(grainstats_additions, orient="index")

    return all_traces_data, grainstats_additions_df, ordered_trace_full_images
