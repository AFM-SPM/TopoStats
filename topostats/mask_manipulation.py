"""Code for manipulating binary masks."""

import logging
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import ndimage
from skimage import filters
from skimage.morphology import label
from pydantic import BaseModel, Field
import networkx as nx

# pylint: disable=no-name-in-module
from skimage.graph import route_through_array

from topostats.logs.logs import LOGGER_NAME
from topostats.utils import convolve_skeleton
from topostats.tracing.skeletonize import getSkeleton
from topostats.measure.geometry import calculate_mask_width_with_skeleton, calculate_pixel_path_distance

LOGGER = logging.getLogger(LOGGER_NAME)


# TODO: DEBUG IMPORTS, REMOVE FOR PR
import matplotlib.pyplot as plt
from topostats.plottingfuncs import Colormap

colormap = Colormap()
cmap = colormap.get_cmap()
vmin = -3.0
vmax = 4.0


def re_add_holes(
    pixel_to_nm_scaling: float,
    orig_mask: npt.NDArray,
    smoothed_mask: npt.NDArray,
    holearea_min_max: tuple[float | int | None, float | int | None] = (2, None),
) -> npt.NDArray:
    """
    Restore holes in masks that were occluded by dilation.

    As Gaussian dilation smoothing methods can close holes in the original mask, this function obtains those holes
    (based on the general background being the first due to padding) and adds them back into the smoothed mask. When
    paired with ``smooth_mask``, this essentially just smooths the outer edge of the mask.

    Parameters
    ----------
    pixel_to_nm_scaling : float
        Pixel to nanometer scaling of the image.
    orig_mask : npt.NDArray
        Original mask.
    smoothed_mask : npt.NDArray
        Original mask but with inner and outer edged smoothed. The smoothing operation may have closed up important
        holes in the mask.
    holearea_min_max : tuple[float | int | None, float | int | None]
        Tuple of minimum and maximum hole area (in nanometers) to replace from the original mask into the smoothed
        mask.

    Returns
    -------
    npt.NDArray
        Smoothed mask with holes restored.
    """
    # handle Nones
    # If both none, do nothing
    if holearea_min_max[0] is None and holearea_min_max[1] is None:
        return smoothed_mask
    # If min is none, set to 0.0 (use float to avoid inferring int type)
    if holearea_min_max[0] is None:
        hole_area_min: float = 0.0
    else:
        hole_area_min = float(holearea_min_max[0])
    # If max is none, set to inf
    if holearea_min_max[1] is None:
        hole_area_max: float = np.inf
    else:
        hole_area_max = float(holearea_min_max[1])

    # obtain px holesizes
    holesize_min_px = hole_area_min / ((pixel_to_nm_scaling) ** 2)
    holesize_max_px = hole_area_max / ((pixel_to_nm_scaling) ** 2)

    # obtain a hole mask
    holes = 1 - orig_mask
    holes = label(holes)
    hole_sizes = [holes[holes == i].size for i in range(1, holes.max() + 1)]
    holes[holes == 1] = 0  # set background to 0 assuming it is the first hole seen (from top left)

    # remove too small or too big holes from mask
    for i, hole_size in enumerate(hole_sizes):
        if hole_size < holesize_min_px or hole_size > holesize_max_px:  # small holes may be fake are left out
            holes[holes == i + 1] = 0
    holes[holes != 0] = 1  # set correct sixe holes to 1

    # replace correct sized holes
    return np.where(holes == 1, 0, smoothed_mask)


def smooth_mask(
    filename: str,
    pixel_to_nm_scaling: float,
    grain: npt.NDArray,
    dilation_iterations: int = 2,
    gaussian_sigma: float | int = 2,
    holearea_min_max: tuple[int | float | None, int | float | None] = (0, None),
) -> npt.NDArray:
    """
    Smooth a grain mask based on the lower number of binary pixels added from dilation or gaussian.

    This method ensures gaussian smoothing isn't too aggressive and covers / creates gaps in the mask.

    Parameters
    ----------
    filename : str
        Filename of the image being processed (for logging purposes).
    pixel_to_nm_scaling : float
        Pixel to nanometer scaling of the image.
    grain : npt.NDArray
        Numpy array of the grain mask.
    dilation_iterations : int
        Number of times to dilate the grain to smooth it. Default is 2.
    gaussian_sigma : float | None
        Gaussian sigma value to smooth the grains after an Otsu threshold. If None, defaults to 2.
    holearea_min_max : tuple[float | int | None]
        Tuple of minimum and maximum hole area (in nanometers) to replace from the original mask into the smoothed
        mask.

    Returns
    -------
    npt.NDArray
        Numpy array of smoothed image.
    """
    # Option to disable the smoothing (i.e. U-Net masks are already smooth)
    if dilation_iterations is None and gaussian_sigma is None:
        LOGGER.debug(f"[{filename}] : no grain smoothing done")
        return grain

    # Option to only do gaussian or dilation
    if dilation_iterations is not None:
        dilation = ndimage.binary_dilation(grain, iterations=dilation_iterations).astype(np.int32)
    else:
        gauss = filters.gaussian(grain, sigma=gaussian_sigma)
        gauss = np.where(gauss > filters.threshold_otsu(gauss) * 1.3, 1, 0)
        gauss = gauss.astype(np.int32)
        LOGGER.debug(f"[{filename}] : smoothing done by gaussian {gaussian_sigma}")
        return re_add_holes(grain, gauss, holearea_min_max)
    if gaussian_sigma is not None:
        gauss = filters.gaussian(grain, sigma=gaussian_sigma)
        gauss = np.where(gauss > filters.threshold_otsu(gauss) * 1.3, 1, 0)
        gauss = gauss.astype(np.int32)
    else:
        LOGGER.debug(f"[{filename}] : smoothing done by dilation {dilation_iterations}")
        return re_add_holes(grain, dilation, holearea_min_max)

    # Competition option between dilation and gaussian mask differences wrt original grains
    if abs(dilation.sum() - grain.sum()) > abs(gauss.sum() - grain.sum()):
        LOGGER.debug(f"[{filename}] : smoothing done by gaussian {gaussian_sigma}")
        return re_add_holes(
            pixel_to_nm_scaling=pixel_to_nm_scaling,
            orig_mask=grain,
            smoothed_mask=gauss,
            holearea_min_max=holearea_min_max,
        )
    LOGGER.debug(f"[{filename}] : smoothing done by dilation {dilation_iterations}")
    return re_add_holes(
        pixel_to_nm_scaling=pixel_to_nm_scaling,
        orig_mask=grain,
        smoothed_mask=dilation,
        holearea_min_max=holearea_min_max,
    )


def keep_only_nonrepeated_endpoints(
    potential_pairs: list[tuple[tuple[int, int], tuple[int, int], float]],
) -> list[tuple[tuple[int, int], tuple[int, int], float]]:
    """
    From a list of potential endpoint pairs to connect, remove any pairs that share endpoints with other pairs.

    Parameters
    ----------
    potential_pairs : list[tuple[tuple[int, int], tuple[int, int], float]]
        List of potential endpoint pairs to connect, with their distance in nanometers.

    Returns
    -------
    list[tuple[tuple[int, int], tuple[int, int], float]]
        List of endpoint pairs with no repeated endpoints.
    """

    used_endpoints: list[tuple[int, int]] = []
    for potential_pair in potential_pairs:
        endpoint_1, endpoint_2, _distance_nm = potential_pair
        used_endpoints.append((endpoint_1[0], endpoint_1[1]))
        used_endpoints.append((endpoint_2[0], endpoint_2[1]))

    repeated_endpoints = set([ep for ep in used_endpoints if used_endpoints.count(ep) > 1])

    pairs_no_repeated_ends: list[tuple[tuple[int, int], tuple[int, int], float]] = []
    for potential_pair in potential_pairs:
        endpoint_1, endpoint_2, _distance_nm = potential_pair
        if (endpoint_1[0], endpoint_1[1]) not in repeated_endpoints and (
            endpoint_2[0],
            endpoint_2[1],
        ) not in repeated_endpoints:
            pairs_no_repeated_ends.append(potential_pair)
        else:
            print(f"excluding pair {endpoint_1}, {endpoint_2} due to repeated endpoints")

    return pairs_no_repeated_ends


class Endpoint(BaseModel):
    """Represents an endpoint in a skeletonised mask."""

    id: int
    position: tuple[int, int]


class Junctionpoint(BaseModel):
    """Represents a junction point in a skeletonised mask."""

    id: int
    position: tuple[int, int]


class ConnectionGroup(BaseModel):
    """Represents a group of endpoints to consider for connection."""

    id: int
    endpoints: dict[int, Endpoint] = Field(default_factory=dict)
    junctionpoints: dict[int, Junctionpoint] = Field(default_factory=dict)
    hard_connected_endpoints: list[tuple[int, int, float]] = Field(default_factory=list)
    close_connectionpoint_pairs: list[tuple[int, int]] = Field(default_factory=list)


# pylint: disable=too-many-locals
def group_connectionpoints(
    connectionpoints: dict[int, Endpoint | Junctionpoint],
    close_pairs: list[tuple[int, int, float]],
    draw_graph: bool = False,
) -> dict[int, ConnectionGroup]:
    """
    Group connectionpoints into connection groups based on interconnections.
    """
    # Split the graph into connected groups

    # Create a graph
    G = nx.Graph()
    # Add the nodes (endpoints) as endpoint IDs
    for connectionpoint_index, _connectionpoint in connectionpoints.items():
        G.add_node(connectionpoint_index)
    # Add edges for each close pair (as endpoint IDs)
    for connectionpoint_1_index, connectionpoint_2_index, _distance_nm in close_pairs:
        G.add_edge(connectionpoint_1_index, connectionpoint_2_index)

    # draw it
    if draw_graph:
        nx.draw(G, with_labels=True)
        plt.show()

    # Get networkx to find connected components
    connected_components: list[set[int]] = list(nx.connected_components(G))

    # Create ConnectionGroup objects for each connected component
    connection_groups: dict[int, ConnectionGroup] = {}
    for group_id, component in enumerate(connected_components):
        connectionpoints_group = {
            connectionpoint_index: connectionpoints[connectionpoint_index] for connectionpoint_index in component
        }
        # Get the close pairs that are within this component
        group_close_pairs = [
            (connectionpoint_1_index, connectionpoint_2_index)
            for connectionpoint_1_index, connectionpoint_2_index, _distance_nm in close_pairs
            if connectionpoint_1_index in component and connectionpoint_2_index in component
        ]
        # Create dictionaries of endpoints and junctionpoints, since they have been combined and we need to separate them
        # again
        group_endpoints = {
            connectionpoint_index: connectionpoint_instance
            for connectionpoint_index, connectionpoint_instance in connectionpoints_group.items()
            if isinstance(connectionpoint_instance, Endpoint)
        }
        group_junctionpoints = {
            connectionpoint_index: connectionpoint_instance
            for connectionpoint_index, connectionpoint_instance in connectionpoints_group.items()
            if isinstance(connectionpoint_instance, Junctionpoint)
        }
        connection_group = ConnectionGroup(
            id=group_id,
            endpoints=group_endpoints,
            junctionpoints=group_junctionpoints,
            close_connectionpoint_pairs=group_close_pairs,
        )
        connection_groups[group_id] = connection_group
    return connection_groups


# pylint: disable=too-many-arguments
def connect_endpoints_with_best_path(
    image: npt.NDArray[np.float32],
    mask: npt.NDArray[np.bool_],
    p2nm: float,
    endpoint_1_coords: tuple[int, int],
    endpoint_2_coords: tuple[int, int],
    endpoint_connection_cost_map_height_maximum: float,
) -> tuple[npt.NDArray[np.uint8], float, float]:
    """
    Connect two endpoints with a path found by pathfinding through a cost map derived from the image heights.
    """
    # create a weight cost map from the image, where 0 is the maximum cost, and the lowest cost is configurable.
    # first create a crop around the two endpoints to speed up pathfinding
    cost_map_bbox_padding_px = 10
    min_y = max(0, min(endpoint_1_coords[0], endpoint_2_coords[0]) - cost_map_bbox_padding_px)
    max_y = min(image.shape[0], max(endpoint_1_coords[0], endpoint_2_coords[0]) + cost_map_bbox_padding_px)
    min_x = max(0, min(endpoint_1_coords[1], endpoint_2_coords[1]) - cost_map_bbox_padding_px)
    max_x = min(image.shape[1], max(endpoint_1_coords[1], endpoint_2_coords[1]) + cost_map_bbox_padding_px)
    cost_map = image[min_y:max_y, min_x:max_x]
    _mask_crop = mask[min_y:max_y, min_x:max_x]
    _image_crop = image[min_y:max_y, min_x:max_x]
    local_endpoint_1 = (endpoint_1_coords[0] - min_y, endpoint_1_coords[1] - min_x)
    local_endpoint_2 = (endpoint_2_coords[0] - min_y, endpoint_2_coords[1] - min_x)
    # clip it to the height bounds
    cost_map = np.clip(
        cost_map,
        a_min=0,
        a_max=endpoint_connection_cost_map_height_maximum,
    )
    # invert it
    cost_map = endpoint_connection_cost_map_height_maximum - cost_map
    # normalise to 0-1
    cost_map = cost_map / endpoint_connection_cost_map_height_maximum

    # find the lowest cost path between the two endpoints
    path, cost = route_through_array(
        cost_map,
        start=local_endpoint_1,
        end=local_endpoint_2,
        fully_connected=True,  # allow diagonal moves
    )

    cost = float(cost)

    # Convert the path back to the original image coordinates
    path = [(y + min_y, x + min_x) for y, x in path]
    # Convert to numpy array for easier indexing
    path = np.array(path)

    # Calculate the distance of the path in nm, taking into account diagnonal distances
    path_distance_nm = calculate_pixel_path_distance(path) * p2nm

    return path, cost, path_distance_nm


def fill_mask_gap_using_path(
    mask: npt.NDArray[np.bool_],
    path: npt.NDArray[np.uint8],
    mean_mask_pixel_width: float,
) -> npt.NDArray[np.bool_]:
    """
    Fill a gap in a binary mask using a given path, dilating the path to match the mean mask width.
    """

    path_mask = np.zeros_like(mask, dtype=bool)
    # Set the path to True
    for y, x in path:
        path_mask[y, x] = True
    # Calculate the dilation iterations needed to reach the mean mask width
    dilation_radius = int(np.ceil(mean_mask_pixel_width / 2))
    dilated_path_array = ndimage.binary_dilation(
        path_mask,
        iterations=dilation_radius,
    )

    # Add the dilated path to the whole mask
    filled_mask = mask | dilated_path_array

    return filled_mask


def find_hard_connected_endpoints(
    skeleton: npt.NDArray[np.bool_],
    endpoints: dict[int, Endpoint],
    max_connection_distance_nm: float,
    p2nm: float,
) -> list[tuple[int, int, float]]:
    """
    Find hard-connected endpoints (with distances) within a maximum connection distance in a skeleton.

    Specifically, finds the pairs of endpoints, and the distances between them in nanometres, that are connected
    via the skeleton within the given maximum connection distance.
    """
    hard_connected_endpoints: list[tuple[int, int, float]] = []

    # Breadth-first search from each endpoint to find reachable endpoints within the max distance
    for endpoint_id, endpoint in endpoints.items():
        start_position = endpoint.position
        visited: set[tuple[int, int]] = set()
        queue: list[tuple[tuple[int, int], float]] = [(start_position, 0.0)]  # (position, distance_nm)
        while queue:
            current_position = queue.pop(0)
            current_coords, current_distance_nm = current_position
            # If alrady visited this position, skip it.
            if current_coords in visited:
                continue
            # Else add it to the visited set
            visited.add(current_coords)
            # Check if this position is another endpoint (not the start endpoint)
            for other_endpoint_id, other_endpoint in endpoints.items():
                if other_endpoint_id != endpoint_id and other_endpoint.position == current_coords:
                    # This is a hard-connection
                    hard_connected_endpoints.append((endpoint_id, other_endpoint_id, current_distance_nm))
            # Explore neighbours (8-connectivity)
            y, x = current_coords
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        # Skip current position
                        continue
                    neighbour_coords = (y + dy, x + dx)
                    # Check if neighbour is within bounds of the skeleton and is part of the skeleton's true pixels
                    if (
                        0 <= neighbour_coords[0] < skeleton.shape[0]
                        and 0 <= neighbour_coords[1] < skeleton.shape[1]
                        and skeleton[neighbour_coords]
                    ):
                        # Calculate distance to neighbour, taking into account diagonal distance
                        step_distance_nm = np.sqrt(dy**2 + dx**2) * p2nm
                        new_distance_nm = current_distance_nm + step_distance_nm
                        # If within max distance, add to queue
                        if new_distance_nm <= max_connection_distance_nm:
                            queue.append((neighbour_coords, new_distance_nm))
    return hard_connected_endpoints


# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=too-many-positional-arguments
def skeletonise_and_join_close_ends(
    filename: str,
    p2nm: float,
    image: npt.NDArray,
    mask: npt.NDArray,
    skeletonisation_holearea_min_max: tuple[int | None, int | None],
    skeletonisation_mask_smoothing_dilation_iterations: int,
    skeletonisation_mask_smoothing_gaussian_sigma: float,
    skeletonisation_method: str,
    skeletonisation_height_bias: float,
    endpoint_connection_distance_nm: float,
    endpoint_connection_cost_map_height_maximum: float,
) -> npt.NDArray:
    """
    Fill-in gaps in grain masks by skeletonising all grains and connecting close endpoints with mask-width wide paths.

    Parameters
    ----------
    filename : str
        Filename of the image.
    p2nm : float
        Pixel to nanometre scaling factor.
    image : npt.NDArray
        2-D Numpy array of the image.
    mask : npt.NDArray
        2-D Numpy array of the grain mask.
    skeletonisation_holearea_min_max : tuple[int | None, int | None]
        Minimum and maximum hole area to fill in the mask smoothing step.
    skeletonisation_mask_smoothing_dilation_iterations : int
        Number of dilation iterations to perform during smoothing of the mask before skeletonisation.
    skeletonisation_mask_smoothing_gaussian_sigma : float
        Sigma of the Gaussian filter to apply during smoothing of the mask before skeletonisation.
    skeletonisation_method : str
        Method to use for skeletonisation.
    skeletonisation_height_bias : float
        Percentage of lowest pixels to remove each skeletonisation iteration. 1 equates to zhang.
    endpoint_connection_distance_nm : float
        Maximum distance between skeleton endpoints to connect (nm).
    endpoint_connection_cost_map_height_maximum : float
        Maximum height to use for the cost map when connecting endpoints. (Should roughly be the maximum height of the
        data in nm).
    plot : bool, optional
        Whether to plot intermediate steps for debugging, by default False.

    Returns
    -------
    npt.NDArray
        2-D Numpy array of the updated grain mask with gaps filled in.
    """

    # skeletonisation_holearea_min_max = (0, None)
    # skeletonisation_mask_smoothing_dilation_iterations = 2
    # skeletonisation_mask_smoothing_gaussian_sigma = 2
    # skeletonisation_method = "topostats"
    # skeletonisation_height_bias = 0.6
    # endpoint_connection_distance_nm = 10
    # endpoint_connection_cost_map_height_maximum = 3.0

    smoothed_mask = smooth_mask(
        filename=filename,
        pixel_to_nm_scaling=p2nm,
        grain=mask,
        gaussian_sigma=skeletonisation_mask_smoothing_gaussian_sigma,
        holearea_min_max=skeletonisation_holearea_min_max,
        dilation_iterations=skeletonisation_mask_smoothing_dilation_iterations,
    )

    # Maybe need to check it doesn't touch the edge of the image like we do in disordered_tracing? unsure.

    # Next step, skeletonize
    skeleton = getSkeleton(
        image=image,
        mask=smoothed_mask,
        method=skeletonisation_method,
        height_bias=skeletonisation_height_bias,
    ).get_skeleton()

    # Calculate the mask width along the skeleton for later
    mean_mask_width_nm = calculate_mask_width_with_skeleton(
        mask=smoothed_mask,
        skeleton=skeleton,
        pixel_to_nm_scaling=p2nm,
    )
    mean_mask_width_px = mean_mask_width_nm / p2nm

    # Now to find the skeleton endpoints and connect close ones.
    convolved_skeleton = convolve_skeleton(skeleton=skeleton)
    # Get the endpoints, value = 2
    endpoint_coords: list[tuple[int, int]] = [tuple(coord) for coord in np.argwhere(convolved_skeleton == 2)]
    # Get the junctions, value = 3
    junction_coords: list[tuple[int, int]] = [tuple(coord) for coord in np.argwhere(convolved_skeleton >= 3)]

    # Create dictionary of connectionpoints
    connectionpoints: dict[int, Endpoint | Junctionpoint] = {}
    connectionpoint_index = 0
    for endpoint_coord in endpoint_coords:
        endpoint = Endpoint(id=connectionpoint_index, position=endpoint_coord)
        connectionpoints[connectionpoint_index] = endpoint
        connectionpoint_index += 1
    for junction_coord in junction_coords:
        junctionpoint = Junctionpoint(id=connectionpoint_index, position=junction_coord)
        connectionpoints[connectionpoint_index] = junctionpoint
        connectionpoint_index += 1

    # For each connectionpoint, determine if any others are close enough to connect
    nearby_connectionpoint_pairs: list[tuple[int, int, float]] = []
    for connectionpoint_1_id, _connectionpoint_1 in connectionpoints.items():
        for connectionpoint_2_id, _connectionpoint_2 in connectionpoints.items():
            if connectionpoint_1_id >= connectionpoint_2_id:
                continue  # avoid double counting
            # calculate absolute distance between them
            connection_distance_nm = float(
                np.linalg.norm((np.array(_connectionpoint_1.position) - np.array(_connectionpoint_2.position)) * p2nm)
            )
            if connection_distance_nm <= endpoint_connection_distance_nm:
                nearby_connectionpoint_pairs.append(
                    (connectionpoint_1_id, connectionpoint_2_id, float(connection_distance_nm))
                )

    # Group neraby endpoints into connection groups based on connectivity
    connection_groups = group_connectionpoints(
        connectionpoints=connectionpoints, close_pairs=nearby_connectionpoint_pairs
    )

    # Now consider each group and decide how to connect them
    for group_id, connection_group in connection_groups.items():
        LOGGER.info(
            f"[{filename}] : processing connection group {group_id} with " f"endpoints : {connection_group.endpoints}"
        )

        # If there is one junctionpoint and one endpoint, connect them directly
        if len(connection_group.junctionpoints) == 1 and len(connection_group.endpoints) == 1:
            endpoint_id = list(connection_group.endpoints.keys())[0]
            junctionpoint_id = list(connection_group.junctionpoints.keys())[0]
            endpoint_coord = connection_group.endpoints[endpoint_id].position
            junctionpoint_coord = connection_group.junctionpoints[junctionpoint_id].position

            path, _connection_cost, _connection_distance_nm = connect_endpoints_with_best_path(
                image=image,
                mask=mask,
                p2nm=p2nm,
                endpoint_1_coords=endpoint_coord,
                endpoint_2_coords=junctionpoint_coord,
                endpoint_connection_cost_map_height_maximum=endpoint_connection_cost_map_height_maximum,
            )

            # Update the skeleton to include the path
            for y, x in path:
                skeleton[y, x] = True

            # Fill the mask gap using the path
            mask = fill_mask_gap_using_path(
                mask=mask,
                path=path,
                mean_mask_pixel_width=mean_mask_width_px,
            )
            # done with this group, move on
            continue

        # If there are only 2 endpoints and no junctionpointsin the group, we can just connect them.
        if len(connection_group.endpoints) == 2 and len(connection_group.junctionpoints) == 0:
            pair = connection_group.close_connectionpoint_pairs[0]
            endpoint_1_id, endpoint_2_id = pair
            endpoint_1_coords = connection_group.endpoints[endpoint_1_id]
            endpoint_2_coords = connection_group.endpoints[endpoint_2_id]

            path, _connection_cost, _connection_distance_nm = connect_endpoints_with_best_path(
                image=image,
                mask=mask,
                p2nm=p2nm,
                endpoint_1_coords=endpoint_1_coords.position,
                endpoint_2_coords=endpoint_2_coords.position,
                endpoint_connection_cost_map_height_maximum=endpoint_connection_cost_map_height_maximum,
            )

            # Update the skeleton to include the path
            for y, x in path:
                skeleton[y, x] = True

            # Fil the mask using the path
            mask = fill_mask_gap_using_path(
                mask=mask,
                path=path,
                mean_mask_pixel_width=mean_mask_width_px,
            )
            # done with this group, move on
            continue

        # If there are 4 endpoints and no junctionpoints, we can try to connect them.
        if len(connection_group.endpoints) == 4 and len(connection_group.junctionpoints) == 0:

            # Find the hard-connected endpoints - endpoints that are connected by paths in the skeleton already.
            hard_connected_endpoints: list[tuple[int, int, float]] = find_hard_connected_endpoints(
                skeleton=skeleton,
                endpoints=connection_group.endpoints,
                max_connection_distance_nm=endpoint_connection_distance_nm,
                p2nm=p2nm,
            )
            connection_group.hard_connected_endpoints = hard_connected_endpoints

            # Identify the start and end nodes by being the only nodes that do not have hard-connections to the other
            # nodes in the group.
            non_hard_connected_endpoint_ids: set[int] = set(connection_group.endpoints.keys())
            for endpoint_1_id, endpoint_2_id, _ in connection_group.hard_connected_endpoints:
                if endpoint_1_id in non_hard_connected_endpoint_ids:
                    non_hard_connected_endpoint_ids.remove(endpoint_1_id)
                if endpoint_2_id in non_hard_connected_endpoint_ids:
                    non_hard_connected_endpoint_ids.remove(endpoint_2_id)
            if len(non_hard_connected_endpoint_ids) != 2:
                LOGGER.info(
                    f"[{filename}] : connection group {group_id} has "
                    f"{len(connection_group.endpoints)} endpoints, "
                    f"but {len(non_hard_connected_endpoint_ids)} non-hard-connected endpoints, "
                    f"currently unsupported, skipping."
                )
                continue

            # So now we have the start and end nodes identified.
            start_node_id, end_node_id = non_hard_connected_endpoint_ids
            _start_node = connection_group.endpoints[start_node_id]
            _end_node = connection_group.endpoints[end_node_id]

            # Find the best path between the start and end node, traversing all nodes, using the shortest distance
            # as the best-path criteria.

            # Since this is just a 4-node system, with a start and end node, there are only two possible paths:
            # start -> node A -> node B -> end
            # start -> node B -> node A -> end

            # The metrics for quality of the options are: total length and total cost.

            # For each option, pathfind using height cost maps to connect the disconnected nodes.

            # Create a list of possible paths by order of node traversal
            node_ids = list(connection_group.endpoints.keys())
            node_ids.remove(start_node_id)
            node_ids.remove(end_node_id)
            node_a_id = node_ids[0]
            node_b_id = node_ids[1]
            path_options = [
                [start_node_id, node_a_id, node_b_id, end_node_id],
                [start_node_id, node_b_id, node_a_id, end_node_id],
            ]
            path_metrics: list[dict[str, Any]] = []  # to store metrics for each path option
            for path_option in path_options:
                total_cost = 0.0
                total_distance_nm: float = 0.0
                path_segments: list[tuple[int, int, float, npt.NDArray[np.uint8] | None]] = []
                # Iterate through each segment of the path
                for i in range(len(path_option) - 1):
                    hard_connection_current_distance_nm: float | None = None
                    segment_start_id = path_option[i]
                    segment_end_id = path_option[i + 1]
                    segment_start_coords = connection_group.endpoints[segment_start_id].position
                    segment_end_coords = connection_group.endpoints[segment_end_id].position
                    # Check if the segment is a hard-connection
                    is_hard_connection = False
                    for hard_connection in connection_group.hard_connected_endpoints:
                        hard_connection_start_id, hard_connection_end_id, hard_connection_distance_nm = hard_connection
                        if (
                            segment_start_id == hard_connection_start_id and segment_end_id == hard_connection_end_id
                        ) or (
                            segment_start_id == hard_connection_end_id and segment_end_id == hard_connection_start_id
                        ):
                            is_hard_connection = True
                            hard_connection_current_distance_nm = hard_connection_distance_nm
                            break
                        if is_hard_connection:
                            assert hard_connection_current_distance_nm is not None
                            total_distance_nm += hard_connection_current_distance_nm
                            path_segments.append(
                                (segment_start_id, segment_end_id, hard_connection_current_distance_nm, None)
                            )
                        else:
                            # pathfind between the two endpoints
                            path, connection_cost, connection_distance_nm = connect_endpoints_with_best_path(
                                image=image,
                                mask=mask,
                                p2nm=p2nm,
                                endpoint_1_coords=segment_start_coords,
                                endpoint_2_coords=segment_end_coords,
                                endpoint_connection_cost_map_height_maximum=endpoint_connection_cost_map_height_maximum,
                            )
                            total_cost += connection_cost
                            total_distance_nm += float(connection_distance_nm)
                            # add to path segments
                            path_segments.append(
                                (segment_start_id, segment_end_id, float(connection_distance_nm), path)
                            )
                    # Store the metrics for this path option
                    path_metrics.append(
                        {
                            "path_option": path_option,
                            "total_cost": total_cost,
                            "total_distance_nm": total_distance_nm,
                            "path_segments": path_segments,
                        }
                    )
                # Select the best path option based on the metrics
                best_path_option = min(path_metrics, key=lambda x: (x["total_distance_nm"]))

                # With the best path option, update the skeleton and mask
                for segment in best_path_option["path_segments"]:
                    segment_start_id, segment_end_id, _segment_distance_nm, path = segment
                    if path is not None:
                        # Non-hard-connection, so update the skeleton and mask
                        for (
                            y,
                            x,
                        ) in path:
                            skeleton[y, x] = True
                        # Fill the mask using the path
                        mask = fill_mask_gap_using_path(
                            mask=mask,
                            path=path,
                            mean_mask_pixel_width=mean_mask_width_px,
                        )

            # done with this group, move on.
            continue

        LOGGER.info(
            f"[{filename}] : connection group {group_id} has "
            f"{len(connection_group.endpoints)} endpoints, "
            f"currently unsupported, skipping."
        )
    return mask


def multi_class_skeletonise_and_join_close_ends(
    class_indices: list[int],
    tensor: npt.NDArray,
    filename: str,
    p2nm: float,
    image: npt.NDArray,
    skeletonisation_holearea_min_max: tuple[int | None, int | None],
    skeletonisation_mask_smoothing_dilation_iterations: int,
    skeletonisation_mask_smoothing_gaussian_sigma: float,
    skeletonisation_method: str,
    skeletonisation_height_bias: float,
    endpoint_connection_distance_nm: float,
    endpoint_connection_cost_map_height_maximum: float,
) -> npt.NDArray:
    """
    Perform joining of close-ends in masks for given classes in a multi-class mask tensor.

    Parameters
    ----------
    class_indices : int
        Index of the class to process in the multi-class mask tensor.
    tensor : npt.NDArray
        Multi-class mask tensor.
    filename : str
        Filename of the image.
    p2nm : float
        Pixel to nanometre scaling factor.
    image : npt.NDArray
        2-D Numpy array of the image.
    skeletonisation_holearea_min_max : tuple[int | None, int | None]
        Minimum and maximum hole area to fill in the mask smoothing step.
    skeletonisation_mask_smoothing_dilation_iterations : int
        Number of dilation iterations to perform during smoothing of the mask before skeletonisation.
    skeletonisation_mask_smoothing_gaussian_sigma : float
        Sigma of the Gaussian filter to apply during smoothing of the mask before skeletonisation.
    skeletonisation_method : str
        Method to use for skeletonisation.
    skeletonisation_height_bias : float
        Percentage of lowest pixels to remove each skeletonisation iteration. 1 equates to zhang.
    endpoint_connection_distance_nm : float
        Maximum distance between skeleton endpoints to connect (nm).
    endpoint_connection_cost_map_height_maximum : float
        Maximum height to use for the cost map when connecting endpoints. (Should roughly be the maximum height of the
        data in nm).

    Returns
    -------
    npt.NDArray
        Updated multi-class mask tensor with close ends joined for the specified class.
    """
    result_tensor = tensor.copy()
    for class_index in class_indices:
        mask = tensor[:, :, class_index]
        updated_mask = skeletonise_and_join_close_ends(
            filename=filename,
            p2nm=p2nm,
            image=image,
            mask=mask,
            skeletonisation_holearea_min_max=skeletonisation_holearea_min_max,
            skeletonisation_mask_smoothing_dilation_iterations=skeletonisation_mask_smoothing_dilation_iterations,
            skeletonisation_mask_smoothing_gaussian_sigma=skeletonisation_mask_smoothing_gaussian_sigma,
            skeletonisation_method=skeletonisation_method,
            skeletonisation_height_bias=skeletonisation_height_bias,
            endpoint_connection_distance_nm=endpoint_connection_distance_nm,
            endpoint_connection_cost_map_height_maximum=endpoint_connection_cost_map_height_maximum,
        )
        result_tensor[:, :, class_index] = updated_mask
    return result_tensor
