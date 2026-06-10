"""Perform close strand correction."""

import copy

import numpy as np
import numpy.typing as npt
from skimage.graph import route_through_array
from scipy.ndimage import convolve
from skimage.morphology import binary_dilation

from topostats.classes import TopoStats


def get_neighbouring_true_pixels(
    mask: npt.NDArray[np.bool_], coord: npt.NDArray[np.integer]
) -> list[npt.NDArray[np.integer]]:
    """
    Get the coordinates of neighbouring true pixels in a binary mask.

    Parameters
    ----------
    mask : npt.NDArray[np.bool_]
        The binary mask.
    coord : npt.NDArray[np.integer]
        The coordinate to get neighbours for.

    Returns
    -------
    list[npt.NDArray[np.integer]]
        A list of coordinates of neighbouring true pixels.
    """
    neighbours = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            neighbour_coord = coord + np.array([dx, dy])
            if (
                neighbour_coord[0] >= 0
                and neighbour_coord[0] < mask.shape[0]
                and neighbour_coord[1] >= 0
                and neighbour_coord[1] < mask.shape[1]
            ):
                if mask[neighbour_coord[0], neighbour_coord[1]]:
                    neighbours.append(neighbour_coord)
    return neighbours


def get_point_along_branch(
    skeleton: npt.NDArray[np.bool_], start_coord: npt.NDArray[np.integer], distance_px: float
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]]:
    """
    Get the coordinate and path taken of a point along a branch at a specified distance from a starting coordinate.

    Parameters
    ----------
    skeleton : npt.NDArray[np.bool_]
        The skeleton image.
    start_coord : npt.NDArray[np.integer]
        The starting coordinate.
    distance_px : float
        The distance in pixels.

    Returns
    -------
    tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]]
        The coordinate of the point along the branch at the specified distance, and the path taken to get there.
    """
    skeleton_tracker = copy.deepcopy(skeleton)
    distance_travelled_px = 0.0
    path_taken = [start_coord]
    current_coord = start_coord
    skeleton_tracker[current_coord[0], current_coord[1]] = False
    neighbours = get_neighbouring_true_pixels(skeleton_tracker, current_coord)
    # note: if we ever encounter more than one neighbour, we stop and return that point, as it's
    # a crossing / node, and we don't want to have to define how to deicde where to go from there.
    while distance_travelled_px < distance_px:
        if len(neighbours) == 0:
            # we've reached the end of the branch before reaching the desired distance, so we return the last point
            break
        if len(neighbours) > 1:
            # we've reached a crossing / node before reaching the desired distance, so we return the current point
            break
        next_coord = neighbours[0]
        distance_travelled_px += np.linalg.norm(next_coord - current_coord)
        current_coord = next_coord
        path_taken.append(current_coord)
        skeleton_tracker[current_coord[0], current_coord[1]] = False
        neighbours = get_neighbouring_true_pixels(skeleton_tracker, current_coord)
    return current_coord, np.array(path_taken)


def correct_close_strands_image(
    topostats_object: TopoStats,
    class_index: int,
    height_threshold_nm: float,
    branch_explore_distance_nm: float,
    cost_image_exponent: float,
    cost_image_base: float,
    crossing_correction_strand_minimum_height_nm: float,
) -> None:
    """Correct close strands in the image."""

    crossing_data = []

    for grain_index, graincrop in topostats_object.require_grain_crops().items():
        grain_image = graincrop.image
        grain_image_normalised = (grain_image - np.min(grain_image)) / (
            np.max(grain_image) - np.min(grain_image) + 1e-8
        )
        grain_cost_image = (1.0 - grain_image_normalised) ** cost_image_exponent + cost_image_base
        pixel_to_nm_scaling = graincrop.pixel_to_nm_scaling
        disordered_trace_images = graincrop.disordered_trace.images
        assert (
            disordered_trace_images is not None
        ), f"Missing attribute: disordered_trace_images for grain {grain_index}"
        pruned_skeleton = disordered_trace_images["pruned_skeleton"]
        result_corrected_skeleton = copy.deepcopy(pruned_skeleton)

        # iterate over the nodes
        for node_index, node in graincrop.nodes.items():
            node_area_skeleton = node.node_area_skeleton
            assert (
                node_area_skeleton is not None
            ), f"Missing attribute: node_area_skeleton for node {node_index} in grain {grain_index}"
            mask_node = np.where(node_area_skeleton == 3, 1, 0)
            mask_branches = np.where(node_area_skeleton == 1, 1, 0)
            # get the maximum height of any node pixel from the image
            node_pixel_coords = np.argwhere(mask_node == 1)
            node_pixel_heights = grain_image[node_pixel_coords[:, 0], node_pixel_coords[:, 1]]
            max_node_pixel_height = np.max(node_pixel_heights)
            if max_node_pixel_height > height_threshold_nm:
                # this is a true crossing (probably)
                crossing_data.append(
                    {
                        "grain_index": grain_index,
                        "node_index": node_index,
                        "true_crossing": True,
                        "max_node_pixel_height": max_node_pixel_height,
                        "coords": node_pixel_coords,
                        "centroid": np.mean(node_pixel_coords, axis=0),
                    }
                )
                # do true crossing adjustment
                # TODO: true crossing adjustments
            else:
                # this is a false crossing (probably)
                crossing_data.append(
                    {
                        "grain_index": grain_index,
                        "node_index": node_index,
                        "true_crossing": False,
                        "max_node_pixel_height": max_node_pixel_height,
                        "coords": node_pixel_coords,
                        "centroid": np.mean(node_pixel_coords, axis=0),
                    }
                )
                # false crossing fixing
                mask_dilated_node = binary_dilation(mask_node, footprint=np.ones((3, 3)))
                mask_dilated_node_branch_intersections = mask_dilated_node * mask_branches
                branch_starts = np.argwhere(mask_dilated_node_branch_intersections == 1)
                if branch_starts.shape[0] != 4:
                    # to fix a false crossing, we need 4 branches - skip this node if not the case.
                    continue
                point_a, path_a = get_point_along_branch(
                    mask_branches, branch_starts[0], branch_explore_distance_nm / pixel_to_nm_scaling
                )
                result_corrected_skeleton[path_a[:, 0], path_a[:, 1]] = False
                point_b, path_b = get_point_along_branch(
                    mask_branches, branch_starts[1], branch_explore_distance_nm / pixel_to_nm_scaling
                )
                result_corrected_skeleton[path_b[:, 0], path_b[:, 1]] = False
                point_c, path_c = get_point_along_branch(
                    mask_branches, branch_starts[2], branch_explore_distance_nm / pixel_to_nm_scaling
                )
                result_corrected_skeleton[path_c[:, 0], path_c[:, 1]] = False
                point_d, path_d = get_point_along_branch(
                    mask_branches, branch_starts[3], branch_explore_distance_nm / pixel_to_nm_scaling
                )
                result_corrected_skeleton[path_d[:, 0], path_d[:, 1]] = False

                # Also remove the node coords from the result skeleton
                result_corrected_skeleton[mask_node == 1] = False

                possible_combinations = [
                    ((point_a, point_b), (point_c, point_d)),
                    ((point_a, point_c), (point_b, point_d)),
                    ((point_a, point_d), (point_b, point_c)),
                ]
                best_combination: (
                    tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]] | None
                ) = None
                highest_lowest_height_along_paths = -np.inf
                for combination in possible_combinations:
                    pair_1, pair_2 = combination
                    # get the costs for pathfinding between the pairs
                    start_1 = pair_1[0]
                    end_1 = pair_1[1]
                    start_2 = pair_2[0]
                    end_2 = pair_2[1]
                    path_1_coords, _path_1_cost = route_through_array(
                        grain_cost_image, start_1, end_1, fully_connected=True
                    )
                    path_1_coords = np.array(path_1_coords)
                    path_1_heights = grain_image[path_1_coords[:, 0], path_1_coords[:, 1]]
                    path_2_coords, _path_2_cost = route_through_array(
                        grain_cost_image, start_2, end_2, fully_connected=True
                    )
                    path_2_coords = np.array(path_2_coords)
                    path_2_heights = grain_image[path_2_coords[:, 0], path_2_coords[:, 1]]

                    lowest_height_along_paths = np.min(np.concatenate([path_1_heights, path_2_heights]))
                    if lowest_height_along_paths > highest_lowest_height_along_paths:
                        highest_lowest_height_along_paths = lowest_height_along_paths
                        best_combination = ((start_1, end_1, path_1_coords), (start_2, end_2, path_2_coords))
                if best_combination is None:
                    raise ValueError("this should never happen, debug this.")
                if highest_lowest_height_along_paths < crossing_correction_strand_minimum_height_nm:
                    # cannot be corrected with confidence.
                    continue
                # apply the correction to the pruned skeleton
                for _, _, path_coords in best_combination:
                    for coord in path_coords:
                        result_corrected_skeleton[coord[0], coord[1]] = 1
        graincrop.skeleton_override = result_corrected_skeleton
        graincrop.overridden_skeleton = pruned_skeleton
