"""Perform close strand correction."""

import copy
import logging

import numpy as np
import numpy.typing as npt
from skimage.graph import route_through_array
from skimage.measure import label
from skimage.morphology import binary_dilation

from topostats.classes import TopoStats
from topostats.logs.logs import LOGGER_NAME

LOGGER = logging.getLogger(LOGGER_NAME)


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


def trace_skeleton_better_through_node(
    node_coords: tuple[int, int],
    peak_coords: tuple[int, int],
    image: npt.NDArray[np.float64],
    skeleton: npt.NDArray[np.bool_],
) -> None:
    euclidean_distance_to_peak_px = np.linalg.norm((peak_coords[0] - node_coords[0]), (peak_coords[1] - node_coords[1]))
    number_of_interpolation_steps = np.max(int(euclidean_distance_to_peak_px * 2.0), 5)

    to_fill_points: list[tuple[int, int]] = []
    for index in range(number_of_interpolation_steps):
        interp_factor = index / (number_of_interpolation_steps - 1)  # since working in range 0 - num -1.
        # find the point along the line proportional to the interpolation factor
        interp_point = (1 - interp_factor) * node_coords + (interp_factor) * peak_coords

        if index == 0:
            to_fill_points.append(node_coords)
        if index == number_of_interpolation_steps - 1:
            to_fill_points.append(peak_coords)

        vector_of_interpolated_point_peak_line = (peak_coords[0] - node_coords[0], peak_coords[1] - node_coords[1])
        vector_of_interpolated_point_peak_line_length_px = np.linalg.norm(vector_of_interpolated_point_peak_line)
        normal_of_interpolated_point_peak_line = (
            np.array([-vector_of_interpolated_point_peak_line[1], vector_of_interpolated_point_peak_line[0]])
            / vector_of_interpolated_point_peak_line_length_px
        )

        # search for the best (highest) point along the normal, tapering the search distance as we get closer to the
        # peak point to prevent sharp turns
        best_point: tuple[int, int] = (int(interp_point[0]), int(interp_point[1]))
        maximum_height = -np.inf
        search_offsets_px = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
        for offset in search_offsets_px:
            # taper the offset as proximity to the peak increases.
            tapered_offset = offset * (1 - interp_factor)
            test_point = interp_point + (normal_of_interpolated_point_peak_line * tapered_offset)
            test_point = (int(test_point[0]), int(test_point[1]))

            # check for image bounds
            if 0 <= test_point[0] < image.shape[1] and 0 <= test_point[1] < image.shape[0]:
                # TODO: check the indexing here
                height = image[test_point[1], test_point[0]]
                if height > maximum_height:
                    maximum_height = height
                    best_point = test_point
        to_fill_points.append(best_point)

    # Fill the skeleton
    for to_fill_point in to_fill_points:
        skeleton[to_fill_point[1], to_fill_point[0]] = True


def correct_close_strands_image(  # noqa: C901
    topostats_object: TopoStats,
    true_peak_threshold_median_multiplier: float,
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
        original_pruned_skeleton = disordered_trace_images["pruned_skeleton"]
        assert (
            np.max(label(original_pruned_skeleton)) == 1
        ), "close strand correction: expected pruned skeleton to only have one connected component for grain"
        f"{grain_index}, but found {np.max(label(original_pruned_skeleton))} - debug this."
        result_corrected_skeleton = copy.deepcopy(original_pruned_skeleton)

        # calculate the median height of the skeleton pixels for the grain
        skeleton_pixel_coords = np.argwhere(original_pruned_skeleton == 1)
        skeleton_pixel_heights = grain_image[skeleton_pixel_coords[:, 0], skeleton_pixel_coords[:, 1]]
        median_skeleton_height = np.median(skeleton_pixel_heights)
        true_peak_threshold = median_skeleton_height * true_peak_threshold_median_multiplier

        # iterate over the nodes
        for node_index, node in graincrop.nodes.items():
            proposed_result_corrected_skeleton = copy.deepcopy(result_corrected_skeleton)
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
            max_node_pixel_coords = node_pixel_coords[np.argmax(node_pixel_heights)]
            if max_node_pixel_height > true_peak_threshold:
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

                # Find node coordinates and clear the paths
                mask_dilated_node = binary_dilation(mask_node, footprint=np.ones((3, 3)))
                mask_dilated_node_branch_intersections = mask_dilated_node * mask_branches
                branch_starts = np.argwhere(mask_dilated_node_branch_intersections == 1)
                if branch_starts.shape[0] != 4:
                    # to fix a true crossing, we need 4 branches - skip this node if not the case.
                    continue

                points: list[tuple[int, int]] = []
                paths = []
                for branch_start in branch_starts:
                    point, path = get_point_along_branch(
                        skeleton=mask_branches,
                        start_coord=branch_start,
                        distance_px=branch_explore_distance_nm / pixel_to_nm_scaling,
                    )
                    points.append((point[0], point[1]))
                    paths.append(path)
                    proposed_result_corrected_skeleton[path[:, 1], path[:, 0]] = False

                # Improve the crossing

                # list the possible combinations of pairs of points to connect
                possible_combinations_indexes = [
                    ((0, 1), (2, 3)),
                    ((0, 2), (1, 3)),
                    ((0, 3), (1, 2)),
                ]

                # get the unit vectors from each node towards the peak pixel coord
                unit_vectors_to_peak_pixel = []
                for point in points:
                    vector = max_node_pixel_coords - point
                    unit_vector = vector / np.linalg.norm(vector)
                    unit_vectors_to_peak_pixel.append(unit_vector)

                # find the combination of pairs of nodes that have the smallest combined dot product of their vectors
                # to the peak. ie - finding the combination where the highest point is closest to the line between the
                # two points
                best_combination_pair = None
                min_dot_product_sum = np.inf
                for combination_indexes in possible_combinations_indexes:
                    pair_0_indexes, pair_1_indexes = combination_indexes
                    combined_to_peak_dot_product = np.dot(
                        unit_vectors_to_peak_pixel[pair_0_indexes[0]], unit_vectors_to_peak_pixel[pair_0_indexes[1]]
                    ) + np.dot(
                        unit_vectors_to_peak_pixel[pair_1_indexes[0]], unit_vectors_to_peak_pixel[pair_1_indexes[1]]
                    )
                    if combined_to_peak_dot_product < min_dot_product_sum:
                        best_combination_pair = combination_indexes

                assert best_combination_pair is not None
                strand_1_start, strand_1_end = (
                    points[best_combination_pair[0][0]],
                    points[best_combination_pair[0][1]],
                )
                strand_2_start, strand_2_end = (
                    points[best_combination_pair[1][0]],
                    points[best_combination_pair[1][1]],
                )

                # trace a more apt line through the node for the skeleton
                trace_skeleton_better_through_node(
                    node_coords=strand_1_start,
                    peak_coords=max_node_pixel_coords,
                    image=grain_image,
                    skeleton=proposed_result_corrected_skeleton,
                )
                trace_skeleton_better_through_node(
                    node_coords=strand_1_end,
                    peak_coords=max_node_pixel_coords,
                    image=grain_image,
                    skeleton=proposed_result_corrected_skeleton,
                )
                trace_skeleton_better_through_node(
                    node_coords=strand_2_start,
                    peak_coords=max_node_pixel_coords,
                    image=grain_image,
                    skeleton=proposed_result_corrected_skeleton,
                )
                trace_skeleton_better_through_node(
                    node_coords=strand_2_end,
                    peak_coords=max_node_pixel_coords,
                    image=grain_image,
                    skeleton=proposed_result_corrected_skeleton,
                )

                # Sanity check for the node before applying the correction
                # if the proposed correction now has disconnected components, skip this correction
                if np.max(label(proposed_result_corrected_skeleton)) != 1:
                    LOGGER.info(
                        f"True crossing improvement :  grain [{grain_index}] node [{node_index}]: Proposed"
                        f"improvement results in disconnected components. Skipping."
                    )
                    continue

                result_corrected_skeleton = proposed_result_corrected_skeleton

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
                proposed_result_corrected_skeleton[path_a[:, 0], path_a[:, 1]] = False
                point_b, path_b = get_point_along_branch(
                    mask_branches, branch_starts[1], branch_explore_distance_nm / pixel_to_nm_scaling
                )
                proposed_result_corrected_skeleton[path_b[:, 0], path_b[:, 1]] = False
                point_c, path_c = get_point_along_branch(
                    mask_branches, branch_starts[2], branch_explore_distance_nm / pixel_to_nm_scaling
                )
                proposed_result_corrected_skeleton[path_c[:, 0], path_c[:, 1]] = False
                point_d, path_d = get_point_along_branch(
                    mask_branches, branch_starts[3], branch_explore_distance_nm / pixel_to_nm_scaling
                )
                proposed_result_corrected_skeleton[path_d[:, 0], path_d[:, 1]] = False

                # Also remove the node coords from the result skeleton
                proposed_result_corrected_skeleton[mask_node == 1] = False

                possible_combinations_indexes = [
                    ((point_a, point_b), (point_c, point_d)),
                    ((point_a, point_c), (point_b, point_d)),
                    ((point_a, point_d), (point_b, point_c)),
                ]
                best_combination: (
                    tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]] | None
                ) = None
                highest_lowest_height_along_paths = -np.inf
                for combination_indexes in possible_combinations_indexes:
                    pair_1, pair_2 = combination_indexes
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
                        proposed_result_corrected_skeleton[coord[0], coord[1]] = 1

                # Sanity check for the node before applying the correction
                # if the proposed correction now has disconnected components, skip this correction
                if np.max(label(proposed_result_corrected_skeleton)) != 1:
                    LOGGER.info(
                        f"Close strand correction :  grain [{grain_index}] node [{node_index}]: Proposed"
                        f"correction results in disconnected components. Skipping."
                    )
                    continue

                result_corrected_skeleton = proposed_result_corrected_skeleton

        graincrop.skeleton_override = result_corrected_skeleton
        graincrop.overridden_skeleton = original_pruned_skeleton
