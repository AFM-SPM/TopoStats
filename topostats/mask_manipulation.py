"""Code for manipulating binary masks."""

import logging

import numpy as np
import numpy.typing as npt
from scipy import ndimage
from skimage import filters
from skimage.morphology import label
from skimage.graph import route_through_array

from topostats.logs.logs import LOGGER_NAME
from topostats.utils import convolve_skeleton
from topostats.tracing.skeletonize import getSkeleton
from topostats.measure.geometry import calculate_mask_width_with_skeleton

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
    endpoint_connection_distance_nm: float,
    endpoint_connection_cost_map_height_maximum: float,
    plot: bool = False,
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
    # plt.imshow(smoothed_mask, cmap="gray")
    # plt.title("Smoothed mask")
    # plt.show()

    # Maybe need to check it doesn't touch the edge of the image like we do in disordered_tracing? unsure.

    # Next step, skeletonize
    skeleton = getSkeleton(
        image=image,
        mask=smoothed_mask,
        method=skeletonisation_method,
        height_bias=0.6,
    ).get_skeleton()

    # Calculate the mask width along the skeleton for later
    mean_mask_width_nm = calculate_mask_width_with_skeleton(
        mask=smoothed_mask,
        skeleton=skeleton,
        pixel_to_nm_scaling=p2nm,
    )
    mean_mask_width_px = mean_mask_width_nm / p2nm

    # fig, ax = plt.subplots(figsize=(10, 10))
    # plt.imshow(skeleton, cmap="gray")
    # plt.title("skeleton")
    # plt.show()

    # Now to find the skeleton endpoints and connect close ones.
    convolved_skeleton = convolve_skeleton(skeleton=skeleton)
    # Get the endpoints, value = 2
    endpoint_coords: list[tuple[int, int]] = [tuple(coord) for coord in np.argwhere(convolved_skeleton == 2)]
    print("endpoints:")
    print(endpoint_coords)

    if plot:
        _fig, _ax = plt.subplots(figsize=(20, 20))
        plt.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        grain_mask_mask = np.ma.masked_where(~mask, mask)
        plt.imshow(grain_mask_mask, cmap="Blues_r", alpha=0.3)
        skeleton_mask = np.ma.masked_where(~convolved_skeleton.astype(bool), convolved_skeleton)
        plt.imshow(skeleton_mask, cmap="viridis", alpha=0.7)
        plt.show()

    # For each endpoint, determine if any others are close enough to connect.
    potential_pairs: list[tuple[tuple[int, int], tuple[int, int], float]] = []
    for i, endpoint_1 in enumerate(endpoint_coords):
        for j, endpoint_2 in enumerate(endpoint_coords):
            if i >= j:
                continue  # avoid double counting
            distance_nm = float(np.linalg.norm((np.array(endpoint_1) - np.array(endpoint_2)) * p2nm))
            if distance_nm <= endpoint_connection_distance_nm:
                potential_pairs.append((endpoint_1, endpoint_2, distance_nm))

    print("potential pairs to connect:")
    for pair in potential_pairs:
        print(pair)

    # for now, for simplicity, let's delete any pairs that are involved in other pairs.
    # Construct a list of all endpoints involved in pairs
    potential_pairs_no_repeated_endpoints = keep_only_nonrepeated_endpoints(potential_pairs)
    print("pairs to connect:")
    for pair in potential_pairs_no_repeated_endpoints:
        print(pair)

    for pair in potential_pairs_no_repeated_endpoints:
        endpoint_1, endpoint_2, distance_nm = pair

        # Connect the pair.

        # try using height biased pathfinding between the two endpoints?
        # create a weight cost map from the image, where 0 is the maximum cost, and the lowest cost is configurable.
        # first create a crop around the two endpoints to speed up pathfinding
        cost_map_bbox_padding_px = 10
        min_y = max(0, min(endpoint_1[0], endpoint_2[0]) - cost_map_bbox_padding_px)
        max_y = min(image.shape[0], max(endpoint_1[0], endpoint_2[0]) + cost_map_bbox_padding_px)
        min_x = max(0, min(endpoint_1[1], endpoint_2[1]) - cost_map_bbox_padding_px)
        max_x = min(image.shape[1], max(endpoint_1[1], endpoint_2[1]) + cost_map_bbox_padding_px)
        cost_map = image[min_y:max_y, min_x:max_x]
        _mask_crop = mask[min_y:max_y, min_x:max_x]
        _image_crop = image[min_y:max_y, min_x:max_x]
        local_endpoint_1 = (endpoint_1[0] - min_y, endpoint_1[1] - min_x)
        local_endpoint_2 = (endpoint_2[0] - min_y, endpoint_2[1] - min_x)
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
        path, _cost = route_through_array(
            cost_map,
            start=local_endpoint_1,
            end=local_endpoint_2,
            fully_connected=True,  # allow diagonal moves
        )

        # Dilate the path to equal the mean mask width in pixels
        path_array = np.zeros_like(cost_map, dtype=bool)
        # Set the path to True
        for y, x in path:
            path_array[y, x] = True
        # Calculate the dilation iterations needed to reach the mean mask width
        dilation_radius = int(np.ceil(mean_mask_width_px / 2))
        dilated_path_array = ndimage.binary_dilation(
            path_array,
            iterations=dilation_radius,
        )

        # Add the dilated path to the whole mask
        dilated_path_array_global = np.zeros_like(mask, dtype=bool)
        dilated_path_array_global[min_y:max_y, min_x:max_x] = dilated_path_array
        mask = mask | dilated_path_array_global

        for y, x in path:
            skeleton[y + min_y, x + min_x] = True

    if plot:
        _fig, _ax = plt.subplots(figsize=(20, 20))
        plt.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        grain_mask_mask = np.ma.masked_where(~mask, mask)
        plt.imshow(grain_mask_mask, cmap="Blues_r", alpha=0.3)
        skeleton_mask = np.ma.masked_where(~skeleton.astype(bool), skeleton)
        plt.imshow(skeleton_mask, cmap="viridis", alpha=0.7)
        plt.title("skeleton after connecting loose ends")
        plt.show()

    return mask


def multi_class_skeletonise_and_join_close_ends(
    class_indices: int,
    tensor: npt.NDArray,
    filename: str,
    p2nm: float,
    image: npt.NDArray,
    skeletonisation_holearea_min_max: tuple[int | None, int | None],
    skeletonisation_mask_smoothing_dilation_iterations: int,
    skeletonisation_mask_smoothing_gaussian_sigma: float,
    skeletonisation_method: str,
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
        mask = tensor[class_index, :, :]
        updated_mask = skeletonise_and_join_close_ends(
            filename=filename,
            p2nm=p2nm,
            image=image,
            mask=mask,
            skeletonisation_holearea_min_max=skeletonisation_holearea_min_max,
            skeletonisation_mask_smoothing_dilation_iterations=skeletonisation_mask_smoothing_dilation_iterations,
            skeletonisation_mask_smoothing_gaussian_sigma=skeletonisation_mask_smoothing_gaussian_sigma,
            skeletonisation_method=skeletonisation_method,
            endpoint_connection_distance_nm=endpoint_connection_distance_nm,
            endpoint_connection_cost_map_height_maximum=endpoint_connection_cost_map_height_maximum,
        )
        result_tensor[class_index, :, :] = updated_mask
    return result_tensor
