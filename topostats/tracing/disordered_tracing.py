"""Generates disordered traces (pruned skeletons) and metrics."""

from __future__ import annotations

import logging
import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd
import skan
import skimage.measure as skimage_measure
from scipy import ndimage
from skimage import filters
from skimage.morphology import label

from topostats.logs.logs import LOGGER_NAME
from topostats.tracing.pruning import prune_skeleton
from topostats.tracing.skeletonize import getSkeleton
from topostats.utils import convolve_skeleton

LOGGER = logging.getLogger(LOGGER_NAME)


class disorderedTrace:  # pylint: disable=too-many-instance-attributes
    """
    Calculate disordered traces for a DNA molecule and calculates statistics from those traces.

    Parameters
    ----------
    image : npt.NDArray
        Cropped image, typically padded beyond the bounding box.
    mask : npt.NDArray
        Labelled mask for the grain, typically padded beyond the bounding box.
    filename : str
        Filename being processed.
    pixel_to_nm_scaling : float
        Pixel to nm scaling.
    min_skeleton_size : int
        Minimum skeleton size below which tracing statistics are not calculated.
    mask_smoothing_params : dict
        Dictionary of parameters to smooth the grain mask for better quality skeletonisation results. Contains
        a gaussian 'sigma' and number of dilation iterations.
    skeletonisation_params : dict
        Skeletonisation Parameters. Method of skeletonisation to use 'topostats' is the original TopoStats
        method. Three methods from scikit-image are available 'zhang', 'lee' and 'thin'.
    pruning_params : dict
        Dictionary of pruning parameters. Contains 'method', 'max_length', 'height_threshold', 'method_values' and
        'method_outlier'.
    n_grain : int
        Grain number being processed (only  used in logging).
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        image: npt.NDArray,
        mask: npt.NDArray,
        filename: str,
        pixel_to_nm_scaling: float,
        min_skeleton_size: int = 10,
        mask_smoothing_params: dict | None = None,
        skeletonisation_params: dict | None = None,
        pruning_params: dict | None = None,
        n_grain: int = None,
    ):
        """
        Calculate disordered traces for a DNA molecule and calculates statistics from those traces.

        Parameters
        ----------
        image : npt.NDArray
            Cropped image, typically padded beyond the bounding box.
        mask : npt.NDArray
            Labelled mask for the grain, typically padded beyond the bounding box.
        filename : str
            Filename being processed.
        pixel_to_nm_scaling : float
            Pixel to nm scaling.
        min_skeleton_size : int
            Minimum skeleton size below which tracing statistics are not calculated.
        mask_smoothing_params : dict
            Dictionary of parameters to smooth the grain mask for better quality skeletonisation results. Contains
            a gaussian 'sigma' and number of dilation iterations.
        skeletonisation_params : dict
            Skeletonisation Parameters. Method of skeletonisation to use 'topostats' is the original TopoStats
            method. Three methods from scikit-image are available 'zhang', 'lee' and 'thin'.
        pruning_params : dict
            Dictionary of pruning parameters. Contains 'method', 'max_length', 'height_threshold', 'method_values' and
            'method_outlier'.
        n_grain : int
            Grain number being processed (only  used in logging).
        """
        self.image = image
        self.mask = mask
        self.filename = filename
        self.pixel_to_nm_scaling = pixel_to_nm_scaling
        self.min_skeleton_size = min_skeleton_size
        self.mask_smoothing_params = mask_smoothing_params
        self.skeletonisation_params = (
            skeletonisation_params if skeletonisation_params is not None else {"method": "zhang"}
        )
        self.pruning_params = pruning_params if pruning_params is not None else {"method": "topostats"}
        self.n_grain = n_grain
        # Images
        self.smoothed_mask = np.zeros_like(image)
        self.skeleton = np.zeros_like(image)
        self.pruned_skeleton = np.zeros_like(image)
        # Trace
        self.disordered_trace = None

        # suppresses scipy splining warnings
        warnings.filterwarnings("ignore")

        LOGGER.debug(f"[{self.filename}] Performing Disordered Tracing")

    def trace_dna(self):
        """Perform the DNA skeletonisation and cleaning pipeline."""
        self.smoothed_mask = self.smooth_mask(self.mask, **self.mask_smoothing_params)
        self.skeleton = getSkeleton(
            self.image,
            self.smoothed_mask,
            method=self.skeletonisation_params["method"],
            height_bias=self.skeletonisation_params["height_bias"],
        ).get_skeleton()
        self.pruned_skeleton = prune_skeleton(
            self.image, self.skeleton, self.pixel_to_nm_scaling, **self.pruning_params.copy()
        )
        self.pruned_skeleton = self.remove_touching_edge(self.pruned_skeleton)
        self.disordered_trace = np.argwhere(self.pruned_skeleton == 1)

        if self.disordered_trace is None:
            LOGGER.warning(f"[{self.filename}] : Grain {self.n_grain} failed to Skeletonise.")
            self.disordered_trace = None
        elif len(self.disordered_trace) < self.min_skeleton_size:
            LOGGER.warning(f"[{self.filename}] : Grain {self.n_grain} skeleton < {self.min_skeleton_size}, skipping.")
            self.disordered_trace = None

    def re_add_holes(
        self,
        orig_mask: npt.NDArray,
        smoothed_mask: npt.NDArray,
        holearea_min_max: tuple[float | int | None] = (2, None),
    ) -> npt.NDArray:
        """
        Restore holes in masks that were occluded by dilation.

        As Gaussian dilation smoothing methods can close holes in the original mask, this function obtains those holes
        (based on the general background being the first due to padding) and adds them back into the smoothed mask. When
        paired with ``smooth_mask``, this essentially just smooths the outer edge of the mask.

        Parameters
        ----------
        orig_mask : npt.NDArray
            Original mask.
        smoothed_mask : npt.NDArray
            Original mask but with inner and outer edged smoothed. The smoothing operation may have closed up important
            holes in the mask.
        holearea_min_max : tuple[float | int | None]
            Tuple of minimum and maximum hole area (in nanometers) to replace from the original mask into the smoothed
            mask.

        Returns
        -------
        npt.NDArray
            Smoothed mask with holes restored.
        """
        # handle none's
        if set(holearea_min_max) == {None}:
            return smoothed_mask
        if None in holearea_min_max:
            none_index = holearea_min_max.index(None)
            holearea_min_max[none_index] = 0 if none_index == 0 else np.inf

        # obtain px holesizes
        holesize_min_px = holearea_min_max[0] / ((self.pixel_to_nm_scaling) ** 2)
        holesize_max_px = holearea_min_max[1] / ((self.pixel_to_nm_scaling) ** 2)

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

    @staticmethod
    def remove_touching_edge(skeleton: npt.NDArray) -> npt.NDArray:
        """
        Remove any skeleton points touching the border (to prevent errors later).

        Parameters
        ----------
        skeleton : npt.NDArray
            A binary array where touching clusters of 1's become 0's if touching the edge of the array.

        Returns
        -------
        npt.NDArray
            Skeleton without points touching the border.
        """
        for edge in [skeleton[0, :-1], skeleton[:-1, -1], skeleton[-1, 1:], skeleton[1:, 0]]:
            uniques = np.unique(edge)
            for i in uniques:
                skeleton[skeleton == i] = 0
        return skeleton

    def smooth_mask(
        self,
        grain: npt.NDArray,
        dilation_iterations: int = 2,
        gaussian_sigma: float | int = 2,
        holearea_min_max: tuple[int | float | None] = (0, None),
    ) -> npt.NDArray:
        """
        Smooth a grain mask based on the lower number of binary pixels added from dilation or gaussian.

        This method ensures gaussian smoothing isn't too aggressive and covers / creates gaps in the mask.

        Parameters
        ----------
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
            Numpy array of smmoothed image.
        """
        dilation = ndimage.binary_dilation(grain, iterations=dilation_iterations).astype(np.int32)
        gauss = filters.gaussian(grain, sigma=gaussian_sigma)
        gauss = np.where(gauss > filters.threshold_otsu(gauss) * 1.3, 1, 0)
        gauss = gauss.astype(np.int32)
        # Add hole to the smooth mask conditional on smallest pixel difference for dilation or the Gaussian smoothing.
        if dilation.sum() > gauss.sum():
            LOGGER.debug(f"[{self.filename}] : smoothing done by gaussian {gaussian_sigma}")
            return self.re_add_holes(grain, gauss, holearea_min_max)
        LOGGER.debug(f"[{self.filename}] : smoothing done by dilation {dilation_iterations}")
        return self.re_add_holes(grain, dilation, holearea_min_max)


def trace_image_disordered(  # pylint: disable=too-many-arguments,too-many-locals
    image: npt.NDArray,
    grains_mask: npt.NDArray,
    filename: str,
    pixel_to_nm_scaling: float,
    min_skeleton_size: int,
    mask_smoothing_params: dict,
    skeletonisation_params: dict,
    pruning_params: dict,
    pad_width: int = 1,
) -> tuple[dict, pd.DataFrame, dict, pd.DataFrame]:
    """
    Processor function for tracing image.

    Parameters
    ----------
    image : npt.NDArray
        Full image as Numpy Array.
    grains_mask : npt.NDArray
        Full image as Grains that are labelled.
    filename : str
        File being processed.
    pixel_to_nm_scaling : float
        Pixel to nm scaling.
    min_skeleton_size : int
        Minimum size of grain in pixels after skeletonisation.
    mask_smoothing_params : dict
        Dictionary of parameters to smooth the grain mask for better quality skeletonisation results. Contains
        a gaussian 'sigma' and number of dilation iterations.
    skeletonisation_params : dict
        Dictionary of options for skeletonisation, options are 'zhang' (scikit-image) / 'lee' (scikit-image) / 'thin'
        (scikitimage) or 'topostats' (original TopoStats method).
    pruning_params : dict
        Dictionary of options for pruning.
    pad_width : int
        Padding to the cropped image mask.

    Returns
    -------
    tuple[dict, pd.DataFrame, dict, pd.DataFrame]
        Binary and integer labeled cropped and full-image masks from skeletonising and pruning the grains in the image.
    """
    # Check both arrays are the same shape - should this be a test instead, why should this ever occur?
    if image.shape != grains_mask.shape:
        raise ValueError(f"Image shape ({image.shape}) and Mask shape ({grains_mask.shape}) should match.")

    cropped_images, cropped_masks, bboxs = prep_arrays(image, grains_mask, pad_width)
    n_grains = len(cropped_images)
    img_base = np.zeros_like(image)
    disordered_trace_crop_data = {}
    grainstats_additions = {}
    disordered_tracing_stats = pd.DataFrame()

    # want to get each cropped image, use some anchor coords to match them onto the image,
    #   and compile all the grain images onto a single image
    all_images = {
        "smoothed_grain": img_base.copy(),
        "skeleton": img_base.copy(),
        "pruned_skeleton": img_base.copy(),
        "branch_indexes": img_base.copy(),
        "branch_types": img_base.copy(),
    }

    LOGGER.info(f"[{filename}] : Calculating Disordered Tracing statistics for {n_grains} grains...")

    for cropped_image_index, cropped_image in cropped_images.items():
        try:
            cropped_mask = cropped_masks[cropped_image_index]
            disordered_trace_images = disordered_trace_grain(
                cropped_image=cropped_image,
                cropped_mask=cropped_mask,
                pixel_to_nm_scaling=pixel_to_nm_scaling,
                mask_smoothing_params=mask_smoothing_params,
                skeletonisation_params=skeletonisation_params,
                pruning_params=pruning_params,
                filename=filename,
                min_skeleton_size=min_skeleton_size,
                n_grain=cropped_image_index,
            )
            LOGGER.debug(f"[{filename}] : Disordered Traced grain {cropped_image_index + 1} of {n_grains}")

            if disordered_trace_images is not None:
                # obtain segment stats
                try:
                    skan_skeleton = skan.Skeleton(
                        np.where(disordered_trace_images["pruned_skeleton"] == 1, cropped_image, 0),
                        spacing=pixel_to_nm_scaling,
                    )
                    skan_df = skan.summarize(skan_skeleton)
                    skan_df = compile_skan_stats(skan_df, skan_skeleton, cropped_image, filename, cropped_image_index)
                    total_branch_length = skan_df["branch_distance"].sum() * 1e-9
                except ValueError:
                    LOGGER.warning(
                        f"[{filename}] : Skeleton for grain {cropped_image_index} has been pruned out of existence."
                    )
                    total_branch_length = 0
                    skan_df = pd.DataFrame()

                disordered_tracing_stats = pd.concat((disordered_tracing_stats, skan_df))

                # obtain stats
                conv_pruned_skeleton = convolve_skeleton(disordered_trace_images["pruned_skeleton"])
                grainstats_additions[cropped_image_index] = {
                    "image": filename,
                    "grain_number": cropped_image_index,
                    "grain_endpoints": np.int64((conv_pruned_skeleton == 2).sum()),
                    "grain_junctions": np.int64((conv_pruned_skeleton == 3).sum()),
                    "total_branch_lengths": total_branch_length,
                }

                # remap the cropped images back onto the original
                for image_name, full_image in all_images.items():
                    crop = disordered_trace_images[image_name]
                    bbox = bboxs[cropped_image_index]
                    full_image[bbox[0] : bbox[2], bbox[1] : bbox[3]] += crop[pad_width:-pad_width, pad_width:-pad_width]
                disordered_trace_crop_data[f"grain_{cropped_image_index}"] = disordered_trace_images
                disordered_trace_crop_data[f"grain_{cropped_image_index}"]["bbox"] = bboxs[cropped_image_index]

        # when skel too small, pruned to 0's, skan -> ValueError -> skipped
        except Exception as e:  # pylint: disable=broad-exception-caught
            LOGGER.error(  # pylint: disable=logging-not-lazy
                f"[{filename}] : Disordered tracing of grain "
                f"{cropped_image_index} failed. Consider raising an issue on GitHub. Error: ",
                exc_info=e,
            )

        # convert stats dict to dataframe
        grainstats_additions_df = pd.DataFrame.from_dict(grainstats_additions, orient="index")

    return disordered_trace_crop_data, grainstats_additions_df, all_images, disordered_tracing_stats


def compile_skan_stats(
    skan_df: pd.DataFrame, skan_skeleton: skan.Skeleton, image: npt.NDArray, filename: str, grain_number: int
) -> pd.DataFrame:
    """
    Obtain and add more stats to the resultant Skan dataframe.

    Parameters
    ----------
    skan_df : pd.DataFrame
        The statistics DataFrame produced by Skan's `summarize` function.
    skan_skeleton : skan.Skeleton
        The graphical representation of the skeleton produced by Skan.
    image : npt.NDArray
        The image the skeleton was produced from.
    filename : str
        Name of the file being processed.
    grain_number : int
        The number of the grain being processed.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the filename, grain_number, branch-distance, branch-type, connected_segments,
        mean-pixel-value, stdev-pixel-value, min-value, median-value, and mid-value.
    """
    skan_df["image"] = filename
    skan_df["branch-type"] = np.int64(skan_df["branch-type"])
    skan_df["grain_number"] = grain_number
    skan_df["connected_segments"] = skan_df.apply(find_connections, axis=1, skan_df=skan_df)
    skan_df["min_value"] = skan_df.apply(lambda x: segment_heights(x, skan_skeleton, image).min(), axis=1)
    skan_df["median_value"] = skan_df.apply(lambda x: np.median(segment_heights(x, skan_skeleton, image)), axis=1)
    skan_df["middle_value"] = skan_df.apply(segment_middles, skan_skeleton=skan_skeleton, image=image, axis=1)

    # Skan-0.11.1 uses dashes in variables names which prevents dot-notation being used to refer to variable names
    # this has been addressed (see https://github.com/jni/skan/pull/215) for now manually convert.
    skan_df.columns = skan_df.columns.str.replace("-", "_")

    # remove unused skan columns
    return skan_df[
        [
            "image",
            "grain_number",
            "branch_distance",
            "branch_type",
            "connected_segments",
            "mean_pixel_value",
            "stdev_pixel_value",
            "min_value",
            "median_value",
            "middle_value",
        ]
    ]


def segment_heights(row: pd.Series, skan_skeleton: skan.Skeleton, image: npt.NDArray) -> npt.NDArray:
    """
    Obtain an ordered list of heights from the skan defined skeleton segment.

    Parameters
    ----------
    row : pd.Series
        A row from the Skan summarize dataframe.
    skan_skeleton : skan.Skeleton
        The graphical representation of the skeleton produced by Skan.
    image : npt.NDArray
        The image the skeleton was produced from.

    Returns
    -------
    npt.NDArray
        Heights along the segment, naturally ordered by Skan.
    """
    coords = skan_skeleton.path_coordinates(row.name)
    return image[coords[:, 0], coords[:, 1]]


def segment_middles(row: pd.Series, skan_skeleton: skan.csr.Skeleton, image: npt.NDArray) -> float:
    """
    Obtain the pixel value in the middle of the ordered segment.

    Parameters
    ----------
    row : pd.Series
        A row from the Skan summarize dataframe.
    skan_skeleton : skan.csr.Skeleton
        The graphical representation of the skeleton produced by Skan.
    image : npt.NDArray
        The image the skeleton was produced from.

    Returns
    -------
    float
        The single or mean pixel value corresponding to the middle coordinate(s) of the segment.
    """
    heights = segment_heights(row, skan_skeleton, image)
    middle_idx, middle_remainder = (len(heights) + 1) // 2 - 1, (len(heights) + 1) % 2
    return heights[[middle_idx, middle_idx + middle_remainder]].mean()


def find_connections(row: pd.Series, skan_df: pd.DataFrame) -> str:
    """
    Compile the neighbouring branch indexes of the row.

    Parameters
    ----------
    row : pd.Series
        A row from the Skan summarize dataframe.
    skan_df : pd.DataFrame
        The statistics DataFrame produced by Skan's `summarize` function.

    Returns
    -------
    str
        A string representation of a list of matching row indices where the node src and dst
        columns match that of the rows.
        String is needed for csv compatibility since csvs can't hold lists.
    """
    connections = skan_df[
        (skan_df["node-id-src"] == row["node-id-src"])
        | (skan_df["node-id-dst"] == row["node-id-dst"])
        | (skan_df["node-id-src"] == row["node-id-dst"])
        | (skan_df["node-id-dst"] == row["node-id-src"])
    ].index.tolist()

    # Remove the index of the current row itself from the list of connections
    connections.remove(row.name)
    return str(connections)


def prep_arrays(
    image: npt.NDArray, labelled_grains_mask: npt.NDArray, pad_width: int
) -> tuple[dict[int, npt.NDArray], dict[int, npt.NDArray]]:
    """
    Take an image and labelled mask and crops individual grains and original heights to a list.

    A second padding is made after cropping to ensure for "edge cases" where grains are close to bounding box edges that
    they are traced correctly. This is accounted for when aligning traces to the whole image mask.

    Parameters
    ----------
    image : npt.NDArray
        Gaussian filtered image. Typically filtered_image.images["gaussian_filtered"].
    labelled_grains_mask : npt.NDArray
        2D Numpy array of labelled grain masks, with each mask being comprised solely of unique integer (not
        zero). Typically this will be output from 'grains.directions[<direction>["labelled_region_02]'.
    pad_width : int
        Cells by which to pad cropped regions by.

    Returns
    -------
    Tuple
        Returns a tuple of two dictionaries, each consisting of cropped arrays.
    """
    # Get bounding boxes for each grain
    region_properties = skimage_measure.regionprops(labelled_grains_mask)
    # Subset image and grains then zip them up
    cropped_images = {}
    cropped_masks = {}

    # for index, grain in enumerate(region_properties):
    #    cropped_image, cropped_bbox = crop_array(image, grain.bbox, pad_width)

    cropped_images = {index: crop_array(image, grain.bbox, pad_width) for index, grain in enumerate(region_properties)}
    cropped_images = {index: np.pad(grain, pad_width=pad_width) for index, grain in cropped_images.items()}
    cropped_masks = {
        index: crop_array(labelled_grains_mask, grain.bbox, pad_width) for index, grain in enumerate(region_properties)
    }
    cropped_masks = {index: np.pad(grain, pad_width=pad_width) for index, grain in cropped_masks.items()}
    cropped_masks = {index: np.where(grain == (index + 1), 1, 0) for index, grain in cropped_masks.items()}
    # Get BBOX coords to remap crops to images
    bboxs = [pad_bounding_box(image.shape, list(grain.bbox), pad_width=pad_width) for grain in region_properties]

    return (cropped_images, cropped_masks, bboxs)


def grain_anchor(array_shape: tuple, bounding_box: list, pad_width: int) -> list:
    """
    Extract anchor (min_row, min_col) from labelled regions and align individual traces over the original image.

    Parameters
    ----------
    array_shape : tuple
        Shape of original array.
    bounding_box : list
        A list of region properties returned by 'skimage.measure.regionprops()'.
    pad_width : int
        Padding for image.

    Returns
    -------
    list(Tuple)
        A list of tuples of the min_row, min_col of each bounding box.
    """
    bounding_coordinates = pad_bounding_box(array_shape, bounding_box, pad_width)
    return (bounding_coordinates[0], bounding_coordinates[1])


def disordered_trace_grain(  # pylint: disable=too-many-arguments
    cropped_image: npt.NDArray,
    cropped_mask: npt.NDArray,
    pixel_to_nm_scaling: float,
    mask_smoothing_params: dict,
    skeletonisation_params: dict,
    pruning_params: dict,
    filename: str = None,
    min_skeleton_size: int = 10,
    n_grain: int = None,
) -> dict:
    """
    Trace an individual grain.

    Tracing involves multiple steps...

    1. Skeletonisation
    2. Pruning of side branches (artefacts from skeletonisation).
    3. Ordering of the skeleton.

    Parameters
    ----------
    cropped_image : npt.NDArray
        Cropped array from the original image defined as the bounding box from the labelled mask.
    cropped_mask : npt.NDArray
        Cropped array from the labelled image defined as the bounding box from the labelled mask. This should have been
        converted to a binary mask.
    pixel_to_nm_scaling : float
        Pixel to nm scaling.
    mask_smoothing_params : dict
        Dictionary of parameters to smooth the grain mask for better quality skeletonisation results. Contains
        a gaussian 'sigma' and number of dilation iterations.
    skeletonisation_params : dict
        Dictionary of skeletonisation parameters, options are 'zhang' (scikit-image) / 'lee' (scikit-image) / 'thin'
        (scikitimage) or 'topostats' (original TopoStats method).
    pruning_params : dict
        Dictionary of pruning parameters.
    filename : str
        File being processed.
    min_skeleton_size : int
        Minimum size of grain in pixels after skeletonisation.
    n_grain : int
        Grain number being processed.

    Returns
    -------
    dict
        Dictionary of the contour length, whether the image is circular or linear, the end-to-end distance and an array
        of coordinates.
    """
    disorderedtrace = disorderedTrace(
        image=cropped_image,
        mask=cropped_mask,
        filename=filename,
        pixel_to_nm_scaling=pixel_to_nm_scaling,
        min_skeleton_size=min_skeleton_size,
        mask_smoothing_params=mask_smoothing_params,
        skeletonisation_params=skeletonisation_params,
        pruning_params=pruning_params,
        n_grain=n_grain,
    )

    disorderedtrace.trace_dna()

    if disorderedtrace.disordered_trace is None:
        return None

    return {
        "original_image": cropped_image,
        "original_grain": cropped_mask,
        "smoothed_grain": disorderedtrace.smoothed_mask,
        "skeleton": disorderedtrace.skeleton,
        "pruned_skeleton": disorderedtrace.pruned_skeleton,
        "branch_types": get_skan_image(
            cropped_image, disorderedtrace.pruned_skeleton, "branch-type"
        ),  # change with Skan new release
        "branch_indexes": get_skan_image(
            cropped_image, disorderedtrace.pruned_skeleton, "node-id-src"
        ),  # change with Skan new release
    }


def get_skan_image(original_image: npt.NDArray, pruned_skeleton: npt.NDArray, skan_column: str) -> npt.NDArray:
    """
    Label each branch with it's Skan branch type label.

    Branch types (+1 compared to Skan docs) are defined as:
    1 = Endpoint-to-endpoint (isolated branch)
    2 = Junction-to-endpoint
    3 = Junction-to-junction
    4 = Isolated cycle

    Parameters
    ----------
    original_image : npt.NDArray
        Height image from which the pruned skeleton is derived from.
    pruned_skeleton : npt.NDArray
        Single pixel thick skeleton mask.
    skan_column : str
        A column from Skan's summarize function to colour the branch segments with.

    Returns
    -------
    npt.NDArray
        2D array where the background is 0, and skeleton branches label as their Skan branch type.
    """
    branch_field_image = np.zeros_like(original_image)
    skeleton_image = np.where(pruned_skeleton == 1, original_image, 0)

    try:
        skan_skeleton = skan.Skeleton(skeleton_image, spacing=1e-9, value_is_height=True)
        res = skan.summarize(skan_skeleton)
        for i, branch_field in enumerate(res[skan_column]):
            path_coords = skan_skeleton.path_coordinates(i)
            if skan_column == "node-id-src":
                branch_field = i
            branch_field_image[path_coords[:, 0], path_coords[:, 1]] = branch_field + 1
    except ValueError:  # when no skeleton to skan
        LOGGER.warning("Skeleton has been pruned out of existence.")

    return branch_field_image


def crop_array(array: npt.NDArray, bounding_box: tuple, pad_width: int = 0) -> npt.NDArray:
    """
    Crop an array.

    Ideally we pad the array that is being cropped so that we have heights outside of the grains bounding box. However,
    in some cases, if a grain is near the edge of the image scan this results in requesting indexes outside of the
    existing image. In which case we get as much of the image padded as possible.

    Parameters
    ----------
    array : npt.NDArray
        2D Numpy array to be cropped.
    bounding_box : Tuple
        Tuple of coordinates to crop, should be of form (min_row, min_col, max_row, max_col).
    pad_width : int
        Padding to apply to bounding box.

    Returns
    -------
    npt.NDArray()
        Cropped array.
    """
    bounding_box = list(bounding_box)
    bounding_box = pad_bounding_box(array.shape, bounding_box, pad_width)
    return array[
        bounding_box[0] : bounding_box[2],
        bounding_box[1] : bounding_box[3],
    ]


def pad_bounding_box(array_shape: tuple, bounding_box: list, pad_width: int) -> list:
    """
    Pad coordinates, if they extend beyond image boundaries stop at boundary.

    Parameters
    ----------
    array_shape : tuple
        Shape of original image.
    bounding_box : list
        List of coordinates 'min_row', 'min_col', 'max_row', 'max_col'.
    pad_width : int
        Cells to pad arrays by.

    Returns
    -------
    list
       List of padded coordinates.
    """
    # Top Row : Make this the first column if too close
    bounding_box[0] = 0 if bounding_box[0] - pad_width < 0 else bounding_box[0] - pad_width
    # Left Column : Make this the first column if too close
    bounding_box[1] = 0 if bounding_box[1] - pad_width < 0 else bounding_box[1] - pad_width
    # Bottom Row : Make this the last row if too close
    bounding_box[2] = array_shape[0] if bounding_box[2] + pad_width > array_shape[0] else bounding_box[2] + pad_width
    # Right Column : Make this the last column if too close
    bounding_box[3] = array_shape[1] if bounding_box[3] + pad_width > array_shape[1] else bounding_box[3] + pad_width
    return bounding_box


# 2023-06-09 - Code that runs dnatracing in parallel across grains, left deliberately for use when we remodularise the
#              entry-points/workflow. Will require that the gaussian filtered array is saved and passed in along with
#              the labelled regions. @ns-rse
#
#
# if __name__ == "__main__":
#     cropped_images, cropped_masks = prep_arrays(image, grains_mask, pad_width)
#     n_grains = len(cropped_images)
#     LOGGER.info(f"[{filename}] : Calculating statistics for {n_grains} grains.")
#     # Process in parallel
#     with Pool(processes=cores) as pool:
#         results = {}
#         with tqdm(total=n_grains) as pbar:
#             x = 0
#             for result in pool.starmap(
#                 trace_grain,
#                 zip(
#                     cropped_images,
#                     cropped_masks,
#                     repeat(pixel_to_nm_scaling),
#                     repeat(filename),
#                     repeat(min_skeleton_size),
#                     repeat(skeletonisation_method),
#                 ),
#             ):
#                 LOGGER.info(f"[{filename}] : Traced grain {x + 1} of {n_grains}")
#                 results[x] = result
#                 x += 1
#                 pbar.update()
#     try:
#         results = pd.DataFrame.from_dict(results, orient="index")
#         results.index.name = "molecule_number"
#     except ValueError as error:
#         LOGGER.error("No grains found in any images, consider adjusting your thresholds.")
#         LOGGER.error(error)
#     return results
