"""Generates disordered traces (pruned skeletons) and metrics."""

from __future__ import annotations

import logging
import warnings

import numpy as np
import numpy.typing as npt
import skimage.measure as skimage_measure
from scipy import ndimage
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import label

from topostats.logs.logs import LOGGER_NAME
from topostats.tracing.pruning import prune_skeleton
from topostats.tracing.skeletonize import getSkeleton

LOGGER = logging.getLogger(LOGGER_NAME)


class disorderedTrace:
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
    convert_nm_to_m : bool
        Convert nanometers to metres.
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

    def __init__(
        self,
        image: npt.NDArray,
        mask: npt.NDArray,
        filename: str,
        pixel_to_nm_scaling: float,
        convert_nm_to_m: bool = True,
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
        convert_nm_to_m : bool
            Convert nanometers to metres.
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
        self.image = image * 1e-9 if convert_nm_to_m else image
        self.mask = mask
        self.filename = filename
        self.pixel_to_nm_scaling = pixel_to_nm_scaling * 1e-9 if convert_nm_to_m else pixel_to_nm_scaling
        self.min_skeleton_size = min_skeleton_size
        self.mask_smoothing_params = (
            mask_smoothing_params
            if mask_smoothing_params is not None
            else {"gaussian_sigma": None, "dilation_iterations": 2}
        )
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
        # LOGGER.info(f"[{self.filename}] : mask_smooth_params : {self.mask_smoothing_params=}")
        self.smoothed_mask = self.smooth_mask(self.mask, **self.mask_smoothing_params)
        self.skeleton = getSkeleton(
            self.smoothed_mask,
            self.mask,
            method=self.skeletonisation_params["method"],
            height_bias=self.skeletonisation_params["height_bias"],
        ).get_skeleton()
        self.pruned_skeleton = prune_skeleton(self.smoothed_mask, self.skeleton, **self.pruning_params.copy())
        self.pruned_skeleton = self.remove_touching_edge(self.pruned_skeleton)
        self.disordered_trace = np.argwhere(self.pruned_skeleton == 1)

        if self.disordered_trace is None:
            LOGGER.info(f"[{self.filename}] : Grain failed to Skeletonise")
        elif len(self.disordered_trace) < self.min_skeleton_size:
            self.disordered_trace = None

    def re_add_holes(
        self, orig_mask: npt.NDArray, smoothed_mask: npt.NDArray, holearea_min_max: list = (4, np.inf)
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
            Original mask but with inner and outer edged smoothed. The smoothing operation may have closed up important holes in the mask.
        holearea_min_max : list
            List of minimum and maximum hole area (in nanometers).

        Returns
        -------
        npt.NDArray
            Smoothed mask with holes restored.
        """
        holesize_min_px = holearea_min_max[0] / ((self.pixel_to_nm_scaling / 1e-9) ** 2)
        holesize_max_px = holearea_min_max[1] / ((self.pixel_to_nm_scaling / 1e-9) ** 2)
        holes = 1 - orig_mask
        holes = label(holes)
        sizes = [holes[holes == i].size for i in range(1, holes.max() + 1)]
        holes[holes == 1] = 0  # set background to 0

        for i, size in enumerate(sizes):
            if size < holesize_min_px or size > holesize_max_px:  # small holes may be fake are left out
                holes[holes == i + 1] = 0
        holes[holes != 0] = 1

        # compare num holes in each mask
        holey_smooth = smoothed_mask.copy()
        holey_smooth[holes == 1] = 0

        return holey_smooth

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
        self, grain: npt.NDArray, dilation_iterations: int = 2, gaussian_sigma: float | int | None = None
    ) -> npt.NDArray:
        """
        Smooth grains based on the lower number of binary pixels added from dilation or gaussian.

        This method ensures gaussian smoothing isn't too aggressive and covers / creates gaps in the mask.

        Parameters
        ----------
        grain : npt.NDArray
            Numpy array of the grain mask.
        dilation_iterations : int
            Number of times to dilate the grain to smooth it. Default is 2.
        gaussian_sigma : float | None
            Gaussian sigma value to smooth the grains after an Otsu threshold. If None, defaults to
            max(grain.shape) / 256.

        Returns
        -------
        npt.NDArray
            Numpy array of smmoothed image.
        """
        gaussian_sigma = max(grain.shape) / 256 if gaussian_sigma is None else gaussian_sigma
        dilation = ndimage.binary_dilation(grain, iterations=dilation_iterations).astype(np.int32)
        gauss = gaussian(grain, sigma=gaussian_sigma)
        gauss[gauss > threshold_otsu(gauss) * 1.3] = 1
        gauss[gauss != 1] = 0
        gauss = gauss.astype(np.int32)
        # Add hole to the smooth mask conditional on smallest pixel difference for dilation or the Gaussian smoothing.
        if dilation.sum() > gauss.sum():
            return self.re_add_holes(grain, gauss)
        return self.re_add_holes(grain, dilation)


def trace_image_disordered(
    image: npt.NDArray,
    grains_mask: npt.NDArray,
    filename: str,
    pixel_to_nm_scaling: float,
    min_skeleton_size: int,
    mask_smoothing_params: dict,
    skeletonisation_params: dict,
    pruning_params: dict,
    pad_width: int = 1,
    cores: int = 1,
) -> dict:
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
    cores : int
        Number of cores to process with.

    Returns
    -------
    tuple[dict, dict]
        Binary and interger labeled cropped and full-image masks from skeletonising and pruning the grains in the image.
    """
    # Check both arrays are the same shape - should this be a test instead, why should this ever occur?
    if image.shape != grains_mask.shape:
        raise ValueError(f"Image shape ({image.shape}) and Mask shape ({grains_mask.shape}) should match.")

    cropped_images, cropped_masks, bboxs = prep_arrays(image, grains_mask, pad_width)
    n_grains = len(cropped_images)
    img_base = np.zeros_like(image)
    disordered_trace_crop_data = {}

    # want to get each cropped image, use some anchor coords to match them onto the image,
    #   and compile all the grain images onto a single image
    all_images = {
        "smoothed_grain": img_base.copy(),
        "skeleton": img_base.copy(),
        "pruned_skeleton": img_base.copy(),
    }

    LOGGER.info(f"[{filename}] : Calculating DNA tracing statistics for {n_grains} grains.")

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
            LOGGER.info(f"[{filename}] : Disordered Traced grain {cropped_image_index + 1} of {n_grains}")

            # remap the cropped images back onto the original
            for image_name, full_image in all_images.items():
                crop = disordered_trace_images[image_name]
                bbox = bboxs[cropped_image_index]
                full_image[bbox[0] : bbox[2], bbox[1] : bbox[3]] += crop[pad_width:-pad_width, pad_width:-pad_width]
            disordered_trace_crop_data[f"grain_{cropped_image_index}"] = disordered_trace_images
            disordered_trace_crop_data[f"grain_{cropped_image_index}"]["bbox"] = bboxs[cropped_image_index]

        except Exception as e:
            LOGGER.warning(f"[{filename}] : Disordered tracing of grain {cropped_image_index} failed with {e}.")

    return disordered_trace_crop_data, all_images


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
    # Flip every labelled region to be 1 instead of its label
    cropped_masks = {index: np.where(grain == 0, 0, 1) for index, grain in cropped_masks.items()}
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


def disordered_trace_grain(
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

    cropped_images = {
        "original_image": cropped_image,
        "original_grain": cropped_mask,
        "smoothed_grain": disorderedtrace.smoothed_mask,
        "skeleton": disorderedtrace.skeleton,
        "pruned_skeleton": disorderedtrace.pruned_skeleton,
    }

    return cropped_images


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