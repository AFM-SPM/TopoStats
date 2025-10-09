"""Functions for processing data."""

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from art import tprint

from topostats import TOPOSTATS_COMMIT, TOPOSTATS_VERSION
from topostats.array_manipulation import re_crop_grain_image_and_mask_to_set_size_nm
from topostats.filters import Filters
from topostats.grains import GrainCrop, GrainCropsDirection, Grains, ImageGrainCrops
from topostats.grainstats import GrainStats
from topostats.io import get_out_path, save_topostats_file
from topostats.logs.logs import LOGGER_NAME
from topostats.measure.curvature import calculate_curvature_stats_image
from topostats.plotting import plot_crossing_linetrace_halfmax
from topostats.plottingfuncs import (
    Images,
    add_pixel_to_nm_to_plotting_config,
)
from topostats.statistics import image_statistics
from topostats.tracing.disordered_tracing import trace_image_disordered
from topostats.tracing.nodestats import nodestats_image
from topostats.tracing.ordered_tracing import ordered_tracing_image
from topostats.tracing.splining import splining_image
from topostats.utils import create_empty_dataframe

# pylint: disable=broad-except
# pylint: disable=line-too-long
# pylint: disable=too-many-arguments
# pylint: disable=too-many-branches
# pylint: disable=too-many-lines
# pylint: disable=too-many-locals
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-statements
# pylint: disable=too-many-nested-blocks
# pylint: disable=unnecessary-dict-index-lookup
# pylint: disable=too-many-lines

LOGGER = logging.getLogger(LOGGER_NAME)


def run_filters(
    unprocessed_image: npt.NDArray,
    pixel_to_nm_scaling: float,
    filename: str,
    filter_out_path: Path,
    core_out_path: Path,
    filter_config: dict,
    plotting_config: dict,
) -> npt.NDArray | None:
    """
    Filter and flatten an image. Optionally plots the results, returning the flattened image.

    Parameters
    ----------
    unprocessed_image : npt.NDArray
        Image to be flattened.
    pixel_to_nm_scaling : float
        Scaling factor for converting pixel length scales to nanometres.
        ie the number of pixels per nanometre.
    filename : str
        File name for the image.
    filter_out_path : Path
        Output directory for step-by-step flattening plots.
    core_out_path : Path
        General output directory for outputs such as the flattened image.
    filter_config : dict
        Dictionary of configuration for the Filters class to use when initialised.
    plotting_config : dict
        Dictionary of configuration for plotting output images.

    Returns
    -------
    npt.NDArray | None
        Either a numpy array of the flattened image, or None if an error occurs or
        flattening is disabled in the configuration.
    """
    if filter_config["run"]:
        filter_config.pop("run")
        LOGGER.debug(f"[{filename}] Image dimensions: {unprocessed_image.shape}")
        LOGGER.info(f"[{filename}] : *** Filtering ***")
        filters = Filters(
            image=unprocessed_image,
            filename=filename,
            pixel_to_nm_scaling=pixel_to_nm_scaling,
            **filter_config,
        )
        filters.filter_image()
        # Optionally plot filter stage
        if plotting_config["run"]:
            plotting_config.pop("run")
            LOGGER.info(f"[{filename}] : Plotting Filtering Images")
            # Generate plots
            for plot_name, array in filters.images.items():
                if plot_name not in ["scan_raw"]:
                    if plot_name == "extracted_channel":
                        array = np.flipud(array.pixels)
                    plotting_config["plot_dict"][plot_name]["output_dir"] = (
                        core_out_path if plotting_config["plot_dict"][plot_name]["core_set"] else filter_out_path
                    )
                    try:
                        Images(array, **plotting_config["plot_dict"][plot_name]).plot_and_save()
                        Images(array, **plotting_config["plot_dict"][plot_name]).plot_histogram_and_save()
                    except AttributeError:
                        LOGGER.info(f"[{filename}] Unable to generate plot : {plot_name}")
            plotting_config["run"] = True
        # Always want the 'z_threshed' plot (aka "Height Thresholded") but in the core_out_path
        plot_name = "z_threshed"
        plotting_config["plot_dict"][plot_name]["output_dir"] = core_out_path
        Images(
            filters.images["gaussian_filtered"],
            filename=filename,
            **plotting_config["plot_dict"][plot_name],
        ).plot_and_save()
        LOGGER.info(f"[{filename}] : Filters stage completed successfully.")
        return filters.images["gaussian_filtered"]
    # Otherwise, return None and warn that initial processing is disabled.
    LOGGER.error(
        "You have not included running the initial filter stage. This is required for all subsequent "
        "stages of processing. Please check your configuration file."
    )
    return None


def run_grains(  # noqa: C901
    image: npt.NDArray,
    pixel_to_nm_scaling: float,
    filename: str,
    grain_out_path: Path,
    core_out_path: Path,
    plotting_config: dict,
    grains_config: dict,
) -> ImageGrainCrops:
    """
    Identify grains (molecules) and optionally plots the results.

    Parameters
    ----------
    image : npt.NDArray
        2d numpy array image to find grains in.
    pixel_to_nm_scaling : float
        Scaling factor for converting pixel length scales to nanometres. I.e. the number of pixels per nanometre.
    filename : str
        Name of file being processed (used in logging).
    grain_out_path : Path
        Output path for step-by-step grain finding plots.
    core_out_path : Path
        General output directory for outputs such as the flattened image with grain masks overlaid.
    plotting_config : dict
        Dictionary of configuration for plotting images.
    grains_config : dict
        Dictionary of configuration for the Grains class to use when initialised.

    Returns
    -------
    dict | None
        Either None in the case of error or grain finding being disabled or a dictionary
        with keys of "above" and or "below" containing binary masks depicting where grains
        have been detected.
    """
    if grains_config["run"]:
        grains_config.pop("run")
        try:
            LOGGER.info(f"[{filename}] : *** Grain Finding ***")
            grains = Grains(
                image=image,
                filename=filename,
                pixel_to_nm_scaling=pixel_to_nm_scaling,
                **grains_config,
            )
            grains.find_grains()
            # Get number of grains found
            num_above = 0 if grains.image_grain_crops.above is None else len(grains.image_grain_crops.above.crops)
            num_below = 0 if grains.image_grain_crops.below is None else len(grains.image_grain_crops.below.crops)
            LOGGER.info(f"[{filename}] : Grains found: {num_above} above, {num_below} below")
            if num_above == 0 and num_below == 0:
                LOGGER.warning(f"[{filename}] : No grains found for either direction.")
        except Exception as e:
            LOGGER.error(
                f"[{filename}] : An error occurred during grain finding, skipping following steps.", exc_info=e
            )
        else:
            # Optionally plot grain finding stage if we have found grains and plotting is required
            if plotting_config["run"]:
                plotting_config.pop("run")
                grain_crop_plot_size_nm = plotting_config["grain_crop_plot_size_nm"]
                LOGGER.info(f"[{filename}] : Plotting Grain Finding Images")
                for direction, image_arrays in grains.mask_images.items():
                    LOGGER.debug(f"[{filename}] : Plotting {direction} Grain Finding Images")
                    grain_out_path_direction = grain_out_path / f"{direction}"
                    # Plot diagnostic full grain images
                    for plot_name, array in image_arrays.items():
                        # Tensor, iterate over each channel
                        filename_base = plotting_config["plot_dict"][plot_name]["filename"]
                        for tensor_class in range(1, array.shape[2]):
                            LOGGER.debug(f"[{filename}] : Plotting {plot_name} image, class {tensor_class}")
                            plotting_config["plot_dict"][plot_name]["output_dir"] = grain_out_path_direction
                            plotting_config["plot_dict"][plot_name]["filename"] = (
                                filename_base + f"_class_{tensor_class}"
                            )
                            Images(
                                data=image,
                                masked_array=array[:, :, tensor_class],
                                **plotting_config["plot_dict"][plot_name],
                            ).plot_and_save()
                    # Plot individual grain masks
                    direction_grain_crops: GrainCropsDirection | None = None
                    if direction == "above":
                        direction_grain_crops = grains.image_grain_crops.above
                    else:
                        direction_grain_crops = grains.image_grain_crops.below
                    if direction_grain_crops is not None:
                        LOGGER.info(f"[{filename}] : Plotting individual grain masks")
                        for grain_number, grain_crop in direction_grain_crops.crops.items():
                            # If the grain_crop_plot_size_nm is -1, just use the grain crop as-is.
                            if grain_crop_plot_size_nm == -1:
                                crop_image = grain_crop.image
                                crop_mask = grain_crop.mask
                            else:
                                try:
                                    # Resize the grain crop to the requested size
                                    crop_image, crop_mask = re_crop_grain_image_and_mask_to_set_size_nm(
                                        filename=filename,
                                        grain_number=grain_number,
                                        grain_bbox=grain_crop.bbox,
                                        pixel_to_nm_scaling=pixel_to_nm_scaling,
                                        full_image=image,
                                        full_mask_tensor=direction_grain_crops.full_mask_tensor,
                                        target_size_nm=grain_crop_plot_size_nm,
                                    )
                                except ValueError as e:
                                    if "crop cannot be re-cropped" in str(e):
                                        LOGGER.error(
                                            "Crop cannot be re-cropped to requested size, skipping plotting "
                                            "this grain.",
                                            exc_info=True,
                                        )
                                        continue

                            # Plot the grain crop without mask
                            plotting_config["plot_dict"]["grain_image"]["filename"] = f"{filename}_grain_{grain_number}"
                            plotting_config["plot_dict"]["grain_image"]["output_dir"] = grain_out_path_direction
                            Images(
                                data=crop_image,
                                **plotting_config["plot_dict"]["grain_image"],
                            ).plot_and_save()
                            # Plot the grain crop with mask
                            plotting_config["plot_dict"]["grain_mask"]["output_dir"] = grain_out_path_direction
                            # Tensor, iterate over channels
                            for tensor_class in range(1, crop_mask.shape[2]):
                                plotting_config["plot_dict"]["grain_mask"][
                                    "filename"
                                ] = f"{filename}_grain_mask_{grain_number}_class_{tensor_class}"
                                Images(
                                    data=crop_image,
                                    masked_array=crop_mask[:, :, tensor_class],
                                    **plotting_config["plot_dict"]["grain_mask"],
                                ).plot_and_save()
                    # Always plot these plots
                    # Make a plot of labelled regions with bounding boxes
                    direction_grain_crops = (
                        grains.image_grain_crops.above if direction == "above" else grains.image_grain_crops.below
                    )

                    if direction_grain_crops is not None:
                        full_mask_tensor = direction_grain_crops.full_mask_tensor

                        # Plot image with overlaid masks
                        plot_name = "mask_overlay"
                        plotting_config["plot_dict"][plot_name]["output_dir"] = core_out_path
                        # Tensor, iterate over each channel
                        for tensor_class in range(1, full_mask_tensor.shape[2]):
                            # Set filename for this class
                            plotting_config["plot_dict"][plot_name][
                                "filename"
                            ] = f"{filename}_{direction}_masked_overlay_class_{tensor_class}"
                            full_mask_tensor_class = full_mask_tensor[:, :, tensor_class]
                            full_mask_tensor_class_labelled = Grains.label_regions(full_mask_tensor_class)
                            full_mask_tensor_class_regionprops = Grains.get_region_properties(
                                full_mask_tensor_class_labelled
                            )
                            Images(
                                data=image,
                                masked_array=full_mask_tensor_class.astype(bool),
                                **plotting_config["plot_dict"][plot_name],
                                region_properties=full_mask_tensor_class_regionprops,
                            ).plot_and_save()
                plotting_config["run"] = True
            else:
                # Otherwise, return None and warn that plotting is disabled for grain finding images
                LOGGER.info(f"[{filename}] : Plotting disabled for Grain Finding Images")
            LOGGER.info(f"[{filename}] : Grain Finding stage completed successfully.")
            return grains.image_grain_crops
    # Otherwise, return None and warn grainstats is disabled
    LOGGER.info(f"[{filename}] Detection of grains disabled, GrainStats will not be run.")
    return ImageGrainCrops(above=None, below=None)


def run_grainstats(
    image_grain_crops: ImageGrainCrops,
    filename: str,
    basename: Path,
    grainstats_config: dict,
    plotting_config: dict,
    grain_out_path: Path,
) -> tuple[pd.DataFrame, dict[int : npt.NDArray], dict[int, GrainCrop]]:
    """
    Calculate grain statistics for an image and optionally plots the results.

    Parameters
    ----------
    image_grain_crops : ImageGrainCrops
        ImageGrainCrops object containing the GrainCrops to calculate stats for.
    filename : str
        Name of the image.
    basename : Path
        Path to directory containing the image.
    grainstats_config : dict
        Dictionary of configuration for the GrainStats class to be used when initialised.
    plotting_config : dict
        Dictionary of configuration for plotting images.
    grain_out_path : Path
        Directory to save optional grain statistics visual information to.

    Returns
    -------
    tuple[pd.DataFrame, dict[int: npt.NDArray], dict[int, GrainCrop]]
        Tuple of a pandas DataFrame containing the statsistics for each grain, a dictionary of images and a
        dictionary of GrainCrop objects.
    """
    # Calculate statistics if required
    if grainstats_config["run"]:
        grainstats_config.pop("run")
        class_names = {index + 1: class_name for index, class_name in enumerate(grainstats_config.pop("class_names"))}
        # Grain Statistics :
        try:
            LOGGER.info(f"[{filename}] : *** Grain Statistics ***")
            grain_plot_dict = {
                key: value
                for key, value in plotting_config["plot_dict"].items()
                if key in ["grain_image", "grain_mask", "grain_mask_image"]
            }
            grainstats_dict = {}
            height_profiles_dict = {}

            # There are two layers to process those above the given threshold and those below
            grain_crops_direction: GrainCropsDirection
            for direction, grain_crops_direction in image_grain_crops.__dict__.items():
                if grain_crops_direction is None:
                    LOGGER.warning(
                        f"No grains exist for the {direction} direction. Skipping grainstats for {direction}."
                    )
                    continue
                grainstats_calculator = GrainStats(
                    grain_crops=grain_crops_direction.crops,
                    direction=direction,
                    base_output_dir=grain_out_path,
                    image_name=filename,
                    plot_opts=grain_plot_dict,
                    **grainstats_config,
                )
                grainstats_dict[direction], height_profiles_dict[direction] = grainstats_calculator.calculate_stats()
                grainstats_dict[direction]["threshold"] = direction
            # Create results dataframe from above and below results
            # Appease pylint and ensure that grainstats_df is always created
            grainstats_df = create_empty_dataframe(column_set="grainstats")
            if "above" in grainstats_dict and "below" in grainstats_dict:
                grainstats_df = pd.concat([grainstats_dict["below"], grainstats_dict["above"]])
            elif "above" in grainstats_dict:
                grainstats_df = grainstats_dict["above"]
            elif "below" in grainstats_dict:
                grainstats_df = grainstats_dict["below"]
            else:
                raise ValueError(
                    "grainstats dictionary has neither 'above' nor 'below' keys. This should be impossible."
                )
            grainstats_df["basename"] = basename.parent
            grainstats_df["class_name"] = grainstats_df["class_number"].map(class_names)
            LOGGER.info(f"[{filename}] : Calculated grainstats for {len(grainstats_df)} grains.")
            LOGGER.info(f"[{filename}] : Grainstats stage completed successfully.")
            return grainstats_df, height_profiles_dict, grainstats_calculator.grain_crops
        except Exception:
            LOGGER.info(
                f"[{filename}] : Errors occurred whilst calculating grain statistics. Returning empty dataframe."
            )
            return create_empty_dataframe(column_set="grainstats"), height_profiles_dict, {}
    else:
        LOGGER.info(
            f"[{filename}] : Calculation of grainstats disabled, returning empty dataframe and empty height_profiles."
        )
        return create_empty_dataframe(column_set="grainstats"), {}, {}


def run_disordered_tracing(
    full_image: npt.NDArray,
    image_grain_crops: ImageGrainCrops,
    pixel_to_nm_scaling: float,
    filename: str,
    basename: str,
    core_out_path: Path,
    tracing_out_path: Path,
    disordered_tracing_config: dict,
    plotting_config: dict,
    grainstats_df: pd.DataFrame = None,
) -> tuple[dict | None, pd.DataFrame, pd.DataFrame]:
    """
    Skeletonise and prune grains, adding results to statistics data frames and optionally plot results.

    Parameters
    ----------
    full_image : npt.ndarray
        Image containing the grains to pass to the tracing function.
    image_grain_crops : ImageGrainCrops
        ImageGrainCrops object containing the GrainCrops to perform tracing on.
    pixel_to_nm_scaling : float
        Scaling factor for converting pixel length scales to nanometers, i.e. the number of pixesl per nanometres (nm).
    filename : str
        Name of the image.
    basename : Path
        Path to directory containing the image.
    core_out_path : Path
        Path to save the core disordered trace image to.
    tracing_out_path : Path
        Path to save the optional, diagnostic disordered trace images to.
    disordered_tracing_config : dict
        Dictionary configuration for obtaining a disordered trace representation of the grains.
    plotting_config : dict
        Dictionary configuration for plotting images.
    grainstats_df : pd.DataFrame | None
        The grain statistics dataframe to be added to. This optional argument defaults to `None` in which case an empty grainstats dataframe is created.

    Returns
    -------
    dict
        Dictionary of "grain_<index>" keys and Nx2 coordinate arrays of the disordered grain trace.
    """
    if disordered_tracing_config["run"]:
        disordered_tracing_config.pop("run")
        LOGGER.info(f"[{filename}] : *** Disordered Tracing ***")

        if grainstats_df is None:
            grainstats_df = create_empty_dataframe(column_set="grainstats")

        disordered_traces = defaultdict()
        disordered_trace_grainstats = pd.DataFrame()
        disordered_tracing_stats_image = pd.DataFrame()
        try:
            grain_crop_direction: GrainCropsDirection
            for direction, grain_crop_direction in image_grain_crops.__dict__.items():
                if grain_crop_direction is None:
                    LOGGER.warning(
                        f"[{filename}] : No grains exist for the {direction} direction. Skipping disordered_tracing for {direction}."
                    )
                    continue

                (
                    disordered_traces_cropped_data,
                    _disordered_trace_grainstats,
                    disordered_tracing_images,
                    disordered_tracing_stats,
                ) = trace_image_disordered(
                    full_image=full_image,
                    grain_crops=grain_crop_direction.crops,
                    filename=filename,
                    pixel_to_nm_scaling=pixel_to_nm_scaling,
                    **disordered_tracing_config,
                )
                # save per image new grainstats stats
                _disordered_trace_grainstats["threshold"] = direction
                disordered_trace_grainstats = pd.concat([disordered_trace_grainstats, _disordered_trace_grainstats])
                disordered_tracing_stats["threshold"] = direction
                disordered_tracing_stats["basename"] = basename.parent
                disordered_tracing_stats_image = pd.concat([disordered_tracing_stats_image, disordered_tracing_stats])
                # append direction results to dict
                disordered_traces[direction] = disordered_traces_cropped_data
                # save plots
                Images(
                    full_image,
                    masked_array=disordered_tracing_images.pop("pruned_skeleton"),
                    output_dir=core_out_path,
                    filename=f"{filename}_{direction}_disordered_trace",
                    **plotting_config["plot_dict"]["pruned_skeleton"],
                ).plot_and_save()
                for plot_name, image_value in disordered_tracing_images.items():
                    Images(
                        full_image,
                        masked_array=image_value,
                        output_dir=tracing_out_path / direction,
                        **plotting_config["plot_dict"][plot_name],
                    ).plot_and_save()
            # merge grainstats data with other dataframe
            resultant_grainstats = (
                pd.merge(
                    grainstats_df, disordered_trace_grainstats, how="outer", on=["image", "threshold", "grain_number"]
                )
                if grainstats_df is not None
                else disordered_trace_grainstats
            )
            LOGGER.info(f"[{filename}] : Disordered Tracing stage completed successfully.")
            return disordered_traces, resultant_grainstats, disordered_tracing_stats_image
        except ValueError as e:
            LOGGER.info(f"[{filename}] : Disordered tracing failed with ValueError {e}")

        except Exception as e:
            LOGGER.info(
                f"[{filename}] : Disordered tracing failed - skipping. Consider raising an issue on GitHub. Error: ",
                exc_info=e,
            )
        return (
            disordered_traces,
            grainstats_df,
            create_empty_dataframe(column_set="disordered_tracing_statistics"),
        )
    LOGGER.info(f"[{filename}] Calculation of Disordered Tracing disabled, returning empty dictionary.")
    return None, grainstats_df, create_empty_dataframe(column_set="disordered_tracing_statistics")


def run_nodestats(  # noqa: C901
    image: npt.NDArray,
    disordered_tracing_data: dict,
    pixel_to_nm_scaling: float,
    filename: str,
    core_out_path: Path,
    tracing_out_path: Path,
    nodestats_config: dict,
    plotting_config: dict,
    grainstats_df: pd.DataFrame = None,
) -> tuple[dict, pd.DataFrame]:
    """
    Analyse crossing points in grains adding results to statistics data frames and optionally plot results.

    Parameters
    ----------
    image : npt.ndarray
        Image containing the DNA to pass to the tracing function.
    disordered_tracing_data : dict
        Dictionary of skeletonised and pruned grain masks. Result from "run_disordered_tracing".
    pixel_to_nm_scaling : float
        Scaling factor for converting pixel length scales to nanometers, i.e. the number of pixels per nanometres (nm).
    filename : str
        Name of the image.
    core_out_path : Path
        Path to save the core NodeStats image to.
    tracing_out_path : Path
        Path to save optional, diagnostic NodeStats images to.
    nodestats_config : dict
        Dictionary configuration for analysing the crossing points.
    plotting_config : dict
        Dictionary configuration for plotting images.
    grainstats_df : pd.DataFrame | None
        The grain statistics dataframe to bee added to. This optional argument defaults to `None` in which case an empty grainstats dataframe is created.

    Returns
    -------
    tuple[dict, pd.DataFrame]
        A NodeStats analysis dictionary and grainstats metrics dataframe.
    """
    if nodestats_config["run"]:
        nodestats_config.pop("run")
        LOGGER.info(f"[{filename}] : *** Nodestats ***")

        if grainstats_df is None:
            grainstats_df = create_empty_dataframe(column_set="grainstats")

        nodestats_whole_data = defaultdict()
        nodestats_grainstats = pd.DataFrame()
        try:
            # run image using directional grain masks
            for direction, disordered_tracing_direction_data in disordered_tracing_data.items():
                (
                    nodestats_data,
                    _nodestats_grainstats,
                    nodestats_full_images,
                    nodestats_branch_images,
                ) = nodestats_image(
                    image=image,
                    disordered_tracing_direction_data=disordered_tracing_direction_data,
                    filename=filename,
                    pixel_to_nm_scaling=pixel_to_nm_scaling,
                    **nodestats_config,
                )
                # save per image new grainstats stats
                _nodestats_grainstats["threshold"] = direction
                nodestats_grainstats = pd.concat([nodestats_grainstats, _nodestats_grainstats])
                # append direction results to dict
                nodestats_whole_data[direction] = {"stats": nodestats_data, "images": nodestats_branch_images}
                # save whole image plots
                Images(
                    filename=f"{filename}_{direction}_nodes",
                    data=image,
                    masked_array=nodestats_full_images.pop("connected_nodes"),
                    output_dir=core_out_path,
                    **plotting_config["plot_dict"]["connected_nodes"],
                ).plot_and_save()
                for plot_name, image_value in nodestats_full_images.items():
                    Images(
                        image,
                        masked_array=image_value,
                        output_dir=tracing_out_path / direction,
                        **plotting_config["plot_dict"][plot_name],
                    ).plot_and_save()
                # plot single node images
                for mol_no, mol_stats in nodestats_data.items():
                    if mol_stats is not None:
                        for node_no, single_node_stats in mol_stats.items():
                            # plot the node and branch_mask images
                            for cropped_image_type, cropped_image in nodestats_branch_images[mol_no]["nodes"][
                                node_no
                            ].items():
                                Images(
                                    nodestats_branch_images[mol_no]["grain"]["grain_image"],
                                    masked_array=cropped_image,
                                    output_dir=tracing_out_path / direction / "nodes",
                                    filename=f"{mol_no}_{node_no}_{cropped_image_type}",
                                    **plotting_config["plot_dict"][cropped_image_type],
                                ).plot_and_save()
                            # plot crossing height linetrace
                            if "all" in plotting_config["image_set"] or "nodestats" in plotting_config["image_set"]:
                                if not single_node_stats["error"]:
                                    fig, _ = plot_crossing_linetrace_halfmax(
                                        branch_stats_dict=single_node_stats["branch_stats"],
                                        mask_cmap=plotting_config["plot_dict"]["node_line_trace"]["mask_cmap"],
                                        title=plotting_config["plot_dict"]["node_line_trace"]["mask_cmap"],
                                    )
                                    fig.savefig(
                                        tracing_out_path
                                        / direction
                                        / "nodes"
                                        / f"{mol_no}_{node_no}_linetrace_halfmax.svg",
                                        format="svg",
                                    )
            # merge grainstats data with other dataframe
            resultant_grainstats = (
                pd.merge(grainstats_df, nodestats_grainstats, how="outer", on=["image", "threshold", "grain_number"])
                if grainstats_df is not None
                else nodestats_grainstats
            )
            LOGGER.info(f"[{filename}] : NodeStats stage completed successfully.")
            # merge all image dictionaries
            return nodestats_whole_data, resultant_grainstats
        except UnboundLocalError as e:
            LOGGER.info(
                f"[{filename}] : NodeStats failed with UnboundLocalError {e} - all skeletons pruned in the Disordered Tracing step."
            )

        except KeyError as e:
            LOGGER.info(
                f"[{filename}] : NodeStats failed with KeyError {e} - no skeletons found from the Disordered Tracing step."
            )
        except Exception as e:
            LOGGER.info(
                f"[{filename}] : NodeStats failed - skipping. Consider raising an issue on GitHub. Error: ", exc_info=e
            )
        return nodestats_whole_data, grainstats_df
    LOGGER.info(f"[{filename}] : Calculation of nodestats disabled, returning empty dataframe.")
    return None, grainstats_df


# need to add in the molstats here
def run_ordered_tracing(
    image: npt.NDArray,
    disordered_tracing_data: dict,
    nodestats_data: dict,
    filename: str,
    basename: Path,
    core_out_path: Path,
    tracing_out_path: Path,
    ordered_tracing_config: dict,
    plotting_config: dict,
    grainstats_df: pd.DataFrame = None,
) -> tuple:
    """
    Order coordinates of traces, adding results to statistics data frames and optionally plot results.

    Parameters
    ----------
    image : npt.ndarray
        Image containing the DNA to pass to the tracing function.
    disordered_tracing_data : dict
        Dictionary of skeletonised and pruned grain masks. Result from "run_disordered_tracing".
    nodestats_data : dict
        Dictionary of images and statistics from the NodeStats analysis. Result from "run_nodestats".
    filename : str
        Name of the image.
    basename : Path
        The path of the files' parent directory.
    core_out_path : Path
        Path to save the core ordered tracing image to.
    tracing_out_path : Path
        Path to save optional, diagnostic ordered trace images to.
    ordered_tracing_config : dict
        Dictionary configuration for obtaining an ordered trace representation of the skeletons.
    plotting_config : dict
        Dictionary configuration for plotting images.
    grainstats_df : pd.DataFrame | None
        The grain statistics dataframe to be added to. This optional argument defaults to `None` in which case an empty grainstats dataframe is created.

    Returns
    -------
    tuple[dict, pd.DataFrame]
        A NodeStats analysis dictionary and grainstats metrics dataframe.
    """
    if ordered_tracing_config["run"]:
        ordered_tracing_config.pop("run")
        LOGGER.info(f"[{filename}] : *** Ordered Tracing ***")

        if grainstats_df is None:
            grainstats_df = create_empty_dataframe(column_set="grainstats")

        ordered_tracing_image_data = defaultdict()
        ordered_tracing_molstats = pd.DataFrame()
        ordered_tracing_grainstats = pd.DataFrame()
        try:
            # run image using directional grain masks
            for direction, disordered_tracing_direction_data in disordered_tracing_data.items():
                # Check if there are grains
                if not disordered_tracing_direction_data:
                    LOGGER.warning(
                        f"[{filename}] : No skeletons exist for the {direction} direction. Skipping ordered_tracing for {direction}."
                    )
                    raise ValueError(f"No skeletons exist for the {direction} direction")
                # if grains are found
                (
                    ordered_tracing_data,
                    _ordered_tracing_grainstats,
                    _ordered_tracing_molstats,
                    ordered_tracing_full_images,
                ) = ordered_tracing_image(
                    image=image,
                    disordered_tracing_direction_data=disordered_tracing_direction_data,
                    nodestats_direction_data=nodestats_data[direction],
                    filename=filename,
                    **ordered_tracing_config,
                )
                # save per image new grainstats stats
                _ordered_tracing_grainstats["threshold"] = direction
                ordered_tracing_grainstats = pd.concat([ordered_tracing_grainstats, _ordered_tracing_grainstats])
                _ordered_tracing_molstats["threshold"] = direction
                ordered_tracing_molstats = pd.concat([ordered_tracing_molstats, _ordered_tracing_molstats])
                # append direction results to dict
                ordered_tracing_image_data[direction] = ordered_tracing_data
                # save whole image plots
                plotting_config["plot_dict"]["ordered_traces"]["core_set"] = True  # fudge around core having own cmap
                Images(
                    filename=f"{filename}_{direction}_ordered_traces",
                    data=image,
                    masked_array=ordered_tracing_full_images.pop("ordered_traces"),
                    output_dir=core_out_path,
                    **plotting_config["plot_dict"]["ordered_traces"],
                ).plot_and_save()
                # save optional diagnostic plots (those with core_set = False)
                for plot_name, image_value in ordered_tracing_full_images.items():
                    Images(
                        image,
                        masked_array=image_value,
                        output_dir=tracing_out_path / direction,
                        **plotting_config["plot_dict"][plot_name],
                    ).plot_and_save()
            # merge grainstats data with other dataframe
            resultant_grainstats = (
                pd.merge(
                    grainstats_df, ordered_tracing_grainstats, how="outer", on=["image", "threshold", "grain_number"]
                )
                if grainstats_df is not None
                else ordered_tracing_grainstats
            )
            ordered_tracing_molstats["basename"] = basename.parent
            LOGGER.info(f"[{filename}] : Ordered Tracing stage completed successfully.")
            # merge all image dictionaries
            return ordered_tracing_image_data, resultant_grainstats, ordered_tracing_molstats
        except ValueError as e:
            LOGGER.info(
                f"[{filename}] : Ordered Tracing failed with ValueError {e} - No skeletons exist for the {direction} direction."
            )

        except KeyError as e:
            LOGGER.info(
                f"[{filename}] : Ordered Tracing failed with KeyError {e} - no skeletons found from the Disordered Tracing step."
            )

        except Exception as e:
            LOGGER.info(
                f"[{filename}] : Ordered Tracing failed - skipping. Consider raising an issue on GitHub. Error: ",
                exc_info=e,
            )
        return (
            ordered_tracing_image_data,
            grainstats_df,
            create_empty_dataframe(column_set="mol_statistics"),
        )

    return None, grainstats_df, create_empty_dataframe(column_set="mol_statistics")


def run_splining(
    image: npt.NDArray,
    ordered_tracing_data: dict,
    pixel_to_nm_scaling: float,
    filename: str,
    core_out_path: Path,
    splining_config: dict,
    plotting_config: dict,
    grainstats_df: pd.DataFrame = None,
    molstats_df: pd.DataFrame = None,
) -> tuple:
    """
    Smooth the ordered trace coordinates, adding results to statistics data frames and optionally plot results.

    Parameters
    ----------
    image : npt.NDArray
        Image containing the DNA to pass to the tracing function.
    ordered_tracing_data : dict
        Dictionary of ordered coordinates. Result from "run_ordered_tracing".
    pixel_to_nm_scaling : float
        Scaling factor for converting pixel length scales to nanometers, i.e. the number of pixels per nanometres (nm).
    filename : str
        Name of the image.
    core_out_path : Path
        Path to save the core ordered tracing image to.
    splining_config : dict
        Dictionary configuration for obtaining an ordered trace representation of the skeletons.
    plotting_config : dict
        Dictionary configuration for plotting images.
    grainstats_df : pd.DataFrame | None
        The grain statistics dataframe to be added to. This optional argument defaults to `None` in which case an empty grainstats dataframe is created.
    molstats_df : pd.DataFrame | None
        The molecule statistics dataframe to be added to. This optional argument defaults to `None` in which case an empty grainstats dataframe is created.

    Returns
    -------
    tuple[dict, pd.DataFrame]
        A smooth curve analysis dictionary and grainstats metrics dataframe.
    """
    if splining_config["run"]:
        splining_config.pop("run")
        LOGGER.info(f"[{filename}] : *** Splining ***")

        if grainstats_df is None:
            grainstats_df = create_empty_dataframe(column_set="grainstats")
        if molstats_df is None:
            molstats_df = create_empty_dataframe(column_set="mol_statistics")

        splined_image_data = defaultdict()
        splining_grainstats = pd.DataFrame()
        splining_molstats = pd.DataFrame()
        try:
            # run image using directional grain masks
            for direction, ordered_tracing_direction_data in ordered_tracing_data.items():
                if not ordered_tracing_direction_data:
                    LOGGER.warning(
                        f"[{filename}] : No grains exist for the {direction} direction. Skipping disordered_tracing for {direction}."
                    )
                    splining_grainstats = create_empty_dataframe(column_set="grainstats")
                    splining_molstats = create_empty_dataframe(column_set="mol_statistics")
                    raise ValueError(f"No grains exist for the {direction} direction")
                # if grains are found
                (
                    splined_data,
                    _splining_grainstats,
                    _splining_molstats,
                ) = splining_image(
                    image=image,
                    ordered_tracing_direction_data=ordered_tracing_direction_data,
                    filename=filename,
                    pixel_to_nm_scaling=pixel_to_nm_scaling,
                    **splining_config,
                )
                # save per image new grainstats stats
                _splining_grainstats["threshold"] = direction
                splining_grainstats = pd.concat([splining_grainstats, _splining_grainstats])
                _splining_molstats["threshold"] = direction
                splining_molstats = pd.concat([splining_molstats, _splining_molstats])
                # append direction results to dict
                splined_image_data[direction] = splined_data
                # Plot traces on each grain individually
                all_splines = []
                for _, grain_dict in splined_data.items():
                    for _, mol_dict in grain_dict.items():
                        all_splines.append(mol_dict["spline_coords"] + mol_dict["bbox"][:2])

                Images(
                    data=image,
                    output_dir=core_out_path,
                    filename=f"{filename}_{direction}_all_splines",
                    plot_coords=all_splines,
                    **plotting_config["plot_dict"]["splined_trace"],
                ).plot_and_save()
            # merge grainstats data with other dataframe
            resultant_grainstats = (
                pd.merge(grainstats_df, splining_grainstats, how="outer", on=["image", "threshold", "grain_number"])
                if grainstats_df is not None
                else splining_grainstats
            )
            # merge molstats data with other dataframe
            resultant_molstats = (
                pd.merge(
                    molstats_df,
                    splining_molstats,
                    how="outer",
                    on=["image", "threshold", "grain_number", "molecule_number"],
                )
                if molstats_df is not None
                else splining_molstats
            )
            LOGGER.info(f"[{filename}] : Splining stage completed successfully.")
            # merge all image dictionaries
            return splined_image_data, resultant_grainstats, resultant_molstats
        except KeyError as e:
            LOGGER.info(
                f"[{filename}] : Splining failed with KeyError {e} - no ordered traces found from the Ordered Tracing step."
            )
            return (
                splined_image_data,
                grainstats_df,
                create_empty_dataframe(column_set="mol_statistics"),
            )
        except Exception as e:
            LOGGER.error(
                f"[{filename}] : Splining failed - skipping. Consider raising an issue on GitHub. Error: ", exc_info=e
            )
            return splined_image_data, grainstats_df, splining_molstats
    return None, grainstats_df, molstats_df


def run_curvature_stats(
    image: np.ndarray,
    cropped_image_data: dict,
    grain_trace_data: dict,
    pixel_to_nm_scaling: float,
    filename: str,
    core_out_path: Path,
    tracing_out_path: Path,
    curvature_config: dict,
    plotting_config: dict,
) -> dict | None:
    """
    Calculate curvature statistics for the traced DNA molecules.

    Currently only works on simple traces, not branched traces.

    Parameters
    ----------
    image : np.ndarray
        AFM image, for plotting purposes.
    cropped_image_data : dict
        Dictionary containing cropped images.
    grain_trace_data : dict
        Dictionary of grain trace data.
    pixel_to_nm_scaling : float
        Scaling factor for converting pixel length scales to nanometres.
        ie the number of pixels per nanometre.
    filename : str
        Name of the image.
    core_out_path : Path
        Path to save the core curvature image to.
    tracing_out_path : Path
        Path to save the optional, diagnostic curvature images to.
    curvature_config : dict
        Dictionary of configuration for running the curvature stats.
    plotting_config : dict
        Dictionary of configuration for plotting images.

    Returns
    -------
    dict
        Dictionary containing curvature statistics.
    """
    if curvature_config["run"]:
        try:
            curvature_config.pop("run")
            LOGGER.info(f"[{filename}] : *** Curvature Stats ***")
            all_directions_grains_curvature_stats_dict: dict = {}
            for direction in grain_trace_data.keys():
                # Pass the traces to the curvature stats function
                grains_curvature_stats_dict = calculate_curvature_stats_image(
                    all_grain_smoothed_data=grain_trace_data[direction],
                    pixel_to_nm_scaling=pixel_to_nm_scaling,
                )

                Images(
                    np.array([[0, 0], [0, 0]]),  # dummy data, as the image is passed in the method call.
                    filename=f"{filename}_{direction}_curvature",
                    output_dir=core_out_path,
                    **plotting_config["plot_dict"]["curvature"],
                ).plot_curvatures(
                    image=image,
                    cropped_images=cropped_image_data[direction],
                    grains_curvature_stats_dict=grains_curvature_stats_dict,
                    all_grain_smoothed_data=grain_trace_data[direction],
                    colourmap_normalisation_bounds=curvature_config["colourmap_normalisation_bounds"],
                )

                Images(
                    np.array([[0, 0], [0, 0]]),  # dummy data, as the image is passed in the method call.
                    output_dir=tracing_out_path / direction / "curvature",
                    **plotting_config["plot_dict"]["curvature_individual_grains"],
                ).plot_curvatures_individual_grains(
                    cropped_images=cropped_image_data[direction],
                    grains_curvature_stats_dict=grains_curvature_stats_dict,
                    all_grains_smoothed_data=grain_trace_data[direction],
                    colourmap_normalisation_bounds=curvature_config["colourmap_normalisation_bounds"],
                )

                all_directions_grains_curvature_stats_dict[direction] = grains_curvature_stats_dict

            return all_directions_grains_curvature_stats_dict
        except Exception as e:
            LOGGER.error(
                f"[{filename}] : Splining failed - skipping. Consider raising an issue on GitHub. Error: ", exc_info=e
            )
            return None
    LOGGER.info(f"[{filename}] : Calculation of Curvature Stats disabled, returning None.")
    return None


def get_out_paths(
    image_path: Path, base_dir: Path, output_dir: Path, filename: str, plotting_config: dict, grain_dirs: bool = True
):
    """
    Determine components of output paths for a given image and plotting config.

    Parameters
    ----------
    image_path : Path
        Path of the image being processed.
    base_dir : Path
        Path of the data folder.
    output_dir : Path
        Base output directory for output data.
    filename : str
        Name of the image being processed.
    plotting_config : dict
        Dictionary of configuration for plotting images.
    grain_dirs : bool
        Whether to create the ``grains`` and ``dnatracing`` sub-directories, by default this is ``True`` but it should
        be set to ``False`` when running just ``Filters``.

    Returns
    -------
    tuple
        Core output path for general file outputs, filter output path for flattening related files and
        grain output path for grain finding related files.
    """
    LOGGER.info(f"Processing : {filename}")
    core_out_path = get_out_path(image_path, base_dir, output_dir).parent / "processed"
    core_out_path.mkdir(parents=True, exist_ok=True)
    filter_out_path = core_out_path / filename / "filters"
    grain_out_path = core_out_path / filename / "grains"
    tracing_out_path = core_out_path / filename / "dnatracing"
    if "core" not in plotting_config["image_set"] and grain_dirs:
        filter_out_path.mkdir(exist_ok=True, parents=True)
        Path.mkdir(grain_out_path / "above", parents=True, exist_ok=True)
        Path.mkdir(grain_out_path / "below", parents=True, exist_ok=True)
        Path.mkdir(tracing_out_path / "above", parents=True, exist_ok=True)
        Path.mkdir(tracing_out_path / "below", parents=True, exist_ok=True)
        Path.mkdir(tracing_out_path / "above" / "nodes", parents=True, exist_ok=True)
        Path.mkdir(tracing_out_path / "below" / "nodes", parents=True, exist_ok=True)
        Path.mkdir(tracing_out_path / "above" / "curvature", parents=True, exist_ok=True)
        Path.mkdir(tracing_out_path / "below" / "curvature", parents=True, exist_ok=True)

    return core_out_path, filter_out_path, grain_out_path, tracing_out_path


def process_scan(
    topostats_object: dict,
    base_dir: str | Path,
    filter_config: dict,
    grains_config: dict,
    grainstats_config: dict,
    disordered_tracing_config: dict,
    nodestats_config: dict,
    ordered_tracing_config: dict,
    splining_config: dict,
    curvature_config: dict,
    plotting_config: dict,
    output_dir: str | Path = "output",
) -> tuple[dict, pd.DataFrame, dict]:
    """
    Process a single image, filtering, finding grains and calculating their statistics.

    Parameters
    ----------
    topostats_object : dict[str, Union[npt.NDArray, Path, float]]
        A dictionary with keys 'image', 'img_path' and 'pixel_to_nm_scaling' containing a file or frames' image, it's
        path and it's pixel to namometre scaling value.
    base_dir : str | Path
        Directory to recursively search for files, if not specified the current directory is scanned.
    filter_config : dict
        Dictionary of configuration options for running the Filter stage.
    grains_config : dict
        Dictionary of configuration options for running the Grain detection stage.
    grainstats_config : dict
        Dictionary of configuration options for running the Grain Statistics stage.
    disordered_tracing_config : dict
        Dictionary configuration for obtaining a disordered trace representation of the grains.
    nodestats_config : dict
        Dictionary of configuration options for running the NodeStats stage.
    ordered_tracing_config : dict
        Dictionary configuration for obtaining an ordered trace representation of the skeletons.
    splining_config : dict
        Dictionary of configuration options for running the splining stage.
    curvature_config : dict
        Dictionary of configuration options for running the curvature stats stage.
    plotting_config : dict
        Dictionary of configuration options for plotting figures.
    output_dir : str | Path
        Directory to save output to, it will be created if it does not exist. If it already exists then it is possible
        that output will be over-written.

    Returns
    -------
    tuple[dict, pd.DataFrame, dict]
        TopoStats dictionary object, DataFrame containing grain statistics and dna tracing statistics,
        and dictionary containing general image statistics.
    """
    core_out_path, filter_out_path, grain_out_path, tracing_out_path = get_out_paths(
        image_path=topostats_object["img_path"],
        base_dir=base_dir,
        output_dir=output_dir,
        filename=topostats_object["filename"],
        plotting_config=plotting_config,
    )

    plotting_config = add_pixel_to_nm_to_plotting_config(plotting_config, topostats_object["pixel_to_nm_scaling"])

    # Flatten Image
    image = run_filters(
        unprocessed_image=topostats_object["image_original"],
        pixel_to_nm_scaling=topostats_object["pixel_to_nm_scaling"],
        filename=topostats_object["filename"],
        filter_out_path=filter_out_path,
        core_out_path=core_out_path,
        filter_config=filter_config,
        plotting_config=plotting_config,
    )
    # Use flattened image if one is returned, else use original image
    topostats_object["image"] = image if image is not None else topostats_object["image_original"]

    # Find Grains :
    image_grain_crops = run_grains(
        image=topostats_object["image"],
        pixel_to_nm_scaling=topostats_object["pixel_to_nm_scaling"],
        filename=topostats_object["filename"],
        grain_out_path=grain_out_path,
        core_out_path=core_out_path,
        plotting_config=plotting_config,
        grains_config=grains_config,
    )

    topostats_object["grain_tensors"] = {}
    topostats_object["grain_tensors"]["above"] = (
        image_grain_crops.above.full_mask_tensor if image_grain_crops.above is not None else None
    )
    topostats_object["grain_tensors"]["below"] = (
        image_grain_crops.below.full_mask_tensor if image_grain_crops.below is not None else None
    )

    if image_grain_crops.above is not None or image_grain_crops.below is not None:
        # Grainstats :
        grainstats_df, height_profiles, _ = run_grainstats(
            image_grain_crops=image_grain_crops,
            filename=topostats_object["filename"],
            basename=topostats_object["img_path"],
            grainstats_config=grainstats_config,
            plotting_config=plotting_config,
            grain_out_path=grain_out_path,
        )
        topostats_object["height_profiles"] = height_profiles

        # Disordered Tracing
        disordered_traces_data, grainstats_df, disordered_tracing_stats = run_disordered_tracing(
            full_image=topostats_object["image"],
            image_grain_crops=image_grain_crops,
            pixel_to_nm_scaling=topostats_object["pixel_to_nm_scaling"],
            filename=topostats_object["filename"],
            basename=topostats_object["img_path"],
            core_out_path=core_out_path,
            tracing_out_path=tracing_out_path,
            disordered_tracing_config=disordered_tracing_config,
            grainstats_df=grainstats_df,
            plotting_config=plotting_config,
        )
        topostats_object["disordered_traces"] = disordered_traces_data

        # Nodestats
        nodestats, grainstats_df = run_nodestats(
            image=topostats_object["image"],
            disordered_tracing_data=topostats_object["disordered_traces"],
            pixel_to_nm_scaling=topostats_object["pixel_to_nm_scaling"],
            filename=topostats_object["filename"],
            core_out_path=core_out_path,
            tracing_out_path=tracing_out_path,
            plotting_config=plotting_config,
            nodestats_config=nodestats_config,
            grainstats_df=grainstats_df,
        )

        # Ordered Tracing
        ordered_tracing, grainstats_df, molstats_df = run_ordered_tracing(
            image=topostats_object["image"],
            disordered_tracing_data=topostats_object["disordered_traces"],
            nodestats_data=nodestats,
            filename=topostats_object["filename"],
            basename=topostats_object["img_path"],
            core_out_path=core_out_path,
            tracing_out_path=tracing_out_path,
            ordered_tracing_config=ordered_tracing_config,
            plotting_config=plotting_config,
            grainstats_df=grainstats_df,
        )
        topostats_object["ordered_traces"] = ordered_tracing
        topostats_object["nodestats"] = nodestats  # looks weird but ordered adds an extra field

        # splining
        splined_data, grainstats_df, molstats_df = run_splining(
            image=topostats_object["image"],
            ordered_tracing_data=topostats_object["ordered_traces"],
            pixel_to_nm_scaling=topostats_object["pixel_to_nm_scaling"],
            filename=topostats_object["filename"],
            core_out_path=core_out_path,
            plotting_config=plotting_config,
            splining_config=splining_config,
            grainstats_df=grainstats_df,
            molstats_df=molstats_df,
        )
        # Add grain trace data to topostats object
        topostats_object["splining"] = splined_data

        # Curvature Stats
        grain_curvature_stats_dict = run_curvature_stats(
            image=topostats_object["image"],
            cropped_image_data=topostats_object["disordered_traces"],
            grain_trace_data=topostats_object["splining"],
            pixel_to_nm_scaling=topostats_object["pixel_to_nm_scaling"],
            filename=topostats_object["filename"],
            core_out_path=core_out_path,
            tracing_out_path=tracing_out_path,
            curvature_config=curvature_config,
            plotting_config=plotting_config,
        )

        topostats_object["grain_curvature_stats"] = grain_curvature_stats_dict

    else:
        LOGGER.warning(f"[{topostats_object['filename']}] : No grains found, skipping grainstats and tracing stages.")
        grainstats_df = create_empty_dataframe(column_set="grainstats")
        molstats_df = create_empty_dataframe(column_set="mol_statistics")
        disordered_tracing_stats = create_empty_dataframe(column_set="disordered_tracing_statistics")
        height_profiles = {}

    # Get image statistics
    LOGGER.info(f"[{topostats_object['filename']}] : *** Image Statistics ***")
    # Provide the raw image if image has not been flattened, else provide the flattened image.
    if topostats_object["image"] is not None:
        image_for_image_stats = topostats_object["image"]
    else:
        image_for_image_stats = topostats_object["image_original"]
    # Calculate image statistics - returns a dictionary
    image_stats = image_statistics(
        image=image_for_image_stats,
        filename=topostats_object["filename"],
        results_df=grainstats_df,
        pixel_to_nm_scaling=topostats_object["pixel_to_nm_scaling"],
    )

    # Save the topostats dictionary object to .topostats file.
    save_topostats_file(
        output_dir=core_out_path,
        filename=str(topostats_object["filename"]),
        topostats_object=topostats_object,
    )

    return (
        topostats_object["img_path"],
        grainstats_df,
        height_profiles,
        image_stats,
        disordered_tracing_stats,
        molstats_df,
    )


def process_filters(
    topostats_object: dict,
    base_dir: str | Path,
    filter_config: dict,
    plotting_config: dict,
    output_dir: str | Path = "output",
) -> tuple[str, bool]:
    """
    Filter an image return the flattened images and save to ''.topostats''.

    Runs just the first key step of flattening images to remove noise, tilt and optionally scars saving to
    ''.topostats'' for subsequent processing and analyses.

    Parameters
    ----------
    topostats_object : dict[str, Union[npt.NDArray, Path, float]]
        A dictionary with keys 'image', 'img_path' and 'pixel_to_nm_scaling' containing a file or frames' image, it's
        path and it's pixel to namometre scaling value.
    base_dir : str | Path
        Directory to recursively search for files, if not specified the current directory is scanned.
    filter_config : dict
        Dictionary of configuration options for running the Filter stage.
    plotting_config : dict
        Dictionary of configuration options for plotting figures.
    output_dir : str | Path
        Directory to save output to, it will be created if it does not exist. If it already exists then it is possible
        that output will be over-written.

    Returns
    -------
    tuple[str, bool]
        A tuple of the image and a boolean indicating if the image was successfully processed.
    """
    core_out_path, filter_out_path, _, _ = get_out_paths(
        image_path=topostats_object["img_path"],
        base_dir=base_dir,
        output_dir=output_dir,
        filename=topostats_object["filename"],
        plotting_config=plotting_config,
        grain_dirs=False,
    )

    plotting_config = add_pixel_to_nm_to_plotting_config(plotting_config, topostats_object["pixel_to_nm_scaling"])

    # Flatten Image
    try:
        image = run_filters(
            unprocessed_image=topostats_object["image_original"],
            pixel_to_nm_scaling=topostats_object["pixel_to_nm_scaling"],
            filename=topostats_object["filename"],
            filter_out_path=filter_out_path,
            core_out_path=core_out_path,
            filter_config=filter_config,
            plotting_config=plotting_config,
        )
        # Use flattened image if one is returned, else use original image
        topostats_object["image"] = image if image is not None else topostats_object["image_original"]

        # Save the topostats dictionary object to .topostats file.
        save_topostats_file(
            output_dir=core_out_path,
            filename=str(topostats_object["filename"]),
            topostats_object=topostats_object,
        )
        return (topostats_object["filename"], True)
    except:  # noqa: E722  # pylint: disable=bare-except
        LOGGER.info(f"Filtering failed for image : {topostats_object['filename']}")
        return (topostats_object["filename"], False)


def process_grains(
    topostats_object: dict,
    base_dir: str | Path,
    grains_config: dict,
    plotting_config: dict,
    output_dir: str | Path = "output",
) -> tuple[str, bool]:
    """
    Detect grains in flattened images and save to ''.topostats''.

    Runs grain detection on flattened images to identify grains and save data to  ''.topostats'' for subsequent
    processing and analyses.

    Parameters
    ----------
    topostats_object : dict[str, Union[npt.NDArray, Path, float]]
        A dictionary with keys 'image', 'img_path' and 'pixel_to_nm_scaling' containing a file or frames' image, it's
        path and it's pixel to namometre scaling value.
    base_dir : str | Path
        Directory to recursively search for files, if not specified the current directory is scanned.
    grains_config : dict
        Dictionary of configuration options for running the Filter stage.
    plotting_config : dict
        Dictionary of configuration options for plotting figures.
    output_dir : str | Path
        Directory to save output to, it will be created if it does not exist. If it already exists then it is possible
        that output will be over-written.

    Returns
    -------
    tuple[str, bool]
        A tuple of the image and a boolean indicating if the image was successfully processed.
    """
    core_out_path, _, grain_out_path, _ = get_out_paths(
        image_path=topostats_object["img_path"],
        base_dir=base_dir,
        output_dir=output_dir,
        filename=topostats_object["filename"],
        plotting_config=plotting_config,
    )
    plotting_config = add_pixel_to_nm_to_plotting_config(plotting_config, topostats_object["pixel_to_nm_scaling"])
    # Find Grains using the filtered image
    try:
        image_grain_crops = run_grains(
            image=topostats_object["image"],
            pixel_to_nm_scaling=topostats_object["pixel_to_nm_scaling"],
            filename=topostats_object["filename"],
            grain_out_path=grain_out_path,
            core_out_path=core_out_path,
            plotting_config=plotting_config,
            grains_config=grains_config,
        )
        topostats_object["grain_tensors"] = {}
        if image_grain_crops.above is not None:
            topostats_object["grain_tensors"]["above"] = image_grain_crops.above.full_mask_tensor
        if image_grain_crops.below is not None:
            topostats_object["grain_tensors"]["below"] = image_grain_crops.below.full_mask_tensor
        # Save the topostats dictionary object to .topostats file.
        save_topostats_file(
            output_dir=core_out_path,
            filename=str(topostats_object["filename"]),
            topostats_object=topostats_object,
        )
        return (topostats_object["filename"], True)
    except:  # noqa: E722  # pylint: disable=bare-except
        LOGGER.info(f"Grain detection failed for image : {topostats_object['filename']}")
        return (topostats_object["filename"], False)


# @ns-rse 2025-03-28 - This function will need updating to work with ImageGrainCrops object once we have updated loading
# the HDF5 dictionaries to such objects.
def process_grainstats(
    topostats_object: dict,
    base_dir: str | Path,
    grainstats_config: dict,
    plotting_config: dict,
    output_dir: str | Path = "output",
) -> tuple[str, bool]:
    """
    Calculate grain statistics in an image where grains have already been detected.

    Runs just the first key step of flattening images to remove noise, tilt and optionally scars saving to
    ''.topostats'' for subsequent processing and analyses.

    Parameters
    ----------
    topostats_object : dict[str, Union[npt.NDArray, Path, float]]
        A dictionary with keys 'image', 'img_path' and 'pixel_to_nm_scaling' containing a file or frames' image, it's
        path and it's pixel to namometre scaling value.
    base_dir : str | Path
        Directory to recursively search for files, if not specified the current directory is scanned.
    grainstats_config : dict
        Dictionary of configuration options for running the Filter stage.
    plotting_config : dict
        Dictionary of configuration options for plotting figures.
    output_dir : str | Path
        Directory to save output to, it will be created if it does not exist. If it already exists then it is possible
        that output will be over-written.

    Returns
    -------
    tuple[str, pd.DataFrame]
        A tuple of the image and a boolean indicating if the image was successfully processed.
    """
    core_out_path, _, grainstats_out_path, _ = get_out_paths(
        image_path=topostats_object["img_path"],
        base_dir=base_dir,
        output_dir=output_dir,
        filename=topostats_object["filename"],
        plotting_config=plotting_config,
    )
    plotting_config = add_pixel_to_nm_to_plotting_config(plotting_config, topostats_object["pixel_to_nm_scaling"])

    # Calculate grainstats if there are any to be detected
    try:
        if "above" in topostats_object["grain_masks"].keys() or "below" in topostats_object["grain_masks"].keys():
            grainstats_df, height_profiles, grain_crops = run_grainstats(
                image_grain_crops=topostats_object["image"],
                filename=topostats_object["filename"],
                basename=topostats_object["img_path"],
                grainstats_config=grainstats_config,
                plotting_config=plotting_config,
                grain_out_path=grainstats_out_path,
            )
            # Save the topostats dictionary object to .topostats file.
            topostats_object["height_profiles"] = height_profiles
            save_topostats_file(
                output_dir=core_out_path,
                filename=str(topostats_object["filename"]),
                topostats_object=topostats_object,
            )
            return (topostats_object["filename"], grainstats_df, height_profiles, grain_crops)
        return (
            topostats_object["filename"],
            create_empty_dataframe(column_set="grainstats"),
            None,
            None,
        )
    except:  # noqa: E722  # pylint: disable=bare-except
        LOGGER.info(f"Grain detection failed for image : {topostats_object['filename']}")
        return (topostats_object["filename"], create_empty_dataframe(column_set="grainstats"), False)


def check_run_steps(  # noqa: C901
    filter_run: bool,
    grains_run: bool,
    grainstats_run: bool,
    disordered_tracing_run: bool,
    nodestats_run: bool,
    ordered_tracing_run: bool,
    splining_run: bool,
) -> None:
    """
    Check options for running steps (Filter, Grain, Grainstats and DNA tracing) are logically consistent.

    This checks that earlier steps required are enabled.

    Parameters
    ----------
    filter_run : bool
        Flag for running Filtering.
    grains_run : bool
        Flag for running Grains.
    grainstats_run : bool
        Flag for running GrainStats.
    disordered_tracing_run : bool
        Flag for running Disordered Tracing.
    nodestats_run : bool
        Flag for running NodeStats.
    ordered_tracing_run : bool
        Flag for running Ordered Tracing.
    splining_run : bool
        Flag for running DNA Tracing.
    """
    LOGGER.debug(f"{filter_run=}")
    LOGGER.debug(f"{grains_run=}")
    LOGGER.debug(f"{grainstats_run=}")
    LOGGER.debug(f"{disordered_tracing_run=}")
    LOGGER.debug(f"{nodestats_run=}")
    LOGGER.debug(f"{ordered_tracing_run=}")
    LOGGER.debug(f"{splining_run=}")
    if splining_run:
        if ordered_tracing_run is False:
            LOGGER.error("Splining enabled but Ordered Tracing disabled. Please check your configuration file.")
        if nodestats_run is False:
            LOGGER.error("Splining enabled but NodeStats disabled. Tracing will use the 'old' method.")
        if disordered_tracing_run is False:
            LOGGER.error("Splining enabled but Disordered Tracing disabled. Please check your configuration file.")
        elif grainstats_run is False:
            LOGGER.error("Splining enabled but Grainstats disabled. Please check your configuration file.")
        elif grains_run is False:
            LOGGER.error("Splining enabled but Grains disabled. Please check your configuration file.")
        elif filter_run is False:
            LOGGER.error("Splining enabled but Filters disabled. Please check your configuration file.")
        else:
            LOGGER.info("Configuration run options are consistent, processing can proceed.")
    elif ordered_tracing_run:
        if disordered_tracing_run is False:
            LOGGER.error(
                "Ordered Tracing enabled but Disordered Tracing disabled. Please check your configuration file."
            )
        elif grainstats_run is False:
            LOGGER.error("NodeStats enabled but Grainstats disabled. Please check your configuration file.")
        elif grains_run is False:
            LOGGER.error("NodeStats enabled but Grains disabled. Please check your configuration file.")
        elif filter_run is False:
            LOGGER.error("NodeStats enabled but Filters disabled. Please check your configuration file.")
        else:
            LOGGER.info("Configuration run options are consistent, processing can proceed.")
    elif nodestats_run:
        if disordered_tracing_run is False:
            LOGGER.error("NodeStats enabled but Disordered Tracing disabled. Please check your configuration file.")
        elif grainstats_run is False:
            LOGGER.error("NodeStats enabled but Grainstats disabled. Please check your configuration file.")
        elif grains_run is False:
            LOGGER.error("NodeStats enabled but Grains disabled. Please check your configuration file.")
        elif filter_run is False:
            LOGGER.error("NodeStats enabled but Filters disabled. Please check your configuration file.")
        else:
            LOGGER.info("Configuration run options are consistent, processing can proceed.")
    elif disordered_tracing_run:
        if grainstats_run is False:
            LOGGER.error("Disordered Tracing enabled but Grainstats disabled. Please check your configuration file.")
        elif grains_run is False:
            LOGGER.error("Disordered Tracing enabled but Grains disabled. Please check your configuration file.")
        elif filter_run is False:
            LOGGER.error("Disordered Tracing enabled but Filters disabled. Please check your configuration file.")
        else:
            LOGGER.info("Configuration run options are consistent, processing can proceed.")
    elif grainstats_run:
        if grains_run is False:
            LOGGER.error("Grainstats enabled but Grains disabled. Please check your configuration file.")
        elif filter_run is False:
            LOGGER.error("Grainstats enabled but Filters disabled. Please check your configuration file.")
        else:
            LOGGER.info("Configuration run options are consistent, processing can proceed.")
    elif grains_run:
        if filter_run is False:
            LOGGER.error("Grains enabled but Filters disabled. Please check your configuration file.")
        else:
            LOGGER.info("Configuration run options are consistent, processing can proceed.")
    else:
        LOGGER.info("Configuration run options are consistent, processing can proceed.")


def completion_message(config: dict, img_files: list, summary_config: dict, images_processed: int) -> None:
    """
    Print a completion message summarising images processed.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    img_files : list
        List of found image paths.
    summary_config : dict
        Configuration for plotting summary statistics.
    images_processed : int
        Pandas DataFrame of results.
    """
    if summary_config is not None:
        distribution_plots_message = str(summary_config["output_dir"])
    else:
        distribution_plots_message = "Disabled. Enable in config 'summary_stats/run' if needed."
    print(
        "\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n"
    )
    tprint("TopoStats", font="twisted")
    LOGGER.info(
        f"\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ COMPLETE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n"
        f"  TopoStats Version           : {TOPOSTATS_VERSION}\n"
        f"  TopoStats Commit            : {TOPOSTATS_COMMIT}\n"
        f"  Base Directory              : {config['base_dir']}\n"
        f"  File Extension              : {config['file_ext']}\n"
        f"  Files Found                 : {len(img_files)}\n"
        f"  Successfully Processed^1    : {images_processed} ({(images_processed * 100) / len(img_files)}%)\n"
        f"  All statistics              : {str(config['output_dir'])}/grain_statistics.csv\n"
        f"  Distribution Plots          : {distribution_plots_message}\n\n"
        f"  Configuration               : {config['output_dir']}/config.yaml\n\n"
        f"  Email                       : topostats@sheffield.ac.uk\n"
        f"  Documentation               : https://afm-spm.github.io/topostats/\n"
        f"  Source Code                 : https://github.com/AFM-SPM/TopoStats/\n"
        f"  Bug Reports/Feature Request : https://github.com/AFM-SPM/TopoStats/issues/new/choose\n"
        f"  Citation File Format        : https://github.com/AFM-SPM/TopoStats/blob/main/CITATION.cff\n\n"
        f"  ^1 Successful processing of an image is detection of grains and calculation of at least\n"
        f"     grain statistics. If these have been disabled the percentage will be 0.\n\n"
        f"  If you encounter bugs/issues or have feature requests please report them at the above URL\n"
        f"  or email us.\n\n"
        f"  If you have found TopoStats useful please consider citing it. A Citation File Format is\n"
        f"  linked above and available from the Source Code page.\n"
        f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n"
    )
