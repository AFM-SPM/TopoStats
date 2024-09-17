"""Functions for processing data."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from topostats import __version__
from topostats.filters import Filters
from topostats.grains import Grains
from topostats.grainstats import GrainStats
from topostats.io import get_out_path, save_topostats_file
from topostats.logs.logs import LOGGER_NAME, setup_logger
from topostats.plotting import plot_crossing_linetrace_halfmax
from topostats.plottingfuncs import Images, add_pixel_to_nm_to_plotting_config
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
# pylint: disable=too-many-statements
# pylint: disable=too-many-nested-blocks
# pylint: disable=unnecessary-dict-index-lookup

LOGGER = setup_logger(LOGGER_NAME)


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
        LOGGER.info(f"[{filename}] Image dimensions: {unprocessed_image.shape}")
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
            if plotting_config["image_set"] == "all":
                filter_out_path.mkdir(parents=True, exist_ok=True)
                LOGGER.debug(f"[{filename}] : Target filter directory created : {filter_out_path}")
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
) -> dict | None:
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
            for direction, _ in grains.region_properties.items():
                LOGGER.info(
                    f"[{filename}] : Grains found for direction {direction} : {len(grains.region_properties[direction])}"
                )
                if len(grains.region_properties[direction]) == 0:
                    LOGGER.warning(f"[{filename}] : No grains found for direction {direction}")
        except Exception as e:
            LOGGER.error(f"[{filename}] : An error occurred during grain finding, skipping grainstats and dnatracing.")
            LOGGER.error(f"[{filename}] : The error: {e}")
        else:
            for direction, region_props in grains.region_properties.items():
                if len(region_props) == 0:
                    LOGGER.warning(f"[{filename}] : No grains found for the {direction} direction.")
            # Optionally plot grain finding stage if we have found grains and plotting is required
            if plotting_config["run"]:
                plotting_config.pop("run")
                LOGGER.info(f"[{filename}] : Plotting Grain Finding Images")
                for direction, image_arrays in grains.directions.items():
                    LOGGER.info(f"[{filename}] : Plotting {direction} Grain Finding Images")
                    grain_out_path_direction = grain_out_path / f"{direction}"
                    if plotting_config["image_set"] == "all":
                        grain_out_path_direction.mkdir(parents=True, exist_ok=True)
                        LOGGER.debug(f"[{filename}] : Target grain directory created : {grain_out_path_direction}")
                    for plot_name, array in image_arrays.items():
                        LOGGER.info(f"[{filename}] : Plotting {plot_name} image")
                        plotting_config["plot_dict"][plot_name]["output_dir"] = grain_out_path_direction
                        Images(array, **plotting_config["plot_dict"][plot_name]).plot_and_save()
                    # Make a plot of coloured regions with bounding boxes
                    plotting_config["plot_dict"]["bounding_boxes"]["output_dir"] = grain_out_path_direction
                    Images(
                        grains.directions[direction]["coloured_regions"],
                        **plotting_config["plot_dict"]["bounding_boxes"],
                        region_properties=grains.region_properties[direction],
                    ).plot_and_save()
                    plotting_config["plot_dict"]["coloured_boxes"]["output_dir"] = grain_out_path_direction
                    Images(
                        grains.directions[direction]["labelled_regions_02"],
                        **plotting_config["plot_dict"]["coloured_boxes"],
                        region_properties=grains.region_properties[direction],
                    ).plot_and_save()
                    # Always want mask_overlay (aka "Height Thresholded with Mask") but in core_out_path
                    plot_name = "mask_overlay"
                    plotting_config["plot_dict"][plot_name]["output_dir"] = core_out_path
                    Images(
                        image,
                        filename=f"{filename}_{direction}_masked",
                        masked_array=grains.directions[direction]["removed_small_objects"],
                        **plotting_config["plot_dict"][plot_name],
                    ).plot_and_save()

                plotting_config["run"] = True

            else:
                # Otherwise, return None and warn that plotting is disabled for grain finding images
                LOGGER.info(f"[{filename}] : Plotting disabled for Grain Finding Images")

            grain_masks = {}
            for direction in grains.directions:
                grain_masks[direction] = grains.directions[direction]["labelled_regions_02"]

            return grain_masks

    # Otherwise, return None and warn grainstats is disabled
    LOGGER.info(f"[{filename}] Detection of grains disabled, GrainStats will not be run.")

    return None


def run_grainstats(
    image: npt.NDArray,
    pixel_to_nm_scaling: float,
    grain_masks: dict,
    filename: str,
    basename: Path,
    grainstats_config: dict,
    plotting_config: dict,
    grain_out_path: Path,
):
    """
    Calculate grain statistics for an image and optionally plots the results.

    Parameters
    ----------
    image : npt.NDArray
        2D numpy array image for grain statistics calculations.
    pixel_to_nm_scaling : float
        Scaling factor for converting pixel length scales to nanometres.
        ie the number of pixels per nanometre.
    grain_masks : dict
        Dictionary of grain masks, keys "above" or "below" with values of 2d numpy
        boolean arrays indicating the pixels that have been masked as grains.
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
    pd.DataFrame
        A pandas DataFrame containing the statsistics for each grain. The index is the
        filename and grain number.
    """
    # Calculate statistics if required
    if grainstats_config["run"]:
        grainstats_config.pop("run")
        # Grain Statistics :
        try:
            LOGGER.info(f"[{filename}] : *** Grain Statistics ***")
            grain_plot_dict = {
                key: value
                for key, value in plotting_config["plot_dict"].items()
                if key in ["grain_image", "grain_mask", "grain_mask_image"]
            }
            grainstats_dict = {}
            # There are two layers to process those above the given threshold and those below
            for direction, _ in grain_masks.items():
                # Check if there are grains
                if np.max(grain_masks[direction]) == 0:
                    LOGGER.warning(
                        f"[{filename}] : No grains exist for the {direction} direction. Skipping grainstats for {direction}."
                    )
                    grainstats_dict[direction] = create_empty_dataframe()
                else:
                    grainstats_dict[direction], grains_plot_data = GrainStats(
                        data=image,
                        labelled_data=grain_masks[direction],
                        pixel_to_nanometre_scaling=pixel_to_nm_scaling,
                        direction=direction,
                        base_output_dir=grain_out_path,
                        image_name=filename,
                        plot_opts=grain_plot_dict,
                        **grainstats_config,
                    ).calculate_stats()
                    grainstats_dict[direction]["threshold"] = direction

                    # Plot grains if required
                    if plotting_config["image_set"] == "all":
                        LOGGER.info(f"[{filename}] : Plotting grain images for direction: {direction}.")
                        for plot_data in grains_plot_data:
                            LOGGER.info(
                                f"[{filename}] : Plotting grain image {plot_data['filename']} for direction: {direction}."
                            )
                            Images(
                                data=plot_data["data"],
                                output_dir=plot_data["output_dir"],
                                filename=plot_data["filename"],
                                **plotting_config["plot_dict"][plot_data["name"]],
                            ).plot_and_save()

            # Create results dataframe from above and below results
            # Appease pylint and ensure that grainstats_df is always created
            grainstats_df = create_empty_dataframe()
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

            return grainstats_df

        except Exception:
            LOGGER.info(
                f"[{filename}] : Errors occurred whilst calculating grain statistics. Returning empty dataframe."
            )
            return create_empty_dataframe()
    else:
        LOGGER.info(f"[{filename}] : Calculation of grainstats disabled, returning empty dataframe.")
        return create_empty_dataframe()


def run_disorderedTrace(
    image: npt.NDArray,
    grain_masks: dict,
    pixel_to_nm_scaling: float,
    filename: str,
    basename: str,
    core_out_path: Path,
    tracing_out_path: Path,
    disordered_tracing_config: dict,
    plotting_config: dict,
    results_df: pd.DataFrame = None,
) -> dict:
    """
    Skeletonise and prune grains, adding results to statistics data frames and optionally plot results.

    Parameters
    ----------
    image : npt.ndarray
        Image containing the grains to pass to the tracing function.
    grain_masks : dict
        Dictionary of grain masks, keys "above" or "below" with values of 2D Numpy boolean arrays indicating the pixels
        that have been masked as grains.
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
    results_df : pd.DataFrame, optional
        The grainstats dataframe to bee added to. by default None.

    Returns
    -------
    dict
        Dictionary of "grain_<index>" keys and Nx2 coordinate arrays of the disordered grain trace.
    """
    if disordered_tracing_config["run"]:
        disordered_tracing_config.pop("run")
        LOGGER.info(f"[{filename}] : *** Disordered Tracing ***")
        disordered_traces = defaultdict()
        grainstats_additions_image = pd.DataFrame()
        disordered_tracing_stats_image = pd.DataFrame()
        try:
            # run image using directional grain masks
            for direction, _ in grain_masks.items():
                # Check if there are grains
                if np.max(grain_masks[direction]) == 0:
                    LOGGER.warning(
                        f"[{filename}] : No grains exist for the {direction} direction. Skipping disordered_tracing for {direction}."
                    )
                    raise ValueError(f"No grains exist for the {direction} direction")

                # if grains are found
                (
                    disordered_traces_cropped_data,
                    grainstats_additions_df,
                    disordered_tracing_images,
                    disordered_tracing_stats,
                ) = trace_image_disordered(
                    image=image,
                    grains_mask=grain_masks[direction],
                    filename=filename,
                    pixel_to_nm_scaling=pixel_to_nm_scaling,
                    **disordered_tracing_config,
                )
                # save per image new grainstats stats
                grainstats_additions_df["threshold"] = direction
                grainstats_additions_image = pd.concat([grainstats_additions_image, grainstats_additions_df])

                disordered_tracing_stats["threshold"] = direction
                disordered_tracing_stats["basename"] = basename.parent
                disordered_tracing_stats_image = pd.concat([disordered_tracing_stats_image, disordered_tracing_stats])

                # append direction results to dict
                disordered_traces[direction] = disordered_traces_cropped_data
                # save plots
                Images(
                    image,
                    masked_array=disordered_tracing_images.pop("pruned_skeleton"),
                    output_dir=core_out_path,
                    filename=f"{filename}_{direction}_disordered_trace",
                    **plotting_config["plot_dict"]["pruned_skeleton"],
                ).plot_and_save()
                for plot_name, image_value in disordered_tracing_images.items():
                    Images(
                        image,
                        masked_array=image_value,
                        output_dir=tracing_out_path / direction,
                        **plotting_config["plot_dict"][plot_name],
                    ).plot_and_save()

            # merge grainstats data with other dataframe
            resultant_grainstats = (
                pd.merge(results_df, grainstats_additions_image, on=["image", "threshold", "grain_number"])
                if results_df is not None
                else grainstats_additions_image
            )

            return disordered_traces, resultant_grainstats, disordered_tracing_stats_image

        except Exception:
            LOGGER.info("Disordered tracing failed - skipping.")
            return {}, results_df, None

    else:
        LOGGER.info(f"[{filename}] Calculation of Disordered Tracing disabled, returning empty dictionary.")
        return {}, results_df, None


def run_nodestats(  # noqa: C901
    image: npt.NDArray,
    disordered_tracing_data: dict,
    pixel_to_nm_scaling: float,
    filename: str,
    core_out_path: Path,
    tracing_out_path: Path,
    nodestats_config: dict,
    plotting_config: dict,
    results_df: pd.DataFrame = None,
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
    results_df : pd.DataFrame, optional
        The grainstats dataframe to bee added to. by default None.

    Returns
    -------
    tuple[dict, pd.DataFrame]
        A NodeStats analysis dictionary and grainstats metrics dataframe.
    """
    if nodestats_config["run"]:
        nodestats_config.pop("run")
        LOGGER.info(f"[{filename}] : *** Nodestats ***")
        nodestats_whole_data = defaultdict()
        grainstats_additions_image = pd.DataFrame()
        try:
            # run image using directional grain masks
            for direction, disordered_tracing_direction_data in disordered_tracing_data.items():
                (
                    nodestats_data,
                    grainstats_additions_df,
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
                grainstats_additions_df["threshold"] = direction
                grainstats_additions_image = pd.concat([grainstats_additions_image, grainstats_additions_df])

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
                            if plotting_config["image_set"] == "all":
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
                LOGGER.info(f"[{filename}] : Finished Plotting NodeStats Images")

            # merge grainstats data with other dataframe
            resultant_grainstats = (
                pd.merge(results_df, grainstats_additions_image, on=["image", "threshold", "grain_number"])
                if results_df is not None
                else grainstats_additions_image
            )

            # merge all image dictionaries
            return nodestats_whole_data, resultant_grainstats

        except Exception as e:
            LOGGER.info(f"NodeStats failed with {e} - skipping.")
            return nodestats_whole_data, grainstats_additions_image

    else:
        LOGGER.info(f"[{filename}] : Calculation of nodestats disabled, returning empty dataframe.")
        return None, results_df


def run_ordered_tracing(
    image: npt.NDArray,
    disordered_tracing_data: dict,
    nodestats_data: dict,
    filename: str,
    core_out_path: Path,
    tracing_out_path: Path,
    ordered_tracing_config: dict,
    plotting_config: dict,
    results_df: pd.DataFrame = None,
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
    core_out_path : Path
        Path to save the core ordered tracing image to.
    tracing_out_path : Path
        Path to save optional, diagnostic ordered trace images to.
    ordered_tracing_config : dict
        Dictionary configuration for obtaining an ordered trace representation of the skeletons.
    plotting_config : dict
        Dictionary configuration for plotting images.
    results_df : pd.DataFrame, optional
        The grainstats dataframe to bee added to. by default None.

    Returns
    -------
    tuple[dict, pd.DataFrame]
        A NodeStats analysis dictionary and grainstats metrics dataframe.
    """
    if ordered_tracing_config["run"]:
        ordered_tracing_config.pop("run")
        LOGGER.info(f"[{filename}] : *** Ordered Tracing ***")
        ordered_tracing_image_data = defaultdict()
        grainstats_additions_image = pd.DataFrame()

        try:
            # run image using directional grain masks
            for direction, disordered_tracing_direction_data in disordered_tracing_data.items():
                # Check if there are grains
                if not disordered_tracing_direction_data:
                    LOGGER.warning(
                        f"[{filename}] : No grains exist for the {direction} direction. Skipping ordered_tracing for {direction}."
                    )
                    raise ValueError(f"No grains exist for the {direction} direction")

                # if grains are found
                (
                    ordered_tracing_data,
                    grainstats_additions_df,
                    ordered_tracing_full_images,
                ) = ordered_tracing_image(
                    image=image,
                    disordered_tracing_direction_data=disordered_tracing_direction_data,
                    nodestats_direction_data=nodestats_data[direction],
                    filename=filename,
                    **ordered_tracing_config,
                )

                # save per image new grainstats stats
                grainstats_additions_df["threshold"] = direction
                grainstats_additions_image = pd.concat([grainstats_additions_image, grainstats_additions_df])

                # append direction results to dict
                ordered_tracing_image_data[direction] = ordered_tracing_data

                # save whole image plots
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

                LOGGER.info(f"[{filename}] : Finished Plotting Ordered Tracing Images")

            # merge grainstats data with other dataframe
            resultant_grainstats = (
                pd.merge(results_df, grainstats_additions_image, on=["image", "threshold", "grain_number"])
                if results_df is not None
                else grainstats_additions_image
            )

            # merge all image dictionaries
            return ordered_tracing_image_data, resultant_grainstats

        except Exception as e:
            LOGGER.info(f"Ordered Tracing failed with {e} - skipping.")
            return ordered_tracing_image_data, results_df

    return None, results_df


def run_splining(
    image: npt.NDArray,
    ordered_tracing_data: dict,
    pixel_to_nm_scaling: float,
    filename: str,
    basename: Path,
    core_out_path: Path,
    splining_config: dict,
    plotting_config: dict,
    results_df: pd.DataFrame = None,
) -> tuple:
    """
    Smooth the ordered trace coordinates, adding results to statistics data frames and optionally plot results.

    Parameters
    ----------
    image : npt.ndarray
        Image containing the DNA to pass to the tracing function.
    ordered_tracing_data : dict
        Dictionary of ordered coordinates. Result from "run_ordered_tracing".
    pixel_to_nm_scaling : float
        Scaling factor for converting pixel length scales to nanometers, i.e. the number of pixels per nanometres (nm).
    filename : str
        Name of the image.
    basename : Path
        Path to directory containing the image.
    core_out_path : Path
        Path to save the core ordered tracing image to.
    splining_config : dict
        Dictionary configuration for obtaining an ordered trace representation of the skeletons.
    plotting_config : dict
        Dictionary configuration for plotting images.
    results_df : pd.DataFrame, optional
        The grainstats dataframe to bee added to. by default None.

    Returns
    -------
    tuple[dict, pd.DataFrame]
        A smooth curve analysis dictionary and grainstats metrics dataframe.
    """
    if splining_config["run"]:
        splining_config.pop("run")
        LOGGER.info(f"[{filename}] : *** Splining ***")
        splined_image_data = defaultdict()
        splining_stats = pd.DataFrame()
        image_molstats_df = pd.DataFrame()

        try:
            # run image using directional grain masks
            for direction, ordered_tracing_direction_data in ordered_tracing_data.items():
                if not ordered_tracing_direction_data:
                    LOGGER.warning(
                        f"[{filename}] : No grains exist for the {direction} direction. Skipping disordered_tracing for {direction}."
                    )
                    splining_stats = create_empty_dataframe()
                    image_molstats_df = create_empty_dataframe(columns=["image", "basename", "threshold"])
                    raise ValueError(f"No grains exist for the {direction} direction")

                # if grains are found
                (
                    splined_data,
                    _splining_stats,
                    molstats_df,
                ) = splining_image(
                    image=image,
                    ordered_tracing_direction_data=ordered_tracing_direction_data,
                    filename=filename,
                    pixel_to_nm_scaling=pixel_to_nm_scaling,
                    **splining_config,
                )

                # save per image new grainstats stats
                _splining_stats["threshold"] = direction
                splining_stats = pd.concat([splining_stats, _splining_stats])
                molstats_df["threshold"] = direction
                image_molstats_df = pd.concat([image_molstats_df, molstats_df])

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
                LOGGER.info(f"[{filename}] : Finished Plotting Splining Images")

            # merge grainstats data with other dataframe
            resultant_grainstats = (
                pd.merge(results_df, splining_stats, on=["image", "threshold", "grain_number"])
                if results_df is not None
                else splining_stats
            )
            image_molstats_df["basename"] = basename.parent

            # merge all image dictionaries
            return splined_image_data, resultant_grainstats, image_molstats_df

        except Exception as e:
            LOGGER.info(f"Splining failed with {e} - skipping.")
            return splined_image_data, splining_stats, image_molstats_df

    return None, results_df, create_empty_dataframe(columns=["image", "basename", "threshold"])


def get_out_paths(image_path: Path, base_dir: Path, output_dir: Path, filename: str, plotting_config: dict):
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
    if plotting_config["image_set"] == "all":
        filter_out_path.mkdir(exist_ok=True, parents=True)
        Path.mkdir(grain_out_path / "above", parents=True, exist_ok=True)
        Path.mkdir(grain_out_path / "below", parents=True, exist_ok=True)
        Path.mkdir(tracing_out_path / "above", parents=True, exist_ok=True)
        Path.mkdir(tracing_out_path / "below", parents=True, exist_ok=True)
        Path.mkdir(tracing_out_path / "above" / "nodes", parents=True, exist_ok=True)
        Path.mkdir(tracing_out_path / "below" / "nodes", parents=True, exist_ok=True)

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
    plotting_config: dict,
    output_dir: str | Path = "output",
) -> tuple[dict, pd.DataFrame, dict]:
    """
    Process a single image, filtering, finding grains and calculating their statistics.

    Parameters
    ----------
    topostats_object : dict[str, Union[npt.NDArray, Path, float]]
        A dictionary with keys 'image', 'img_path' and 'px_2_nm' containing a file or frames' image, it's path and it's
        pixel to namometre scaling value.
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
    image_flattened = run_filters(
        unprocessed_image=topostats_object["image_original"],
        pixel_to_nm_scaling=topostats_object["pixel_to_nm_scaling"],
        filename=topostats_object["filename"],
        filter_out_path=filter_out_path,
        core_out_path=core_out_path,
        filter_config=filter_config,
        plotting_config=plotting_config,
    )
    # Use flattened image if one is returned, else use original image
    topostats_object["image_flattened"] = (
        image_flattened if image_flattened is not None else topostats_object["image_original"]
    )

    # Find Grains :
    grain_masks = run_grains(
        image=topostats_object["image_flattened"],
        pixel_to_nm_scaling=topostats_object["pixel_to_nm_scaling"],
        filename=topostats_object["filename"],
        grain_out_path=grain_out_path,
        core_out_path=core_out_path,
        plotting_config=plotting_config,
        grains_config=grains_config,
    )
    # Update grain masks if new grain masks are returned. Else keep old grain masks. Topostats object's "grain_masks"
    # defaults to an empty dictionary so this is safe.
    topostats_object["grain_masks"] = grain_masks if grain_masks is not None else topostats_object["grain_masks"]

    if "above" in topostats_object["grain_masks"].keys() or "below" in topostats_object["grain_masks"].keys():
        # Grainstats :
        results_df = run_grainstats(
            image=topostats_object["image_flattened"],
            pixel_to_nm_scaling=topostats_object["pixel_to_nm_scaling"],
            grain_masks=topostats_object["grain_masks"],
            filename=topostats_object["filename"],
            basename=topostats_object["img_path"],
            grainstats_config=grainstats_config,
            plotting_config=plotting_config,
            grain_out_path=grain_out_path,
        )

        # Disordered Tracing
        disordered_traces_data, results_df, disordered_tracing_stats = run_disorderedTrace(
            image=topostats_object["image_flattened"],
            grain_masks=topostats_object["grain_masks"],
            pixel_to_nm_scaling=topostats_object["pixel_to_nm_scaling"],
            filename=topostats_object["filename"],
            basename=topostats_object["img_path"],
            core_out_path=core_out_path,
            tracing_out_path=tracing_out_path,
            disordered_tracing_config=disordered_tracing_config,
            results_df=results_df,
            plotting_config=plotting_config,
        )
        topostats_object["disordered_traces"] = disordered_traces_data

        # Nodestats
        nodestats, results_df = run_nodestats(
            image=topostats_object["image_flattened"],
            disordered_tracing_data=topostats_object["disordered_traces"],
            pixel_to_nm_scaling=topostats_object["pixel_to_nm_scaling"],
            filename=topostats_object["filename"],
            core_out_path=core_out_path,
            tracing_out_path=tracing_out_path,
            plotting_config=plotting_config,
            nodestats_config=nodestats_config,
            results_df=results_df,
        )
        topostats_object["nodestats"] = nodestats

        # Ordered Tracing
        ordered_tracing, results_df = run_ordered_tracing(
            image=topostats_object["image_flattened"],
            disordered_tracing_data=topostats_object["disordered_traces"],
            nodestats_data=nodestats,
            filename=topostats_object["filename"],
            core_out_path=core_out_path,
            tracing_out_path=tracing_out_path,
            ordered_tracing_config=ordered_tracing_config,
            plotting_config=plotting_config,
            results_df=results_df,
        )
        topostats_object["ordered_traces"] = ordered_tracing

        # splining
        splined_data, results_df, molstats_df = run_splining(
            image=topostats_object["image_flattened"],
            ordered_tracing_data=topostats_object["ordered_traces"],
            pixel_to_nm_scaling=topostats_object["pixel_to_nm_scaling"],
            filename=topostats_object["filename"],
            basename=topostats_object["img_path"],
            core_out_path=core_out_path,
            plotting_config=plotting_config,
            splining_config=splining_config,
            results_df=results_df,
        )

        # Add grain trace data to topostats object
        topostats_object["splining"] = splined_data

    else:
        results_df = create_empty_dataframe()
        molstats_df = create_empty_dataframe()
        disordered_tracing_stats = create_empty_dataframe()

    # Get image statistics
    LOGGER.info(f"[{topostats_object['filename']}] : *** Image Statistics ***")
    # Provide the raw image if image has not been flattened, else provide the flattened image.
    if topostats_object["image_flattened"] is not None:
        image_for_image_stats = topostats_object["image_flattened"]
    else:
        image_for_image_stats = topostats_object["image_original"]
    # Calculate image statistics - returns a dictionary
    image_stats = image_statistics(
        image=image_for_image_stats,
        filename=topostats_object["filename"],
        results_df=results_df,
        pixel_to_nm_scaling=topostats_object["pixel_to_nm_scaling"],
    )

    # Save the topostats dictionary object to .topostats file.
    save_topostats_file(
        output_dir=core_out_path, filename=str(topostats_object["filename"]), topostats_object=topostats_object
    )

    return topostats_object["img_path"], results_df, image_stats, disordered_tracing_stats, molstats_df


def check_run_steps(  # noqa: C901
    filter_run: bool,
    grains_run: bool,
    grainstats_run: bool,
    disordered_tracing_run: bool,
    nodestats_run: bool,
    splining_run: bool,
) -> None:  # noqa: C901
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
    splining_run : bool
        Flag for running DNA Tracing.
    """
    if splining_run:
        if nodestats_run is False:
            LOGGER.error("Splining enabled but NodeStats is disabled. Tracing will use the 'old' method.")
        elif disordered_tracing_run is False:
            LOGGER.error("Splining enabled but Disordered Tracing is disabled. Please check your configuration file.")
        elif grainstats_run is False:
            LOGGER.error("Tracing enabled but Grainstats disabled. Please check your configuration file.")
        elif grains_run is False:
            LOGGER.error("Tracing enabled but Grains disabled. Please check your configuration file.")
        elif filter_run is False:
            LOGGER.error("Tracing enabled but Filters disabled. Please check your configuration file.")
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
    img_files : list()
        List of found image paths.
    summary_config : dict(
        Configuration for plotting summary statistics.
    images_processed : int
        Pandas DataFrame of results.
    """
    if summary_config is not None:
        distribution_plots_message = str(summary_config["output_dir"])
    else:
        distribution_plots_message = "Disabled. Enable in config 'summary_stats/run' if needed."
    LOGGER.info(
        f"\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ COMPLETE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n"
        f"  TopoStats Version           : {__version__}\n"
        f"  Base Directory              : {config['base_dir']}\n"
        f"  File Extension              : {config['file_ext']}\n"
        f"  Files Found                 : {len(img_files)}\n"
        f"  Successfully Processed^1    : {images_processed} ({(images_processed * 100) / len(img_files)}%)\n"
        f"  All statistics              : {str(config['output_dir'])}/all_statistics.csv\n"
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
        f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n"
    )
