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
from topostats.tracing.dnatracing import dnatrace_image
from topostats.tracing.nodestats import nodestats_image
from topostats.utils import create_empty_dataframe

# pylint: disable=broad-except
# pylint: disable=line-too-long
# pylint: disable=too-many-arguments
# pylint: disable=too-many-branches
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
    LOGGER.info(f"[{filename}] Detection of grains disabled, returning empty data frame.")

    return None


def run_grainstats(
    image: npt.NDArray,
    pixel_to_nm_scaling: float,
    grain_masks: dict,
    filename: str,
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
    core_out_path: Path,
    tracing_out_path: Path,
    disordered_tracing_config: dict,
    plotting_config: dict,
) -> dict:
    """
    Trace DNA molecule for the supplied grains adding results to statistics data frames and optionally plot results.

    Parameters
    ----------
    image : npt.ndarray
        Image containing the DNA to pass to the tracing function.
    grain_masks : dict
        Dictionary of grain masks, keys "above" or "below" with values of 2D Numpy boolean arrays indicating the pixels
        that have been masked as grains.
    pixel_to_nm_scaling : float
        Scaling factor for converting pixel length scales to nanometers, i.e. the number of pixesl per nanometres (nm).
    filename : str
        Name of the image.
    tracing_out_path : Path
        Dictionary to save optional DNA tracing visual information to.
    disordered_tracing_config : dict
        Dictionary configuration for obtaining a disordered trace representation of the grains.
    plotting_config : dict
        Dictionary configuration for plotting images.

    Returns
    -------
    dict
        Dictionary of "grain_<index>" keys and Nx2 coordinate arrays of the disordered grain trace.
    """
    if disordered_tracing_config["run"]:
        disordered_tracing_config.pop("run")
        LOGGER.info(f"[{filename}] : *** Disordered Tracing ***")
        disordered_traces = defaultdict()
        try:
            # run image using directional grain masks
            for direction, _ in grain_masks.items():
                disordered_traces_cropped_data, disordered_tracing_images = trace_image_disordered(
                    image=image,
                    grains_mask=grain_masks[direction],
                    filename=filename,
                    pixel_to_nm_scaling=pixel_to_nm_scaling,
                    **disordered_tracing_config,
                )
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

            return disordered_traces

        except Exception:
            LOGGER.info("Disordered tracing failed - skipping.")
            return disordered_traces


def run_nodestats(
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

    if nodestats_config["run"]:
        nodestats_config.pop("run")
        LOGGER.info(f"[{filename}] : *** Nodestats ***")
        nodestats_image_data = defaultdict()
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
                nodestats_image_data[direction] = nodestats_data

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

                # plot sinlge node images
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
                LOGGER.info(f"[{filename}] : Finished Plotting DNA Tracing Images")

            # merge grainstats data with other dataframe
            resultant_grainstats = (
                pd.merge(results_df, grainstats_additions_image, on=["image", "threshold", "grain_number"])
                if results_df is not None
                else grainstats_additions_image
            )

            # merge all image dictionaries
            return nodestats_image_data, resultant_grainstats

        except Exception as e:
            LOGGER.info(f"NodeStats failed with {e} - skipping.")
            return nodestats_image_data, resultant_grainstats


# noqa: C901
def run_dnatracing(
    image: npt.NDArray,
    grain_masks: dict,
    pixel_to_nm_scaling: float,
    image_path: Path,
    filename: str,
    core_out_path: Path,
    tracing_out_path: Path,
    dnatracing_config: dict,
    plotting_config: dict,
    results_df: pd.DataFrame = None,
) -> tuple[dict, pd.DataFrame]:
    """
    Trace DNA molecule for the supplied grains adding results to statistics data frames and optionally plot results.

    Image containing the DNA to pass to the tracing function.

    Parameters
    ----------
    image : npt.NDArray
        Dictionary of grain masks, keys "above" or "below" with values of 2D Numpy boolean arrays indicating the pixels
        that have been masked as grains.
    grain_masks : dict
        Scaling factor for converting pixel length scales to nanometers, i.e. the number of pixesl per nanometres (nm).
    pixel_to_nm_scaling : float
        Pat to the original image file (used for DataFrame indexing).
    image_path : Path
        Name of the image.
    filename : str
        General output directory for outputs such as the grain statistics dataframe.
    core_out_path : Path
        Directory to save optional DNA tracing visual information to.
    tracing_out_path : Path
        Dictionary to save optional DNA tracing visual information to.
    dnatracing_config : dict
        Dictionary configuration for the DNA tracing.
    plotting_config : dict
        Dictionary configuration for plotting images.
    results_df : pd.DataFrame
        Pandas DataFrame containing grain statistics.

    Returns
    -------
    tuple[dict, pd.DataFrame]
        Dictionary of results and Pandas DataFrame containing grain statistics and dna tracing statistics. Keys are file
        path and molecule number.
    """
    # Create empty dataframe is none is passed
    if results_df is None:
        results_df = create_empty_dataframe()

    # Run dnatracing
    # try:
    grain_trace_data = None
    if dnatracing_config["run"]:
        dnatracing_config.pop("run")
        LOGGER.info(f"[{filename}] : *** DNA Tracing ***")
        tracing_stats = defaultdict()
        grain_trace_data = defaultdict()
        for direction, _ in grain_masks.items():
            tracing_results = dnatrace_image(
                image=image,
                grains_mask=grain_masks[direction],
                filename=filename,
                pixel_to_nm_scaling=pixel_to_nm_scaling,
                **dnatracing_config,
            )
            tracing_stats[direction] = tracing_results["grain_statistics"]
            tracing_stats[direction]["threshold"] = direction
            ordered_traces = tracing_results["all_ordered_traces"]
            cropped_images: dict[int, npt.NDArray] = tracing_results["cropped_images"]

            grain_trace_data[direction] = {
                "cropped_images": cropped_images,
                "ordered_traces": tracing_results["all_ordered_traces"],
                "splined_traces": tracing_results["all_splined_traces"],
                "ordered_trace_heights": tracing_results["all_ordered_trace_heights"],
                "ordered_trace_cumulative_distances": tracing_results["all_ordered_trace_cumulative_distances"],
                "tracingstats": tracing_results["dnatracing_statistics"],
            }

            # Plot traces for the whole image
            Images(
                image,
                output_dir=core_out_path,
                filename=f"{filename}_{direction}_traced",
                plot_coords=tracing_results["splined_traces_image_frame"],
                **plotting_config["plot_dict"]["all_molecule_traces"],
            ).plot_and_save()

            # Plot traces on each grain individually
            for grain_index, mol_dict in enumerate(ordered_traces.values()):
                Images(
                    cropped_images[grain_index],
                    output_dir=tracing_out_path / direction,
                    filename=f"{filename}_grain_trace_{grain_index}",
                    plot_coords=list(mol_dict.values()),
                    **plotting_config["plot_dict"]["single_molecule_trace"],
                ).plot_and_save()

            plot_names = {
                "visual": tracing_results["all_images"]["visual"],
                "ordered_trace": tracing_results["all_images"]["ordered_traces"],
                "fitted_trace": tracing_results["all_images"]["fitted_traces"],
            }
            for plot_name, image_value in plot_names.items():
                Images(
                    image,
                    masked_array=image_value,
                    output_dir=tracing_out_path / direction,
                    **plotting_config["plot_dict"][plot_name],
                ).plot_and_save()

        # Set create tracing_stats_df from above and below results
        if "above" in tracing_stats and "below" in tracing_stats:
            tracing_stats_df = pd.concat([tracing_stats["below"], tracing_stats["above"]])
        elif "above" in tracing_stats:
            tracing_stats_df = tracing_stats["above"]
        elif "below" in tracing_stats:
            tracing_stats_df = tracing_stats["below"]
        LOGGER.info(f"[{filename}] : Combining {list(tracing_stats.keys())} grain statistics and dnatracing statistics")
        # NB - Merge on image, molecule and threshold because we may have above and below molecules which
        #      gives duplicate molecule numbers as they are processed separately, if tracing stats
        #      are not available (because skeleton was too small), grainstats are still retained.
        results = results_df.merge(tracing_stats_df, on=["image", "threshold", "grain_number"], how="left")
        results["basename"] = image_path.parent

        return results, grain_trace_data

    # Otherwise, return the passed in dataframe and warn that tracing is disabled
    LOGGER.info(f"[{filename}] Calculation of DNA Tracing disabled, returning grainstats data frame.")
    results = results_df
    results["basename"] = image_path.parent

    return results, grain_trace_data
    # except Exception:
    #     # If no results we need a dummy dataframe to return.
    #     LOGGER.warning(
    #         f"[{filename}] : Errors occurred whilst calculating DNA tracing statistics, " "returning grain statistics"
    #     )
    #     results = results_df
    #     results["basename"] = image_path.parent
    #     grain_trace_data = None
    #     return results, grain_trace_data


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
    dnatracing_config: dict,
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
    dnatracing_config : dict
        Dictionary of configuration options for running the DNA Tracing stage.
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
            grainstats_config=grainstats_config,
            plotting_config=plotting_config,
            grain_out_path=grain_out_path,
        )

        # Disordered Tracing
        disordered_traces = run_disorderedTrace(
            image=topostats_object["image_flattened"],
            grain_masks=topostats_object["grain_masks"],
            pixel_to_nm_scaling=topostats_object["pixel_to_nm_scaling"],
            filename=topostats_object["filename"],
            core_out_path=core_out_path,
            tracing_out_path=tracing_out_path,
            disordered_tracing_config=disordered_tracing_config,
            plotting_config=plotting_config,
        )
        topostats_object["disordered_traces"] = (
            disordered_traces if disordered_traces is not None else topostats_object["grain_masks"]
        )

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

        # DNAtracing
        results_df, grain_trace_data = run_dnatracing(
            image=topostats_object["image_flattened"],
            grain_masks=topostats_object["grain_masks"],
            pixel_to_nm_scaling=topostats_object["pixel_to_nm_scaling"],
            filename=topostats_object["filename"],
            core_out_path=core_out_path,
            tracing_out_path=tracing_out_path,
            image_path=topostats_object["img_path"],
            plotting_config=plotting_config,
            dnatracing_config=dnatracing_config,
            results_df=results_df,
        )

        # Add grain trace data to topostats object
        # topostats_object["grain_trace_data"] = grain_trace_data

    else:
        results_df = create_empty_dataframe()

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

    return topostats_object["img_path"], results_df, image_stats


def check_run_steps(
    filter_run: bool,
    grains_run: bool,
    grainstats_run: bool,
    disordered_tracing_run: bool,
    nodestats_run: bool,
    dnatracing_run: bool,
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
    dnatracing_run : bool
        Flag for running DNA Tracing.
    """
    if dnatracing_run:
        if nodestats_run is False:
            LOGGER.error("DNA tracing enabled but NodeStats is disabled. Tracing will use the 'old' method.")
        elif disordered_tracing_run is False:
            LOGGER.error(
                "DNA tracing enabled but Disordered Tracing is disabled. Please check your configuration file."
            )
        elif grainstats_run is False:
            LOGGER.error("DNA tracing enabled but Grainstats disabled. Please check your configuration file.")
        elif grains_run is False:
            LOGGER.error("DNA tracing enabled but Grains disabled. Please check your configuration file.")
        elif filter_run is False:
            LOGGER.error("DNA tracing enabled but Filters disabled. Please check your configuration file.")
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
