"""Functions for processing data."""

import logging
from copy import deepcopy
from pathlib import Path

import numpy as np

# import pandas as pd
from art import tprint

from topostats import TOPOSTATS_BASE_VERSION, TOPOSTATS_COMMIT
from topostats.array_manipulation import re_crop_grain_image_and_mask_to_set_size_nm
from topostats.classes import TopoStats
from topostats.filters import Filters
from topostats.grains import Grains
from topostats.grainstats import GrainStats
from topostats.io import get_out_path, save_topostats_file
from topostats.logs.logs import LOGGER_NAME
from topostats.measure.curvature import calculate_curvature_stats_image
from topostats.plotting import plot_crossing_linetrace_halfmax
from topostats.plottingfuncs import (
    Images,
    add_pixel_to_nm_to_plotting_config,
)
from topostats.tracing.disordered_tracing import trace_image_disordered
from topostats.tracing.nodestats import nodestats_image
from topostats.tracing.ordered_tracing import ordered_tracing_image
from topostats.tracing.splining import splining_image

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


def run_filters(  # noqa: C901
    topostats_object: TopoStats,
    filter_out_path: Path,
    core_out_path: Path,
    filter_config: dict | None = None,
    plotting_config: dict | None = None,
) -> None:
    """
    Filter and flatten an image. Optionally plots the results, returning the flattened image.

    Parameters
    ----------
    topostats_object : TopoStats
        TopoStats object to be filtered.
    filter_out_path : Path
        Output directory for step-by-step flattening plots.
    core_out_path : Path
        General output directory for outputs such as the flattened image.
    filter_config : dict
        Dictionary of configuration for the Filters class to use when initialised.
    plotting_config : dict
        Dictionary of configuration for plotting output images.
    """
    filter_config = deepcopy(topostats_object.config["filter"]) if filter_config is None else deepcopy(filter_config)
    plotting_config = (
        deepcopy(topostats_object.config["plotting"]) if plotting_config is None else deepcopy(plotting_config)
    )
    filter_out_path = (
        core_out_path / f"{topostats_object.filename}" / "filters" if filter_out_path is None else filter_out_path
    )
    if filter_config["run"]:
        filter_config.pop("run")
        try:
            LOGGER.debug(f"[{topostats_object.filename}] Image dimensions: {topostats_object.image_original.shape}")
            LOGGER.info(f"[{topostats_object.filename}] : *** Filtering ***")
            filters = Filters(
                topostats_object=topostats_object,
                **filter_config,
            )
            filters.filter_image()
            LOGGER.info(f"[{topostats_object.filename}] : Filters stage completed successfully.")
        except Exception as e:
            LOGGER.error(
                f"[{topostats_object.filename}] : An error occurred during filtering. Skipping subsequent steps.",
                exc_info=e,
            )
        # Optionally plot filter stage
        else:
            if plotting_config["run"]:
                try:
                    plotting_config.pop("run")
                    Path(filter_out_path).mkdir(parents=True, exist_ok=True)
                    # Generate plots
                    for plot_name, array in filters.images.items():
                        if plot_name not in ["scan_raw"]:
                            if plot_name == "extracted_channel":
                                array = np.flipud(array.pixels)
                            plotting_config["plot_dict"][plot_name]["output_dir"] = (
                                core_out_path
                                if plotting_config["plot_dict"][plot_name]["core_set"]
                                else filter_out_path
                            )
                            try:
                                # ns-rse 2025-12-03 Could perhaps move logic for plotting here rather incurring cost of
                                # instantiating only to find the given plot is not required.
                                LOGGER.debug(
                                    f"[{topostats_object.filename}] [run_filter] : Plotting array : {plot_name=}"
                                )
                                Images(array, **plotting_config["plot_dict"][plot_name]).plot_and_save()
                                Images(array, **plotting_config["plot_dict"][plot_name]).plot_histogram_and_save()
                            except AttributeError:
                                # If scar removal isn't run scar_mask plot always fails with Attribute error, only log
                                # other failures
                                if plot_name != "scar_mask" or topostats_object.config["filter"]["remove_scars"]["run"]:
                                    LOGGER.info(f"[{topostats_object.filename}] Unable to generate plot : {plot_name}")
                                else:
                                    continue
                    # Always want the 'z_threshed' plot (aka "Height Thresholded") but in the core_out_path
                    plotting_config["plot_dict"]["z_threshed"]["output_dir"] = core_out_path
                    Images(
                        topostats_object.image,
                        filename=topostats_object.filename,
                        **plotting_config["plot_dict"]["z_threshed"],
                    ).plot_and_save()
                    LOGGER.info(f"[{topostats_object.filename}] : Filters plotting completed successfully.")
                except Exception as e:
                    LOGGER.error(
                        f"[{topostats_object.filename}] : Plotting filtering failed. Consider raising an issue on"
                        "GitHub. Error : ",
                        exc_info=e,
                    )
        return
    # Otherwise, return None and warn that initial processing is disabled.
    LOGGER.error(
        "Your configuration disables running the initial filter stage. This is required for all subsequent "
        "stages of processing. Please correct your configuration file."
    )
    return


def run_grains(  # noqa: C901
    topostats_object: TopoStats,
    grain_out_path: Path | None,
    core_out_path: Path,
    plotting_config: dict | None = None,
    grains_config: dict | None = None,
) -> None:
    """
    Identify grains (molecules) and optionally plots the results.

    Parameters
    ----------
    topostats_object : TopoStats
        TopoStats object for grain detection.
    grain_out_path : Path
        Output path for step-by-step grain finding plots.
    core_out_path : Path
        General output directory for outputs such as the flattened image with grain masks overlaid.
    plotting_config : dict
        Dictionary of configuration for plotting images.
    grains_config : dict
        Dictionary of configuration for the Grains class to use when initialised.
    """
    grains_config = topostats_object.config["grains"].copy() if grains_config is None else grains_config
    plotting_config = (
        deepcopy(topostats_object.config["plotting"]) if plotting_config is None else deepcopy(plotting_config)
    )
    grain_out_path = (
        core_out_path / f"{topostats_object.filename}" / "grains" if grain_out_path is None else grain_out_path
    )
    if grains_config["run"]:
        grains_config.pop("run")
        try:
            LOGGER.info(f"[{topostats_object.filename}] : *** Grain Finding ***")
            grains = Grains(
                topostats_object=topostats_object,
                **grains_config,
            )
            grains.find_grains()
            LOGGER.info(
                f"[{topostats_object.filename}] : Grain Finding stage completed."
                f"Grains found : {len(topostats_object.grain_crops)}"
            )
            if len(topostats_object.grain_crops) == 0:
                LOGGER.warning(f"[{topostats_object.filename}] : No grains found.")
        except Exception as e:
            LOGGER.error(
                f"[{topostats_object.filename}] : An error occurred during grain finding, skipping following steps.",
                exc_info=e,
            )
        else:
            if plotting_config["run"]:
                try:
                    # Optionally plot grain finding stage if we have found grains and plotting is required
                    plotting_config.pop("run")
                    grain_out_path.mkdir(parents=True, exist_ok=True)
                    grain_crop_plot_size_nm = plotting_config["grain_crop_plot_size_nm"]
                    # @ns-rse : 2025-10-30 Need to think through carefully what this becomes and which directory things are
                    # to be in as we no longer have a direction and should be using topostats_object.grain_crops
                    for _, image_arrays in grains.mask_images.items():
                        LOGGER.debug(f"[{topostats_object.filename}] : Plotting Grain Diagnostic Images")
                        # Plot diagnostic full grain images
                        for plot_name, array in image_arrays.items():
                            # Tensor, iterate over each channel
                            filename_base = plotting_config["plot_dict"][plot_name]["filename"]
                            for tensor_class in range(1, array.shape[2]):
                                LOGGER.info(
                                    f"[{topostats_object.filename}] : Plotting {plot_name} image, class {tensor_class}"
                                )
                                plotting_config["plot_dict"][plot_name]["output_dir"] = grain_out_path
                                plotting_config["plot_dict"][plot_name]["filename"] = (
                                    filename_base + f"_class_{tensor_class}"
                                )
                                Images(
                                    data=topostats_object.image,
                                    masked_array=array[:, :, tensor_class],
                                    **plotting_config["plot_dict"][plot_name],
                                ).plot_and_save()
                        # Plot individual grain masks
                        if topostats_object.grain_crops not in ({}, None):
                            LOGGER.info(f"[{topostats_object.filename}] : Plotting individual grain masks")
                            for grain_number, grain_crop in topostats_object.grain_crops.items():
                                # If the grain_crop_plot_size_nm is -1, just use the grain crop as-is.
                                if grain_crop_plot_size_nm == -1:
                                    crop_image = grain_crop.image
                                    crop_mask = grain_crop.mask
                                else:
                                    try:
                                        LOGGER.info(
                                            f"[{topostats_object.filename}] : Resizing grain crop {grain_number}"
                                        )
                                        crop_image, crop_mask = re_crop_grain_image_and_mask_to_set_size_nm(
                                            filename=topostats_object.filename,
                                            grain_number=grain_number,
                                            grain_bbox=grain_crop.bbox,
                                            pixel_to_nm_scaling=topostats_object.pixel_to_nm_scaling,
                                            full_image=topostats_object.image,
                                            full_mask_tensor=topostats_object.full_mask_tensor,
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
                                plotting_config["plot_dict"]["grain_image"]["filename"] = (
                                    f"{topostats_object.filename}_grain_{grain_number}"
                                )
                                plotting_config["plot_dict"]["grain_image"]["output_dir"] = grain_out_path
                                Images(
                                    data=crop_image,
                                    **plotting_config["plot_dict"]["grain_image"],
                                ).plot_and_save()
                                # Plot the grain crop with mask
                                plotting_config["plot_dict"]["grain_mask"]["output_dir"] = grain_out_path
                                # Tensor, iterate over channels
                                for tensor_class in range(1, crop_mask.shape[2]):
                                    plotting_config["plot_dict"]["grain_mask"]["filename"] = (
                                        f"{topostats_object.filename}_grain_mask_{grain_number}_class_{tensor_class}"
                                    )
                                    Images(
                                        data=crop_image,
                                        masked_array=crop_mask[:, :, tensor_class],
                                        **plotting_config["plot_dict"]["grain_mask"],
                                    ).plot_and_save()
                        # Make a plot of labelled regions with bounding boxes
                        if topostats_object.grain_crops is not None:
                            # Plot image with overlaid masks
                            plot_name = "mask_overlay"
                            plotting_config["plot_dict"][plot_name]["output_dir"] = core_out_path
                            # Tensor, iterate over each channel
                            for tensor_class in range(1, topostats_object.full_mask_tensor.shape[2]):
                                # Set filename for this class
                                plotting_config["plot_dict"][plot_name]["filename"] = (
                                    f"{topostats_object.filename}_{grain_number}_masked_overlay_class_{tensor_class}"
                                )
                                full_mask_tensor_class = topostats_object.full_mask_tensor[:, :, tensor_class]
                                full_mask_tensor_class_regionprops = Grains.get_region_properties(
                                    Grains.label_regions(full_mask_tensor_class)
                                )
                                Images(
                                    data=topostats_object.image,
                                    masked_array=full_mask_tensor_class.astype(bool),
                                    **plotting_config["plot_dict"][plot_name],
                                    region_properties=full_mask_tensor_class_regionprops,
                                ).plot_and_save()
                    LOGGER.info(f"[{topostats_object.filename}] : Grain plotting completed successfully.")
                except Exception as e:
                    LOGGER.error(
                        f"[{topostats_object.filename}] : Plotting grains failed. Consider raising an issue on"
                        "GitHub. Error",
                        exc_info=e,
                    )
            else:
                LOGGER.info(f"[{topostats_object.filename}] : Plotting disabled for Grain Finding Images")
        return
    # Otherwise, return None and warn grainstats is disabled
    LOGGER.info(f"[{topostats_object.filename}] Detection of grains disabled, GrainStats will not be run.")
    return


def run_grainstats(
    topostats_object: TopoStats,
    grain_out_path: Path | None,
    core_out_path: Path,
    grainstats_config: dict | None = None,
    plotting_config: dict | None = None,
) -> None:
    """
    Calculate grain statistics for an image and optionally plots the results.

    Parameters
    ----------
    topostats_object : TopoStats
        TopoStats object post grain detection for which statistics are to be calculated.
    grain_out_path : Path
        Directory to save optional grain statistics visual information to.
    core_out_path : Path
        General output directory for outputs such as the flattened image with grain masks overlaid.
    grainstats_config : dict, optional
        Dictionary of configuration for the GrainStats class to be used when initialised.
    plotting_config : dict, optional
        Dictionary of configuration for plotting images.
    """
    grainstats_config = topostats_object.config["grainstats"].copy() if grainstats_config is None else grainstats_config
    plotting_config = (
        deepcopy(topostats_object.config["plotting"]) if plotting_config is None else deepcopy(plotting_config)
    )
    grain_out_path = (
        core_out_path / f"{topostats_object.filename}" / "grains" if grain_out_path is None else grain_out_path
    )
    if grainstats_config["run"]:
        grainstats_config.pop("run")
        # ns-rse 2025-12-03 : Pop the `class_names`, not used by GrainStats why are these part of the
        # "grainstats_config"? (Must be popped from `grainstats_config` though as not arg to GrainStats)
        _ = {index + 1: class_name for index, class_name in enumerate(grainstats_config.pop("class_names"))}
        # Grain Statistics :
        try:
            LOGGER.info(f"[{topostats_object.filename}] : *** Grain Statistics ***")
            grain_plot_dict = {
                key: value
                for key, value in plotting_config["plot_dict"].items()
                if key in ["grain_image", "grain_mask", "grain_mask_image"]
            }
            GrainStats(
                topostats_object=topostats_object,
                base_output_dir=grain_out_path,
                plot_opts=grain_plot_dict,
                **grainstats_config,
            )
            LOGGER.info(
                f"[{topostats_object.filename}] : Calculated grainstats for {len(topostats_object.grain_crops)} grains."
            )
            LOGGER.info(f"[{topostats_object.filename}] : Grainstats stage completed successfully.")
            return
        except Exception as e:
            LOGGER.info(
                f"[{topostats_object.filename}] : Errors occurred whilst calculating grain statistics. Returning empty dataframe.",
                exc_info=e,
            )
            return
    LOGGER.info(f"[{topostats_object.filename}] : Calculation of grainstats disabled.")
    return


def run_disordered_tracing(  # noqa: C901
    topostats_object: TopoStats,
    core_out_path: Path,
    tracing_out_path: Path,  # pylint: disable=unused-argument
    disordered_tracing_config: dict | None = None,
    plotting_config: dict | None = None,
) -> None:
    """
    Skeletonise and prune grains, adding results to statistics data frames and optionally plot results.

    Parameters
    ----------
    topostats_object : TopoStats
        TopoStats object for processing, should have had grain detection performed prior to disordered tracing otherwise
        there are no grains to trace.
    core_out_path : Path
        Path to save the core disordered trace image to.
    tracing_out_path : Path
        Path to save the optional, diagnostic disordered trace images to.
    disordered_tracing_config : dict
        Dictionary configuration for obtaining a disordered trace representation of the grains.
    plotting_config : dict
        Dictionary configuration for plotting images.
    """
    disordered_tracing_config = (
        topostats_object.config["disordered_tracing"]
        if disordered_tracing_config is None
        else disordered_tracing_config
    )
    plotting_config = (
        deepcopy(topostats_object.config["plotting"]) if plotting_config is None else deepcopy(plotting_config)
    )
    tracing_out_path = (
        core_out_path / f"{topostats_object.filename}" / "dnatracing" if tracing_out_path is None else tracing_out_path
    )
    if disordered_tracing_config["run"]:
        disordered_tracing_config.pop("run")
        LOGGER.info(f"[{topostats_object.filename}] : *** Disordered Tracing ***")
        if topostats_object.grain_crops is None:
            LOGGER.warning(f"[{topostats_object.filename}] : No grains exist. Skipping disordered tracing.")
            return
        try:
            trace_image_disordered(
                topostats_object=topostats_object,
                **disordered_tracing_config,
            )
            LOGGER.info(f"[{topostats_object.filename}] : Disordered Tracing stage completed successfully.")
        except ValueError as e:
            LOGGER.info(f"[{topostats_object.filename}] : Disordered tracing failed with ValueError {e}")
        except AttributeError as e:
            if topostats_object.image_grain_stats is None:
                LOGGER.info(f"[{topostats_object.filename}] : Missing image_grain_stats attribute.")
            else:
                LOGGER.info(f"[{topostats_object.filename}] : Disordered tracing failed with AttributeError {e}")
        except Exception as e:
            LOGGER.info(
                f"[{topostats_object.filename}] : Disordered tracing failed - skipping. Consider raising an issue on GitHub. Error: ",
                exc_info=e,
            )
        # Plot results
        else:
            try:
                tracing_out_path.mkdir(parents=True, exist_ok=True)
                for grain_number, grain_crop in topostats_object.grain_crops.items():
                    # Plot pruned skeletons
                    LOGGER.info(
                        f"[{topostats_object.filename}] : Plotting disordered traces for grain {grain_number + 1}."
                    )
                    Images(
                        data=grain_crop.image,
                        masked_array=grain_crop.disordered_trace.images["pruned_skeleton"],
                        output_dir=tracing_out_path,
                        filename=f"{topostats_object.filename}_grain_{grain_number}_disordered_trace",
                        **plotting_config["plot_dict"]["pruned_skeleton"],
                    ).plot_and_save()
                    # Plot other disordered tracing stages (skeleton, branch_types and branch_indexes)
                    for plot_name, image_value in grain_crop.disordered_trace.images.items():
                        # Skip plotting the image and grain themselves and pruned_skeleton (plotted above)
                        if plot_name in {"image", "grain", "pruned_skeleton"}:
                            continue
                        try:
                            # ns-rse 2025-12-04 : fudge to get filenames consistent
                            config_filename = plotting_config["plot_dict"][plot_name].pop("filename")
                            filename = f"{topostats_object.filename}_grain_{grain_number}_" + config_filename[3:]
                            Images(
                                data=grain_crop.image,
                                masked_array=image_value,
                                output_dir=tracing_out_path,
                                filename=filename,
                                **plotting_config["plot_dict"][plot_name],
                            ).plot_and_save()
                            plotting_config["plot_dict"][plot_name]["filename"] = config_filename
                        except KeyError:
                            LOGGER.warning(
                                f"[{topostats_object.filename}] : !!! No configuration to plot `{plot_name}` !!!\n\n "
                                "If you  are NOT using a custom plotting configuration then please raise an issue on"
                                "GitHub to report this problem."
                            )
                # ns-rse 2025-12-10 : Reconsider if these plots are useful, if they are we need a configurable way of
                # plotting them (which requires distinguishing them from those generated during ordered tracing)
                for plot_name in ["smoothed_mask", "skeleton", "branch_indexes", "branch_types"]:
                    Images(
                        data=topostats_object.image,
                        masked_array=topostats_object.full_image_plots[plot_name],
                        output_dir=tracing_out_path,  # / direction,
                        **plotting_config["plot_dict"][plot_name],
                    ).plot_and_save()

                LOGGER.info(f"[{topostats_object.filename}] : Disordered trace plotting completed successfully.")
            except Exception as e:
                LOGGER.error(
                    f"[{topostats_object.filename}] : Plotting disordered traces failed. Consider raising an issue on"
                    "GitHub. Error : ",
                    exc_info=e,
                )
        return
    LOGGER.info(f"[{topostats_object.filename}] Disordered Tracing disabled.")
    return


def run_nodestats(  # noqa: C901
    topostats_object: TopoStats,
    core_out_path: Path,  # pylint: disable=unused-argument
    tracing_out_path: Path,  # pylint: disable=unused-argument
    nodestats_config: dict | None = None,
    plotting_config: dict | None = None,
) -> None:
    """
    Analyse crossing points in grains adding results to statistics data frames and optionally plot results.

    Parameters
    ----------
    topostats_object : TopoStats
        TopoStats object for processing, should have had disordered tracing performed first.
    core_out_path : Path
        Path to save the core NodeStats image to.
    tracing_out_path : Path
        Path to save optional, diagnostic NodeStats images to.
    nodestats_config : dict
        Dictionary configuration for analysing the crossing points.
    plotting_config : dict
        Dictionary configuration for plotting images.
    """
    nodestats_config = topostats_object.config["nodestats"] if nodestats_config is None else nodestats_config
    plotting_config = (
        deepcopy(topostats_object.config["plotting"]) if plotting_config is None else deepcopy(plotting_config)
    )
    tracing_out_path = (
        core_out_path / f"{topostats_object.filename}" / "dnatracing" / "nodes"
        if tracing_out_path is None
        else tracing_out_path / "nodes"
    )
    if nodestats_config["run"]:
        nodestats_config.pop("run")
        LOGGER.info(f"[{topostats_object.filename}] : *** Nodestats ***")
        if topostats_object.grain_crops is None:
            LOGGER.warning(f"[{topostats_object.filename}] : No grains exist. Skipping nodestats tracing.")
            return
        try:
            nodestats_image(
                topostats_object=topostats_object,
                **nodestats_config,
            )
            LOGGER.info(f"[{topostats_object.filename}] : NodeStats stage completed successfully.")
        except UnboundLocalError as e:
            LOGGER.info(
                f"[{topostats_object.filename}] : NodeStats failed with UnboundLocalError {e} - all skeletons pruned in the Disordered Tracing step."
            )
        except KeyError as e:
            LOGGER.info(
                f"[{topostats_object.filename}] : NodeStats failed with KeyError {e} - no skeletons found from the Disordered Tracing step."
            )
        except Exception as e:
            LOGGER.info(
                f"[{topostats_object.filename}] : NodeStats failed - skipping. Consider raising an issue on GitHub. Error: ",
                exc_info=e,
            )
        else:
            try:
                tracing_out_path.mkdir(parents=True, exist_ok=True)
                # For each node within each grain make three plots
                for grain_number, grain_crop in topostats_object.grain_crops.items():
                    for node_number, node in grain_crop.nodes.items():
                        LOGGER.info(
                            f"[{topostats_object.filename}] : Plotting Nodestats Grain {grain_number + 1} (Node {node_number})"
                        )
                        Images(
                            data=grain_crop.image,
                            masked_array=node.node_area_skeleton,
                            output_dir=tracing_out_path,
                            filename=f"grain_{grain_number}_node_{node_number}_node_area_skeleton.png",
                            **plotting_config["plot_dict"]["node_area_skeleton"],
                        ).plot_and_save()
                        Images(
                            data=grain_crop.image,
                            masked_array=node.node_branch_mask,
                            output_dir=tracing_out_path,
                            filename=f"grain_{grain_number}_node_{node_number}_node_branch_mask.png",
                            **plotting_config["plot_dict"]["node_branch_mask"],
                        ).plot_and_save()
                        Images(
                            data=grain_crop.image,
                            masked_array=node.node_avg_mask,
                            output_dir=tracing_out_path,
                            filename=f"grain_{grain_number}_node_{node_number}_node_avg_mask.png",
                            **plotting_config["plot_dict"]["node_avg_mask"],
                        ).plot_and_save()
                        if "all" in plotting_config["image_set"] or "nodestats" in plotting_config["image_set"]:
                            if not node.error:
                                fig, _ = plot_crossing_linetrace_halfmax(
                                    branch_stats=node.branch_stats,
                                    mask_cmap=plotting_config["plot_dict"]["node_line_trace"]["mask_cmap"],
                                    title=plotting_config["plot_dict"]["node_line_trace"]["title"],
                                )
                                fig.savefig(
                                    tracing_out_path / f"grain_{grain_number}_node_{node_number}_linetrace_halfmax.png",
                                    format="png",
                                )
                LOGGER.info(f"[{topostats_object.filename}] : Nodestats plotting completed successfully.")
            except Exception as e:
                LOGGER.error(
                    f"[{topostats_object.filename}] : Plotting nodestats failed. Consider raising an issue on"
                    "GitHub. Error : ",
                    exc_info=e,
                )
        return
    LOGGER.info(f"[{topostats_object.filename}] : Calculation of nodestats disabled, returning empty dataframe.")
    return


# need to add in the molstats here
def run_ordered_tracing(
    topostats_object: TopoStats,
    core_out_path: Path,
    tracing_out_path: Path,
    ordered_tracing_config: dict | None = None,
    plotting_config: dict | None = None,
) -> None:
    """
    Order coordinates of traces, adding results to statistics data frames and optionally plot results.

    Parameters
    ----------
    topostats_object : TopoStats
        TopoStats object for processing, should have had nodestats processing performed first.
    core_out_path : Path
        Path to save the core ordered tracing image to.
    tracing_out_path : Path
        Path to save optional, diagnostic ordered trace images to.
    ordered_tracing_config : dict
        Dictionary configuration for obtaining an ordered trace representation of the skeletons.
    plotting_config : dict
        Dictionary configuration for plotting images.
    """
    ordered_tracing_config = (
        topostats_object.config["ordered_tracing"] if ordered_tracing_config is None else ordered_tracing_config
    )
    plotting_config = topostats_object.config["plotting"] if plotting_config is None else plotting_config
    if ordered_tracing_config["run"]:
        ordered_tracing_config.pop("run")
        if topostats_object.grain_crops is None:
            LOGGER.warning(f"[{topostats_object.filename}] : No grains exist. Skipping ordered tracing.")
            return
        try:
            LOGGER.info(f"[{topostats_object.filename}] : *** Ordered Tracing ***")
            ordered_tracing_image(
                topostats_object=topostats_object,
                **ordered_tracing_config,
            )
            LOGGER.info(f"[{topostats_object.filename}] : Ordered Tracing stage completed successfully.")
        except ValueError as e:
            LOGGER.info(
                f"[{topostats_object.filename}] : Ordered Tracing failed with ValueError {e} - No skeletons exist."
            )
        except KeyError as e:
            LOGGER.info(
                f"[{topostats_object.filename}] : Ordered Tracing failed with KeyError {e} - no skeletons found from the Disordered Tracing step."
            )
        except Exception as e:
            LOGGER.info(
                f"[{topostats_object.filename}] : Ordered Tracing failed - skipping. Consider raising an issue on GitHub. Error: ",
                exc_info=e,
            )
        else:
            if plotting_config["run"]:
                try:
                    plotting_config["plot_dict"]["ordered_traces"][
                        "core_set"
                    ] = True  # fudge around core having own cmap
                    # ns-rse 2025-11-27 : What is being plotted here?
                    Images(
                        filename=f"{topostats_object.filename}_ordered_traces",
                        data=topostats_object.image,
                        masked_array=topostats_object.full_image_plots["ordered_traces"],
                        output_dir=core_out_path,
                        **plotting_config["plot_dict"]["ordered_traces"],
                    ).plot_and_save()

                    # save optional diagnostic plots (those with core_set = False)
                    # ns-rse 2025-12-10 : Reconsider if these plots are useful, if they are we need a configurable way of
                    # plotting them (which requires distinguishing them from those generated during disordered tracing)
                    for plot_name in ["all_molecules", "over_under", "trace_segments"]:
                        Images(
                            data=topostats_object.image,
                            masked_array=topostats_object.full_image_plots[plot_name],
                            output_dir=tracing_out_path,  # / direction,
                            **plotting_config["plot_dict"][plot_name],
                        ).plot_and_save()
                    LOGGER.info(f"[{topostats_object.filename}] : Ordered tracing plotting completed successfully.")
                except Exception as e:
                    LOGGER.error(
                        f"[{topostats_object.filename}] : Plotting ordered traces failed. Consider raising an issue on"
                        "GitHub. Error : ",
                        exc_info=e,
                    )
        return
    return


def run_splining(
    topostats_object: TopoStats,
    core_out_path: Path,
    splining_config: dict | None = None,
    plotting_config: dict | None = None,
) -> None:
    """
    Smooth the ordered trace coordinates and optionally plot results.

    Parameters
    ----------
    topostats_object : TopoStats
        TopoStats object to be splined.
    core_out_path : Path
        Path to save the core ordered tracing image to.
    splining_config : dict
        Dictionary configuration for obtaining an ordered trace representation of the skeletons.
    plotting_config : dict
        Dictionary configuration for plotting images.
    """
    splining_config = topostats_object.config["splining"] if splining_config is None else splining_config
    plotting_config = topostats_object.config["plotting"] if plotting_config is None else plotting_config
    if splining_config["run"]:
        splining_config.pop("run")
        if topostats_object.grain_crops is None:
            LOGGER.warning(f"[{topostats_object.filename}] : No grains exist. Skipping splining.")
            return
        try:
            LOGGER.info(f"[{topostats_object.filename}] : *** Splining ***")
            splining_image(
                topostats_object=topostats_object,
                **splining_config,
            )
            LOGGER.info(f"[{topostats_object.filename}] : Splining stage completed successfully.")
        except KeyError as e:
            LOGGER.info(
                f"[{topostats_object.filename}] : Splining failed with KeyError {e} - no ordered traces found from the Ordered Tracing step."
            )
        except Exception as e:
            LOGGER.error(
                f"[{topostats_object.filename}] : Splining failed - skipping. Consider raising an issue on GitHub. Error: ",
                exc_info=e,
            )
        else:
            if plotting_config["run"]:
                try:
                    # Extract coordinates for all splines into a single list for overlaying
                    all_splines = []
                    for _, grain_crop in topostats_object.grain_crops.items():
                        for _, molecule in grain_crop.ordered_trace.molecule_data.items():
                            all_splines.append(molecule.splined_coords + grain_crop.bbox[:2])
                    Images(
                        data=topostats_object.image,
                        output_dir=core_out_path,
                        filename=f"{topostats_object.filename}_all_splines",
                        # ns-rse 2025-12-03 : Need to pull out data and construct all_splines
                        plot_coords=all_splines,
                        **plotting_config["plot_dict"]["splined_trace"],
                    ).plot_and_save()
                    LOGGER.info(f"[{topostats_object.filename}] : Splining plotting completed successfully.")
                except Exception as e:
                    LOGGER.error(
                        f"[{topostats_object.filename}] : Plotting splines failed. Consider raising an issue on"
                        "GitHub. Error : ",
                        exc_info=e,
                    )
        return
    return


def run_curvature_stats(
    topostats_object: TopoStats,
    core_out_path: Path,  # pylint: disable=unused-argument
    tracing_out_path: Path,  # pylint: disable=unused-argument
    curvature_config: dict | None = None,
    plotting_config: dict | None = None,
) -> None:
    """
    Calculate curvature statistics for the traced DNA molecules.

    Currently only works on simple traces, not branched traces.

    Parameters
    ----------
    topostats_object : TopoStats
        ``TopoStats`` object post splining, all ``Molecules`` within the ``grain_crops`` attribute (a dictionary of
        ``GrainCrop`` should have ``splined_coords`` attributes populated.
    core_out_path : Path
        Path to save the core curvature image to.
    tracing_out_path : Path
        Path to save the optional, diagnostic curvature images to.
    curvature_config : dict
        Dictionary of configuration for running the curvature stats.
    plotting_config : dict
        Dictionary of configuration for plotting images.
    """
    curvature_config = topostats_object.config["curvature"].copy() if curvature_config is None else curvature_config
    plotting_config = (
        deepcopy(topostats_object.config["plotting"]) if plotting_config is None else deepcopy(plotting_config)
    )
    if curvature_config["run"]:
        if topostats_object.grain_crops is None:
            LOGGER.warning(f"[{topostats_object.filename}] : No grains exist. Skipping splining.")
            return
        try:
            curvature_config.pop("run")
            LOGGER.info(f"[{topostats_object.filename}] : *** Curvature Stats ***")
            # Pass the traces to the curvature stats function
            calculate_curvature_stats_image(topostats_object=topostats_object, **curvature_config)
            LOGGER.info(f"[{topostats_object.filename}] : Curvature stage completed successfully.")
        except Exception as e:
            LOGGER.error(
                f"[{topostats_object.filename}] : Curvature calculation failed. Consider raising an issue on GitHub. Error: ",
                exc_info=e,
            )
        else:
            try:
                if plotting_config["run"]:
                    # Setup dictionaries to aggregate components for the all image plot
                    all_curvatures = {}
                    all_smooth = {}
                    all_images = {}
                    colourmap_normalisation_bounds = plotting_config["plot_dict"]["curvature_individual_grains"].pop(
                        "colourmap_normalisation_bounds"
                    )
                    for grain_number, grain_crop in topostats_object.grain_crops.items():
                        all_curvatures[grain_number] = {}
                        all_smooth[grain_number] = {}
                        all_images[grain_number] = {}
                        for molecule_number, molecule in grain_crop.ordered_trace.molecule_data.items():
                            print(f"\n{grain_number=} : {molecule_number}\n")
                            Images(
                                np.array([[0, 0], [0, 0]]),  # dummy data, as the image is passed in the method call.
                                output_dir=tracing_out_path / "curvature",
                                **plotting_config["plot_dict"]["curvature_individual_grains"],
                            ).plot_curvatures_individual_grain(
                                grain_crop=grain_crop,
                                grain_number=grain_number,
                                colourmap_normalisation_bounds=colourmap_normalisation_bounds,
                            )
                            all_curvatures[grain_number][molecule_number] = molecule.curvature_stats
                            all_smooth[grain_number][molecule_number] = molecule.splined_coords
                            all_images[grain_number][molecule_number] = grain_crop.image
                    colourmap_normalisation_bounds = plotting_config["plot_dict"]["curvature"].pop(
                        "colourmap_normalisation_bounds"
                    )
                    Images(
                        np.array([[0, 0], [0, 0]]),  # dummy data, as the image is passed in the method call.
                        filename=f"{topostats_object.filename}_curvature",
                        output_dir=core_out_path,
                        **plotting_config["plot_dict"]["curvature"],
                    ).plot_curvatures(
                        image=topostats_object.image,
                        grain_crops=topostats_object.grain_crops,
                        colourmap_normalisation_bounds=colourmap_normalisation_bounds,
                    )
                    LOGGER.info(f"[{topostats_object.filename}] : Curvature plotting completed successfully.")
            except Exception as e:
                LOGGER.error(
                    f"[{topostats_object.filename}] : Plotting curvature failed. Consider raising an issue on"
                    "GitHub. Error : ",
                    exc_info=e,
                )
            return
        return
    LOGGER.info(f"[{topostats_object.filename}] : Calculation of Curvature Stats disabled, returning None.")
    return


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
    # ns-rse 2025-12-25 Do we want to keep tracing nested or promote disordered/ordered/nodes/curvature/splining up a
    # level?
    tracing_out_path = core_out_path / filename / "dnatracing"
    if "core" not in plotting_config["image_set"] and grain_dirs:
        filter_out_path.mkdir(exist_ok=True, parents=True)
        grain_out_path.mkdir(exist_ok=True, parents=True)
        # ns-rse 2025-12-05 Do we want disordered and ordered subdirectories?
        Path.mkdir(tracing_out_path / "nodes", parents=True, exist_ok=True)
        Path.mkdir(tracing_out_path / "curvature", parents=True, exist_ok=True)
        Path.mkdir(tracing_out_path / "splining", parents=True, exist_ok=True)

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
) -> TopoStats:
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
    TopoStats
        ``TopoStats`` object post-processing.
    """
    # Setup configuration, we use that from the topostats_object.config if not explicitly given an option
    config = topostats_object.config.copy()
    base_dir = config["base_dir"] if base_dir is None else base_dir
    filter_config = config["filter"] if filter_config is None else filter_config
    grains_config = config["grains"] if grains_config is None else grains_config
    grainstats_config = config["grainstats"] if grainstats_config is None else grainstats_config
    disordered_tracing_config = (
        config["disordered_tracing"] if disordered_tracing_config is None else disordered_tracing_config
    )
    nodestats_config = config["nodestats"] if nodestats_config is None else nodestats_config
    ordered_tracing_config = config["ordered_tracing"] if ordered_tracing_config is None else ordered_tracing_config
    splining_config = config["splining"] if splining_config is None else splining_config
    curvature_config = config["curvature"] if curvature_config is None else curvature_config
    plotting_config = config["plotting"].copy() if plotting_config is None else plotting_config
    output_dir = config["output_dir"]

    # Get output paths
    core_out_path, filter_out_path, grain_out_path, tracing_out_path = get_out_paths(
        image_path=topostats_object.img_path,
        base_dir=base_dir,
        output_dir=output_dir,
        filename=topostats_object.filename,
        plotting_config=plotting_config,
    )

    plotting_config = add_pixel_to_nm_to_plotting_config(plotting_config, topostats_object.pixel_to_nm_scaling)
    # Flatten Image
    run_filters(
        topostats_object=topostats_object,
        filter_out_path=filter_out_path,
        core_out_path=core_out_path,
        filter_config=filter_config,
        plotting_config=plotting_config,
    )
    # Find Grains :
    run_grains(
        topostats_object=topostats_object,
        grain_out_path=grain_out_path,
        core_out_path=core_out_path,
        plotting_config=plotting_config,
        grains_config=grains_config,
    )
    if topostats_object.grain_crops is not None:
        # Grainstats :
        run_grainstats(
            topostats_object=topostats_object,
            grainstats_config=grainstats_config,
            plotting_config=plotting_config,
            grain_out_path=grain_out_path,
            core_out_path=core_out_path,
        )

        # Disordered Tracing
        run_disordered_tracing(
            topostats_object=topostats_object,
            core_out_path=core_out_path,
            tracing_out_path=tracing_out_path,
            disordered_tracing_config=disordered_tracing_config,
            plotting_config=plotting_config,
        )

        # Nodestats
        run_nodestats(
            topostats_object=topostats_object,
            core_out_path=core_out_path,
            tracing_out_path=tracing_out_path,
            plotting_config=plotting_config,
            nodestats_config=nodestats_config,
        )

        # Ordered Tracing
        run_ordered_tracing(
            topostats_object=topostats_object,
            core_out_path=core_out_path,
            tracing_out_path=tracing_out_path,
            ordered_tracing_config=ordered_tracing_config,
            plotting_config=plotting_config,
        )

        # splining
        run_splining(
            topostats_object=topostats_object,
            core_out_path=core_out_path,
            plotting_config=plotting_config,
            splining_config=splining_config,
        )

        # Curvature Stats
        run_curvature_stats(
            topostats_object=topostats_object,
            core_out_path=core_out_path,
            tracing_out_path=tracing_out_path,
            curvature_config=curvature_config,
            plotting_config=plotting_config,
        )

    else:
        LOGGER.warning(f"[{topostats_object.filename}] : No grains found, skipping grainstats and tracing stages.")

    # Get image statistics
    LOGGER.info(f"[{topostats_object.filename}] : *** Image Statistics ***")
    # ns-rse 2025-12-12 Add @tobyallwood methods here for pulling statistics out of topostats_object

    # Save the topostats dictionary object to .topostats file.
    save_topostats_file(
        output_dir=core_out_path,
        filename=str(topostats_object.filename),
        topostats_object=topostats_object,
    )
    # ns-rse 2025-12-12 Return the Pandas data fraome from  @tobyallwood methods rather than topostats_object
    return topostats_object


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
    # Setup configuration, we use that from the topostats_object.config if not explicitly given an option
    config = topostats_object.config.copy()
    base_dir = config["base_dir"] if base_dir is None else base_dir
    filter_config = config["filter"] if filter_config is None else filter_config
    plotting_config = config["plotting"] if plotting_config is None else plotting_config
    output_dir = config["output_dir"]
    core_out_path, filter_out_path, _, _ = get_out_paths(
        image_path=topostats_object.img_path,
        base_dir=base_dir,
        output_dir=output_dir,
        filename=topostats_object.filename,
        plotting_config=plotting_config,
        grain_dirs=False,
    )
    plotting_config = add_pixel_to_nm_to_plotting_config(plotting_config, topostats_object.pixel_to_nm_scaling)
    # Flatten Image
    try:
        run_filters(
            topostats_object=topostats_object,
            filter_out_path=filter_out_path,
            core_out_path=core_out_path,
            filter_config=filter_config,
            plotting_config=plotting_config,
        )
        # Save the topostats dictionary object to .topostats file.
        save_topostats_file(
            output_dir=core_out_path,
            filename=str(topostats_object.filename),
            topostats_object=topostats_object,
        )
        return (topostats_object.filename, True)
    except:  # noqa: E722  # pylint: disable=bare-except
        LOGGER.info(f"Filtering failed for image : {topostats_object.filename}")
        return (topostats_object.filename, False)


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
    # Setup configuration, we use that from the topostats_object.config if not explicitly given an option
    config = topostats_object.config.copy()
    base_dir = config["base_dir"] if base_dir is None else base_dir
    grains_config = config["grains"] if grains_config is None else grains_config
    plotting_config = config["plotting"] if plotting_config is None else plotting_config
    output_dir = config["output_dir"]
    core_out_path, _, grain_out_path, _ = get_out_paths(
        image_path=topostats_object.img_path,
        base_dir=base_dir,
        output_dir=output_dir,
        filename=topostats_object.filename,
        plotting_config=plotting_config,
    )
    plotting_config = add_pixel_to_nm_to_plotting_config(plotting_config, topostats_object.pixel_to_nm_scaling)
    # Find Grains using the filtered image
    try:
        run_grains(
            topostats_object=topostats_object,
            grain_out_path=grain_out_path,
            core_out_path=core_out_path,
            plotting_config=plotting_config,
            grains_config=grains_config,
        )
        # Save the topostats dictionary object to .topostats file.
        save_topostats_file(
            output_dir=core_out_path,
            filename=str(topostats_object.filename),
            topostats_object=topostats_object,
        )
        return (topostats_object.filename, True)
    except:  # noqa: E722  # pylint: disable=bare-except
        LOGGER.info(f"Grain detection failed for image : {topostats_object.filename}")
        return (topostats_object.filename, False)


def process_grainstats(
    topostats_object: TopoStats,
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
    topostats_object : TopoStats
        A ``TopoStats`` object.
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
    # Setup configuration, we use that from the topostats_object.config if not explicitly given an option
    config = topostats_object.config.copy()
    base_dir = config["base_dir"] if base_dir is None else base_dir
    grainstats_config = config["grainstats"] if grainstats_config is None else grainstats_config
    plotting_config = config["plotting"] if plotting_config is None else plotting_config
    output_dir = config["output_dir"]
    core_out_path, _, grain_out_path, _ = get_out_paths(
        image_path=topostats_object.img_path,
        base_dir=base_dir,
        output_dir=output_dir,
        filename=topostats_object.filename,
        plotting_config=plotting_config,
    )
    plotting_config = add_pixel_to_nm_to_plotting_config(plotting_config, topostats_object.pixel_to_nm_scaling)

    # Calculate grainstats if there are any to be detected
    if topostats_object.grain_crops is not None:
        try:
            run_grainstats(
                topostats_object=topostats_object,
                grainstats_config=grainstats_config,
                plotting_config=plotting_config,
                grain_out_path=grain_out_path,
                core_out_path=core_out_path,
            )
            # Save the topostats dictionary object to .topostats file.
            save_topostats_file(
                output_dir=core_out_path,
                filename=str(topostats_object.filename),
                topostats_object=topostats_object,
            )
        except:  # noqa: E722  # pylint: disable=bare-except
            LOGGER.info(f"Grain detection failed for image : {topostats_object.filename}")
            return topostats_object
    LOGGER.info(f"[{topostats_object.filename}] : No grains present, GrainStats skipped.")
    return topostats_object


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
        f"  TopoStats Version           : {TOPOSTATS_BASE_VERSION}\n"
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
