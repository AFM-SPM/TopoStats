"""Functions for procesing data."""
from collections import defaultdict
from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd

from topostats.filters import Filters
from topostats.grains import Grains
from topostats.grainstats import GrainStats
from topostats.io import get_out_path
from topostats.logs.logs import setup_logger, LOGGER_NAME
from topostats.plottingfuncs import Images
from topostats.tracing.dnatracing import dnaTrace, traceStats
from topostats.utils import create_empty_dataframe

# pylint: disable=broad-except
# pylint: disable=line-too-long
# pylint: disable=too-many-arguments
# pylint: disable=too-many-branches
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=unnecessary-dict-index-lookup

LOGGER = setup_logger(LOGGER_NAME)


def process_scan(
    img_path_px2nm: Dict[str, Union[np.ndarray, Path, float]],
    base_dir: Union[str, Path],
    filter_config: dict,
    grains_config: dict,
    grainstats_config: dict,
    dnatracing_config: dict,
    plotting_config: dict,
    output_dir: Union[str, Path] = "output",
) -> None:
    """Process a single image, filtering, finding grains and calculating their statistics.

    Parameters
    ----------
    img_path_px2nm : Dict[str, Union[np.ndarray, Path, float]]
        A dictionary with keys 'image', 'img_path' and 'px_2_nm' containing a file or frames' image, it's path and it's pixel to namometre scaling value.
    base_dir : Union[str, Path]
        Directory to recursively search for files, if not specified the current directory is scanned.
    filter_config : dict
        Dictionary of configuration options for running the Filter stage.
    grains_config : dict
        Dictionary of configuration options for running the Grain detection stage.
    grainstats_config : dict
        Dictionary of configuration options for running the Grain Statistics stage.
    dnatracing_config : dict
        Dictionary of configuration options for running the DNA Tracing stage.
    plotting_config : dict
        Dictionary of configuration options for plotting figures.
    output_dir : Union[str, Path]
        Directory to save output to, it will be created if it does not exist. If it already exists then it is possible
        that output will be over-written.


    Results
    -------
    None

    Results are written to CSV and images produced in configuration options request them.
    """

    image = img_path_px2nm["image"]
    image_path = img_path_px2nm["img_path"]
    pixel_to_nm_scaling = img_path_px2nm["px_2_nm"]
    filename = image_path.name

    LOGGER.info(f"Processing : {filename}")
    core_out_path = get_out_path(image_path, base_dir, output_dir).parent / "processed"
    core_out_path.mkdir(parents=True, exist_ok=True)
    filter_out_path = core_out_path / filename / "filters"
    filter_out_path.mkdir(exist_ok=True, parents=True)
    grain_out_path = core_out_path / filename / "grains"
    Path.mkdir(grain_out_path / "upper", parents=True, exist_ok=True)
    Path.mkdir(grain_out_path / "lower", parents=True, exist_ok=True)

    # Filter Image
    if filter_config["run"]:
        filter_config.pop("run")
        LOGGER.info(f"[{filename}] : *** Filtering ***")
        filtered_image = Filters(
            image,
            filename,
            pixel_to_nm_scaling,
            **filter_config,
        )
        filtered_image.filter_image()

        # Optionally plot filter stage
        if plotting_config["run"]:
            plotting_config.pop("run")
            LOGGER.info(f"[{filename}] : Plotting Filtering Images")
            # Update PLOT_DICT with pixel_to_nm_scaling (can't add _output_dir since it changes)
            plot_opts = {"pixel_to_nm_scaling": pixel_to_nm_scaling}
            for image, options in plotting_config["plot_dict"].items():
                plotting_config["plot_dict"][image] = {**options, **plot_opts}
            # Generate plots
            for plot_name, array in filtered_image.images.items():
                if plot_name not in ["scan_raw"]:
                    if plot_name == "extracted_channel":
                        array = np.flipud(array.pixels)
                    plotting_config["plot_dict"][plot_name]["output_dir"] = filter_out_path
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
            filtered_image.images["gaussian_filtered"],
            filename=filename,
            **plotting_config["plot_dict"][plot_name],
        ).plot_and_save()

    # Find Grains :
    if grains_config["run"]:
        grains_config.pop("run")
        try:
            LOGGER.info(f"[{filename}] : *** Grain Finding ***")
            grains = Grains(
                image=filtered_image.images["gaussian_filtered"],
                filename=filename,
                pixel_to_nm_scaling=pixel_to_nm_scaling,
                **grains_config,
            )
            grains.find_grains()
        except IndexError:
            LOGGER.info(f"[{filename}] : No grains were detected, skipping Grain Statistics and DNA Tracing.")
        except ValueError:
            LOGGER.info(f"[{filename}] : No image, it is all masked.")
            results = create_empty_dataframe()
        if grains.region_properties is None:
            results = create_empty_dataframe()
        # Optionally plot grain finding stage
        if plotting_config["run"] and grains.region_properties is not None:
            plotting_config.pop("run")
            LOGGER.info(f"[{filename}] : Plotting Grain Finding Images")
            for direction, image_arrays in grains.directions.items():
                for plot_name, array in image_arrays.items():
                    plotting_config["plot_dict"][plot_name]["output_dir"] = grain_out_path / f"{direction}"
                    Images(array, **plotting_config["plot_dict"][plot_name]).plot_and_save()
                # Make a plot of coloured regions with bounding boxes
                plotting_config["plot_dict"]["bounding_boxes"]["output_dir"] = grain_out_path / f"{direction}"
                Images(
                    grains.directions[direction]["coloured_regions"],
                    **plotting_config["plot_dict"]["bounding_boxes"],
                    region_properties=grains.region_properties[direction],
                ).plot_and_save()
                plotting_config["plot_dict"]["coloured_boxes"]["output_dir"] = grain_out_path / f"{direction}"
                Images(
                    grains.directions[direction]["labelled_regions_02"],
                    **plotting_config["plot_dict"]["coloured_boxes"],
                    region_properties=grains.region_properties[direction],
                ).plot_and_save()
                # Always want mask_overlay (aka "Height Thresholded with Mask") but in core_out_path
                plot_name = "mask_overlay"
                plotting_config["plot_dict"][plot_name]["output_dir"] = core_out_path
                Images(
                    filtered_image.images["gaussian_filtered"],
                    filename=f"{filename}_{direction}_masked",
                    masked_array=grains.directions[direction]["removed_small_objects"],
                    **plotting_config["plot_dict"][plot_name],
                ).plot_and_save()

            plotting_config["run"] = True

        # Grainstats :
        #
        # There are two layers to process those above the given threshold and those below, use dictionary comprehension
        # to pass over these.
        if grainstats_config["run"] and grains.region_properties is not None:
            grainstats_config.pop("run")
            # Grain Statistics :
            try:
                LOGGER.info(f"[{filename}] : *** Grain Statistics ***")
                grain_plot_dict = {
                    key: value
                    for key, value in plotting_config["plot_dict"].items()
                    if key in ["grain_image", "grain_mask", "grain_mask_image"]
                }
                grainstats = {}
                for direction, _ in grains.directions.items():
                    grainstats[direction] = GrainStats(
                        data=filtered_image.images["gaussian_filtered"],
                        labelled_data=grains.directions[direction]["labelled_regions_02"],
                        pixel_to_nanometre_scaling=pixel_to_nm_scaling,
                        direction=direction,
                        base_output_dir=grain_out_path,
                        image_name=filename,
                        plot_opts=grain_plot_dict,
                        **grainstats_config,
                    ).calculate_stats()
                    grainstats[direction]["statistics"]["threshold"] = direction
                # Set tracing_stats_df in light of direction
                if grains_config["direction"] == "both":
                    grainstats_df = pd.concat([grainstats["lower"]["statistics"], grainstats["upper"]["statistics"]])
                elif grains_config["direction"] == "upper":
                    grainstats_df = grainstats["upper"]["statistics"]
                elif grains_config["direction"] == "lower":
                    grainstats_df = grainstats["lower"]["statistics"]
                # Run dnatracing
                if dnatracing_config["run"]:
                    dnatracing_config.pop("run")
                    LOGGER.info(f"[{filename}] : *** DNA Tracing ***")
                    dna_traces = defaultdict()
                    tracing_stats = defaultdict()
                    for direction, _ in grainstats.items():
                        dna_traces[direction] = dnaTrace(
                            full_image_data=filtered_image.images["gaussian_filtered"].T,
                            grains=grains.directions[direction]["labelled_regions_02"],
                            filename=filename,
                            pixel_size=pixel_to_nm_scaling,
                            **dnatracing_config,
                        )
                        dna_traces[direction].trace_dna()
                        tracing_stats[direction] = traceStats(trace_object=dna_traces[direction], image_path=image_path)
                        tracing_stats[direction].df["threshold"] = direction
                    # Set tracing_stats_df in light of direction
                    if grains_config["direction"] == "both":
                        tracing_stats_df = pd.concat([tracing_stats["lower"].df, tracing_stats["upper"].df])
                    elif grains_config["direction"] == "upper":
                        tracing_stats_df = tracing_stats["upper"].df
                    elif grains_config["direction"] == "lower":
                        tracing_stats_df = tracing_stats["lower"].df
                    LOGGER.info(f"[{filename}] : Combining {direction} grain statistics and dnatracing statistics")
                    # NB - Merge on molecule and threshold because we may have upper and lower molecueles which gives
                    #      duplicate molecule numbers as they are processed separately
                    results = grainstats_df.merge(tracing_stats_df, on=["molecule_number", "threshold"])
                else:
                    results = grainstats_df
                    results["image"] = filename
                    results["basename"] = image_path.parent
            except Exception:
                # If no results we need a dummy dataframe to return.
                LOGGER.info(
                    f"[{filename}] : Errors occurred whilst calculating grain statistics and DNA tracing statistics."
                )
                results = create_empty_dataframe()

    return image_path, results
