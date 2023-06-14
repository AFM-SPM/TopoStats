"""Functions for procesing data."""
from collections import defaultdict
from pathlib import Path
from typing import Dict, Union, List

import numpy as np
import pandas as pd

from topostats import __version__
from topostats.filters import Filters
from topostats.grains import Grains
from topostats.grainstats import GrainStats
from topostats.io import get_out_path, save_array
from topostats.logs.logs import setup_logger, LOGGER_NAME
from topostats.plottingfuncs import Images
from topostats.tracing.dnatracing import trace_image
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


def filter_wrapper(
    unprocessed_image: np.ndarray,
    pixel_to_nm_scaling: float,
    filename: str,
    filter_out_path: Path,
    core_out_path: Path,
    filter_config: dict,
    plotting_config: dict,
) -> np.ndarray:
    if filter_config["run"]:
        filter_config.pop("run")
        LOGGER.info(f"[{filename}] Image dimensions: {unprocessed_image.shape}")
        LOGGER.info(f"[{filename}] : *** Filtering ***")
        filtered_image = Filters(
            image=unprocessed_image,
            filename=filename,
            pixel_to_nm_scaling=pixel_to_nm_scaling,
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
        # Save the z_threshed image (aka "Height_Thresholded") numpy array
        save_array(
            array=filtered_image.images["gaussian_filtered"],
            outpath=core_out_path,
            filename=filename,
            array_type="height_thresholded",
        )

        return filtered_image.images["gaussian_filtered"]

    else:
        LOGGER.error(
            "You have not included running the initial filter stage. This is required for all subsequent "
            "stages of processing. Please check your configuration file."
        )

        return unprocessed_image


def grains_wrapper(
    image: np.ndarray,
    pixel_to_nm_scaling: float,
    filename: str,
    grain_out_path: Path,
    core_out_path: Path,
    plotting_config: dict,
    grains_config: dict,
):
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
            for direction, _ in grains.directions.items():
                LOGGER.info(
                    f"[{filename}] : Grains found for direction {direction} : {len(grains.region_properties[direction])}"
                )
                if len(grains.region_properties[direction]) == 0:
                    LOGGER.warning(f"[{filename}] : No grains found for direction {direction}")
        except Exception as e:
            LOGGER.error(f"[{filename}] : An error occured during grain finding, skipping grainstats and dnatracing.")
            LOGGER.error(f"[{filename}] : The error: {e}")
        else:
            for direction, region_props in grains.region_properties.items():
                if len(region_props) == 0:
                    LOGGER.warning(f"[{filename}] : No grains found for the {direction} direction.")
            # Optionally plot grain finding stage if we have found grains and plotting is required
            if plotting_config["run"]:
                plotting_config.pop("run")
                LOGGER.info(f"[{filename}] : Plotting Grain Finding Images")
                grain_masks = {}
                for direction, image_arrays in grains.directions.items():
                    LOGGER.info(f"[{filename}] : Plotting {direction} Grain Finding Images")
                    for plot_name, array in image_arrays.items():
                        LOGGER.info(f"[{filename}] : Plotting {plot_name} image")
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
                        image,
                        filename=f"{filename}_{direction}_masked",
                        masked_array=grains.directions[direction]["removed_small_objects"],
                        **plotting_config["plot_dict"][plot_name],
                    ).plot_and_save()

                    grain_masks[direction] = grains.directions[direction]["labelled_regions_02"]

                plotting_config["run"] = True

                return grain_masks

            else:
                LOGGER.info(f"[{filename}] : Plotting disabled for Grain Finding Images")

                return None
    else:
        LOGGER.info(f"[{filename}] Detection of grains disabled, returning empty data frame.")
        return None


def grainstats_wrapper(
    image: np.ndarray,
    pixel_to_nm_scaling: float,
    grain_masks: dict,
    filename: str,
    grainstats_config: dict,
    plotting_config: dict,
    grain_out_path: Path,
):
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
            grainstats = {}
            # There are two layers to process those above the given threshold and those below
            for direction, _ in grain_masks.items():
                if len(grain_masks[direction]) == 0:
                    LOGGER.warning(
                        f"[{filename}] : No grains exist for the {direction} direction. Skipping grainstats and DNAtracing."
                    )
                    grainstats[direction] = create_empty_dataframe()
                else:
                    grainstats[direction], grains_plot_data = GrainStats(
                        data=image,
                        labelled_data=grain_masks[direction],
                        pixel_to_nanometre_scaling=pixel_to_nm_scaling,
                        direction=direction,
                        base_output_dir=grain_out_path,
                        image_name=filename,
                        plot_opts=grain_plot_dict,
                        **grainstats_config,
                    ).calculate_stats()
                    grainstats[direction]["threshold"] = direction
                    # Plot grains
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
            # Set grainstats_df in light of direction
            if "above" in grain_masks.keys() and "below" in grain_masks.keys():
                grainstats_df = pd.concat([grainstats["below"], grainstats["above"]])
            elif "above" in grain_masks.keys():
                grainstats_df = grainstats["above"]
            elif "below" in grain_masks.keys():
                grainstats_df = grainstats["below"]

            return grainstats_df

        except Exception:
            LOGGER.info(f"[{filename}] : Errors occurred whilst calculating grain statistics. Skipping DNAtracing.")
            return create_empty_dataframe()


def dnatracing_wrapper(
    image: np.ndarray,
    grain_masks: np.ndarray,
    pixel_to_nm_scaling: float,
    image_path: Path,
    filename: str,
    dnatracing_config: dict,
    results_df: pd.DataFrame = None,
):
    # Run dnatracing
    try:
        if dnatracing_config["run"]:
            dnatracing_config.pop("run")
            LOGGER.info(f"[{filename}] : *** DNA Tracing ***")
            tracing_stats = defaultdict()
            for direction, _ in grain_masks.items():
                tracing_stats[direction] = trace_image(
                    image=image,
                    grains_mask=grain_masks[direction],
                    filename=filename,
                    pixel_to_nm_scaling=pixel_to_nm_scaling,
                    **dnatracing_config,
                )
                tracing_stats[direction]["threshold"] = direction
            # Set tracing_stats_df in light of direction
            if "above" in grain_masks.keys() and "below" in grain_masks.keys():
                tracing_stats_df = pd.concat([tracing_stats["below"], tracing_stats["above"]])
            elif "above" in grain_masks.keys():
                tracing_stats_df = tracing_stats["above"]
            elif "below" in grain_masks.keys():
                tracing_stats_df = tracing_stats["below"]
            LOGGER.info(f"[{filename}] : Combining {grain_masks.keys()} grain statistics and dnatracing statistics")
            # NB - Merge on image, molecule and threshold because we may have above and below molecules which
            #      gives duplicate molecule numbers as they are processed separately, if tracing stats
            #      are not available (because skeleton was too small), grainstats are still retained.
            results = results_df.merge(tracing_stats_df, on=["image", "threshold", "molecule_number"], how="left")
            results["basename"] = results_df.parent
        else:
            LOGGER.info(f"[{filename}] Calculation of DNA Tracing disabled, returning grainstats data frame.")
            results = results_df
            results["basename"] = image_path.parent

            return results

    except Exception:
        # If no results we need a dummy dataframe to return.
        LOGGER.warning(
            f"[{filename}] : Errors occurred whilst calculating DNA tracing statistics, " "returning grain statistics"
        )
        results = results_df
        results["basename"] = image_path.parent

        return results


def get_out_paths(image_path: Path, base_dir: Path, output_dir: Path, filename: str, plotting_config: dict):
    LOGGER.info(f"Processing : {filename}")
    core_out_path = get_out_path(image_path, base_dir, output_dir).parent / "processed"
    core_out_path.mkdir(parents=True, exist_ok=True)
    filter_out_path = core_out_path / filename / "filters"
    grain_out_path = core_out_path / filename / "grains"
    if plotting_config["image_set"] == "all":
        filter_out_path.mkdir(exist_ok=True, parents=True)
        Path.mkdir(grain_out_path / "above", parents=True, exist_ok=True)
        Path.mkdir(grain_out_path / "below", parents=True, exist_ok=True)

    return core_out_path, filter_out_path, grain_out_path


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

    unprocessed_image = img_path_px2nm["image"]
    image_path = img_path_px2nm["img_path"]
    pixel_to_nm_scaling = img_path_px2nm["px_2_nm"]
    filename = image_path.name

    core_out_path, filter_out_path, grain_out_path = get_out_paths(
        image_path, base_dir, output_dir, filename, plotting_config
    )

    # Filter Image
    flattened_image = filter_wrapper(
        unprocessed_image=unprocessed_image,
        pixel_to_nm_scaling=pixel_to_nm_scaling,
        filename=filename,
        filter_out_path=filter_out_path,
        core_out_path=core_out_path,
        filter_config=filter_config,
        plotting_config=plotting_config,
    )

    # Find Grains :
    grain_masks = grains_wrapper(
        image=flattened_image,
        pixel_to_nm_scaling=pixel_to_nm_scaling,
        filename=filename,
        grain_out_path=grain_out_path,
        core_out_path=core_out_path,
        plotting_config=plotting_config,
        grains_config=grains_config,
    )

    if grain_masks is not None:
        # Grainstats :
        results_df = grainstats_wrapper(
            image=flattened_image,
            pixel_to_nm_scaling=pixel_to_nm_scaling,
            grain_masks=grain_masks,
            filename=filename,
            grainstats_config=grainstats_config,
            plotting_config=plotting_config,
            grain_out_path=grain_out_path,
        )

        # DNAtracing
        results_df = dnatracing_wrapper(
            image=flattened_image,
            pixel_to_nm_scaling=pixel_to_nm_scaling,
            grain_masks=grain_masks,
            filename=filename,
            image_path=image_path,
            dnatracing_config=dnatracing_config,
            results_df=results_df,
        )

    else:
        results_df = create_empty_dataframe()

    return image_path, results_df


def check_run_steps(filter_run: bool, grains_run: bool, grainstats_run: bool, dnatracing_run: bool) -> None:
    """Check options for running steps (Filter, Grain, Grainstats and DNA tracing) are logically consistent.

    This checks that earlier steps required are enabled.

    Parameters
    ----------
    filter_run: bool
        Flag for running Filtering.
    grains_run: bool
        Flag for running Grains.
    grainstats_run: bool
        Flag for running GrainStats.
    dnatracing_run: bool
        Flag for running DNA Tracing.

    Returns
    -------
    None
    """
    if dnatracing_run:
        if grainstats_run is False:
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


def completion_message(config: Dict, img_files: List, summary_config: Dict, images_processed: int) -> None:
    """Print a completion message summarising images processed.

    Parameters
    ----------
    config: dict
        Configuration dictionary.
    img_files: list()
        List of found image paths.
    summary_config: dict(
        Configuration for plotting summary statistics.
    images_processed: int
        Pandas DataFrame of results.

    Results
    -------
    None
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
        f"  Configuration               : {config['output_dir']}/config.yaml\n"
        f"  All statistics              : {str(config['output_dir'])}/all_statistics.csv\n"
        f"  Distribution Plots          : {distribution_plots_message}\n\n"
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
