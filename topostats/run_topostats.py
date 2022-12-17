"""Run TopoStats

This provides an entry point for running TopoStats as a command line programme.
"""
import argparse as arg
from collections import defaultdict
from functools import partial
import importlib.resources as pkg_resources
from multiprocessing import Pool
from pathlib import Path
import sys
from typing import Union, Dict
import yaml

import pandas as pd
import numpy as np
from tqdm import tqdm

from topostats.filters import Filters
from topostats.grains import Grains
from topostats.grainstats import GrainStats
from topostats.io import read_yaml, write_yaml, LoadScans
from topostats.logs.logs import setup_logger, LOGGER_NAME
from topostats.plottingfuncs import Images
from topostats.tracing.dnatracing import dnaTrace, traceStats
from topostats.utils import (
    find_images,
    get_out_path,
    update_config,
    convert_path,
    create_empty_dataframe,
    folder_grainstats,
)
from topostats.validation import validate_config, DEFAULT_CONFIG_SCHEMA, PLOTTING_SCHEMA

LOGGER = setup_logger(LOGGER_NAME)


# pylint: disable=broad-except
# pylint: disable=line-too-long
# pylint: disable=too-many-arguments
# pylint: disable=too-many-branches
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=unnecessary-dict-index-lookup


def create_parser() -> arg.ArgumentParser:
    """Create a parser for reading options."""
    parser = arg.ArgumentParser(
        description="Process AFM images. Additional arguments over-ride those in the configuration file."
    )
    parser.add_argument(
        "-c",
        "--config_file",
        dest="config_file",
        required=False,
        help="Path to a YAML configuration file.",
    )
    parser.add_argument(
        "--create-config-file",
        dest="create_config_file",
        type=str,
        required=False,
        help="Filename to write a sample YAML configuration file to (should end in '.yaml').",
    )
    parser.add_argument(
        "-b",
        "--base_dir",
        dest="base_dir",
        type=str,
        required=False,
        help="Base directory to scan for images.",
    )
    parser.add_argument(
        "-f",
        "--file_ext",
        dest="file_ext",
        type=str,
        required=False,
        help="File extension to scan for.",
    )
    parser.add_argument(
        "--channel",
        dest="channel",
        type=str,
        required=False,
        help="Channel to extract.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        dest="output_dir",
        type=str,
        required=False,
        help="Output directory to write results to.",
    )
    parser.add_argument(
        "-s",
        "--save_plots",
        dest="save_plots",
        type=bool,
        required=False,
        help="Whether to save plots.",
    )
    parser.add_argument(
        "-t", "--threshold_method", dest="threshold_method", required=False, help="Method used for thresholding."
    )
    parser.add_argument(
        "--otsu_threshold_multiplier",
        dest="otsu_threshold_multiplier",
        required=False,
        help="Factor to scale threshold during grain finding.",
    )
    parser.add_argument("-m", "--mask", dest="mask", type=bool, required=False, help="Mask the image.")
    parser.add_argument("-q", "--quiet", dest="quiet", type=bool, required=False, help="Toggle verbosity.")
    parser.add_argument(
        "-w",
        "--warnings",
        dest="quiet",
        type=str,
        required=False,
        help="Whether to ignore warnings.",
    )
    return parser


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
    filename = image_path.stem

    LOGGER.info(f"Processing : {filename}")
    _output_dir = get_out_path(image_path, base_dir, output_dir).parent / "Processed"
    _output_dir.mkdir(parents=True, exist_ok=True)

    if plotting_config["image_set"] == "core":
        filter_out_path = _output_dir
    else:
        filter_out_path = Path(_output_dir) / filename / "filters"
        filter_out_path.mkdir(exist_ok=True, parents=True)
        Path.mkdir(_output_dir / filename / "grains" / "upper", parents=True, exist_ok=True)
        Path.mkdir(_output_dir / filename / "grains" / "lower", parents=True, exist_ok=True)

    # Filter Image :
    if filter_config["run"]:
        filter_config.pop("run")
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
            plot_opts = {
                "pixel_to_nm_scaling": pixel_to_nm_scaling,
            }
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
            plot_name = "z_threshed"
            plotting_config["plot_dict"][plot_name]["output_dir"] = Path(_output_dir)
            Images(
                filtered_image.images["gaussian_filtered"],
                filename=filename + "_processed",
                **plotting_config["plot_dict"][plot_name],
            ).plot_and_save()
            plotting_config["run"] = True

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
                output_dir = Path(_output_dir) / filename / "grains" / f"{direction}"
                for plot_name, array in image_arrays.items():
                    plotting_config["plot_dict"][plot_name]["output_dir"] = output_dir
                    Images(array, **plotting_config["plot_dict"][plot_name]).plot_and_save()
                # Make a plot of coloured regions with bounding boxes
                plotting_config["plot_dict"]["bounding_boxes"]["output_dir"] = output_dir
                Images(
                    grains.directions[direction]["coloured_regions"],
                    **plotting_config["plot_dict"]["bounding_boxes"],
                    region_properties=grains.region_properties[direction],
                ).plot_and_save()
                plotting_config["plot_dict"]["coloured_boxes"]["output_dir"] = output_dir
                Images(
                    grains.directions[direction]["labelled_regions_02"],
                    **plotting_config["plot_dict"]["coloured_boxes"],
                    region_properties=grains.region_properties[direction],
                ).plot_and_save()

                plot_name = "mask_overlay"
                plotting_config["plot_dict"][plot_name]["output_dir"] = Path(_output_dir)
                Images(
                    filtered_image.images["gaussian_filtered"],
                    filename=f"{filename}_{direction}_processed_masked",
                    data2=grains.directions[direction]["removed_small_objects"],
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
                        base_output_dir=_output_dir / "grains",
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
                    results = grainstats_df.merge(tracing_stats_df, on=["Molecule Number", "threshold"])
                else:
                    results = grainstats_df
                    results["Image Name"] = filename
                    results["Basename"] = image_path.parent

            except Exception:
                # If no results we need a dummy dataframe to return.
                LOGGER.info(
                    f"[{filename}] : Errors occurred whilst calculating grain statistics and DNA tracing statistics."
                )
                results = create_empty_dataframe()

    return image_path, results


def main():
    """Run processing."""

    # Parse command line options, load config (or default) and update with command line options
    parser = create_parser()
    args = parser.parse_args()
    if args.config_file is not None:
        config = read_yaml(args.config_file)
    else:
        default_config = pkg_resources.open_text(__package__, "default_config.yaml")
        config = yaml.safe_load(default_config.read())
    config = update_config(config, args)
    config["output_dir"] = convert_path(config["output_dir"])

    # Validate configuration
    validate_config(config, schema=DEFAULT_CONFIG_SCHEMA, config_type="YAML configuration file")

    config["output_dir"].mkdir(parents=True, exist_ok=True)

    # Write sample configuration if asked to do so and exit
    if args.create_config_file:
        write_yaml(
            config,
            output_dir="./",
            config_file=args.create_config_file,
            header_message="Sample configuration file auto-generated",
        )
        LOGGER.info(f"A sample configuration has been written to : ./{args.create_config_file}")
        LOGGER.info(
            "Please refer to the documentation on how to use the configuration file : \n\n"
            "https://afm-spm.github.io/TopoStats/usage.html#configuring-topostats\n"
            "https://afm-spm.github.io/TopoStats/configuration.html"
        )
        sys.exit()
    # Load plotting_dictionary and validate
    plotting_dictionary = pkg_resources.open_text(__package__, "plotting_dictionary.yaml")
    config["plotting"]["plot_dict"] = yaml.safe_load(plotting_dictionary.read())
    validate_config(config["plotting"]["plot_dict"], schema=PLOTTING_SCHEMA, config_type="YAML configuration file")

    # FIXME : Make this a function and from topostats.utils import update_plot_dict and write tests
    # Update the config["plotting"]["plot_dict"] with plotting options
    for image, options in config["plotting"]["plot_dict"].items():
        config["plotting"]["plot_dict"][image] = {
            **options,
            "save_format": config["plotting"]["save_format"],
            "image_set": config["plotting"]["image_set"],
            "colorbar": config["plotting"]["colorbar"],
            "axes": config["plotting"]["axes"],
            "cmap": config["plotting"]["cmap"],
            "zrange": config["plotting"]["zrange"],
            "histogram_log_axis": config["plotting"]["histogram_log_axis"],
        }
        if image not in ["z_threshed", "mask_overlay", "grain_image", "grain_mask_image"]:
            config["plotting"]["plot_dict"][image].pop("zrange")

    LOGGER.info(f"Configuration file loaded from      : {args.config_file}")
    LOGGER.info(f"Scanning for images in              : {config['base_dir']}")
    LOGGER.info(f"Output directory                    : {str(config['output_dir'])}")
    LOGGER.info(f"Looking for images with extension   : {config['file_ext']}")
    img_files = find_images(config["base_dir"], file_ext=config["file_ext"])
    LOGGER.info(f"Images with extension {config['file_ext']} in {config['base_dir']} : {len(img_files)}")
    if len(img_files) == 0:
        LOGGER.error(f"No images with extension {config['file_ext']} in {config['base_dir']}")
        LOGGER.error("Please check your configuration and directories.")
        sys.exit()
    LOGGER.info(f'Thresholding method (Filtering)     : {config["filter"]["threshold_method"]}')
    LOGGER.info(f'Thresholding method (Grains)        : {config["grains"]["threshold_method"]}')

    if config["quiet"]:
        LOGGER.setLevel("ERROR")

    processing_function = partial(
        process_scan,
        base_dir=config["base_dir"],
        filter_config=config["filter"],
        grains_config=config["grains"],
        grainstats_config=config["grainstats"],
        dnatracing_config=config["dnatracing"],
        plotting_config=config["plotting"],
        output_dir=config["output_dir"],
    )

    all_scan_data = LoadScans(img_files, **config["loading"])
    all_scan_data.get_data()
    scan_data_dict = all_scan_data.img_dic

    with Pool(processes=config["cores"]) as pool:
        results = defaultdict()
        with tqdm(
            total=len(img_files),
            desc=f"Processing images from {config['base_dir']}, results are under {config['output_dir']}",
        ) as pbar:
            for img, result in pool.imap_unordered(
                processing_function,
                scan_data_dict.values(),
            ):
                results[str(img)] = result
                pbar.update()
    results = pd.concat(results.values())
    results.reset_index()
    results.to_csv(config["output_dir"] / "all_statistics.csv", index=False)
    LOGGER.info(
        (
            f"All statistics combined for {len(img_files)} images(s) are "
            f"saved to : {str(config['output_dir'] / 'all_statistics.csv')}"
        )
    )
    folder_grainstats(config["output_dir"], config["base_dir"], results)

    # Write config to file
    LOGGER.info(f"Writing configuration to : {config['output_dir']}/config.yaml")
    write_yaml(config, output_dir=config["output_dir"])


if __name__ == "__main__":
    main()
