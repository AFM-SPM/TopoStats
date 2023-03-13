"""Run TopoStats

This provides an entry point for running TopoStats as a command line programme.
"""
import argparse as arg
from collections import defaultdict
from functools import partial
import importlib.resources as pkg_resources
import logging
from multiprocessing import Pool
from pprint import pformat
import sys
import yaml

import pandas as pd
from tqdm import tqdm

from topostats._version import __version__
from topostats.io import find_files, read_yaml, save_folder_grainstats, write_yaml, LoadScans
from topostats.logs.logs import LOGGER_NAME
from topostats.plotting import toposum
from topostats.processing import check_run_steps, completion_message, process_scan
from topostats.utils import update_config, update_plotting_config
from topostats.validation import validate_config, DEFAULT_CONFIG_SCHEMA, PLOTTING_SCHEMA, SUMMARY_SCHEMA

# We already setup the logger in __init__.py and it is idempotent so calling it here returns the same object as from
# __init__.py
# Ref : https://stackoverflow.com/a/57799639/1444043
# LOGGER = setup_logger(LOGGER_NAME)
LOGGER = logging.getLogger(LOGGER_NAME)


# pylint: disable=too-many-branches
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=unnecessary-dict-index-lookup
# pylint: disable=too-many-nested-blocks


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
        "-s",
        "--summary_config",
        dest="summary_config",
        required=False,
        help="Path to a YAML configuration file for summary plots and statistics.",
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
        "-j",
        "--cores",
        dest="cores",
        type=int,
        required=False,
        help="Number of CPU cores to use when processing.",
    )
    parser.add_argument(
        "-l",
        "--log_level",
        dest="log_level",
        type=str,
        required=False,
        help="Logging level to use, default is 'info' for verbose output use 'debug'.",
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
        "--save_plots",
        dest="save_plots",
        type=bool,
        required=False,
        help="Whether to save plots.",
    )
    parser.add_argument("-m", "--mask", dest="mask", type=bool, required=False, help="Mask the image.")
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"Installed version of TopoStats : {__version__}",
        help="Report the current version of TopoStats that is installed.",
    )
    parser.add_argument(
        "-w",
        "--warnings",
        dest="warnings",
        type=bool,
        required=False,
        help="Whether to ignore warnings.",
    )
    return parser


def main(args=None):
    """Find and process all files."""

    # Parse command line options, load config (or default) and update with command line options
    parser = create_parser()
    args = parser.parse_args() if args is None else parser.parse_args(args)
    if args.config_file is not None:
        config = read_yaml(args.config_file)
    else:
        default_config = pkg_resources.open_text(__package__, "default_config.yaml")
        config = yaml.safe_load(default_config.read())
    config = update_config(config, args)

    # Set logging level
    if config["log_level"] == "warning":
        LOGGER.setLevel("WARNING")
    elif config["log_level"] == "error":
        LOGGER.setLevel("ERROR")
    elif config["log_level"] == "debug":
        LOGGER.setLevel("DEBUG")
    else:
        LOGGER.setLevel("INFO")
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
    validate_config(
        config["plotting"]["plot_dict"], schema=PLOTTING_SCHEMA, config_type="YAML plotting configuration file"
    )

    # Check earlier stages of processing are enabled for later.
    check_run_steps(
        filter_run=config["filter"]["run"],
        grains_run=config["grains"]["run"],
        grainstats_run=config["grainstats"]["run"],
        dnatracing_run=config["dnatracing"]["run"],
    )
    # Update the config["plotting"]["plot_dict"] with plotting options
    config["plotting"] = update_plotting_config(config["plotting"])

    LOGGER.info(f"Configuration file loaded from      : {args.config_file}")
    LOGGER.info(f"Scanning for images in              : {config['base_dir']}")
    LOGGER.info(f"Output directory                    : {str(config['output_dir'])}")
    LOGGER.info(f"Looking for images with extension   : {config['file_ext']}")
    img_files = find_files(config["base_dir"], file_ext=config["file_ext"])
    LOGGER.info(f"Images with extension {config['file_ext']} in {config['base_dir']} : {len(img_files)}")
    if len(img_files) == 0:
        LOGGER.error(f"No images with extension {config['file_ext']} in {config['base_dir']}")
        LOGGER.error("Please check your configuration and directories.")
        sys.exit()
    LOGGER.info(f'Thresholding method (Filtering)     : {config["filter"]["threshold_method"]}')
    LOGGER.info(f'Thresholding method (Grains)        : {config["grains"]["threshold_method"]}')
    LOGGER.debug(f"Configuration after update         : \n{pformat(config, indent=4)}")  # noqa : T203

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
    try:
        results = pd.concat(results.values())
    except ValueError as error:
        LOGGER.error("No grains found in any images, consider adjusting your thresholds.")
        LOGGER.error(error)

    # Summary Statistics and Plots
    if config["summary_stats"]["run"]:
        # Load summary plots/statistics configuration and validate, location depends on command line args or value in
        # any config file given, if neither are provided the default topostats/summary_config.yaml is loaded
        if args.summary_config is not None:
            summary_config = read_yaml(args.summary_config)
        elif config["summary_stats"]["config"] is not None:
            summary_config = read_yaml(config["summary_stats"]["config"])
        else:
            summary_yaml = pkg_resources.open_text(__package__, "summary_config.yaml")
            summary_config = yaml.safe_load(summary_yaml.read())
        summary_config = update_config(summary_config, args)
        validate_config(summary_config, SUMMARY_SCHEMA, config_type="YAML summarisation config")
        # We never want to load data from CSV as we are using the data that has just been processed.
        summary_config.pop("csv_file")

        # Load variable to label mapping
        plotting_yaml = pkg_resources.open_text(__package__, "var_to_label.yaml")
        summary_config["var_to_label"] = yaml.safe_load(plotting_yaml.read())
        LOGGER.info("[plotting] Default variable to labels mapping loaded.")

        # If we don't have a dataframe there is nothing to plot
        if isinstance(results, pd.DataFrame):
            # If summary_config["output_dir"] does not match or is not a sub-dir of config["output_dir"] it
            # needs creating
            summary_config["output_dir"] = config["output_dir"] / "summary_distributions"
            summary_config["output_dir"].mkdir(parents=True, exist_ok=True)
            LOGGER.info(f"Summary plots and statistics will be saved to : {summary_config['output_dir']}")

            # Plot summaries
            summary_config["df"] = results.reset_index()
            toposum(summary_config)
        else:
            LOGGER.info(
                "There are no results to plot, either you have disabled grains/grainstats/dnatracing or there "
                "have been errors, please check the log for further information."
            )

    # Write statistics to CSV
    if isinstance(results, pd.DataFrame):
        results.reset_index(inplace=True)
        results.set_index(["image", "threshold", "molecule_number"], inplace=True)
        results.to_csv(config["output_dir"] / "all_statistics.csv", index=True)
        save_folder_grainstats(config["output_dir"], config["base_dir"], results)
        results.reset_index(inplace=True)  # So we can access unique image names
        images_processed = len(results["image"].unique())
    else:
        images_processed = 0
    # Write config to file
    config["plotting"].pop("plot_dict")
    write_yaml(config, output_dir=config["output_dir"])
    LOGGER.debug(f"Images processed : {images_processed}")
    completion_message(config, img_files, summary_config, images_processed)


if __name__ == "__main__":
    main()
