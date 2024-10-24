"""
Run TopoStats.

This provides an entry point for running TopoStats as a command line programme.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from functools import partial
from importlib import resources
from multiprocessing import Pool
from pprint import pformat

import pandas as pd
import yaml
from tqdm import tqdm

from topostats.io import (
    LoadScans,
    dict_to_json,
    find_files,
    merge_mappings,
    read_yaml,
    save_folder_grainstats,
    write_yaml,
)
from topostats.logs.logs import LOGGER_NAME
from topostats.plotting import toposum
from topostats.processing import check_run_steps, completion_message, process_scan
from topostats.utils import update_config, update_plotting_config
from topostats.validation import DEFAULT_CONFIG_SCHEMA, PLOTTING_SCHEMA, SUMMARY_SCHEMA, validate_config

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


def reconcile_config_args(args: argparse.Namespace | None) -> dict:
    """
    Reconcile command line arguments with the default configuration.
    
    Command line arguments take precedence over the default configuration. If a partial configuration file is specified (with '-c' or '--config-file') the defaults are over-ridden by these values (internally the configuration dictionary is updated with these values). Any other command line arguments take precedence over both the default and those supplied in a configuration file (again the dictionary is updated).
    
    The final configuration is validated before processing begins.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments passed into TopoStats.

    Returns
    -------
    dict
        The configuration dictionary.
    """
    default_config_raw = (resources.files(__package__) / "default_config.yaml").read_text()
    default_config = yaml.safe_load(default_config_raw)
    if args is not None:
        config_file_arg: str | None = args.config_file
        if config_file_arg is not None:
            config = read_yaml(config_file_arg)
            # Merge the loaded config with the default config to fill in any defaults that are missing
            # Make sure to prioritise the loaded config, so it overrides the default
            config = merge_mappings(map1=default_config, map2=config)
        else:
            # If no config file is provided, use the default config
            config = default_config
    else:
        # If no args are provided, use the default config
        config = default_config

    # Override the config with command line arguments passed in, eg --output_dir ./output/
    if args is not None:
        config = update_config(config, args)

    # Validate configuration
    validate_config(config, schema=DEFAULT_CONFIG_SCHEMA, config_type="YAML configuration file")

    return config


def run_topostats(args: argparse.Namespace | None = None) -> None:  # noqa: C901
    """
    Find and process all files.

    Parameters
    ----------
    args : None
        Arguments.
    """
    # Parse command line options, load config (or default) and update with command line options
    config = handle_config_file(args=args)

    # Set logging level
    if config["log_level"] == "warning":
        LOGGER.setLevel("WARNING")
    elif config["log_level"] == "error":
        LOGGER.setLevel("ERROR")
    elif config["log_level"] == "debug":
        LOGGER.setLevel("DEBUG")
    else:
        LOGGER.setLevel("INFO")

    # Create base output directory
    config["output_dir"].mkdir(parents=True, exist_ok=True)

    # Load plotting_dictionary and validate then update with command line options
    plotting_dictionary = (resources.files(__package__) / "plotting_dictionary.yaml").read_text()
    config["plotting"]["plot_dict"] = yaml.safe_load(plotting_dictionary)
    validate_config(
        config["plotting"]["plot_dict"], schema=PLOTTING_SCHEMA, config_type="YAML plotting configuration file"
    )
    config["plotting"] = update_config(config["plotting"], args)

    # Check earlier stages of processing are enabled for later.
    check_run_steps(
        filter_run=config["filter"]["run"],
        grains_run=config["grains"]["run"],
        grainstats_run=config["grainstats"]["run"],
        disordered_tracing_run=config["disordered_tracing"]["run"],
        nodestats_run=config["nodestats"]["run"],
        ordered_tracing_run=config["ordered_tracing"]["run"],
        splining_run=config["splining"]["run"],
    )
    # Ensures each image has all plotting options which are passed as **kwargs
    config["plotting"] = update_plotting_config(config["plotting"])
    LOGGER.debug(f"Plotting configuration after update :\n{pformat(config['plotting'], indent=4)}")

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
    LOGGER.info(f"Thresholding method (Filtering)     : {config['filter']['threshold_method']}")
    LOGGER.info(f"Thresholding method (Grains)        : {config['grains']['threshold_method']}")
    LOGGER.debug(f"Configuration after update         : \n{pformat(config, indent=4)}")  # noqa : T203

    processing_function = partial(
        process_scan,
        base_dir=config["base_dir"],
        filter_config=config["filter"],
        grains_config=config["grains"],
        grainstats_config=config["grainstats"],
        disordered_tracing_config=config["disordered_tracing"],
        nodestats_config=config["nodestats"],
        ordered_tracing_config=config["ordered_tracing"],
        splining_config=config["splining"],
        plotting_config=config["plotting"],
        output_dir=config["output_dir"],
    )

    all_scan_data = LoadScans(img_files, **config["loading"])
    all_scan_data.get_data()
    # Get a dictionary of all the image data dictionaries.
    # Keys are the image names
    # Values are the individual image data dictionaries
    scan_data_dict = all_scan_data.img_dict

    with Pool(processes=config["cores"]) as pool:
        results = defaultdict()
        image_stats_all = defaultdict()
        mols_results = defaultdict()
        disordered_trace_results = defaultdict()
        height_profile_all = defaultdict()
        with tqdm(
            total=len(img_files),
            desc=f"Processing images from {config['base_dir']}, results are under {config['output_dir']}",
        ) as pbar:
            for (
                img,
                result,
                height_profiles,
                individual_image_stats_df,
                disordered_trace_result,
                mols_result,
            ) in pool.imap_unordered(
                processing_function,
                scan_data_dict.values(),
            ):
                results[str(img)] = result.dropna(axis=1, how="all")
                disordered_trace_results[str(img)] = disordered_trace_result.dropna(axis=1, how="all")
                mols_results[str(img)] = mols_result.dropna(axis=1, how="all")
                pbar.update()

                # Add the dataframe to the results dict
                image_stats_all[str(img)] = individual_image_stats_df.dropna(axis=1, how="all")

                # Combine all height profiles
                height_profile_all[str(img)] = height_profiles

                # Display completion message for the image
                LOGGER.info(f"[{img.name}] Processing completed.")

    LOGGER.info(f"Saving image stats to : {config['output_dir']}/image_stats.csv.")
    # Concatenate all the dictionary's values into a dataframe. Ignore the keys since
    # the dataframes have the file names in them already.
    image_stats_all_df = pd.concat(image_stats_all.values())
    image_stats_all_df.to_csv(config["output_dir"] / "image_stats.csv")

    try:
        results = pd.concat(results.values())
    except ValueError as error:
        LOGGER.error("No grains found in any images, consider adjusting your thresholds.")
        LOGGER.error(error)

    try:
        disordered_trace_results = pd.concat(disordered_trace_results.values())
    except ValueError as error:
        LOGGER.error("No skeletons found in any images, consider adjusting disordered tracing parameters.")
        LOGGER.error(error)

    try:
        mols_results = pd.concat(mols_results.values())
    except ValueError as error:
        LOGGER.error("No mols found in any images, consider adjusting ordered tracing / splining parameters.")
        LOGGER.error(error)
    # If requested save height profiles
    if config["grainstats"]["extract_height_profile"]:
        LOGGER.info(f"Saving all height profiles to {config['output_dir']}/height_profiles.json")
        dict_to_json(data=height_profile_all, output_dir=config["output_dir"], filename="height_profiles.json")

    # Summary Statistics and Plots
    if config["summary_stats"]["run"]:
        # Load summary plots/statistics configuration and validate, location depends on command line args or value in
        # any config file given, if neither are provided the default topostats/summary_config.yaml is loaded
        if args.summary_config is not None:
            summary_config = read_yaml(args.summary_config)
        elif config["summary_stats"]["config"] is not None:
            summary_config = read_yaml(config["summary_stats"]["config"])
        else:
            summary_yaml = (resources.files(__package__) / "summary_config.yaml").read_text()
            summary_config = yaml.safe_load(summary_yaml)

        # Do not pass command line arguments to toposum as they clash with process command line arguments
        summary_config = update_config(summary_config, config["plotting"])

        validate_config(summary_config, SUMMARY_SCHEMA, config_type="YAML summarisation config")
        # We never want to load data from CSV as we are using the data that has just been processed.
        summary_config.pop("csv_file")

        # Load variable to label mapping
        plotting_yaml = (resources.files(__package__) / "var_to_label.yaml").read_text()
        summary_config["var_to_label"] = yaml.safe_load(plotting_yaml)
        LOGGER.info("[plotting] Default variable to labels mapping loaded.")

        # If we don't have a dataframe or we do and it is all NaN there is nothing to plot
        if isinstance(results, pd.DataFrame) and not results.isna().values.all():
            if results.shape[0] > 1:
                # If summary_config["output_dir"] does not match or is not a sub-dir of config["output_dir"] it
                # needs creating
                summary_config["output_dir"] = config["output_dir"] / "summary_distributions"
                summary_config["output_dir"].mkdir(parents=True, exist_ok=True)
                LOGGER.info(f"Summary plots and statistics will be saved to : {summary_config['output_dir']}")

                # Plot summaries
                summary_config["df"] = results.reset_index()
                toposum(summary_config)
            else:
                LOGGER.warning(
                    "There are fewer than two grains that have been detected, so"
                    " summary plots cannot be made for this image."
                )
        else:
            LOGGER.warning(
                "There are no results to plot, either...\n\n"
                "* you have disabled grains/grainstats etc.\n"
                "* no grains have been detected across all scans.\n"
                "* there have been errors.\n\n"
                "If you are not expecting to detect grains please consider disabling"
                "grains/grainstats etc/plotting/summary_stats. If you are expecting to detect grains"
                " please check log-files for further information."
            )
    else:
        summary_config = None

    # Write statistics to CSV if there is data.
    if isinstance(results, pd.DataFrame) and not results.isna().values.all():
        results.reset_index(inplace=True)
        results.set_index(["image", "threshold", "grain_number"], inplace=True)
        results.to_csv(config["output_dir"] / "all_statistics.csv", index=True)
        save_folder_grainstats(config["output_dir"], config["base_dir"], results, "grain_stats")
        results.reset_index(inplace=True)  # So we can access unique image names
        images_processed = len(results["image"].unique())
    else:
        images_processed = 0
        LOGGER.warning("There are no grainstats statistics to write to CSV.")

    if isinstance(disordered_trace_results, pd.DataFrame) and not disordered_trace_results.isna().values.all():
        disordered_trace_results.reset_index(inplace=True)
        disordered_trace_results.set_index(["image", "threshold", "grain_number"], inplace=True)
        disordered_trace_results.to_csv(config["output_dir"] / "all_disordered_segment_statistics.csv", index=True)
        save_folder_grainstats(
            config["output_dir"], config["base_dir"], disordered_trace_results, "disordered_trace_stats"
        )
        disordered_trace_results.reset_index(inplace=True)  # So we can access unique image names
    else:
        LOGGER.warning("There are no disordered tracing statistics to write to CSV.")

    if isinstance(mols_results, pd.DataFrame) and not mols_results.isna().values.all():
        mols_results.reset_index(drop=True, inplace=True)
        mols_results.set_index(["image", "threshold", "grain_number"], inplace=True)
        mols_results.to_csv(config["output_dir"] / "all_mol_statistics.csv", index=True)
        save_folder_grainstats(config["output_dir"], config["base_dir"], mols_results, "mol_stats")
        mols_results.reset_index(inplace=True)  # So we can access unique image names
    else:
        LOGGER.warning("There are no molecule tracing statistics to write to CSV.")
    # Write config to file
    config["plotting"].pop("plot_dict")
    write_yaml(config, output_dir=config["output_dir"])
    LOGGER.debug(f"Images processed : {images_processed}")
    # Update config with plotting defaults for printing
    completion_message(config, img_files, summary_config, images_processed)
