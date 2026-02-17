"""
Run TopoStats modules.

This provide entry points for running TopoStats as a command line programme. Each function within this module is a
wrapper which runs various functions from the ''processing'' module in parallel.
"""

import argparse
import logging
import re
import sys
from collections import defaultdict
from functools import partial
from importlib import resources
from multiprocessing import Pool
from pathlib import Path
from pprint import pformat

import pandas as pd
import yaml
from tqdm import tqdm

from topostats.config import reconcile_config_args, update_config, update_plotting_config
from topostats.io import (
    LoadScans,
    extract_height_profiles,
    find_files,
    read_yaml,
    write_csv,
    write_yaml,
)
from topostats.logs.logs import LOGGER_NAME
from topostats.plotting import toposum
from topostats.processing import (
    check_run_steps,
    completion_message,
    process_filters,
    process_grains,
    process_grainstats,
    process_scan,
    run_disordered_tracing,
    run_nodestats,
    run_ordered_tracing,
    run_splining,
)
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


def _set_logging(log_level: str | None) -> None:
    """
    Set the logging level.

    Parameters
    ----------
    log_level : str
        String for the desired log-level.
    """
    if log_level == "warning":
        LOGGER.setLevel("WARNING")
    elif log_level == "error":
        LOGGER.setLevel("ERROR")
    elif log_level == "debug":
        LOGGER.setLevel("DEBUG")
    else:
        LOGGER.setLevel("INFO")


def _log_setup(config: dict, args: argparse.Namespace | None, img_files: dict) -> None:
    """
    Log the current configuration.

    Parameters
    ----------
    config : dict
        Dictionary of configuration options.
    args : argparse.Namespace | None
        Arguments function was invoked with.
    img_files : dict
        Dictionary of image files that have been found.
    """
    LOGGER.debug(f"Plotting configuration after update :\n{pformat(config['plotting'], indent=4)}")

    LOGGER.info(f"Configuration file loaded from      : {args.config_file}")
    LOGGER.info(f"Scanning for images in              : {config['base_dir']}")
    LOGGER.info(f"Output directory                    : {str(config['output_dir'])}")
    LOGGER.info(f"Looking for images with extension   : {config['file_ext']}")
    LOGGER.info(f"Images with extension {config['file_ext']} in {config['base_dir']} : {len(img_files)}")
    if len(img_files) == 0:
        LOGGER.error(f"No images with extension {config['file_ext']} in {config['base_dir']}")
        LOGGER.error("Please check your configuration and directories.")
        sys.exit()
    LOGGER.info(f"Thresholding method (Filtering)     : {config['filter']['threshold_method']}")
    LOGGER.info(f"Thresholding method (Grains)        : {config['grains']['threshold_method']}")
    LOGGER.debug(f"Configuration after update         : \n{pformat(config, indent=4)}")  # noqa: T203


def _parse_configuration(args: argparse.Namespace | None = None) -> tuple[dict, dict]:
    """
    Load configurations, validate and check run steps are consistent.

    Parameters
    ----------
    args : argparse.Namespace | None
        Arguments.

    Returns
    -------
    tuple[dict, dict]
        Returns the dictionary of configuration options and a dictionary of image files found on the input path.
    """
    # Parse command line options, load config (or default) and update with command line options
    config = reconcile_config_args(args=args)

    # Validate configuration
    validate_config(config, schema=DEFAULT_CONFIG_SCHEMA, config_type="YAML configuration file")

    # Set logging level
    _set_logging(config["log_level"])

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
    img_files = find_files(config["base_dir"], file_ext=config["file_ext"])
    _log_setup(config, args, img_files)
    return config, img_files


def process(args: argparse.Namespace | None = None) -> None:  # noqa: C901
    """
    Find and process all files.

    Parameters
    ----------
    args : None
        Arguments.
    """
    config, img_files = _parse_configuration(args)
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
        curvature_config=config["curvature"],
        plotting_config=config["plotting"],
        output_dir=config["output_dir"],
    )
    # Ensure we load the original images as we are running the whole pipeline
    if config["file_ext"] == ".topostats":
        config["loading"]["extract"] = "raw"

    output_full_stats = config["output_stats"] == "full"

    all_scan_data = LoadScans(img_files, config=config)
    all_scan_data.get_data()
    # Get a dictionary of all the image data dictionaries.
    # Keys are the image names
    # Values are the individual image data dictionaries
    scan_data_dict = all_scan_data.img_dict

    with Pool(processes=config["cores"]) as pool:
        grain_stats_all = defaultdict()
        topostats_object_all = defaultdict()
        image_stats_all = defaultdict()
        disordered_tracing_all = defaultdict()
        branch_stats_all = defaultdict()
        molecule_stats_all = defaultdict()
        with tqdm(
            total=len(img_files),
            desc=f"Processing images from {config['base_dir']}, results are under {config['output_dir']}",
        ) as pbar:
            for (
                filename,
                grain_stats_df,
                topostats_object,
                image_stats_df,
                disordered_tracing_df,
                branch_stats_df,
                molecule_stats_df,
            ) in pool.imap_unordered(
                processing_function,
                scan_data_dict.values(),
            ):
                # Append each images returned dataframes to the dictionaries
                if grain_stats_df is not None:
                    grain_stats_all[str(filename)] = grain_stats_df.dropna(axis=1, how="all")
                topostats_object_all[str(filename)] = topostats_object
                if image_stats_df is not None:
                    image_stats_all[str(filename)] = image_stats_df.dropna(axis=1, how="all")
                if disordered_tracing_df is not None:
                    disordered_tracing_all[str(filename)] = disordered_tracing_df.dropna(axis=1, how="all")
                if branch_stats_df is not None:
                    branch_stats_all[str(filename)] = branch_stats_df.dropna(axis=1, how="all")
                if molecule_stats_df is not None:
                    molecule_stats_all[str(filename)] = molecule_stats_df.dropna(axis=1, how="all")

                pbar.update()
                # Display completion message for the image
                LOGGER.info(f"[{filename}] Processing completed.")
    # Concatenate all the dictionary's values into a dataframe. Ignore the keys since
    # the dataframes have the file names in them already.
    image_stats_all_df = pd.concat(image_stats_all.values())
    image_stats_all_df.to_csv(config["output_dir"] / "image_statistics.csv")

    # Molecule statistics - required as we need to average end-to-end and contour length across grains, even if not
    # being explicitly written to CSV themselves
    try:
        molecule_stats_all = pd.concat(molecule_stats_all.values())
        grain_stats_additions = pd.concat(
            [
                # Sum the contour length of molecules within each grain
                molecule_stats_all[["image", "grain_number", "contour_length"]]
                .groupby(["image", "grain_number"])
                .sum(),
                # Mean end to end distance across molecules within each grain
                molecule_stats_all[["image", "grain_number", "end_to_end_distance"]]
                .groupby(["image", "grain_number"])
                .mean(),
            ],
            axis=1,
        )
        grain_stats_additions.columns = ["total_contour_length", "mean_end_to_end_distance"]
    except ValueError as error:
        LOGGER.error(
            "No molecules found in any images."
            "Either enable tracing or consider adjusting ordered tracing / splining parameters."
        )
        LOGGER.error(error)
        grain_stats_additions = None

    # ns-rse 2025-12-23 - there is a common pattern here, could we abstract this to a factory method?
    if len(grain_stats_all) > 0:
        try:
            grain_stats_all = pd.concat(grain_stats_all.values())
            grain_stats_all.reset_index(inplace=True)
            grain_stats_all.set_index(["image", "grain_number"], inplace=True)
        except ValueError as error:
            LOGGER.error("No grains found in any images, consider adjusting your thresholds.")
            LOGGER.error(error)
        if grain_stats_additions is not None:
            grain_stats_all = grain_stats_all.merge(grain_stats_additions, on=["image", "grain_number"])
        else:
            LOGGER.warning("No molecule statistics to merge with grain statistics.")
        # Write statistics to CSV if there is data.
        if isinstance(grain_stats_all, pd.DataFrame) and not grain_stats_all.isna().values.all():
            grain_stats_all = write_csv(
                df=grain_stats_all,
                dataset="grain_stats",
                # Reset after above merge
                names=["image", "grain_number"],
                index=["image", "grain_number", "class", "subgrain"],
                output_dir=config["output_dir"],
                base_dir=config["base_dir"],
            )
            LOGGER.info(f"Saved grain stats to : {config['output_dir']}/grain_statistics.csv.")
    else:
        images_processed = 0
        LOGGER.warning("There are no grainstats statistics to write to CSV.")

    # Optional output files
    if output_full_stats:
        if branch_stats_all is not None:
            # Matched branch statistics
            try:
                branch_stats_all = pd.concat(branch_stats_all.values())
            except ValueError as error:
                LOGGER.error("No skeletons found in any images, consider adjusting disordered tracing parameters.")
                LOGGER.error(error)
            if isinstance(branch_stats_all, pd.DataFrame) and not branch_stats_all.isna().values.all():
                branch_stats_all = write_csv(
                    df=branch_stats_all,
                    dataset="matched_branch_stats",
                    names=["grain_number", "node", "branch"],
                    index=["image", "grain_number", "node", "branch"],
                    output_dir=config["output_dir"],
                    base_dir=config["base_dir"],
                )
                LOGGER.info(f"Saved matched branch stats to : {config['output_dir']}/matched_branch_statistics.csv.")
        else:
            LOGGER.warning("There are no matched branch statistics to write to CSV.")
        # Disordered trace statistics
        if disordered_tracing_all is not None:
            try:
                disordered_tracing_all = pd.concat(disordered_tracing_all.values())
            except ValueError as error:
                LOGGER.error("No skeletons found in any images, consider adjusting disordered tracing parameters.")
                LOGGER.error(error)
            if isinstance(disordered_tracing_all, pd.DataFrame) and not disordered_tracing_all.isna().values.all():
                disordered_tracing_all = write_csv(
                    df=disordered_tracing_all,
                    dataset="branch_statistics",
                    names=["grain_number", "index"],
                    index=["image", "grain_number", "index"],
                    output_dir=config["output_dir"],
                    base_dir=config["base_dir"],
                )
                LOGGER.info(f"Saved disordered tracing stats to : {config['output_dir']}/branch_statistics.csv.")
        else:
            LOGGER.warning("There are no disordered tracing statistics to write to CSV.")

        # Molecule statistics
        if molecule_stats_all is not None:
            if isinstance(molecule_stats_all, pd.DataFrame) and not molecule_stats_all.isna().values.all():
                molecule_stats_all = write_csv(
                    df=molecule_stats_all,
                    dataset="mol_stats",
                    names=None,
                    index=["image", "grain_number"],
                    output_dir=config["output_dir"],
                    base_dir=config["base_dir"],
                )
                LOGGER.info(f"Saved molecule stats to : {config['output_dir']}/molecule_statistics.csv.")
        else:
            LOGGER.warning("There are no molecule tracing statistics to write to CSV.")

    else:
        LOGGER.info(
            "Writing 'molecule_statistics.csv', 'branch_statistics.csv' and'matched_branch_statistics.csv' skipped"
        )

    # If requested save height profiles
    if config["grainstats"]["extract_height_profile"]:
        extract_height_profiles(
            topostats_object_all=topostats_object_all,
            output_dir=config["output_dir"],
            filename="height_profiles.json",
        )

    # Write config to file
    config["plotting"].pop("plot_dict")
    write_yaml(config, output_dir=config["output_dir"])
    images_processed = len(grain_stats_all["image"].unique())
    LOGGER.debug(f"Images processed : {images_processed}")

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
        if isinstance(grain_stats_all, pd.DataFrame) and not grain_stats_all.isna().values.all():
            if grain_stats_all.shape[0] > 1:
                # If summary_config["output_dir"] does not match or is not a sub-dir of config["output_dir"] it
                # needs creating
                summary_config["output_dir"] = Path(config["output_dir"]) / "summary_distributions"
                summary_config["output_dir"].mkdir(parents=True, exist_ok=True)
                LOGGER.info(f"Summary plots and statistics will be saved to : {summary_config['output_dir']}")

                # Plot summaries
                summary_config["df"] = grain_stats_all.reset_index()
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
                " grains/grainstats etc/plotting/summary_stats. If you are expecting to detect grains"
                " please check log-files for further information."
            )
    else:
        summary_config = None

    # Update config with plotting defaults for printing
    completion_message(config, img_files, summary_config, images_processed)


# WIP these will be added from args once the dictionary keys have been wrangled
# pylint: disable=no-value-for-parameter
# pylint: disable=unused-argument


def filters(args: argparse.Namespace | None = None) -> None:
    """
    Load files from disk and run filtering.

    Parameters
    ----------
    args : None
        Arguments.
    """
    config, img_files = _parse_configuration(args)
    # If loading existing .topostats files the images need filtering again so we need to extract the raw image
    if config["file_ext"] == ".topostats":
        config["loading"]["extract"] = "raw"
    all_scan_data = LoadScans(img_files, config=config)
    all_scan_data.get_data()

    processing_function = partial(
        process_filters,
        base_dir=config["base_dir"],
        filter_config=config["filter"],
        plotting_config=config["plotting"],
        output_dir=config["output_dir"],
    )

    with Pool(processes=config["cores"]) as pool:
        results = defaultdict()
        with tqdm(
            total=len(img_files),
            desc=f"Processing images from {config['base_dir']}, results are under {config['output_dir']}",
        ) as pbar:
            for img, result in pool.imap_unordered(
                processing_function,
                all_scan_data.img_dict.values(),
            ):
                results[str(img)] = result
                pbar.update()

                # Display completion message for the image
                LOGGER.info(f"[{img}] Filtering completed.")

    # Write config to file
    config["plotting"].pop("plot_dict")
    write_yaml(config, output_dir=config["output_dir"])
    LOGGER.debug(f"Images processed : {len(results)}")
    # Update config with plotting defaults for printing
    completion_message(config, img_files, summary_config=None, images_processed=sum(results.values()))


def grains(args: argparse.Namespace | None = None) -> None:
    """
    Load files from disk and run grain finding.

    Parameters
    ----------
    args : None
        Arguments.
    """
    config, img_files = _parse_configuration(args)
    # Triggers extraction of filtered images from existing .topostats files
    if config["file_ext"] == ".topostats":
        config["loading"]["extract"] = "grains"
    all_scan_data = LoadScans(img_files, config=config)
    all_scan_data.get_data()

    processing_function = partial(
        process_grains,
        base_dir=config["base_dir"],
        grains_config=config["grains"],
        plotting_config=config["plotting"],
        output_dir=config["output_dir"],
    )
    with Pool(processes=config["cores"]) as pool:
        results = defaultdict()
        with tqdm(
            total=len(img_files),
            desc=f"Processing images from {config['base_dir']}, results are under {config['output_dir']}",
        ) as pbar:
            for img, result in pool.imap_unordered(
                processing_function,
                all_scan_data.img_dict.values(),
            ):
                results[str(img)] = result
                pbar.update()

                # Display completion message for the image
                LOGGER.info(f"[{img}] Grain detection completed (NB - Filtering was *not* re-run).")

    # Write config to file
    config["plotting"].pop("plot_dict")
    write_yaml(config, output_dir=config["output_dir"])
    LOGGER.debug(f"Images processed : {len(results)}")
    # Update config with plotting defaults for printing
    completion_message(config, img_files, summary_config=None, images_processed=sum(results.values()))


def grainstats(args: argparse.Namespace | None = None) -> None:
    """
    Load files from disk and run grainstats.

    Parameters
    ----------
    args : None
        Arguments.
    """
    config, img_files = _parse_configuration(args)  # pylint: disable=unused-variable
    # Triggers extraction of filtered images from existing .topostats files
    if config["file_ext"] == ".topostats":
        config["loading"]["extract"] = "grainstats"
    all_scan_data = LoadScans(img_files, config=config)
    all_scan_data.get_data()
    processing_function = partial(
        process_grainstats,
        base_dir=config["base_dir"],
        grainstats_config=config["grainstats"],
        plotting_config=config["plotting"],
        output_dir=config["output_dir"],
    )
    with Pool(processes=config["cores"]) as pool:
        grain_stats_all = defaultdict()
        topostats_object_all = defaultdict()
        with tqdm(
            total=len(img_files),
            desc=f"Processing images from {config['base_dir']}, results are under {config['output_dir']}",
        ) as pbar:
            for (
                filename,
                topostats_object,
                grain_stats_df,
            ) in pool.imap_unordered(
                processing_function,
                all_scan_data.img_dict.values(),
            ):
                topostats_object_all[filename] = topostats_object
                grain_stats_all[filename] = grain_stats_df
                pbar.update()

                # Display completion message for the image
                LOGGER.info(f"[{filename}] Grainstats completed (NB - Filtering was *not* re-run).")

    # Concatenate all the dictionary's values into a dataframe. Ignore the keys since
    # the dataframes have the file names in them already.
    try:
        grain_stats_all_df = pd.concat(grain_stats_all.values())
        grain_stats_all_df.to_csv(config["output_dir"] / "image_statistics.csv")
    except ValueError as error:
        LOGGER.error("No grains found in any images, consider adjusting your thresholds.")
        LOGGER.error(error)
    # If requested save height profiles
    if config["grainstats"]["extract_height_profile"]:
        extract_height_profiles(
            topostats_object_all=topostats_object_all,
            output_dir=config["output_dir"],
            filename="height_profiles.json",
        )

    # Write config to file
    config["plotting"].pop("plot_dict")
    write_yaml(config, output_dir=config["output_dir"])
    LOGGER.debug(f"Images processed : {len(grain_stats_all)}")
    # Update config with plotting defaults for printing
    completion_message(config, img_files, summary_config=None, images_processed=grain_stats_all_df.shape[0])


def disordered_tracing(args: argparse.Namespace | None = None) -> None:
    """
    Load files from disk and run grainstats.

    Parameters
    ----------
    args : None
        Arguments.
    """
    config, img_files = _parse_configuration(args)  # pylint: disable=unused-variable
    run_disordered_tracing()


def nodestats(args: argparse.Namespace | None = None) -> None:
    """
    Load files from disk and run grainstats.

    Parameters
    ----------
    args : None
        Arguments.
    """
    config, img_files = _parse_configuration(args)  # pylint: disable=unused-variable
    run_nodestats()


def ordered_tracing(args: argparse.Namespace | None = None) -> None:
    """
    Load files from disk and run grainstats.

    Parameters
    ----------
    args : None
        Arguments.
    """
    config, img_files = _parse_configuration(args)  # pylint: disable=unused-variable
    run_ordered_tracing()


def splining(args: argparse.Namespace | None = None) -> None:
    """
    Load files from disk and run grainstats.

    Parameters
    ----------
    args : None
        Arguments.
    """
    config, img_files = _parse_configuration(args)  # pylint: disable=unused-variable
    run_splining()


def bruker_rename(args: argparse.Namespace | None = None) -> None:
    """
    Find files old-format Bruker files in the specified directory and append the suffix ``.spm``.

    Parameters
    ----------
    args : argparse.Namespace | None
        Arguments.
    """
    # Parse command line options, load config (or default) and update with command line options
    config = reconcile_config_args(args=args)

    # Validate configuration
    validate_config(config, schema=DEFAULT_CONFIG_SCHEMA, config_type="YAML configuration file")

    # Set logging level
    _set_logging(config["log_level"])

    assert (
        config["file_ext"] == ".spm"
    ), "Can only rename old .spm files, change your file-ext in config or command line"
    all_spm_files = find_files(config["base_dir"], file_ext=config["file_ext"])
    LOGGER.info(f"Total Bruker files found : {len(all_spm_files)}")
    OLD_BRUKER_RE = re.compile(r"\.\d+$")
    old_spm_files = [spm_file for spm_file in all_spm_files if OLD_BRUKER_RE.match(spm_file.suffix)]
    LOGGER.info(f"Old style files found    : {len(old_spm_files)}")
    LOGGER.info("Renaming files...")
    # Could rename files using list comprehension (no logging though)
    # [spm_file.rename(f"{spm_file}.spm") for spm_file in old_spm_files]
    # Instead loop with logging showing each rename
    for spm_file in old_spm_files:
        spm_file.rename(f"{spm_file}.spm")
        LOGGER.info(f"{spm_file.relative_to(config['base_dir'])} > {spm_file.relative_to(config['base_dir'])}.spm")
