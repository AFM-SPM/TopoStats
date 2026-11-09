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
from pkgutil import get_data
from pprint import pformat

import pandas as pd
import yaml
from tqdm import tqdm

import topostats
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
    default_config = get_data(package=topostats.__package__, resource="default_config.yaml")
    default_config = yaml.full_load(default_config)
    config = reconcile_config_args(args=args, default_config=default_config)

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
        curvature_run=config["curvature"]["run"],
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
        close_strand_correction_config=config["close_strand_correction"],
        nodestats_config=config["nodestats"],
        ordered_tracing_config=config["ordered_tracing"],
        splining_config=config["splining"],
        curvature_config=config["curvature"],
        plotting_config=config["plotting"],
        grain_qa_config=config["grain_qa"],
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

    # ns-rse 2025-12-23 - there is a common pattern here, could we abstract this to a factory method?
    if len(grain_stats_all) > 0:
        try:
            grain_stats_all = pd.concat(grain_stats_all.values())
            grain_stats_all = grain_stats_all.reset_index()
            grain_stats_all = grain_stats_all.set_index(["image", "grain_number"])
        except ValueError as error:
            LOGGER.error("No grains found in any images, consider adjusting your thresholds.")
            LOGGER.error(error)
        # Write statistics to CSV if there is data.
        if isinstance(grain_stats_all, pd.DataFrame) and not grain_stats_all.isna().values.all():
            # debugging: save stats of success to csv
            # count the number of rows with non NaN "total_contour_length" in the grain_stats_df
            if grain_stats_all is not None:
                from topostats import TOPOSTATS_COMMIT

                # get the current git commit
                total_contour_length_minimum_threshold = 400e-9
                total_contour_length_maximum_threshold = 460e-9
                num_within_total_contour_length_threshold = (
                    grain_stats_all["total_contour_length"]
                    .between(total_contour_length_minimum_threshold, total_contour_length_maximum_threshold)
                    .sum()
                )

                num_successfully_processed = grain_stats_all["total_contour_length"].notnull().sum()
                num_grains_total = len(grain_stats_all)
                num_within_total_contour_length_threshold = (
                    grain_stats_all["total_contour_length"]
                    .between(total_contour_length_minimum_threshold, total_contour_length_maximum_threshold)
                    .sum()
                )
                text = ""
                text += f"TopoStats commit: {TOPOSTATS_COMMIT}\n"
                text += f"Number of grains successfully processed: {num_successfully_processed} / {num_grains_total} ({num_successfully_processed / num_grains_total * 100:.2f}%)\n"
                text += f"Number of successful grains with total_contour_length between {total_contour_length_minimum_threshold} and {total_contour_length_maximum_threshold}: {num_within_total_contour_length_threshold} / {num_successfully_processed} ({num_within_total_contour_length_threshold / num_successfully_processed * 100:.2f}%)\n"
                LOGGER.info(text)
                # save to file too
                with open(config["output_dir"] / "grain_stats_qa.txt", "w") as f:
                    f.write(text)
            grain_stats_all = write_csv(
                df=grain_stats_all,
                dataset="grain_stats",
                # Reset after above merge
                names=["image", "grain_number"],
                index=["image", "grain_number", "class", "subgrain"],
                output_dir=config["output_dir"],
            )
            LOGGER.info(f"Saved grain stats to : {config['output_dir']}/grain_statistics.csv.")
    else:
        images_processed = 0
        LOGGER.warning("There are no grainstats statistics to write to CSV.")

    # Optional output files
    if output_full_stats:
        if branch_stats_all:
            # Matched branch statistics
            try:
                branch_stats_all = pd.concat(branch_stats_all.values())
            except ValueError as error:
                LOGGER.error(error)
            if isinstance(branch_stats_all, pd.DataFrame) and not branch_stats_all.isna().values.all():
                branch_stats_all = write_csv(
                    df=branch_stats_all,
                    dataset="matched_branch_stats",
                    names=["grain_number", "node", "branch"],
                    index=["image", "grain_number", "node", "branch"],
                    output_dir=config["output_dir"],
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

    # Update config with plotting defaults for printing
    completion_message(config, img_files, images_processed)


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
    completion_message(config, img_files, images_processed=sum(results.values()))


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
    completion_message(config, img_files, images_processed=sum(results.values()))


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
    completion_message(config, img_files, images_processed=grain_stats_all_df.shape[0])


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
    default_config = get_data(package=topostats.__package__, resource="default_config.yaml")
    default_config = yaml.full_load(default_config)
    config = reconcile_config_args(args=args, default_config=default_config)

    # Validate configuration
    validate_config(config, schema=DEFAULT_CONFIG_SCHEMA, config_type="YAML configuration file")

    # Set logging level
    _set_logging(config["log_level"])

    assert config["file_ext"] == ".spm", (
        "Can only rename old .spm files, change your file-ext in config or command line"
    )
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
