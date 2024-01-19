"""Entry point for all TopoStats programs.

Parses command-line arguments and passes input on to the relevant functions / modules.
"""
import argparse as arg
import sys

from topostats import __version__
from topostats.io import write_config_with_comments
from topostats.plotting import run_toposum
from topostats.run_topostats import run_topostats


def create_parser() -> arg.ArgumentParser:
    """Create a parser for reading options."""
    parser = arg.ArgumentParser(
        description="Run various programs relating to AFM data. Add the name of the program you wish to run."
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"Installed version of TopoStats: {__version__}",
        help="Report the current version of TopoStats that is installed",
    )

    subparsers = parser.add_subparsers(title="program", description="Available programs, listed below:", dest="program")

    # Create a sub-parsers for different stages of processing and tasks
    process_parser = subparsers.add_parser(
        "process",
        description="Process AFM images. Additional arguments over-ride those in the configuration file.",
        help="Process AFM images. Additional arguments over-ride those in the configuration file.",
    )
    process_parser.add_argument(
        "-c",
        "--config-file",
        dest="config_file",
        required=False,
        help="Path to a YAML configuration file.",
    )
    process_parser.add_argument(
        "-s",
        "--summary-config",
        dest="summary_config",
        required=False,
        help="Path to a YAML configuration file for summary plots and statistics.",
    )
    process_parser.add_argument(
        "--matplotlibrc",
        dest="matplotlibrc",
        required=False,
        help="Path to a matplotlibrc file.",
    )
    process_parser.add_argument(
        "-b",
        "--base-dir",
        dest="base_dir",
        type=str,
        required=False,
        help="Base directory to scan for images.",
    )
    process_parser.add_argument(
        "-j",
        "--cores",
        dest="cores",
        type=int,
        required=False,
        help="Number of CPU cores to use when processing.",
    )
    process_parser.add_argument(
        "-l",
        "--log-level",
        dest="log_level",
        type=str,
        required=False,
        help="Logging level to use, default is 'info' for verbose output use 'debug'.",
    )
    process_parser.add_argument(
        "-f",
        "--file-ext",
        dest="file_ext",
        type=str,
        required=False,
        help="File extension to scan for.",
    )
    process_parser.add_argument(
        "--channel",
        dest="channel",
        type=str,
        required=False,
        help="Channel to extract.",
    )
    process_parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        type=str,
        required=False,
        help="Output directory to write results to.",
    )
    process_parser.add_argument(
        "--save-plots",
        dest="save_plots",
        type=bool,
        required=False,
        help="Whether to save plots.",
    )
    process_parser.add_argument("-m", "--mask", dest="mask", type=bool, required=False, help="Mask the image.")
    process_parser.add_argument(
        "-w",
        "--warnings",
        dest="warnings",
        type=bool,
        required=False,
        help="Whether to ignore warnings.",
    )
    process_parser.set_defaults(func=run_topostats)

    toposum_parser = subparsers.add_parser(
        "summary",
        description="Plotting and summary of TopoStats output statistics.",
        help="Plotting and summary of TopoStats output statistics.",
    )
    toposum_parser.add_argument("-i", "--input_csv", dest="csv_file", required=False, help="Path to CSV file to plot.")
    toposum_parser.add_argument(
        "-c",
        "--config-file",
        dest="config_file",
        required=False,
        help="Path to a YAML plotting dictionary that maps variable names to labels.",
    )
    toposum_parser.add_argument(
        "-l",
        "--var-to-label",
        dest="var_to_label",
        required=False,
        help="Path to a YAML plotting dictionary that maps variable names to labels.",
    )
    toposum_parser.add_argument(
        "--create-config-file",
        dest="create_config_file",
        type=str,
        required=False,
        help="Filename to write a sample YAML configuration file to (should end in '.yaml').",
    )
    toposum_parser.add_argument(
        "--create-label-file",
        dest="create_label_file",
        type=str,
        required=False,
        help="Filename to write a sample YAML label file to (should end in '.yaml').",
    )
    toposum_parser.set_defaults(func=run_toposum)

    load_parser = subparsers.add_parser(
        "load",
        description="Load and save all images as .topostats files for subsequent processing.",
        help="Load and save all images as .topostats files for subsequent processing.",
    )
    load_parser.add_argument(
        "-c",
        "--config-file",
        dest="config_file",
        required=False,
        help="Path to a YAML configuration file.",
    )

    filter_parser = subparsers.add_parser(
        "filter",
        description="Load and filter images, saving as .topostats files for subsequent processing.",
        help="Load and filter images, saving as .topostats files for subsequent processing.",
    )
    filter_parser.add_argument(
        "-c",
        "--config-file",
        dest="config_file",
        required=False,
        help="Path to a YAML configuration file.",
    )

    grain_parser = subparsers.add_parser(
        "grains",
        description="Load filtered images from '.topostats' files and detect grains.",
        help="Load filtered images from '.topostats' files and detect grains.",
    )
    grain_parser.add_argument(
        "-c",
        "--config-file",
        dest="config_file",
        required=False,
        help="Path to a YAML configuration file.",
    )

    grainstats_parser = subparsers.add_parser(
        "grainstats",
        description="Load images with grains from '.topostats' files and calculate statistics.",
        help="Load images with grains from '.topostats' files and calculate statistics.",
    )
    grainstats_parser.add_argument(
        "-c",
        "--config-file",
        dest="config_file",
        required=False,
        help="Path to a YAML configuration file.",
    )

    dnatracing_parser = subparsers.add_parser(
        "dnatracing",
        description="Load images with grains from '.topostats' files and trace DNA molecules.",
        help="Load images with grains from '.topostats' files and trace DNA molecules.",
    )
    dnatracing_parser.add_argument(
        "-c",
        "--config-file",
        dest="config_file",
        required=False,
        help="Path to a YAML configuration file.",
    )

    tracingstats_parser = subparsers.add_parser(
        "tracingstats",
        description="Load images with grains from '.topostats' files and trace DNA molecules.",
        help="Load images with grains from '.topostats' files and trace DNA molecules.",
    )
    tracingstats_parser.add_argument(
        "-c",
        "--config-file",
        dest="config_file",
        required=False,
        help="Path to a YAML configuration file.",
    )

    create_config_parser = subparsers.add_parser(
        "create-config",
        description="Create a configuration file using the defaults.",
        help="Create a configuration file using the defaults.",
    )
    create_config_parser.add_argument(
        "-f",
        "--filename",
        dest="filename",
        required=False,
        default="config.yaml",
        help="Name of YAML file to save configuration to (default 'config.yaml').",
    )
    create_config_parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        required=False,
        default="./",
        help="Path to where the YAML file should be saved (default './' the current directory).",
    )
    create_config_parser.add_argument(
        "-c",
        "--config",
        dest="config",
        default=None,
        help="Configuration to use, currently only one is supported, the 'default'.",
    )
    create_config_parser.set_defaults(func=write_config_with_comments)

    create_matplotlibrc_parser = subparsers.add_parser(
        "create-matplotlibrc",
        description="Create a Matplotlibrc parameters file.",
        help="Create a Matplotlibrc parameters file using the defaults.",
    )
    create_matplotlibrc_parser.add_argument(
        "-f",
        "--filename",
        dest="filename",
        required=False,
        default="topostats.mplstyle",
        help="Name of file to save Matplotlibrc configuration to (default 'topostats.mplstyle').",
    )
    create_matplotlibrc_parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        required=False,
        default="./",
        help="Path to where the YAML file should be saved (default './' the current directory).",
    )
    create_matplotlibrc_parser.add_argument(
        "-c",
        "--config",
        dest="config",
        default="topostats.mplstyle",
        help="Matplotlibrc style file to use, currently only one is supported, the 'topostats.mplstyle'.",
    )
    create_matplotlibrc_parser.set_defaults(func=write_config_with_comments)

    return parser


def entry_point(manually_provided_args=None, testing=False) -> None:
    """Entry point for all TopoStats programs."""
    # Parse command line options, load config (or default) and update with command line options
    parser = create_parser()
    args = parser.parse_args() if manually_provided_args is None else parser.parse_args(manually_provided_args)

    # No program specified, print help and exit
    if not args.program:
        parser.print_help()
        sys.exit()

    if testing:
        return args

    # call the relevant function
    args.func(args)

    return None


def create_legacy_run_topostats_parser() -> arg.ArgumentParser:
    """Create a parser reading options for the 'run_topostats' processing entry point."""
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


def create_legacy_toposum_parser() -> arg.ArgumentParser:
    """Create a parser reading options for the legacy 'toposum' summarize entry point."""
    parser = arg.ArgumentParser(
        description="Summarise and plot histograms, kernel density estimates and scatter plots of TopoStats"
        "grain and DNA Tracing statistics."
    )
    parser.add_argument("-i", "--input_csv", dest="csv_file", required=False, help="Path to CSV file to plot.")
    parser.add_argument(
        "-c",
        "--config_file",
        dest="config_file",
        required=False,
        help="Path to a YAML configuration file.",
    )
    parser.add_argument(
        "-l",
        "--var_to_label",
        dest="var_to_label",
        required=False,
        help="Path to a YAML plotting dictionary that maps variable names to labels.",
    )
    parser.add_argument(
        "--create-config-file",
        dest="create_config_file",
        type=str,
        required=False,
        help="Filename to write a sample YAML configuration file to (should end in '.yaml').",
    )
    parser.add_argument(
        "--create-label-file",
        dest="create_label_file",
        type=str,
        required=False,
        help="Filename to write a sample YAML label file to (should end in '.yaml').",
    )
    return parser


def legacy_run_topostats_entry_point(args=None, testing=False) -> None:
    """Legacy entry point for the run_topostats processing function."""
    parser = create_legacy_run_topostats_parser()
    args = parser.parse_args() if args is None else parser.parse_args(args)

    if testing:
        return args

    run_topostats(args=args)

    return None


def legacy_toposum_entry_point(args=None, testing=False) -> None:
    """Legacy entry point for the toposum summarizing function."""
    parser = create_legacy_toposum_parser()
    args = parser.parse_args() if args is None else parser.parse_args(args)

    if testing:
        return args

    run_toposum(args=args)

    return None
