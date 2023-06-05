"""Entry point for all TopoStats programs. Parses command-line arguments and passes input on to the relevant
functions / modules.
"""

import sys
import argparse as arg

from topostats import __version__
from topostats.run_topostats import run_topostats
from topostats.plotting import run_toposum


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

    # run_topostats parser
    process_parser = subparsers.add_parser(
        "process",
        description="Process AFM images. Additional arguments over-ride those in the configuration file.",
        help="Process AFM images. Additional arguments over-ride those in the configuration file.",
    )
    process_parser.add_argument(
        "-c",
        "--config_file",
        dest="config_file",
        required=False,
        help="Path to a YAML configuration file.",
    )
    process_parser.add_argument(
        "--create_config_file",
        dest="create_config_file",
        required=False,
        help="Filename to write a sample YAML configuration file to (should end in '.yaml').",
    )
    process_parser.add_argument(
        "-s",
        "--summary_config",
        dest="summary_config",
        required=False,
        help="Path to a YAML configuration file for summary plots and statistics.",
    )
    process_parser.add_argument(
        "-b",
        "--base_dir",
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
        "--log_level",
        dest="log_level",
        type=str,
        required=False,
        help="Logging level to use, default is 'info' for verbose output use 'debug'.",
    )
    process_parser.add_argument(
        "-f",
        "--file_ext",
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
        "--output_dir",
        dest="output_dir",
        type=str,
        required=False,
        help="Output directory to write results to.",
    )
    process_parser.add_argument(
        "--save_plots",
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

    # toposum parser
    toposum_parser = subparsers.add_parser(
        "summary",
        description="Plotting and summary of TopoStats output statistics.",
        help="Plotting and summary of TopoStats output statistics.",
    )
    toposum_parser.add_argument("-i", "--input_csv", dest="csv_file", required=False, help="Path to CSV file to plot.")
    toposum_parser.add_argument(
        "-c",
        "--config_file",
        dest="config_file",
        required=False,
        help="Path to a YAML plotting dictionary that maps variable names to labels.",
    )
    toposum_parser.add_argument(
        "-l",
        "--var_to_label",
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

    return parser


def entry_point(manually_provided_args=None, testing=False) -> None:
    """Entry point for all TopoStats programs."""

    # Parse command line options, load config (or default) and update with command line options
    parser = create_parser()
    args = parser.parse_args() if manually_provided_args is None else parser.parse_args(manually_provided_args)
    if not args.program:
        # no program specified, print help and exit
        parser.print_help()
        sys.exit()

    if testing:
        return args

    # call the relevant function
    args.func(args)

    return None


def create_legacy_process_parser() -> arg.ArgumentParser:
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


def legacy_process_entry_point(args=None) -> None:
    """Legacy entry point for the run_topostats processing function."""

    parser = create_legacy_process_parser()
    args = parser.parse_args() if args is None else parser.parse_args(args)
    run_topostats(args=args)


def legacy_toposum_entry_point(args=None) -> None:
    """Legacy entry point for the toposum summarizing function."""
    parser = create_legacy_toposum_parser()
    args = parser.parse_args() if args is None else parser.parse_args(args)
    run_toposum(args=args)
