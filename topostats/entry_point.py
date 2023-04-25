"""Entry point for all TopoStats programs. Parses command-line arguments and passes input on to the relevant
functions / modules."""

import argparse as arg


def dummy_run(args):
    """Dummy function for testing parsers' delegation."""
    print("running topostats")
    print("args:")
    print(args)
    print(args.config_file)


def dummy_toposum(args):
    """Dummy function for testing parsers' delegation."""
    print("running toposum")
    print("args:")
    print(args)
    print(args.config_file)


def create_parser() -> arg.ArgumentParser:
    """Create a parser for reading options."""
    parser = arg.ArgumentParser(
        description="Run various programs relating to AFM data. Add the name of the program you wish to run."
    )

    subparsers = parser.add_subparsers(
        title="programs", description="Available programs, listed below:", dest="programs"
    )

    # run_topostats parser
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument(
        "-c", "--config_file", dest="config_file", required=False, help="Path to a YAML configuration file."
    )
    run_parser.set_defaults(func=dummy_run)

    # toposum parser
    toposum_parser = subparsers.add_parser("summary")
    toposum_parser.add_argument(
        "-c",
        "--config_file",
        dest="config_file",
        required=False,
        help="Path to a YAML plotting dictionary that maps variable names to labels.",
    )
    toposum_parser.set_defaults(func=dummy_toposum)

    return parser


def main():
    """Entry point for all TopoStats programs."""

    print("running topostats main command")

    parser = create_parser()
    args = parser.parse_args()
    print("args:")
    print(args)
    args.func(args)
