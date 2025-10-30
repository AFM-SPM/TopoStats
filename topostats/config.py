"""Functions and tools for working with configuration files."""

import logging
from argparse import Namespace
from collections.abc import MutableMapping
from datetime import datetime
from pathlib import Path
from pkgutil import get_data
from pprint import pformat
from typing import TypeVar

import yaml

from topostats import CONFIG_DOCUMENTATION_REFERENCE
from topostats.io import read_yaml
from topostats.logs.logs import LOGGER_NAME
from topostats.utils import convert_path

MutableMappingType = TypeVar("MutableMappingType", bound="MutableMapping")

LOGGER = logging.getLogger(LOGGER_NAME)


def reconcile_config_args(args: Namespace | None) -> dict:
    """
    Reconcile command line arguments with the default configuration.

    Command line arguments take precedence over the default configuration. If a partial configuration file is specified
    (with '-c' or '--config-file') the defaults are over-ridden by these values (internally the configuration
    dictionary is updated with these values). Any other command line arguments take precedence over both the default
    and those supplied in a configuration file (again the dictionary is updated).

    The final configuration is validated before processing begins.

    Parameters
    ----------
    args : Namespace
        Command line arguments passed into TopoStats.

    Returns
    -------
    dict
        The configuration dictionary.
    """
    update_module(args=args)
    default_config = get_data(package=args.module, resource="default_config.yaml")
    default_config = yaml.full_load(default_config)
    if args is not None:
        if args.config_file is not None:
            config = read_yaml(str(args.config_file))
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

    return config


def update_module(
    args: Namespace,
    topostats_modules: tuple = (
        "bruker-rename",
        "create-config",
        "curvature",
        "disordered_tracing",
        "filter",
        "grains",
        "grainstats",
        "nodestats",
        "ordered_tracing",
        "process",
        "splining",
    ),
) -> None:
    """
    Update the `args.module` argument if processing TopoStats objects.

    This function allows the sub-parser command to map to the pipeline we wish to use. For now TopoStats has sub-parsers
    but it is the intention to introduce sub-sub-parsers for other modules such that eventually we invoke ``topostats```
    with a module argument followed by the step of processing.

    >>> topostats topostats filter
    >>> topostats topostats process
    >>> topostats afmslicer slice
    >>> topostats afmslicer process
    >>> topostats perovstats process

    Parameters
    ----------
    args : Namespace
        Default arguments that need parsing and updating.
    topostats_modules : tuple
        List of module names that are unique to TopoStats.
    """
    if args.module in topostats_modules:
        args.module = "topostats"


def merge_mappings(map1: MutableMappingType, map2: MutableMappingType) -> MutableMappingType:
    """
    Merge two mappings (dictionaries), with priority given to the second mapping.

    Note: Using a Mapping should make this robust to any mapping type, not just dictionaries. MutableMapping was needed
    as Mapping is not a mutable type, and this function needs to be able to change the dictionaries.

    Parameters
    ----------
    map1 : MutableMapping
        First mapping to merge, with secondary priority.
    map2 : MutableMapping
        Second mapping to merge, with primary priority.

    Returns
    -------
    dict
        Merged dictionary.
    """
    # Iterate over the second mapping
    for key, value in map2.items():
        # If the value is another mapping, then recurse
        if isinstance(value, MutableMapping):
            # If the key is not in the first mapping, add it as an empty dictionary before recursing
            map1[key] = merge_mappings(map1.get(key, {}), value)
        else:
            # Else simply add / override the key value pair
            map1[key] = value
    return map1


def write_config_with_comments(args: Namespace = None) -> None:  # noqa: C901
    """
    Write a sample configuration with in-line comments.

    This function is not designed to be used interactively but can be, just call it without any arguments and it will
    write a configuration to './config.yaml'.

    Parameters
    ----------
    args : Namespace
        A Namespace object parsed from argparse with values for 'filename'.
    """
    output_dir = Path("./") if args.output_dir is None else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger_msg = "A sample configuration has been written to"
    # Update args.module
    update_module(args)
    if args.config == "default" or args.config is None:
        try:
            config = get_data(package=args.module, resource="default_config.yaml")
            filename = "default_config.yaml" if args.filename is None else args.filename
        except FileNotFoundError as exc:
            raise (
                FileNotFoundError(f"There is no configuration for module {args.module} called 'default_config.yaml'")
            ) from exc
    elif args.config == "simple":
        try:
            config = get_data(package=args.module, resource="simple_config.yaml")
            filename = "simple_config.yaml" if args.filename is None else args.filename
        except FileNotFoundError as exc:
            raise (
                FileNotFoundError(f"There is no configuration for module {args.module} called 'simple_config.yaml'")
            ) from exc
    elif args.config == "mplstyle":
        try:
            config = get_data(package=args.module, resource="topostats.mplstyle")
            filename = "topostats.mplstyle" if args.filename is None else args.filename
        except FileNotFoundError as exc:
            raise (
                FileNotFoundError(f"There is no configuration for module {args.module} called 'topostats.mplstyle'")
            ) from exc
    elif args.config == "var_to_label":
        try:
            config = get_data(package=args.module, resource="var_to_label.yaml")
            filename = "var_to_label.yaml" if args.filename is None else args.filename
        except FileNotFoundError as exc:
            raise (
                FileNotFoundError(f"There is no configuration for module {args.module} called 'var_to_label.yaml'")
            ) from exc
    else:
        valid_config = ["default", "simple", "mplstyle", "var_to_label"]
        raise ValueError(f"Invalid configuration file option, valid options are\n{valid_config}")

    if ".yaml" not in str(filename) and ".yml" not in str(filename) and ".mplstyle" not in str(filename):
        config_path = output_dir / f"{filename}.yaml"
    else:
        config_path = output_dir / filename

    try:
        with config_path.open("w", encoding="utf-8") as f:
            f.write(f"# Config file generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{CONFIG_DOCUMENTATION_REFERENCE}")
            f.write(config.decode("utf-8"))
    except AttributeError as e:
        raise e

    LOGGER.info(f"{logger_msg} : {str(config_path)}")


def update_config(config: dict, args: dict | Namespace) -> dict:
    """
    Update the configuration with any arguments.

    Parameters
    ----------
    config : dict
        Dictionary of configuration (typically read from YAML file specified with '-c/--config <filename>').
    args : Namespace
        Command line arguments.

    Returns
    -------
    dict
        Dictionary updated with command arguments.
    """
    args = vars(args) if isinstance(args, Namespace) else args

    config_keys = config.keys()
    for arg_key, arg_value in args.items():
        if isinstance(arg_value, dict):
            update_config(config, arg_value)
        else:
            if arg_key in config_keys and arg_value is not None:
                original_value = config[arg_key]
                config[arg_key] = arg_value
                LOGGER.debug(f"Updated config config[{arg_key}] : {original_value} > {arg_value} ")
    if "base_dir" in config.keys():
        config["base_dir"] = convert_path(config["base_dir"])
    if "output_dir" in config.keys():
        config["output_dir"] = convert_path(config["output_dir"])
    return config


def update_plotting_config(plotting_config: dict) -> dict:
    """
    Update the plotting config for each of the plots in plot_dict.

    Ensures that each entry has all the plotting configuration values that are needed.

    Parameters
    ----------
    plotting_config : dict
        Plotting configuration to be updated.

    Returns
    -------
    dict
        Updated plotting configuration.
    """
    main_config = plotting_config.copy()
    for opt in ["plot_dict", "run"]:
        main_config.pop(opt)
    LOGGER.debug(
        f"Main plotting options that need updating/adding to plotting dict :\n{pformat(main_config, indent=4)}"
    )
    for image, options in plotting_config["plot_dict"].items():
        main_config_temp = main_config.copy()
        LOGGER.debug(f"Dictionary for image : {image}")
        LOGGER.debug(f"{pformat(options, indent=4)}")
        # First update options with values that exist in main_config
        # We must however be careful not to update the colourmap for diagnostic traces
        if (
            not plotting_config["plot_dict"][image]["core_set"]
            and "mask_cmap" in plotting_config["plot_dict"][image].keys()
        ):
            main_config_temp.pop("mask_cmap")
        plotting_config["plot_dict"][image] = update_config(options, main_config_temp)
        LOGGER.debug(f"Updated values :\n{pformat(plotting_config['plot_dict'][image])}")
        # Then combine the remaining key/values we need from main_config that don't already exist
        for key_main, value_main in main_config_temp.items():
            if key_main not in plotting_config["plot_dict"][image]:
                plotting_config["plot_dict"][image][key_main] = value_main
        LOGGER.debug(f"After adding missing configuration options :\n{pformat(plotting_config['plot_dict'][image])}")
        # Make it so that binary images do not have the user-defined z-scale
        # applied, but non-binary images do.
        if plotting_config["plot_dict"][image]["image_type"] == "binary":
            plotting_config["plot_dict"][image]["zrange"] = [None, None]

    return plotting_config
