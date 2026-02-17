"""Test the config module function(s)."""

import argparse
import logging
from pathlib import Path
from pkgutil import get_data

import pytest
import yaml

from topostats.config import (
    merge_mappings,
    reconcile_config_args,
    update_config,
    update_module,
    update_plotting_config,
    write_config_with_comments,
)
from topostats.logs.logs import LOGGER_NAME
from topostats.validation import DEFAULT_CONFIG_SCHEMA, validate_config

BASE_DIR = Path.cwd()

default_config = get_data(package="topostats", resource="default_config.yaml")
DEFAULT_CONFIG = yaml.full_load(default_config)


@pytest.mark.parametrize(
    ("config"),
    [
        pytest.param("default", id="default"),
        pytest.param("simple", id="simple"),
    ],
)
def test_reconcile_config_args_no_config(config: str) -> None:
    """Test the handling of config file function with no config."""
    args = argparse.Namespace(
        program="process",
        config_file=None,
        module="topostats",
        config=config,
    )
    config = reconcile_config_args(args=args, default_config=DEFAULT_CONFIG)

    # Check that the config passes the schema
    validate_config(config, schema=DEFAULT_CONFIG_SCHEMA, config_type="YAML configuration file")


def test_reconcile_config_args_no_config_with_overrides() -> None:
    """Test the handle config file function with no config and overrides."""
    args = argparse.Namespace(
        program="process",
        config_file=None,
        output_dir="./dummy_output_dir",
        module="topostats",
    )
    config = reconcile_config_args(args=args, default_config=DEFAULT_CONFIG)

    # Check that the overrides have been applied
    assert config["output_dir"] == Path("./dummy_output_dir")
    # Check that the config still passes the schema
    validate_config(config, schema=DEFAULT_CONFIG_SCHEMA, config_type="YAML configuration file")


def test_reconcile_config_args_full_config() -> None:
    """Test the handle config file function with a full config."""
    args = argparse.Namespace(
        program="process",
        config_file=f"{BASE_DIR / 'topostats' / 'default_config.yaml'}",
        module="topostats",
    )

    config = reconcile_config_args(args=args, default_config=DEFAULT_CONFIG)

    # Check that the config passes the schema
    validate_config(config, schema=DEFAULT_CONFIG_SCHEMA, config_type="YAML configuration file")


def test_reconcile_config_args_partial_config() -> None:
    """Test the reconcile_config_args function with a partial config."""
    args = argparse.Namespace(
        program="process",
        config_file=f"{BASE_DIR / 'tests' / 'resources' / 'test_partial_config.yaml'}",
        module="topostats",
    )
    config = reconcile_config_args(args=args, default_config=DEFAULT_CONFIG)

    # Check that the partial config has overridden the default config
    assert config["filter"]["threshold_method"] == "absolute"
    # Check that the config still passes the schema
    validate_config(config, schema=DEFAULT_CONFIG_SCHEMA, config_type="YAML configuration file")


def test_reconcile_config_args_partial_config_with_overrides() -> None:
    """Test the reconcile_config_args function with a partial config and overrides."""
    args = argparse.Namespace(
        program="process",
        config_file=f"{BASE_DIR / 'tests' / 'resources' / 'test_partial_config.yaml'}",
        output_dir="./dummy_output_dir",
        module="topostats",
    )
    config = reconcile_config_args(args=args, default_config=DEFAULT_CONFIG)

    # Check that the partial config has overridden the default config
    assert config["filter"]["threshold_method"] == "absolute"
    # Check that the overrides have been applied
    assert config["output_dir"] == Path("./dummy_output_dir")
    # Check that the config still passes the schema
    validate_config(config, schema=DEFAULT_CONFIG_SCHEMA, config_type="YAML configuration file")


@pytest.mark.parametrize(
    ("args", "modules", "arg_module"),
    [
        pytest.param(
            argparse.Namespace(module="process"),
            None,
            "topostats",
            id="topostats process",
        ),
        pytest.param(
            argparse.Namespace(module="filter"),
            None,
            "topostats",
            id="topostats filter",
        ),
        pytest.param(
            argparse.Namespace(module="grains"),
            None,
            "topostats",
            id="topostats grains",
        ),
        pytest.param(
            argparse.Namespace(module="grainstats"),
            None,
            "topostats",
            id="topostats grainstats",
        ),
        pytest.param(
            argparse.Namespace(module="disordered_tracing"),
            None,
            "topostats",
            id="topostats disordered_tracing",
        ),
        pytest.param(
            argparse.Namespace(module="nodestats"),
            None,
            "topostats",
            id="topostats nodestats",
        ),
        pytest.param(
            argparse.Namespace(module="ordered_tracing"),
            None,
            "topostats",
            id="topostats ordered_tracing",
        ),
        pytest.param(
            argparse.Namespace(module="splining"),
            None,
            "topostats",
            id="topostats splining",
        ),
        pytest.param(
            argparse.Namespace(module="curvature"),
            None,
            "topostats",
            id="topostats curvature",
        ),
        pytest.param(
            argparse.Namespace(module="afmslicer"),
            None,
            "afmslicer",
            id="afmslicer",
        ),
        pytest.param(
            argparse.Namespace(module="a_new_module"),
            (
                "a",
                "new",
                "module",
                "a_new_module",
            ),
            "topostats",  # if 'module' is in modules we expect 'topostats'
            id="custom module list",
        ),
    ],
)
def test_update_module(args: argparse.Namespace, modules: tuple, arg_module: str) -> None:
    """Test for config.update_module()."""
    # We want to test that the default module list of the function works
    if modules is None:
        update_module(args=args)
    # ...as well as custom lists
    else:
        update_module(args=args, topostats_modules=modules)
    assert args.module == arg_module


@pytest.mark.parametrize(
    ("dict1", "dict2", "expected_merged_dict"),
    [
        pytest.param(
            {"a": 1, "b": 2},
            {"c": 3, "d": 4},
            {"a": 1, "b": 2, "c": 3, "d": 4},
            id="two dicts, no common keys",
        ),
        pytest.param(
            {"a": 1, "b": 2},
            {"b": 3, "c": 4},
            {"a": 1, "b": 3, "c": 4},
            id="two dicts, one common key, testing priority of second dict",
        ),
        # Nested dictionaries
        pytest.param(
            {"a": 1, "b": {"c": 2, "d": 3}},
            {"b": {"c": 4, "e": 5}},
            {"a": 1, "b": {"c": 4, "d": 3, "e": 5}},
            id="nested dictionaries, one common key in nested dict, testing priority of second dict",
        ),
    ],
)
def test_merge_mappings(dict1: dict, dict2: dict, expected_merged_dict: dict) -> None:
    """Test merging of mappings."""
    merged_dict = merge_mappings(dict1, dict2)
    assert merged_dict == expected_merged_dict


@pytest.mark.parametrize(
    ("filename", "config", "expected_filename"),
    [
        pytest.param(
            "test_config_with_comments.yaml", None, "test_config_with_comments.yaml", id="filename, no config name"
        ),
        pytest.param(
            "test_config_with_comments",
            None,
            "test_config_with_comments.yaml",
            id="filename without yaml extension, no config name",
        ),
        pytest.param(None, "default", "config.yaml", id="no filename, default config"),
        pytest.param(None, None, "config.yaml", id="no filename, no config name"),
        pytest.param("test_simple_config.yaml", "simple", "test_simple_config.yaml", id="filename, simple config"),
        pytest.param(None, "simple", "simple_config.yaml", id="no filename, simple config"),
        pytest.param("custom.mplstyle", "mplstyle", "custom.mplstyle", id="filename, mplstyle"),
        pytest.param(None, "mplstyle", "topostats.mplstyle", id="no filename, mplstyle"),
        pytest.param(
            "custom_var_to_label.yaml", "var_to_label", "custom_var_to_label.yaml", id="filename, var_to_label"
        ),
        pytest.param(None, "var_to_label", "var_to_label.yaml", id="no filename, var_to_label"),
    ],
)
def test_write_config_with_comments(tmp_path: Path, filename: str, config: str, expected_filename: str) -> None:
    """Test writing of config file with comments.

    If and when specific configurations for different sample types are introduced then the parametrisation can be
    extended to allow these adding their names under "config" and introducing specific parameters that may differe
    between the configuration files.
    """
    # Setup argparse.Namespace with the tests parameters
    args = argparse.Namespace()
    args.filename = filename
    args.output_dir = tmp_path
    args.config = config
    args.module = "topostats"
    # Write default config with comments to file
    write_config_with_comments(args)
    assert Path(tmp_path / expected_filename).is_file()
    # Read the written config
    with Path.open(tmp_path / expected_filename, encoding="utf-8") as f:
        written_config = f.read()
    # Validate that the written config has comments in it
    assert "Config file generated" in written_config
    assert "For more information on configuration and how to use it" in written_config
    # Validate some of the parameters are present conditional on config type
    if config == "default":
        assert "loading:" in written_config
        assert "gaussian_mode: nearest" in written_config
        assert "style: topostats.mplstyle" in written_config
        assert "pixel_interpolation: null" in written_config
    elif config == "simple":
        assert "cores: 2" in written_config
        assert "row_alignment_quantile: 0.5" in written_config
    elif config == "mplstyle":
        assert "#### MATPLOTLIBRC FORMAT" in written_config
        assert "image.cmap:             nanoscope" in written_config
    elif config == "var_to_label":
        assert "grain_mean: Mean Height" in written_config
        assert "radius_median: Median Radius" in written_config


@pytest.mark.parametrize(
    ("module", "config", "error"),
    [
        pytest.param(
            "topostats",
            "nonsense",
            ValueError,
            id="topostats module for 'nonsense' config",
        ),
        pytest.param(
            "other_package",
            "default",
            AttributeError,
            id="other_package module for 'default' config",
        ),
    ],
)
def test_write_config_with_comments_user_warning(module: str, config: str, error, tmp_path: Path) -> None:
    """Tests if errors are raised when an attempt is made to request a configuration file type that does not exist."""
    args = argparse.Namespace()
    args.filename = "config.yaml"
    args.output_dir = tmp_path
    args.config = config
    args.module = module
    with pytest.raises(error):
        write_config_with_comments(args)


def test_update_config(caplog) -> None:
    """Test updating configuration."""
    caplog.set_level(logging.DEBUG, LOGGER_NAME)
    sample_config = {"a": 1, "b": 2, "c": "something", "base_dir": "here", "output_dir": "there"}
    new_values = {"c": "something new"}
    updated_config = update_config(config=sample_config, args=new_values)

    assert isinstance(updated_config, dict)
    assert "Updated config config[c] : something > something new" in caplog.text
    assert updated_config["c"] == "something new"


@pytest.mark.parametrize(
    ("image_name", "core_set", "title", "zrange"),
    [
        pytest.param("extracted_channel", False, "Raw Height", [0, 3], id="Non-binary image, not in core, with title"),
        pytest.param("z_threshed", True, "Height Thresholded", [0, 3], id="Non-binary image, in core, with title"),
        pytest.param(
            "grain_mask", False, "", [None, None], id="Binary image, not in core, set no title"
        ),  # binary image
        pytest.param(
            "grain_mask_image", False, "", [0, 3], id="Non-binary image, not in core, set no title"
        ),  # non-binary image
    ],
)
def test_update_plotting_config(
    process_scan_config: dict, image_name: str, core_set: bool, title: str, zrange: tuple
) -> None:
    """Ensure values are added to each image in plot_dict."""
    process_scan_config["plotting"] = update_plotting_config(process_scan_config["plotting"])
    assert process_scan_config["plotting"]["plot_dict"][image_name]["core_set"] == core_set
    # Only check titles for images that have titles. grain_image, grain_mask, grain_mask_image don't
    # have titles since they're created dynamically.
    if title in ["extracted_channel", "z_threshed"]:
        assert process_scan_config["plotting"]["plot_dict"][image_name]["title"] == title
    # Ensure that both types (binary, non-binary) of image have the correct z-ranges
    # ([None, None] for binary, user defined for non-binary)
    assert process_scan_config["plotting"]["plot_dict"][image_name]["zrange"] == zrange


@pytest.mark.parametrize(
    ("plotting_config", "target_config"),
    [
        pytest.param(
            {
                "run": True,
                "savefig_dpi": None,
                "pixel_interpolation": None,
                "plot_dict": {
                    "extracted_channel": {
                        "filename": "00-raw_heightmap",
                        "image_type": "non-binary",
                        "core_set": False,
                        "savefig_dpi": 100,
                    }
                },
            },
            {
                "run": True,
                "savefig_dpi": None,
                "pixel_interpolation": None,
                "plot_dict": {
                    "extracted_channel": {
                        "filename": "00-raw_heightmap",
                        "image_type": "non-binary",
                        "savefig_dpi": 100,
                        "core_set": False,
                        "pixel_interpolation": None,
                    }
                },
            },
            id="DPI None in main, extracted channel DPI should stay at 100",
        ),
        pytest.param(
            {
                "run": True,
                "savefig_dpi": 600,
                "pixel_interpolation": None,
                "plot_dict": {
                    "extracted_channel": {
                        "filename": "00-raw_heightmap",
                        "image_type": "non-binary",
                        "savefig_dpi": 100,
                        "core_set": False,
                    }
                },
            },
            {
                "run": True,
                "savefig_dpi": 600,
                "pixel_interpolation": None,
                "plot_dict": {
                    "extracted_channel": {
                        "filename": "00-raw_heightmap",
                        "image_type": "non-binary",
                        "savefig_dpi": 600,
                        "core_set": False,
                        "pixel_interpolation": None,
                    }
                },
            },
            id="DPI 600 in main, extracted channel DPI should update to 600",
        ),
    ],
)
def test_udpate_plotting_config_adding_required_options(plotting_config: dict, target_config: dict, caplog) -> None:
    """Only updates plotting_dict parameters from parent plotting config if value is not None."""
    caplog.set_level(logging.DEBUG, LOGGER_NAME)
    update_plotting_config(plotting_config)
    assert plotting_config == target_config
    if plotting_config["savefig_dpi"] == 600:
        assert "100 > 600" in caplog.text
