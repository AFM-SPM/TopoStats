"""Test end-to-end running of topostats."""

import argparse
import logging
from pathlib import Path

import pandas as pd
import pytest
from AFMReader import topostats

from topostats.entry_point import entry_point
from topostats.logs.logs import LOGGER_NAME
from topostats.run_modules import _set_logging, reconcile_config_args
from topostats.validation import DEFAULT_CONFIG_SCHEMA, validate_config

BASE_DIR = Path.cwd()


def test_reconcile_config_args_no_config() -> None:
    """Test the handle config file function with no config."""
    args = argparse.Namespace(
        program="process",
        config_file=None,
    )
    config = reconcile_config_args(args=args)

    # Check that the config passes the schema
    validate_config(config, schema=DEFAULT_CONFIG_SCHEMA, config_type="YAML configuration file")


def test_reconcile_config_args_no_config_with_overrides() -> None:
    """Test the handle config file function with no config and overrides."""
    args = argparse.Namespace(
        program="process",
        config_file=None,
        output_dir="./dummy_output_dir",
    )
    config = reconcile_config_args(args=args)

    # Check that the overrides have been applied
    assert config["output_dir"] == Path("./dummy_output_dir")
    # Check that the config still passes the schema
    validate_config(config, schema=DEFAULT_CONFIG_SCHEMA, config_type="YAML configuration file")


def test_reconcile_config_args_full_config() -> None:
    """Test the handle config file function with a full config."""
    args = argparse.Namespace(program="process", config_file=f"{BASE_DIR / 'topostats' / 'default_config.yaml'}")

    config = reconcile_config_args(args=args)

    # Check that the config passes the schema
    validate_config(config, schema=DEFAULT_CONFIG_SCHEMA, config_type="YAML configuration file")


def test_reconcile_config_args_partial_config() -> None:
    """Test the reconcile_config_args function with a partial config."""
    args = argparse.Namespace(
        program="process", config_file=f"{BASE_DIR / 'tests' / 'resources' / 'test_partial_config.yaml'}"
    )
    config = reconcile_config_args(args=args)

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
    )
    config = reconcile_config_args(args=args)

    # Check that the partial config has overridden the default config
    assert config["filter"]["threshold_method"] == "absolute"
    # Check that the overrides have been applied
    assert config["output_dir"] == Path("./dummy_output_dir")
    # Check that the config still passes the schema
    validate_config(config, schema=DEFAULT_CONFIG_SCHEMA, config_type="YAML configuration file")


@pytest.mark.parametrize(
    ("log_level", "effective_level"),
    [
        pytest.param("debug", 10, id="log level debug"),
        pytest.param("info", 20, id="log level warning"),
        pytest.param("warning", 30, id="log level warning"),
        pytest.param("error", 40, id="log level error"),
    ],
)
def test_set_logging(log_level: str, effective_level: int) -> None:
    """Test setting log-level."""
    LOGGER = logging.getLogger(LOGGER_NAME)
    _set_logging(log_level)
    assert LOGGER.getEffectiveLevel() == effective_level


@pytest.mark.parametrize("option", [("-h"), ("--help")])
def test_run_topostats_main_help(capsys, option) -> None:
    """Test the -h/--help flag to run_topostats."""
    try:
        entry_point(manually_provided_args=["process", option])
    except SystemExit:
        pass
    assert "Process AFM images." in capsys.readouterr().out


def test_run_topostats_process_all(caplog) -> None:
    """Test run_topostats completes without error when no arguments are given."""
    caplog.set_level(logging.INFO)
    # Explicitly force loading of topostats/default_config.yaml as I couldn't work out how to invoke process_all()
    # without any arguments as it defaults to 'sys.argv' as this is wrapped within pytest it picks up the arguments
    # pytest was invoked with (see thread on StackOverflow at https://stackoverflow.com/a/55260580/1444043)
    entry_point(
        manually_provided_args=[
            "--config",
            f"{BASE_DIR / 'topostats' / 'default_config.yaml'}",
            "--base-dir",
            "./tests/resources/test_image/",
            "--file-ext",
            ".topostats",
            "--extract",
            "all",
            "process",
        ]
    )
    assert "~~~~~~~~~~~~~~~~~~~~ COMPLETE ~~~~~~~~~~~~~~~~~~~~" in caplog.text
    assert "Successfully Processed^1    : 1 (100.0%)" in caplog.text


def test_run_topostats_process_debug(caplog) -> None:
    """Test run_topostats with debugging and check DEBUG messages are logged."""
    # Set the logging level of the topostats logger
    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        entry_point(
            manually_provided_args=[
                "--config",
                f"{BASE_DIR / 'topostats' / 'default_config.yaml'}",
                "-l",
                "debug",
                "--base-dir",
                "./tests/resources/test_image/",
                "--file-ext",
                ".topostats",
                "process",
            ]
        )
        assert "Configuration after update         :" in caplog.text
        assert "File extension : .topostats" in caplog.text
        assert "Images processed : 1" in caplog.text
        assert "~~~~~~~~~~~~~~~~~~~~ COMPLETE ~~~~~~~~~~~~~~~~~~~~" in caplog.text


def test_filters(caplog) -> None:
    """Test running the filters module.

    We use the command line entry point to test that _just_ filters runs.
    """
    caplog.set_level(logging.INFO)
    entry_point(
        manually_provided_args=[
            "--config",
            f"{BASE_DIR / 'topostats' / 'default_config.yaml'}",
            "--base-dir",
            "./tests/resources/test_image/",
            "--file-ext",
            ".topostats",
            "filter",  # This is the sub-command we wish to test, it will call run_modules.filters()
        ]
    )
    assert "Looking for images with extension   : .topostats" in caplog.text
    assert "[minicircle_small] Filtering completed." in caplog.text
    # Load the output and check the keys
    data = topostats.load_topostats("output/processed/minicircle_small.topostats")
    assert list(data.keys()) == [
        "filename",
        "image",
        "image_original",
        "img_path",
        "pixel_to_nm_scaling",
        "topostats_file_version",
    ]


def test_grains(caplog) -> None:
    """Test running the grains module.

    We use the command line entry point to test that _just_ grains runs.
    """
    caplog.set_level(logging.INFO)
    entry_point(
        manually_provided_args=[
            "--config",
            f"{BASE_DIR / 'topostats' / 'default_config.yaml'}",
            "--base-dir",
            "./tests/resources/test_image/",
            "--file-ext",
            ".topostats",
            "grains",  # This is the sub-command we wish to test, it will call run_modules.grains()
        ]
    )
    assert "Looking for images with extension   : .topostats" in caplog.text
    assert "[minicircle_small] Grain detection completed (NB - Filtering was *not* re-run)." in caplog.text
    # Load the output and check the keys
    data = topostats.load_topostats("output/processed/minicircle_small.topostats")
    assert list(data.keys()) == [
        "filename",
        "grain_tensors",
        "image",
        "image_original",
        "img_path",
        "pixel_to_nm_scaling",
        "topostats_file_version",
    ]


@pytest.mark.xfail(reason="Awaiting update of AFMReader to reconstruct `image_grain_crops` with correct classes")
def test_grainstats(caplog) -> None:
    """Test running the grainstats module.

    We use the command line entry point to test that _just_ grains runs.
    """
    caplog.set_level(logging.INFO)
    entry_point(
        manually_provided_args=[
            "--config",
            f"{BASE_DIR / 'topostats' / 'default_config.yaml'}",
            "--base-dir",
            "./tests/resources/test_image/",
            "--file-ext",
            ".topostats",
            "grainstats",  # This is the sub-command we wish to test, it will call run_modules.grains()
        ]
    )
    assert "Looking for images with extension   : .topostats" in caplog.text
    assert "[minicircle_small] Grainstats completed (NB - Filtering was *not* re-run)." in caplog.text
    # Load the output and check the keys
    data = pd.read_csv("output/image_stats.csv")
    assert list(data.columns) == [
        "Unnamed: 0",
        "image",
        "basename",
        "grain_number",
        "area",
        "area_cartesian_bbox",
        "aspect_ratio",
        "bending_angle",
        "centre_x",
        "centre_y",
        "circular",
        "contour_length",
        "end_to_end_distance",
        "height_max",
        "height_mean",
        "height_median",
        "height_min",
        "max_feret",
        "min_feret",
        "radius_max",
        "radius_mean",
        "radius_median",
        "radius_min",
        "smallest_bounding_area",
        "smallest_bounding_length",
        "smallest_bounding_width",
        "threshold",
        "volume",
    ]
    assert data.shape == (3, 23)
