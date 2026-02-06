"""Test end-to-end running of topostats."""

import logging
from pathlib import Path

import pandas as pd
import pytest
from AFMReader import topostats
from syrupy.matchers import path_type

from topostats.classes import GrainCrop, TopoStats
from topostats.entry_point import entry_point
from topostats.io import dict_to_topostats
from topostats.logs.logs import LOGGER_NAME
from topostats.run_modules import _set_logging

BASE_DIR = Path.cwd()


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


@pytest.mark.parametrize(
    ("attributes"),
    [
        pytest.param(
            [
                "filename",
                "image",
                "image_original",
                "img_path",
                "pixel_to_nm_scaling",
                "config",
                "grain_crops",
                "topostats_version",
            ],
            id="file_version <= 2.3",
        ),
    ],
)
def test_filters(attributes: dict, caplog) -> None:
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
    # Load the output file with AFMReader check its a dictionary and convert to TopoStats
    data = topostats.load_topostats("output/processed/topostats/minicircle_small.topostats")
    assert isinstance(data, dict)
    topostats_object = dict_to_topostats(dictionary=data)
    assert isinstance(topostats_object, TopoStats)
    for attribute in attributes:
        assert hasattr(topostats_object, attribute)


@pytest.mark.parametrize(
    ("attributes"),
    [
        pytest.param(
            [
                "config",
                "filename",
                "full_mask_tensor",
                "grain_crops",
                "image",
                "image_original",
                "img_path",
                "pixel_to_nm_scaling",
                "topostats_version",
            ],
            id="running grains",
        ),
    ],
)
def test_grains(attributes: dict, caplog, tmp_path: Path) -> None:
    """Test running the grains module.

    We use the command line entry point to test that _just_ grains runs.
    """
    caplog.set_level(logging.DEBUG)
    entry_point(
        manually_provided_args=[
            "--config",
            f"{BASE_DIR / 'topostats' / 'default_config.yaml'}",
            "--base-dir",
            "./tests/resources/test_image/",
            "--file-ext",
            ".topostats",
            "--output-dir",
            f"{tmp_path}/output",
            "grains",  # This is the sub-command we wish to test, it will call run_modules.grains()
        ]
    )
    assert "Looking for images with extension   : .topostats" in caplog.text
    assert "[minicircle_small] Grain detection completed (NB - Filtering was *not* re-run)." in caplog.text
    # Load the output file with AFMReader check its a dictionary and convert to TopoStats
    data = topostats.load_topostats(tmp_path / "output/processed/topostats/minicircle_small.topostats")
    assert isinstance(data, dict)
    topostats_object = dict_to_topostats(dictionary=data)
    assert isinstance(topostats_object, TopoStats)
    for attribute in attributes:
        assert hasattr(topostats_object, attribute)
    for _, grain_crop in topostats_object.grain_crops.items():
        assert isinstance(grain_crop, GrainCrop)


def test_grainstats(caplog, snapshot) -> None:
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
    # Load the output and check contents
    data = pd.read_csv("output/image_statistics.csv")
    data.drop(["basename"], axis=1, inplace=True)
    assert data.to_string(float_format="{:.4e}".format) == snapshot(
        matcher=path_type(types=(float,), replacer=lambda data, _: round(data, 4))
    )


def test_bruker_rename(tmp_path: Path, caplog) -> None:
    """Test renaming of old Bruker files."""
    # Create 100 dummy test files
    for x in range(1, 101):
        test_file = tmp_path / f"test.{x:{0}>3}"
        test_file.touch(exist_ok=True)
    # Create a single dummy with .spm extension
    tmp_spm = tmp_path / "test.spm"
    tmp_spm.touch(exist_ok=True)
    caplog.set_level(logging.INFO)
    entry_point(
        manually_provided_args=[
            "--base-dir",
            f"{tmp_path}",
            "bruker-rename",  # This is the sub-command we wish to test, it will call run_modules.grains()
        ]
    )
    assert "Total Bruker files found : 101" in caplog.text
    assert "Old style files found    : 100" in caplog.text
    assert "test.001 > test.001.spm" in caplog.text
    assert "test.051 > test.051.spm" in caplog.text
    assert "test.100 > test.100.spm" in caplog.text


@pytest.mark.parametrize(
    ("file_ext"),
    [
        pytest.param(".asd", id="asd"),
        pytest.param(".ibw", id="ibw"),
        pytest.param(".gwy", id="gwy"),
        pytest.param(".jpk", id="jpk"),
        pytest.param(".stp", id="stp"),
        pytest.param(".top", id="top"),
        pytest.param(".topostats", id="topostats"),
    ],
)
def test_bruker_rename_assertion_error(file_ext: str) -> None:
    """Test AssertionError is raised when file_ext for renaming old Bruker files is wrong."""
    with pytest.raises(AssertionError):
        entry_point(
            manually_provided_args=[
                "--file-ext",
                file_ext,
                "bruker-rename",  # This is the sub-command we wish to test, it will call run_modules.grains()
            ]
        )
