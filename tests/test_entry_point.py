"""Test the entry point of TopoStats and its ability to correctly direct to programs."""

from pathlib import Path
from typing import Callable
import pytest

from topostats.entry_point import (
    entry_point,
    legacy_run_topostats_entry_point,
    legacy_toposum_entry_point,
)
from topostats.run_topostats import run_topostats
from topostats.plotting import run_toposum


# Test "help" arguments
@pytest.mark.parametrize("option", ("-h", "--help"))
def test_entry_point_help(capsys, option) -> None:
    """Test the help argument of the general entry point."""

    try:
        entry_point(manually_provided_args=[option])
    except SystemExit:
        pass
    output = capsys.readouterr().out

    assert "usage:" in output and "program" in output


@pytest.mark.parametrize("option", ("-h", "--help"))
def test_entry_point_process_help(capsys, option):
    """Test the help argument of the process program."""

    try:
        entry_point(manually_provided_args=["process", option])
    except SystemExit:
        pass
    output = capsys.readouterr().out

    assert "usage:" in output and "process" in output


@pytest.mark.parametrize("option", ("-h", "--help"))
def test_entry_point_summary_help(capsys, option):
    """Test the help argument of the summary program."""

    try:
        entry_point(manually_provided_args=["summary", option])
    except SystemExit:
        pass
    output = capsys.readouterr().out

    assert "usage:" in output and "summary" in output


# Test that the right functions are returned with the right arguments
@pytest.mark.parametrize(
    "options, expected_function, expected_arg_name, expected_arg_value",
    [
        (
            [
                "process",
                "-c dummy/config/dir/config.yaml",
            ],
            run_topostats,
            "config_file",
            " dummy/config/dir/config.yaml",
        ),
        (
            [
                "process",
                "--config",
                "dummy/config/dir/config.yaml",
            ],
            run_topostats,
            "config_file",
            "dummy/config/dir/config.yaml",
        ),
        (
            [
                "summary",
                "-l dummy/config/dir/var_to_label.yaml",
            ],
            run_toposum,
            "var_to_label",
            " dummy/config/dir/var_to_label.yaml",
        ),
    ],
)
def test_entry_point(
    options: list, expected_function: Callable, expected_arg_name: str, expected_arg_value: str
) -> None:
    """Test the entry point, ensuring the correct function is called for each program, and arguments
    are carried through."""

    returned_args = entry_point(options, testing=True)
    # convert argparse's Namespace object to dictionary
    returned_args_dict = vars(returned_args)

    # check that the correct function is collected
    assert returned_args.func == expected_function

    # check that the argument has successfully been passed through into the dictionary
    assert returned_args_dict[expected_arg_name] == expected_arg_value


def test_entry_point_create_config_file(tmp_path: Path) -> None:
    """Test that the entry point is able to produce a default config file when asked to."""

    with pytest.raises(SystemExit):
        entry_point(manually_provided_args=["process", "--create_config_file", f"{tmp_path}/test_create_config.yaml"])

    assert Path(f"{tmp_path}/test_create_config.yaml").is_file()


# Test that the right functions are returned with the right arguments
@pytest.mark.parametrize(
    "options, expected_arg_name, expected_arg_value",
    [
        (
            [
                "-c dummy/config/dir/config.yaml",
            ],
            "config_file",
            " dummy/config/dir/config.yaml",
        ),
        (
            [
                "--config",
                "dummy/config/dir/config.yaml",
            ],
            "config_file",
            "dummy/config/dir/config.yaml",
        ),
    ],
)
def test_legacy_run_topostats_entry_point(options: list, expected_arg_name: str, expected_arg_value: str) -> None:
    """Test the run_topostats legacy entry point, ensuring the arguments
    are parsed and carried through correctly."""

    returned_args = legacy_run_topostats_entry_point(options, testing=True)
    # Convert argparse's Namespace object to dictionary
    returned_args_dict = vars(returned_args)

    assert returned_args_dict[expected_arg_name] == expected_arg_value


def test_legacy_run_topostats_entry_point_create_config_file(tmp_path: Path) -> None:
    """Test that the run_topostats legacy entry point is able to produce a default config file
    when asked to."""

    with pytest.raises(SystemExit):
        legacy_run_topostats_entry_point(
            args=["--create-config-file", f"{tmp_path}/test_legacy_run_topostats_create_config.yaml"]
        )

    assert Path(f"{tmp_path}/test_legacy_run_topostats_create_config.yaml").is_file()


def test_legacy_toposum_entry_point_create_config_file(tmp_path: Path) -> None:
    """Test that the toposum legacy entry point is able to produce a default config file
    when asked to."""

    with pytest.raises(SystemExit):
        legacy_toposum_entry_point(
            args=["--create-config-file", f"{tmp_path}/test_legacy_toposum_create_config_file.yaml"]
        )

    assert Path(f"{tmp_path}/test_legacy_toposum_create_config_file.yaml").is_file()


def test_legacy_toposum_entry_point_create_label_file(tmp_path: Path) -> None:
    """Test that the toposum legacy entry point is able to produce a default label file
    when asked to."""

    with pytest.raises(SystemExit):
        legacy_toposum_entry_point(
            args=["--create-label-file", f"{tmp_path}/test_legacy_toposum_create_label_file.yaml"]
        )

    assert Path(f"{tmp_path}/test_legacy_toposum_create_label_file.yaml").is_file()
