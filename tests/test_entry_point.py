"""Test the entry point of TopoStats and its ability to correctly direct to programs."""

from collections.abc import Callable
from pathlib import Path

import pytest

from topostats.entry_point import (
    entry_point,
    legacy_run_topostats_entry_point,
    legacy_toposum_entry_point,
)
from topostats.io import write_config_with_comments
from topostats.plotting import run_toposum
from topostats.run_topostats import run_topostats


# Test "help" arguments
@pytest.mark.parametrize("option", [("-h"), ("--help")])
def test_entry_point_help(capsys, option) -> None:
    """Test the help argument of the general entry point."""
    try:
        entry_point(manually_provided_args=[option])
    except SystemExit:
        pass
    output = capsys.readouterr().out

    assert "usage:" in output
    assert "program" in output


@pytest.mark.parametrize(
    (("argument", "option")),
    [
        ("process", "-h"),
        ("process", "--help"),
        ("summary", "-h"),
        ("summary", "--help"),
        ("load", "-h"),
        ("load", "--help"),
        ("filter", "-h"),
        ("filter", "--help"),
        ("grains", "-h"),
        ("grains", "--help"),
        ("grainstats", "-h"),
        ("grainstats", "--help"),
        ("create-config", "-h"),
        ("create-config", "--help"),
    ],
)
def test_entry_point_subprocess_help(capsys, argument: str, option: str) -> None:
    """Test the help argument to the master and sub entry points."""
    try:
        entry_point(manually_provided_args=[argument, option])
    except SystemExit:
        pass
    output = capsys.readouterr().out

    assert "usage:" in output
    assert argument in output


# Test that the right functions are returned with the right arguments
@pytest.mark.parametrize(
    ("options", "expected_function", "expected_arg_name", "expected_arg_value"),
    [
        pytest.param(
            [
                "process",
                "-c",
                "dummy/config/dir/config.yaml",
            ],
            run_topostats,
            "config_file",
            Path("dummy/config/dir/config.yaml"),
            id="Process with config file argument",
        ),
        pytest.param(
            [
                "process",
                "-b",
                "/tmp/",  # noqa: S108
            ],
            run_topostats,
            "base_dir",
            Path("/tmp/"),  # noqa: S108
            id="Process with base dir argument",
        ),
        pytest.param(
            [
                "create-config",
                "--filename",
                "dummy/config/dir/config.yaml",
            ],
            write_config_with_comments,
            "filename",
            Path("dummy/config/dir/config.yaml"),
            id="Create config with output filename",
        ),
        pytest.param(
            [
                "summary",
                "-l",
                "dummy/config/dir/var_to_label.yaml",
            ],
            run_toposum,
            "var_to_label",
            Path("dummy/config/dir/var_to_label.yaml"),
            id="Summary with label file.",
        ),
    ],
)
def test_entry_point(
    options: list, expected_function: Callable, expected_arg_name: str, expected_arg_value: str
) -> None:
    """Ensure the correct function is called for each program, and arguments are carried through."""
    returned_args = entry_point(options, testing=True)
    # convert argparse's Namespace object to dictionary
    returned_args_dict = vars(returned_args)
    # check that the correct function is collected
    assert returned_args.func == expected_function
    # check that the argument has successfully been passed through into the dictionary
    assert returned_args_dict[expected_arg_name] == expected_arg_value


def test_entry_point_create_config_file(tmp_path: Path) -> None:
    """Test that the entry point is able to produce a default config file when asked to."""
    entry_point(
        manually_provided_args=[
            "create-config",
            "--filename",
            "test_create_config.yaml",
            "--output-dir",
            f"{tmp_path}",
        ]
    )
    assert Path(f"{tmp_path}/test_create_config.yaml").is_file()


# Test that the right functions are returned with the right arguments
@pytest.mark.parametrize(
    ("options", "expected_arg_name", "expected_arg_value"),
    [
        pytest.param(
            [
                "-c",
                "dummy/config/dir/config.yaml",
            ],
            "config_file",
            Path("dummy/config/dir/config.yaml"),
            id="Test using -c flag for config file",
        ),
        pytest.param(
            [
                "--config",
                "dummy/config/dir/config.yaml",
            ],
            "config_file",
            Path("dummy/config/dir/config.yaml"),
            id="Test using --config flag for config file",
        ),
    ],
)
def test_legacy_run_topostats_entry_point(options: list, expected_arg_name: str, expected_arg_value: str) -> None:
    """Ensure the arguments are parsed and carried through correctly to legacy entry point."""
    returned_args = legacy_run_topostats_entry_point(options, testing=True)
    # Convert argparse's Namespace object to dictionary
    returned_args_dict = vars(returned_args)

    assert returned_args_dict[expected_arg_name] == expected_arg_value


def test_legacy_toposum_entry_point_create_config_file(tmp_path: Path) -> None:
    """Ensure the toposum legacy entry point is able to produce a default config file."""
    with pytest.raises(SystemExit):
        legacy_toposum_entry_point(
            args=["--create-config-file", f"{tmp_path}/test_legacy_toposum_create_config_file.yaml"]
        )

    assert Path(f"{tmp_path}/test_legacy_toposum_create_config_file.yaml").is_file()


def test_legacy_toposum_entry_point_create_label_file(tmp_path: Path) -> None:
    """Ensure the toposum legacy entry point is able to produce a default label file."""
    with pytest.raises(SystemExit):
        legacy_toposum_entry_point(
            args=["--create-label-file", f"{tmp_path}/test_legacy_toposum_create_label_file.yaml"]
        )

    assert Path(f"{tmp_path}/test_legacy_toposum_create_label_file.yaml").is_file()
