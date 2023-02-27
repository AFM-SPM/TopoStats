"""Test end-to-end running of topostats."""
import logging
from pathlib import Path

import pytest

from topostats.io import LoadScans
from topostats.logs.logs import LOGGER_NAME
from topostats.run_topostats import process_scan, main as run_topostats_main

BASE_DIR = Path.cwd()


@pytest.mark.parametrize("option", ("-h", "--help"))
def test_run_topostats_main_help(capsys, option) -> None:
    """Test the -h/--help flag to run_topostats."""
    try:
        run_topostats_main([option])
    except SystemExit:
        pass
    assert "Process AFM images." in capsys.readouterr().out


def test_run_topostats_process_all(caplog) -> None:
    """Test run_topostats completes without error when no arguments are given."""
    caplog.set_level(logging.INFO)
    # Explicitly force loading of topostats/default_config.yaml as I couldn't work out how to invoke process_all()
    # without any arguments as it defaults to 'sys.argv' as this is wrapped within pytest it picks up the arguments
    # pytest was invoked with (see thread on StackOverflow at https://stackoverflow.com/a/55260580/1444043)
    run_topostats_main(args=["--config", f"{BASE_DIR / 'topostats' / 'default_config.yaml'}"])
    assert "~~~~~~~~~~~~~~~~~~~~ COMPLETE ~~~~~~~~~~~~~~~~~~~~" in caplog.text
    assert "Successfully Processed^1    : 1 (100.0%)" in caplog.text


def test_run_topostats_process_debug(caplog) -> None:
    """Test run_topostats with debugging and check DEBUG messages are logged"""
    # Set the logging level of the topostats logger
    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        run_topostats_main(args=["--config", f"{BASE_DIR / 'topostats' / 'default_config.yaml'}", "-l", "debug"])
        assert "Configuration after update         :" in caplog.text
        assert "File extension : .spm" in caplog.text
        assert "Images processed : 1" in caplog.text
        assert "~~~~~~~~~~~~~~~~~~~~ COMPLETE ~~~~~~~~~~~~~~~~~~~~" in caplog.text
