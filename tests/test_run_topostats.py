"""Test end-to-end running of topostats."""
import logging
from pathlib import Path

import pytest

from topostats.run_topostats import main as run_topostats_main

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
    assert "Successfully Processed      : 1 (100.0%)" in caplog.text
