"""Tests for logging"""
from pathlib import Path
import logging
from topostats.logs.logs import setup_logger


def test_setup_logger() -> None:
    """Test logger setup"""
    LOGGER = setup_logger()

    assert isinstance(LOGGER, logging.Logger)
