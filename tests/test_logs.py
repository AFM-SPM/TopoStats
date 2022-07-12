"""Tests for logging"""
import logging
from topostats.logs.logs import setup_logger


def test_setup_logger() -> None:
    """Test logger setup"""
    logger = setup_logger()

    assert isinstance(logger, logging.Logger)
