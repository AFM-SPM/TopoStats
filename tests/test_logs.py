"""Tests for logging."""

import logging

import pytest

from topostats.logs.logs import LOGGER_NAME, setup_logger

LOGGER = setup_logger(LOGGER_NAME)


def test_setup_logger(caplog) -> None:
    """Test logger setup."""
    info_msg = "This is a test message"
    LOGGER.info(info_msg)
    assert isinstance(LOGGER, logging.Logger)
    assert info_msg in caplog.text


@pytest.mark.parametrize(
    ("log_level", "message"),
    [
        (logging.DEBUG, "DEBUG : This is a debug log message."),
        (logging.INFO, "INFO : This is an info log message."),
        (logging.WARNING, "WARNING : This is a warning log message."),
        (logging.CRITICAL, "CRITICAL : This is a critical log message."),
    ],
)
def test_debug(caplog, log_level, message) -> None:
    """Test logging debug messages."""
    caplog.set_level(log_level, logger=LOGGER_NAME)
    if "DEBUG" in message:
        LOGGER.debug(message)
    elif "INFO" in message:
        LOGGER.info(message)
    elif "WARNING" in message:
        LOGGER.warning(message)
    elif "CRITICAL" in message:
        LOGGER.critical(message)
    with caplog.at_level(log_level):
        assert isinstance(LOGGER, logging.Logger)
        assert message in caplog.text
