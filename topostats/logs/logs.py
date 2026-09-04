"""Standardise logging."""

from pathlib import Path
import logging
import sys
from datetime import datetime
from pathlib import Path

# pylint: disable=assignment-from-no-return

start = datetime.now()
LOG_INFO_FORMATTER = logging.Formatter(
    fmt="[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s", datefmt="%a, %d %b %Y %H:%M:%S"
)
LOG_ERROR_FORMATTER = logging.Formatter(
    fmt="[%(asctime)s] [%(levelname)-8s] [%(name)s] [%(filename)s] [%(lineno)s] %(message)s",
    datefmt="%a, %d %b %Y %H:%M:%S",
)

LOGGER_NAME = "topostats"


def setup_logger(output_dir: Path, log_name: str = LOGGER_NAME) -> logging.Logger:
    """
    Logger setup.

    The logger for the module is initialised when the module is loaded (as this functions is called from
    __init__.py). This creates two stream handlers, one for general output and one for errors which are formatted
    differently (there is greater information in the error formatter). To use in modules import the 'LOGGER_NAME' and
    create a logger as shown in the Examples, it will inherit the formatting and direction of messages to the correct
    stream.

    Parameters
    ----------
    output_dir : Path
        Directory where the log file will be saved.
    log_name : str
        Name under which logging information occurs.

    Returns
    -------
    logging.Logger
        Logger object.

    Examples
    --------
    To use the logger in (sub-)modules have the following.

        import logging
        from topostats.logs.logs import LOGGER_NAME

        LOGGER = logging.getLogger(LOGGER_NAME)
        setup_logger(output_dir=config["output_dir"], log_name=LOGGER_NAME)
        LOGGER.info('This is a log message.')
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # Prevent log messages from being propagated to the root logger

    # Check if the logger already has handlers to avoid adding duplicate handlers
    if logger.handlers:
        return logger

    # If the logger has no handlers, set up the stream handlers
    log_file = output_dir / f"topostats-{start.strftime('%Y-%m-%d-%H-%M-%S')}.log"

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(LOG_ERROR_FORMATTER)
    logger.addHandler(file_handler)

    return logger
