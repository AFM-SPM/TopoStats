"""Standardise logging."""

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


def setup_logger(log_name: str = LOGGER_NAME) -> logging.Logger:
    """
    Logger setup.

    The logger for the module is initialised when the module is loaded (as this functions is called from
    __init__.py). This creates two stream handlers, one for general output and one for errors which are formatted
    differently (there is greater information in the error formatter). To use in modules import the 'LOGGER_NAME' and
    create a logger as shown in the Examples, it will inherit the formatting and direction of messages to the correct
    stream.

    Parameters
    ----------
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

        LOGGER.info('This is a log message.')
    """
    out_stream_handler = logging.StreamHandler(sys.stdout)
    out_stream_handler.setLevel(logging.DEBUG)
    out_stream_handler.setFormatter(LOG_INFO_FORMATTER)

    err_stream_handler = logging.StreamHandler(sys.stderr)
    err_stream_handler.setLevel(logging.ERROR)
    err_stream_handler.setFormatter(LOG_ERROR_FORMATTER)

    file_handler = logging.FileHandler(Path().cwd().stem + f"-{start.strftime('%Y-%m-%d-%H-%M-%S')}.log")
    file_handler.setFormatter(LOG_ERROR_FORMATTER)

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    logger.propagate = True
    if not logger.handlers:
        logger.addHandler(out_stream_handler)
        logger.addHandler(err_stream_handler)
        logger.addHandler(file_handler)

    return logger
