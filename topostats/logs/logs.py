"""
Standardise logging.
"""
import sys
from pathlib import Path
from datetime import datetime
import logging

# pylint: disable=assignment-from-no-return

start = datetime.now()
LOG_CONFIG = logging.basicConfig(filename=str(str(Path().cwd()) + start.strftime('%Y-%m-%d-%H-%M-%S') + '.log'),
                                 filemode='w')
LOG_FORMATTER = logging.Formatter(fmt='[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s',
                                  datefmt='%a, %d %b %Y %H:%M:%S')
LOG_ERROR_FORMATTER = logging.Formatter(
    fmt='[%(asctime)s] [%(levelname)-8s] [%(name)s] [%(filename)s] [%(lineno)s] %(message)s',
    datefmt='%a, %d %b %Y %H:%M:%S')

LOGGER_NAME = 'topostats'


def setup_logger(log_name: str = LOGGER_NAME) -> logging.Logger:
    """Setup a standard logger

    Parameters
    ----------
    log_name : str
        Name under which logging information occurs.

    Returns
    -------
    logging.Logger
        Logger object.
    """
    out_stream_handler = logging.StreamHandler(sys.stdout)
    out_stream_handler.setLevel(logging.DEBUG)
    out_stream_handler.setFormatter(LOG_FORMATTER)

    err_stream_handler = logging.StreamHandler(sys.stderr)
    err_stream_handler.setLevel(logging.ERROR)
    err_stream_handler.setFormatter(LOG_ERROR_FORMATTER)

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(out_stream_handler)
        logger.addHandler(err_stream_handler)

    return logger
