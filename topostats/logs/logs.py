"""Standardise logging."""

from datetime import datetime
from pathlib import Path

from loguru import logger as loguru

# pylint: disable=assignment-from-no-return


def setup_loguru(output_dir: Path, level: str) -> None:
    """
    Set up the loguru logger to log to a file.

    Parameters
    ----------
    output_dir : Path
        The directory to save the log file to.
    level : str
        The logging level to use. Can be one of "DEBUG", "INFO", "WARNING", "ERROR", or "CRITICAL".
    """
    log_file = output_dir / f"topostats-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    # remove any existing handlers
    loguru.remove()
    # add a new handler that logs to the specified file
    loguru.add(
        log_file,
        level=level.upper(),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        backtrace=True,
        diagnose=True,
        enqueue=True,
    )
    # note:
    # enqueue: True is used to ensure that log messages are written to the file in the order they are received, even
    # if multiple threads are logging at the same time.
    # backtrace: True is used to include the full traceback in the log message when an exception is raised.
    # diagnose: True is used to include the exception's arguments in the log message.
