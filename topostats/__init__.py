"""Topostats"""
from importlib.metadata import version

from .logs.logs import setup_logger

LOGGER = setup_logger()

release = version("topostats")
__version__ = ".".join(release.split("."[:2]))
