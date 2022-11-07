"""Topostats"""
from .logs.logs import setup_logger
from . import _version

LOGGER = setup_logger()

__version__ = _version.get_versions()["version"]
