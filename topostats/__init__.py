"""Topostats."""

import logging
import os
from importlib.metadata import version

import snoop
from matplotlib import colormaps
from packaging.version import Version

from .logs.logs import LOGGER_NAME

from .theme import Colormap

# Disable TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


__version__ = version("topostats")
__release__ = ".".join(__version__.split(".")[:-2])

TOPOSTATS_VERSION = Version(__version__)
if TOPOSTATS_VERSION.is_prerelease and TOPOSTATS_VERSION.is_devrelease:
    TOPOSTATS_BASE_VERSION = str(TOPOSTATS_VERSION.base_version)
    TOPOSTATS_COMMIT = str(TOPOSTATS_VERSION).split("+g")[1]
else:
    TOPOSTATS_BASE_VERSION = str(TOPOSTATS_VERSION)
    TOPOSTATS_COMMIT = ""
CONFIG_DOCUMENTATION_REFERENCE = """# For more information on configuration and how to use it:
# https://afm-spm.github.io/TopoStats/latest/usage/configuration/\n"""

colormaps.register(cmap=Colormap("nanoscope").get_cmap())
colormaps.register(cmap=Colormap("gwyddion").get_cmap())

# Disable snoop
snoop.install(enabled=False)

LOGGER = logging.getLogger(LOGGER_NAME)


def get_topostats_version() -> Version:
    """
    Get the version of TopoStats.

    Returns
    -------
    Version
        The version of TopoStats.
    """
    return TOPOSTATS_VERSION


def get_topostats_commit() -> str:
    """
    Get the commit of TopoStats.

    Returns
    -------
    str
        The commit of TopoStats.
    """
    return TOPOSTATS_COMMIT
