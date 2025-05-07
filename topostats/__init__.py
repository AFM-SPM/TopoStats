"""Topostats."""

import os
from importlib.metadata import version

import snoop
from matplotlib import colormaps

from .logs.logs import setup_logger
from .theme import Colormap

# Disable TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

LOGGER = setup_logger()

__version__ = version("topostats")
__release__ = ".".join(__version__.split(".")[:-2])

TOPOSTATS_DETAILS = version("topostats").split("+g")
TOPOSTATS_VERSION = TOPOSTATS_DETAILS[0]
TOPOSTATS_COMMIT = TOPOSTATS_DETAILS[1].split(".d")[0]

colormaps.register(cmap=Colormap("nanoscope").get_cmap())
colormaps.register(cmap=Colormap("gwyddion").get_cmap())

# Disable snoop
snoop.install(enabled=False)
