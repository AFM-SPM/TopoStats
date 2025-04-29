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

colormaps.register(cmap=Colormap("nanoscope").get_cmap())
colormaps.register(cmap=Colormap("gwyddion").get_cmap())

# Disable snoop
snoop.install(enabled=False)
