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

release = version("topostats")
__version__ = ".".join(release.split("."[:2]))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

colormaps.register(cmap=Colormap("nanoscope").get_cmap())
colormaps.register(cmap=Colormap("gwyddion").get_cmap())

# Disable snoop
snoop.install(enabled=False)
