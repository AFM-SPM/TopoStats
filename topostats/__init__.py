"""Topostats."""
from importlib.metadata import version
import matplotlib.pyplot as plt

from .logs.logs import setup_logger
from .theme import Colormap

LOGGER = setup_logger()

release = version("topostats")
__version__ = ".".join(release.split("."[:2]))

plt.register_cmap(cmap=Colormap("nanoscope").get_cmap())
plt.register_cmap(cmap=Colormap("gwyddion").get_cmap())
