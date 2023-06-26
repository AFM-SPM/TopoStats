"""Custom Bruker Nanoscope colorscale."""
import logging
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import numpy as np

from topostats.logs.logs import LOGGER_NAME

LOGGER = logging.getLogger(LOGGER_NAME)


class Colormap:
    """Class for setting the Colormap"""

    def __init__(self, name: str = "nanoscope"):
        self.name = name
        self.cmap = None
        self.set(self.name)

    def __str__(self):
        return f"TopoStats Colormap: {self.name}"

    def set(self, name: str):
        """Set the ColorMap.

        Parameters
        ----------
        name: str
            Name of colormap
        """
        if name.lower() == "nanoscope":
            self.cmap = self.nanoscope()
        elif name.lower() == "gwyddion":
            self.cmap = self.gwyddion()
        elif name.lower() == "blu":
            self.cmap = self.blu()
        elif name.lower() == "multi":
            self.cmap = self.multi()
        else:
            # Get one of the matplotlib colormaps
            self.cmap = mpl.colormaps[name]
        LOGGER.debug(f"[theme] Colormap set to : {name}")

    def get_cmap(self):
        """Return the matplotlib.cm colormap object"""
        return self.cmap

    @staticmethod
    def nanoscope():
        """
        Returns a matplotlib compatible colormap that replicates the Bruker Nanoscope colorscale
        The colormap is implemented in Gwyddion's GwyGradient via 'Nanoscope.txt'
        """
        cdict = {
            "red": (
                (0.0, 0.0, 0.0),
                (0.124464, 0.0, 0.0),
                (0.236052, 0.0670103, 0.0670103),
                (0.371245, 0.253338, 0.253338),
                (0.472103, 0.392344, 0.392344),
                (0.611588, 0.584587, 0.584587),
                (0.708155, 0.717678, 0.717678),
                (0.714052, 0.725806, 0.725806),
                (0.890558, 0.969072, 0.969072),
                (0.933476, 0.987464, 0.987464),
                (0.944709, 0.992278, 0.992278),
                (0.965682, 0.995207, 0.995207),
                (0.971401, 0.996006, 0.996006),
                (1, 1, 1),
            ),
            "green": (
                (0.0, 0.0, 0.0),
                (0.124464, 0.0, 0.0),
                (0.236052, 0.0, 0.0),
                (0.371245, 0.0, 0.0),
                (0.472103, 0.0721649, 0.0721649),
                (0.611588, 0.334114, 0.334114),
                (0.708155, 0.515464, 0.515464),
                (0.714052, 0.527471, 0.527471),
                (0.890558, 0.886843, 0.886843),
                (0.933476, 0.974227, 0.974227),
                (0.944709, 0.980523, 0.980523),
                (0.965682, 0.992278, 0.992278),
                (0.971401, 0.993565, 0.993565),
                (1, 1, 1),
            ),
            "blue": (
                (0.0, 0.0, 0.0),
                (0.124464, 0.0, 0.0),
                (0.236052, 0.0, 0.0),
                (0.371245, 0.0, 0.0),
                (0.472103, 0.0, 0.0),
                (0.611588, 0.0, 0.0),
                (0.708155, 0.252575, 0.252575),
                (0.714052, 0.268, 0.268),
                (0.890558, 0.76343, 0.76343),
                (0.933476, 0.883897, 0.883897),
                (0.944709, 0.915426, 0.915426),
                (0.965682, 0.974293, 0.974293),
                (0.971401, 0.990347, 0.990347),
                (1, 1, 1),
            ),
        }

        colormap = LinearSegmentedColormap("nanoscope", cdict)
        return colormap

    @staticmethod
    def gwyddion():
        """The RGBA colour map for the Gwyddion.net colour gradient."""
        N = 4  # Number of values
        vals = np.ones((N, 4))  # Initialise the array to be full of 1.0
        vals[0] = [0.0, 0.0, 0.0, 1]
        vals[1] = [168 / 256, 40 / 256, 15 / 256, 1.0]
        vals[2] = [243 / 256, 194 / 256, 93 / 256, 1.0]
        vals[3] = [1.0, 1.0, 1.0, 1.0]

        colormap = LinearSegmentedColormap.from_list("gwyddion", vals, N=256)
        return colormap

    @staticmethod
    def blu():
        "RGBA colour map of just the colour blue."
        return ListedColormap([[32 / 256, 226 / 256, 205 / 256]], "blu", N=256)
    
    @staticmethod
    def multi():
        "11 distinct colours."
        return ListedColormap([
            np.array([230, 25, 75])/256,
            np.array([60, 180, 75])/256,
            np.array([255, 225, 25])/256,
            np.array([0, 130, 200])/256,
            np.array([245, 130, 48])/256,
            np.array([70, 240, 240])/256,
            np.array([240, 50, 230])/256,
            np.array([250, 190, 212])/256,
            np.array([0, 128, 128])/256,
            np.array([220, 190, 255])/256,
            np.array([170, 110, 40])/256
            ],
            "multi", N=None)
