"""Custom Bruker Nanoscope colorscale."""

import logging

import matplotlib as mpl
import matplotlib.cm
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from topostats.logs.logs import LOGGER_NAME

LOGGER = logging.getLogger(LOGGER_NAME)


class Colormap:
    """
    Class for setting the Colormap.

    Parameters
    ----------
    name : str
        Name of colormap to use.
    """

    def __init__(self, name: str = "nanoscope"):
        """
        Initialise the class.

        Parameters
        ----------
        name : str
            Name of colormap to use.
        """
        self.name = name
        self.cmap = None
        self.set_cmap(self.name)

    def __str__(self) -> str:
        """
        Return string representation of object.

        Returns
        -------
        str
            String detailing the colormap.
        """
        return f"TopoStats Colormap: {self.name}"

    def set_cmap(self, name: str) -> None:
        """
        Set the ColorMap.

        Parameters
        ----------
        name : str
            Name of the colormap to return.
        """
        if name.lower() == "nanoscope":
            self.cmap = self.nanoscope()
        elif name.lower() == "gwyddion":
            self.cmap = self.gwyddion()
        elif name.lower() == "blue":
            self.cmap = self.blue()
        elif name.lower() == "blue_purple_green":
            self.cmap = self.blue_purple_green()
        else:
            # Get one of the matplotlib colormaps
            self.cmap = mpl.colormaps[name]
        LOGGER.debug(f"[theme] Colormap set to : {name}")

    def get_cmap(self) -> matplotlib.cm:
        """
        Return the matplotlib.cm colormap object.

        Returns
        -------
        matplotlib.cm
            Matplotlib Color map object.
        """
        return self.cmap

    @staticmethod
    def nanoscope() -> LinearSegmentedColormap:
        """
        Matplotlib compatible colormap that replicates the Bruker Nanoscope colorscale.

        The colormap is implemented in Gwyddion's GwyGradient via 'Nanoscope.txt'.

        Returns
        -------
        LinearSegmentedColormap
            MatplotLib LinearSegmentedColourmap that replicates Bruker Nanoscope colorscale.
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

        return LinearSegmentedColormap("nanoscope", cdict)

    @staticmethod
    def gwyddion() -> LinearSegmentedColormap:
        """
        Set RGBA colour map for the Gwyddion.net colour gradient.

        Returns
        -------
        LinearSegmentedColormap
            The 'gwyddion' colormap.
        """
        N = 4  # Number of values
        vals = np.ones((N, 4))  # Initialise the array to be full of 1.0
        vals[0] = [0.0, 0.0, 0.0, 1]
        vals[1] = [168 / 256, 40 / 256, 15 / 256, 1.0]
        vals[2] = [243 / 256, 194 / 256, 93 / 256, 1.0]
        vals[3] = [1.0, 1.0, 1.0, 1.0]

        return LinearSegmentedColormap.from_list("gwyddion", vals, N=256)

    @staticmethod
    def blue() -> ListedColormap:
        """
        Set RGBA colour map of just the colour blue.

        Returns
        -------
        ListedColormap
            The 'blue' colormap.
        """
        return ListedColormap([[32 / 256, 226 / 256, 205 / 256]], "blue", N=256)

    @staticmethod
    def blue_purple_green() -> ListedColormap:
        """
        RGBA colour map of just the colour blue/purple/green.

        Returns
        -------
        ListedColormap
            The 'blue/purple/green' colormap.
        """
        return ListedColormap(
            [[0 / 256, 157 / 256, 229 / 256], [255 / 256, 100 / 256, 225 / 256], [0 / 256, 1, 139 / 256]],
            "blue_purple_green",
            N=3,
        )
