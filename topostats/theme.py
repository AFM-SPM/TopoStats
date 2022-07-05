from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


class Colormap:
    def __init__(self, name: str = "nanoscope"):
        self.name = name
        self.cmap = None
        self.set(self.name)
    def __str__(self):
        return f"TopoStats Colormap: {self.name}"

    def set(self, name: str):
        """ Set the colormap to the defined colormap with name = name """
        # This function could probably do with some better string comparisons to ensure the correct
        # colormap is picked up
        if name.lower() == "nanoscope":
            #print("Using the Nanoscope colormap for plotting")
            self.cmap = self.nanoscope()
        else:
            # Get one of the matplotlib colormaps
            self.cmap = cm.get_cmap(name)

    def get_cmap(self):
        """ Return the matplotlib.cm colormap object """
        return self.cmap

    def nanoscope(self):
        """
            Returns a matplotlib compatible colormap that replicates the Bruker Nanoscope colorscale
            The colormap is implemented in Gwyddion's GwyGradient via 'Nanoscope.txt'

            Original Nanoscope.txt

            Gwyddion resource GwyGradient
            0 0 0 0 1
            0.124464 0 0 0 1
            0.236052 0.0670103 0 0 1
            0.371245 0.253338 0 0 1
            0.472103 0.392344 0.0721649 0 1
            0.611588 0.584587 0.334114 0 1
            0.708155 0.717678 0.515464 0.252575 1
            0.714052 0.725806 0.527471 0.268 1
            0.890558 0.969072 0.886843 0.76343 1
            0.933476 0.987464 0.974227 0.883897 1
            0.944709 0.992278 0.980523 0.915426 1
            0.965682 0.995207 0.992278 0.974293 1
            0.971401 0.996006 0.993565 0.990347 1
            1 1 1 1 1

            return colormap
        """

        n = 8 # Number of values
        vals = np.ones((n, 4)) # Initialise the array to be full of 1.0
        vals[0] = [0., 0., 0., 1] # Lowest value colour => black
        vals[1] = [0.0670103, 0., 0., 1.]
        vals[2] = [0.07, 0.01, 0.01, 1.]
        vals[3] = [0.14, 0.02, 0., 1.]
        vals[4] = [0.38, 0.1, 0., 1.]
        vals[5] = [0.51, 0.30, 0.11, 1.]
        vals[6] = [0.78, 0.56, 0.42, 1.]
        vals[7] = [1., 1., 1., 1.] # Highest value colour => white
        colormap = LinearSegmentedColormap.from_list("nanoscope", vals, N=256)
        return colormap
