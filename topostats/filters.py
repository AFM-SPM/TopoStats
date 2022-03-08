import numpy as np

"""Contains filter functions that take a 2D array representing an image as an input, as well as necessary parameters, and return a 2D array of the same size representing the filtered image."""


def amplify(image: np.array, level: float) -> np.array:
    """The amplify filter mulitplies the value of all pixels by the `level` argument.

    :param image: A 2D raster image
    :param level: Filter level
    :return: A filtered 2D raster image
    """

    return image * level
