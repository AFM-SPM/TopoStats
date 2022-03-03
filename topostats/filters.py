import numpy as np

"""Contains filter functions that take a 2D array representing an image as an input, as well as necessary parameters, and return a 2D array of the same size representing the filtered image."""


def turner(image: np.array, level: float) -> np.array:
    """The Turner filter removes all noise in an image by replacing the value of all pixels with the `level` argument.
    
    This is, clearly, a silly example and will eventually be removed."""
    return np.ones_like(image) * level
