"""Utilities for tracing"""
from typing import Callable, Union

import numpy as np


class orderTrace:  # pylint: disable=too-few-public-methods
    """Class for ordering co-ordinates of a skeleton"""

    def __init__(self, coordinates: Union[np.ndarray, list]):
        """Initialise the class."""
        self.coordinates = coordinates

    def order(self, shape: str) -> np.ndarray:
        """Order a trace of the given shape.

        Parameters
        ----------
        shape: str
            The shape of the molecule, should be either 'linear' or 'circle'.capitalize()

        Returns
        -------
        Callable"""
        return self._order(shape)

    def _order(self, shape: str) -> Callable:
        """Creator for the method of ordering to use.

        Parameters
        ----------
        shape : str
            The shape of the molecule, should be either 'linear' or 'circle'.

        Returns
        -------
        Callable
            Returns the appropriate method for ordering the coordinates.

        """
        if shape == "linear":
            return self._linear()
        if shape == "circle":
            return self._circle()
        raise ValueError(shape)

    def _linear(self) -> np.ndarray:
        """Order linear skeleton."""

    def _circle(self) -> np.ndarray:
        """Order circular skeleton."""

    #    def _dilate(self):

    def _find_end_point(self):
        """Find end-points of linear grains."""


class Neighbours:  # pylint: disable=too-few-public-methods
    """Class for summarising features of neighbouring pixels."""

    def __init__(self):
        """Initialise the class."""
