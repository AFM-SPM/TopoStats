"""Module for calculating curvature statistics."""
import logging

# from pathlib import Path

import numpy as np

from topostats.logs.logs import LOGGER_NAME

LOGGER = logging.getLogger(LOGGER_NAME)

# Disable white space before colon (black and flake8 confict)
# noqa: E203


class Curvature:
    def __init__(self):
        """Initialise the class."""

    def find_curvature(self, molecule_coordinates: np.ndarray, circular: bool, edge_order: int = 2):
        """Find the curvature for an individual molecule.

        Parameters
        ----------
        molecule_coordinates: np.ndarray
        Coordinates of the simplified splined trace of a molecule. These are returned by dnaTracing.
        circular: bool
        Whether the molecule has been determined as being circular or not.
        edge_order: int(
        Gradient is passed to numpy.gradient and Gradient is calculated using N-th order accurate differences at
        boundaries.

        Returns
        -------
        """

        length = len(molecule_coordinates)
        if circular:
            longlist = np.concatenate((molecule_coordinates, molecule_coordinates, molecule_coordinates))
            dx = np.gradient(longlist, axis=0)[:, 0]
            dy = np.gradient(longlist, axis=0)[:, 1]
            d2x = np.gradient(dx)
            d2y = np.gradient(dy)

            dx = dx[length : 2 * length]  # NOQA
            dy = dy[length : 2 * length]  # NOQA
            d2x = d2x[length : 2 * length]  # NOQA
            d2y = d2y[length : 2 * length]  # NOQA
        else:
            dx = np.gradient(molecule_coordinates, edge_order, axis=0)[:, 0]
            dy = np.gradient(molecule_coordinates, edge_order, axis=0)[:, 1]
            d2x = np.gradient(dx)
            d2y = np.gradient(dy)

    # def _extract_coordinates(molecule) -> np.ndarray:
    #     """Extract the coordinates for the points"""
    #     curve = []
    #     contour = 0
    #     # for i in len(molecule):
    #     pass

    def _calculate_local_curvature(dx, dy, d2x, d2y) -> float:
        """Calculate the local curvature between points."""
        return ((d2x * dy) - (dx * d2y)) / ((dx**2) + (dy**2)) ** 1.5
