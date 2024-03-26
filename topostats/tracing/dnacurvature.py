"""Module for calculating curvature statistics."""

import logging

# from pathlib import Path
import numpy as np

from topostats.logs.logs import LOGGER_NAME

LOGGER = logging.getLogger(LOGGER_NAME)

# Disable white space before colon
# black and flake8 conflict https://black.readthedocs.io/en/stable/faq.html#why-are-flake8-s-e203-and-w503-violated
# noqa: E203


class Curvature:
    """
    Class for determining the curvature of molecules.

    Parameters
    ----------
    molecule_coordinates : np.ndarray
        Coordinates of the simplified splined trace of a molecule. These are returned by dnaTracing.
    circular : bool
        Whether the image is circular or not.
    """

    def __init__(
        self,
        molecule_coordinates: np.ndarray,
        circular: bool,
    ):
        """
        Initialise the class.

        Parameters
        ----------
        molecule_coordinates : np.ndarray
            Coordinates of the simplified splined trace of a molecule. These are returned by dnaTracing.
        circular : bool
            Whether the image is circular or not.
        """
        self.molecule_coordinates = molecule_coordinates
        self.circular = circular
        self.n_points = len(molecule_coordinates)
        self.first_derivative = None
        self.second_derivative = None
        self.local_curvature = None

    def calculate_derivatives(self, edge_order: int = 1) -> None:
        """
        Find the curvature for an individual molecule.

        Parameters
        ----------
        edge_order : int
            Gradient is passed to numpy.gradient and Gradient is calculated using N-th order accurate differences at
            boundaries. Also used to expand the array by the necessary number of coordinates at either end to form a
            loop for the calculations.
        """
        # If circular we need gradients correctly calculated at the start and end and so the array has the number of
        # points used in np.gradient(edge_order) from the end attached to the start and the same from the start attached
        # to the end.
        edge_order_boundary = edge_order + 1
        if self.circular:
            coordinates = np.vstack(
                (
                    self.molecule_coordinates[-edge_order_boundary:],
                    self.molecule_coordinates,
                    self.molecule_coordinates[:edge_order_boundary],
                )
            )
        else:
            coordinates = self.molecule_coordinates
        self.first_derivative = np.gradient(coordinates, edge_order, axis=0)
        self.second_derivative = np.gradient(self.first_derivative, edge_order, axis=0)
        # Now trim the arrays back to the appropriate size
        if self.circular:
            self.first_derivative = self.first_derivative[edge_order_boundary:-edge_order_boundary]
            self.second_derivative = self.second_derivative[edge_order_boundary:-edge_order_boundary]

    # def _extract_coordinates(molecule) -> np.ndarray:
    #     """Extract the coordinates for the points"""
    #     curve = []
    #     contour = 0
    #     # for i in len(molecule):
    #     pass

    def _calculate_local_curvature(self) -> float:
        """Calculate the local curvature between points."""
        self.local_curvature = (
            (self.second_derivative[:, 0] * self.first_derivative[:, 1])
            - (self.first_derivative[:, 0] * self.second_derivative[:, 1])
        ) / ((self.first_derivative[:, 0] ** 2) + (self.first_derivative[:, 1] ** 2)) ** 1.5
