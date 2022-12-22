"""Calculate statistics on molecules."""
import logging
from typing import Dict, Tuple
import numpy as np

from topostats.logs.logs import LOGGER_NAME
from topostats.tracing.tracedna import traceDNA

LOGGER = logging.getLogger(LOGGER_NAME)


class DNAstatistics:  # pylint: disable=too-few-public-methods
    """Calculate statistics on a single grain."""

    def __init__(self, grain: traceDNA) -> Dict:
        self.grain = grain

    def calculate_stats(self):
        """Calculate Statistics"""
        self.measure_contour_length()
        # self.report_basic_stats()

    def molecule_heights(self, mask: np.ndarray = None) -> Tuple[float]:
        """Calculate the average height of a molecule.

        The mean, median and mode of the supplied image is calcualted.

        Parameters
        ----------
        image : np.ndarray
            2D Numpy array, values are heights in the x/y plane. The array
        method : str
            The type of average to return.

        Returns
        -------
        float
            The calculated average.
        """

    def measure_contour_length(self) -> float:
        """Measure the length of the grain.

        Returns
        -------
        float
            The length of the grain.
        """
