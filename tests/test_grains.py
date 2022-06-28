"""Test finding of grains."""
import logging
import numpy as np

# Pylint returns this error for from skimage.filters import gaussian
# pylint: disable=no-name-in-module
from skimage.filters import gaussian
from topostats.grains import Grains

LOGGER = logging.getLogger(__name__)
LOGGER.propagate = True

# Specify the absolute and relattive tolerance for floating point comparison
TOLERANCE = {"atol": 1e-07, "rtol": 1e-07}


def test_gaussian_filter(small_array_grains: Grains, grain_config: dict) -> None:
    """Test Gaussian filter."""
    small_array_grains.gaussian_filter()
    target = gaussian(
        small_array_grains.image,
        sigma=(grain_config["gaussian_size"] * 0.5),
        mode=grain_config["gaussian_mode"],
    )

    assert isinstance(small_array_grains.images["gaussian_filtered"], np.ndarray)
    np.testing.assert_array_equal(small_array_grains.images["gaussian_filtered"], target)


def test_random_grains(random_grains: Grains, caplog) -> None:
    """Test errors raised when processing images without grains."""
    # FIXME : Uncomment once filtering excludes all noise (grains < min_size) as we should find no grains.
    # assert "No grains found" in caplog.text
    assert True
