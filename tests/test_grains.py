"""Test finding of grains."""
import numpy as np

# Pylint returns this error for from skimage.filters import gaussian
# pylint: disable=no-name-in-module
from skimage.filters import gaussian
from topostats.grains import Grains

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
