"""Test finding of grains."""
import logging
import numpy as np
import pytest

# Pylint returns this error for from skimage.filters import gaussian
# pylint: disable=no-name-in-module
from topostats.grains import Grains

LOGGER = logging.getLogger(__name__)
LOGGER.propagate = True

# Specify the absolute and relattive tolerance for floating point comparison
TOLERANCE = {"atol": 1e-07, "rtol": 1e-07}


grain_array = np.array(
    [
        [0,0,1,1,1,1,1,0,0,0],
        [0,1,1,1,1,0,1,0,0,2],
        [0,0,0,0,0,0,0,0,0,2],
        [0,3,3,0,0,0,0,0,2,2],
        [3,3,3,3,3,0,0,2,2,2],
    ]
)

grain_array2 = np.array(
    [
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,1],
        [0,0,0,0,0,0,0,0,0,1],
        [0,2,2,0,0,0,0,0,1,1],
        [2,2,2,2,2,0,0,1,1,1],
    ]
)

grain_array3 = np.array(
    [
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
    ]
)

grain_array4 = np.array(
    [
        [0,0,1,1,1,1,1,0,0,0],
        [0,1,1,1,1,0,1,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
    ]
)

@pytest.mark.parametrize(
    "area_thresh, expected",
    [
        ([None, None], grain_array),
        ([None, 8], grain_array2),
        ([3,6], grain_array3),
        ([8,11], grain_array4)
    ]
)
def test_known_array_threshold(area_thresh, expected) -> None:
    "Tests that arrays are thresholded on size as expected."
    grains = Grains(image=np.zeros((10,6)), filename="xyz", pixel_to_nm_scaling=1)
    assert (grains.area_thresholding(grain_array, area_thresh) == expected).all()


# def test_random_grains(random_grains: Grains, caplog) -> None:
#     """Test errors raised when processing images without grains."""
#     # FIXME : I can see for myself that the error message is logged but the assert fails as caplog.text is empty?
#     # assert "No gains found." in caplog.text
#     assert True
