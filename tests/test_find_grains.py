"""Test finding of grains."""
import numpy as np

from skimage.filters import gaussian
from topostats.find_grains import (quadratic, get_lower_threshold, gaussian_filter, boolean_image, tidy_border,
                                   remove_objects, label_regions, colour_regions, region_properties, get_bounding_boxes,
                                   save_region_stats)

# Specify the absolute and relattive tolerance for floating point comparison
TOLERANCE = {'atol': 1e-07, 'rtol': 1e-07}


def test_gaussian_filter(small_array: np.array, gaussian_size: float = 2, dx: float = 1, mode: str = 'nearest') -> None:
    """Test Gaussian filter."""
    gaussian_filtered = gaussian_filter(small_array)
    target = gaussian(small_array, sigma=(gaussian_size / dx), mode=mode)

    assert isinstance(gaussian_filtered, np.ndarray)
    np.testing.assert_array_equal(gaussian_filtered, target)


def test_boolean_image(small_array: np.array, threshold: float = 0.5) -> None:
    """Test creation of boolean array."""
    boolean_array = boolean_image(small_array, threshold)
    target = small_array > threshold

    assert isinstance(boolean_array, np.ndarray)
    assert np.issubdtype(boolean_array.dtype, np.bool_)
    assert boolean_array.sum() == 44
    np.testing.assert_array_equal(boolean_array, target)
