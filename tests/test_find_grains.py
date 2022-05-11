"""Test finding of grains."""
import numpy as np

from skimage.filters import gaussian
from topostats.find_grains import (
    quadratic,
    gaussian_filter,
    tidy_border,
    remove_objects,
    label_regions,
    colour_regions,
    region_properties,
    get_bounding_boxes,
    save_region_stats,
)

# Specify the absolute and relattive tolerance for floating point comparison
TOLERANCE = {"atol": 1e-07, "rtol": 1e-07}


def test_gaussian_filter(
    small_array: np.array, gaussian_size: float = 2, pixel_nm_scaling: float = 1, mode: str = "nearest"
) -> None:
    """Test Gaussian filter."""
    gaussian_filtered = gaussian_filter(small_array, gaussian_size, pixel_nm_scaling)
    target = gaussian(small_array, sigma=(gaussian_size / pixel_nm_scaling), mode=mode)

    assert isinstance(gaussian_filtered, np.ndarray)
    np.testing.assert_array_equal(gaussian_filtered, target)


def test_quadratic(small_array) -> None:
    """Test quadratic function."""
    values = {"a": 2, "b": 3, "c": 4}
    small_array_quadratic = quadratic(small_array, values["a"], values["b"], values["c"])
    target = (values["a"] * small_array**2) + (values["b"] * small_array) + values["c"]

    np.testing.assert_array_equal(target, small_array_quadratic)
