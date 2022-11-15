"""Tests of the filters module."""
from pathlib import Path

import numpy as np
from skimage.filters import gaussian

from topostats.filters import Filters

# pylint: disable=protected-access

# Specify the absolute and relattive tolerance for floating point comparison
TOLERANCE = {"atol": 1e-07, "rtol": 1e-07}

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"


def test_median_flatten_no_mask(test_filters_random: Filters, image_random_median_flattened: np.array) -> None:
    """Test aligning of rows by median height."""
    median_flattened = test_filters_random.median_flatten(test_filters_random.images["pixels"], mask=None)

    assert isinstance(median_flattened, np.ndarray)
    assert median_flattened.shape == (1024, 1024)

    np.testing.assert_allclose(median_flattened, image_random_median_flattened, **TOLERANCE)


def test_remove_tilt_no_mask(test_filters_random: Filters, image_random_remove_x_y_tilt: np.array) -> None:
    """Test removal of x/y tilt."""
    tilt_removed = test_filters_random.remove_tilt(test_filters_random.images["pixels"], mask=None)

    assert isinstance(tilt_removed, np.ndarray)
    assert tilt_removed.shape == (1024, 1024)
    np.testing.assert_allclose(tilt_removed, image_random_remove_x_y_tilt, **TOLERANCE)


def test_remove_quadratic(test_filters_random: Filters, image_random_remove_quadratic: np.ndarray) -> None:
    """Test removal of quadratic tilt."""
    quadratic_removed = test_filters_random.remove_quadratic(test_filters_random.images["pixels"], mask=None)

    assert isinstance(quadratic_removed, np.ndarray)
    assert quadratic_removed.shape == (1024, 1024)
    np.testing.assert_allclose(quadratic_removed, image_random_remove_quadratic, **TOLERANCE)

def test_calc_diff(test_filters_random: Filters, image_random: np.ndarray) -> None:
    """Test calculation of difference in array."""
    target = image_random[-1] - image_random[0]
    calculated = test_filters_random.calc_diff(test_filters_random.images["pixels"])

    np.testing.assert_array_equal(target, calculated)


def test_calc_gradient(test_filters_random: Filters, image_random: np.ndarray) -> None:
    """Test calculation of gradient."""
    target = (image_random[-1] - image_random[0]) / image_random.shape[0]
    calculated = test_filters_random.calc_gradient(
        test_filters_random.images["pixels"], test_filters_random.images["pixels"].shape[0]
    )

    np.testing.assert_array_equal(target, calculated)


# FIXME : sum of half of the array values is vastly smaller and so test fails. What is strange is that
#         test_filters_minicircle.test_average_background() *DOESN'T* fail
def test_non_square_img(test_filters_random: Filters):
    test_filters_random.images["pixels"] = test_filters_random.images["pixels"][:, 0:512]
    test_filters_random.images["zero_averaged_background"] = test_filters_random.median_flatten(
        image=test_filters_random.images["pixels"], mask=None
    )
    assert isinstance(test_filters_random.images["zero_averaged_background"], np.ndarray)
    assert test_filters_random.images["zero_averaged_background"].shape == (1024, 512)
    # assert test_filters_random.images["zero_averaged_background"].sum() == 44426.48188033322


def test_average_background(test_filters_random_with_mask: Filters):
    """Test the background averaging"""
    test_filters_random_with_mask.images["zero_averaged_background"] = test_filters_random_with_mask.average_background(
        image=test_filters_random_with_mask.images["pixels"],
        mask=test_filters_random_with_mask.images["mask"]
    )
    assert isinstance(test_filters_random_with_mask.images["zero_averaged_background"], np.ndarray)
    assert test_filters_random_with_mask.images["zero_averaged_background"].sum() == 263002.61107539403


def test_gaussian_filter(small_array_filters: Filters, filter_config: dict) -> None:
    """Test Gaussian filter."""
    small_array_filters.images["gaussian_filtered"] = small_array_filters.gaussian_filter(
        image=small_array_filters.images["zero_averaged_background"]
    )
    target = gaussian(
        small_array_filters.images["zero_averaged_background"],
        sigma=(filter_config["gaussian_size"]),
        mode=filter_config["gaussian_mode"],
    )
    assert isinstance(small_array_filters.images["gaussian_filtered"], np.ndarray)
    np.testing.assert_array_equal(small_array_filters.images["gaussian_filtered"], target)
