"""Tests of the filters module."""
# + pylint: disable=invalid-name
import numpy as np

import pytest

from topostats.filters import Filters


def test_median_flatten_unmasked(minicircle_initial_median_flatten: Filters) -> None:
    """Test initial alignment of rows without mask."""
    assert isinstance(minicircle_initial_median_flatten.images["initial_median_flatten"], np.ndarray)
    assert minicircle_initial_median_flatten.images["initial_median_flatten"].shape == (1024, 1024)
    assert minicircle_initial_median_flatten.images["initial_median_flatten"].sum() == pytest.approx(40879.633073103985)


def test_remove_tilt_unmasked(minicircle_initial_tilt_removal: Filters) -> None:
    """Test removal of tilt without mask."""
    assert isinstance(minicircle_initial_tilt_removal.images["initial_tilt_removal"], np.ndarray)
    assert minicircle_initial_tilt_removal.images["initial_tilt_removal"].shape == (1024, 1024)
    assert minicircle_initial_tilt_removal.images["initial_tilt_removal"].sum() == pytest.approx(-1673257.7284464256)


def test_remove_quadratic_unmasked(minicircle_initial_quadratic_removal: Filters) -> None:
    """Test removal of quadratic with mask."""
    assert isinstance(minicircle_initial_quadratic_removal.images["initial_quadratic_removal"], np.ndarray)
    assert minicircle_initial_quadratic_removal.images["initial_quadratic_removal"].shape == (1024, 1024)
    assert minicircle_initial_quadratic_removal.images["initial_quadratic_removal"].sum() == pytest.approx(
        -1672538.8578738687
    )


def test_get_threshold_otsu(minicircle_threshold_otsu: np.array) -> None:
    """Test calculation of threshold."""
    assert isinstance(minicircle_threshold_otsu.thresholds, dict)
    assert minicircle_threshold_otsu.thresholds["upper"] == pytest.approx(-0.7471511148088736)


def test_get_threshold_stddev(minicircle_threshold_stddev: np.array) -> None:
    """Test calculation of threshold."""
    assert isinstance(minicircle_threshold_stddev.thresholds, dict)
    assert minicircle_threshold_stddev.thresholds == pytest.approx(
        {"lower": -8.283235478318018, "upper": -0.9269936645505799}
    )


def test_get_threshold_abs(minicircle_threshold_abs: np.array) -> None:
    """Test calculation of threshold."""
    assert isinstance(minicircle_threshold_abs.thresholds, dict)
    assert minicircle_threshold_abs.thresholds == {"upper": 1.5, "lower": -1.5}


def test_get_mask(minicircle_mask: Filters) -> None:
    """Test derivation of mask."""
    assert isinstance(minicircle_mask.images["mask"], np.ndarray)
    assert minicircle_mask.images["mask"].shape == (1024, 1024)
    assert minicircle_mask.images["mask"].sum() == 83095


def test_median_flatten_masked(minicircle_masked_median_flatten: Filters) -> None:
    """Test alignment of rows without mask."""
    assert isinstance(minicircle_masked_median_flatten.images["masked_median_flatten"], np.ndarray)
    assert minicircle_masked_median_flatten.images["masked_median_flatten"].shape == (1024, 1024)
    assert minicircle_masked_median_flatten.images["masked_median_flatten"].sum() == pytest.approx(169538.4636641923)


def test_remove_x_y_tilt_masked(minicircle_masked_tilt_removal: Filters) -> None:
    """Test removal of tilt with mask."""
    assert isinstance(minicircle_masked_tilt_removal.images["masked_tilt_removal"], np.ndarray)
    assert minicircle_masked_tilt_removal.images["masked_tilt_removal"].shape == (1024, 1024)
    assert minicircle_masked_tilt_removal.images["masked_tilt_removal"].sum() == pytest.approx(163752.28336856866)


def test_remove_quadratic_masked(minicircle_masked_quadratic_removal: Filters) -> None:
    """Test removal of quadratic with mask."""
    assert isinstance(minicircle_masked_quadratic_removal.images["masked_quadratic_removal"], np.ndarray)
    assert minicircle_masked_quadratic_removal.images["masked_quadratic_removal"].shape == (1024, 1024)
    assert minicircle_masked_quadratic_removal.images["masked_quadratic_removal"].sum() == pytest.approx(
        169411.34987849486
    )


def test_gaussian_filter(minicircle_grain_gaussian_filter: Filters) -> None:
    """Test gaussian filter applied to background."""
    assert isinstance(minicircle_grain_gaussian_filter.images["gaussian_filtered"], np.ndarray)
    assert minicircle_grain_gaussian_filter.images["gaussian_filtered"].shape == (1024, 1024)
    assert minicircle_grain_gaussian_filter.images["gaussian_filtered"].sum() == pytest.approx(169409.44307011212)
