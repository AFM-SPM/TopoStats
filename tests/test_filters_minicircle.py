"""Tests of the filters module."""
# + pylint: disable=invalid-name
import numpy as np
import pytest

from topostats.filters import Filters


def test_median_flatten_unmasked(minicircle_initial_median_flatten: Filters) -> None:
    """Test initial alignment of rows without mask."""
    assert isinstance(minicircle_initial_median_flatten.images["initial_median_flatten"], np.ndarray)
    assert minicircle_initial_median_flatten.images["initial_median_flatten"].shape == (64, 64)
    assert minicircle_initial_median_flatten.images["initial_median_flatten"].sum() == pytest.approx(733.1903924295203)


def test_remove_tilt_unmasked(minicircle_initial_tilt_removal: Filters) -> None:
    """Test removal of tilt without mask."""
    assert isinstance(minicircle_initial_tilt_removal.images["initial_tilt_removal"], np.ndarray)
    assert minicircle_initial_tilt_removal.images["initial_tilt_removal"].shape == (64, 64)
    assert minicircle_initial_tilt_removal.images["initial_tilt_removal"].sum() == pytest.approx(-928.581525771142)


def test_remove_quadratic_unmasked(minicircle_initial_quadratic_removal: Filters) -> None:
    """Test removal of quadratic with mask."""
    assert isinstance(minicircle_initial_quadratic_removal.images["initial_quadratic_removal"], np.ndarray)
    assert minicircle_initial_quadratic_removal.images["initial_quadratic_removal"].shape == (64, 64)
    assert minicircle_initial_quadratic_removal.images["initial_quadratic_removal"].sum() == pytest.approx(
        -805.8570677819365
    )


def test_get_threshold_otsu(minicircle_threshold_otsu: np.array) -> None:
    """Test calculation of threshold."""
    assert isinstance(minicircle_threshold_otsu.thresholds, dict)
    assert minicircle_threshold_otsu.thresholds["above"] == pytest.approx(0.4990958836019106)


def test_get_threshold_stddev(minicircle_threshold_stddev: np.array) -> None:
    """Test calculation of threshold."""
    assert isinstance(minicircle_threshold_stddev.thresholds, dict)
    assert minicircle_threshold_stddev.thresholds == pytest.approx(
        {"below": -7.484708050736529, "above": 0.4990958836019106}
    )


def test_get_threshold_abs(minicircle_threshold_abs: np.array) -> None:
    """Test calculation of threshold."""
    assert isinstance(minicircle_threshold_abs.thresholds, dict)
    assert minicircle_threshold_abs.thresholds == {"above": 1.5, "below": -1.5}


def test_get_mask(minicircle_mask: Filters) -> None:
    """Test derivation of mask."""
    assert isinstance(minicircle_mask.images["mask"], np.ndarray)
    assert minicircle_mask.images["mask"].shape == (64, 64)
    assert minicircle_mask.images["mask"].sum() == 590


def test_median_flatten_masked(minicircle_masked_median_flatten: Filters) -> None:
    """Test alignment of rows without mask."""
    assert isinstance(minicircle_masked_median_flatten.images["masked_median_flatten"], np.ndarray)
    assert minicircle_masked_median_flatten.images["masked_median_flatten"].shape == (64, 64)
    assert minicircle_masked_median_flatten.images["masked_median_flatten"].sum() == pytest.approx(1341.2916822670288)


def test_remove_x_y_tilt_masked(minicircle_masked_tilt_removal: Filters) -> None:
    """Test removal of tilt with mask."""
    assert isinstance(minicircle_masked_tilt_removal.images["masked_tilt_removal"], np.ndarray)
    assert minicircle_masked_tilt_removal.images["masked_tilt_removal"].shape == (64, 64)
    assert minicircle_masked_tilt_removal.images["masked_tilt_removal"].sum() == pytest.approx(1324.5046829300152)


def test_remove_quadratic_masked(minicircle_masked_quadratic_removal: Filters) -> None:
    """Test removal of quadratic with mask."""
    assert isinstance(minicircle_masked_quadratic_removal.images["masked_quadratic_removal"], np.ndarray)
    assert minicircle_masked_quadratic_removal.images["masked_quadratic_removal"].shape == (64, 64)
    assert minicircle_masked_quadratic_removal.images["masked_quadratic_removal"].sum() == pytest.approx(
        1324.635491325745
    )


def test_gaussian_filter(minicircle_grain_gaussian_filter: Filters) -> None:
    """Test gaussian filter applied to background."""
    assert isinstance(minicircle_grain_gaussian_filter.images["gaussian_filtered"], np.ndarray)
    assert minicircle_grain_gaussian_filter.images["gaussian_filtered"].shape == (64, 64)
    assert minicircle_grain_gaussian_filter.images["gaussian_filtered"].sum() == pytest.approx(1325.8136941472153)
