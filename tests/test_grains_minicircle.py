"""Tests for Find Grains."""
import numpy as np

import pytest
from skimage.measure._regionprops import RegionProperties

from topostats.grains import Grains


def test_threshold_otsu(minicircle_grain_threshold_otsu: Grains) -> None:
    """Test threshold calculation"""
    assert isinstance(minicircle_grain_threshold_otsu.thresholds, dict)
    assert minicircle_grain_threshold_otsu.thresholds["upper"] == pytest.approx(1.6955888564850083)


def test_threshold_stddev(minicircle_grain_threshold_stddev: Grains) -> None:
    """Test threshold calculation"""
    assert isinstance(minicircle_grain_threshold_stddev.thresholds, dict)
    assert minicircle_grain_threshold_stddev.thresholds == pytest.approx(
        {"lower": -5.934646110767899, "upper": 0.7711821893707078}
    )


def test_threshold_abs(minicircle_grain_threshold_abs: Grains) -> None:
    """Test threshold calculation"""
    assert isinstance(minicircle_grain_threshold_abs.thresholds, dict)
    assert minicircle_grain_threshold_abs.thresholds == {"upper": 1.0, "lower": -1.0}


def test_mask_minicircle(minicircle_grain_mask: Grains) -> None:
    """Test creation of boolean array for clearing borders."""
    assert isinstance(minicircle_grain_mask.directions["upper"]["mask_grains"], np.ndarray)
    assert minicircle_grain_mask.directions["upper"]["mask_grains"].shape == (1024, 1024)
    assert minicircle_grain_mask.directions["upper"]["mask_grains"].sum() == 54964


def test_clear_border(minicircle_grain_clear_border: Grains) -> None:
    """Test creation of boolean array for clearing borders."""
    assert isinstance(minicircle_grain_clear_border.directions["upper"]["tidied_border"], np.ndarray)
    assert minicircle_grain_clear_border.directions["upper"]["tidied_border"].shape == (1024, 1024)
    assert minicircle_grain_clear_border.directions["upper"]["tidied_border"].sum() == 50793


def test_remove_noise(minicircle_grain_remove_noise: Grains) -> None:
    """Test creation of boolean array for clearing borders."""
    assert isinstance(minicircle_grain_remove_noise.directions["upper"]["removed_noise"], np.ndarray)
    assert minicircle_grain_remove_noise.directions["upper"]["removed_noise"].shape == (1024, 1024)
    assert minicircle_grain_remove_noise.directions["upper"]["removed_noise"].sum() == 45813


def test_calc_minimum_grain_size_pixels(minicircle_minimum_grain_size) -> None:
    """Test calculation of minimum grain size in pixels."""
    assert isinstance(minicircle_minimum_grain_size.minimum_grain_size, float)
    assert minicircle_minimum_grain_size.minimum_grain_size == 1559.25


def test_remove_small_objects(minicircle_small_objects_removed: Grains) -> None:
    """Test removal of small objects."""
    assert isinstance(minicircle_small_objects_removed.directions["upper"]["removed_small_objects"], np.ndarray)
    assert minicircle_small_objects_removed.directions["upper"]["removed_small_objects"].shape == (1024, 1024)
    assert minicircle_small_objects_removed.directions["upper"]["removed_small_objects"].sum() == 41960


def test_area_thresholding(minicircle_area_thresholding: Grains) -> None:
    """Test removal of small objects via absolute area thresholding."""
    assert isinstance(minicircle_area_thresholding.directions["upper"]["removed_small_objects"], np.ndarray)
    assert minicircle_area_thresholding.directions["upper"]["removed_small_objects"].shape == (1024, 1024)
    assert minicircle_area_thresholding.directions["upper"]["removed_small_objects"].sum() == 398845


def test_label_regions(minicircle_grain_labelled_post_removal: Grains) -> None:
    """Test removal of small objects."""
    assert isinstance(minicircle_grain_labelled_post_removal.directions["upper"]["labelled_regions_02"], np.ndarray)
    assert minicircle_grain_labelled_post_removal.directions["upper"]["labelled_regions_02"].shape == (1024, 1024)
    assert minicircle_grain_labelled_post_removal.directions["upper"]["labelled_regions_02"].sum() == 479645


def test_region_properties(minicircle_grain_region_properties_post_removal: np.array) -> None:
    """Test removal of small objects."""
    assert isinstance(minicircle_grain_region_properties_post_removal, list)
    assert len(minicircle_grain_region_properties_post_removal) == 22
    for x in minicircle_grain_region_properties_post_removal:
        assert isinstance(x, RegionProperties)


def test_colour_regions(minicircle_grain_coloured: Grains) -> None:
    """Test removal of small objects."""
    assert isinstance(minicircle_grain_coloured.directions["upper"]["coloured_regions"], np.ndarray)
    assert minicircle_grain_coloured.directions["upper"]["coloured_regions"].shape == (1024, 1024, 3)
    assert minicircle_grain_coloured.directions["upper"]["coloured_regions"].sum() == 60991.13700000002
