"""Tests for Find Grains."""
import numpy as np

import pytest
from skimage.measure._regionprops import RegionProperties

from topostats.grains import Grains


def test_threshold_otsu(minicircle_grain_threshold_otsu: Grains) -> None:
    """Test threshold calculation"""
    assert isinstance(minicircle_grain_threshold_otsu.thresholds, dict)
    assert minicircle_grain_threshold_otsu.thresholds["above"] == pytest.approx(1.3131356733305124)


def test_threshold_stddev(minicircle_grain_threshold_stddev: Grains) -> None:
    """Test threshold calculation"""
    assert isinstance(minicircle_grain_threshold_stddev.thresholds, dict)
    assert minicircle_grain_threshold_stddev.thresholds == pytest.approx(
        {"below": -5.901722094162684, "above": 0.9495125861734497}
    )


def test_threshold_abs(minicircle_grain_threshold_abs: Grains) -> None:
    """Test threshold calculation"""
    assert isinstance(minicircle_grain_threshold_abs.thresholds, dict)
    assert minicircle_grain_threshold_abs.thresholds == {"above": 1.0, "below": -1.0}


def test_mask_minicircle(minicircle_grain_mask: Grains) -> None:
    """Test creation of boolean array for clearing borders."""
    assert isinstance(minicircle_grain_mask.directions["above"]["mask_grains"], np.ndarray)
    assert minicircle_grain_mask.directions["above"]["mask_grains"].shape == (64, 64)
    assert minicircle_grain_mask.directions["above"]["mask_grains"].sum() == 499


def test_clear_border(minicircle_grain_clear_border: Grains) -> None:
    """Test creation of boolean array for clearing borders."""
    assert isinstance(minicircle_grain_clear_border.directions["above"]["tidied_border"], np.ndarray)
    assert minicircle_grain_clear_border.directions["above"]["tidied_border"].shape == (64, 64)
    assert minicircle_grain_clear_border.directions["above"]["tidied_border"].sum() == 400


def test_remove_noise(minicircle_grain_remove_noise: Grains) -> None:
    """Test creation of boolean array for clearing borders."""
    assert isinstance(minicircle_grain_remove_noise.directions["above"]["removed_noise"], np.ndarray)
    assert minicircle_grain_remove_noise.directions["above"]["removed_noise"].shape == (64, 64)
    #    assert minicircle_grain_remove_noise.directions["above"]["removed_noise"].sum() == 45813
    assert minicircle_grain_remove_noise.directions["above"]["removed_noise"].sum() == 392


def test_calc_minimum_grain_size_pixels(minicircle_minimum_grain_size) -> None:
    """Test calculation of minimum grain size in pixels."""
    assert isinstance(minicircle_minimum_grain_size.minimum_grain_size, float)
    assert minicircle_minimum_grain_size.minimum_grain_size == 122.25
    print(minicircle_minimum_grain_size.minimum_grain_size)
    assert False


def test_remove_small_objects(minicircle_small_objects_removed: Grains) -> None:
    """Test removal of small objects."""
    assert isinstance(minicircle_small_objects_removed.directions["above"]["removed_small_objects"], np.ndarray)
    assert minicircle_small_objects_removed.directions["above"]["removed_small_objects"].shape == (64, 64)
    assert minicircle_small_objects_removed.directions["above"]["removed_small_objects"].sum() == 392


def test_area_thresholding(minicircle_area_thresholding: Grains) -> None:
    """Test removal of small objects via absolute area thresholding."""
    assert isinstance(minicircle_area_thresholding.directions["above"]["removed_small_objects"], np.ndarray)
    assert minicircle_area_thresholding.directions["above"]["removed_small_objects"].shape == (64, 64)
    assert minicircle_area_thresholding.directions["above"]["removed_small_objects"].sum() == 799


def test_label_regions(minicircle_grain_labelled_post_removal: Grains) -> None:
    """Test removal of small objects."""
    assert isinstance(minicircle_grain_labelled_post_removal.directions["above"]["labelled_regions_02"], np.ndarray)
    assert minicircle_grain_labelled_post_removal.directions["above"]["labelled_regions_02"].shape == (64, 64)
    assert minicircle_grain_labelled_post_removal.directions["above"]["labelled_regions_02"].sum() == 799


def test_region_properties(minicircle_grain_region_properties_post_removal: np.array) -> None:
    """Test removal of small objects."""
    assert isinstance(minicircle_grain_region_properties_post_removal, list)
    assert len(minicircle_grain_region_properties_post_removal) == 3
    for x in minicircle_grain_region_properties_post_removal:
        assert isinstance(x, RegionProperties)


def test_colour_regions(minicircle_grain_coloured: Grains) -> None:
    """Test removal of small objects."""
    assert isinstance(minicircle_grain_coloured.directions["above"]["coloured_regions"], np.ndarray)
    assert minicircle_grain_coloured.directions["above"]["coloured_regions"].shape == (64, 64, 3)
    assert minicircle_grain_coloured.directions["above"]["coloured_regions"].sum() == 533.0
