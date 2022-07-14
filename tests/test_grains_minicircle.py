"""Tests for Find Grains."""
import numpy as np

import pytest
from skimage.measure._regionprops import RegionProperties

from topostats.grains import Grains
from topostats.plottingfuncs import plot_and_save

# Specify the absolute and relattive tolerance for floating point comparison
TOLERANCE = {"atol": 1e-07, "rtol": 1e-07}


def test_threshold_otsu(minicircle_grain_threshold_otsu: Grains) -> None:
    """Test threshold calculation"""
    assert isinstance(minicircle_grain_threshold_otsu.thresholds, dict)
    assert minicircle_grain_threshold_otsu.thresholds["upper"] == 1.7566312251807688


def test_threshold_stddev(minicircle_grain_threshold_stddev: Grains) -> None:
    """Test threshold calculation"""
    assert isinstance(minicircle_grain_threshold_stddev.thresholds, dict)
    assert minicircle_grain_threshold_stddev.thresholds == {"upper": 0.8323036290677261, "lower": -0.5092456579760813}


def test_threshold_abs(minicircle_grain_threshold_abs: Grains) -> None:
    """Test threshold calculation"""
    assert isinstance(minicircle_grain_threshold_abs.thresholds, dict)
    assert minicircle_grain_threshold_abs.thresholds == {"upper": 1.0, "lower": -1.0}


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_gaussian_filter_minicircle(minicircle_grain_gaussian_filter: Grains, plotting_config: dict, tmpdir) -> None:
    """Test Gaussian filter."""
    assert isinstance(minicircle_grain_gaussian_filter.images["gaussian_filtered"], np.ndarray)
    assert minicircle_grain_gaussian_filter.images["gaussian_filtered"].shape == (1024, 1024)
    assert minicircle_grain_gaussian_filter.images["gaussian_filtered"].sum() == 169373.26937962
    fig, _ = plot_and_save(
        minicircle_grain_gaussian_filter.images["gaussian_filtered"],
        tmpdir,
        "08-gaussian-filtered.png",
        title="Gaussian Filter",
        **plotting_config
    )
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_mask_minicircle(minicircle_grain_mask: Grains, plotting_config: dict, tmpdir) -> None:
    """Test creation of boolean array for clearing borders."""
    assert isinstance(minicircle_grain_mask.directions["upper"]["mask_grains"], np.ndarray)
    assert minicircle_grain_mask.directions["upper"]["mask_grains"].shape == (1024, 1024)
    assert minicircle_grain_mask.directions["upper"]["mask_grains"].sum() == 52674
    fig, _ = plot_and_save(
        minicircle_grain_mask.directions["upper"]["mask_grains"],
        tmpdir,
        "09-boolean.png",
        title="Boolean Mask",
        **plotting_config
    )
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_clear_border(minicircle_grain_clear_border: Grains, plotting_config: dict, tmpdir) -> None:
    """Test creation of boolean array for clearing borders."""
    assert isinstance(minicircle_grain_clear_border.directions["upper"]["tidied_border"], np.ndarray)
    assert minicircle_grain_clear_border.directions["upper"]["tidied_border"].shape == (1024, 1024)
    assert minicircle_grain_clear_border.directions["upper"]["tidied_border"].sum() == 48700
    fig, _ = plot_and_save(
        minicircle_grain_clear_border.directions["upper"]["tidied_border"],
        tmpdir,
        "10-clear_border.png",
        title="Clear Borders",
        **plotting_config
    )
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_remove_noise(minicircle_grain_remove_noise: Grains, plotting_config: dict, tmpdir) -> None:
    """Test creation of boolean array for clearing borders."""
    assert isinstance(minicircle_grain_remove_noise.directions["upper"]["removed_noise"], np.ndarray)
    assert minicircle_grain_remove_noise.directions["upper"]["removed_noise"].shape == (1024, 1024)
    assert minicircle_grain_remove_noise.directions["upper"]["removed_noise"].sum() == 44054
    fig, _ = plot_and_save(
        minicircle_grain_remove_noise.directions["upper"]["removed_noise"],
        tmpdir,
        "11-remove_noise.png",
        title="Noise Removed",
        **plotting_config
    )
    return fig


def test_calc_minimum_grain_size_pixels(minicircle_minimum_grain_size) -> None:
    """Test calculation of minimum grain size in pixels."""
    assert isinstance(minicircle_minimum_grain_size.minimum_grain_size, float)
    assert minicircle_minimum_grain_size.minimum_grain_size == 1529.25


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_remove_small_objects(minicircle_small_objects_removed: Grains, plotting_config: dict, tmpdir) -> None:
    """Test removal of small objects."""
    assert isinstance(minicircle_small_objects_removed.directions["upper"]["removed_small_objects"], np.ndarray)
    assert minicircle_small_objects_removed.directions["upper"]["removed_small_objects"].shape == (1024, 1024)
    assert minicircle_small_objects_removed.directions["upper"]["removed_small_objects"].sum() == 40573
    fig, _ = plot_and_save(
        minicircle_small_objects_removed.directions["upper"]["removed_small_objects"],
        tmpdir,
        "11-small_objects_removed.png",
        title="Small Objects Removed",
        **plotting_config
    )
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_label_regions(minicircle_grain_labelled_post_removal: Grains, plotting_config: dict, tmpdir) -> None:
    """Test removal of small objects."""
    assert isinstance(minicircle_grain_labelled_post_removal.directions["upper"]["labelled_regions_02"], np.ndarray)
    assert minicircle_grain_labelled_post_removal.directions["upper"]["labelled_regions_02"].shape == (1024, 1024)
    assert minicircle_grain_labelled_post_removal.directions["upper"]["labelled_regions_02"].sum() == 465604
    fig, _ = plot_and_save(
        minicircle_grain_labelled_post_removal.directions["upper"]["labelled_regions_02"],
        tmpdir,
        "12-labelled.png",
        title="Labelled Regions",
        **plotting_config
    )
    return fig


def test_region_properties(minicircle_grain_region_properties_post_removal: np.array) -> None:
    """Test removal of small objects."""
    assert isinstance(minicircle_grain_region_properties_post_removal, list)
    assert len(minicircle_grain_region_properties_post_removal) == 22
    for x in minicircle_grain_region_properties_post_removal:
        assert isinstance(x, RegionProperties)


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_colour_regions(minicircle_grain_coloured: Grains, plotting_config: dict, tmpdir) -> None:
    """Test removal of small objects."""
    assert isinstance(minicircle_grain_coloured.directions["upper"]["coloured_regions"], np.ndarray)
    assert minicircle_grain_coloured.directions["upper"]["coloured_regions"].shape == (1024, 1024, 3)
    assert minicircle_grain_coloured.directions["upper"]["coloured_regions"].sum() == 59179.71000000001
    fig, _ = plot_and_save(
        minicircle_grain_coloured.directions["upper"]["coloured_regions"],
        tmpdir,
        "14-coloured_regions.png",
        title="Coloured Regions",
        **plotting_config
    )
    return fig
