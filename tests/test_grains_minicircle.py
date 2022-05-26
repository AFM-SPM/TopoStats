"""Tests for Find Grains."""
import numpy as np

import pytest
from skimage.measure._regionprops import RegionProperties

from topostats.grains import Grains
from topostats.plottingfuncs import plot_and_save

# Specify the absolute and relattive tolerance for floating point comparison
TOLERANCE = {"atol": 1e-07, "rtol": 1e-07}


def test_threshold(minicircle_grain_threshold: Grains) -> None:
    """Test threshold calculation"""
    assert isinstance(minicircle_grain_threshold.threshold, float)
    assert minicircle_grain_threshold.threshold == 1.7566312251807688


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_gaussian_filter_minicircle(minicircle_grain_gaussian_filter: Grains, tmpdir) -> None:
    """Test Gaussian filter."""
    assert isinstance(minicircle_grain_gaussian_filter.images["gaussian_filtered"], np.ndarray)
    assert minicircle_grain_gaussian_filter.images["gaussian_filtered"].shape == (1024, 1024)
    assert minicircle_grain_gaussian_filter.images["gaussian_filtered"].sum() == 169373.26937962
    fig, _ = plot_and_save(
        minicircle_grain_gaussian_filter.images["gaussian_filtered"],
        tmpdir,
        "08-gaussian-filtered.png",
        title="Gaussian Filter",
    )
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_mask_minicircle(minicircle_grain_mask: np.array, tmpdir) -> None:
    """Test creation of boolean array for clearing borders."""
    assert isinstance(minicircle_grain_mask.images["mask_grains"], np.ndarray)
    assert minicircle_grain_mask.images["mask_grains"].shape == (1024, 1024)
    assert minicircle_grain_mask.images["mask_grains"].sum() == 52674
    fig, _ = plot_and_save(minicircle_grain_mask.images["mask_grains"], tmpdir, "09-boolean.png", title="Boolean Mask")
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_clear_border(minicircle_grain_clear_border: np.array, tmpdir) -> None:
    """Test creation of boolean array for clearing borders."""
    assert isinstance(minicircle_grain_clear_border.images["tidied_border"], np.ndarray)
    assert minicircle_grain_clear_border.images["tidied_border"].shape == (1024, 1024)
    assert minicircle_grain_clear_border.images["tidied_border"].sum() == 48700
    fig, _ = plot_and_save(
        minicircle_grain_clear_border.images["tidied_border"], tmpdir, "10-clear_border.png", title="Clear Borders"
    )
    return fig


def test_calc_minimum_grain_size_pixels(minicircle_minimum_grain_size) -> None:
    """Test calculation of minimum grain size in pixels."""
    assert isinstance(minicircle_minimum_grain_size.minimum_grain_size, float)
    assert minicircle_minimum_grain_size.minimum_grain_size == 1553.25


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_remove_small_objects(minicircle_small_objects_removed: np.array, tmpdir) -> None:
    """Test removal of small objects."""
    assert isinstance(minicircle_small_objects_removed.images["objects_removed"], np.ndarray)
    assert minicircle_small_objects_removed.images["objects_removed"].shape == (1024, 1024)
    assert minicircle_small_objects_removed.images["objects_removed"].sum() == 40573
    fig, _ = plot_and_save(
        minicircle_small_objects_removed.images["objects_removed"],
        tmpdir,
        "11-small_objects_removed.png",
        title="Small Objects Removed",
    )
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_label_regions(minicircle_grain_labelled_post_removal: np.array, tmpdir) -> None:
    """Test removal of small objects."""
    assert isinstance(minicircle_grain_labelled_post_removal.images["labelled_regions"], np.ndarray)
    assert minicircle_grain_labelled_post_removal.images["labelled_regions"].shape == (1024, 1024)
    assert minicircle_grain_labelled_post_removal.images["labelled_regions"].sum() == 465604
    fig, _ = plot_and_save(
        minicircle_grain_labelled_post_removal.images["labelled_regions"],
        tmpdir,
        "12-labelled.png",
        title="Labelled Regions",
    )
    return fig


def test_region_properties(minicircle_grain_region_properties_post_removal: np.array) -> None:
    """Test removal of small objects."""
    assert isinstance(minicircle_grain_region_properties_post_removal.region_properties, list)
    assert len(minicircle_grain_region_properties_post_removal.region_properties) == 22
    for x in minicircle_grain_region_properties_post_removal.region_properties:
        assert isinstance(x, RegionProperties)


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_colour_regions(minicircle_grain_coloured: np.array, tmpdir) -> None:
    """Test removal of small objects."""
    assert isinstance(minicircle_grain_coloured.images["coloured_regions"], np.ndarray)
    assert minicircle_grain_coloured.images["coloured_regions"].shape == (1024, 1024, 3)
    assert minicircle_grain_coloured.images["coloured_regions"].sum() == 59179.71000000001
    fig, _ = plot_and_save(
        minicircle_grain_coloured.images["coloured_regions"],
        tmpdir,
        "14-coloured_regions.png",
        title="Coloured Regions",
    )
    return fig
