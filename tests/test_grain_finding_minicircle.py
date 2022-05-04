"""Tests of the filters module."""
import numpy as np

import pytest
from skimage.measure._regionprops import RegionProperties

from topostats.plottingfuncs import plot_and_save
from topostats.find_grains import get_lower_threshold


def test_lower_threshold(minicircle_zero_average_background: np.array, grain_config: dict) -> None:
    """Test calculation of lower threshold"""
    lower_threshold = get_lower_threshold(minicircle_zero_average_background, grain_config['lower_threshold'])
    assert isinstance(lower_threshold, float)
    assert lower_threshold == 1.7005720647153635


@pytest.mark.mpl_image_compare(baseline_dir='resources/img/')
def test_gaussian_filter_minicircle(minicircle_grain_gaussian_filter: np.array, tmpdir) -> None:
    """Test Gaussian filter."""
    assert isinstance(minicircle_grain_gaussian_filter, np.ndarray)
    assert minicircle_grain_gaussian_filter.shape == (1024, 1024)
    assert minicircle_grain_gaussian_filter.sum() == 134285.453051399
    fig, _ = plot_and_save(minicircle_grain_gaussian_filter,
                           tmpdir / '08-gaussian-filtered.png',
                           title='Gaussian Filter')
    return fig


@pytest.mark.mpl_image_compare(baseline_dir='resources/img/')
def test_boolean_minicircle(minicircle_grain_boolean: np.array, tmpdir) -> None:
    """Test creation of boolean array for clearing borders."""
    assert isinstance(minicircle_grain_boolean, np.ndarray)
    assert minicircle_grain_boolean.shape == (1024, 1024)
    assert minicircle_grain_boolean.sum() == 48331
    fig, _ = plot_and_save(minicircle_grain_boolean, tmpdir / '09-boolean.png', title='Boolean Mask')
    return fig


@pytest.mark.mpl_image_compare(baseline_dir='resources/img/')
def test_clear_border(minicircle_grain_clear_border: np.array, tmpdir) -> None:
    """Test creation of boolean array for clearing borders."""
    assert isinstance(minicircle_grain_clear_border, np.ndarray)
    assert minicircle_grain_clear_border.shape == (1024, 1024)
    assert minicircle_grain_clear_border.sum() == 42059
    fig, _ = plot_and_save(minicircle_grain_clear_border, tmpdir / '10-clear_border.png', title='Clear Borders')
    return fig


@pytest.mark.mpl_image_compare(baseline_dir='resources/img/')
def test_remove_small_objects(minicircle_grain_small_objects_removed: np.array, tmpdir) -> None:
    """Test removal of small objects."""
    assert isinstance(minicircle_grain_small_objects_removed, np.ndarray)
    assert minicircle_grain_small_objects_removed.shape == (1024, 1024)
    assert minicircle_grain_small_objects_removed.sum() == 36334
    fig, _ = plot_and_save(minicircle_grain_small_objects_removed,
                           tmpdir / '11-small_objects_removed.png',
                           title='Small Objects Removed')
    return fig


@pytest.mark.mpl_image_compare(baseline_dir='resources/img/')
def test_label_regions(minicircle_grain_labelled: np.array, tmpdir) -> None:
    """Test removal of small objects."""
    assert isinstance(minicircle_grain_labelled, np.ndarray)
    assert minicircle_grain_labelled.shape == (1024, 1024)
    assert minicircle_grain_labelled.sum() == 404547
    fig, _ = plot_and_save(minicircle_grain_labelled, tmpdir / '12-labelled.png', title='Labelled Regions')
    return fig


@pytest.mark.mpl_image_compare(baseline_dir='resources/img/')
def test_colour_regions(minicircle_grain_coloured: np.array, tmpdir) -> None:
    """Test removal of small objects."""
    assert isinstance(minicircle_grain_coloured, np.ndarray)
    assert minicircle_grain_coloured.shape == (1024, 1024, 3)
    assert minicircle_grain_coloured.sum() == 53997.467000000004
    fig, _ = plot_and_save(minicircle_grain_coloured, tmpdir / '14-coloured_regions.png', title='Coloured Regions')
    return fig


def test_region_properties(minicircle_grain_region_properties: np.array) -> None:
    """Test removal of small objects."""
    assert isinstance(minicircle_grain_region_properties, list)
    assert len(minicircle_grain_region_properties) == 21
    for x in minicircle_grain_region_properties:
        assert isinstance(x, RegionProperties)
