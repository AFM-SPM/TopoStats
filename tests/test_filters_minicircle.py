"""Tests of the filters module."""
import numpy as np

import pytest
from pySPM.SPM import SPM_image
from pySPM.Bruker import Bruker

from topostats.plottingfuncs import plot_and_save

# Specify the absolute and relattive tolerance for floating point comparison
TOLERANCE = {'atol': 1e-07, 'rtol': 1e-07}


def test_load_scan(minicircle) -> None:
    """Test loading of image"""

    assert isinstance(minicircle, Bruker)


def test_extract_channel(minicircle_channel) -> None:
    """Test extraction of channel."""
    assert isinstance(minicircle_channel, SPM_image)


@pytest.mark.mpl_image_compare(baseline_dir='resources/img/')
def test_extract_pixels(minicircle_pixels, tmpdir) -> None:
    """Test extraction of channel."""
    assert isinstance(minicircle_pixels, np.ndarray)
    assert minicircle_pixels.shape == (1024, 1024)
    assert minicircle_pixels.sum() == 30695369.188316286
    fig, _ = plot_and_save(minicircle_pixels, tmpdir / '01-raw_heightmap.png', title='Raw Height')
    return fig


@pytest.mark.mpl_image_compare(baseline_dir='resources/img/')
def test_align_rows_unmasked(minicircle_initial_align: np.array, tmpdir) -> None:
    """Test alignment of rows without mask."""
    assert isinstance(minicircle_initial_align, np.ndarray)
    assert minicircle_initial_align.shape == (1024, 1024)
    assert minicircle_initial_align.sum() == 30754588.04587935
    fig, _ = plot_and_save(minicircle_initial_align,
                           tmpdir / '02-initial_align_rows_unmasked.png',
                           title='Initial Align (Unmasked)')
    return fig


@pytest.mark.mpl_image_compare(baseline_dir='resources/img/')
def test_remove_x_y_tilt_unmasked(minicircle_initial_tilt_removal: np.array, tmpdir) -> None:
    """Test removal of tilt without mask."""
    assert isinstance(minicircle_initial_tilt_removal, np.ndarray)
    assert minicircle_initial_tilt_removal.shape == (1024, 1024)
    assert minicircle_initial_tilt_removal.sum() == 29060254.477173675
    fig, _ = plot_and_save(minicircle_initial_tilt_removal,
                           tmpdir / '03-initial_tilt_removal_unmasked.png',
                           title='Initial Tilt Removal (Unmasked)')
    return fig


def test_get_threshold(minicircle_threshold: np.array) -> None:
    """Test calculation of threshold."""
    assert isinstance(minicircle_threshold, float)
    assert minicircle_threshold == 28.58495414588038


@pytest.mark.mpl_image_compare(baseline_dir='resources/img/')
def test_get_mask(minicircle_mask: np.array, tmpdir) -> None:
    """Test derivation of mask."""
    assert isinstance(minicircle_mask, np.ndarray)
    assert minicircle_mask.shape == (1024, 1024)
    assert minicircle_mask.sum() == 82159
    fig, _ = plot_and_save(minicircle_mask, tmpdir / '04-binary_mask.png', title='Binary Mask')
    return fig


@pytest.mark.mpl_image_compare(baseline_dir='resources/img/')
def test_align_rows_masked(minicircle_masked_align: np.array, tmpdir) -> None:
    """Test alignment of rows without mask."""
    assert isinstance(minicircle_masked_align, np.ndarray)
    assert minicircle_masked_align.shape == (1024, 1024)
    assert minicircle_masked_align.sum() == 29068698.626152653
    fig, _ = plot_and_save(minicircle_masked_align,
                           tmpdir / '05-masked_align_rows.png',
                           title='Secondary Align (Masked)')
    return fig


@pytest.mark.mpl_image_compare(baseline_dir='resources/img/')
def test_remove_x_y_tilt_masked(minicircle_masked_tilt_removal: np.array, tmpdir) -> None:
    """Test removal of tilt without mask."""
    assert isinstance(minicircle_masked_tilt_removal, np.ndarray)
    assert minicircle_masked_tilt_removal.shape == (1024, 1024)
    assert minicircle_masked_tilt_removal.sum() == 29067389.916735478
    fig, _ = plot_and_save(minicircle_masked_tilt_removal,
                           tmpdir / '06-secondary_tilt_removal_masked.png',
                           title='Secondary Tilt Removal (Masked)')
    return fig


@pytest.mark.mpl_image_compare(baseline_dir='resources/img/')
def test_zero_average_background(minicircle_zero_average_background: np.array, tmpdir) -> None:
    """Test zero-averaging of background."""
    assert isinstance(minicircle_zero_average_background, np.ndarray)
    assert minicircle_zero_average_background.shape == (1024, 1024)
    assert minicircle_zero_average_background.sum() == 134305.44906065107
    fig, _ = plot_and_save(minicircle_zero_average_background,
                           tmpdir / '07-zero_average_background.png',
                           title='Zero Average Background')
    return fig
