"""Tests of the filters module."""
# + pylint: disable=invalid-name
import numpy as np

import pytest
from pySPM.SPM import SPM_image

from topostats.filters import Filters
from topostats.plottingfuncs import plot_and_save

# Specify the absolute and relattive tolerance for floating point comparison
TOLERANCE = {"atol": 1e-07, "rtol": 1e-07}


def test_extract_img_name(minicircle_filename: Filters) -> None:
    """Test extracting image name."""
    assert isinstance(minicircle_filename, Filters)
    assert isinstance(minicircle_filename.filename, str)
    assert minicircle_filename.filename == "minicircle"


def test_load_scan(minicircle_load_scan: Filters) -> None:
    """Test loading of scan."""
    assert isinstance(minicircle_load_scan, Filters)


def test_make_output_directory(minicircle_make_output_directory: Filters, tmpdir) -> None:
    """Test loading of scan."""
    target_dir = tmpdir / minicircle_make_output_directory.filename
    assert target_dir.exists()


def test_extract_channel(minicircle_channel: Filters) -> None:
    """Test extraction of channel."""
    assert isinstance(minicircle_channel, Filters)
    assert isinstance(minicircle_channel.images["extracted_channel"], SPM_image)


def test_extract_pixel_to_nm_scaling(minicircle_extract_pixel_to_nm_scaling: Filters) -> None:
    """Test extraction of pixels to nanometer scaling."""
    assert isinstance(minicircle_extract_pixel_to_nm_scaling, Filters)
    assert minicircle_extract_pixel_to_nm_scaling.pixel_to_nm_scaling == 0.4940029296875


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_extract_pixels(minicircle_pixels, tmpdir) -> None:
    """Test extraction of channel."""
    assert isinstance(minicircle_pixels.images["pixels"], np.ndarray)
    assert minicircle_pixels.images["pixels"].shape == (1024, 1024)
    assert minicircle_pixels.images["pixels"].sum() == 30695369.188316286
    fig, _ = plot_and_save(minicircle_pixels.images["pixels"], tmpdir, "01-raw_heightmap.png", title="Raw Height")
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_align_rows_unmasked(minicircle_initial_align: np.array, tmpdir) -> None:
    """Test initial alignment of rows without mask."""
    assert isinstance(minicircle_initial_align.images["initial_align"], np.ndarray)
    assert minicircle_initial_align.images["initial_align"].shape == (1024, 1024)
    assert minicircle_initial_align.images["initial_align"].sum() == 30754588.04587935
    fig, _ = plot_and_save(
        minicircle_initial_align.images["initial_align"],
        tmpdir,
        "02-initial_align_rows_unmasked.png",
        title="Initial Align (Unmasked)",
    )
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_remove_x_y_tilt_unmasked(minicircle_initial_tilt_removal: np.array, tmpdir) -> None:
    """Test removal of tilt without mask."""
    assert isinstance(minicircle_initial_tilt_removal.images["initial_tilt_removal"], np.ndarray)
    assert minicircle_initial_tilt_removal.images["initial_tilt_removal"].shape == (1024, 1024)
    assert minicircle_initial_tilt_removal.images["initial_tilt_removal"].sum() == 29060254.477173675
    fig, _ = plot_and_save(
        minicircle_initial_tilt_removal.images["initial_tilt_removal"],
        tmpdir,
        "03-initial_tilt_removal_unmasked.png",
        title="Initial Tilt Removal (Unmasked)",
    )
    return fig


def test_get_threshold(minicircle_threshold: np.array) -> None:
    """Test calculation of threshold."""
    assert isinstance(minicircle_threshold.threshold, float)
    assert minicircle_threshold.threshold == 28.58495414588038


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_get_mask(minicircle_mask: np.array, tmpdir) -> None:
    """Test derivation of mask."""
    assert isinstance(minicircle_mask.images["mask"], np.ndarray)
    assert minicircle_mask.images["mask"].shape == (1024, 1024)
    assert minicircle_mask.images["mask"].sum() == 82159
    fig, _ = plot_and_save(minicircle_mask.images["mask"], tmpdir, "04-binary_mask.png", title="Binary Mask")
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_align_rows_masked(minicircle_masked_align: np.array, tmpdir) -> None:
    """Test alignment of rows without mask."""
    assert isinstance(minicircle_masked_align.images["masked_align"], np.ndarray)
    assert minicircle_masked_align.images["masked_align"].shape == (1024, 1024)
    assert minicircle_masked_align.images["masked_align"].sum() == 29068698.626152653
    fig, _ = plot_and_save(
        minicircle_masked_align.images["masked_align"],
        tmpdir,
        "05-masked_align_rows.png",
        title="Secondary Align (Masked)",
    )
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_remove_x_y_tilt_masked(minicircle_masked_tilt_removal: np.array, tmpdir) -> None:
    """Test removal of tilt without mask."""
    assert isinstance(minicircle_masked_tilt_removal.images["masked_tilt_removal"], np.ndarray)
    assert minicircle_masked_tilt_removal.images["masked_tilt_removal"].shape == (1024, 1024)
    assert minicircle_masked_tilt_removal.images["masked_tilt_removal"].sum() == 29067389.916735478
    fig, _ = plot_and_save(
        minicircle_masked_tilt_removal.images["masked_tilt_removal"],
        tmpdir,
        "06-secondary_tilt_removal_masked.png",
        title="Secondary Tilt Removal (Masked)",
    )
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_average_background(minicircle_zero_average_background: np.array, tmpdir) -> None:
    """Test zero-averaging of background."""
    assert isinstance(minicircle_zero_average_background.images["zero_averaged_background"], np.ndarray)
    assert minicircle_zero_average_background.images["zero_averaged_background"].shape == (1024, 1024)
    assert minicircle_zero_average_background.images["zero_averaged_background"].sum() == 134305.44906065107
    fig, _ = plot_and_save(
        minicircle_zero_average_background.images["zero_averaged_background"],
        tmpdir,
        "07-zero_average_background.png",
        title="Zero Average Background",
    )
    return fig
