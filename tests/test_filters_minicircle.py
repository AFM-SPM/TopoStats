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


def test_make_output_directory(tmpdir) -> None:
    """Test loading of scan."""
    assert tmpdir.exists()


def test_extract_channel(minicircle_channel: Filters) -> None:
    """Test extraction of channel."""
    assert isinstance(minicircle_channel, Filters)
    assert isinstance(minicircle_channel.images["extracted_channel"], SPM_image)


def test_extract_pixel_to_nm_scaling(minicircle_pixels: Filters) -> None:
    """Test extraction of pixels to nanometer scaling."""
    assert isinstance(minicircle_pixels, Filters)
    assert minicircle_pixels.pixel_to_nm_scaling == 0.4940029296875


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_extract_pixels(minicircle_pixels: Filters, plotting_config: dict, tmpdir) -> None:
    """Test extraction of channel."""
    print(f"######### plotting_config : {plotting_config}")
    assert isinstance(minicircle_pixels.images["pixels"], np.ndarray)
    assert minicircle_pixels.images["pixels"].shape == (1024, 1024)
    assert minicircle_pixels.images["pixels"].sum() == 30695369.188316286
    fig, _ = plot_and_save(
        minicircle_pixels.images["pixels"],
        tmpdir,
        "01-raw_heightmap.png",
        title="Raw Height",
        **plotting_config,
    )
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_align_rows_unmasked(minicircle_initial_align: Filters, plotting_config: dict, tmpdir) -> None:
    """Test initial alignment of rows without mask."""
    assert isinstance(minicircle_initial_align.images["initial_align"], np.ndarray)
    assert minicircle_initial_align.images["initial_align"].shape == (1024, 1024)
    assert minicircle_initial_align.images["initial_align"].sum() == 30754588.04587935
    fig, _ = plot_and_save(
        minicircle_initial_align.images["initial_align"],
        tmpdir,
        "02-initial_align_rows_unmasked.png",
        title="Initial Align (Unmasked)",
        **plotting_config,
    )
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_remove_x_y_tilt_unmasked(minicircle_initial_tilt_removal: Filters, plotting_config: dict, tmpdir) -> None:
    """Test removal of tilt without mask."""
    assert isinstance(minicircle_initial_tilt_removal.images["initial_tilt_removal"], np.ndarray)
    assert minicircle_initial_tilt_removal.images["initial_tilt_removal"].shape == (1024, 1024)
    assert minicircle_initial_tilt_removal.images["initial_tilt_removal"].sum() == 29060254.477173675
    fig, _ = plot_and_save(
        minicircle_initial_tilt_removal.images["initial_tilt_removal"],
        tmpdir,
        "03-initial_tilt_removal_unmasked.png",
        title="Initial Tilt Removal (Unmasked)",
        **plotting_config,
    )
    return fig


def test_get_threshold_otsu(minicircle_threshold_otsu: np.array) -> None:
    """Test calculation of threshold."""
    assert isinstance(minicircle_threshold_otsu.thresholds, dict)
    assert minicircle_threshold_otsu.thresholds["upper"] == 28.58495414588038


def test_get_threshold_stddev(minicircle_threshold_stddev: np.array) -> None:
    """Test calculation of threshold."""
    assert isinstance(minicircle_threshold_stddev.thresholds, dict)
    assert minicircle_threshold_stddev.thresholds == {"upper": 28.382985321353974, "lower": 27.045051324866566}


def test_get_threshold_abs(minicircle_threshold_abs: np.array) -> None:
    """Test calculation of threshold."""
    assert isinstance(minicircle_threshold_abs.thresholds, dict)
    assert minicircle_threshold_abs.thresholds == {"upper": 1.5, "lower": -1.5}


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_get_mask(minicircle_mask: Filters, plotting_config: dict, tmpdir) -> None:
    """Test derivation of mask."""
    plotting_config["type"] = "binary"
    assert isinstance(minicircle_mask.images["mask"], np.ndarray)
    assert minicircle_mask.images["mask"].shape == (1024, 1024)
    assert minicircle_mask.images["mask"].sum() == 82159
    fig, _ = plot_and_save(
        minicircle_mask.images["mask"], tmpdir, "04-binary_mask.png", title="Binary Mask", **plotting_config
    )
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_align_rows_masked(minicircle_masked_align: Filters, plotting_config: dict, tmpdir) -> None:
    """Test alignment of rows without mask."""
    assert isinstance(minicircle_masked_align.images["masked_align"], np.ndarray)
    assert minicircle_masked_align.images["masked_align"].shape == (1024, 1024)
    assert minicircle_masked_align.images["masked_align"].sum() == 29077928.954810616
    fig, _ = plot_and_save(
        minicircle_masked_align.images["masked_align"],
        tmpdir,
        "05-masked_align_rows.png",
        title="Secondary Align (Masked)",
        **plotting_config,
    )
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_remove_x_y_tilt_masked(minicircle_masked_tilt_removal: Filters, plotting_config: dict, tmpdir) -> None:
    """Test removal of tilt without mask."""
    assert isinstance(minicircle_masked_tilt_removal.images["masked_tilt_removal"], np.ndarray)
    assert minicircle_masked_tilt_removal.images["masked_tilt_removal"].shape == (1024, 1024)
    assert minicircle_masked_tilt_removal.images["masked_tilt_removal"].sum() == 29074955.22136825
    fig, _ = plot_and_save(
        minicircle_masked_tilt_removal.images["masked_tilt_removal"],
        tmpdir,
        "06-secondary_tilt_removal_masked.png",
        title="Secondary Tilt Removal (Masked)",
        **plotting_config,
    )
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_average_background(minicircle_zero_average_background: Filters, plotting_config: dict, tmpdir) -> None:
    """Test zero-averaging of background."""
    assert isinstance(minicircle_zero_average_background.images["zero_averaged_background"], np.ndarray)
    assert minicircle_zero_average_background.images["zero_averaged_background"].shape == (1024, 1024)
    assert minicircle_zero_average_background.images["zero_averaged_background"].sum() == 169375.41754769627
    fig, _ = plot_and_save(
        minicircle_zero_average_background.images["zero_averaged_background"],
        tmpdir,
        "07-zero_average_background.png",
        title="Zero Average Background",
        **plotting_config,
    )
    return fig


# FIXME (2022-07-11): More tests of alignment/tilt removal and average background when methods other than otsu are used
