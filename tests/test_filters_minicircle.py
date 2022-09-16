"""Tests of the filters module."""
# + pylint: disable=invalid-name
import numpy as np

import pytest
from pySPM.SPM import SPM_image

from topostats.filters import Filters
from topostats.plottingfuncs import Images

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


def test_make_output_directory(tmp_path) -> None:
    """Test loading of scan."""
    assert tmp_path.exists()


def test_extract_channel(minicircle_channel: Filters) -> None:
    """Test extraction of channel."""
    assert isinstance(minicircle_channel, Filters)
    assert isinstance(minicircle_channel.images["extracted_channel"], SPM_image)


def test_extract_pixel_to_nm_scaling(minicircle_pixels: Filters) -> None:
    """Test extraction of pixels to nanometer scaling."""
    assert isinstance(minicircle_pixels, Filters)
    assert minicircle_pixels.pixel_to_nm_scaling == 0.4940029296875


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_extract_pixels(minicircle_pixels: Filters, plotting_config: dict, plot_dict: dict, tmp_path) -> None:
    """Test extraction of channel."""
    assert isinstance(minicircle_pixels.images["pixels"], np.ndarray)
    assert minicircle_pixels.images["pixels"].shape == (1024, 1024)
    assert minicircle_pixels.images["pixels"].sum() == 30695369.188316286
    plotting_config = {**plotting_config, **plot_dict["pixels"]}
    fig, _ = Images(
        data=minicircle_pixels.images["pixels"],
        output_dir=tmp_path,
        pixel_to_nm_scaling_factor=minicircle_pixels.pixel_to_nm_scaling,
        **plotting_config,
    ).plot_and_save()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_align_rows_unmasked(
    minicircle_initial_align: Filters, plotting_config: dict, plot_dict: dict, tmp_path
) -> None:
    """Test initial alignment of rows without mask."""
    assert isinstance(minicircle_initial_align.images["initial_align"], np.ndarray)
    assert minicircle_initial_align.images["initial_align"].shape == (1024, 1024)
    assert minicircle_initial_align.images["initial_align"].sum() == 30754588.04587935
    plotting_config = {**plotting_config, **plot_dict["initial_align"]}
    fig, _ = Images(
        data=minicircle_initial_align.images["initial_align"],
        output_dir=tmp_path,
        pixel_to_nm_scaling_factor=minicircle_initial_align.pixel_to_nm_scaling,
        **plotting_config,
    ).plot_and_save()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_remove_x_y_tilt_unmasked(
    minicircle_initial_tilt_removal: Filters, plotting_config: dict, plot_dict: dict, tmp_path
) -> None:
    """Test removal of tilt without mask."""
    assert isinstance(minicircle_initial_tilt_removal.images["initial_tilt_removal"], np.ndarray)
    assert minicircle_initial_tilt_removal.images["initial_tilt_removal"].shape == (1024, 1024)
    assert minicircle_initial_tilt_removal.images["initial_tilt_removal"].sum() == 29060254.477173675
    plotting_config = {**plotting_config, **plot_dict["initial_tilt_removal"]}
    fig, _ = Images(
        data=minicircle_initial_tilt_removal.images["initial_tilt_removal"],
        output_dir=tmp_path,
        pixel_to_nm_scaling_factor=minicircle_initial_tilt_removal.pixel_to_nm_scaling,
        **plotting_config,
    ).plot_and_save()
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
def test_get_mask(minicircle_mask: Filters, plotting_config: dict, plot_dict: dict, tmp_path) -> None:
    """Test derivation of mask."""
    plotting_config["type"] = "binary"
    assert isinstance(minicircle_mask.images["mask"], np.ndarray)
    assert minicircle_mask.images["mask"].shape == (1024, 1024)
    assert minicircle_mask.images["mask"].sum() == 82159
    plotting_config = {**plotting_config, **plot_dict["mask"]}
    fig, _ = Images(
        data=minicircle_mask.images["mask"],
        output_dir=tmp_path,
        pixel_to_nm_scaling_factor=minicircle_mask.pixel_to_nm_scaling,
        **plotting_config,
    ).plot_and_save()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_align_rows_masked(minicircle_masked_align: Filters, plotting_config: dict, plot_dict: dict, tmp_path) -> None:
    """Test alignment of rows without mask."""
    assert isinstance(minicircle_masked_align.images["masked_align"], np.ndarray)
    assert minicircle_masked_align.images["masked_align"].shape == (1024, 1024)
    assert minicircle_masked_align.images["masked_align"].sum() == 29077928.954810616
    plotting_config = {**plotting_config, **plot_dict["masked_align"]}
    fig, _ = Images(
        data=minicircle_masked_align.images["masked_align"],
        output_dir=tmp_path,
        pixel_to_nm_scaling_factor=minicircle_masked_align.pixel_to_nm_scaling,
        **plotting_config,
    ).plot_and_save()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_remove_x_y_tilt_masked(
    minicircle_masked_tilt_removal: Filters, plotting_config: dict, plot_dict: dict, tmp_path
) -> None:
    """Test removal of tilt without mask."""
    assert isinstance(minicircle_masked_tilt_removal.images["masked_tilt_removal"], np.ndarray)
    assert minicircle_masked_tilt_removal.images["masked_tilt_removal"].shape == (1024, 1024)
    assert minicircle_masked_tilt_removal.images["masked_tilt_removal"].sum() == 29074955.22136825
    plotting_config = {**plotting_config, **plot_dict["masked_tilt_removal"]}
    fig, _ = Images(
        data=minicircle_masked_tilt_removal.images["masked_tilt_removal"],
        output_dir=tmp_path,
        pixel_to_nm_scaling_factor=minicircle_masked_tilt_removal.pixel_to_nm_scaling,
        **plotting_config,
    ).plot_and_save()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_average_background(
    minicircle_zero_average_background: Filters, plotting_config: dict, plot_dict: dict, tmp_path
) -> None:
    """Test zero-averaging of background."""
    assert isinstance(minicircle_zero_average_background.images["zero_averaged_background"], np.ndarray)
    assert minicircle_zero_average_background.images["zero_averaged_background"].shape == (1024, 1024)
    assert minicircle_zero_average_background.images["zero_averaged_background"].sum() == 169375.4175476962
    plotting_config = {**plotting_config, **plot_dict["zero_averaged_background"]}
    fig, _ = Images(
        data=minicircle_zero_average_background.images["zero_averaged_background"],
        output_dir=tmp_path,
        pixel_to_nm_scaling_factor=minicircle_zero_average_background.pixel_to_nm_scaling,
        **plotting_config,
    ).plot_and_save()
    return fig

@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_gaussian_filter(
    minicircle_grain_gaussian_filter: Filters, plotting_config: dict, plot_dict: dict, tmp_path
) -> None:
    """Test gaussian filter applied to background."""
    assert isinstance(minicircle_grain_gaussian_filter.images["gaussian_filtered"], np.ndarray)
    assert minicircle_grain_gaussian_filter.images["gaussian_filtered"].shape == (1024, 1024)
    assert minicircle_grain_gaussian_filter.images["gaussian_filtered"].sum() == 169373.05336999876
    plotting_config = {**plotting_config, **plot_dict["gaussian_filtered"]}
    fig, _ = Images(
        data=minicircle_grain_gaussian_filter.images["gaussian_filtered"],
        output_dir=tmp_path,
        pixel_to_nm_scaling_factor=minicircle_grain_gaussian_filter.pixel_to_nm_scaling,
        **plotting_config,
    ).plot_and_save()
    return fig

# FIXME (2022-07-11): More tests of alignment/tilt removal and average background when methods other than otsu are used
