"""Tests for Find Grains."""
import numpy as np

import pytest
from skimage.measure._regionprops import RegionProperties

from topostats.grains import Grains
from topostats.plottingfuncs import Images

# Specify the absolute and relattive tolerance for floating point comparison
TOLERANCE = {"atol": 1e-07, "rtol": 1e-07}


def test_threshold_otsu(minicircle_grain_threshold_otsu: Grains) -> None:
    """Test threshold calculation"""
    assert isinstance(minicircle_grain_threshold_otsu.thresholds, dict)
    assert minicircle_grain_threshold_otsu.thresholds["upper"] == 1.6774424068410074


def test_threshold_stddev(minicircle_grain_threshold_stddev: Grains) -> None:
    """Test threshold calculation"""
    assert isinstance(minicircle_grain_threshold_stddev.thresholds, dict)
    assert minicircle_grain_threshold_stddev.thresholds == {"upper": 0.7712698500699151, "lower": -0.4482163882893693}


def test_threshold_abs(minicircle_grain_threshold_abs: Grains) -> None:
    """Test threshold calculation"""
    assert isinstance(minicircle_grain_threshold_abs.thresholds, dict)
    assert minicircle_grain_threshold_abs.thresholds == {"upper": 1.0, "lower": -1.0}


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_mask_minicircle(minicircle_grain_mask: Grains, plotting_config: dict, plot_dict: dict, tmp_path) -> None:
    """Test creation of boolean array for clearing borders."""
    plotting_config["image_type"] = "binary"
    assert isinstance(minicircle_grain_mask.directions["upper"]["mask_grains"], np.ndarray)
    assert minicircle_grain_mask.directions["upper"]["mask_grains"].shape == (1024, 1024)
    assert minicircle_grain_mask.directions["upper"]["mask_grains"].sum() == 55648
    plotting_config = {**plotting_config, **plot_dict["mask_grains"]}
    fig, _ = Images(
        data=minicircle_grain_mask.directions["upper"]["mask_grains"],
        output_dir=tmp_path,
        pixel_to_nm_scaling=minicircle_grain_mask.pixel_to_nm_scaling,
        **plotting_config,
    ).plot_and_save()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_clear_border(minicircle_grain_clear_border: Grains, plotting_config: dict, plot_dict: dict, tmp_path) -> None:
    """Test creation of boolean array for clearing borders."""
    plotting_config["image_type"] = "binary"
    assert isinstance(minicircle_grain_clear_border.directions["upper"]["tidied_border"], np.ndarray)
    assert minicircle_grain_clear_border.directions["upper"]["tidied_border"].shape == (1024, 1024)
    assert minicircle_grain_clear_border.directions["upper"]["tidied_border"].sum() == 51457
    plotting_config = {**plotting_config, **plot_dict["tidied_border"]}
    fig, _ = Images(
        data=minicircle_grain_clear_border.directions["upper"]["tidied_border"],
        output_dir=tmp_path,
        pixel_to_nm_scaling=minicircle_grain_clear_border.pixel_to_nm_scaling,
        **plotting_config,
    ).plot_and_save()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_remove_noise(minicircle_grain_remove_noise: Grains, plotting_config: dict, plot_dict: dict, tmp_path) -> None:
    """Test creation of boolean array for clearing borders."""
    plotting_config["image_type"] = "binary"
    assert isinstance(minicircle_grain_remove_noise.directions["upper"]["removed_noise"], np.ndarray)
    assert minicircle_grain_remove_noise.directions["upper"]["removed_noise"].shape == (1024, 1024)
    assert minicircle_grain_remove_noise.directions["upper"]["removed_noise"].sum() == 46484
    plotting_config = {**plotting_config, **plot_dict["removed_noise"]}
    fig, _ = Images(
        data=minicircle_grain_remove_noise.directions["upper"]["removed_noise"],
        output_dir=tmp_path,
        pixel_to_nm_scaling=minicircle_grain_remove_noise.pixel_to_nm_scaling,
        **plotting_config,
    ).plot_and_save()
    return fig


def test_calc_minimum_grain_size_pixels(minicircle_minimum_grain_size) -> None:
    """Test calculation of minimum grain size in pixels."""
    assert isinstance(minicircle_minimum_grain_size.minimum_grain_size, float)
    assert minicircle_minimum_grain_size.minimum_grain_size == 1570.25


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_remove_small_objects(
    minicircle_small_objects_removed: Grains, plotting_config: dict, plot_dict: dict, tmp_path
) -> None:
    """Test removal of small objects."""
    plotting_config["image_type"] = "binary"
    assert isinstance(minicircle_small_objects_removed.directions["upper"]["removed_small_objects"], np.ndarray)
    assert minicircle_small_objects_removed.directions["upper"]["removed_small_objects"].shape == (1024, 1024)
    assert minicircle_small_objects_removed.directions["upper"]["removed_small_objects"].sum() == 42378
    plotting_config = {**plotting_config, **plot_dict["removed_small_objects"]}
    fig, _ = Images(
        data=minicircle_small_objects_removed.directions["upper"]["removed_small_objects"],
        output_dir=tmp_path,
        pixel_to_nm_scaling=minicircle_small_objects_removed.pixel_to_nm_scaling,
        **plotting_config,
    ).plot_and_save()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_area_thresholding(
    minicircle_area_thresholding: Grains, plotting_config: dict, plot_dict: dict, tmp_path
) -> None:
    """Test removal of small objects via absolute area thresholding."""
    plotting_config["image_type"] = "binary"
    assert isinstance(minicircle_area_thresholding.directions["upper"]["removed_small_objects"], np.ndarray)
    assert minicircle_area_thresholding.directions["upper"]["removed_small_objects"].shape == (1024, 1024)
    assert minicircle_area_thresholding.directions["upper"]["removed_small_objects"].sum() == 442441
    plotting_config = {**plotting_config, **plot_dict["removed_small_objects"]}
    fig, _ = Images(
        data=minicircle_area_thresholding.directions["upper"]["removed_small_objects"],
        output_dir=tmp_path,
        pixel_to_nm_scaling=minicircle_area_thresholding.pixel_to_nm_scaling,
        **plotting_config,
    ).plot_and_save()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_label_regions(
    minicircle_grain_labelled_post_removal: Grains, plotting_config: dict, plot_dict: dict, tmp_path
) -> None:
    """Test removal of small objects."""
    plotting_config["image_type"] = "binary"
    assert isinstance(minicircle_grain_labelled_post_removal.directions["upper"]["labelled_regions_02"], np.ndarray)
    assert minicircle_grain_labelled_post_removal.directions["upper"]["labelled_regions_02"].shape == (1024, 1024)
    assert minicircle_grain_labelled_post_removal.directions["upper"]["labelled_regions_02"].sum() == 484819
    plotting_config = {**plotting_config, **plot_dict["labelled_regions_02"]}
    fig, _ = Images(
        data=minicircle_grain_labelled_post_removal.directions["upper"]["labelled_regions_02"],
        output_dir=tmp_path,
        pixel_to_nm_scaling=minicircle_grain_labelled_post_removal.pixel_to_nm_scaling,
        **plotting_config,
    ).plot_and_save()
    return fig


def test_region_properties(minicircle_grain_region_properties_post_removal: np.array) -> None:
    """Test removal of small objects."""
    assert isinstance(minicircle_grain_region_properties_post_removal, list)
    assert len(minicircle_grain_region_properties_post_removal) == 22
    for x in minicircle_grain_region_properties_post_removal:
        assert isinstance(x, RegionProperties)


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_colour_regions(minicircle_grain_coloured: Grains, plotting_config: dict, plot_dict: dict, tmp_path) -> None:
    """Test removal of small objects."""
    plotting_config["image_type"] = "binary"
    assert isinstance(minicircle_grain_coloured.directions["upper"]["coloured_regions"], np.ndarray)
    assert minicircle_grain_coloured.directions["upper"]["coloured_regions"].shape == (1024, 1024, 3)
    assert minicircle_grain_coloured.directions["upper"]["coloured_regions"].sum() == 61714.27500000001
    plotting_config = {**plotting_config, **plot_dict["coloured_regions"]}
    fig, _ = Images(
        data=minicircle_grain_coloured.directions["upper"]["coloured_regions"],
        output_dir=tmp_path,
        pixel_to_nm_scaling=minicircle_grain_coloured.pixel_to_nm_scaling,
        **plotting_config,
    ).plot_and_save()
    return fig
