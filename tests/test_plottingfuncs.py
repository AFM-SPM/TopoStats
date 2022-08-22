"""Tests of plotting functions."""
import pytest
import numpy as np

from topostats.filters import Filters
from topostats.grains import Grains
from topostats.plottingfuncs import plot_and_save


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_plot_and_save_no_colorbar(minicircle_pixels: Filters, plotting_config: dict, tmpdir) -> None:
    """Test plotting without colorbar"""
    fig, _ = plot_and_save(
        data=minicircle_pixels.images["pixels"],
        output_dir=tmpdir,
        filename="01-raw_heightmap.png",
        pixel_to_nm_scaling_factor=minicircle_pixels.pixel_to_nm_scaling,
        title="Raw Height",
        colorbar=False,
    )
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_plot_and_save_colorbar(minicircle_pixels: Filters, plotting_config: dict, tmpdir) -> None:
    """Test plotting with colorbar"""
    fig, _ = plot_and_save(
        data=minicircle_pixels.images["pixels"],
        output_dir=tmpdir,
        filename="01-raw_heightmap.png",
        pixel_to_nm_scaling_factor=minicircle_pixels.pixel_to_nm_scaling,
        title="Raw Height",
        colorbar=True,
    )
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_plot_and_save_colorbar_afmhot(minicircle_pixels: Filters, plotting_config: dict, tmpdir) -> None:
    """Test plotting with colorbar"""
    plotting_config["cmap"] = "afmhot"
    fig, _ = plot_and_save(
        data=minicircle_pixels.images["pixels"],
        output_dir=tmpdir,
        filename="01-raw_heightmap.png",
        pixel_to_nm_scaling_factor=minicircle_pixels.pixel_to_nm_scaling,
        title="Raw Height",
        colorbar=True,
        cmap="afmhot",
    )
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_plot_and_save_bounding_box(
    minicircle_grain_coloured: Grains,
    minicircle_grain_region_properties_post_removal: Grains,
    plotting_config: dict,
    tmpdir,
) -> None:
    """Test plotting bounding boxes"""
    plotting_config["type"] = "binary"
    fig, _ = plot_and_save(
        data=minicircle_grain_coloured.directions["upper"]["coloured_regions"],
        output_dir=tmpdir,
        filename="15-coloured_regions.png",
        pixel_to_nm_scaling_factor=minicircle_grain_coloured.pixel_to_nm_scaling,
        title="Coloured Regions",
        **plotting_config,
        region_properties=minicircle_grain_region_properties_post_removal,
    )
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_plot_and_save_zrange(minicircle_grain_gaussian_filter: Grains, plotting_config: dict, tmpdir) -> None:
    """Test plotting with colorbar"""
    plotting_config["zrange"] = [-10, 10]
    plotting_config["core_set"] = True
    fig, _ = plot_and_save(
        data=minicircle_grain_gaussian_filter.images["gaussian_filtered"],
        output_dir=tmpdir,
        filename="08_5-z_threshold.png",
        pixel_to_nm_scaling_factor=minicircle_grain_gaussian_filter.pixel_to_nm_scaling,
        title="Raw Height",
        **plotting_config,
    )
    return fig
