"""Tests of plotting functions."""
import pytest

from topostats.filters import Filters
from topostats.plottingfuncs import plot_and_save


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_plot_and_save_no_colorbar(minicircle_pixels: Filters, plotting_config: dict, tmpdir) -> None:
    """Test plotting without colorbar"""
    plotting_config["colorbar"] = False
    fig, _ = plot_and_save(
        minicircle_pixels.images["pixels"], tmpdir, "01-raw_heightmap.png", title="Raw Height", **plotting_config
    )
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_plot_and_save_colorbar(minicircle_pixels: Filters, plotting_config: dict, tmpdir) -> None:
    """Test plotting with colorbar"""
    fig, _ = plot_and_save(
        minicircle_pixels.images["pixels"], tmpdir, "01-raw_heightmap.png", title="Raw Height", **plotting_config
    )
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_plot_and_save_colorbar_afmhot(minicircle_pixels: Filters, plotting_config: dict, tmpdir) -> None:
    """Test plotting with colorbar"""
    plotting_config["cmap"] = "afmhot"
    fig, _ = plot_and_save(
        minicircle_pixels.images["pixels"], tmpdir, "01-raw_heightmap.png", title="Raw Height", **plotting_config
    )
    return fig
