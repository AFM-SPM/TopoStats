"""Tests of plotting functions."""
import pytest

from topostats.plottingfuncs import plot_and_save


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_plot_and_save_no_colorbar(minicircle_pixels, tmpdir) -> None:
    """Test plotting without colorbar"""
    fig, _ = plot_and_save(
        minicircle_pixels.images["pixels"], tmpdir, "01-raw_heightmap.png", title="Raw Height", colorbar=False
    )
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_plot_and_save_colorbar(minicircle_pixels, tmpdir) -> None:
    """Test plotting with colorbar"""
    fig, _ = plot_and_save(
        minicircle_pixels.images["pixels"], tmpdir, "01-raw_heightmap.png", title="Raw Height", colorbar=True
    )
    return fig
