"""Tests of plotting functions."""
from types import NoneType
import pytest
from pathlib import Path
import numpy as np
import skimage.io as io

from topostats.filters import Filters
from topostats.grains import Grains
from topostats.plottingfuncs import Images

@pytest.mark.parametrize("data2, axes_colorbar, region_properties", [
(np.random.rand(10,10), True, None),
(None, True, None),
(None, False, True)
])
def test_save_figure(
    data2: np.ndarray,
    axes_colorbar: bool,
    region_properties: bool,
    minicircle_grain_region_properties_post_removal: Grains,
    tmp_path: Path):
    """Tests that an image is saved and a figure returned"""
    if region_properties:
        region_properties=minicircle_grain_region_properties_post_removal
    fig, ax = Images(
        data=np.random.rand(10,10),
        output_dir=tmp_path,
        filename="result.png",
        data2=data2,
        colorbar=axes_colorbar,
        axes=axes_colorbar,
        region_properties=region_properties,
    ).save_figure()
    assert Path(tmp_path / "result.png").exists()
    assert not isinstance(fig, type(None))
    assert not  isinstance(ax, type(None))


def test_save_array_figure(tmp_path: Path):
    """Tests that the image array is saved"""
    Images(
        data=np.random.rand(10,10),
        output_dir=tmp_path,
        filename="result.png",
    ).save_array_figure()
    assert Path(tmp_path / "result.png").exists()


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_plot_and_save_no_colorbar(minicircle_pixels: Filters, plotting_config: dict, tmp_path) -> None:
    """Test plotting without colorbar"""
    plotting_config["colorbar"] = False
    fig, _ = Images(
        data=minicircle_pixels.images["pixels"],
        output_dir=tmp_path,
        filename="01-raw_heightmap.png",
        title="Raw Height",
        **plotting_config
    ).plot_and_save()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_plot_and_save_colorbar(minicircle_pixels: Filters, plotting_config: dict, tmp_path) -> None:
    """Test plotting with colorbar"""
    fig, _ = Images(
        data=minicircle_pixels.images["pixels"],
        output_dir=tmp_path,
        filename="01-raw_heightmap.png",
        title="Raw Height",
        **plotting_config
    ).plot_and_save()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_plot_and_save_axes(minicircle_pixels: Filters, plotting_config: dict, tmp_path) -> None:
    """Test plotting without axes"""
    plotting_config["axes"] = False
    fig, _ = Images(
        data=minicircle_pixels.images["pixels"],
        output_dir=tmp_path,
        filename="01-raw_heightmap.png",
        title="Raw Height", 
        **plotting_config
    ).plot_and_save()
    return fig


def test_plot_and_save_no_axes_no_colorbar(minicircle_pixels: Filters, plotting_config: dict, tmp_path) -> None:
    """Test plotting without axes and without the colourbar."""
    plotting_config["axes"] = False
    plotting_config["colorbar"] = False
    Images(
        data=minicircle_pixels.images["pixels"],
        output_dir=tmp_path,
        filename="01-raw_heightmap.png",
        title="Raw Height",
        **plotting_config
    ).plot_and_save()
    img = io.imread(tmp_path/"01-raw_heightmap.png")
    assert np.sum(img) == 448788105
    assert img.shape == (1024,1024,4)


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_plot_and_save_colorbar_afmhot(minicircle_pixels: Filters, plotting_config: dict, tmp_path) -> None:
    """Test plotting with colorbar"""
    plotting_config["cmap"] = "afmhot"
    fig, _ = Images(
        data=minicircle_pixels.images["pixels"],
        output_dir=tmp_path,
        filename="01-raw_heightmap.png",
        title="Raw Height",
        **plotting_config
    ).plot_and_save()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_plot_and_save_bounding_box(
    minicircle_grain_coloured: Grains,
    minicircle_grain_region_properties_post_removal: Grains,
    plotting_config: dict,
    tmp_path,
) -> None:
    """Test plotting bounding boxes"""
    plotting_config["type"] = "binary"
    fig, _ = Images(
        data=minicircle_grain_coloured.directions["upper"]["coloured_regions"],
        output_dir=tmp_path,
        filename="15-coloured_regions.png",
        title="Coloured Regions",
        **plotting_config,
        region_properties=minicircle_grain_region_properties_post_removal
    ).plot_and_save()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_plot_and_save_zrange(minicircle_grain_gaussian_filter: Grains, plotting_config: dict, tmp_path) -> None:
    """Tests plotting of the zrange scaled image"""
    plotting_config["zrange"] = [-10, 10]
    plotting_config["core_set"] = True
    fig, _ = Images(
        data=minicircle_grain_gaussian_filter.images["gaussian_filtered"],
        output_dir=tmp_path,
        filename="08_5-z_threshold.png",
        title="Raw Height",
        **plotting_config
    ).plot_and_save()
    return fig
