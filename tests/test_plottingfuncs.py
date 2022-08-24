"""Tests of plotting functions."""
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
    image_random: np.ndarray,
    minicircle_grain_region_properties_post_removal: Grains,
    tmp_path: Path):
    """Tests that an image is saved and a figure returned"""
    if region_properties:
        region_properties=minicircle_grain_region_properties_post_removal
    fig, ax = Images(
        data=image_random,
        output_dir=tmp_path,
        filename="result.png",
        data2=data2,
        colorbar=axes_colorbar,
        axes=axes_colorbar,
        region_properties=region_properties,
    ).save_figure()
    assert Path(tmp_path / "result.png").exists()
    assert fig is not None
    assert ax is not None


def test_save_array_figure(tmp_path: Path):
    """Tests that the image array is saved"""
    Images(
        data=np.random.rand(10,10),
        output_dir=tmp_path,
        filename="result.png",
    ).save_array_figure()
    assert Path(tmp_path / "result.png").exists()


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_plot_and_save_no_colorbar(minicircle_pixels: Filters, tmp_path: Path) -> None:
    """Test plotting without colorbar"""
    fig, _ = Images(
        data=minicircle_pixels.images["pixels"],
        output_dir=tmp_path,
        filename="01-raw_heightmap.png",
        pixel_to_nm_scaling_factor=minicircle_pixels.pixel_to_nm_scaling,
        title="Raw Height",
        colorbar=False,
        axes=True,
        image_set="all",
    ).plot_and_save()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_plot_and_save_colorbar(minicircle_pixels: Filters, tmp_path: Path) -> None:
    """Test plotting with colorbar"""
    fig, _ = Images(
        data=minicircle_pixels.images["pixels"],
        output_dir=tmp_path,
        filename="01-raw_heightmap.png",
        pixel_to_nm_scaling_factor=minicircle_pixels.pixel_to_nm_scaling,
        title="Raw Height",
        colorbar=True,
        axes=True,
        image_set="all",
    ).plot_and_save()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_plot_and_save_no_axes(minicircle_pixels: Filters, plotting_config: dict, tmp_path: Path) -> None:
    """Test plotting without axes"""
    plotting_config["axes"] = False
    print(plotting_config)
    fig, _ = Images(
        data=minicircle_pixels.images["pixels"],
        output_dir=tmp_path,
        filename="01-raw_heightmap.png",
        title="Raw Height", 
        **plotting_config
    ).plot_and_save()
    return fig


def test_plot_and_save_no_axes_no_colorbar(minicircle_pixels: Filters, plotting_config: dict, tmp_path: Path) -> None:
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
def test_plot_and_save_colorbar_afmhot(minicircle_pixels: Filters, tmp_path: Path, plotting_config: dict
    ) -> None:
    """Test plotting with colorbar"""
    plotting_config["cmap"] = "afmhot"
    fig, _ = Images(
        data=minicircle_pixels.images["pixels"],
        output_dir=tmp_path,
        filename="01-raw_heightmap.png",
        pixel_to_nm_scaling_factor=minicircle_pixels.pixel_to_nm_scaling,
        title="Raw Height",
        colorbar=True,
        axes=True,
        cmap="afmhot",
        image_set="all"
    ).plot_and_save()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_plot_and_save_bounding_box(
    minicircle_grain_coloured: Grains,
    minicircle_grain_region_properties_post_removal: Grains,
    plotting_config: dict,
    tmp_path: Path,
) -> None:
    """Test plotting bounding boxes"""
    plotting_config["type"] = "binary"
    fig, _ = Images(
        data=minicircle_grain_coloured.directions["upper"]["coloured_regions"],
        output_dir=tmp_path,
        filename="15-coloured_regions.png",
        pixel_to_nm_scaling_factor=minicircle_grain_coloured.pixel_to_nm_scaling,
        title="Coloured Regions",
        **plotting_config,
        region_properties=minicircle_grain_region_properties_post_removal,
    ).plot_and_save()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_plot_and_save_zrange(minicircle_grain_gaussian_filter: Grains, plotting_config: dict, tmp_path: Path) -> None:
    """Tests plotting of the zrange scaled image"""
    plotting_config["zrange"] = [-10, 10]
    plotting_config["core_set"] = True
    fig, _ = Images(
        data=minicircle_grain_gaussian_filter.images["gaussian_filtered"],
        output_dir=tmp_path,
        filename="08_5-z_threshold.png",
        pixel_to_nm_scaling_factor=minicircle_grain_gaussian_filter.pixel_to_nm_scaling,
        title="Raw Height",
        **plotting_config,
    ).plot_and_save()
    return fig
