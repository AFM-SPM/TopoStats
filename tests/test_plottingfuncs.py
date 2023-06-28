"""Tests of plotting functions."""
from pathlib import Path

import pytest
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
from skimage import io

from topostats.grains import Grains
from topostats.io import LoadScans
from topostats.plottingfuncs import dilate_binary_image, Images


DPI = 300.0
RNG = np.random.default_rng(seed=1000)
ARRAY = RNG.random((10, 10))
MASK = RNG.uniform(low=0, high=1, size=ARRAY.shape) > 0.5


@pytest.mark.parametrize(
    "binary_image, dilation_iterations, expected",
    [
        (
            np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]),
            1,
            np.array([[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]]),
        ),
        (
            np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]),
            2,
            np.array([[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]]),
        ),
    ],
)
def test_dilate_binary_image(binary_image: np.ndarray, dilation_iterations: int, expected: np.ndarray) -> None:
    "Test the dilate binary images function of plottingfuncs.py."

    result = dilate_binary_image(binary_image=binary_image, dilation_iterations=dilation_iterations)

    print(result)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "masked_array, axes_colorbar, region_properties",
    [(np.random.rand(10, 10), True, None), (None, True, None), (None, False, True)],
)
def test_save_figure(
    masked_array: np.ndarray,
    axes_colorbar: bool,
    region_properties: bool,
    image_random: np.ndarray,
    minicircle_grain_region_properties_post_removal: Grains,
    tmp_path: Path,
):
    """Tests that an image is saved and a figure returned"""
    if region_properties:
        region_properties = minicircle_grain_region_properties_post_removal
    fig, ax = Images(
        data=image_random,
        output_dir=tmp_path,
        filename="result",
        masked_array=masked_array,
        colorbar=axes_colorbar,
        axes=axes_colorbar,
        region_properties=region_properties,
    ).save_figure()
    assert Path(tmp_path / "result.png").exists()
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def test_save_array_figure(tmp_path: Path):
    """Tests that the image array is saved"""
    Images(
        data=np.random.rand(10, 10),
        output_dir=tmp_path,
        filename="result",
    ).save_array_figure()
    assert Path(tmp_path / "result.png").exists()


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_plot_and_save_no_colorbar(load_scan_data: LoadScans, tmp_path: Path) -> None:
    """Test plotting without colorbar"""
    fig, _ = Images(
        data=load_scan_data.image,
        output_dir=tmp_path,
        filename="01-raw_heightmap",
        pixel_to_nm_scaling=load_scan_data.pixel_to_nm_scaling,
        title="Raw Height",
        colorbar=False,
        axes=True,
        image_set="all",
    ).plot_and_save()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_plot_histogram_and_save(load_scan_data: LoadScans, tmp_path: Path) -> None:
    """Test plotting histograms"""
    fig, _ = Images(
        load_scan_data.image, output_dir=tmp_path, filename="histogram", image_set="all"
    ).plot_histogram_and_save()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_plot_and_save_colorbar(load_scan_data: LoadScans, tmp_path: Path) -> None:
    """Test plotting with colorbar"""
    fig, _ = Images(
        data=load_scan_data.image,
        output_dir=tmp_path,
        filename="01-raw_heightmap",
        pixel_to_nm_scaling=load_scan_data.pixel_to_nm_scaling,
        title="Raw Height",
        colorbar=True,
        axes=True,
        image_set="all",
    ).plot_and_save()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_plot_and_save_no_axes(load_scan_data: LoadScans, plotting_config: dict, tmp_path: Path) -> None:
    """Test plotting without axes"""
    plotting_config["axes"] = False
    fig, _ = Images(
        data=load_scan_data.image,
        output_dir=tmp_path,
        filename="01-raw_heightmap",
        title="Raw Height",
        **plotting_config,
    ).plot_and_save()
    return fig


def test_plot_and_save_no_axes_no_colorbar(load_scan_data: LoadScans, plotting_config: dict, tmp_path: Path) -> None:
    """Test plotting without axes and without the colourbar."""
    plotting_config["axes"] = False
    plotting_config["colorbar"] = False
    Images(
        data=load_scan_data.image,
        output_dir=tmp_path,
        filename="01-raw_heightmap",
        title="Raw Height",
        **plotting_config,
    ).plot_and_save()
    img = io.imread(tmp_path / "01-raw_heightmap.png")
    assert np.sum(img) == 461143075
    assert img.shape == (1024, 1024, 4)


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_plot_and_save_colorbar_afmhot(load_scan_data: LoadScans, tmp_path: Path, plotting_config: dict) -> None:
    """Test plotting with colorbar"""
    plotting_config["cmap"] = "afmhot"
    fig, _ = Images(
        data=load_scan_data.image,
        output_dir=tmp_path,
        filename="01-raw_heightmap",
        pixel_to_nm_scaling=load_scan_data.pixel_to_nm_scaling,
        title="Raw Height",
        colorbar=True,
        axes=True,
        cmap="afmhot",
        image_set="all",
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
    plotting_config["image_type"] = "binary"
    fig, _ = Images(
        data=minicircle_grain_coloured.directions["above"]["coloured_regions"],
        output_dir=tmp_path,
        filename="15-coloured_regions",
        pixel_to_nm_scaling=minicircle_grain_coloured.pixel_to_nm_scaling,
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
        filename="08_5-z_threshold",
        pixel_to_nm_scaling=minicircle_grain_gaussian_filter.pixel_to_nm_scaling,
        title="Raw Height",
        **plotting_config,
    ).plot_and_save()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_plot_and_save_non_square_bounding_box(
    minicircle_grain_coloured: Grains,
    minicircle_grain_region_properties_post_removal: Grains,
    plotting_config: dict,
    tmp_path: Path,
) -> None:
    """Test plotting bounding boxes"""
    plotting_config["image_type"] = "binary"
    fig, _ = Images(
        data=minicircle_grain_coloured.image[:, 0:512],
        output_dir=tmp_path,
        filename="15-coloured_regions.png",
        pixel_to_nm_scaling=minicircle_grain_coloured.pixel_to_nm_scaling,
        title="Coloured Regions",
        **plotting_config,
        region_properties=minicircle_grain_region_properties_post_removal,
    ).plot_and_save()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_mask_cmap(plotting_config: dict, tmp_path: Path) -> None:
    """Test the plotting of a mask with a different colourmap (blu)."""
    plotting_config["mask_cmap"] = "blu"
    fig, _ = Images(
        data=ARRAY,
        output_dir=tmp_path,
        filename="colour.png",
        masked_array=MASK,
        **plotting_config,
    ).plot_and_save()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/", savefig_kwargs={"dpi": DPI})
def test_high_dpi(minicircle_grain_gaussian_filter: Grains, plotting_config: dict, tmp_path: Path) -> None:
    """Test plotting with high DPI."""
    plotting_config["dpi"] = DPI
    fig, _ = Images(
        data=minicircle_grain_gaussian_filter.images["gaussian_filtered"],
        output_dir=tmp_path,
        filename="high_dpi",
        **plotting_config,
    ).plot_and_save()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_mask_dilation(plotting_config: dict, tmp_path: Path) -> None:
    """Test the plotting of a mask with a different colourmap (blu)."""
    plotting_config["mask_cmap"] = "blu"
    mask = np.zeros((1024, 1024))
    mask[500, :] = 1
    fig, _ = Images(
        data=RNG.random((1024, 1024)),
        output_dir=tmp_path,
        filename="mask_dilation",
        masked_array=mask,
        **plotting_config,
    ).plot_and_save()
    return fig
