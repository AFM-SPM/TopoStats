"""Tests of plotting functions."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from topostats.grains import Grains
from topostats.io import LoadScans
from topostats.plottingfuncs import (
    Images,
    add_pixel_to_nm_to_plotting_config,
    dilate_binary_image,
    load_mplstyle,
    plot_curvatures,
    set_n_ticks,
)

DPI = 300.0
RNG = np.random.default_rng(seed=1000)
ARRAY = RNG.random((10, 10))
MASK = RNG.uniform(low=0, high=1, size=ARRAY.shape) > 0.5


def test_add_pixel_to_nm_to_plotting_config(plotting_config_with_plot_dict):
    """Test addition of pixel to nm scaling to individual plot configurations."""
    plotting_config = add_pixel_to_nm_to_plotting_config(
        plotting_config=plotting_config_with_plot_dict, pixel_to_nm_scaling=1.23456789
    )

    # Ensure that every plot's config has the correct pixel to nm scaling factor.
    for _, plot_config in plotting_config["plot_dict"].items():
        if plot_config["pixel_to_nm_scaling"] != 1.23456789:
            raise AssertionError()


@pytest.mark.parametrize(
    ("style", "axes_titlesize", "font_size", "image_cmap", "savefig_format"),
    [
        ("topostats.mplstyle", 18, 16.0, "nanoscope", "png"),
        ("tests/resources/my_custom.mplstyle", 34, 5.0, "viridis", "svg"),
    ],
)
def test_load_mplstyle(style: str, axes_titlesize: int, font_size: float, image_cmap: str, savefig_format: str) -> None:
    """Test loading of topostats.mplstyle and a custom style."""
    load_mplstyle(style)
    assert mpl.rcParams["axes.titlesize"] == axes_titlesize
    assert mpl.rcParams["font.size"] == font_size
    assert mpl.rcParams["image.cmap"] == image_cmap
    assert mpl.rcParams["savefig.format"] == savefig_format


@pytest.mark.parametrize(
    ("binary_image", "dilation_iterations", "expected"),
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
    """Test the dilate binary images function of plottingfuncs.py."""
    result = dilate_binary_image(binary_image=binary_image, dilation_iterations=dilation_iterations)
    np.testing.assert_array_equal(result, expected)


rng = np.random.default_rng()


@pytest.mark.parametrize(
    ("masked_array", "axes_colorbar", "region_properties"),
    [(rng.random((10, 10)), True, None), (None, True, None), (None, False, True)],
)
def test_save_figure(
    masked_array: np.ndarray,
    axes_colorbar: bool,
    region_properties: bool,
    image_random: np.ndarray,
    minicircle_grain_region_properties_post_removal: Grains,
    tmp_path: Path,
):
    """Tests that an image is saved and a figure returned."""
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


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_plot_and_save_no_colorbar(load_scan_data: LoadScans, plotting_config: dict, tmp_path: Path) -> None:
    """Test plotting without colorbar."""
    plotting_config["colorbar"] = False
    fig, _ = Images(
        data=load_scan_data.image,
        output_dir=tmp_path,
        filename="01-raw_heightmap",
        pixel_to_nm_scaling=load_scan_data.pixel_to_nm_scaling,
        title="Raw Height",
        **plotting_config,
    ).plot_and_save()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_plot_histogram_and_save(load_scan_data: LoadScans, tmp_path: Path) -> None:
    """Test plotting histograms."""
    fig, _ = Images(
        load_scan_data.image, output_dir=tmp_path, filename="histogram", image_set="all"
    ).plot_histogram_and_save()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_plot_and_save_colorbar_and_axes(load_scan_data: LoadScans, plotting_config: dict, tmp_path: Path) -> None:
    """Test plotting with colorbar and axes (True in default_config.yaml)."""
    fig, _ = Images(
        data=load_scan_data.image,
        output_dir=tmp_path,
        filename="01-raw_heightmap",
        pixel_to_nm_scaling=load_scan_data.pixel_to_nm_scaling,
        title="Raw Height",
        **plotting_config,
    ).plot_and_save()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_plot_and_save_no_axes(load_scan_data: LoadScans, plotting_config: dict, tmp_path: Path) -> None:
    """Test plotting without axes."""
    plotting_config["axes"] = False
    fig, _ = Images(
        data=load_scan_data.image,
        output_dir=tmp_path,
        filename="01-raw_heightmap",
        title="Raw Height",
        **plotting_config,
    ).plot_and_save()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_plot_and_save_no_axes_no_colorbar(load_scan_data: LoadScans, plotting_config: dict, tmp_path: Path) -> None:
    """Test plotting without axes and without the colourbar."""
    plotting_config["axes"] = False
    plotting_config["colorbar"] = False
    fig, _ = Images(
        data=load_scan_data.image,
        output_dir=tmp_path,
        filename="01-raw_heightmap",
        title="Raw Height",
        **plotting_config,
    ).plot_and_save()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_plot_and_save_colorbar_afmhot(load_scan_data: LoadScans, tmp_path: Path, plotting_config: dict) -> None:
    """Test plotting with colorbar."""
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
    """Test plotting bounding boxes."""
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
    """Tests plotting of the zrange scaled image."""
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
    """Test plotting bounding boxes."""
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
    plotting_config["mask_cmap"] = "blue"
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
    plotting_config["savefig_dpi"] = DPI
    fig, _ = Images(
        data=minicircle_grain_gaussian_filter.images["gaussian_filtered"],
        output_dir=tmp_path,
        filename="high_dpi",
        **plotting_config,
    ).plot_and_save()
    return fig


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (
            [2, 2],
            (np.array([0, 8]), np.array([0, 10])),
        ),
        (
            [3, 4],
            (np.array([0, 4, 8]), np.array([0, 3, 6, 10])),
        ),
    ],
)
def test_set_n_ticks(n: list[int, int], expected: np.ndarray) -> None:
    """Test the function to set the number of x and y axis ticks."""
    arr = np.arange(0, 80).reshape(10, 8)
    _, ax = plt.subplots(1, 1)
    ax.imshow(arr)

    set_n_ticks(ax, n)

    xticks = ax.get_xticks()
    assert len(xticks) == n[0]
    assert (xticks == expected[0]).all()

    yticks = ax.get_yticks()
    assert len(yticks) == n[1]
    assert (yticks == expected[1]).all()


def test_plot_curvatures() -> None:
    """Test plotting of curvatures."""
    image = np.array(
        [
            [0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2],
            [0.2, 1.2, 1.1, 1.1, 1.1, 1.1, 0.2, 0.1],
            [0.1, 1.1, 1.1, 0.2, 0.1, 1.2, 1.1, 0.2],
            [0.2, 1.1, 0.2, 0.1, 0.2, 0.1, 1.1, 0.1],
            [0.1, 1.1, 0.1, 0.2, 0.1, 0.2, 1.1, 0.2],
            [0.2, 0.1, 1.1, 0.1, 0.2, 0.1, 1.1, 0.1],
            [0.1, 0.2, 1.1, 1.1, 1.1, 1.1, 0.1, 0.2],
            [0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1],
        ]
    )

    grain_0_curvature_stats = {
        "grain_trace_nm": np.array(
            [
                [1, 1],
                [1, 2],
                [1, 3],
                [1, 4],
                [1, 5],
                [1, 6],
                [2, 6],
                [3, 6],
                [4, 6],
                [5, 6],
                [6, 5],
                [6, 4],
                [6, 3],
                [6, 2],
                [5, 1],
                [4, 1],
                [3, 1],
                [2, 1],
            ]
        ),
        "grain_curvature": np.array(
            [
                0.1,
                0.2,
                0.1,
                0.2,
                0.1,
                0.2,
                0.1,
                0.2,
                0.1,
                0.2,
                0.1,
                0.2,
                0.1,
                0.2,
                0.1,
                0.2,
                0.1,
                0.2,
            ]
        ),
        "curvature_defects_binary_array": np.array(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
            ]
        ),
    }

    grains_curvture_stats_dict = {
        0: grain_0_curvature_stats,
    }

    plot_curvatures(
        image=image,
        grains_curvature_stats_dict=grains_curvture_stats_dict,
        pixel_to_nm_scaling=1.0,
    )
