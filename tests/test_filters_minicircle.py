"""Tests of the filters module."""
# + pylint: disable=invalid-name
import numpy as np

import pytest

from topostats.filters import Filters
from topostats.plottingfuncs import Images

# Specify the absolute and relattive tolerance for floating point comparison
TOLERANCE = {"atol": 1e-06, "rtol": 1e-06}


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_median_flatten_unmasked(
    minicircle_initial_align: Filters, plotting_config: dict, plot_dict: dict, tmp_path
) -> None:
    """Test initial alignment of rows without mask."""
    assert isinstance(minicircle_initial_align.images["initial_align"], np.ndarray)
    assert minicircle_initial_align.images["initial_align"].shape == (1024, 1024)
    assert minicircle_initial_align.images["initial_align"].sum() == pytest.approx(40879.633073103985)
    plotting_config = {**plotting_config, **plot_dict["initial_align"]}
    fig, _ = Images(
        data=minicircle_initial_align.images["initial_align"],
        output_dir=tmp_path,
        pixel_to_nm_scaling=minicircle_initial_align.pixel_to_nm_scaling,
        **plotting_config,
    ).plot_and_save()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_remove_tilt_unmasked(
    minicircle_initial_tilt_removal: Filters, plotting_config: dict, plot_dict: dict, tmp_path
) -> None:
    """Test removal of tilt without mask."""
    assert isinstance(minicircle_initial_tilt_removal.images["initial_tilt_removal"], np.ndarray)
    assert minicircle_initial_tilt_removal.images["initial_tilt_removal"].shape == (1024, 1024)
    assert minicircle_initial_tilt_removal.images["initial_tilt_removal"].sum() == pytest.approx(-1673257.7284464256)
    plotting_config = {**plotting_config, **plot_dict["initial_tilt_removal"]}
    fig, _ = Images(
        data=minicircle_initial_tilt_removal.images["initial_tilt_removal"],
        output_dir=tmp_path,
        pixel_to_nm_scaling=minicircle_initial_tilt_removal.pixel_to_nm_scaling,
        **plotting_config,
    ).plot_and_save()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_remove_quadratic_unmasked(
    minicircle_initial_quadratic_removal: Filters, plotting_config: dict, plot_dict: dict, tmp_path
) -> None:
    """Test removal of quadratic with mask."""
    assert isinstance(minicircle_initial_quadratic_removal.images["initial_quadratic_removal"], np.ndarray)
    assert minicircle_initial_quadratic_removal.images["initial_quadratic_removal"].shape == (1024, 1024)
    assert minicircle_initial_quadratic_removal.images["initial_quadratic_removal"].sum() == pytest.approx(
        -1672538.8578738687
    )
    plotting_config = {**plotting_config, **plot_dict["initial_quadratic_removal"]}
    fig, _ = Images(
        data=minicircle_initial_quadratic_removal.images["initial_quadratic_removal"],
        output_dir=tmp_path,
        pixel_to_nm_scaling=minicircle_initial_quadratic_removal.pixel_to_nm_scaling,
        **plotting_config,
    ).plot_and_save()
    return fig


def test_get_threshold_otsu(minicircle_threshold_otsu: np.array) -> None:
    """Test calculation of threshold."""
    assert isinstance(minicircle_threshold_otsu.thresholds, dict)
    assert minicircle_threshold_otsu.thresholds["upper"] == pytest.approx(-0.7471511148088736)


def test_get_threshold_stddev(minicircle_threshold_stddev: np.array) -> None:
    """Test calculation of threshold."""
    assert isinstance(minicircle_threshold_stddev.thresholds, dict)
    assert minicircle_threshold_stddev.thresholds == pytest.approx(
        {"lower": -8.283235478318018, "upper": -0.9269936645505799}
    )


def test_get_threshold_abs(minicircle_threshold_abs: np.array) -> None:
    """Test calculation of threshold."""
    assert isinstance(minicircle_threshold_abs.thresholds, dict)
    assert minicircle_threshold_abs.thresholds == {"upper": 1.5, "lower": -1.5}


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_get_mask(minicircle_mask: Filters, plotting_config: dict, plot_dict: dict, tmp_path) -> None:
    """Test derivation of mask."""
    plotting_config["image_type"] = "binary"
    assert isinstance(minicircle_mask.images["mask"], np.ndarray)
    assert minicircle_mask.images["mask"].shape == (1024, 1024)
    assert minicircle_mask.images["mask"].sum() == 83095
    plotting_config = {**plotting_config, **plot_dict["mask"]}
    fig, _ = Images(
        data=minicircle_mask.images["mask"],
        output_dir=tmp_path,
        pixel_to_nm_scaling=minicircle_mask.pixel_to_nm_scaling,
        **plotting_config,
    ).plot_and_save()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_median_flatten_masked(
    minicircle_masked_align: Filters, plotting_config: dict, plot_dict: dict, tmp_path
) -> None:
    """Test alignment of rows without mask."""
    assert isinstance(minicircle_masked_align.images["masked_align"], np.ndarray)
    assert minicircle_masked_align.images["masked_align"].shape == (1024, 1024)
    assert minicircle_masked_align.images["masked_align"].sum() == pytest.approx(169538.4636641923)
    plotting_config = {**plotting_config, **plot_dict["masked_align"]}
    fig, _ = Images(
        data=minicircle_masked_align.images["masked_align"],
        output_dir=tmp_path,
        pixel_to_nm_scaling=minicircle_masked_align.pixel_to_nm_scaling,
        **plotting_config,
    ).plot_and_save()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_remove_x_y_tilt_masked(
    minicircle_masked_tilt_removal: Filters, plotting_config: dict, plot_dict: dict, tmp_path
) -> None:
    """Test removal of tilt with mask."""
    assert isinstance(minicircle_masked_tilt_removal.images["masked_tilt_removal"], np.ndarray)
    assert minicircle_masked_tilt_removal.images["masked_tilt_removal"].shape == (1024, 1024)
    assert minicircle_masked_tilt_removal.images["masked_tilt_removal"].sum() == pytest.approx(163752.28336856866)
    plotting_config = {**plotting_config, **plot_dict["masked_tilt_removal"]}
    fig, _ = Images(
        data=minicircle_masked_tilt_removal.images["masked_tilt_removal"],
        output_dir=tmp_path,
        pixel_to_nm_scaling=minicircle_masked_tilt_removal.pixel_to_nm_scaling,
        **plotting_config,
    ).plot_and_save()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_remove_quadratic_masked(
    minicircle_masked_quadratic_removal: Filters, plotting_config: dict, plot_dict: dict, tmp_path
) -> None:
    """Test removal of quadratic with mask."""
    assert isinstance(minicircle_masked_quadratic_removal.images["masked_quadratic_removal"], np.ndarray)
    assert minicircle_masked_quadratic_removal.images["masked_quadratic_removal"].shape == (1024, 1024)
    assert minicircle_masked_quadratic_removal.images["masked_quadratic_removal"].sum() == pytest.approx(
        169411.34987849486
    )
    plotting_config = {**plotting_config, **plot_dict["masked_quadratic_removal"]}
    fig, _ = Images(
        data=minicircle_masked_quadratic_removal.images["masked_quadratic_removal"],
        output_dir=tmp_path,
        pixel_to_nm_scaling=minicircle_masked_quadratic_removal.pixel_to_nm_scaling,
        **plotting_config,
    ).plot_and_save()
    return fig


# @pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
# def test_average_background(
#     minicircle_zero_average_background: Filters, plotting_config: dict, plot_dict: dict, tmp_path
# ) -> None:
#     """Test zero-averaging of background."""
#     assert isinstance(minicircle_zero_average_background.images["zero_averaged_background"], np.ndarray)
#     assert minicircle_zero_average_background.images["zero_averaged_background"].shape == (1024, 1024)
#     assert minicircle_zero_average_background.images["zero_averaged_background"].sum() == 169375.4175476962
#     plotting_config = {**plotting_config, **plot_dict["zero_averaged_background"]}
#     fig, _ = Images(
#         data=minicircle_zero_average_background.images["zero_averaged_background"],
#         output_dir=tmp_path,
#         pixel_to_nm_scaling=minicircle_zero_average_background.pixel_to_nm_scaling,
#         **plotting_config,
#     ).plot_and_save()
#     return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_gaussian_filter(
    minicircle_grain_gaussian_filter: Filters, plotting_config: dict, plot_dict: dict, tmp_path
) -> None:
    """Test gaussian filter applied to background."""
    assert isinstance(minicircle_grain_gaussian_filter.images["gaussian_filtered"], np.ndarray)
    assert minicircle_grain_gaussian_filter.images["gaussian_filtered"].shape == (1024, 1024)
    assert minicircle_grain_gaussian_filter.images["gaussian_filtered"].sum() == pytest.approx(169409.44307011212)
    plotting_config = {**plotting_config, **plot_dict["gaussian_filtered"]}
    fig, _ = Images(
        data=minicircle_grain_gaussian_filter.images["gaussian_filtered"],
        output_dir=tmp_path,
        pixel_to_nm_scaling=minicircle_grain_gaussian_filter.pixel_to_nm_scaling,
        **plotting_config,
    ).plot_and_save()
    return fig
