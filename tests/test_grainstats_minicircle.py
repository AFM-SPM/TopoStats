"""Tests for the grainstats module."""
from pathlib import Path
import imghdr

import pytest

from topostats.grainstats import GrainStats

# Specify the absolute and relattive tolerance for floating point comparison
TOLERANCE = {"atol": 1e-07, "rtol": 1e-07}

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"


def test_grainstats_regression(regtest, minicircle_grainstats: GrainStats) -> None:
    """Regression tests for grainstats."""
    statistics, _ = minicircle_grainstats.calculate_stats()
    print(statistics.to_string(), file=regtest)


# @pytest.mark.parametrize(
#     "value, expected",
#     [
#         ("core", False),
#         ("all", True),
#     ],
# )
# def test_image_set(minicircle_grainstats: GrainStats, tmp_path: Path, value, expected):
#     """Tests for the correct outputs when image_set is varied"""
#     minicircle_grainstats.save_cropped_grains = True
#     minicircle_grainstats.plot_opts["grain_image"]["image_set"] = value
#     minicircle_grainstats.plot_opts["grain_mask"]["image_set"] = value
#     minicircle_grainstats.plot_opts["grain_mask_image"]["image_set"] = value
#     minicircle_grainstats.base_output_dir = tmp_path / "grains"
#     minicircle_grainstats.calculate_stats()
#     assert Path.exists(tmp_path / "grains/upper" / "None_grain_image_0.png") is True
#     assert Path.exists(tmp_path / "grains/upper" / "None_grain_mask_0.png") == expected
#     assert Path.exists(tmp_path / "grains/upper" / "None_grain_mask_image_0.png") == expected


# @pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
# def test_cropped_image(minicircle_grainstats: GrainStats, tmp_path: Path):
#     """Tests that produced cropped images have not changed."""
#     grain_centre = 547, 794  # centre of grain 7
#     length = int(minicircle_grainstats.cropped_size / (2 * minicircle_grainstats.pixel_to_nanometre_scaling))
#     cropped_grain_image = minicircle_grainstats.get_cropped_region(
#         image=minicircle_grainstats.data, length=length, centre=np.asarray(grain_centre)
#     )
#     assert cropped_grain_image.shape == (81, 81)
#     fig, _ = Images(
#         data=cropped_grain_image,
#         output_dir=tmp_path,
#         filename="cropped_grain_7.png",
#         pixel_to_nm_scaling=minicircle_grainstats.pixel_to_nanometre_scaling,
#         image_type="non-binary",
#         image_set="all",
#         core_set=True,
#     ).plot_and_save()
#     return fig


@pytest.mark.parametrize("extension", [("png"), ("tiff")])
def test_save_format(minicircle_grainstats: GrainStats, tmp_path: Path, extension: str):
    "Tests if save format applied to cropped images"
    minicircle_grainstats.save_cropped_grains = True
    minicircle_grainstats.plot_opts["grain_image"]["save_format"] = extension
    minicircle_grainstats.base_output_dir = tmp_path
    minicircle_grainstats.calculate_stats()
    assert imghdr.what(tmp_path / f"upper/None_grain_image_0.{extension}") == extension
