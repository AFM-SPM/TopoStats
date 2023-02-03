"""Tests for the grainstats module."""
from pathlib import Path
import imghdr

import pytest
import numpy as np

from topostats.grainstats import GrainStats


def test_grainstats_regression(regtest, minicircle_grainstats: GrainStats) -> None:
    """Regression tests for grainstats."""
    statistics = minicircle_grainstats.calculate_stats()
    print(statistics["statistics"].to_string(), file=regtest)


@pytest.mark.parametrize(
    "value",
    [
        (True),
        (False),
    ],
)
def test_save_cropped_grains(minicircle_grainstats: GrainStats, tmp_path: Path, value: bool) -> None:
    """Tests if save_cropped_grains option only creates the grains dir when True"""
    minicircle_grainstats.save_cropped_grains = value
    minicircle_grainstats.base_output_dir = tmp_path / "grains"
    minicircle_grainstats.calculate_stats()
    assert Path.exists(tmp_path / "grains") == value


@pytest.mark.parametrize(
    "value, expected",
    [
        ("core", False),
        ("all", True),
    ],
)
def test_image_set(minicircle_grainstats: GrainStats, tmp_path: Path, value: str, expected: bool) -> None:
    """Tests for the correct outputs when image_set is varied"""
    minicircle_grainstats.save_cropped_grains = True
    minicircle_grainstats.plot_opts["grain_image"]["image_set"] = value
    minicircle_grainstats.plot_opts["grain_mask"]["image_set"] = value
    minicircle_grainstats.plot_opts["grain_mask_image"]["image_set"] = value
    minicircle_grainstats.base_output_dir = tmp_path / "grains"
    minicircle_grainstats.calculate_stats()
    assert Path.exists(tmp_path / "grains/upper" / "None_grain_image_0.png") is True
    assert Path.exists(tmp_path / "grains/upper" / "None_grain_mask_0.png") == expected
    assert Path.exists(tmp_path / "grains/upper" / "None_grain_mask_image_0.png") == expected


def test_cropped_image(minicircle_grainstats: GrainStats) -> None:
    """Tests that produced cropped images have not changed."""
    grain_centre = 547, 794  # centre of grain 7
    length = int(minicircle_grainstats.cropped_size / (2 * minicircle_grainstats.pixel_to_nanometre_scaling))
    cropped_grain_image = minicircle_grainstats.get_cropped_region(
        image=minicircle_grainstats.data, length=length, centre=np.asarray(grain_centre)
    )
    assert cropped_grain_image.shape == (81, 81)


@pytest.mark.parametrize("extension", [("png"), ("tiff")])
def test_save_format(minicircle_grainstats: GrainStats, tmp_path: Path, extension: str) -> None:
    "Tests if save format applied to cropped images"
    minicircle_grainstats.save_cropped_grains = True
    minicircle_grainstats.plot_opts["grain_image"]["save_format"] = extension
    minicircle_grainstats.base_output_dir = tmp_path
    minicircle_grainstats.calculate_stats()
    assert imghdr.what(tmp_path / f"upper/None_grain_image_0.{extension}") == extension
