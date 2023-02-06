"""Tests for the grainstats module."""
from pathlib import Path
import imghdr

import pytest

import numpy as np

from topostats.grainstats import GrainStats

# Specify the absolute and relattive tolerance for floating point comparison
TOLERANCE = {"atol": 1e-07, "rtol": 1e-07}

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"


def test_grainstats_regression(regtest, minicircle_grainstats: GrainStats) -> None:
    """Regression tests for grainstats."""
    statistics, _ = minicircle_grainstats.calculate_stats()
    print(statistics.to_string(), file=regtest)


# @pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_cropped_image(minicircle_grainstats: GrainStats):
    """Tests that produced cropped images have not changed."""
    grain_centre = 547, 794  # centre of grain 7
    length = int(minicircle_grainstats.cropped_size / (2 * minicircle_grainstats.pixel_to_nanometre_scaling))
    cropped_grain_image = minicircle_grainstats.get_cropped_region(
        image=minicircle_grainstats.data, length=length, centre=np.asarray(grain_centre)
    )
    assert cropped_grain_image.shape == (81, 81)

    expected = np.load(RESOURCES / "test_cropped_grains.npy")

    np.testing.assert_array_equal(cropped_grain_image, expected)


@pytest.mark.parametrize("extension", [("png"), ("tiff")])
def test_save_format(minicircle_grainstats: GrainStats, tmp_path: Path, extension: str):
    "Tests if save format applied to cropped images"
    minicircle_grainstats.save_cropped_grains = True
    minicircle_grainstats.plot_opts["grain_image"]["save_format"] = extension
    minicircle_grainstats.base_output_dir = tmp_path
    minicircle_grainstats.calculate_stats()
    assert imghdr.what(tmp_path / f"upper/None_grain_image_0.{extension}") == extension
