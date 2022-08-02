"""Tests for the grainstats module."""
from pathlib import Path

import pandas as pd
import pytest
from topostats.grainstats import GrainStats

# Specify the absolute and relattive tolerance for floating point comparison
TOLERANCE = {"atol": 1e-07, "rtol": 1e-07}

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"


def test_grainstats(minicircle_grainstats: GrainStats, minicircle_grainstats_20220526: pd.DataFrame) -> None:
    """Test the overall GrainStats class."""
    statistics = minicircle_grainstats.calculate_stats()

    pd.testing.assert_frame_equal(statistics["statistics"], minicircle_grainstats_20220526)


@pytest.mark.parametrize("value", [
(True),
(False),
])
def test_save_cropped_grains(minicircle_grainstats: GrainStats, tmpdir, value):
    # need to run grainstats with config option True and see if it is there.
    minicircle_grainstats.save_cropped_grains = value
    minicircle_grainstats.base_output_dir = Path(tmpdir) / "grains"
    minicircle_grainstats.calculate_stats()
    assert Path.exists(Path(tmpdir) / "grains") == value

@pytest.mark.parametrize("value, expected", [
("core", False),
("all", True),
])
def test_image_set(minicircle_grainstats: GrainStats, tmpdir, value, expected):
    # need to run grainstats with config option True and see if it is there.
    minicircle_grainstats.save_cropped_grains = True
    minicircle_grainstats.image_set = value
    minicircle_grainstats.base_output_dir = Path(tmpdir) / "grains"
    minicircle_grainstats.calculate_stats()
    assert Path.exists(Path(tmpdir) / "grains/minicircle" / "None_processed_grain_0.png") == True
    assert Path.exists(Path(tmpdir) / "grains/minicircle" / "None_grain_image_0.png") == expected
    assert Path.exists(Path(tmpdir) / "grains/minicircle" / "None_grainmask_0.png") == expected
    assert Path.exists(Path(tmpdir) / "grains/minicircle" / "None_grain_image_0.png") == expected
    assert Path.exists(Path(tmpdir) / "grains/minicircle" / "None_grain_image_0.png") == expected