"""Tests for the grainstats module."""
from pathlib import Path

import numpy as np

from topostats.grainstats import GrainStats

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

    np.testing.assert_array_almost_equal(cropped_grain_image, expected, decimal=4)
