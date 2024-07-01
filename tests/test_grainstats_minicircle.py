"""Tests for the grainstats module."""

from pathlib import Path

import numpy as np

from topostats.grainstats import GrainStats

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"


def test_grainstats_regression(regtest, minicircle_grainstats: GrainStats) -> None:
    """Regression tests for grainstats."""
    statistics, _, _ = minicircle_grainstats.calculate_stats()
    print(statistics.to_string(), file=regtest)


# @pytest.mark.mpl_image_compare(baseline_dir="resources/img/")
def test_cropped_image(minicircle_grainstats: GrainStats):
    """Tests that produced cropped images have not changed."""
    grain_centre = 39, 20  # centre of grain 7
    length = int(minicircle_grainstats.cropped_size / (2 * minicircle_grainstats.pixel_to_nanometre_scaling))
    cropped_grain_image = minicircle_grainstats.get_cropped_region(
        image=minicircle_grainstats.data, length=length, centre=np.asarray(grain_centre)
    )
    assert cropped_grain_image.shape == (21, 21)
    expected = np.load(RESOURCES / "test_cropped_grains.npy")

    np.testing.assert_array_almost_equal(cropped_grain_image, expected, decimal=4)


TARGET_HEIGHTS = [
    np.asarray(
        [
            1.17694587,
            1.85086245,
            2.09765305,
            1.88914328,
            1.52008864,
            1.28891122,
            1.38427535,
            1.6722701,
            1.9646043,
            1.98017076,
            1.72077622,
            1.51154282,
            1.49449833,
            1.57948445,
            1.6704149,
            1.75805464,
            1.64567489,
            1.41241941,
            1.07542518,
            0.83595287,
            0.8562599,
            1.16540297,
            1.6008892,
            1.90177378,
            1.6989038,
            1.02200928,
        ]
    ),
    np.asarray(
        [
            1.42323652,
            2.02262952,
            1.79344518,
            1.02490132,
            0.36515528,
            0.09889628,
            0.06873168,
            0.11120055,
            0.09410178,
            0.13544439,
            0.4786223,
            1.02653842,
            1.62401822,
            2.08260828,
            2.30009634,
            2.42498608,
            2.45261784,
            2.46582296,
            2.48040368,
            2.349282,
            2.03633221,
            1.41081186,
        ]
    ),
    np.asarray(
        [
            1.02590392,
            1.78240737,
            1.84437298,
            1.18464512,
            0.48976822,
            0.15957786,
            0.24568885,
            0.52280458,
            0.74339471,
            0.8098793,
            0.68745325,
            0.5540719,
            0.45205617,
            0.3369843,
            0.44261602,
            0.63158219,
            0.92347483,
            1.44330873,
            1.8803693,
            2.05368918,
            1.84420664,
            1.26933633,
        ]
    ),
]


def test_trace_extract_height_profile(minicircle_grainstats: GrainStats) -> None:
    """Test extraction of height profiles of minicircle.spm."""
    minicircle_grainstats.extract_height_profile = True
    _, _, height_profiles = minicircle_grainstats.calculate_stats()
    assert isinstance(height_profiles, dict)
    assert len(height_profiles) == 3
    for mol, heights in height_profiles.items():
        np.testing.assert_array_almost_equal(heights, TARGET_HEIGHTS[mol])
