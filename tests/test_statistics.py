"""Tests for image statistics."""

import numpy as np
import pytest

from topostats.io import dict_almost_equal
from topostats.statistics import image_statistics, roughness_rms


def test_image_statistics(image_random: np.ndarray) -> None:
    """Test the image_statistics function of statistics.py."""
    # Collect the output dataframe
    output_image_stats = image_statistics(
        image=image_random,
        filename="test_image_file",
        pixel_to_nm_scaling=0.5,
        n_grains=4,
    )
    expected = {
        "image": "test_image_file",
        "image_size_x_m": 5.12e-07,
        "image_size_y_m": 5.12e-07,
        "image_area_m2": 2.62144e-13,
        "image_size_x_px": 1024,
        "image_size_y_px": 1024,
        "image_area_px2": 1048576,
        "grains": 4,
        "grains_per_m2": 15258789062499.998,
        "rms_roughness": np.float64(5.772928703606123e-10),
    }

    assert list(output_image_stats.keys()) == list(expected.keys())
    assert dict_almost_equal(output_image_stats, expected)


@pytest.mark.parametrize(
    ("test_image", "expected"),
    [
        (
            np.array(
                [
                    [0, 0],
                    [0, 0],
                ]
            ),
            0.0,
        ),
        (
            np.array(
                [
                    [1, 1],
                    [1, 1],
                ]
            ),
            1.0,
        ),
        (
            np.array(
                [
                    [-1, -1],
                    [-1, -1],
                ]
            ),
            1.0,
        ),
        (
            np.array(
                [
                    [1, 2],
                    [3, 4],
                ]
            ),
            2.7386127875258306,
        ),
    ],
)
def test_roughness_rms(test_image, expected):
    """Test the rms (root-mean-square) roughness calculation."""
    roughness = roughness_rms(test_image)
    np.testing.assert_almost_equal(roughness, expected)
