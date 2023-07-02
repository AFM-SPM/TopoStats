"""Tests for image statistics"""

import pytest
import numpy as np
import pandas as pd

from topostats.statistics import image_statistics, roughness_rms


def test_image_statistics(image_random: np.ndarray) -> None:
    """Test the image_statistics function of statistics.py."""

    # Construct the results dataframe that image_stats takes as an argument
    results_data = {
        "molecule_number": [0, 1, 2, 3, 4, 5],
        "threshold": ["above", "above", "below", "above", "below", "above"],
    }
    results_df = pd.DataFrame(results_data)

    # Collect the output dataframe
    output_image_stats = image_statistics(
        image=image_random,
        filename="test_image_file",
        results_df=results_df,
        pixel_to_nm_scaling=0.5,
    )

    # Construct the expected dataframe
    expected_columns = [
        "Image",
        "image_size_x_m",
        "image_size_y_m",
        "image_area_m2",
        "image_size_x_px",
        "image_size_y_px",
        "image_area_px2",
        "grains_number_above",
        "grains_per_m2_above",
        "grains_number_below",
        "grains_per_m2_below",
        "rms_roughness",
    ]
    expected_data = [
        [
            "test_image_file",
            5.12e-07,
            5.12e-07,
            2.62144e-13,
            1024,
            1024,
            1048576,
            4,
            15258789062499.998,
            2,
            7629394531249.999,
            5.772928703606123e-10,
        ]
    ]
    expected_df = pd.DataFrame(expected_data, columns=expected_columns)
    expected_df.set_index("Image", inplace=True)

    # This was the only way I could find to do it, as pandas' assert_frame_equal will
    # take 5e-10 and 4e-10 to be equal.
    expected_values = np.array(expected_df.values[0])
    output_values = np.array(output_image_stats.values[0])
    np.testing.assert_array_equal(output_values, expected_values)

    pd.testing.assert_frame_equal(output_image_stats, expected_df)


@pytest.mark.parametrize(
    "test_image, expected",
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
