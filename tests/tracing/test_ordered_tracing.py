# Disable ruff 301 - pickle loading is unsafe, but we don't care for tests
# ruff: noqa: S301
"""Test the ordered tracing module."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from topostats.tracing.ordered_tracing import ordered_tracing_image

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"

# pylint: disable=unspecified-encoding
# pylint: disable=too-many-locals


@pytest.mark.parametrize(
    (
        "image_filename",
        "disordered_tracing_direction_data_filename",
        "nodestats_data_filename",
        "nodestats_branch_images_filename",
        "filename",
        "pixel_to_nm_scaling",
    ),
    [
        pytest.param(
            "example_catenanes.npy",
            "example_catenanes_disordered_crop_data.pkl",
            "example_catenanes_nodestats_data.pkl",
            "example_catenanes_nodestats_branch_images.pkl",
            "catenane",  # filename
            1.0,  # pixel_to_nm_scaling
            id="catenane",
        ),
    ],
)
def test_ordered_tracing_image(
    image_filename: str,
    disordered_tracing_direction_data_filename: str,
    nodestats_data_filename: str,
    nodestats_branch_images_filename: str,
    filename: str,
    pixel_to_nm_scaling: float,
) -> None:
    """Test the ordered tracing image method of ordered tracing."""
    # disordered_tracing_direction_data is the disordered tracing data
    # for a particular threshold direction.

    # nodestats_direction_data contains both nodestats_data and nodestats_branch_images

    # Load the required data
    image = np.load(RESOURCES / image_filename)

    with Path.open(RESOURCES / disordered_tracing_direction_data_filename, "rb") as f:
        disordered_tracing_direction_data = pickle.load(f)

    with Path.open(RESOURCES / nodestats_data_filename, "rb") as f:
        nodestats_data = pickle.load(f)

    with Path.open(RESOURCES / nodestats_branch_images_filename, "rb") as f:
        nodestats_branch_images = pickle.load(f)

    nodestats_whole_data = {"stats": nodestats_data, "images": nodestats_branch_images}

    result_ordered_tracing_data, result_grainstats_additions_df, result_ordered_tracing_full_images = (
        ordered_tracing_image(
            image=image,
            disordered_tracing_direction_data=disordered_tracing_direction_data,
            nodestats_direction_data=nodestats_whole_data,
            filename=filename,
            pixel_to_nm_scaling=pixel_to_nm_scaling,
            ordering_method="nodestats",
            pad_width=1,
        )
    )

    # # Debugging - grab variables to show images
    # variable_ordered_traces = result_ordered_tracing_full_images["ordered_traces"]
    # variable_all_molecules = result_ordered_tracing_full_images["all_molecules"]
    # variable_over_under = result_ordered_tracing_full_images["over_under"]
    # variable_trace_segments = result_ordered_tracing_full_images["trace_segments"]

    # Save the results to update the test data
    if filename == "catenane":
        # Save result ordered tracing data as pickle
        with Path.open(RESOURCES / "example_catenanes_ordered_tracing_data.pkl", "wb") as f:
            pickle.dump(result_ordered_tracing_data, f)

        # Save result grainstats additions as csv
        result_grainstats_additions_df.to_csv(RESOURCES / "example_catenanes_ordered_tracing_grainstats_additions.csv")

        # Save result ordered tracing full images as pickle
        with Path.open(RESOURCES / "example_catenanes_ordered_tracing_full_images.pkl", "wb") as f:
            pickle.dump(result_ordered_tracing_full_images, f)

    # Load the expected results
    with Path.open(RESOURCES / "example_catenanes_ordered_tracing_data.pkl", "rb") as f:
        expected_ordered_tracing_data = pickle.load(f)

    expected_grainstats_additions_df = pd.read_csv(
        RESOURCES / "example_catenanes_ordered_tracing_grainstats_additions.csv", index_col=0
    )

    with Path.open(RESOURCES / "example_catenanes_ordered_tracing_full_images.pkl", "rb") as f:
        expected_ordered_tracing_full_images = pickle.load(f)

    # Check the results
    np.testing.assert_equal(result_ordered_tracing_data, expected_ordered_tracing_data)
    pd.testing.assert_frame_equal(result_grainstats_additions_df, expected_grainstats_additions_df)
    np.testing.assert_equal(result_ordered_tracing_full_images, expected_ordered_tracing_full_images)
