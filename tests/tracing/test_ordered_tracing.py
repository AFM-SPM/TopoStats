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
GENERAL_RESOURCES = BASE_DIR / "tests" / "resources"
ORDERED_TRACING_RESOURCES = BASE_DIR / "tests" / "resources" / "tracing" / "ordered_tracing"
NODESTATS_RESOURCES = BASE_DIR / "tests" / "resources" / "tracing" / "nodestats"
DISORDERED_TRACING_RESOURCES = BASE_DIR / "tests" / "resources" / "tracing" / "disordered_tracing"

# pylint: disable=unspecified-encoding
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments


@pytest.mark.parametrize(
    (
        "image_filename",
        "disordered_tracing_direction_data_filename",
        "nodestats_data_filename",
        "nodestats_branch_images_filename",
        "filename",
        "expected_ordered_tracing_data_filename",
        "expected_ordered_tracing_grainstats_filename",
        "expected_molstats_filename",
        "expected_ordered_tracing_full_images_filename",
    ),
    [
        pytest.param(
            "example_catenanes.npy",
            "catenanes_disordered_tracing_crop_data.pkl",
            "catenanes_nodestats_data.pkl",
            "catenanes_nodestats_branch_images.pkl",
            "catenane",  # filename
            "catenanes_ordered_tracing_data.pkl",
            "catenanes_ordered_tracing_grainstats.csv",
            "catenanes_ordered_tracing_molstats.csv",
            "catenanes_ordered_tracing_full_images.pkl",
            id="catenane",
        ),
        pytest.param(
            "example_rep_int.npy",
            "rep_int_disordered_tracing_crop_data.pkl",
            "rep_int_nodestats_data_no_pair_odd_branches.pkl",
            "rep_int_nodestats_branch_images_no_pair_odd_branches.pkl",
            "replication_intermediate",  # filename
            "rep_int_ordered_tracing_data.pkl",
            "rep_int_ordered_tracing_grainstats.csv",
            "rep_int_ordered_tracing_molstats.csv",
            "rep_int_ordered_tracing_full_images.pkl",
            id="replication_intermediate",
        ),
    ],
)
def test_ordered_tracing_image(
    image_filename: str,
    disordered_tracing_direction_data_filename: str,
    nodestats_data_filename: str,
    nodestats_branch_images_filename: str,
    filename: str,
    expected_ordered_tracing_data_filename: str,
    expected_ordered_tracing_grainstats_filename: str,
    expected_molstats_filename: str,
    expected_ordered_tracing_full_images_filename: str,
) -> None:
    """Test the ordered tracing image method of ordered tracing."""
    # disordered_tracing_direction_data is the disordered tracing data
    # for a particular threshold direction.

    # nodestats_direction_data contains both nodestats_data and nodestats_branch_images

    # Load the required data
    image = np.load(GENERAL_RESOURCES / image_filename)

    with Path.open(DISORDERED_TRACING_RESOURCES / disordered_tracing_direction_data_filename, "rb") as f:
        disordered_tracing_direction_data = pickle.load(f)

    with Path.open(NODESTATS_RESOURCES / nodestats_data_filename, "rb") as f:
        nodestats_data = pickle.load(f)

    with Path.open(NODESTATS_RESOURCES / nodestats_branch_images_filename, "rb") as f:
        nodestats_branch_images = pickle.load(f)

    nodestats_whole_data = {"stats": nodestats_data, "images": nodestats_branch_images}

    (
        result_ordered_tracing_data,
        result_ordered_tracing_grainstats,
        result_molstats_df,
        result_ordered_tracing_full_images,
    ) = ordered_tracing_image(
        image=image,
        disordered_tracing_direction_data=disordered_tracing_direction_data,
        nodestats_direction_data=nodestats_whole_data,
        filename=filename,
        ordering_method="nodestats",
        pad_width=1,
    )

    # # Debugging - grab variables to show images
    # variable_ordered_traces = result_ordered_tracing_full_images["ordered_traces"]
    # variable_all_molecules = result_ordered_tracing_full_images["all_molecules"]
    # variable_over_under = result_ordered_tracing_full_images["over_under"]
    # variable_trace_segments = result_ordered_tracing_full_images["trace_segments"]

    # # Save result ordered tracing data as pickle
    # with Path.open(ORDERED_TRACING_RESOURCES / expected_ordered_tracing_data_filename, "wb") as f:
    #     pickle.dump(result_ordered_tracing_data, f)

    # # Save result grainstats additions as csv
    # result_ordered_tracing_grainstats.to_csv(ORDERED_TRACING_RESOURCES / expected_ordered_tracing_grainstats_filename)

    # # Save the molstats dataframe as csv
    # result_molstats_df.to_csv(ORDERED_TRACING_RESOURCES / expected_molstats_filename)

    # # Save result ordered tracing full images as pickle
    # with Path.open(ORDERED_TRACING_RESOURCES / expected_ordered_tracing_full_images_filename, "wb") as f:
    #     pickle.dump(result_ordered_tracing_full_images, f)

    # Load the expected results
    with Path.open(ORDERED_TRACING_RESOURCES / expected_ordered_tracing_data_filename, "rb") as f:
        expected_ordered_tracing_data = pickle.load(f)

    expected_ordered_tracing_grainstats = pd.read_csv(
        ORDERED_TRACING_RESOURCES / expected_ordered_tracing_grainstats_filename, index_col=0
    )

    expected_molstats_df = pd.read_csv(ORDERED_TRACING_RESOURCES / expected_molstats_filename, index_col=0)

    with Path.open(ORDERED_TRACING_RESOURCES / expected_ordered_tracing_full_images_filename, "rb") as f:
        expected_ordered_tracing_full_images = pickle.load(f)

    # Check the results
    np.testing.assert_equal(result_ordered_tracing_data, expected_ordered_tracing_data)
    pd.testing.assert_frame_equal(result_ordered_tracing_grainstats, expected_ordered_tracing_grainstats)
    pd.testing.assert_frame_equal(result_molstats_df, expected_molstats_df)
    np.testing.assert_equal(result_ordered_tracing_full_images, expected_ordered_tracing_full_images)