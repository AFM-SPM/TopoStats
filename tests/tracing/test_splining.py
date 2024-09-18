# Disable ruff 301 - pickle loading is unsafe, but we don't care for tests
# ruff: noqa: S301
"""Test the splining module."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from topostats.tracing.splining import splining_image

BASE_DIR = Path.cwd()
GENERAL_RESOURCES = BASE_DIR / "tests" / "resources"
SPLINING_RESOURCES = BASE_DIR / "tests" / "resources" / "tracing" / "splining"
ORDERED_TRACING_RESOURCES = BASE_DIR / "tests" / "resources" / "tracing" / "ordered_tracing"

# pylint: disable=unspecified-encoding
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments


@pytest.mark.parametrize(
    (
        "image_filename",
        "ordered_tracing_direction_data_filename",
        "pixel_to_nm_scaling",
        "splining_method",
        "rolling_window_size",
        "spline_step_size",
        "spline_linear_smoothing",
        "spline_circular_smoothing",
        "spline_degree",
        "filename",
        "expected_all_splines_data_filename",
        "expected_grainstats_additions_filename",
        "expected_molstats_filename",
    ),
    [
        pytest.param(
            "example_catenanes.npy",
            "catenanes_ordered_tracing_data.pkl",
            1.0,  # pixel_to_nm_scaling
            # Splining parameters
            "rolling_window",  # splining_method
            20e-9,  # rolling_window_size
            7.0e-9,  # spline_step_size
            5.0,  # spline_linear_smoothing
            5.0,  # spline_circular_smoothing
            3,  # spline_degree
            "catenane",  # filename
            "catenanes_splining_data.pkl",
            "catenanes_splining_grainstats_additions.csv",
            "catenanes_splining_molstats.csv",
            id="catenane",
        ),
        pytest.param(
            "example_rep_int.npy",
            "rep_int_ordered_tracing_data.pkl",
            1.0,  # pixel_to_nm_scaling
            # Splining parameters
            "rolling_window",  # splining_method
            20e-9,  # rolling_window_size
            7.0e-9,  # spline_step_size
            5.0,  # spline_linear_smoothing
            5.0,  # spline_circular_smoothing
            3,  # spline_degree
            "replication_intermediate",  # filename
            "rep_int_splining_data.pkl",
            "rep_int_splining_grainstats_additions.csv",
            "rep_int_splining_molstats.csv",
            id="replication_intermediate",
        ),
    ],
)
def test_splining_image(
    image_filename: str,
    ordered_tracing_direction_data_filename: str,
    pixel_to_nm_scaling: float,
    splining_method: str,
    rolling_window_size: float,
    spline_step_size: float,
    spline_linear_smoothing: float,
    spline_circular_smoothing: float,
    spline_degree: int,
    filename: str,
    expected_all_splines_data_filename: str,
    expected_grainstats_additions_filename: str,
    expected_molstats_filename: str,
) -> None:
    """Test the splining_image function of the splining module."""
    # Load the data
    image = np.load(GENERAL_RESOURCES / image_filename)

    # Load the ordered tracing direction data
    with Path.open(ORDERED_TRACING_RESOURCES / ordered_tracing_direction_data_filename, "rb") as file:
        ordered_tracing_direction_data = pickle.load(file)

    result_all_splines_data, result_grainstats_additions_df, result_molstats_df = splining_image(
        image=image,
        ordered_tracing_direction_data=ordered_tracing_direction_data,
        pixel_to_nm_scaling=pixel_to_nm_scaling,
        filename=filename,
        method=splining_method,
        rolling_window_size=rolling_window_size,
        spline_step_size=spline_step_size,
        spline_linear_smoothing=spline_linear_smoothing,
        spline_circular_smoothing=spline_circular_smoothing,
        spline_degree=spline_degree,
    )

    # Debugging
    # Spline coords is Nx2 array of spline coordinates
    # Visualise the spline coordinates
    # import matplotlib.pyplot as plt
    # import numpy.typing as npt

    # fig, ax = plt.subplots(figsize=(10, 10))
    # ax.imshow(image, cmap="gray")
    # # Array of lots of matplotlib colours
    # lots_of_colours = [
    #     "blue",
    #     "green",
    #     "red",
    #     "cyan",
    #     "magenta",
    #     "yellow",
    #     "black",
    #     "white",
    #     "orange",
    #     "purple",
    # ]
    # for grain_key_index, grain_key in enumerate(result_all_splines_data.keys()):
    #     print(f"Grain key: {grain_key}")
    #     for mol_key_index, mol_key in enumerate(result_all_splines_data[grain_key].keys()):
    #         spline_coords: npt.NDArray[np.float32] = result_all_splines_data[grain_key][mol_key]["spline_coords"]
    #         bbox = result_all_splines_data[grain_key][mol_key]["bbox"]
    #         bbox_min_col = bbox[0]
    #         bbox_min_row = bbox[1]
    #         previous_point = spline_coords[0]
    #         colour = lots_of_colours[mol_key_index + grain_key_index * 3 % len(lots_of_colours)]
    #         for point in spline_coords[1:]:
    #             ax.plot(
    #                 [
    #                     previous_point[1] / pixel_to_nm_scaling + bbox_min_row,
    #                     point[1] / pixel_to_nm_scaling + bbox_min_row,
    #                 ],
    #                 [
    #                     previous_point[0] / pixel_to_nm_scaling + bbox_min_col,
    #                     point[0] / pixel_to_nm_scaling + bbox_min_col,
    #                 ],
    #                 color=colour,
    #                 linewidth=2,
    #             )
    #             previous_point = point
    # plt.show()

    # # Save the results to update the test data
    # # Save result splining data as pickle
    # with Path.open(SPLINING_RESOURCES / expected_all_splines_data_filename, "wb") as file:
    #     pickle.dump(result_all_splines_data, file)

    # # Save result grainstats additions as csv
    # result_grainstats_additions_df.to_csv(SPLINING_RESOURCES / expected_grainstats_additions_filename)

    # # Save result molstats as csv
    # result_molstats_df.to_csv(SPLINING_RESOURCES / expected_molstats_filename)

    # Load the expected results
    with Path.open(SPLINING_RESOURCES / expected_all_splines_data_filename, "rb") as file:
        expected_all_splines_data = pickle.load(file)

    expected_grainstats_additions_df = pd.read_csv(
        SPLINING_RESOURCES / expected_grainstats_additions_filename, index_col=0
    )
    expected_molstats_df = pd.read_csv(SPLINING_RESOURCES / expected_molstats_filename, index_col=0)

    # Check the results
    np.testing.assert_equal(result_all_splines_data, expected_all_splines_data)
    pd.testing.assert_frame_equal(result_grainstats_additions_df, expected_grainstats_additions_df)
    pd.testing.assert_frame_equal(result_molstats_df, expected_molstats_df)
