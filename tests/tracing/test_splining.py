# Disable ruff 301 - pickle loading is unsafe, but we don't care for tests
# ruff: noqa: S301
"""Test the splining module."""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from topostats.tracing.splining import splineTrace, splining_image, windowTrace

BASE_DIR = Path.cwd()
GENERAL_RESOURCES = BASE_DIR / "tests" / "resources"
SPLINING_RESOURCES = BASE_DIR / "tests" / "resources" / "tracing" / "splining"
ORDERED_TRACING_RESOURCES = BASE_DIR / "tests" / "resources" / "tracing" / "ordered_tracing"

# pylint: disable=unspecified-encoding
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments

PIXEL_TRACE = np.array(
    [[0, 0], [0, 1], [0, 2], [0, 3], [1, 3], [2, 3], [3, 3], [3, 2], [3, 1], [3, 0], [2, 0], [1, 0]]
).astype(np.int32)


def plot_spline_debugging(
    image: npt.NDArray[np.float32],
    result_all_splines_data: dict,
    pixel_to_nm_scaling: float,
) -> None:
    """
    Plot splines of an image overlaid on the image.

    Used for debugging changes to the splining code & visually ensuring the splines are correct.

    Parameters
    ----------
    image : npt.NDArray[np.float32]
        Image to plot the splines on.
    result_all_splines_data : dict
        Dictionary containing the spline coordinates for each molecule.
    pixel_to_nm_scaling : float
        Pixel to nm scaling factor.
    """
    _, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap="gray")
    # Array of lots of matplotlib colours
    lots_of_colours = [
        "blue",
        "green",
        "red",
        "cyan",
        "magenta",
        "yellow",
        "black",
        "white",
        "orange",
        "purple",
    ]
    for grain_key_index, grain_key in enumerate(result_all_splines_data.keys()):
        print(f"Grain key: {grain_key}")
        for mol_key_index, mol_key in enumerate(result_all_splines_data[grain_key].keys()):
            spline_coords: npt.NDArray[np.float32] = result_all_splines_data[grain_key][mol_key]["spline_coords"]
            bbox = result_all_splines_data[grain_key][mol_key]["bbox"]
            bbox_min_col = bbox[0]
            bbox_min_row = bbox[1]
            previous_point = spline_coords[0]
            colour = lots_of_colours[mol_key_index + grain_key_index * 3 % len(lots_of_colours)]
            for point in spline_coords[1:]:
                ax.plot(
                    [
                        previous_point[1] / pixel_to_nm_scaling + bbox_min_row,
                        point[1] / pixel_to_nm_scaling + bbox_min_row,
                    ],
                    [
                        previous_point[0] / pixel_to_nm_scaling + bbox_min_col,
                        point[0] / pixel_to_nm_scaling + bbox_min_col,
                    ],
                    color=colour,
                    linewidth=2,
                )
                previous_point = point
    plt.show()


@pytest.mark.parametrize(
    ("tuple_list", "expected_result"),
    [
        (
            [(1, 2, 3), (1, 2, 3), (1, 2, 3), (1, 2, 3)],
            [(1, 2, 3)],
        ),
        (
            [(1, 2, 3), (1, 2, 3), (4, 5, 6), (4, 5, 6), (7, 8, 9), (10, 11, 12), (10, 11, 12)],
            [(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)],
        ),
        ([np.array((1, 2, 3)), np.array((1, 2, 3)), np.array((1, 2, 3)), np.array((1, 2, 3))], [(1, 2, 3)]),
    ],
)
def test_remove_duplicate_consecutive_tuples(tuple_list: list[tuple], expected_result: list[tuple]) -> None:
    """Test the remove_duplicate_consecutive_tuples function of splining.py."""
    result = splineTrace.remove_duplicate_consecutive_tuples(tuple_list)

    np.testing.assert_array_equal(result, expected_result)


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
        "expected_splining_grainstats_filename",
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
            "catenanes_splining_grainstats.csv",
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
            "rep_int_splining_grainstats.csv",
            "rep_int_splining_molstats.csv",
            id="replication_intermediate",
        ),
    ],
)
def test_splining_image(  # pylint: disable=too-many-positional-arguments
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
    expected_splining_grainstats_filename: str,
    expected_molstats_filename: str,
) -> None:
    """Test the splining_image function of the splining module."""
    # Load the data
    image = np.load(GENERAL_RESOURCES / image_filename)

    # Load the ordered tracing direction data
    with Path.open(ORDERED_TRACING_RESOURCES / ordered_tracing_direction_data_filename, "rb") as file:
        ordered_tracing_direction_data = pickle.load(file)

    result_all_splines_data, result_splining_grainstats, result_molstats_df = splining_image(
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

    # When updating the test, you will want to verify that the splines are correct. Use
    # plot_spline_debugging to plot the splines on the image.

    # # Save the results to update the test data
    # # Save result splining data as pickle
    # with Path.open(SPLINING_RESOURCES / expected_all_splines_data_filename, "wb") as file:
    #     pickle.dump(result_all_splines_data, file)

    # # Save result grainstats additions as csv
    # result_splining_grainstats.to_csv(SPLINING_RESOURCES / expected_splining_grainstats_filename)

    # # Save result molstats as csv
    # result_molstats_df.to_csv(SPLINING_RESOURCES / expected_molstats_filename)

    # Load the expected results
    with Path.open(SPLINING_RESOURCES / expected_all_splines_data_filename, "rb") as file:
        expected_all_splines_data = pickle.load(file)

    expected_splining_grainstats = pd.read_csv(SPLINING_RESOURCES / expected_splining_grainstats_filename, index_col=0)
    expected_molstats_df = pd.read_csv(SPLINING_RESOURCES / expected_molstats_filename, index_col=0)

    # Check the results
    np.testing.assert_equal(result_all_splines_data, expected_all_splines_data)
    pd.testing.assert_frame_equal(result_splining_grainstats, expected_splining_grainstats)
    pd.testing.assert_frame_equal(result_molstats_df, expected_molstats_df)


@pytest.mark.parametrize(
    ("pixel_trace", "rolling_window_size", "pixel_to_nm_scaling", "expected_pooled_trace"),
    [
        pytest.param(
            PIXEL_TRACE,
            np.float64(1.0),
            1.0,
            np.array(
                [
                    [0.0, 1.0],
                    [0.0, 2.0],
                    [0.0, 3.0],
                    [1.0, 3.0],
                    [2.0, 3.0],
                    [3.0, 3.0],
                    [3.0, 2.0],
                    [3.0, 1.0],
                    [3.0, 0.0],
                    [2.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 0.0],
                ]
            ).astype(np.float64),
            id="4x4 box starting at 0, 0 with window size 1",
        ),
        pytest.param(
            PIXEL_TRACE,
            np.float64(2.0),
            1.0,
            np.array(
                [
                    [0.0, 1.5],
                    [0.0, 2.5],
                    [0.5, 3.0],
                    [1.5, 3.0],
                    [2.5, 3.0],
                    [3.0, 2.5],
                    [3.0, 1.5],
                    [3.0, 0.5],
                    [2.5, 0.0],
                    [1.5, 0.0],
                    [0.5, 0.0],
                    [0.0, 0.5],
                ]
            ).astype(np.float64),
            id="4x4 box starting at 0, 0 with window size 2",
        ),
        pytest.param(
            PIXEL_TRACE,
            np.float64(5.5),
            1.0,
            np.array(
                [
                    [1.0, 2.5],
                    [1.5, 2.666666666],
                    [2.0, 2.5],
                    [2.5, 2.0],
                    [2.666666666, 1.5],
                    [2.5, 1.0],
                    [2.0, 0.5],
                    [1.5, 0.333333333],
                    [1.0, 0.5],
                    [0.5, 1.0],
                    [0.333333333, 1.5],
                    [0.5, 2.0],
                ]
            ).astype(np.float64),
            id="4x4 box starting at 0, 0 with window size 5.5",
        ),
        pytest.param(
            PIXEL_TRACE,
            np.float64(2.0),
            0.5,
            np.array(
                [
                    [0.25, 2.25],
                    [0.75, 2.75],
                    [1.5, 3.0],
                    [2.25, 2.75],
                    [2.75, 2.25],
                    [3.0, 1.5],
                    [2.75, 0.75],
                    [2.25, 0.25],
                    [1.5, 0.0],
                    [0.75, 0.25],
                    [0.25, 0.75],
                    [0.0, 1.5],
                ]
            ).astype(np.float64),
            id="4x4 box starting at 0, 0 with window size 2 and scaling 2",
        ),
    ],
)
def test_pool_trace_circular(
    pixel_trace: npt.NDArray[np.int32],
    rolling_window_size: np.float64,
    pixel_to_nm_scaling: float,
    expected_pooled_trace: npt.NDArray[np.float64],
) -> None:
    """Test of the pool_trace_circular function of the windowTrace class."""
    result_pooled_trace = windowTrace.pool_trace_circular(pixel_trace, rolling_window_size, pixel_to_nm_scaling)

    np.testing.assert_allclose(result_pooled_trace, expected_pooled_trace, atol=1e-6)


@pytest.mark.parametrize(
    ("pixel_trace", "rolling_window_size", "pixel_to_nm_scaling", "expected_pooled_trace"),
    [
        pytest.param(
            PIXEL_TRACE,
            np.float64(1.0),
            1.0,
            np.array(
                [
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 2.0],
                    [0.0, 3.0],
                    [1.0, 3.0],
                    [2.0, 3.0],
                    [3.0, 3.0],
                    [3.0, 2.0],
                    [3.0, 1.0],
                    [3.0, 0.0],
                    [2.0, 0.0],
                    [1.0, 0.0],
                ]
            ).astype(np.float64),
            id="4x4 box starting at 0, 0 with window size 1",
        ),
        pytest.param(
            PIXEL_TRACE,
            np.float64(2.0),
            1.0,
            np.array(
                [
                    [0.0, 0.0],
                    [0.0, 0.5],
                    [0.0, 1.5],
                    [0.0, 2.5],
                    [0.5, 3.0],
                    [1.5, 3.0],
                    [2.5, 3.0],
                    [3.0, 2.5],
                    [3.0, 1.5],
                    [3.0, 0.5],
                    [2.5, 0.0],
                    [1.5, 0.0],
                    [1.0, 0.0],
                ]
            ).astype(np.float64),
            id="4x4 box starting at 0, 0 with window size 2",
        ),
        pytest.param(
            PIXEL_TRACE,
            np.float64(5.5),
            1.0,
            np.array(
                [
                    [0.0, 0.0],
                    [0.5, 2.0],
                    [1.0, 2.5],
                    [1.5, 2.66666666666],
                    [2.0, 2.5],
                    [2.5, 2.0],
                    [2.666666666, 1.5],
                    [2.5, 1.0],
                    [1.0, 0.0],
                ]
            ).astype(np.float64),
            id="4x4 box starting at 0, 0 with window size 5.5",
        ),
        pytest.param(
            PIXEL_TRACE,
            np.float64(2.0),
            2.0,
            np.array(
                [
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 2.0],
                    [0.0, 3.0],
                    [1.0, 3.0],
                    [2.0, 3.0],
                    [3.0, 3.0],
                    [3.0, 2.0],
                    [3.0, 1.0],
                    [3.0, 0.0],
                    [2.0, 0.0],
                    [1.0, 0.0],
                ]
            ).astype(np.float64),
            id="4x4 box starting at 0, 0 with window size 2 and scaling 2",
        ),
    ],
)
def test_pool_trace_linear(
    pixel_trace: npt.NDArray[np.int32],
    rolling_window_size: np.float64,
    pixel_to_nm_scaling: float,
    expected_pooled_trace: npt.NDArray[np.float64],
) -> None:
    """Test of the pool_trace_circular function of the windowTrace class."""
    result_pooled_trace = windowTrace.pool_trace_linear(pixel_trace, rolling_window_size, pixel_to_nm_scaling)

    np.testing.assert_allclose(result_pooled_trace, expected_pooled_trace, atol=1e-6)
