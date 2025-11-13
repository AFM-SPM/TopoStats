# Disable ruff 301 - pickle loading is unsafe, but we don't care for tests
# ruff: noqa: S301
"""Test the splining module."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pytest

from topostats.classes import Molecule
from topostats.tracing.splining import (
    interpolate_between_two_points_distance,
    resample_points_regular_interval,
    splineTrace,
    splining_image,
    windowTrace,
)

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"
TRACING_RESOURCES = RESOURCES / "tracing"
SPLINING_RESOURCES = TRACING_RESOURCES / "splining"
ORDERED_TRACING_RESOURCES = TRACING_RESOURCES / "ordered_tracing"

# pylint: disable=unspecified-encoding
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments

PIXEL_TRACE = np.array(
    [[0, 0], [0, 1], [0, 2], [0, 3], [1, 3], [2, 3], [3, 3], [3, 2], [3, 1], [3, 0], [2, 0], [1, 0]]
).astype(np.int32)
MOLECULE = Molecule(
    ordered_coords=np.array(
        [[0, 0], [0, 1], [0, 2], [0, 3], [1, 3], [2, 3], [3, 3], [3, 2], [3, 1], [3, 0], [2, 0], [1, 0]]
    ).astype(np.int32)
)


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
            splined_coords: npt.NDArray[np.float32] = result_all_splines_data[grain_key][mol_key]["splined_coords"]
            bbox = result_all_splines_data[grain_key][mol_key]["bbox"]
            bbox_min_col = bbox[0]
            bbox_min_row = bbox[1]
            previous_point = splined_coords[0]
            colour = lots_of_colours[mol_key_index + grain_key_index * 3 % len(lots_of_colours)]
            for point in splined_coords[1:]:
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
        "splining_fixture",
        "splining_method",
        "rolling_window_size",
        "spline_step_size",
        "spline_linear_smoothing",
        "spline_circular_smoothing",
        "spline_degree",
    ),
    [
        pytest.param(
            "catenane_splining",
            "rolling_window",  # splining_method
            20e-9,  # rolling_window_size
            7.0e-9,  # spline_step_size
            5.0,  # spline_linear_smoothing
            5.0,  # spline_circular_smoothing
            3,  # spline_degree
            id="catenane",
        ),
        pytest.param(
            "rep_int_splining",
            "rolling_window",  # splining_method
            20e-9,  # rolling_window_size
            7.0e-9,  # spline_step_size
            5.0,  # spline_linear_smoothing
            5.0,  # spline_circular_smoothing
            3,  # spline_degree
            id="replication_intermediate",
        ),
    ],
)
def test_splining_image(  # pylint: disable=too-many-positional-arguments
    splining_fixture: str,
    splining_method: str,
    rolling_window_size: float,
    spline_step_size: float,
    spline_linear_smoothing: float,
    spline_circular_smoothing: float,
    spline_degree: int,
    request,
    snapshot,
) -> None:
    """Test the splining_image function of the splining module."""
    topostats_object = request.getfixturevalue(splining_fixture)

    _, result_splining_grainstats, result_molstats_df = splining_image(
        topostats_object=topostats_object,
        method=splining_method,
        rolling_window_size=rolling_window_size,
        spline_step_size=spline_step_size,
        spline_linear_smoothing=spline_linear_smoothing,
        spline_circular_smoothing=spline_circular_smoothing,
        spline_degree=spline_degree,
    )

    # When updating the test, you will want to verify that the splines are correct. Use
    # plot_spline_debugging to plot the splines on the image.

    # Check the results, no easy way to iterate over these as loops would over-write the snapshot
    assert topostats_object.grain_crops[0].ordered_trace.molecule_data[0] == snapshot
    assert topostats_object.grain_crops[0].ordered_trace.molecule_data[1] == snapshot
    if topostats_object.filename == "catenane":
        assert topostats_object.grain_crops[1].ordered_trace.molecule_data[0] == snapshot
        assert topostats_object.grain_crops[1].ordered_trace.molecule_data[1] == snapshot
    elif topostats_object.filename == "replication_intermediate":
        assert topostats_object.grain_crops[0].ordered_trace.molecule_data[2] == snapshot
    assert result_splining_grainstats.to_string(float_format="%.6e") == snapshot
    assert result_molstats_df.to_string(float_format="%.6e") == snapshot


@pytest.mark.parametrize(
    ("pixel_trace", "rolling_window_size", "pixel_to_nm_scaling", "expected_pooled_trace"),
    [
        pytest.param(
            MOLECULE,
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
            MOLECULE,
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
            MOLECULE,
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
            MOLECULE,
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
            MOLECULE,
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
            MOLECULE,
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
            MOLECULE,
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
            MOLECULE,
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


@pytest.mark.parametrize(
    ("point1", "point2", "distance", "expected_point"),
    [
        pytest.param(
            np.array([0.0, 0.0]),
            np.array([10.0, 0.0]),
            5.0,
            np.array([5.0, 0.0]),
            id="horizontal_interpolation",
        ),
        pytest.param(
            np.array([0.0, 0.0]),
            np.array([0.0, 10.0]),
            5.0,
            np.array([0.0, 5.0]),
            id="vertical_interpolation",
        ),
        pytest.param(
            np.array([0.0, 0.0]),
            np.array([10.0, 10.0]),
            5.0,
            np.array([3.535534, 3.535534]),
            id="diagonal_interpolation",
        ),
        pytest.param(
            np.array([-5.0, 0.0]),
            np.array([5.0, 0.0]),
            5.0,
            np.array([0.0, 0.0]),
            id="negative_interpolation",
        ),
    ],
)
def test_interpolate_between_two_points_distance(
    point1: npt.NDArray[np.float32],
    point2: npt.NDArray[np.float32],
    distance: np.float32,
    expected_point: npt.NDArray[np.float32],
) -> None:
    """Test the interpolation between two points."""
    interpolated_point = interpolate_between_two_points_distance(point1, point2, distance)
    assert isinstance(interpolated_point, np.ndarray)
    assert interpolated_point.shape == (2,)
    np.testing.assert_allclose(interpolated_point, expected_point, atol=1e-6)


def test_resample_points_regular_interval() -> None:
    """Test the resampling of points at regular intervals."""
    points = np.load(SPLINING_RESOURCES / "molecule_coords_irregular_spacing.npy")
    resampled_points = resample_points_regular_interval(points, 1.0, circular=True)
    # check that each point is approximately the right distance apart
    resampled_distances = np.linalg.norm(resampled_points[1:] - resampled_points[:-1], axis=1)
    assert np.all(np.isclose(resampled_distances, 1.0, atol=0.01))
