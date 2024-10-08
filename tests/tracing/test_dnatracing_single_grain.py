"""Tests for tracing single molecules."""

from pathlib import Path

import numpy as np
import pytest

from topostats.tracing.dnatracing import (
    crop_array,
    dnaTrace,
    grain_anchor,
    pad_bounding_box,
    trace_grain,
)

# This is required because of the inheritance used throughout
# pylint: disable=redefined-outer-name
BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"
PIXEL_SIZE = 0.4940029296875
MIN_SKELETON_SIZE = 10
PAD_WIDTH = 30

LINEAR_IMAGE = np.load(RESOURCES / "dnatracing_image_linear.npy")
LINEAR_MASK = np.load(RESOURCES / "dnatracing_mask_linear.npy")
CIRCULAR_IMAGE = np.load(RESOURCES / "dnatracing_image_circular.npy")
CIRCULAR_MASK = np.load(RESOURCES / "dnatracing_mask_circular.npy")


@pytest.fixture()
def dnatrace_linear(process_scan_config: dict) -> dnaTrace:
    """dnaTrace object instantiated with a single linear grain."""  # noqa: D403
    tracing_config = process_scan_config["dnatracing"]
    tracing_config.pop("run")
    tracing_config.pop("pad_width")
    return dnaTrace(
        image=LINEAR_IMAGE,
        mask=LINEAR_MASK,
        filename="linear",
        pixel_to_nm_scaling=PIXEL_SIZE,
        **tracing_config,
    )


@pytest.fixture()
def dnatrace_circular(process_scan_config: dict) -> dnaTrace:
    """dnaTrace object instantiated with a single linear grain."""  # noqa: D403
    tracing_config = process_scan_config["dnatracing"]
    tracing_config.pop("run")
    tracing_config.pop("pad_width")
    return dnaTrace(
        image=CIRCULAR_IMAGE,
        mask=CIRCULAR_MASK,
        filename="circular",
        pixel_to_nm_scaling=PIXEL_SIZE,
        **tracing_config,
    )


@pytest.mark.parametrize(
    ("dnatrace", "gauss_image_sum"),
    [
        pytest.param("dnatrace_linear", 5.517763534147536e-06, id="linear"),
        pytest.param("dnatrace_circular", 6.126947266262167e-06, id="circular"),
    ],
)
def test_gaussian_filter(dnatrace: dnaTrace, gauss_image_sum: float, request) -> None:
    """Test of the method."""
    dnatrace = request.getfixturevalue(dnatrace)
    dnatrace.gaussian_filter()
    assert dnatrace.smoothed_mask.sum() == pytest.approx(gauss_image_sum)


@pytest.mark.parametrize(
    ("dnatrace", "skeletonisation_method", "length", "start", "end"),
    [
        pytest.param(
            "dnatrace_linear", "topostats", 120, np.asarray([28, 47]), np.asarray([106, 87]), id="linear topostats"
        ),
        pytest.param(
            "dnatrace_circular", "topostats", 150, np.asarray([59, 59]), np.asarray([113, 54]), id="circular topostats"
        ),
        pytest.param("dnatrace_linear", "zhang", 170, np.asarray([28, 47]), np.asarray([106, 87]), id="linear zhang"),
        pytest.param(
            "dnatrace_circular", "zhang", 184, np.asarray([43, 95]), np.asarray([113, 54]), id="circular zhang"
        ),
        pytest.param("dnatrace_linear", "lee", 130, np.asarray([27, 45]), np.asarray([106, 87]), id="linear lee"),
        pytest.param("dnatrace_circular", "lee", 177, np.asarray([45, 93]), np.asarray([114, 53]), id="circular lee"),
        pytest.param("dnatrace_linear", "thin", 187, np.asarray([27, 45]), np.asarray([106, 83]), id="linear thin"),
        pytest.param("dnatrace_circular", "thin", 190, np.asarray([38, 85]), np.asarray([115, 52]), id="circular thin"),
    ],
)
def test_get_disordered_trace(  # pylint: disable=too-many-positional-arguments
    dnatrace: dnaTrace, skeletonisation_method: str, length: int, start: tuple, end: tuple, request
) -> None:
    """Test of get_disordered_trace the method."""
    dnatrace = request.getfixturevalue(dnatrace)
    dnatrace.skeletonisation_method = skeletonisation_method
    dnatrace.gaussian_filter()
    dnatrace.get_disordered_trace()
    assert isinstance(dnatrace.disordered_trace, np.ndarray)
    assert len(dnatrace.disordered_trace) == length
    np.testing.assert_array_equal(dnatrace.disordered_trace[0,], start)
    np.testing.assert_array_equal(dnatrace.disordered_trace[-1,], end)


@pytest.mark.skip(reason="Need to correctly prune arrays first.")
@pytest.mark.parametrize(
    ("dnatrace", "mol_is_circular"),
    [
        # pytest.param("dnatrace_linear"), False, id="linear"),
        pytest.param("dnatrace_circular", True, id="circular"),
    ],
)
def test_linear_or_circular(dnatrace: dnaTrace, mol_is_circular: int, request) -> None:
    """Test of the linear_or_circular method."""
    dnatrace = request.getfixturevalue(dnatrace)
    dnatrace.min_skeleton_size = MIN_SKELETON_SIZE
    dnatrace.gaussian_filter()
    dnatrace.get_disordered_trace()
    # Modified as mol_is_circular is no longer an attribute and the method linear_or_circular() returns Boolean instead
    assert dnatrace.linear_or_circular(dnatrace.disordered_trace) == mol_is_circular


@pytest.mark.skip(reason="Need to correctly prune arrays first.")
@pytest.mark.parametrize(
    ("dnatrace", "length", "start", "end"),
    [
        pytest.param("dnatrace_linear", 118, np.asarray([28, 48]), np.asarray([88, 70]), id="linear"),
        pytest.param("dnatrace_circular", 151, np.asarray([59, 59]), np.asarray([59, 59]), id="circular"),
    ],
)
def test_get_ordered_traces(dnatrace: dnaTrace, length: int, start: np.array, end: np.array, request) -> None:
    """Test of the get_ordered_traces method.

    Note the coordinates at the start and end differ from the fixtures for test_get_disordered_trace, but that the
    circular molecule starts and ends in the same place but the linear doesn't (even though it is currently reported as
    being circular!).
    """
    dnatrace = request.getfixturevalue(dnatrace)
    dnatrace.gaussian_filter()
    dnatrace.get_disordered_trace()
    dnatrace.linear_or_circular(dnatrace.disordered_trace)
    dnatrace.get_ordered_traces()
    assert isinstance(dnatrace.ordered_trace, np.ndarray)
    assert len(dnatrace.ordered_trace) == length
    np.testing.assert_array_equal(
        dnatrace.ordered_trace[0,],
        start,
    )
    np.testing.assert_array_almost_equal(
        dnatrace.ordered_trace[-1,],
        end,
    )


@pytest.mark.skip(reason="Need to correctly prune arrays first.")
@pytest.mark.parametrize(
    ("dnatrace", "length", "start", "end"),
    [
        pytest.param("dnatrace_linear", 118, 8.8224769e-10, 1.7610771e-09, id="linear"),
        pytest.param("dnatrace_circular", 151, 2.5852866e-09, 2.5852866e-09, id="circular"),
    ],
)
def test_get_ordered_trace_heights(dnatrace: dnaTrace, length: int, start: float, end: float, request) -> None:
    """Test of the get_trace_heights method."""
    dnatrace = request.getfixturevalue(dnatrace)
    dnatrace.gaussian_filter()
    dnatrace.get_disordered_trace()
    dnatrace.linear_or_circular(dnatrace.disordered_trace)
    dnatrace.get_ordered_traces()
    dnatrace.get_ordered_trace_heights()
    assert isinstance(dnatrace.ordered_trace_heights, np.ndarray)
    assert len(dnatrace.ordered_trace_heights) == length
    assert dnatrace.ordered_trace_heights[0] == pytest.approx(start, abs=1e-12)
    assert dnatrace.ordered_trace_heights[-1] == pytest.approx(end, abs=1e-12)


@pytest.mark.skip(reason="Need to correctly prune arrays first.")
@pytest.mark.parametrize(
    ("dnatrace", "length", "start", "end"),
    [
        pytest.param("dnatrace_linear", 118, 0.0, 6.8234101e-08, id="linear"),
        pytest.param("dnatrace_circular", 151, 0.0, 8.3513084e-08, id="circular"),
    ],
)
def test_ordered_get_trace_cumulative_distances(
    dnatrace: dnaTrace, length: int, start: float, end: float, request
) -> None:
    """Test of the get_trace_cumulative_distances method."""
    dnatrace = request.getfixturevalue(dnatrace)
    dnatrace.gaussian_filter()
    dnatrace.get_disordered_trace()
    dnatrace.linear_or_circular(dnatrace.disordered_trace)
    dnatrace.get_ordered_traces()
    dnatrace.get_ordered_trace_heights()
    dnatrace.get_ordered_trace_cumulative_distances()
    assert isinstance(dnatrace.ordered_trace_cumulative_distances, np.ndarray)
    assert len(dnatrace.ordered_trace_cumulative_distances) == length
    assert dnatrace.ordered_trace_cumulative_distances[0] == pytest.approx(start, abs=1e-11)
    assert dnatrace.ordered_trace_cumulative_distances[-1] == pytest.approx(end, abs=1e-11)
    # Check that the cumulative distance is always increasing
    assert np.all(np.diff(dnatrace.ordered_trace_cumulative_distances) > 0)


@pytest.mark.skip(reason="Need to correctly prune arrays first.")
@pytest.mark.parametrize(
    ("coordinate_list", "pixel_to_nm_scaling", "target_list"),
    [
        pytest.param(np.asarray([[1, 1], [1, 2]]), 1.0, np.asarray([0.0, 1.0]), id="Horizontal line; scaling 1.0"),
        pytest.param(np.asarray([[1, 1], [1, 2]]), 0.5, np.asarray([0.0, 0.5]), id="Horizontal line; scaling 0.5"),
        pytest.param(np.asarray([[1, 1], [2, 2]]), 1.0, np.asarray([0.0, np.sqrt(2)]), id="Diagonal line; scaling 1.0"),
        pytest.param(
            np.asarray([[1, 1], [2, 2], [3, 2], [4, 2], [4, 3]]),
            1.0,
            np.asarray([0.0, np.sqrt(2), np.sqrt(2) + 1.0, np.sqrt(2) + 2.0, np.sqrt(2) + 3.0]),
            id="Complex line; scaling 1.0",
        ),
    ],
)
def test_coord_dist(coordinate_list: list, pixel_to_nm_scaling: float, target_list: list) -> None:
    """Test of the coord_dist method."""
    cumulative_distance_list = dnaTrace.coord_dist(coordinate_list, pixel_to_nm_scaling)

    assert isinstance(cumulative_distance_list, np.ndarray)
    assert cumulative_distance_list.shape[0] == target_list.shape[0]
    np.testing.assert_array_almost_equal(cumulative_distance_list, target_list)


@pytest.mark.skip(reason="Need to correctly prune arrays first.")
@pytest.mark.parametrize(
    ("dnatrace", "length", "start", "end"),
    [
        pytest.param("dnatrace_linear", 118, np.asarray([28, 49]), np.asarray([88, 75]), id="linear"),
        pytest.param("dnatrace_circular", 151, np.asarray([58, 58]), np.asarray([58, 58]), id="circular"),
    ],
)
def test_get_fitted_traces(dnatrace: dnaTrace, length: int, start: np.array, end: np.array, request) -> None:
    """Test of the method get_fitted_traces()."""
    dnatrace = request.getfixturevalue(dnatrace)
    dnatrace.gaussian_filter()
    dnatrace.get_disordered_trace()
    dnatrace.linear_or_circular(dnatrace.disordered_trace)
    dnatrace.get_ordered_traces()
    dnatrace.linear_or_circular(dnatrace.disordered_trace)
    dnatrace.get_fitted_traces()
    assert isinstance(dnatrace.fitted_trace, np.ndarray)
    assert len(dnatrace.fitted_trace) == length
    np.testing.assert_array_equal(
        dnatrace.fitted_trace[0,],
        start,
    )
    np.testing.assert_array_almost_equal(
        dnatrace.fitted_trace[-1,],
        end,
    )


@pytest.mark.skip(reason="Need to correctly prune arrays first.")
@pytest.mark.parametrize(
    ("dnatrace", "length", "start", "end"),
    [
        pytest.param(
            "dnatrace_linear",
            1652,
            np.asarray([35.357143, 46.714286]),
            np.asarray([35.357143, 46.714286]),
            id="linear",
        ),
        pytest.param(
            "dnatrace_circular",
            2114,
            np.asarray([59.285714, 65.428571]),
            np.asarray([59.285714, 65.428571]),
            id="circular",
        ),
    ],
)
def test_get_splined_traces(dnatrace: dnaTrace, length: int, start: np.array, end: np.array, request) -> None:
    """Test of the method for get_splined_traces()."""
    dnatrace = request.getfixturevalue(dnatrace)
    dnatrace.gaussian_filter()
    dnatrace.get_disordered_trace()
    dnatrace.linear_or_circular(dnatrace.disordered_trace)
    dnatrace.get_ordered_traces()
    dnatrace.linear_or_circular(dnatrace.disordered_trace)
    dnatrace.get_fitted_traces()
    dnatrace.get_splined_traces()
    assert isinstance(dnatrace.splined_trace, np.ndarray)
    assert len(dnatrace.splined_trace) == length
    np.testing.assert_array_almost_equal(
        dnatrace.splined_trace[0,],
        start,
    )
    np.testing.assert_array_almost_equal(
        dnatrace.splined_trace[-1,],
        end,
    )


@pytest.mark.skip(reason="Need to correctly prune arrays first.")
@pytest.mark.parametrize(
    ("dnatrace", "contour_length"),
    [
        pytest.param("dnatrace_linear", 9.040267985905398e-08, id="linear"),
        pytest.param("dnatrace_circular", 7.617314045334366e-08, id="circular"),
    ],
)
def test_measure_contour_length(dnatrace: dnaTrace, contour_length: float, request) -> None:
    """Test of the method measure_contour_length()."""
    dnatrace = request.getfixturevalue(dnatrace)
    dnatrace.gaussian_filter()
    dnatrace.get_disordered_trace()
    dnatrace.linear_or_circular(dnatrace.disordered_trace)
    dnatrace.get_ordered_traces()
    dnatrace.linear_or_circular(dnatrace.disordered_trace)
    dnatrace.get_fitted_traces()
    dnatrace.get_splined_traces()
    dnatrace.measure_contour_length()
    assert dnatrace.contour_length == pytest.approx(contour_length)


@pytest.mark.skip(reason="Need to correctly prune arrays first.")
@pytest.mark.parametrize(
    ("dnatrace", "end_to_end_distance"),
    [
        pytest.param("dnatrace_linear", 0, id="linear"),
        pytest.param("dnatrace_circular", 0, id="circular"),
    ],
)
def test_measure_end_to_end_distance(dnatrace: dnaTrace, end_to_end_distance: float, request) -> None:
    """Test of the method measure_end_to_end_distance()."""
    dnatrace = request.getfixturevalue(dnatrace)
    dnatrace.gaussian_filter()
    dnatrace.get_disordered_trace()
    dnatrace.linear_or_circular(dnatrace.disordered_trace)
    dnatrace.get_ordered_traces()
    dnatrace.linear_or_circular(dnatrace.disordered_trace)
    dnatrace.get_fitted_traces()
    dnatrace.get_splined_traces()
    dnatrace.measure_end_to_end_distance()
    assert dnatrace.end_to_end_distance == pytest.approx(end_to_end_distance)


@pytest.mark.parametrize(
    ("bounding_box", "pad_width", "target_array"),
    [
        (
            # Top right, no padding, does not extend beyond border
            (1, 1, 4, 4),
            0,
            np.asarray(
                [
                    [1, 0, 0],
                    [0, 0, 0],
                    [0, 0, 1],
                ]
            ),
        ),
        (
            # Top right, 1 cell padding, does not extend beyond border
            (1, 1, 4, 4),
            1,
            np.asarray(
                [
                    [0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0],
                ]
            ),
        ),
        (
            # Top right, 2 cell padding, extends beyond border
            (1, 1, 4, 4),
            2,
            np.asarray(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ]
            ),
        ),
        (
            # Bottom left, no padding, does not extend beyond border
            (6, 6, 9, 9),
            0,
            np.asarray(
                [
                    [1, 0, 0],
                    [0, 0, 0],
                    [0, 0, 1],
                ]
            ),
        ),
        (
            # Bottom left, one cell padding, does not extend beyond border
            (6, 6, 9, 9),
            1,
            np.asarray(
                [
                    [0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0],
                ]
            ),
        ),
        (
            # Bottom left, two cell padding, extends beyond border
            (6, 6, 9, 9),
            2,
            np.asarray(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0],
                ]
            ),
        ),
        (
            # Bottom left, three cell padding, extends beyond border
            (6, 6, 9, 9),
            3,
            np.asarray(
                [
                    [1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ]
            ),
        ),
    ],
)
def test_crop_array(bounding_box: tuple, pad_width: int, target_array: list) -> None:
    """Test cropping of arrays."""
    array = np.zeros((10, 10))
    upper_box = np.asarray([[1, 1], [3, 3]])
    lower_box = np.asarray([[6, 6], [8, 8]])
    array[upper_box[:, 0], upper_box[:, 1]] = 1
    array[lower_box[:, 0], lower_box[:, 1]] = 1
    cropped_array = crop_array(array, bounding_box, pad_width)
    np.testing.assert_array_equal(cropped_array, target_array)


@pytest.mark.parametrize(
    ("array_shape", "bounding_box", "pad_width", "target_coordinates"),
    [
        pytest.param((10, 10), [1, 1, 5, 5], 1, [0, 0, 6, 6], id="1x5 box with pad width of 1"),
        pytest.param((10, 10), [1, 1, 5, 5], 3, [0, 0, 8, 8], id="1x5 box with pad width of 3"),
        pytest.param((10, 10), [4, 4, 5, 5], 1, [3, 3, 6, 6], id="1x1 box with pad width of 1"),
        pytest.param((10, 10), [4, 4, 5, 5], 3, [1, 1, 8, 8], id="1x3 box with pad width of 3"),
        pytest.param(
            (10, 10), [4, 4, 5, 5], 6, [0, 0, 10, 10], id="1x5 box with pad width of 6 (exceeds image boundary)"
        ),
    ],
)
def test_pad_bounding_box(array_shape: tuple, bounding_box: list, pad_width: int, target_coordinates: tuple) -> None:
    """Test the padding ofbounding boxes."""
    padded_bounding_box = pad_bounding_box(array_shape, bounding_box, pad_width)
    assert padded_bounding_box == target_coordinates


@pytest.mark.parametrize(
    ("array_shape", "bounding_box", "pad_width", "target_coordinates"),
    [
        pytest.param((10, 10), [1, 1, 5, 5], 1, (0, 0), id="1x5 box pad width of 1"),
        pytest.param((10, 10), [1, 1, 5, 5], 3, (0, 0), id="1x5 box pad width of 3"),
        pytest.param((10, 10), [4, 4, 5, 5], 1, (3, 3), id="1x1 box pad width of 1"),
        pytest.param((10, 10), [4, 4, 5, 5], 3, (1, 1), id="1x1 box pad width of 3"),
        pytest.param((10, 10), [4, 4, 5, 5], 6, (0, 0), id="1x1 box pad width of 6"),
    ],
)
def test_grain_anchor(array_shape: tuple, bounding_box: list, pad_width: int, target_coordinates: tuple) -> None:
    """Test the extraction of padded bounding boxes."""
    padded_grain_anchor = grain_anchor(array_shape, bounding_box, pad_width)
    assert padded_grain_anchor == target_coordinates


# @ns-rse failing linting needs updating
# @pytest.mark.parametrize(
#     (
#         "cropped_image",
#         "cropped_mask",
#         "filename",
#         "skeletonisation_method",
#         "end_to_end_distance",
#         "circular",
#         "contour_length",
#     ),
#     [
#         (
#             LINEAR_IMAGE,
#             LINEAR_MASK,
#             "linear_test_topostats",
#             "topostats",
#             3.115753758716346e-08,
#             False,
#             5.684734982126664e-08,
#         ),
#         (
#             CIRCULAR_IMAGE,
#             CIRCULAR_MASK,
#             "circular_test_topostats",
#             "topostats",
#             0,
#             True,
#             7.617314045334366e-08,
#         ),
#         (
#             LINEAR_IMAGE,
#             LINEAR_MASK,
#             "linear_test_zhang",
#             "zhang",
#             2.6964685842539566e-08,
#             False,
#             6.194694383968303e-08,
#         ),
#         (
#             CIRCULAR_IMAGE,
#             CIRCULAR_MASK,
#             "circular_test_zhang",
#             "zhang",
#             9.636691058914389e-09,
#             False,
#             8.187508931608563e-08,
#         ),
#         (
#             LINEAR_IMAGE,
#             LINEAR_MASK,
#             "linear_test_lee",
#             "lee",
#             3.197879765453915e-08,
#             False,
#             5.655032001817721e-08,
#         ),
#         (
#             CIRCULAR_IMAGE,
#             CIRCULAR_MASK,
#             "circular_test_lee",
#             "lee",
#             8.261640682714017e-09,
#             False,
#             8.062559919860788e-08,
#         ),
#         (
#             LINEAR_IMAGE,
#             LINEAR_MASK,
#             "linear_test_thin",
#             "thin",
#             4.068855894099921e-08,
#             False,
#             5.518856387362746e-08,
#         ),
#         (
#             CIRCULAR_IMAGE,
#             CIRCULAR_MASK,
#             "circular_test_thin",
#             "thin",
#             3.638262839374549e-08,
#             False,
#             3.6512544238919716e-08,
#         ),
#     ],
# )
# def test_trace_grain(  # pylint: disable=too-many-positional-arguments
#     cropped_image: np.ndarray,
#     cropped_mask: np.ndarray,
#     filename: str,
#     skeletonisation_method: str,
#     end_to_end_distance: float,
#     circular: bool,
#     contour_length: float,
# ) -> None:
#     """Test trace_grain function for tracing a single grain."""
#     trace_stats = trace_grain(
#         cropped_image=cropped_image,
#         cropped_mask=cropped_mask,
#         pixel_to_nm_scaling=PIXEL_SIZE,
#         filename=filename,
#         min_skeleton_size=MIN_SKELETON_SIZE,
#         skeletonisation_method=skeletonisation_method,
#     )
#     assert trace_stats["image"] == filename
#     assert trace_stats["end_to_end_distance"] == pytest.approx(end_to_end_distance)
#     assert trace_stats["circular"] == circular
#     assert trace_stats["contour_length"] == pytest.approx(contour_length)
