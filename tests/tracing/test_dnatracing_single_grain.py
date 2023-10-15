"""Tests for tracing single molecules."""
from pathlib import Path

import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture

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
def dnatrace_linear() -> dnaTrace:
    """dnaTrace object instantiated with a single linear grain."""  # noqa: D403
    return dnaTrace(
        image=LINEAR_IMAGE,
        grain=LINEAR_MASK,
        filename="linear",
        pixel_to_nm_scaling=PIXEL_SIZE,
        min_skeleton_size=MIN_SKELETON_SIZE,
        skeletonisation_method="topostats",
    )


@pytest.fixture()
def dnatrace_circular() -> dnaTrace:
    """dnaTrace object instantiated with a single linear grain."""  # noqa: D403
    return dnaTrace(
        image=CIRCULAR_IMAGE,
        grain=CIRCULAR_MASK,
        filename="circular",
        pixel_to_nm_scaling=PIXEL_SIZE,
        min_skeleton_size=MIN_SKELETON_SIZE,
        skeletonisation_method="topostats",
    )


@pytest.mark.parametrize(
    ("dnatrace", "gauss_image_sum"),
    [
        (lazy_fixture("dnatrace_linear"), 5.517763534147536e-06),
        (lazy_fixture("dnatrace_circular"), 6.126947266262167e-06),
    ],
)
def test_gaussian_filter(dnatrace: dnaTrace, gauss_image_sum: float) -> None:
    """Test of the method."""
    dnatrace.gaussian_filter()
    assert dnatrace.gauss_image.sum() == pytest.approx(gauss_image_sum)


@pytest.mark.parametrize(
    ("dnatrace", "skeletonisation_method", "length", "start", "end"),
    [
        (lazy_fixture("dnatrace_linear"), "topostats", 120, np.asarray([28, 47]), np.asarray([106, 87])),
        (lazy_fixture("dnatrace_circular"), "topostats", 150, np.asarray([59, 59]), np.asarray([113, 54])),
        (lazy_fixture("dnatrace_linear"), "zhang", 170, np.asarray([28, 47]), np.asarray([106, 87])),
        (lazy_fixture("dnatrace_circular"), "zhang", 184, np.asarray([43, 95]), np.asarray([113, 54])),
        (lazy_fixture("dnatrace_linear"), "lee", 130, np.asarray([27, 45]), np.asarray([106, 87])),
        (lazy_fixture("dnatrace_circular"), "lee", 177, np.asarray([45, 93]), np.asarray([114, 53])),
        (lazy_fixture("dnatrace_linear"), "thin", 187, np.asarray([27, 45]), np.asarray([106, 83])),
        (lazy_fixture("dnatrace_circular"), "thin", 190, np.asarray([38, 85]), np.asarray([115, 52])),
    ],
)
def test_get_disordered_trace(
    dnatrace: dnaTrace, skeletonisation_method: str, length: int, start: tuple, end: tuple
) -> None:
    """Test of get_disordered_trace the method."""
    dnatrace.skeletonisation_method = skeletonisation_method
    dnatrace.gaussian_filter()
    dnatrace.get_disordered_trace()
    assert isinstance(dnatrace.disordered_trace, np.ndarray)
    assert len(dnatrace.disordered_trace) == length
    np.testing.assert_array_equal(dnatrace.disordered_trace[0,], start)
    np.testing.assert_array_equal(dnatrace.disordered_trace[-1,], end)


# Currently linear molecule isn't detected as linear, although it was when selecting and extracting in a Notebook
@pytest.mark.parametrize(
    ("dnatrace", "mol_is_circular"),
    [
        # (lazy_fixture("dnatrace_linear"), False),
        (lazy_fixture("dnatrace_circular"), True),
    ],
)
def test_linear_or_circular(dnatrace: dnaTrace, mol_is_circular: int) -> None:
    """Test of the linear_or_circular method."""
    dnatrace.min_skeleton_size = MIN_SKELETON_SIZE
    dnatrace.gaussian_filter()
    dnatrace.get_disordered_trace()
    dnatrace.linear_or_circular(dnatrace.disordered_trace)
    assert dnatrace.mol_is_circular == mol_is_circular


@pytest.mark.parametrize(
    ("dnatrace", "length", "start", "end"),
    [
        (lazy_fixture("dnatrace_linear"), 118, np.asarray([28, 48]), np.asarray([88, 70])),
        (lazy_fixture("dnatrace_circular"), 151, np.asarray([59, 59]), np.asarray([59, 59])),
    ],
)
def test_get_ordered_traces(dnatrace: dnaTrace, length: int, start: np.array, end: np.array) -> None:
    """Test of the get_ordered_traces method.

    Note the co-ordinates at the start and end differ from the fixtures for test_get_disordered_trace, but that the
    circular molecule starts and ends in the same place but the linear doesn't (even though it is currently reported as
    being circular!).
    """
    dnatrace.gaussian_filter()
    dnatrace.get_disordered_trace()
    dnatrace.linear_or_circular(dnatrace.disordered_trace)
    dnatrace.get_ordered_traces()
    assert isinstance(dnatrace.ordered_trace, np.ndarray)
    assert len(dnatrace.ordered_trace) == length
    np.testing.assert_array_equal(dnatrace.ordered_trace[0,], start)
    np.testing.assert_array_almost_equal(dnatrace.ordered_trace[-1,], end)


@pytest.mark.parametrize(
    ("dnatrace", "length", "start", "end"),
    [
        (lazy_fixture("dnatrace_linear"), 118, np.asarray([28, 49]), np.asarray([88, 75])),
        (lazy_fixture("dnatrace_circular"), 151, np.asarray([58, 58]), np.asarray([58, 58])),
    ],
)
def test_get_fitted_traces(dnatrace: dnaTrace, length: int, start: np.array, end: np.array) -> None:
    """Test of the method get_fitted_traces()."""
    dnatrace.gaussian_filter()
    dnatrace.get_disordered_trace()
    dnatrace.linear_or_circular(dnatrace.disordered_trace)
    dnatrace.get_ordered_traces()
    dnatrace.linear_or_circular(dnatrace.disordered_trace)
    dnatrace.get_fitted_traces()
    assert isinstance(dnatrace.fitted_trace, np.ndarray)
    assert len(dnatrace.fitted_trace) == length
    np.testing.assert_array_equal(dnatrace.fitted_trace[0,], start)
    np.testing.assert_array_almost_equal(dnatrace.fitted_trace[-1,], end)


@pytest.mark.parametrize(
    ("dnatrace", "length", "start", "end"),
    [
        (
            lazy_fixture("dnatrace_linear"),
            1652,
            np.asarray([35.542197, 46.683159]),
            np.asarray([35.542197, 46.683159]),
        ),
        (
            lazy_fixture("dnatrace_circular"),
            2114,
            np.asarray([58.999833, 65.47856]),
            np.asarray([58.999833, 65.47856]),
        ),
    ],
)
def test_get_splined_traces(dnatrace: dnaTrace, length: int, start: np.array, end: np.array) -> None:
    """Test of the method for get_splined_traces()."""
    dnatrace.gaussian_filter()
    dnatrace.get_disordered_trace()
    dnatrace.linear_or_circular(dnatrace.disordered_trace)
    dnatrace.get_ordered_traces()
    dnatrace.linear_or_circular(dnatrace.disordered_trace)
    dnatrace.get_fitted_traces()
    dnatrace.get_splined_traces()
    assert isinstance(dnatrace.splined_trace, np.ndarray)
    assert len(dnatrace.splined_trace) == length
    np.testing.assert_array_almost_equal(dnatrace.splined_trace[0,], start)
    np.testing.assert_array_almost_equal(dnatrace.splined_trace[-1,], end)


@pytest.mark.parametrize(
    ("dnatrace", "contour_length"),
    [
        (lazy_fixture("dnatrace_linear"), 8.910711063861682e-08),
        (lazy_fixture("dnatrace_circular"), 7.574136072208753e-08),
    ],
)
def test_measure_contour_length(dnatrace: dnaTrace, contour_length: float) -> None:
    """Test of the method measure_contour_length()."""
    dnatrace.gaussian_filter()
    dnatrace.get_disordered_trace()
    dnatrace.linear_or_circular(dnatrace.disordered_trace)
    dnatrace.get_ordered_traces()
    dnatrace.linear_or_circular(dnatrace.disordered_trace)
    dnatrace.get_fitted_traces()
    dnatrace.get_splined_traces()
    dnatrace.measure_contour_length()
    assert dnatrace.contour_length == pytest.approx(contour_length)


# Currently need an actual linear grain to test this.
@pytest.mark.parametrize(
    ("dnatrace", "end_to_end_distance"),
    [
        (lazy_fixture("dnatrace_linear"), 0),
        (lazy_fixture("dnatrace_circular"), 0),
    ],
)
def test_measure_end_to_end_distance(dnatrace: dnaTrace, end_to_end_distance: float) -> None:
    """Test of the method measure_end_to_end_distance()."""
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
        ((10, 10), [1, 1, 5, 5], 1, [0, 0, 6, 6]),
        ((10, 10), [1, 1, 5, 5], 3, [0, 0, 8, 8]),
        ((10, 10), [4, 4, 5, 5], 1, [3, 3, 6, 6]),
        ((10, 10), [4, 4, 5, 5], 3, [1, 1, 8, 8]),
        ((10, 10), [4, 4, 5, 5], 6, [0, 0, 10, 10]),
    ],
)
def test_pad_bounding_box(array_shape: tuple, bounding_box: list, pad_width: int, target_coordinates: tuple) -> None:
    """Test the padding ofbounding boxes."""
    padded_bounding_box = pad_bounding_box(array_shape, bounding_box, pad_width)
    assert padded_bounding_box == target_coordinates


@pytest.mark.parametrize(
    ("array_shape", "bounding_box", "pad_width", "target_coordinates"),
    [
        ((10, 10), [1, 1, 5, 5], 1, (0, 0)),
        ((10, 10), [1, 1, 5, 5], 3, (0, 0)),
        ((10, 10), [4, 4, 5, 5], 1, (3, 3)),
        ((10, 10), [4, 4, 5, 5], 3, (1, 1)),
        ((10, 10), [4, 4, 5, 5], 6, (0, 0)),
    ],
)
def test_grain_anchor(array_shape: tuple, bounding_box: list, pad_width: int, target_coordinates: tuple) -> None:
    """Test the extraction of padded bounding boxes."""
    padded_grain_anchor = grain_anchor(array_shape, bounding_box, pad_width)
    assert padded_grain_anchor == target_coordinates


@pytest.mark.parametrize(
    (
        "cropped_image",
        "cropped_mask",
        "filename",
        "skeletonisation_method",
        "end_to_end_distance",
        "circular",
        "contour_length",
    ),
    [
        (
            LINEAR_IMAGE,
            LINEAR_MASK,
            "linear_test_topostats",
            "topostats",
            3.115753758716346e-08,
            False,
            5.684734982126664e-08,
        ),
        (
            CIRCULAR_IMAGE,
            CIRCULAR_MASK,
            "circular_test_topostats",
            "topostats",
            0,
            True,
            7.574136072208753e-08,
        ),
        (
            LINEAR_IMAGE,
            LINEAR_MASK,
            "linear_test_zhang",
            "zhang",
            2.6964685842539566e-08,
            False,
            6.194694383968303e-08,
        ),
        (
            CIRCULAR_IMAGE,
            CIRCULAR_MASK,
            "circular_test_zhang",
            "zhang",
            9.636691058914389e-09,
            False,
            8.187508931608563e-08,
        ),
        (
            LINEAR_IMAGE,
            LINEAR_MASK,
            "linear_test_lee",
            "lee",
            3.197879765453915e-08,
            False,
            5.655032001817721e-08,
        ),
        (
            CIRCULAR_IMAGE,
            CIRCULAR_MASK,
            "circular_test_lee",
            "lee",
            8.261640682714017e-09,
            False,
            8.062559919860788e-08,
        ),
        (
            LINEAR_IMAGE,
            LINEAR_MASK,
            "linear_test_thin",
            "thin",
            4.068855894099921e-08,
            False,
            5.518856387362746e-08,
        ),
        (
            CIRCULAR_IMAGE,
            CIRCULAR_MASK,
            "circular_test_thin",
            "thin",
            3.638262839374549e-08,
            False,
            3.6512544238919716e-08,
        ),
    ],
)
def test_trace_grain(
    cropped_image: np.ndarray,
    cropped_mask: np.ndarray,
    filename: str,
    skeletonisation_method: str,
    end_to_end_distance: float,
    circular: bool,
    contour_length: float,
) -> None:
    """Test trace_grain function for tracing a single grain."""
    trace_stats = trace_grain(
        cropped_image=cropped_image,
        cropped_mask=cropped_mask,
        pixel_to_nm_scaling=PIXEL_SIZE,
        filename=filename,
        min_skeleton_size=MIN_SKELETON_SIZE,
        skeletonisation_method=skeletonisation_method,
    )
    assert trace_stats["image"] == filename
    assert trace_stats["end_to_end_distance"] == pytest.approx(end_to_end_distance)
    assert trace_stats["circular"] == circular
    assert trace_stats["contour_length"] == pytest.approx(contour_length)
