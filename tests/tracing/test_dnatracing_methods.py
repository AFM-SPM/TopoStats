"""Additional tests of dnaTracing methods."""

from pathlib import Path

import numpy as np
import pytest

from topostats.tracing.dnatracing import dnaTrace, round_splined_traces
from topostats.tracing.tracingfuncs import reorderTrace

# This is required because of the inheritance used throughout
# pylint: disable=redefined-outer-name
BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"
PIXEL_SIZE = 0.4940029296875

LINEAR_IMAGE = np.load(RESOURCES / "dnatracing_image_linear.npy")
LINEAR_MASK = np.load(RESOURCES / "dnatracing_mask_linear.npy")
CIRCULAR_IMAGE = np.load(RESOURCES / "dnatracing_image_circular.npy")
CIRCULAR_MASK = np.load(RESOURCES / "dnatracing_mask_circular.npy")
MIN_SKELETON_SIZE = 10


@pytest.fixture()
def dnatrace() -> dnaTrace:
    """dnaTrace object for use in tests."""  # noqa: D403
    return dnaTrace(
        image=np.asarray([[1]]),
        mask=None,
        filename="test.spm",
        pixel_to_nm_scaling=PIXEL_SIZE,
        min_skeleton_size=MIN_SKELETON_SIZE,
    )


@pytest.fixture()
def dnatrace_spline(dnatrace: dnaTrace) -> dnaTrace:
    """Instantiate a dnaTrace object for splining tests."""
    dnatrace.pixel_to_nm_scaling = 1.0
    dnatrace.n_grain = 1
    return dnatrace


@pytest.mark.parametrize(
    ("trace_image", "step_size_m", "mol_is_circular", "smoothness", "expected_spline_image"),
    # Note that some images here have gaps in them. This is NOT an issue with the splining, but
    # a natural result of rounding the floating point coordinates to integers. We only do this
    # to provide a visualisation for easy test comparison and does not affect results.
    [
        # Test circular with no smoothing. The shape remains unchanged. The gap is just a result
        # of rounding the floating point coordinates to integers for this neat visualisation
        # and can be ignored.
        (
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 1, 1, 1, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
            1.0,
            True,
            0.0,
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 0, 1, 1, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 1, 1, 1, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
        ),
        # Test circular with smoothing. The shape is smoothed out.
        (
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 1, 1, 1, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
            1.0,
            True,
            10.0,
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 1, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
        ),
        # Another circular with no smoothing. Relatively unchanged
        (
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                    [0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
            10.0,
            True,
            0.0,
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                    [0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
        ),
        # Another circular with smoothing. The shape is smoothed out.
        (
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                    [0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
            10.0,
            True,
            5.0,
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
                    [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
        ),
        # Simple line with smoothing unchanged because too simple.
        (
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
            5.0,
            False,
            5.0,
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
        ),
        # Another line with smoothing unchanged because too simple.
        (
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            ),
            4.0,
            False,
            5.0,
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            ),
        ),
        # Complex line without smoothing, unchanged.
        (
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
            5.0,
            False,
            0.0,
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
        ),
        # Complex line with smoothing, smoothed out.
        (
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
            5.0,
            False,
            5.0,
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
        ),
    ],
)
def test_get_splined_traces(
    dnatrace_spline: dnaTrace,
    trace_image: np.ndarray,
    step_size_m: float,
    mol_is_circular: bool,
    smoothness: float,
    expected_spline_image: np.ndarray,
) -> None:
    """Test the get_splined_traces() function of dnatracing.py."""
    # For development visualisations - keep this in for future use
    # plt.imsave('./fitted_trace.png', trace_image)

    # Obtain the coords of the pixels from our test image
    trace_coords = np.argwhere(trace_image == 1)

    # Get an ordered trace from our test trace images
    if mol_is_circular:
        fitted_trace, _whether_trace_completed = reorderTrace.circularTrace(trace_coords)
    else:
        fitted_trace = reorderTrace.linearTrace(trace_coords)

    # Set dnatrace smoothness accordingly
    if mol_is_circular:
        dnatrace_spline.spline_circular_smoothing = smoothness
    else:
        dnatrace_spline.spline_linear_smoothing = smoothness

    # Generate splined trace
    dnatrace_spline.fitted_trace = fitted_trace
    dnatrace_spline.step_size_m = step_size_m
    # Fixed pixel to nm scaling since changing this is redundant due to the internal effect being linked to
    # the step_size_m divided by this value, so changing both doesn't make sense.
    dnatrace_spline.mol_is_circular = mol_is_circular

    # Spline the traces
    dnatrace_spline.get_splined_traces()

    # Extract the splined trace
    splined_trace = dnatrace_spline.splined_trace

    # This is just for easier human-readable tests. Turn the splined coords into a visualisation.
    splined_image = np.zeros_like(trace_image)
    splined_image[np.round(splined_trace[:, 0]).astype(int), np.round(splined_trace[:, 1]).astype(int)] = 1

    # For development visualisations - keep this in for future use
    # plt.imsave(f'./test_splined_image_{splined_image.shape}.png', splined_image)

    np.testing.assert_array_equal(splined_image, expected_spline_image)


def test_round_splined_traces():
    """Test the round splined traces function of dnatracing.py."""
    splined_traces = {"0": np.array([[1.2, 2.3], [3.4, 4.5]]), "1": None, "2": np.array([[5.6, 6.7], [7.8, 8.9]])}
    expected_result = {"0": np.array([[1, 2], [3, 4]]), "1": None, "2": np.array([[6, 7], [8, 9]])}
    result = round_splined_traces(splined_traces)
    np.testing.assert_equal(result, expected_result)
