"""Test the splining module."""

import numpy as np
import numpy.typing as npt
import pytest

from topostats.tracing.splining import windowTrace

# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals


@pytest.mark.parametrize(
    ("pixel_trace", "rolling_window_size", "pixel_to_nm_scaling", "expected_pooled_trace"),
    [
        pytest.param(
            np.array(
                [[0, 0], [0, 1], [0, 2], [0, 3], [1, 3], [2, 3], [3, 3], [3, 2], [3, 1], [3, 0], [2, 0], [1, 0]]
            ).astype(np.int32),
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
            np.array(
                [[0, 0], [0, 1], [0, 2], [0, 3], [1, 3], [2, 3], [3, 3], [3, 2], [3, 1], [3, 0], [2, 0], [1, 0]]
            ).astype(np.int32),
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
            np.array(
                [[0, 0], [0, 1], [0, 2], [0, 3], [1, 3], [2, 3], [3, 3], [3, 2], [3, 1], [3, 0], [2, 0], [1, 0]]
            ).astype(np.int32),
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
            np.array(
                [[0, 0], [0, 1], [0, 2], [0, 3], [1, 3], [2, 3], [3, 3], [3, 2], [3, 1], [3, 0], [2, 0], [1, 0]]
            ).astype(np.int32),
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
