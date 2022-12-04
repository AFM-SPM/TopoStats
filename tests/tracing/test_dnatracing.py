"""Tests of dnatracing methods."""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from topostats.tracing.dnatracing import dnaTrace, traceStats

# pylint: disable=protected-access

# keeping target dictionary as may want to use these grains for more tests

mask_n_points = {
    "masks": {
        1: np.array(
            [
                [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        ),
        2: np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ]
        ),
        3: np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        ),
        4: np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            ]
        ),
        },
    "points": {
        1: "a"
        }}

multigrain = np.array(
    [
        [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 2],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 2],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 2],
        [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
        [0, 0, 3, 3, 3, 3, 0, 0, 0, 0, 0],
        [3, 3, 0, 0, 0, 0, 3, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 0],
    ])


@pytest.mark.parametrize(
    "multi_grain, no_grains",
    [(multigrain, 3)],
)
def test_binary_img_to_dict(multi_grain: np.ndarray, no_grains: int) -> None:
    """Test that a binary image can be separated into arrays."""
    mask = multi_grain.copy()
    mask[mask !=0 ] = 0
    test_dict = dnaTrace.binary_img_to_dict(multi_grain)
    assert len(test_dict) == no_grains
    for grain in range(1, no_grains + 1):
        assert test_dict[grain].shape == multi_grain.shape
        assert (test_dict[grain] == np.where(multi_grain == grain, 1, 0)).all


# @pytest.mark.mpl_image_compare(baseline_dir="../resources/img/")
# def test_minicircle_dnatracing_skeletons_plot(regtest, minicircle_dnatracing: traceStats) -> None:
#     """Test the skeletons returned by dnatracing."""
#     mol_array = minicircle_dnatracing._get_grain_array(1)
#     fig, ax = plt.subplots(1, 1, figsize=(8, 8))
#     ax.imshow(mol_array)
#     return fig


def test_tracestats(regtest, minicircle_tracestats: traceStats) -> None:
    """Regression tests for DNA trace statistics."""
    print(minicircle_tracestats.to_string(), file=regtest)
