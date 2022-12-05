"""Tests of dnatracing methods."""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from topostats.tracing.dnatracing import dnaTrace, traceStats

# pylint: disable=protected-access

# keeping target dictionary as may want to use these grains for more tests

DILTATE_TARGETS = {
    1: np.asarray(
        [
            [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]),
    2: np.asarray(
        [
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        ]),
    3: np.asarray(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        ])
    }

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


def test_binary_img_to_dict() -> None:
    """Test that a binary image can be separated into arrays."""
    mask = multigrain.copy()
    mask[mask !=0 ] = 0
    test_dict = dnaTrace.binary_img_to_dict(multigrain)
    assert len(test_dict) == 3
    for grain in range(1, 3 + 1):
        assert test_dict[grain].shape == multigrain.shape
        assert (test_dict[grain] == np.where(multigrain == grain, 1, 0)).all


def test_dilate(dnatracing_dilate: dnaTrace) -> None:
    """Tests dilate function outputs."""
    assert len(dnatracing_dilate.grains) == 3 # ensure is not empty
    for grain_no, grain_array in dnatracing_dilate.grains.items():
        assert grain_array.shape == DILTATE_TARGETS[grain_no].shape
        assert (grain_array == DILTATE_TARGETS[grain_no]).all()


def test_dnatracing_get_disordered_trace(dnatracing_disordered_traces: dnaTrace):
    """Tests grain coordinates are extracted."""
    for grain_no, coords in dnatracing_disordered_traces.disordered_traces.items():
        assert len(coords) == DILATE_TARGETS[grain_no].sum()



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
    assert 1 == 1
