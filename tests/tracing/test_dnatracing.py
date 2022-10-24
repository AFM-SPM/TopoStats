"""Tests of dnatracing methods."""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from topostats.tracing.dnatracing import dnaTrace, traceStats

# pylint: disable=protected-access


def test_get_grain_array(test_dnatracing: pd.DataFrame) -> None:
    """Test extraction of grain array."""
    target = {
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
    }
    for x in range(1, 4):
        np.testing.assert_array_equal(test_dnatracing._get_grain_array(x), target[x])


@pytest.mark.parametrize(
    "molecule, array_sum",
    [
        (1, 2859),
        (2, 1711),
        (3, 1802),
        (4, 1646),
        (5, 1725),
        (6, 1715),
        (7, 1756),
        (8, 1882),
        (9, 2119),
        (10, 1930),
        (11, 2065),
        (12, 1999),
        (13, 1947),
        (14, 1920),
        (15, 1797),
        (16, 1924),
        (17, 1987),
        (18, 2029),
        (19, 1849),
        (20, 2004),
        (21, 1895),
        (22, 1817),
    ],
)
def test_minicircle_dnatracing_skeleton_size(regtest, minicircle_dnatracing: traceStats, molecule, array_sum) -> None:
    """Test the skeletons returned by dnatracing."""
    assert minicircle_dnatracing._get_grain_array(molecule).sum() == array_sum
    mol_array = minicircle_dnatracing._get_grain_array(1)


def test_minicircle_dnatracing_skeletons(regtest, minicircle_dnatracing: traceStats) -> None:
    """Test the skeletons returned by dnatracing."""
    mol_array = minicircle_dnatracing._get_grain_array(1)
    assert isinstance(mol_array, np.ndarray)
    print(np.array2string(mol_array), file=regtest)


@pytest.mark.mpl_image_compare(baseline_dir="../resources/img/")
def test_minicircle_dnatracing_skeletons_plot(regtest, minicircle_dnatracing: traceStats) -> None:
    """Test the skeletons returned by dnatracing."""
    mol_array = minicircle_dnatracing._get_grain_array(1)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(mol_array)
    return fig


def test_tracestats(regtest, minicircle_tracestats: traceStats) -> None:
    """Regression tests for DNA trace statistics."""
    print(minicircle_tracestats.to_string(), file=regtest)
