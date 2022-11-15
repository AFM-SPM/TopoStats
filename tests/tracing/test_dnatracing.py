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
        (1, 2872),
        (2, 1726),
        (3, 1770),
        (4, 1624),
        (5, 1706),
        (6, 1695),
        (7, 1729),
        (8, 1855),
        (9, 2085),
        (10, 1900),
        (11, 2054),
        (12, 2003),
        (13, 1919),
        (14, 1887),
        (15, 1766),
        (16, 1898),
        (17, 1965),
        (18, 2000),
        (19, 1818),
        (20, 1967),
        (21, 1895),
        (22, 1826),
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
