"""Tests for the maths functions."""

import pytest
import numpy as np
import pandas as pd

from topostats.maths import round_sig_figs, round_sig_figs_dataframe


@pytest.mark.parametrize(
    "value, sig_figs, expected",
    [
        (
            1.234567e-8,
            4,
            1.235e-8,
        ),
        (
            1.234567e8,
            4,
            1.235e8,
        ),
        (
            1,
            4,
            1,
        ),
        (
            1.234567e-8,
            2,
            1.2e-8,
        ),
        (
            1.234567e8,
            3,
            1.23e8,
        ),
        (
            1,
            4,
            1,
        ),
        (
            np.nan,
            4,
            np.nan,
        ),
    ],
)
def test_round_sig_figs(value: float, sig_figs: int, expected: float) -> None:
    """Test the round_sig_figs function of maths.py, ensuring that the value is
    rounded to a given number of significant figures."""

    result = round_sig_figs(value, sig_figs)

    np.testing.assert_equal(result, expected)


def test_round_sig_figs_dataframe() -> None:
    """Test the round_sig_figs_dataframe function of maths.py, ensuring that
    DataFrame floating point values are rounded to a given number of significant figures."""

    data = {
        "A": [1.234567e18, 2.345678321e18],
        "B": [3.456789312e18, 4.567890312e18],
        "C": [5, 6],
        "D": ["Abc", "def"],
        "E": [True, False],
    }
    data_df = pd.DataFrame(data)

    sig_figs = 4

    rounded_df = round_sig_figs_dataframe(dataframe=data_df, sig_figs=sig_figs)

    expected = {
        "A": [1.235e18, 2.346e18],
        "B": [3.457e18, 4.568e18],
        "C": [5, 6],
        "D": ["Abc", "def"],
        "E": [True, False],
    }

    expected_df = pd.DataFrame(expected)

    pd.testing.assert_frame_equal(expected_df, rounded_df)
