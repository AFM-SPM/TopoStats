"""Tests the dnatracing module"""
import pandas as pd


def test_dnatracing_minicircle(minicircle_dnatracing: pd.DataFrame, minicircle_dnastats: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(minicircle_dnatracing, minicircle_dnastats)
