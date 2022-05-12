"""Tests for the grainstats module."""
from pathlib import Path
import pandas as pd
import numpy as np

from topostats.grainstats import GrainStats

# Specify the absolute and relattive tolerance for floating point comparison
TOLERANCE = {'atol': 1e-07, 'rtol': 1e-07}

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / 'tests' / 'resources'


def test_grainstats(minicircle_grainstats: GrainStats, minicircle_grainstats_20220509: pd.DataFrame) -> None:
    """Test the overall GrainStats class."""
    statistics = minicircle_grainstats.calculate_stats()
    # Uncomment to save the results, modify the date to reflect the day on which you are updating things.
    # statistics.to_csv(RESOURCES / 'minicircle_grainstats_20220509.csv', index=True)
    pd.testing.assert_frame_equal(statistics, minicircle_grainstats_20220509)
    assert False
