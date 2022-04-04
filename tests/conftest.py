"""Fixtures for testing"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / 'tests' / 'resources'


@pytest.fixture
def image_random() -> np.array:
    rng = np.random.default_rng(seed=1000)
    return rng.random((1024, 1024))


@pytest.fixture
def image_random_row_quantiles() -> np.array:
    return np.loadtxt(RESOURCES / 'image_random_row_quantiles.csv',
                      delimiter=',')


@pytest.fixture
def image_random_col_quantiles() -> np.array:
    return np.loadtxt(RESOURCES / 'image_random_col_quantiles.csv',
                      delimiter=',')


@pytest.fixture
def image_random_aligned_rows() -> np.array:
    df = pd.read_csv(RESOURCES / 'image_random_aligned_rows.csv.bz2',
                     header=None)
    return df.to_numpy()


@pytest.fixture
def image_random_remove_x_y_tilt() -> np.array:
    df = pd.read_csv(RESOURCES / 'image_random_remove_x_y_tilt.csv.bz2',
                     header=None)
    return df.to_numpy()
