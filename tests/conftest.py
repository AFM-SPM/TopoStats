"""Fixtures for testing"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / 'tests' / 'resources'


RNG = np.random.default_rng(seed=1000)
SMALL_ARRAY_SIZE = (10, 10)


@pytest.fixture
def image_random() -> np.array:
    rng = np.random.default_rng(seed=1000)
    return rng.random((1024, 1024))


@pytest.fixture
def small_array() -> np.array:
    return RNG.random(SMALL_ARRAY_SIZE)

@pytest.fixture
def small_mask() -> np.array:
    return RNG.uniform(low=0, high=1, size=SMALL_ARRAY_SIZE) > 0.5

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

@pytest.fixture
def image_random_mask() -> np.array:
    df = pd.read_csv(RESOURCES / 'image_random_mask.csv.bz2',
                     header = None)
    return df.to_numpy()

@pytest.fixture
def image_random_row_quantiles_masked() -> np.array:
    return np.loadtxt(RESOURCES / 'image_random_row_quantiles_masked.csv',
                      delimiter=',')


@pytest.fixture
def image_random_col_quantiles_masked() -> np.array:
    return np.loadtxt(RESOURCES / 'image_random_col_quantiles_masked.csv',
                      delimiter=',')
