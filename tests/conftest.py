"""Fixtures for testing"""
# pylint: disable=no-name-in-module
# pylint: disable=redefined-outer-name
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import pytest

from pySPM.SPM import SPM_image
from pySPM.Bruker import Bruker

from topostats.filters import (extract_img_name, extract_channel, extract_pixel_to_nm_scaling, extract_pixels,
                               align_rows, remove_x_y_tilt, average_background)
from topostats.find_grains import (gaussian_filter, tidy_border, remove_objects, label_regions,
                                   minimum_grain_size_pixels, colour_regions, region_properties)
from topostats.io import load_scan, read_yaml
from topostats.utils import get_mask, get_threshold

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / 'tests' / 'resources'

RNG = np.random.default_rng(seed=1000)
SMALL_ARRAY_SIZE = (10, 10)
THRESHOLD = 0.5
CHANNEL = 'Height'


@pytest.fixture
def sample_config() -> dict:
    """Sample configuration"""
    return read_yaml(RESOURCES / 'sample_config.yaml')


@pytest.fixture
def grain_config(sample_config) -> dict:
    """Configurations for grain finding."""
    return sample_config['grains']


@pytest.fixture
def image_random() -> np.array:
    """Random 1024x1024 image for testing."""
    rng = np.random.default_rng(seed=1000)
    return rng.random((1024, 1024))


@pytest.fixture
def small_array() -> np.array:
    """Small (10x10) image array for testing"""
    return RNG.random(SMALL_ARRAY_SIZE)


@pytest.fixture
def small_mask() -> np.array:
    """Small (10x10) mask array for testing."""
    return RNG.uniform(low=0, high=1, size=SMALL_ARRAY_SIZE) > 0.5


@pytest.fixture
def image_random_row_quantiles() -> np.array:
    """Expected row quantiles (unmasked)"""
    return np.loadtxt(RESOURCES / 'image_random_row_quantiles.csv', delimiter=',')


@pytest.fixture
def image_random_col_quantiles() -> np.array:
    """Expected column quantiles (unmasked)"""
    return np.loadtxt(RESOURCES / 'image_random_col_quantiles.csv', delimiter=',')


@pytest.fixture
def image_random_aligned_rows() -> np.array:
    """Expected aligned rows (unmasked)."""
    df = pd.read_csv(RESOURCES / 'image_random_aligned_rows.csv.bz2', header=None)
    return df.to_numpy()


@pytest.fixture
def image_random_remove_x_y_tilt() -> np.array:
    """Expected removed tilt (unmasked)."""
    df = pd.read_csv(RESOURCES / 'image_random_remove_x_y_tilt.csv.bz2', header=None)
    return df.to_numpy()


@pytest.fixture
def image_random_mask() -> np.array:
    """Expected mask."""
    df = pd.read_csv(RESOURCES / 'image_random_mask.csv.bz2', header=None)
    return df.to_numpy()


@pytest.fixture
def image_random_row_quantiles_masked() -> np.array:
    """Expected row quantiles (masked)."""
    return np.loadtxt(RESOURCES / 'image_random_row_quantiles_masked.csv', delimiter=',')


@pytest.fixture
def image_random_col_quantiles_masked() -> np.array:
    """Expected column quantiles (masked)."""
    return np.loadtxt(RESOURCES / 'image_random_col_quantiles_masked.csv', delimiter=',')


@pytest.fixture
def minicircle() -> Bruker:
    """Load a file."""
    return load_scan(RESOURCES / 'minicircle.spm')


@pytest.fixture
def minicircle_filename() -> str:
    """Extract the filename"""
    return extract_img_name(RESOURCES / 'minicircle.spm')


@pytest.fixture
def minicircle_channel(minicircle) -> SPM_image:
    """Extract the image channel."""
    return extract_channel(minicircle, channel=CHANNEL)


@pytest.fixture
def minicircle_pixel_to_nm(minicircle_channel) -> float:
    """Extract the pixel to nanometer scaling."""
    return extract_pixel_to_nm_scaling(minicircle_channel)


@pytest.fixture
def minicircle_pixels(minicircle_channel):
    """Extract Pixels"""
    return extract_pixels(minicircle_channel)


@pytest.fixture
def minicircle_initial_align(minicircle_pixels: np.array) -> np.array:
    """Initial align on unmasked data."""
    return align_rows(minicircle_pixels, mask=None)


@pytest.fixture
def minicircle_initial_tilt_removal(minicircle_initial_align: np.array) -> np.array:
    """Initial x/y tilt removal on unmasked data."""
    return remove_x_y_tilt(minicircle_initial_align, mask=None)


@pytest.fixture
def minicircle_threshold(minicircle_initial_tilt_removal: np.array) -> float:
    """Calculate threshold."""
    return get_threshold(minicircle_initial_tilt_removal)


@pytest.fixture
def minicircle_mask(minicircle_initial_tilt_removal: np.array, minicircle_threshold: float) -> float:
    """Derive mask based on threshold."""
    return get_mask(minicircle_initial_tilt_removal, minicircle_threshold)


@pytest.fixture
def minicircle_masked_align(minicircle_initial_tilt_removal: np.array, minicircle_mask: np.array) -> np.array:
    """Secondary alignment using mask."""
    return align_rows(minicircle_initial_tilt_removal, mask=minicircle_mask)


@pytest.fixture
def minicircle_masked_tilt_removal(minicircle_masked_align: np.array, minicircle_mask: np.array) -> np.array:
    """Secondary x/y tilt removal using mask."""
    return remove_x_y_tilt(minicircle_masked_align, mask=minicircle_mask)


@pytest.fixture
def minicircle_zero_average_background(minicircle_masked_tilt_removal: np.array, minicircle_mask: np.array) -> np.array:
    """Zero average background"""
    return average_background(minicircle_masked_tilt_removal, minicircle_mask)


## Derive fixtures for grain finding
@pytest.fixture
def minicircle_grain_gaussian_filter(minicircle_zero_average_background: np.array, minicircle_pixel_to_nm: float,
                                     grain_config: dict) -> np.array:
    """Apply Gaussian filter."""
    return gaussian_filter(minicircle_zero_average_background,
                           gaussian_size=grain_config['gaussian_size'],
                           pixel_nm_scaling=minicircle_pixel_to_nm,
                           mode=grain_config['mode'])


@pytest.fixture
def minicircle_grain_boolean(minicircle_zero_average_background: np.array, minicircle_grain_gaussian_filter: np.array,
                             grain_config: dict) -> np.array:
    """Boolean mask."""
    threshold = get_threshold(minicircle_zero_average_background) * grain_config['threshold_multiplier']
    return get_mask(minicircle_grain_gaussian_filter, threshold=threshold)


@pytest.fixture
def minicircle_grain_clear_border(minicircle_grain_boolean: np.array) -> np.array:
    """Cleared borders."""
    return tidy_border(minicircle_grain_boolean)


@pytest.fixture
def minicircle_grain_labelled_all(minicircle_grain_clear_border: np.array, grain_config: dict) -> np.array:
    """Labelled regions."""
    return label_regions(minicircle_grain_clear_border, background=grain_config['background'])


@pytest.fixture
def minicircle_minimum_grain_size_pixels(minicircle_grain_labelled_all: np.array, grain_config: dict) -> float:
    """Minimum grain size."""
    return minimum_grain_size_pixels(image=minicircle_grain_labelled_all, background=grain_config['background'])


@pytest.fixture
def minicircle_grain_small_objects_removed(minicircle_grain_clear_border: np.array,
                                           minicircle_minimum_grain_size_pixels: float) -> np.array:
    # grain_config: dict) -> np.array:
    """Small objects removed."""
    return remove_objects(minicircle_grain_clear_border, minimum_grain_size_pixels=minicircle_minimum_grain_size_pixels)


@pytest.fixture
def minicircle_grain_labelled_post_removal(minicircle_grain_small_objects_removed: np.array,
                                           grain_config: dict) -> np.array:
    """Labelled regions."""
    return label_regions(minicircle_grain_small_objects_removed, background=grain_config['background'])


@pytest.fixture
def minicircle_grain_region_properties_post_removal(minicircle_grain_labelled_post_removal: np.array) -> np.array:
    """Region properties."""
    return region_properties(minicircle_grain_labelled_post_removal)


@pytest.fixture
def minicircle_grain_coloured(minicircle_grain_labelled_post_removal: np.array) -> np.array:
    """Coloured regions."""
    return colour_regions(minicircle_grain_labelled_post_removal)
