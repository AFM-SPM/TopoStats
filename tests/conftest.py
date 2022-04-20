"""Fixtures for testing"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from pySPM.SPM import SPM_image
from pySPM.Bruker import Bruker
from skimage.filters import threshold_otsu
from skimage import filters as skimage_filters
from skimage import segmentation as skimage_segmentation
from skimage import measure as skimage_measure
from skimage import morphology as skimage_morphology
from skimage import color as skimage_color

from topostats.filters import *
from topostats.filters import load_scan, extract_channel, extract_pixels
from topostats.plottingfuncs import plot_and_save

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / 'tests' / 'resources'

RNG = np.random.default_rng(seed=1000)
SMALL_ARRAY_SIZE = (10, 10)
THRESHOLD = 0.5
CHANNEL = 'Height'


@pytest.fixture
def grain_config() -> dict:
    return {
        'gaussian_size': 2,
        'dx': 1,
        'upper_height_threshold_rms_multiplier': 1,
        'lower_threshold': 1.7,
        'minimum_grain_size': 800,
        'background': 0
    }


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
    df = pd.read_csv(RESOURCES / 'image_random_mask.csv.bz2', header=None)
    return df.to_numpy()


@pytest.fixture
def image_random_row_quantiles_masked() -> np.array:
    return np.loadtxt(RESOURCES / 'image_random_row_quantiles_masked.csv',
                      delimiter=',')


@pytest.fixture
def image_random_col_quantiles_masked() -> np.array:
    return np.loadtxt(RESOURCES / 'image_random_col_quantiles_masked.csv',
                      delimiter=',')


@pytest.fixture
def minicircle() -> Bruker:
    """Load a file."""
    return load_scan(RESOURCES / 'minicircle.spm')


@pytest.fixture
def minicircle_channel(minicircle) -> SPM_image:
    """Extract the image channel."""
    return extract_channel(minicircle, channel=CHANNEL)


@pytest.fixture
def minicircle_pixels(minicircle_channel):
    """Extract Pixels"""
    return extract_pixels(minicircle_channel)


@pytest.fixture
def minicircle_initial_align(minicircle_pixels: np.array) -> np.array:
    """Initial align on unmasked data."""
    return align_rows(minicircle_pixels, mask=None)


@pytest.fixture
def minicircle_initial_tilt_removal(
        minicircle_initial_align: np.array) -> np.array:
    """Initial x/y tilt removal on unmasked data."""
    return remove_x_y_tilt(minicircle_initial_align, mask=None)


@pytest.fixture
def minicircle_threshold(minicircle_initial_tilt_removal: np.array) -> float:
    """Calculate threshold."""
    return get_threshold(minicircle_initial_tilt_removal)


@pytest.fixture
def minicircle_mask(minicircle_initial_tilt_removal: np.array,
                    minicircle_threshold: float) -> float:
    """Derive mask based on threshold."""
    return get_mask(minicircle_initial_tilt_removal, minicircle_threshold)


@pytest.fixture
def minicircle_masked_align(minicircle_initial_tilt_removal: np.array,
                            minicircle_mask: np.array) -> np.array:
    """Secondary alignment using mask."""
    return align_rows(minicircle_initial_tilt_removal, mask=minicircle_mask)


@pytest.fixture
def minicircle_masked_tilt_removal(minicircle_masked_align: np.array,
                                   minicircle_mask: np.array) -> np.array:
    """Secondary x/y tilt removal using mask."""
    return remove_x_y_tilt(minicircle_masked_align, mask=minicircle_mask)


@pytest.fixture
def minicircle_zero_average_background(
        minicircle_masked_tilt_removal: np.array,
        minicircle_mask: np.array) -> np.array:
    """Zero average background"""
    return average_background(minicircle_masked_tilt_removal, minicircle_mask)


## Derive fixtures for grain finding
@pytest.fixture
def minicircle_gaussian_filter(minicircle_zero_average_background: np.array,
                               grain_config: dict) -> np.array:
    """Apply Gaussian filter."""
    return gaussian_filter(minicircle_masked_tilt_removal)


@pytest.fixture
def minicircle_flattened_boolean(minicircle_masked_tilt_removal: np.array,
                                 grain_config: dict) -> np.array:
    return boolean_image(minicircle_flattened, grain_config['lower_threshold'])


@pytest.fixture
def minicircle_flattened_tidy_border(
        minicircle_flattened_boolean: np.array) -> np.array:
    border_tidied = tidy_border(minicircle_flattened_boolean)
    plot_and_save(
        border_tidied,
        BASE_DIR / 'tmp' / 'check' / '09_data_boolean_border_cleared.png')
    return border_tidied


@pytest.fixture
def minicircle_flattened_small_objects_culled(
        minicircle_flattened_tidy_border: np.array,
        grain_config: dict) -> np.array:
    small_objects_culled = remove_small_objects(
        minicircle_flattened_tidy_border, grain_config['minimum_grain_size'],
        grain_config['dx'])
    plot_and_save(
        small_objects_culled,
        BASE_DIR / 'tmp' / 'check' / '10_data_cull_small_objects.png')
    return small_objects_culled


@pytest.fixture
def minicircle_flattened_labelled(
        minicircle_flattened_small_objects_culled: np.array,
        grain_config: dict) -> np.array:
    labelled = label_regions(minicircle_flattened_small_objects_culled,
                             background=grain_config['background'])
    plot_and_save(
        labelled,
        BASE_DIR / 'tmp' / 'check' / '11_data_boolean_labelled_image.png')
    return labelled


@pytest.fixture
def minicircle_flattened_coloured(
        minicircle_flattened_labelled: np.array) -> np.array:
    labelled = label_regions(minicircle_flattened_labelled)
    plot_and_save(
        labelled, BASE_DIR / 'tmp' / 'check' /
        '12_data_boolean_color_labelled_image.png')
    return labelled


@pytest.fixture
def minicircle_flattened_region_properties(
        minicircle_flattened_labelled: np.array,
        grain_config: dict) -> np.array:
    labelled = region_properties(minicircle_flattened_labelled)
    # plot_and_save(labelled, BASE_DIR / 'tmp' / 'check' / '12a_data_boolean_region_properties.png')
    return labelled
