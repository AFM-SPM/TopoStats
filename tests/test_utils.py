"""Test utils."""
from pathlib import Path

import numpy as np

from topostats.utils import convert_path, find_images, update_config, get_threshold, get_mask


def test_convert_path(tmpdir):
    """Test path conversion."""
    test_dir = str(tmpdir)
    converted_path = convert_path(test_dir)

    assert isinstance(converted_path, Path)
    assert tmpdir == converted_path


def test_find_images():
    """Test finding images"""
    found_images = find_images(base_dir='tests/', file_ext='.spm')

    assert isinstance(found_images, list)
    assert len(found_images) == 1
    assert isinstance(found_images[0], Path)
    assert 'minicircle.spm' in str(found_images[0])


def test_update_config(caplog):
    """Test updating configuration."""
    SAMPLE_CONFIG = {'a': 1, 'b': 2, 'c': 'something', 'base_dir': 'here', 'output_dir': 'there'}
    NEW_VALUES = {'c': 'something new'}
    updated_config = update_config(SAMPLE_CONFIG, NEW_VALUES)

    assert isinstance(updated_config, dict)
    assert 'Updated config config[c] : something > something new' in caplog.text
    assert updated_config['c'] == 'something new'


def test_get_threshold(image_random: np.array):
    """Test calculation of threshold."""
    threshold = get_threshold(image_random)
    expected_threshold = 0.4980470463263117

    assert threshold == expected_threshold


def test_get_mask(image_random: np.array, image_random_mask: np.array):
    """Test generation of mask"""
    threshold = get_threshold(image_random)
    mask = get_mask(image_random, threshold)

    np.testing.assert_array_equal(mask, image_random_mask)
