"""Test utils."""
from pathlib import Path

import numpy as np
import pytest

from topostats.utils import convert_path, find_images, update_config, get_thresholds


THRESHOLD_OPTIONS = {"otsu_threshold_multiplier": 1.7, "deviation_from_mean": 1, "absolute": (-1.5, 1.5)}


def test_convert_path(tmpdir) -> None:
    """Test path conversion."""
    test_dir = str(tmpdir)
    converted_path = convert_path(test_dir)

    assert isinstance(converted_path, Path)
    assert tmpdir == converted_path


def test_find_images() -> None:
    """Test finding images"""
    found_images = find_images(base_dir="tests/", file_ext=".spm")

    assert isinstance(found_images, list)
    assert len(found_images) == 1
    assert isinstance(found_images[0], Path)
    assert "minicircle.spm" in str(found_images[0])


def test_update_config(caplog) -> None:
    """Test updating configuration."""
    sample_config = {"a": 1, "b": 2, "c": "something", "base_dir": "here", "output_dir": "there"}
    new_values = {"c": "something new"}
    updated_config = update_config(sample_config, new_values)

    assert isinstance(updated_config, dict)
    assert "Updated config config[c] : something > something new" in caplog.text
    assert updated_config["c"] == "something new"


def test_get_thresholds_otsu(image_random: np.ndarray) -> None:
    """Test of get_thresholds() method otsu threshold."""
    thresholds = get_thresholds(image=image_random, threshold_method="otsu", **THRESHOLD_OPTIONS)

    assert isinstance(thresholds, dict)
    assert thresholds == {"upper": 0.8466799787547299}


def test_get_thresholds_stddev(image_random: np.ndarray) -> None:
    """Test of get_thresholds() method with mean threshold."""
    thresholds = get_thresholds(image=image_random, threshold_method="std_dev", **THRESHOLD_OPTIONS)
    assert isinstance(thresholds, dict)
    assert thresholds == {"upper": 0.7886033762450778, "lower": 0.21127903661568803}

    with pytest.raises(TypeError):
        thresholds = get_thresholds(image=image_random, threshold_method="std_dev", deviation_from_mean=None)


def test_get_thresholds_absolute(image_random: np.ndarray) -> None:
    """Test of get_thresholds() method with absolute threshold."""
    thresholds = get_thresholds(image=image_random, threshold_method="absolute", **THRESHOLD_OPTIONS)
    assert isinstance(thresholds, dict)
    assert thresholds == {"upper": 1.5, "lower": -1.5}


def test_get_thresholds_type_error(image_random: np.ndarray) -> None:
    """Test a TypeError is raised if a non-string value is passed to get_thresholds()"""
    with pytest.raises(TypeError):
        get_thresholds(image=image_random, threshold_method=6.4, **THRESHOLD_OPTIONS)


def test_get_thresholds_value_error(image_random: np.ndarray) -> None:
    """Test a ValueError is raised if an invalid value is passed to get_thresholds()"""
    with pytest.raises(ValueError):
        get_thresholds(image=image_random, threshold_method="mean", **THRESHOLD_OPTIONS)
