"""Test utils."""
from pathlib import Path

import numpy as np
import pytest

from topostats.io import read_yaml
from topostats.utils import convert_path, update_config, get_thresholds, update_plotting_config


THRESHOLD_OPTIONS = {
    "otsu_threshold_multiplier": 1.7,
    "threshold_std_dev": {"lower": 10.0, "upper": 1.0},
    "absolute": {"lower": -1.5, "upper": 1.5},
}


def test_convert_path(tmp_path: Path) -> None:
    """Test path conversion."""
    test_dir = str(tmp_path)
    converted_path = convert_path(test_dir)

    assert isinstance(converted_path, Path)
    assert tmp_path == converted_path


def test_update_config(caplog) -> None:
    """Test updating configuration."""
    sample_config = {"a": 1, "b": 2, "c": "something", "base_dir": "here", "output_dir": "there"}
    new_values = {"c": "something new"}
    updated_config = update_config(sample_config, new_values)

    assert isinstance(updated_config, dict)
    assert "Updated config config[c] : something > something new" in caplog.text
    assert updated_config["c"] == "something new"


def test_update_plotting_config(process_scan_config: dict) -> None:
    """Test that the plotting config is correctly updated."""

    expected = read_yaml("./tests/resources/updated_plotting_dict.yaml")

    process_scan_config["plotting"] = update_plotting_config(process_scan_config["plotting"])

    assert process_scan_config == expected


def test_get_thresholds_otsu(image_random: np.ndarray) -> None:
    """Test of get_thresholds() method otsu threshold."""
    thresholds = get_thresholds(image=image_random, threshold_method="otsu", **THRESHOLD_OPTIONS)

    assert isinstance(thresholds, dict)
    assert thresholds == {"upper": 0.8466799787547299}


def test_get_thresholds_stddev(image_random: np.ndarray) -> None:
    """Test of get_thresholds() method with mean threshold."""
    thresholds = get_thresholds(image=image_random, threshold_method="std_dev", **THRESHOLD_OPTIONS)
    assert isinstance(thresholds, dict)
    assert thresholds == {"lower": -2.3866804917165663, "upper": 0.7886033762450778}

    with pytest.raises(TypeError):
        thresholds = get_thresholds(image=image_random, threshold_method="std_dev")


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
