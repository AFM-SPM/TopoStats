"""Test utils."""
from pathlib import Path

import numpy as np
import pytest

from topostats.utils import (
    convert_path,
    update_config,
    get_thresholds,
    update_plotting_config,
    create_empty_dataframe,
    ALL_STATISTICS_COLUMNS,
)


THRESHOLD_OPTIONS = {
    "otsu_threshold_multiplier": 1.7,
    "threshold_std_dev": {"below": 10.0, "above": 1.0},
    "absolute": {"below": -1.5, "above": 1.5},
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


@pytest.mark.parametrize(
    "image_name, core_set, title, image_type, zrange",
    [
        ("extracted_channel", False, "Raw Height", "non-binary", [0, 3]),
        ("z_threshed", True, "Height Thresholded", "non-binary", [0, 3]),
        ("grain_image", False, "", [0, 3]),  # non-binary image
        ("grain_mask", False, "", [None, None]),  # binary image
        ("grain_mask_image", False, "", [0, 3]),  # non-binary image
    ],
)
def test_update_plotting_config(
    process_scan_config: dict, image_name: str, core_set: bool, title: str, zrange: tuple
) -> None:
    """Test that update_plotting_config correctly fills in values
    for each image in the plotting dictionary plot_dict."""
    process_scan_config["plotting"] = update_plotting_config(process_scan_config["plotting"])
    assert process_scan_config["plotting"]["plot_dict"][image_name]["core_set"] == core_set
    # Only check titles for images that have titles. grain_image, grain_mask, grain_mask_image don't
    # have titles since they're created dynamically.
    if title in ["extracted_channel", "z_threshed"]:
        assert process_scan_config["plotting"]["plot_dict"][image_name]["title"] == title
    # Ensure that both types (binary, non-binary) of image have the correct z-ranges
    # ([None, None] for binary, user defined for non-binary)
    assert process_scan_config["plotting"]["plot_dict"][image_name]["zrange"] == zrange


def test_get_thresholds_otsu(image_random: np.ndarray) -> None:
    """Test of get_thresholds() method otsu threshold."""
    thresholds = get_thresholds(image=image_random, threshold_method="otsu", **THRESHOLD_OPTIONS)

    assert isinstance(thresholds, dict)
    assert thresholds == {"above": 0.8466799787547299}


def test_get_thresholds_stddev(image_random: np.ndarray) -> None:
    """Test of get_thresholds() method with mean threshold."""
    thresholds = get_thresholds(image=image_random, threshold_method="std_dev", **THRESHOLD_OPTIONS)
    assert isinstance(thresholds, dict)
    assert thresholds == {"below": -2.3866804917165663, "above": 0.7886033762450778}

    with pytest.raises(TypeError):
        thresholds = get_thresholds(image=image_random, threshold_method="std_dev")


def test_get_thresholds_absolute(image_random: np.ndarray) -> None:
    """Test of get_thresholds() method with absolute threshold."""
    thresholds = get_thresholds(image=image_random, threshold_method="absolute", **THRESHOLD_OPTIONS)
    assert isinstance(thresholds, dict)
    assert thresholds == {"above": 1.5, "below": -1.5}


def test_get_thresholds_type_error(image_random: np.ndarray) -> None:
    """Test a TypeError is raised if a non-string value is passed to get_thresholds()"""
    with pytest.raises(TypeError):
        get_thresholds(image=image_random, threshold_method=6.4, **THRESHOLD_OPTIONS)


def test_get_thresholds_value_error(image_random: np.ndarray) -> None:
    """Test a ValueError is raised if an invalid value is passed to get_thresholds()"""
    with pytest.raises(ValueError):
        get_thresholds(image=image_random, threshold_method="mean", **THRESHOLD_OPTIONS)


def test_create_empty_dataframe() -> None:
    """Test the empty dataframe is created correctly."""
    empty_df = create_empty_dataframe(ALL_STATISTICS_COLUMNS)

    assert empty_df.index.name == "molecule_number"
    assert "molecule_number" not in empty_df.columns
    assert empty_df.shape == (0, 26)
    assert {"image", "basename", "area"}.intersection(empty_df.columns)
