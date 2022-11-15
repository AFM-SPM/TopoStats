"""Test utils."""
from pathlib import Path

import numpy as np
import pytest
import pandas as pd

from topostats.utils import convert_path, find_images, get_out_path, update_config, get_thresholds, folder_grainstats


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


def test_find_images() -> None:
    """Test finding images"""
    found_images = find_images(base_dir="tests/", file_ext=".spm")

    assert isinstance(found_images, list)
    assert len(found_images) == 1
    assert isinstance(found_images[0], Path)
    assert "minicircle.spm" in str(found_images[0])


@pytest.mark.parametrize(
    "image_path, base_dir, output_dir, expected",
    [
        # Full absolute path for test.spm
        (
            Path("/heres/some/random/path/images/test.spm"),
            Path("/heres/some/random/path"),
            Path("output/here"),
            Path("output/here/images/"),
        ),
        # Relative path for test.spm, this raises a ValueError
        (
            Path("images/test.spm"),
            Path("/heres/some/random/path"),
            Path("output/here"),
            Path("output/here/images/"),
        ),
        # No file with suffix
        (Path("images/"), Path("/heres/some/random/path"), Path("output/here"), Path("output/here/images/")),
        # Absolute path for output
        (
            Path("/heres/some/random/path/images/test.spm"),
            Path("/heres/some/random/path"),
            Path("/an/absolute/path"),
            Path("/an/absolute/path/images"),
        ),
        # Absolute path for output and relative path for image
        (
            Path("images/test.spm"),
            Path("/heres/some/random/path"),
            Path("/an/absolute/path"),
            Path("/an/absolute/path/images"),
        ),
    ],
)
def test_get_out_path(image_path: Path, base_dir: Path, output_dir: Path, expected: Path) -> None:
    """Test output directories"""
    image_path = Path(image_path)
    base_dir = Path(base_dir)
    output_dir = Path(output_dir)
    out_path = get_out_path(image_path, base_dir, output_dir)
    assert isinstance(out_path, Path)
    assert out_path == expected


def test_get_out_path_attributeerror() -> None:
    """Test get_out_path() raises AttribteError when passed a string instead of a Path() for image_path."""
    with pytest.raises(AttributeError):
        get_out_path(
            image_path="images/test.spm", base_dir=Path("/heres/some/random/path"), output_dir=Path("output/here")
        )


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
    assert thresholds == {"lower": -2.3866804917165663, "upper": 0.7886033762450778}

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


def test_folder_grainstats(tmp_path: Path, minicircle_tracestats: pd.DataFrame) -> None:
    """Test a folder-wide grainstats file is made"""
    input_path = tmp_path / "minicircle"
    minicircle_tracestats["Basename"] = input_path / "subfolder"
    true_out_path = tmp_path / "subfolder" / "Processed"
    Path.mkdir(true_out_path, parents=True)
    folder_grainstats(tmp_path, input_path, minicircle_tracestats)
    assert Path(true_out_path / "folder_grainstats.csv").exists()
