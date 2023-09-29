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
    _get_mask,
    get_and_combine_directional_masks,
)


THRESHOLD_OPTIONS = {
    "otsu_threshold_multiplier": 1.7,
    "threshold_std_dev": {"below": [10.0, None], "above": [1.0, None]},
    "absolute": {"below": [-1.5, None], "above": [1.5, None]},
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
    ("image_name", "core_set", "title", "zrange"),
    [
        ("extracted_channel", False, "Raw Height", [0, 3]),
        ("z_threshed", True, "Height Thresholded", [0, 3]),
        ("grain_image", False, "", [0, 3]),  # non-binary image
        ("grain_mask", False, "", [None, None]),  # binary image
        ("grain_mask_image", False, "", [0, 3]),  # non-binary image
    ],
)
def test_update_plotting_config(
    process_scan_config: dict, image_name: str, core_set: bool, title: str, zrange: tuple
) -> None:
    """Ensure values are added to each image in plot_dict."""
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
    assert set(thresholds.keys()) == {"above", "below"}
    assert set(thresholds["above"].keys()) == {"minimum", "maximum"}
    assert thresholds["above"]["minimum"] == 0.8466799787547299
    assert np.isposinf(thresholds["above"]["maximum"]), "Maximum value should be infinite for otsu above threshold"
    assert thresholds["below"] is None


@pytest.mark.parametrize(
    ("config_std_dev_thresholds", "expected_value_thresholds"),
    [
        (
            {"below": [10.0, None], "above": [1.0, None]},
            {
                "below": {"minimum": -2.3866804917165663, "maximum": -np.Infinity},
                "above": {"minimum": 0.7886033762450778, "maximum": np.Infinity},
            },
        ),
        (
            {"below": None, "above": [None, 1.0]},
            {"below": None, "above": {"minimum": -np.Infinity, "maximum": 0.7886033762450778}},
        ),
    ],
)
def test_get_thresholds_stddev(
    image_random: np.ndarray, config_std_dev_thresholds: dict, expected_value_thresholds: dict
) -> None:
    """Test of get_thresholds() method with mean threshold."""
    THRESHOLD_OPTIONS["threshold_std_dev"] = config_std_dev_thresholds

    thresholds = get_thresholds(image=image_random, threshold_method="std_dev", **THRESHOLD_OPTIONS)
    assert isinstance(thresholds, dict)
    # assert thresholds == {"below": {"minimum": -2.3866804917165663, "maximum"}, "above": {0.7886033762450778}
    assert set(thresholds.keys()) == {"above", "below"}
    # Check the values of the thresholds. Annoyingly it's complicated to do since np.Infinity != np.Infinity.
    for direction in ["above", "below"]:
        if expected_value_thresholds[direction] is None:
            assert thresholds[direction] is None
        else:
            if np.isposinf(expected_value_thresholds[direction]["minimum"]):
                assert np.isposinf(thresholds[direction]["minimum"])
            if np.isneginf(expected_value_thresholds[direction]["minimum"]):
                assert np.isneginf(thresholds[direction]["minimum"])
            else:
                assert expected_value_thresholds[direction]["minimum"] == thresholds[direction]["minimum"]
            if np.isposinf(expected_value_thresholds[direction]["maximum"]):
                assert np.isposinf(thresholds[direction]["maximum"])
            if np.isneginf(expected_value_thresholds[direction]["maximum"]):
                assert np.isneginf(thresholds[direction]["maximum"])
            else:
                assert expected_value_thresholds[direction]["maximum"] == thresholds[direction]["maximum"]

    if expected_value_thresholds["above"] is None:
        assert thresholds["above"] is None

    with pytest.raises(TypeError):
        thresholds = get_thresholds(image=image_random, threshold_method="std_dev")


def test_get_thresholds_absolute(image_random: np.ndarray) -> None:
    """Test of get_thresholds() method with absolute threshold."""
    thresholds = get_thresholds(image=image_random, threshold_method="absolute", **THRESHOLD_OPTIONS)
    assert isinstance(thresholds, dict)
    assert thresholds["above"]["minimum"] == 1.5
    assert np.isposinf(thresholds["above"]["maximum"])
    assert thresholds["below"]["minimum"] == -1.5
    assert np.isneginf(thresholds["below"]["maximum"])


def test_get_thresholds_type_error(image_random: np.ndarray) -> None:
    """Test a TypeError is raised if a non-string value is passed to get_thresholds()."""
    with pytest.raises(TypeError):
        get_thresholds(image=image_random, threshold_method=6.4, **THRESHOLD_OPTIONS)


def test_get_thresholds_value_error(image_random: np.ndarray) -> None:
    """Test a ValueError is raised if an invalid value is passed to get_thresholds()."""
    with pytest.raises(ValueError):  # noqa: PT011
        get_thresholds(image=image_random, threshold_method="mean", **THRESHOLD_OPTIONS)


def test_create_empty_dataframe() -> None:
    """Test the empty dataframe is created correctly."""
    empty_df = create_empty_dataframe(ALL_STATISTICS_COLUMNS)

    assert empty_df.index.name == "molecule_number"
    assert "molecule_number" not in empty_df.columns
    assert empty_df.shape == (0, 26)
    assert {"image", "basename", "area"}.intersection(empty_df.columns)


@pytest.mark.parametrize(
    ("minimum_threshold", "maximum_threshold", "threshold_direction", "expected"),
    [
        (
            1.0,
            4.0,
            "above",
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1],
                    [1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            ),
        ),
        (
            -2.5,
            2.0,
            "above",
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1],
                    [1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            ),
        ),
        (
            4.0,
            np.Infinity,
            "above",
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1],
                ]
            ),
        ),
        (
            -np.Infinity,
            4.0,
            "above",
            np.array(
                [
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            ),
        ),
        (
            -1.0,
            -4.0,
            "below",
            np.array(
                [
                    [0, 0, 0, 1, 1],
                    [1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            ),
        ),
        (
            2.5,
            -2.0,
            "below",
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            ),
        ),
        (
            -4.0,
            -np.Infinity,
            "below",
            np.array(
                [
                    [1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            ),
        ),
        (
            np.Infinity,
            -4.0,
            "below",
            np.array(
                [
                    [0, 0, 0, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                ]
            ),
        ),
    ],
)
def test_get_mask(
    minimum_threshold: float,
    maximum_threshold: float,
    threshold_direction: str,
    expected: np.ndarray,
) -> None:
    """Test the _get_mask fuction of utils.py, ensuring that the correct values in images are masked."""
    image = np.array(
        [
            [-5.0, -4.5, -4.0, -3.5, -3.0],
            [-2.5, -2.0, -1.5, -1.0, -0.5],
            [+0.0, +0.5, +1.0, +1.5, +2.0],
            [+2.5, +3.0, +3.5, +4.0, +4.5],
            [+5.0, +5.5, +6.0, +7.5, +8.0],
        ]
    )

    thresholds = {"minimum": minimum_threshold, "maximum": maximum_threshold}

    result = _get_mask(image=image, thresholds=thresholds, threshold_direction=threshold_direction)

    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    ("thresholds", "expected"),
    [
        (
            {
                "above": {"minimum": 1.0, "maximum": 4.0},
                "below": {"minimum": -1.0, "maximum": -4.0},
            },
            np.array(
                [
                    [0, 0, 0, 1, 1],
                    [1, 1, 1, 0, 0],
                    [0, 0, 0, 1, 1],
                    [1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            ),
        ),
        (
            {"above": {"minimum": 1.0, "maximum": 4.0}, "below": None},
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1],
                    [1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            ),
        ),
        (
            {
                "above": None,
                "below": {"minimum": -1.0, "maximum": -4.0},
            },
            np.array(
                [
                    [0, 0, 0, 1, 1],
                    [1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            ),
        ),
    ],
)
def test_get_and_combine_directional_masks(thresholds: dict, expected: np.ndarray) -> None:
    """Test the get_and_combine_directional_masks function of utils.py.

    Ensures that the correct directional masks are added together and returned.
    """
    image = np.array(
        [
            [-5.0, -4.5, -4.0, -3.5, -3.0],
            [-2.5, -2.0, -1.5, -1.0, -0.5],
            [+0.0, +0.5, +1.0, +1.5, +2.0],
            [+2.5, +3.0, +3.5, +4.0, +4.5],
            [+5.0, +5.5, +6.0, +7.5, +8.0],
        ]
    )

    result = get_and_combine_directional_masks(image, thresholds=thresholds)

    np.testing.assert_array_equal(result, expected)
