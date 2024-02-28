"""Test the utils module."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from topostats.utils import (
    ALL_STATISTICS_COLUMNS,
    bound_padded_coordinates_to_image,
    convert_path,
    create_empty_dataframe,
    get_thresholds,
    update_config,
    update_plotting_config,
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
    ("image_name", "core_set", "title", "zrange"),
    [
        pytest.param("extracted_channel", False, "Raw Height", [0, 3], id="Non-binary image, not in core, with title"),
        pytest.param("z_threshed", True, "Height Thresholded", [0, 3], id="Non-binary image, in core, with title"),
        pytest.param(
            "grain_mask", False, "", [None, None], id="Binary image, not in core, set no title"
        ),  # binary image
        pytest.param(
            "grain_mask_image", False, "", [0, 3], id="Non-binary image, not in core, set no title"
        ),  # non-binary image
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


@pytest.mark.parametrize(
    ("plotting_config", "target_config"),
    [
        pytest.param(
            {
                "run": True,
                "savefig_dpi": None,
                "pixel_interpolation": None,
                "plot_dict": {
                    "extracted_channel": {
                        "filename": "00-raw_heightmap",
                        "image_type": "non-binary",
                        "savefig_dpi": 100,
                    }
                },
            },
            {
                "run": True,
                "savefig_dpi": None,
                "pixel_interpolation": None,
                "plot_dict": {
                    "extracted_channel": {
                        "filename": "00-raw_heightmap",
                        "image_type": "non-binary",
                        "savefig_dpi": 100,
                        "pixel_interpolation": None,
                    }
                },
            },
            id="DPI None in main, extracted channel DPI should stay at 100",
        ),
        pytest.param(
            {
                "run": True,
                "savefig_dpi": 600,
                "pixel_interpolation": None,
                "plot_dict": {
                    "extracted_channel": {
                        "filename": "00-raw_heightmap",
                        "image_type": "non-binary",
                        "savefig_dpi": 100,
                    }
                },
            },
            {
                "run": True,
                "savefig_dpi": 600,
                "pixel_interpolation": None,
                "plot_dict": {
                    "extracted_channel": {
                        "filename": "00-raw_heightmap",
                        "image_type": "non-binary",
                        "savefig_dpi": 600,
                        "pixel_interpolation": None,
                    }
                },
            },
            id="DPI 600 in main, extracted channel DPI should update to 600",
        ),
    ],
)
def test_udpate_plotting_config_adding_required_options(plotting_config: dict, target_config: dict, caplog) -> None:
    """Only updates plotting_dict parameters from parent plotting config if value is not None."""
    update_plotting_config(plotting_config)
    assert plotting_config == target_config
    if plotting_config["savefig_dpi"] == 600:
        assert "100 > 600" in caplog.text


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
    ("image", "padding", "expected"),
    [
        (
            np.asarray(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            ),
            1,  # Padding
            (2, 2),
        ),
        (
            np.asarray(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            ),
            2,  # Padding
            (2, 2),
        ),
        # Padding is 3 which means the range of values this is intended to be used with would be the coordinates (2, 2),
        # minus the padding which would give (-1, -1), and so instead we shift the coordinates over so that the padding
        # will start at (0, 0)
        (
            np.asarray(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            ),
            3,  # Padding
            (3, 3),
        ),
        # With a padding of 2 the resulting coordinates to start the padding would be outside of the image range, so
        # again we want to have this point shifted so that padding starts at (0, 0)
        (
            np.asarray(
                [
                    [0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            ),
            2,  # Padding
            (2, 2),
        ),
        (
            np.asarray(
                [
                    [1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            ),
            2,  # Padding
            (2, 2),
        ),
        # We now check the other corners
        (
            np.asarray(
                [
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0],
                    [0, 1, 0, 1, 0],
                    [1, 0, 0, 0, 1],
                ]
            ),
            2,  # Padding
            (2, 2),
        ),
        # And for completeness check the edges
        (
            np.asarray(
                [
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [1, 1, 0, 1, 1],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                ]
            ),
            2,  # Padding
            (2, 2),
        ),
        # Check with larger padding (have to split these)
        (
            np.asarray(
                [
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            ),
            3,  # Padding
            (3, 3),
        ),
        (
            np.asarray(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                ]
            ),
            3,  # Padding
            (2, 2),
        ),
    ],
)
def test_bound_padded_coordinates_to_image(image: npt.NDArray, padding: int, expected: tuple) -> None:
    """Test that padding points does not exceed the image dimensions."""
    coordinates = np.argwhere(image == 1)
    for coordinate in coordinates:
        padded_coords = bound_padded_coordinates_to_image(coordinate, padding, image.shape)
        assert padded_coords == expected
