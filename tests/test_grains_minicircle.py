"""Tests for Find Grains."""

import numpy as np
import pytest

from topostats.grains import Grains


def test_threshold_otsu(minicircle_grain_threshold_otsu: Grains) -> None:
    """Test threshold calculation."""
    assert isinstance(minicircle_grain_threshold_otsu.thresholds, dict)
    assert minicircle_grain_threshold_otsu.thresholds["above"] == pytest.approx([0.7724327490179487])


def test_threshold_stddev(minicircle_grain_threshold_stddev: Grains) -> None:
    """Test threshold calculation."""
    assert isinstance(minicircle_grain_threshold_stddev.thresholds, dict)
    assert minicircle_grain_threshold_stddev.thresholds == pytest.approx(
        {"below": [-5.901722094162684], "above": [0.9495125861734497]}
    )


def test_threshold_abs(minicircle_grain_threshold_abs: Grains) -> None:
    """Test threshold calculation."""
    assert isinstance(minicircle_grain_threshold_abs.thresholds, dict)
    assert minicircle_grain_threshold_abs.thresholds == {"above": [1.0], "below": [-1.0]}


def test_mask_minicircle(minicircle_grain_traditional_thresholding: Grains) -> None:
    """Test creation of boolean array for clearing borders."""
    assert isinstance(minicircle_grain_traditional_thresholding.mask_images["above"]["thresholded_grains"], np.ndarray)
    assert minicircle_grain_traditional_thresholding.mask_images["above"]["thresholded_grains"].shape == (64, 64, 2)
    assert minicircle_grain_traditional_thresholding.mask_images["above"]["thresholded_grains"][:, :, 1].sum() == 670


def test_clear_border(minicircle_grain_clear_border: Grains) -> None:
    """Test creation of boolean array for clearing borders."""
    assert isinstance(minicircle_grain_clear_border.mask_images["above"]["tidied_border"], np.ndarray)
    assert minicircle_grain_clear_border.mask_images["above"]["tidied_border"].shape == (64, 64, 2)
    assert minicircle_grain_clear_border.mask_images["above"]["tidied_border"][:, :, 1].sum() == 525


def test_remove_objects_too_small_to_process(minicircle_grain_remove_objects_too_small_to_process: Grains) -> None:
    """Test creation of boolean array for clearing borders."""
    assert isinstance(
        minicircle_grain_remove_objects_too_small_to_process.mask_images["above"][
            "removed_objects_too_small_to_process"
        ],
        np.ndarray,
    )
    assert minicircle_grain_remove_objects_too_small_to_process.mask_images["above"][
        "removed_objects_too_small_to_process"
    ].shape == (
        64,
        64,
        2,
    )
    assert (
        minicircle_grain_remove_objects_too_small_to_process.mask_images["above"][
            "removed_objects_too_small_to_process"
        ][:, :, 1].sum()
        == 511
    )


def test_area_thresholding(minicircle_grain_area_thresholding: Grains) -> None:
    """Test area thresholding."""
    assert isinstance(minicircle_grain_area_thresholding.mask_images["above"]["area_thresholded"], np.ndarray)
    assert minicircle_grain_area_thresholding.mask_images["above"]["area_thresholded"].shape == (64, 64, 2)
    assert minicircle_grain_area_thresholding.mask_images["above"]["area_thresholded"][:, :, 1].sum() == 511
