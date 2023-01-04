"""Tests for the scars module."""
from pathlib import Path
import numpy as np
import pytest

from topostats.scars import Scars

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"


def test_remove_scars(synthetic_scar_removal: Scars):
    """Test removal of scars"""

    scars_removed, scar_mask = synthetic_scar_removal.remove_scars()

    target_img = np.load(RESOURCES / "test_scars_synthetic_remove_scars.npy")
    target_mask = np.load(RESOURCES / "test_scars_synthetic_remove_scars_mask.npy")

    np.testing.assert_array_equal(scars_removed, target_img)
    np.testing.assert_array_equal(scar_mask, target_mask)


@pytest.mark.parametrize(
    "potential_positive_scar, target, threshold_low, max_scar_width",
    [
        (
            np.array(
                [
                    [0],
                    [3],
                    [3],
                    [3],
                    [0],
                ]
            ),
            np.array(
                [
                    [0],
                    [3],
                    [3],
                    [3],
                    [0],
                ]
            ),
            2,
            3,
        ),
        (
            np.array(
                [
                    [0],
                    [3],
                    [2.1],
                    [3],
                    [0],
                ]
            ),
            np.array(
                [
                    [0],
                    [3],
                    [2.1],
                    [3],
                    [0],
                ]
            ),
            2,
            3,
        ),
        (
            np.array(
                [
                    [0],
                    [3],
                    [3],
                    [2.1],
                    [0],
                ]
            ),
            np.array(
                [
                    [0],
                    [3],
                    [3],
                    [2.1],
                    [0],
                ]
            ),
            2,
            3,
        ),
        (
            np.array(
                [
                    [0],
                    [2.1],
                    [3],
                    [3],
                    [0],
                ]
            ),
            np.array(
                [
                    [0],
                    [2.1],
                    [3],
                    [3],
                    [0],
                ]
            ),
            2,
            3,
        ),
        (
            np.array(
                [
                    [0],
                    [3],
                    [2],
                    [3],
                    [0],
                ]
            ),
            np.array(
                [
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                ]
            ),
            2,
            3,
        ),
        (
            np.array(
                [
                    [0],
                    [3],
                    [3],
                    [3],
                    [3],
                    [3],
                ]
            ),
            np.array(
                [
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                ]
            ),
            2,
            3,
        ),
        (
            np.array(
                [
                    [0],
                    [3],
                    [3],
                    [3],
                ]
            ),
            np.array(
                [
                    [0],
                    [0],
                    [0],
                    [0],
                ]
            ),
            2,
            3,
        ),
    ],
)
def test_mark_if_positive_scar(potential_positive_scar, target, threshold_low, max_scar_width):
    """Test the mark_if_positive_scar method of the Scars class."""

    marked = np.zeros(potential_positive_scar.shape)

    Scars.mark_if_positive_scar(
        row_col=(0, 0),
        stddev=1.0,
        img=potential_positive_scar,
        marked=marked,
        threshold_low=threshold_low,
        max_scar_width=max_scar_width,
    )

    np.testing.assert_array_equal(marked, target)


@pytest.mark.parametrize(
    "potential_negative_scar, target, threshold_low, max_scar_width",
    [
        (
            np.array(
                [
                    [4],
                    [1],
                    [1],
                    [1],
                    [4],
                ]
            ),
            np.array(
                [
                    [0],
                    [3],
                    [3],
                    [3],
                    [0],
                ]
            ),
            2,
            3,
        ),
        (
            np.array(
                [
                    [4],
                    [1],
                    [1.9],
                    [1],
                    [4],
                ]
            ),
            np.array(
                [
                    [0],
                    [3],
                    [2.1],
                    [3],
                    [0],
                ]
            ),
            2,
            3,
        ),
        (
            np.array(
                [
                    [4],
                    [1],
                    [1],
                    [1.9],
                    [4],
                ]
            ),
            np.array(
                [
                    [0],
                    [3],
                    [3],
                    [2.1],
                    [0],
                ]
            ),
            2,
            3,
        ),
        (
            np.array(
                [
                    [4],
                    [1.9],
                    [1],
                    [1],
                    [4],
                ]
            ),
            np.array(
                [
                    [0],
                    [2.1],
                    [3],
                    [3],
                    [0],
                ]
            ),
            2,
            3,
        ),
        (
            np.array(
                [
                    [4],
                    [1],
                    [2],
                    [1],
                    [4],
                ]
            ),
            np.array(
                [
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                ]
            ),
            2,
            3,
        ),
        (
            np.array(
                [
                    [4],
                    [1],
                    [1],
                    [1],
                    [1],
                    [4],
                ]
            ),
            np.array(
                [
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                ]
            ),
            2,
            3,
        ),
        (
            np.array(
                [
                    [4],
                    [1],
                    [1],
                    [1],
                ]
            ),
            np.array(
                [
                    [0],
                    [0],
                    [0],
                    [0],
                ]
            ),
            2,
            3,
        ),
    ],
)
def test_mark_if_negative_scar(potential_negative_scar, target, threshold_low, max_scar_width):
    """Test the mark_if_negative_scar method of the Scars class."""

    marked = np.zeros(potential_negative_scar.shape)

    Scars.mark_if_negative_scar(
        row_col=(0, 0),
        stddev=1.0,
        img=potential_negative_scar,
        marked=marked,
        threshold_low=threshold_low,
        max_scar_width=max_scar_width,
    )

    np.testing.assert_array_equal(marked, target)


def test_spread_scars():
    """Test the spread scars method of the Scars class."""

    marked_mask = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 1, 0, 0],
            [0, 2, 0, 0, 2, 0, 0],
            [0, 2, 1, 0, 2, 0, 0],
            [0, 2, 0, 1, 2, 0, 0],
            [0, 2, 1, 1, 0, 2, 0],
            [0, 2, 1, 1, 1, 0, 0],
            [0, 2, 0, 1, 1, 2, 0],
            [0, 1, 2, 1, 1, 2, 0],
            [0, 2, 1, 1, 1, 2, 0],
        ]
    )

    target = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 1, 0, 0],
            [0, 2, 0, 0, 2, 0, 0],
            [0, 2, 2, 0, 2, 0, 0],
            [0, 2, 0, 2, 2, 0, 0],
            [0, 2, 2, 2, 0, 2, 0],
            [0, 2, 2, 2, 2, 0, 0],
            [0, 2, 0, 2, 2, 2, 0],
            [0, 2, 2, 2, 2, 2, 0],
            [0, 2, 2, 2, 2, 2, 0],
        ]
    )

    Scars.spread_scars(marked=marked_mask, threshold_low=1, threshold_high=2)

    np.testing.assert_array_equal(marked_mask, target)


def test_remove_short_scars():
    """Test the remove_short_scars method of the Scars class."""

    mask = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 2, 2, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 2, 2, 2, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 2, 2, 2, 2, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 2, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )

    target = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )

    Scars.remove_short_scars(mask, threshold_high=2, min_scar_length=3)

    np.testing.assert_array_equal(mask, target)


def test_mark_scars(synthetic_scars_image, synthetic_marked_scars):
    """Test the mark_scars method of the Scars class."""

    marked = Scars.mark_scars(
        img=synthetic_scars_image,
        direction="positive",
        threshold_low=1.5,
        threshold_high=1.8,
        max_scar_width=2,
        min_scar_length=1,
    )

    np.testing.assert_array_equal(marked, synthetic_marked_scars)


def test_remove_marked_scars(synthetic_scars_image, synthetic_marked_scars):
    """Test the remove_marked_scars method of the Scars class."""

    Scars.remove_marked_scars(synthetic_scars_image, synthetic_marked_scars)

    target = np.load(RESOURCES / "test_scars_synthetic_remove_marked_scars.npy")

    np.testing.assert_array_equal(synthetic_scars_image, target)
