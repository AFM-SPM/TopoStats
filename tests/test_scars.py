"""Tests for the scars module."""
from pathlib import Path
import numpy as np

from topostats.scars import Scars

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"


def test_interpolation():
    """Test interpolation"""


def test_detect_scars():
    """Test detection of scars"""


def test_remove_scars():
    """Test removal of scars"""

    # Fetch synthetic image
    synthetic_image = np.load(RESOURCES / "synthetic_scar_image.npy")

    scars_removed = Scars(
        img=synthetic_image,
        removal_iterations=2,
        threshold_low=1.5,
        threshold_high=1.8,
        max_scar_width=2,
        min_scar_length=1,
    ).remove_scars()

    target = np.load(RESOURCES / "synthetic_scar_removal.npy")
    np.testing.assert_array_equal(scars_removed, target)
