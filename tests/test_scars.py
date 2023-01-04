"""Tests for the scars module."""
from pathlib import Path
import numpy as np

from topostats.scars import Scars

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"


def test_remove_scars(synthetic_scar_removal: Scars):
    """Test removal of scars"""

    scars_removed, scar_mask = synthetic_scar_removal.remove_scars()

    target_img = np.load(RESOURCES / "synthetic_scar_removal.npy")
    target_mask = np.load(RESOURCES / "synthetic_scar_removal_mask.npy")

    np.testing.assert_array_equal(scars_removed, target_img)
    np.testing.assert_array_equal(scar_mask, target_mask)
