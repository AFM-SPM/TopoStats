"""Tests for the main module components defined in ''__init__.py''."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

import topostats
from topostats.grains import ImageGrainCrops

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"
GRAINCROP_DIR = RESOURCES / "graincrop"

# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments


@pytest.mark.parametrize(
    (
        "topostats_object",
        "image_grain_crops",
        "filename",
        "pixel_to_nm_scaling",
        "topostats_version",
        "img_path",
        "image",
        "image_original",
    ),
    [
        pytest.param(
            "topostats_catenanes_2_4_0",
            "imagegraincrops_catenanes",
            "example_catenanes.spm",
            0.488,
            "2.4.0",
            str(GRAINCROP_DIR),
            None,
            None,
            id="catenane v2.4.0",
        ),
        pytest.param(
            "topostats_rep_int_2_4_0",
            "imagegraincrops_rep_int",
            "example_rep_int.spm",
            0.488,
            "2.4.0",
            str(GRAINCROP_DIR),
            None,
            None,
            id="catenanes v2.4.0",
        ),
    ],
)
def test_topostats_to_dict(
    topostats_object: topostats.TopoStats,
    image_grain_crops: ImageGrainCrops,
    filename: str,
    pixel_to_nm_scaling: float,
    topostats_version: str,
    img_path: str,
    image: npt.NDArray | None,
    image_original: npt.NDArray | None,
    request,
) -> None:
    """Test conversion of TopoStats object to dictionary."""
    topostats_object = request.getfixturevalue(topostats_object)
    image_grain_crops = request.getfixturevalue(image_grain_crops)
    expected = {
        "image_grain_crops": image_grain_crops,
        "filename": filename,
        "pixel_to_nm_scaling": pixel_to_nm_scaling,
        "topostats_version": topostats_version,
        "img_path": Path(img_path),
        "image": image,
        "image_original": image_original,
    }
    np.testing.assert_array_equal(topostats_object.topostats_to_dict(), expected)
