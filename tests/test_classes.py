"""Tests of TopoStats classes."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from topostats.classes import DisorderedTrace, GrainCrop, GrainCropsDirection, ImageGrainCrops, TopoStats
from topostats.io import dict_almost_equal

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"
GRAINCROP_DIR = RESOURCES / "graincrop"

# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments

SEED = 4092024
rng = np.random.default_rng(SEED)


@pytest.mark.skip(reason="Need to generate a dummy disordered_trace")
def test_disordered_trace_to_dict(dummy_disordered_trace: DisorderedTrace) -> None:
    """Test the DisorderedTrace.disordered_trace_to_dict() method."""
    expected = {
        "images": dummy_disordered_trace.images,
        "grain_endpoints": dummy_disordered_trace.grain_endpoints,
        "grain_junctions": dummy_disordered_trace.grain_junctions,
        "total_branch_length_nm": dummy_disordered_trace.total_branch_length_nm,
        "grain_width_mean_nm": dummy_disordered_trace.grain_width_mean_nm,
    }
    assert dict_almost_equal(dummy_disordered_trace.disordered_trace_to_dict(), expected)


@pytest.mark.xfail(reason="Awaiting adding attributes for skeleton, height_profile, disordered traces, nodes.")
def test_grain_crop_to_dict(dummy_graincrop: GrainCrop) -> None:
    """Test the GrainCrop.grain_crop_to_dict() method."""
    expected = {
        "image": dummy_graincrop.image,
        "mask": dummy_graincrop.mask,
        "padding": dummy_graincrop.padding,
        "bbox": dummy_graincrop.bbox,
        "pixel_to_nm_scaling": dummy_graincrop.pixel_to_nm_scaling,
        "filename": dummy_graincrop.filename,
        "stats": dummy_graincrop.stats,
        "height_profiles": dummy_graincrop.height_profiles,
    }
    np.testing.assert_array_equal(dummy_graincrop.grain_crop_to_dict(), expected)


def test_grain_crop_direction_to_dict(dummy_graincropsdirection: GrainCropsDirection) -> None:
    """Test the GrainCropDirection.grain_crop_direction_to_dict() method."""
    expected = {
        "crops": dummy_graincropsdirection.crops,
        "full_mask_tensor": dummy_graincropsdirection.full_mask_tensor,
    }
    assert dict_almost_equal(dummy_graincropsdirection.grain_crops_direction_to_dict(), expected)


def test_image_grain_crop_to_dict(dummy_graincropsdirection: GrainCropsDirection) -> None:
    """Test the GrainCropDirection.grain_crop_direction_to_dict() method."""
    dummy_image_grain_crop = ImageGrainCrops(above=dummy_graincropsdirection, below=dummy_graincropsdirection)
    expected = {
        "above": dummy_graincropsdirection,
        "below": dummy_graincropsdirection,
    }
    assert dict_almost_equal(dummy_image_grain_crop.image_grain_crops_to_dict(), expected)


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
            id="rep_int v2.4.0",
        ),
    ],
)
def test_topostats_to_dict(
    topostats_object: TopoStats,
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
            rng.random((10, 10)),
            rng.random((10, 10)),
            id="catenane v2.4.0",
        ),
        pytest.param(
            "topostats_rep_int_2_4_0",
            "imagegraincrops_rep_int",
            "example_rep_int.spm",
            0.488,
            "2.4.0",
            str(GRAINCROP_DIR),
            rng.random((10, 10)),
            rng.random((10, 10)),
            id="tep_int v2.4.0",
        ),
    ],
)
def test_topostats_eq(
    topostats_object: TopoStats,
    image_grain_crops: ImageGrainCrops,
    filename: str,
    pixel_to_nm_scaling: float,
    topostats_version: str,
    img_path: str,
    image: npt.NDArray | None,
    image_original: npt.NDArray | None,
    request,
) -> None:
    """Test the TopoStats.__eq__ method."""
    topostats_object = request.getfixturevalue(topostats_object)
    topostats_object.image = image
    topostats_object.image_original = image_original
    image_grain_crops = request.getfixturevalue(image_grain_crops)
    expected = TopoStats(
        image_grain_crops, filename, pixel_to_nm_scaling, img_path, image, image_original, topostats_version
    )
    assert topostats_object == expected
