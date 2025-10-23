"""Tests of TopoStats classes."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from topostats.classes import (
    DisorderedTrace,
    GrainCrop,
    GrainCropsDirection,
    ImageGrainCrops,
    MatchedBranch,
    Molecule,
    Node,
    OrderedTrace,
    TopoStats,
)
from topostats.io import dict_almost_equal

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"
GRAINCROP_DIR = RESOURCES / "graincrop"

# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments

SEED = 4092024
rng = np.random.default_rng(SEED)


def test_molecule_to_dict(dummy_molecule: Molecule) -> None:
    """Test the Node.node_to_dict() method."""
    expected = {
        "circular": dummy_molecule.circular,
        "topology": dummy_molecule.topology,
        "topology_flip": dummy_molecule.topology_flip,
        "ordered_coords": dummy_molecule.ordered_coords,
        "heights": dummy_molecule.heights,
        "distances": dummy_molecule.distances,
    }
    assert dict_almost_equal(dummy_molecule.molecule_to_dict(), expected)


def test_ordered_trace_to_dict(dummy_ordered_trace: OrderedTrace) -> None:
    """Test the OrderedTrace.ordered_trace_to_dict() method."""
    expected = {
        "tracing_stats": dummy_ordered_trace.tracing_stats,
        "grain_molstats": dummy_ordered_trace.grain_molstats,
        "ordered_trace_data": dummy_ordered_trace.ordered_trace_data,
        "molecules": dummy_ordered_trace.molecules,
        "writhe": dummy_ordered_trace.writhe,
        "pixel_to_nm_scaling": dummy_ordered_trace.pixel_to_nm_scaling,
        "images": dummy_ordered_trace.images,
        "error": dummy_ordered_trace.error,
    }
    assert dict_almost_equal(dummy_ordered_trace.ordered_trace_to_dict(), expected)


def test_node_to_dict(dummy_node: Node) -> None:
    """Test the Node.node_to_dict() method."""
    expected = {
        "error": dummy_node.error,
        "pixel_to_nm_scaling": dummy_node.pixel_to_nm_scaling,
        "branch_stats": dummy_node.branch_stats,
        "unmatched_branch_stats": dummy_node.unmatched_branch_stats,
        "node_coords": dummy_node.node_coords,
        "confidence": dummy_node.confidence,
        "reduced_node_area": dummy_node.reduced_node_area,
        "node_area_skeleton": dummy_node.node_area_skeleton,
        "node_branch_mask": dummy_node.node_branch_mask,
        "node_avg_mask": dummy_node.node_avg_mask,
    }
    assert dict_almost_equal(dummy_node.node_to_dict(), expected)


def test_matched_branch_to_dict(dummy_matched_branch: MatchedBranch) -> None:
    """Test the MatchedBranch.matched_branch_to_dict() method."""
    expected = {
        "ordered_coords": dummy_matched_branch.ordered_coords,
        "heights": dummy_matched_branch.heights,
        "distances": dummy_matched_branch.distances,
        "fwhm": dummy_matched_branch.fwhm,
        "angles": dummy_matched_branch.angles,
    }
    assert dict_almost_equal(dummy_matched_branch.matched_branch_to_dict(), expected)


def test_disordered_trace_to_dict(dummy_disordered_trace: DisorderedTrace) -> None:
    """Test the DisorderedTrace.disordered_trace_to_dict() method."""
    expected = {
        "images": dummy_disordered_trace.images,
        "grain_endpoints": dummy_disordered_trace.grain_endpoints,
        "grain_junctions": dummy_disordered_trace.grain_junctions,
        "total_branch_length": dummy_disordered_trace.total_branch_length,
        "grain_width_mean": dummy_disordered_trace.grain_width_mean,
    }
    assert dict_almost_equal(dummy_disordered_trace.disordered_trace_to_dict(), expected)


@pytest.mark.skip(reason="2025-10-15 - awaiting tests to be written")
def test_grain_crop_debug_locate_difference() -> None:
    """Test the GrainCrop.debug_locate_difference() method."""
    pass


def test_grain_crop_to_dict(dummy_graincrop: GrainCrop) -> None:
    """Test the GrainCrop.grain_crop_to_dict() method."""
    expected = {
        "padding": dummy_graincrop.padding,
        "image": dummy_graincrop.image,
        "mask": dummy_graincrop.mask,
        "bbox": dummy_graincrop.bbox,
        "pixel_to_nm_scaling": dummy_graincrop.pixel_to_nm_scaling,
        "thresholds": dummy_graincrop.thresholds,
        "filename": dummy_graincrop.filename,
        "stats": dummy_graincrop.stats,
        "height_profiles": dummy_graincrop.height_profiles,
        "skeleton": dummy_graincrop.skeleton,
        "disordered_trace": dummy_graincrop.disordered_trace,
        "nodes": dummy_graincrop.nodes,
        "ordered_trace": dummy_graincrop.ordered_trace,
        "threshold_method": dummy_graincrop.threshold_method,
    }
    assert dict_almost_equal(dummy_graincrop.grain_crop_to_dict(), expected)


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
        "img_path",
    ),
    [
        pytest.param(
            "topostats_catenanes_2_4_0",
            "imagegraincrops_catenanes",
            str(GRAINCROP_DIR),
            id="catenane v2.4.0",
        ),
        pytest.param(
            "topostats_rep_int_2_4_0",
            "imagegraincrops_rep_int",
            str(GRAINCROP_DIR),
            id="rep_int v2.4.0",
        ),
    ],
)
def test_topostats_to_dict(
    topostats_object: TopoStats,
    image_grain_crops: ImageGrainCrops,
    img_path: str,
    request,
) -> None:
    """Test conversion of TopoStats object to dictionary."""
    topostats_object = request.getfixturevalue(topostats_object)
    image_grain_crops = request.getfixturevalue(image_grain_crops)
    expected = {
        "image_grain_crops": image_grain_crops,
        "filename": topostats_object.filename,
        "pixel_to_nm_scaling": topostats_object.pixel_to_nm_scaling,
        "img_path": img_path,
        "image": topostats_object.image,
        "image_original": topostats_object.image_original,
        "full_mask_tensor": topostats_object.full_mask_tensor,
        "topostats_version": topostats_object.topostats_version,
        "config": topostats_object.config,
    }
    assert topostats_object.topostats_to_dict() == expected


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
        "config",
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
            "default_config",
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
            "default_config",
            id="tep_int v2.4.0",
        ),
    ],
)
def test_topostats_eq(
    topostats_object: str,
    image_grain_crops: str,
    filename: str,
    pixel_to_nm_scaling: float,
    topostats_version: str,
    img_path: str,
    image: npt.NDArray | None,
    image_original: npt.NDArray | None,
    config: str,
    request,
) -> None:
    """Test the TopoStats.__eq__ method."""
    topostats_object = request.getfixturevalue(topostats_object)
    topostats_object.image = image
    topostats_object.image_original = image_original
    image_grain_crops = request.getfixturevalue(image_grain_crops)
    config = request.getfixturevalue(config)
    expected = TopoStats(
        image_grain_crops=image_grain_crops,
        filename=filename,
        pixel_to_nm_scaling=pixel_to_nm_scaling,
        img_path=img_path,
        image=image,
        image_original=image_original,
        topostats_version=topostats_version,
        config=config,
    )
    assert topostats_object == expected
