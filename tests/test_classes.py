"""Tests of TopoStats classes."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from topostats.classes import (
    DisorderedTrace,
    GrainCrop,
    MatchedBranch,
    Molecule,
    Node,
    OrderedTrace,
    TopoStats,
    UnMatchedBranch,
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
        "splined_coords": dummy_molecule.splined_coords,
        "contour_length": dummy_molecule.contour_length,
        "end_to_end_distance": dummy_molecule.end_to_end_distance,
        "heights": dummy_molecule.heights,
        "distances": dummy_molecule.distances,
        "curvature_stats": dummy_molecule.curvature_stats,
        "bbox": dummy_molecule.bbox,
    }
    assert dict_almost_equal(dummy_molecule.molecule_to_dict(), expected)


# pylint: disable=implicit-str-concat
def test_molecule_str(dummy_molecule: Molecule) -> None:
    """Test the Molecule.__str__() method."""
    expected = (
        "circular : True\n"
        "topology : a\n"
        "topology flip : maybe\n"
        "number of ordered coords : ()\n"
        "number of spline coords : None\n"
        "contour length : None\n"
        "end to end distance : None\n"
        "bounding box coords : None"
    )
    assert str(dummy_molecule) == expected


def test_ordered_trace_to_dict(dummy_ordered_trace: OrderedTrace) -> None:
    """Test the OrderedTrace.ordered_trace_to_dict() method."""
    expected = {
        "tracing_stats": dummy_ordered_trace.tracing_stats,
        "grain_molstats": dummy_ordered_trace.grain_molstats,
        "molecule_data": dummy_ordered_trace.molecule_data,
        "molecules": dummy_ordered_trace.molecules,
        "writhe": dummy_ordered_trace.writhe,
        "pixel_to_nm_scaling": dummy_ordered_trace.pixel_to_nm_scaling,
        "images": dummy_ordered_trace.images,
        "error": dummy_ordered_trace.error,
    }
    assert dict_almost_equal(dummy_ordered_trace.ordered_trace_to_dict(), expected)


def test_ordered_trace_str(dummy_ordered_trace: OrderedTrace) -> None:
    """Test the OrderedTrace.str() method."""
    expected = (
        "number of molecules : 2\nnumber of images : 4\nwrithe : negative\npixel to nm scaling : 1.0\nerror : True"
    )
    assert str(dummy_ordered_trace) == expected


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
        "writhe": dummy_node.writhe,
    }
    assert dict_almost_equal(dummy_node.node_to_dict(), expected)


def test_node_str(dummy_node: Node) -> None:
    """Test the Node.__str__() method."""
    expected = (
        "error : False\n"
        "pixel to nm scaling (nm) : 1.0\n"
        "number of matched branches : 2\n"
        "number of unmatched branches : 1\n"
        "number of coords : 2\n"
        "confidence : 0.987654\n"
        "reduced node area : 10.987654321"
    )
    assert str(dummy_node) == expected


def test_matched_branch_to_dict(dummy_matched_branch: MatchedBranch) -> None:
    """Test the MatchedBranch.matched_branch_to_dict() method."""
    expected = {
        "ordered_coords": dummy_matched_branch.ordered_coords,
        "heights": dummy_matched_branch.heights,
        "distances": dummy_matched_branch.distances,
        "fwhm": dummy_matched_branch.fwhm,
        "fwhm_half_maxs": dummy_matched_branch.fwhm_half_maxs,
        "fwhm_peaks": dummy_matched_branch.fwhm_peaks,
        "angles": dummy_matched_branch.angles,
    }
    assert dict_almost_equal(dummy_matched_branch.matched_branch_to_dict(), expected)


def test_matched_branch_str(dummy_matched_branch: MatchedBranch) -> None:
    """Test the MatchedBranch.__str__() method."""
    expected = (
        "number of coords : 4\n"
        "full width half maximum : 1.1\n"
        "full width half maximum maximums : [2.1, 3.4]\n"
        "full width half maximum peaks : [15.1, 89.1]\n"
        "angles : 143.163"
    )
    assert str(dummy_matched_branch) == expected


def test_unmatched_branch_to_dict(dummy_unmatched_branch: UnMatchedBranch) -> None:
    """Test the UnMatchedBranch.unmatched_branch_to_dict() method."""
    expected = {
        "angles": dummy_unmatched_branch.angles,
    }
    assert dict_almost_equal(dummy_unmatched_branch.unmatched_branch_to_dict(), expected)


def test_unmatched_branch_str(dummy_unmatched_branch: UnMatchedBranch) -> None:
    """Test the UnMatchedBranch.__str__() method."""
    expected = "angles : [143.163, 69.234, 12.465]"
    assert str(dummy_unmatched_branch) == expected


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


def test_disordered_trace_str(dummy_disordered_trace: DisorderedTrace) -> None:
    """Test the DisorderedTrace.__str__() method."""
    expected = (
        "generated images : pruned_skeleton, skeleton, smoothed_mask, branch_indexes, branch_types\n"
        "grain endpoints : 2\n"
        "grain junctions : 3\n"
        "total branch length (nm) : 12.3456789\n"
        "mean grain width (nm) : 3.14152"
    )
    assert str(dummy_disordered_trace) == expected


@pytest.mark.skip(reason="2025-10-15 - awaiting tests to be written")
def test_grain_crop_debug_locate_difference() -> None:
    """Test the GrainCrop.debug_locate_difference() method."""


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


def test_grain_crop_str(dummy_graincrop: GrainCrop) -> None:
    """Test the GrainCrop.__str__() method."""
    expected = (
        "filename : dummy\n"
        "image shape : (10, 10)\n"
        "skeleton shape : (10, 10)\n"
        "mask shape : (10, 10, 2)\n"
        "padding : 2\n"
        "thresholds : (1, 2)\n"
        "threshold method : None\n"
        "bounding box coords : (1, 1, 11, 11)\n"
        "pixel to nm scaling : 1.0\n"
        "number of nodes : 2"
    )
    assert str(dummy_graincrop) == expected


@pytest.mark.parametrize(
    (
        "topostats_object",
        "img_path",
    ),
    [
        pytest.param(
            "topostats_catenanes_2_4_0",
            str(GRAINCROP_DIR),
            id="catenane v2.4.0",
        ),
        pytest.param(
            "topostats_rep_int_2_4_0",
            str(GRAINCROP_DIR),
            id="rep_int v2.4.0",
        ),
    ],
)
def test_topostats_to_dict(
    topostats_object: TopoStats,
    img_path: str,
    request,
) -> None:
    """Test conversion of TopoStats object to dictionary."""
    topostats_object = request.getfixturevalue(topostats_object)
    expected = {
        "filename": topostats_object.filename,
        "grain_crops": topostats_object.grain_crops,
        "pixel_to_nm_scaling": topostats_object.pixel_to_nm_scaling,
        "img_path": img_path,
        "image": topostats_object.image,
        "image_original": topostats_object.image_original,
        "full_mask_tensor": topostats_object.full_mask_tensor,
        "topostats_version": topostats_object.topostats_version,
        "config": topostats_object.config,
        "full_image_plots": topostats_object.full_image_plots,
    }
    assert topostats_object.topostats_to_dict() == expected


# Needs updating to switch to grain_crop rather than image_grain_crop
@pytest.mark.parametrize(
    (
        "topostats_object",
        "grain_crops",
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
            "graincrops_catenanes",
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
            "graincrops_rep_int",
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
    grain_crops: str,
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
    topostats_object: TopoStats = request.getfixturevalue(topostats_object)
    topostats_object.image: npt.NDArray = image
    topostats_object.image_original: npt.NDArray = image_original
    grain_crops: dict[int:GrainCrop] = request.getfixturevalue(grain_crops)
    config = request.getfixturevalue(config)
    expected = TopoStats(
        grain_crops=grain_crops,
        filename=filename,
        pixel_to_nm_scaling=pixel_to_nm_scaling,
        img_path=img_path,
        image=image,
        image_original=image_original,
        topostats_version=topostats_version,
        config=config,
    )
    assert topostats_object == expected


@pytest.mark.parametrize(
    (
        "topostats_object",
        "grain_crops",
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
            "graincrops_catenanes",
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
            "graincrops_rep_int",
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
def test_topostats_str(
    topostats_object: str,
    grain_crops: str,
    filename: str,
    pixel_to_nm_scaling: float,
    topostats_version: str,
    img_path: str,
    image: npt.NDArray | None,
    image_original: npt.NDArray | None,
    config: str,
    request,
) -> None:
    """Test the TopoStats.__str__ method."""
    topostats_object = request.getfixturevalue(topostats_object)
    topostats_object.image = image
    topostats_object.image_original = image_original
    grain_crops = request.getfixturevalue(grain_crops)
    config = request.getfixturevalue(config)
    expected = (
        f"number of grain crops : {len(grain_crops)}\n"
        f"filename : {filename}\n"
        f"pixel to nm scaling : {pixel_to_nm_scaling}\n"
        f"image shape (px) : {image.shape}\n"
        f"image path : {img_path}\n"
        f"TopoStats version : {topostats_version}"
    )
    assert str(topostats_object) == expected
