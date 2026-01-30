"""Tests of TopoStats classes."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from syrupy.matchers import path_type

from topostats.classes import (
    DisorderedTrace,
    GrainCrop,
    MatchedBranch,
    Molecule,
    Node,
    OrderedTrace,
    TopoStats,
    UnMatchedBranch,
    convert_to_dict,
)

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"
GRAINCROP_DIR = RESOURCES / "graincrop"

# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments

SEED = 4092024
rng = np.random.default_rng(SEED)


# pylint: disable=implicit-str-concat
def test_molecule_str(dummy_molecule: Molecule) -> None:
    """Test the Molecule.__str__() method."""
    expected = (
        "circular : True\n"
        "topology : a\n"
        "topology flip : maybe\n"
        "number of ordered coords : 4\n"
        "number of spline coords : 4\n"
        "contour length : 1.023e-07\n"
        "end to end distance : 3.456e-08\n"
        "bounding box coords : (1, 2, 3, 4)"
    )
    assert str(dummy_molecule) == expected


def test_molecule_collate_molecule_statistics(dummy_molecule: Molecule, snapshot) -> None:
    """Test the Molecule.collate_molecule_statistics() method."""
    molecule_statistics = dummy_molecule.collate_molecule_statistics()
    assert isinstance(molecule_statistics, dict)
    assert molecule_statistics == snapshot(matcher=path_type(types=(float,), replacer=lambda data, _: round(data, 8)))


def test_ordered_trace_str(dummy_ordered_trace: OrderedTrace) -> None:
    """Test the OrderedTrace.str() method."""
    expected = (
        "number of molecules : 2\nnumber of images : 4\nwrithe : negative\npixel to nm scaling : 1.0\nerror : True"
    )
    assert str(dummy_ordered_trace) == expected


def test_node_str(dummy_node: Node) -> None:
    """Test the Node.__str__() method."""
    expected = (
        "error : False\n"
        "pixel to nm scaling (nm) : 1.0\n"
        "number of matched branches : 2\n"
        "number of unmatched branches : 1\n"
        "number of coords : 2\n"
        "confidence : 0.987654\n"
        "reduced node area : [10.98765432]"
    )
    assert str(dummy_node) == expected


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


def test_unmatched_branch_str(dummy_unmatched_branch: UnMatchedBranch) -> None:
    """Test the UnMatchedBranch.__str__() method."""
    expected = "angles : [143.163, 69.234, 12.465]"
    assert str(dummy_unmatched_branch) == expected


def test_disordered_trace_str(dummy_disordered_trace: DisorderedTrace) -> None:
    """Test the DisorderedTrace.__str__() method."""
    expected = (
        "generated images : pruned_skeleton, skeleton, smoothed_mask, branch_indexes, branch_types\n"
        "grain endpoints : 2\n"
        "grain junctions : 3\n"
        "total branch length (nm) : 12.3456789\n"
        "mean grain width (nm) : 3.14152\n"
        "number of branches : 1"
    )
    assert str(dummy_disordered_trace) == expected


@pytest.mark.skip(reason="2025-10-15 - awaiting tests to be written")
def test_grain_crop_debug_locate_difference() -> None:
    """Test the GrainCrop.debug_locate_difference() method."""


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
        "topostats_object_fixture",
        "grain_crops_fixture",
        "filename",
        "pixel_to_nm_scaling",
        "topostats_version",
        "img_path",
        "image",
        "basename",
    ),
    [
        pytest.param(
            "topostats_catenanes_2_4_0",
            "graincrops_catenanes",
            "example_catenanes.spm",
            0.488,
            "2.4.0",
            str(GRAINCROP_DIR),
            rng.random((400, 400)),
            "tests/resources",
            id="catenane v2.4.0",
        ),
        pytest.param(
            "topostats_rep_int_2_4_0",
            "graincrops_rep_int",
            "example_rep_int.spm",
            0.488,
            "2.4.0",
            str(GRAINCROP_DIR),
            rng.random((350, 350)),
            "tests/resources",
            id="rep_int v2.4.0",
        ),
    ],
)
def test_topostats_str(
    topostats_object_fixture: str,
    grain_crops_fixture: str,
    filename: str,
    pixel_to_nm_scaling: float,
    topostats_version: str,
    img_path: str,
    image: npt.NDArray | None,
    basename: str,
    request,
) -> None:
    """Test the TopoStats.__str__ method."""
    topostats_object = request.getfixturevalue(topostats_object_fixture)
    grain_crops = request.getfixturevalue(grain_crops_fixture)
    expected = (
        f"number of grain crops : {len(grain_crops)}\n"
        f"filename : {filename}\n"
        f"pixel to nm scaling : {pixel_to_nm_scaling}\n"
        f"image shape (px) : {image.shape}\n"
        f"image path : {img_path}\n"
        f"TopoStats version : {topostats_version}\n"
        f"Basename : {basename}"
    )
    assert str(topostats_object) == expected


topostats_test_data = [
    pytest.param(
        "topostats_catenanes_2_4_0",
        "graincrops_catenanes",
        "example_catenanes.spm",
        0.488,
        "2.4.0",
        str(GRAINCROP_DIR),
        rng.random((10, 10)),
        rng.random((10, 10)),
        rng.random((10, 10, 2)),
        "default_config",
        "tests/resources",
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
        rng.random((10, 10, 2)),
        "default_config",
        "tests/resources",
        id="rep_int v2.4.0",
    ),
]


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
        "full_mask_tensor",
        "config",
        "basename",
    ),
    topostats_test_data,
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
    full_mask_tensor: npt.NDArray | None,
    config: str,
    basename: str,
    request,
) -> None:
    """Test the TopoStats.__eq__ method."""
    topostats_object = request.getfixturevalue(topostats_object)
    topostats_object.image = image
    topostats_object.image_original = image_original
    topostats_object.full_mask_tensor = full_mask_tensor
    grain_crops = request.getfixturevalue(grain_crops)
    config = request.getfixturevalue(config)
    expected = TopoStats(
        grain_crops=grain_crops,
        filename=filename,
        pixel_to_nm_scaling=pixel_to_nm_scaling,
        img_path=img_path,
        image=image,
        image_original=image_original,
        topostats_version=topostats_version,
        full_mask_tensor=full_mask_tensor,
        config=config,
        basename=basename,
    )
    assert topostats_object == expected


@pytest.mark.parametrize(
    ("dummy_class_fixture"),
    [
        pytest.param(
            "dummy_molecule",
            id="molecule",
        ),
        pytest.param(
            "dummy_ordered_trace",
            id="ordered-trace",
        ),
        pytest.param(
            "dummy_node",
            id="node",
        ),
        pytest.param(
            "dummy_unmatched_branch",
            id="unmatched-branch",
        ),
        pytest.param(
            "dummy_matched_branch",
            id="matched-branch",
        ),
        pytest.param(
            "dummy_disordered_trace",
            id="disordered-trace",
        ),
        pytest.param(
            "dummy_graincrop",
            id="graincrop",
        ),
        pytest.param(
            "topostats_catenanes_2_4_0",
            id="topostats_catenanes",
        ),
        pytest.param(
            "topostats_rep_int_2_4_0",
            id="topostats_rep_int",
        ),
    ],
)
def test_convert_to_dict(
    dummy_class_fixture: str,
    snapshot,
    request,
) -> None:
    """Test converting each class to a dictionary using the convert_to_dict() method."""
    dummy_class = request.getfixturevalue(dummy_class_fixture)
    # If we have a `TopoStats` object we need to set a fixed 'img_path' attribute which changes based on system
    if dummy_class_fixture in ("topostats_catenanes_2_4_0", "topostats_rep_int_2_4_0"):
        dummy_class.img_path = "/path/to/file/"
    assert convert_to_dict(dummy_class) == snapshot(
        matcher=path_type(types=(float,), replacer=lambda data, _: round(data, 8))
    )
