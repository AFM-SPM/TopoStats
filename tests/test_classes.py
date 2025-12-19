"""Tests of TopoStats classes."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
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
    convert_to_dict,
    prepare_data_for_df,
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
        "number of ordered coords : ()\n"
        "number of spline coords : None\n"
        "contour length : None\n"
        "end to end distance : None\n"
        "bounding box coords : None"
    )
    assert str(dummy_molecule) == expected


def test_molecule_stats_to_df(dummy_molecule: Molecule, snapshot) -> None:
    """Test the GrainCrop.stats_to_df() method."""
    assert isinstance(dummy_molecule.stats_to_df(), pd.DataFrame)
    assert dummy_molecule.stats_to_df().to_string() == snapshot


def test_ordered_trace_str(dummy_ordered_trace: OrderedTrace, capsys: pytest.CaptureFixture) -> None:
    """Test the OrderedTrace.str() method."""
    expected = (
        "number of molecules : 2\nnumber of images : 4\nwrithe : negative\npixel to nm scaling : 1.0\nerror : True"
    )
    assert str(dummy_ordered_trace) == expected


@pytest.mark.skip(reason="output is wrong only one row, there should be two, one for each molecule")
def test_ordered_trace_stats_to_df(dummy_ordered_trace: OrderedTrace, snapshot) -> None:
    """Test the GrainCrop.stats_to_df() method."""
    assert isinstance(dummy_ordered_trace.stats_to_df(), pd.DataFrame)
    assert dummy_ordered_trace.stats_to_df().to_string() == snapshot


def test_node_str(dummy_node: Node, capsys: pytest.CaptureFixture) -> None:
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


@pytest.mark.skip(reason="output is wrong only one row, there should be two, one for each molecule")
def test_node_stats_to_df(dummy_node: Node, snapshot) -> None:
    """Test the GrainCrop.stats_to_df() method."""
    assert isinstance(dummy_node.stats_to_df(), pd.DataFrame)
    assert dummy_node.stats_to_df().to_string() == snapshot


def test_matched_branch_str(dummy_matched_branch: MatchedBranch, capsys: pytest.CaptureFixture) -> None:
    """Test the MatchedBranch.__str__() method."""
    expected = (
        "number of coords : 4\n"
        "full width half maximum : 1.1\n"
        "full width half maximum maximums : [2.1, 3.4]\n"
        "full width half maximum peaks : [15.1, 89.1]\n"
        "angles : 143.163"
    )
    assert str(dummy_matched_branch) == expected


@pytest.mark.skip(reason="output is wrong only one row with nested dictionaries in columns, need to expand")
def test_matched_branch_stats_to_df(dummy_matched_branch: MatchedBranch, snapshot) -> None:
    """Test the GrainCrop.stats_to_df() method."""
    assert isinstance(dummy_matched_branch.stats_to_df(), pd.DataFrame)
    assert dummy_matched_branch.stats_to_df().to_string() == snapshot


def test_unmatched_branch_str(dummy_unmatched_branch: UnMatchedBranch, capsys: pytest.CaptureFixture) -> None:
    """Test the UnMatchedBranch.__str__() method."""
    expected = "angles : [143.163, 69.234, 12.465]"
    assert str(dummy_unmatched_branch) == expected


@pytest.mark.skip(
    reason="output is wrong only one row, do we need angles for unmatched branches? Probably to align with matched"
)
def test_unmatched_branch_stats_to_df(dummy_unmatched_branch: UnMatchedBranch, snapshot) -> None:
    """Test the GrainCrop.stats_to_df() method."""
    assert isinstance(dummy_unmatched_branch.stats_to_df(), pd.DataFrame)
    assert dummy_unmatched_branch.stats_to_df().to_string() == snapshot


def test_disordered_trace_str(dummy_disordered_trace: DisorderedTrace, capsys: pytest.CaptureFixture) -> None:
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


@pytest.mark.skip(reason="output is wrong only one row with nested dictionaries in columns, need to expand")
def test_disordered_trace_stats_to_df(dummy_disordered_trace: DisorderedTrace, snapshot) -> None:
    """Test the GrainCrop.stats_to_df() method."""
    assert isinstance(dummy_disordered_trace.stats_to_df(), pd.DataFrame)
    assert dummy_disordered_trace.stats_to_df().to_string() == snapshot


@pytest.mark.skip(reason="2025-10-15 - awaiting tests to be written")
def test_grain_crop_debug_locate_difference() -> None:
    """Test the GrainCrop.debug_locate_difference() method."""


def test_grain_crop_str(dummy_graincrop: GrainCrop, capsys: pytest.CaptureFixture) -> None:
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


def test_grain_crop_stats_to_df(dummy_graincrop: GrainCrop) -> None:
    """Test the GrainCrop.stats_to_df() method."""
    node_names = {k: v.__class__.__name__ for k, v in dummy_graincrop.nodes.items()}
    expected = {
        "padding": dummy_graincrop.padding,
        "image": dummy_graincrop.image,
        "mask": dummy_graincrop.mask,
        "bbox": dummy_graincrop.bbox,
        "pixel_to_nm_scaling": dummy_graincrop.pixel_to_nm_scaling,
        "thresholds": dummy_graincrop.thresholds,
        "threshold_method": dummy_graincrop.threshold_method,
        "filename": dummy_graincrop.filename,
        "height_profiles": dummy_graincrop.height_profiles,
        "stats": dummy_graincrop.stats,
        "skeleton": dummy_graincrop.skeleton,
        "disordered_trace": dummy_graincrop.disordered_trace.__class__.__name__,
        "nodes": node_names,
        "ordered_trace": dummy_graincrop.ordered_trace.__class__.__name__,
    }
    assert isinstance(dummy_graincrop.stats_to_df(), pd.DataFrame)
    assert dummy_graincrop.stats_to_df().to_string() == snapshot


@pytest.mark.parametrize(
    (
        "topostats_object",
        "grain_crops",
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
            rng.random((10, 10)),
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
            "tests/resources",
            id="rep_int v2.4.0",
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
    basename: str,
    request,
    capsys: pytest.CaptureFixture,
) -> None:
    """Test the TopoStats.__str__ method."""
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


@pytest.mark.skip(reason="output is wrong only one row with nested dictionaries in columns, need to expand")
@pytest.mark.parametrize(
    (
        "topostats_object",
        "grain_crops",
        "filename",
        "pixel_to_nm_scaling",
        "topostats_version",
        "img_path",
        "image",
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
            id="rep_int v2.4.0",
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
    request,
    capsys,
) -> None:
    """Test the TopoStats.__str__ method."""
    topostats_object = request.getfixturevalue(topostats_object)
    topostats_object.image = image
    grain_crops = request.getfixturevalue(grain_crops)
    expected = (
        f"number of grain crops : {len(grain_crops)}\n"
        f"filename : {filename}\n"
        f"pixel to nm scaling : {pixel_to_nm_scaling}\n"
        f"image shape (px) : {image.shape}\n"
        f"image path : {img_path}\n"
        f"TopoStats version : {topostats_version}"
    )
    assert str(topostats_object) == expected
    print(topostats_object)
    captured = capsys.readouterr()
    assert captured.out.strip() == expected


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
def test_topostats_stats_to_df(
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
    snapshot,
) -> None:
    """Test the TopoStats.stats_to_df() method."""
    topostats_object = request.getfixturevalue(topostats_object)
    topostats_object.image = image
    topostats_object.image_original = image_original
    topostats_object.full_mask_tensor = full_mask_tensor
    grain_crops = request.getfixturevalue(grain_crops)
    config = request.getfixturevalue(config)
    assert isinstance(topostats_object.stats_to_df(), pd.DataFrame)
    assert topostats_object.stats_to_df().to_string() == snapshot


@pytest.mark.parametrize(
    ("dummy_class"),
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
    dummy_class,
    snapshot,
    request,
) -> None:
    """Test converting each class to a dictionary using the convert_to_dict() method."""
    dummy_class = request.getfixturevalue(dummy_class)
    assert convert_to_dict(dummy_class) == snapshot


@pytest.mark.parametrize(
    ("stats_mapping", "topostats_object"),
    [
        pytest.param(
            "molecule_statistics",
            "topostats_rep_int_2_4_0",
            id="molecule_statistics",
        ),
        pytest.param(
            "grain_statistics",
            "topostats_rep_int_2_4_0",
            id="grain_statistics",
        ),
        pytest.param(
            "image_statistics",
            "topostats_rep_int_2_4_0",
            id="image_statistics",
        ),
        pytest.param(
            "branch_statistics",
            "topostats_rep_int_2_4_0",
            id="branch_statistics",
        ),
    ],
)
def test_prepare_data_for_df(
    stats_mapping: str,
    topostats_object: str,
    snapshot,
    request,
) -> None:
    """Test creating subset dictionaries for DataFrame creation using the prepare_data_for_df() method."""
    topostats_object = request.getfixturevalue(topostats_object)
    prepared_data = prepare_data_for_df(topostats_object, stats_mapping)
    assert prepared_data == snapshot
