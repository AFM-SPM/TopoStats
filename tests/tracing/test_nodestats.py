# Disable ruff 301 - pickle loading is unsafe, but we don't care for tests.
# ruff: noqa: S301
"""Test the nodestats module."""

import pickle
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from pytest_lazyfixture import lazy_fixture

# pylint: disable=import-error
# pylint: disable=no-name-in-module
from topostats.io import dict_almost_equal
from topostats.tracing.nodestats import nodeStats, nodestats_image

BASE_DIR = Path.cwd()
GENERAL_RESOURCES = BASE_DIR / "tests" / "resources"
DISORDERED_TRACING_RESOURCES = GENERAL_RESOURCES / "tracing" / "disordered_tracing"
NODESTATS_RESOURCES = BASE_DIR / "tests" / "resources" / "tracing" / "nodestats"

# from topostats.tracing.nodestats import nodeStats

# pylint: disable=unnecessary-pass
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-many-lines


# @pytest.mark.parametrize()
def test_get_node_stats() -> None:
    """Test of get_node_stats() method of nodeStats class."""
    pass


def test_check_node_errorless() -> None:
    """Test of check_node_errorless() method of nodeStats class."""
    pass


def test_skeleton_image_to_graph() -> None:
    """Test of skeleton_image_to_graph() method of nodeStats class."""
    pass


def test_graph_to_skeleton_image() -> None:
    """Test of graph_to_skeleton_image() method of nodeStats class."""
    pass


def test_tidy_branches() -> None:
    """Test of tidy_branches() method of nodeStats class."""
    pass


def test_keep_biggest_object() -> None:
    """Test of keep_biggest_object() method of nodeStats class."""
    pass


def test_connect_close_nodes() -> None:
    """Test of connect_close_nodes() method of nodeStats class."""
    pass


def test_highlight_node_centres() -> None:
    """Test of highlight_node_centres() method of nodeStats class."""
    pass


def test_connect_extended_nodes() -> None:
    """Test of connect_extended_nodes() method of nodeStats class."""
    pass


@pytest.mark.parametrize(
    ("connected_nodes", "expected_nodes"),
    [
        pytest.param(
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 3, 1, 1, 1, 1, 1, 1, 1, 3, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
            id="theta_grain",
        ),
        pytest.param(
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0, 0, 3, 0, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0, 0, 3, 0, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
            id="figure_8",
        ),
    ],
)
def test_connect_extended_nodes_nearest(
    connected_nodes: npt.NDArray[np.number], expected_nodes: npt.NDArray[np.number]
) -> None:
    """Test of connect_extended_nodes_nearest() method of nodeStats class.

    Needs a test for theta topology and figure 8.
    """
    nodestats = nodeStats(
        filename="dummy",
        image=np.array([[0, 0, 0], [0, 1.5, 0], [0, 0, 0]]),
        mask=np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
        smoothed_mask=np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
        skeleton=connected_nodes.astype(bool),
        px_2_nm=np.float64(1.0),
        n_grain=0,
        node_joining_length=0.0,
        node_extend_dist=14.0,
        branch_pairing_length=20.0,
        pair_odd_branches=True,
    )
    nodestats.whole_skel_graph = nodestats.skeleton_image_to_graph(nodestats.skeleton)
    result = nodestats.connect_extended_nodes_nearest(connected_nodes, node_extend_dist=8.0)

    assert np.array_equal(result, expected_nodes)


def test_find_branch_starts() -> None:
    """Test of find_branch_starts() method of nodeStats class."""
    pass


# Create nodestats class using the cats image - will allow running the code for diagnostics
def test_analyse_nodes(
    nodestats_catenane: nodeStats,
) -> None:
    """Test of analyse_nodes() method of nodeStats class."""
    nodestats_catenane.analyse_nodes(max_branch_length=20)

    node_dict_result = nodestats_catenane.node_dicts
    image_dict_result = nodestats_catenane.image_dict

    # Debugging
    # Save the results to overwrite expected results
    # with Path(RESOURCES / "nodestats_analyse_nodes_catenane_node_dict.pkl").open("wb") as f:
    #     pickle.dump(node_dict_result, f)

    # with Path(RESOURCES / "nodestats_analyse_nodes_catenane_image_dict.pkl").open("wb") as f:
    #     pickle.dump(image_dict_result, f)

    # np.save(
    #     RESOURCES / "nodestats_analyse_nodes_catenane_all_connected_nodes.npy", nodestats_catenane.all_connected_nodes
    # )

    # Load the nodestats catenane node dict from pickle
    with Path(GENERAL_RESOURCES / "nodestats_analyse_nodes_catenane_node_dict.pkl").open("rb") as f:
        expected_nodestats_catenane_node_dict = pickle.load(f)

    # Load the nodestats catenane image dict from pickle
    with Path(GENERAL_RESOURCES / "nodestats_analyse_nodes_catenane_image_dict.pkl").open("rb") as f:
        expected_nodestats_catenane_image_dict = pickle.load(f)

    # Load the nodestats catenane all connected nodes from pickle
    with Path(GENERAL_RESOURCES / "nodestats_analyse_nodes_catenane_all_connected_nodes.npy").open("rb") as f:
        expected_nodestats_catenane_all_connected_nodes = np.load(f)

    np.testing.assert_equal(node_dict_result, expected_nodestats_catenane_node_dict)
    np.testing.assert_equal(image_dict_result, expected_nodestats_catenane_image_dict)
    np.testing.assert_array_equal(
        nodestats_catenane.all_connected_nodes, expected_nodestats_catenane_all_connected_nodes
    )


@pytest.mark.parametrize(
    (
        "branch_under_over_order",
        "matched_branches_filename",
        "masked_image_filename",
        "branch_start_coords",
        "ordered_branches_filename",
        "pairs",
        "average_trace_advised",
        "image_shape",
        "expected_branch_image_filename",
        "expected_average_image_filename",
    ),
    [
        pytest.param(
            np.array([0, 1]),
            "catenane_node_0_matched_branches_analyse_node_branches.pkl",
            "catenane_node_0_masked_image.pkl",
            np.array([np.array([278, 353]), np.array([279, 351]), np.array([281, 352]), np.array([281, 354])]),
            "catenane_node_0_ordered_branches.pkl",
            np.array([(1, 3), (2, 0)]),
            True,
            (755, 621),
            "catenane_node_0_branch_image.npy",
            "catenane_node_0_avg_image.npy",
        )
    ],
)
def test_add_branches_to_labelled_image(
    branch_under_over_order: npt.NDArray[np.int32],
    matched_branches_filename: str,
    masked_image_filename: str,
    branch_start_coords: npt.NDArray[np.int32],
    ordered_branches_filename: str,
    pairs: npt.NDArray[np.int32],
    average_trace_advised: bool,
    image_shape: tuple[int, int],
    expected_branch_image_filename: str,
    expected_average_image_filename: str,
) -> None:
    """Test of add_branches_to_labelled_image() method of nodeStats class."""
    # Load the matched branches
    with Path(GENERAL_RESOURCES / f"{matched_branches_filename}").open("rb") as f:
        matched_branches: dict[int, dict[str, npt.NDArray[np.number]]] = pickle.load(f)

    # Load the masked image
    with Path(GENERAL_RESOURCES / f"{masked_image_filename}").open("rb") as f:
        masked_image: dict[int, dict[str, npt.NDArray[np.bool_]]] = pickle.load(f)

    # Load the ordered branches
    with Path(GENERAL_RESOURCES / f"{ordered_branches_filename}").open("rb") as f:
        ordered_branches: list[npt.NDArray[np.int32]] = pickle.load(f)

    # Load the branch image
    expected_branch_image: npt.NDArray[np.int32] = np.load(GENERAL_RESOURCES / expected_branch_image_filename)

    # Load the average image
    expected_average_image: npt.NDArray[np.float64] = np.load(GENERAL_RESOURCES / expected_average_image_filename)

    result_branch_image, result_average_image = nodeStats.add_branches_to_labelled_image(
        branch_under_over_order=branch_under_over_order,
        matched_branches=matched_branches,
        masked_image=masked_image,
        branch_start_coords=branch_start_coords,
        ordered_branches=ordered_branches,
        pairs=pairs,
        average_trace_advised=average_trace_advised,
        image_shape=image_shape,
    )

    np.testing.assert_equal(result_branch_image, expected_branch_image)
    np.testing.assert_equal(result_average_image, expected_average_image)


# FIXME Need a test for not pairing odd branches. Will need a test image with 3-nodes.
@pytest.mark.parametrize(
    (
        "p_to_nm",
        "reduced_node_area_filename",
        "branch_start_coords",
        "max_length_px",
        "reduced_skeleton_graph_filename",
        "image",
        "average_trace_advised",
        "node_coord",
        "pair_odd_branches",
        "filename",
        "resolution_threshold",
        "expected_pairs",
        "expected_matched_branches_filename",
        "expected_ordered_branches_filename",
        "expected_masked_image_filename",
        "expected_branch_under_over_order",
        "expected_conf",
        "expected_singlet_branch_vectors",
    ),
    [
        pytest.param(
            0.18124609,
            "catenane_node_0_reduced_node_area.npy",
            np.array([np.array([278, 353]), np.array([279, 351]), np.array([281, 352]), np.array([281, 354])]),
            110.34720989566988,
            "catenane_node_0_reduced_skeleton_graph.pkl",
            lazy_fixture("catenane_image"),
            True,
            (np.int32(280), np.int32(353)),
            True,
            "catenane_test_image",
            np.float64(1000 / 512),
            np.array([(1, 3), (2, 0)]),
            "catenane_node_0_matched_branches_analyse_node_branches.pkl",
            "catenane_node_0_ordered_branches.pkl",
            "catenane_node_0_masked_image.pkl",
            np.array([0, 1]),
            0.48972025484111525,
            [
                np.array([-0.97044686, -0.24131493]),
                np.array([0.10375883, -0.99460249]),
                np.array([0.98972257, -0.14300081]),
                np.array([0.46367343, 0.88600618]),
            ],
            id="node 0",
        )
    ],
)
def test_analyse_node_branches(
    p_to_nm: float,
    reduced_node_area_filename: npt.NDArray[np.int32],
    branch_start_coords: npt.NDArray[np.int32],
    max_length_px: np.int32,
    reduced_skeleton_graph_filename: npt.NDArray[np.int32],
    image: npt.NDArray[np.float64],
    average_trace_advised: bool,
    node_coord: tuple[np.int32, np.int32],
    pair_odd_branches: np.bool_,
    filename: str,
    resolution_threshold: np.float64,
    expected_pairs: npt.NDArray[np.int32],
    expected_matched_branches_filename: str,
    expected_ordered_branches_filename: str,
    expected_masked_image_filename: str,
    expected_branch_under_over_order: npt.NDArray[np.int32],
    expected_conf: float,
    expected_singlet_branch_vectors: list[npt.NDArray[np.int32]],
) -> None:
    """Test of analyse_node_branches() method of nodeStats class."""
    # Load the reduced node area
    reduced_node_area = np.load(GENERAL_RESOURCES / f"{reduced_node_area_filename}")

    # Load the reduced skeleton graph
    with Path(GENERAL_RESOURCES / f"{reduced_skeleton_graph_filename}").open("rb") as f:
        reduced_skeleton_graph = pickle.load(f)

    (
        result_pairs,
        result_matched_branches,
        result_ordered_branches,
        result_masked_image,
        result_branch_idx_order,
        result_conf,
        result_singlet_branch_vectors,
    ) = nodeStats.analyse_node_branches(
        p_to_nm=np.float64(p_to_nm),
        reduced_node_area=reduced_node_area,
        branch_start_coords=branch_start_coords,
        max_length_px=max_length_px,
        reduced_skeleton_graph=reduced_skeleton_graph,
        image=image,
        average_trace_advised=average_trace_advised,
        node_coord=node_coord,
        pair_odd_branches=pair_odd_branches,
        filename=filename,
        resolution_threshold=resolution_threshold,
    )

    # Load expected matched branches
    with Path(GENERAL_RESOURCES / f"{expected_matched_branches_filename}").open("rb") as f:
        expected_matched_branches = pickle.load(f)

    # Load expected masked image
    with Path(GENERAL_RESOURCES / f"{expected_masked_image_filename}").open("rb") as f:
        expected_masked_image = pickle.load(f)
    # Load expected ordered branches
    with Path(GENERAL_RESOURCES / f"{expected_ordered_branches_filename}").open("rb") as f:
        expected_ordered_branches = pickle.load(f)

    np.testing.assert_equal(result_pairs, expected_pairs)
    np.testing.assert_equal(result_matched_branches, expected_matched_branches)
    np.testing.assert_equal(result_ordered_branches, expected_ordered_branches)
    np.testing.assert_equal(result_masked_image, expected_masked_image)
    np.testing.assert_equal(result_branch_idx_order, expected_branch_under_over_order)
    np.testing.assert_almost_equal(result_conf, expected_conf, decimal=6)
    np.testing.assert_almost_equal(result_singlet_branch_vectors, expected_singlet_branch_vectors, decimal=6)


@pytest.mark.parametrize(
    (
        "pairs",
        "ordered_branches_filename",
        "reduced_skeleton_graph_filename",
        "image",
        "average_trace_advised",
        "node_coords",
        "filename",
        "expected_matched_branches_filename",
        "expected_masked_image_filename",
    ),
    [
        pytest.param(
            np.array([[1, 3], [2, 0]]),
            "catenane_node_0_ordered_branches.pkl",
            "catenane_node_0_reduced_skeleton_graph.pkl",
            lazy_fixture("catenane_image"),
            True,
            (280, 353),
            "catenane_test_image",
            "catenane_node_0_matched_branches_join_matching_branches_through_node.pkl",
            "catenane_node_0_masked_image.pkl",
            id="node 0",
        ),
        pytest.param(
            np.array([[0, 3], [2, 1]]),
            "catenane_node_1_ordered_branches.pkl",
            "catenane_node_1_reduced_skeleton_graph.pkl",
            lazy_fixture("catenane_image"),
            True,
            (312, 237),
            "catenane_test_image",
            "catenane_node_1_matched_branches_join_matching_branches_through_node.pkl",
            "catenane_node_1_masked_image.pkl",
            id="node 1",
        ),
        pytest.param(
            np.array([[3, 1], [0, 2]]),
            "catenane_node_2_ordered_branches.pkl",
            "catenane_node_2_reduced_skeleton_graph.pkl",
            lazy_fixture("catenane_image"),
            True,
            (407, 438),
            "catenane_test_image",
            "catenane_node_2_matched_branches_join_matching_branches_through_node.pkl",
            "catenane_node_2_masked_image.pkl",
            id="node 2",
        ),
        pytest.param(
            np.array([[1, 3], [2, 0]]),
            "catenane_node_3_ordered_branches.pkl",
            "catenane_node_3_reduced_skeleton_graph.pkl",
            lazy_fixture("catenane_image"),
            True,
            (451, 224),
            "catenane_test_image",
            "catenane_node_3_matched_branches_join_matching_branches_through_node.pkl",
            "catenane_node_3_masked_image.pkl",
            id="node 3",
        ),
        pytest.param(
            np.array([[1, 3], [2, 0]]),
            "catenane_node_4_ordered_branches.pkl",
            "catenane_node_4_reduced_skeleton_graph.pkl",
            lazy_fixture("catenane_image"),
            True,
            (558, 194),
            "catenane_test_image",
            "catenane_node_4_matched_branches_join_matching_branches_through_node.pkl",
            "catenane_node_4_masked_image.pkl",
            id="node 4",
        ),
    ],
)
def test_join_matching_branches_through_node(
    pairs: npt.NDArray[np.int32],
    ordered_branches_filename: str,
    reduced_skeleton_graph_filename: str,
    image: npt.NDArray[np.float64],
    average_trace_advised: bool,
    node_coords: tuple[np.int32, np.int32],
    filename: str,
    expected_matched_branches_filename: str,
    expected_masked_image_filename: str,
) -> None:
    """Test of join_matching_branches_through_node() method of nodeStats class."""
    # Load the ordered branches
    with Path(GENERAL_RESOURCES / f"{ordered_branches_filename}").open("rb") as f:
        ordered_branches = pickle.load(f)

    # Load the reduced skeleton graph
    with Path(GENERAL_RESOURCES / f"{reduced_skeleton_graph_filename}").open("rb") as f:
        reduced_skeleton_graph = pickle.load(f)

    # Load expected matched branches
    with Path(GENERAL_RESOURCES / f"{expected_matched_branches_filename}").open("rb") as f:
        expected_matched_branches = pickle.load(f)

    # Load expected masked image
    with Path(GENERAL_RESOURCES / f"{expected_masked_image_filename}").open("rb") as f:
        expected_masked_image = pickle.load(f)

    result_matched_branches, result_masked_image = nodeStats.join_matching_branches_through_node(
        pairs=pairs,
        ordered_branches=ordered_branches,
        reduced_skeleton_graph=reduced_skeleton_graph,
        image=image,
        average_trace_advised=average_trace_advised,
        node_coords=node_coords,
        filename=filename,
    )

    np.testing.assert_equal(result_matched_branches, expected_matched_branches)
    np.testing.assert_equal(result_masked_image, expected_masked_image)


@pytest.mark.parametrize(
    ("reduced_node_area", "branch_start_coords", "max_length_px", "expected_ordered_branches", "expected_vectors"),
    [
        pytest.param(
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 3, 0, 0, 0, 0],
                    [0, 0, 1, 1, 3, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 3, 1, 0, 0],
                    [0, 0, 0, 0, 0, 3, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
            np.array([[4, 3], [3, 4], [6, 5], [5, 6]]),
            2,
            [
                np.array([[4, 3], [4, 2]]),
                np.array([[3, 4], [2, 4]]),
                np.array([[6, 5], [5, 6]]),
                np.array([[5, 6], [6, 7]]),
            ],
            [
                np.array([0.0, -1.0]),
                np.array([-1.0, 0.0]),
                np.array([-0.70710678, 0.70710678]),
                np.array([0.70710678, 0.70710678]),
            ],
        ),
    ],
)
def test_get_ordered_branches_and_vectors(
    reduced_node_area: npt.NDArray[np.int32],
    branch_start_coords: npt.NDArray[np.int32],
    max_length_px: np.int32,
    expected_ordered_branches: list[npt.NDArray[np.int32]],
    expected_vectors: list[npt.NDArray[np.int32]],
) -> None:
    """Test of get_ordered_branches_and_vectors() method of nodeStats class."""
    result_ordered_branches, result_vectors = nodeStats.get_ordered_branches_and_vectors(
        reduced_node_area=reduced_node_area, branch_start_coords=branch_start_coords, max_length_px=max_length_px
    )
    np.testing.assert_equal(result_ordered_branches, expected_ordered_branches)
    np.testing.assert_almost_equal(result_vectors, expected_vectors, decimal=6)


def test_sq() -> None:
    """Test of sq() method of nodeStats class."""
    pass


def test_tri() -> None:
    """Test of tri() method of nodeStats class."""
    pass


def test_auc() -> None:
    """Test of auc() method of nodeStats class."""
    pass


def test_get_two_combinations() -> None:
    """Test of get_two_combinations() method of nodeStats class."""
    pass


def test_cross_confidence() -> None:
    """Test of cross_confidence() method of nodeStats class."""
    pass


def test_recip() -> None:
    """Test of recip() method of nodeStats class."""
    pass


def test_per_diff() -> None:
    """Test of per_diff() method of nodeStats class."""
    pass


def test_detect_ridges() -> None:
    """Test of detect_ridges() method of nodeStats class."""
    pass


def test_get_box_lims() -> None:
    """Test of get_box_lims() method of nodeStats class."""
    pass


def test_order_branch() -> None:
    """Test of order_branch() method of nodeStats class."""
    pass


def test_order_branch_from_start() -> None:
    """Test of order_branch_from_start() method of nodeStats class."""
    pass


def test_local_area_sum() -> None:
    """Test of local_area_sum() method of nodeStats class."""
    pass


def test_get_vector() -> None:
    """Test of get_vector() method of nodeStats class."""
    pass


def test_calc_angles() -> None:
    """Test of calc_angles() method of nodeStats class."""
    pass


def test_pair_vectors() -> None:
    """Test of pair_vectors() method of nodeStats class."""
    pass


def test_best_matches() -> None:
    """Test of best_matches() method of nodeStats class."""
    pass


def test_create_weighted_graph() -> None:
    """Test of create_weighted_graph() method of nodeStats class."""
    pass


def test_pair_angles() -> None:
    """Test of pair_angles() method of nodeStats class."""
    pass


def test_gaussian() -> None:
    """Test of gaussian() method of nodeStats class."""
    pass


def test_fwhm() -> None:
    """Test of fwhm() method of nodeStats class."""
    pass


def test_fwhm2() -> None:
    """Test of fwhm2() method of nodeStats class."""
    pass


def test_peak_height() -> None:
    """Test of peak_height() method of nodeStats class."""
    pass


def test_lin_interp() -> None:
    """Test of lin_interp() method of nodeStats class."""
    pass


def test_close_coords() -> None:
    """Test of close_coords() method of nodeStats class."""
    pass


def test_order_branches() -> None:
    """Test of order_branches() method of nodeStats class."""
    pass


def test_binary_line() -> None:
    """Test of binary_line() method of nodeStats class."""
    pass


def test_coord_dist() -> None:
    """Test of coord_dist() method of nodeStats class."""
    pass


def test_coord_dist_rad() -> None:
    """Test of coord_dist_rad() method of nodeStats class."""
    pass


def test_above_below_value_idx() -> None:
    """Test of above_below_value_idx() method of nodeStats class."""
    pass


def test_average_height_trace() -> None:
    """Test of average_height_trace() method of nodeStats class."""
    pass


def test_fill_holes() -> None:
    """Test of fill_holes() method of nodeStats class."""
    pass


def test_remove_re_entering_branches() -> None:
    """Test of remove_re_entering_branches() method of nodeStats class."""
    pass


@pytest.mark.parametrize(
    ("node_image", "node_coordinate", "expected_node_image"),
    [
        pytest.param(
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 3, 0, 1, 0, 0, 0],
                    [0, 1, 0, 0, 0, 1, 1, 0, 3, 0, 0, 0, 0],
                    [0, 0, 1, 0, 1, 0, 0, 0, 0, 3, 1, 0, 0],
                    [0, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 1, 0],
                    [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
            np.array([6, 7]),
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 3, 0, 1, 0, 0, 0],
                    [0, 1, 0, 0, 0, 1, 1, 0, 3, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
        )
    ],
)
def test_only_centre_branches(
    node_image: npt.NDArray[np.int32],
    node_coordinate: npt.NDArray[np.int32],
    expected_node_image: npt.NDArray[np.int32],
) -> None:
    """Test of only_centre_branches() method of nodeStats class."""
    result_node_image = nodeStats.only_centre_branches(node_image, node_coordinate)

    np.testing.assert_equal(result_node_image, expected_node_image)


def test_average_uniques() -> None:
    """Test of average_uniques() method of nodeStats class."""
    pass


def test_compile_trace() -> None:
    """Test of compile_trace() method of nodeStats class."""
    pass


def test_get_minus_img() -> None:
    """Test of get_minus_img() method of nodeStats class."""
    pass


def test_remove_common_values() -> None:
    """Test of remove_common_values() method of nodeStats class."""
    pass


def test_trace() -> None:
    """Test of trace() method of nodeStats class."""
    pass


def test_reduce_rows() -> None:
    """Test of reduce_rows() method of nodeStats class."""
    pass


def test_get_trace_segment() -> None:
    """Test of get_trace_segment() method of nodeStats class."""
    pass


def test_comb_xyzs() -> None:
    """Test of comb_xyzs() method of nodeStats class."""
    pass


def test_remove_duplicates() -> None:
    """Test of remove_duplicates() method of nodeStats class."""
    pass


def test_order_from_end() -> None:
    """Test of order_from_end() method of nodeStats class."""
    pass


def test_get_trace_idxs() -> None:
    """Test of get_trace_idxs() method of nodeStats class."""
    pass


def test_get_visual_img() -> None:
    """Test of get_visual_img() method of nodeStats class."""
    pass


def test_average_crossing_confs() -> None:
    """Test of average_crossing_confs() method of nodeStats class."""
    pass


def test_minimum_crossing_confs() -> None:
    """Test minimum_crossing_confs() method of nodeStats class."""
    pass


@pytest.mark.parametrize(
    (
        "image_filename",
        "pixel_to_nm_scaling",
        "disordered_tracing_crop_data_filename",
        "node_joining_length",
        "node_extend_dist",
        "branch_pairing_length",
        "pair_odd_branches",
        "expected_nodestats_data_filename",
        "expected_nodestats_grainstats_filename",
        "expected_nodestats_all_images_filename",
        "expected_nodestats_branch_images_filename",
    ),
    [
        pytest.param(
            "example_catenanes.npy",
            # Pixel to nm scaling
            0.488,
            "catenanes_disordered_tracing_crop_data.pkl",
            # Node joining length
            7.0,
            # Node extend distance
            14.0,
            # Branch pairing length
            20.0,
            # Pair odd branches
            True,
            "catenanes_nodestats_data.pkl",
            "catenanes_nodestats_grainstats.csv",
            "catenanes_nodestats_all_images.pkl",
            "catenanes_nodestats_branch_images.pkl",
            id="catenane",
        ),
        pytest.param(
            "example_rep_int.npy",
            # Pixel to nm scaling
            0.488,
            "rep_int_disordered_tracing_crop_data.pkl",
            # Node joining length
            7.0,
            # Node extend distance
            14.0,
            # Branch pairing length
            20.0,
            # Pair odd branches
            False,
            "rep_int_nodestats_data_no_pair_odd_branches.pkl",
            "rep_int_nodestats_grainstats_no_pair_odd_branches.csv",
            "rep_int_nodestats_all_images_no_pair_odd_branches.pkl",
            "rep_int_nodestats_branch_images_no_pair_odd_branches.pkl",
            id="replication_intermediate, not pairing odd branches",
        ),
        pytest.param(
            "example_rep_int.npy",
            # Pixel to nm scaling
            0.488,
            "rep_int_disordered_tracing_crop_data.pkl",
            # Node joining length
            7.0,
            # Node extend distance
            14.0,
            # Branch pairing length
            20.0,
            # Pair odd branches
            True,
            "rep_int_nodestats_data_pair_odd_branches.pkl",
            "rep_int_nodestats_grainstats_pair_odd_branches.csv",
            "rep_int_nodestats_all_images_pair_odd_branches.pkl",
            "rep_int_nodestats_branch_images_pair_odd_branches.pkl",
            id="replication_intermediate, pairing odd branches",
        ),
    ],
)
def test_nodestats_image(
    image_filename: str,
    pixel_to_nm_scaling: float,
    disordered_tracing_crop_data_filename: str,
    node_joining_length: float,
    node_extend_dist: float,
    branch_pairing_length: float,
    pair_odd_branches: bool,
    expected_nodestats_data_filename: str,
    expected_nodestats_grainstats_filename: str,
    expected_nodestats_all_images_filename: str,
    expected_nodestats_branch_images_filename: str,
) -> None:
    """Test of nodestats_image() method of nodeStats class."""
    # Load the image
    image = np.load(GENERAL_RESOURCES / image_filename)
    # load disordered_tracing_crop_data from pickle
    with Path(DISORDERED_TRACING_RESOURCES / disordered_tracing_crop_data_filename).open("rb") as f:
        disordered_tracing_crop_data = pickle.load(f)

    (
        result_nodestats_data,
        result_nodestats_grainstats,
        result_nodestats_all_images,
        result_nodestats_branch_images,
    ) = nodestats_image(
        image=image,
        disordered_tracing_direction_data=disordered_tracing_crop_data,
        filename="test_image",
        pixel_to_nm_scaling=pixel_to_nm_scaling,
        node_joining_length=node_joining_length,
        node_extend_dist=node_extend_dist,
        branch_pairing_length=branch_pairing_length,
        pair_odd_branches=pair_odd_branches,
        pad_width=1,
    )

    # # DEBUGGING (For viewing images)
    # convolved_skeletons = result_all_images["convolved_skeletons"]
    # node_centres = result_all_images["node_centres"]
    # connected_nodes = result_all_images["connected_nodes"]

    # Save the results

    # Save the result_nodestats_data
    # with Path(NODESTATS_RESOURCES / expected_nodestats_data_filename).open("wb") as f:
    #     pickle.dump(result_nodestats_data, f)

    # # Save the result_stats_df as a csv
    # result_nodestats_grainstats_df.to_csv(
    #     NODESTATS_RESOURCES / expected_nodestats_grainstats_filename
    # )

    # # Save the result_all_images
    # with Path(NODESTATS_RESOURCES / expected_nodestats_all_images_filename).open("wb") as f:
    #     pickle.dump(result_nodestats_all_images, f)

    # # Save the result_nodestats_branch_images
    # with Path(NODESTATS_RESOURCES / expected_nodestats_branch_images_filename).open("wb") as f:
    #     pickle.dump(result_nodestats_branch_images, f)

    # Load expected data

    # Load the expected nodestats data
    with Path(NODESTATS_RESOURCES / expected_nodestats_data_filename).open("rb") as f:
        expected_nodestats_data = pickle.load(f)

    # Load the expected grainstats additions
    expected_nodestats_grainstats = pd.read_csv(
        NODESTATS_RESOURCES / expected_nodestats_grainstats_filename, index_col=0
    )

    # Load the expected all images
    with Path(NODESTATS_RESOURCES / expected_nodestats_all_images_filename).open("rb") as f:
        expected_all_images = pickle.load(f)

    # Load the expected nodestats branch images
    with Path(NODESTATS_RESOURCES / expected_nodestats_branch_images_filename).open("rb") as f:
        expected_nodestats_branch_images = pickle.load(f)

    assert dict_almost_equal(result_nodestats_data, expected_nodestats_data, abs_tol=1e-3)
    pd.testing.assert_frame_equal(result_nodestats_grainstats, expected_nodestats_grainstats)
    assert dict_almost_equal(result_nodestats_all_images, expected_all_images)
    assert dict_almost_equal(result_nodestats_branch_images, expected_nodestats_branch_images)
