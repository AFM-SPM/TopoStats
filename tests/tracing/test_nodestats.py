# Disable ruff 301 - pickle loading is unsafe, but we don't care for tests.
# ruff: noqa: S301
"""Test the nodestats module."""

import pickle
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

# pylint: disable=import-error
# pylint: disable=no-name-in-module
from topostats.tracing.nodestats import nodeStats, nodestats_image

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"
TRACING_RESOURCES = RESOURCES / "tracing"
DISORDERED_TRACING_RESOURCES = TRACING_RESOURCES / "disordered_tracing"
NODESTATS_RESOURCES = TRACING_RESOURCES / "nodestats"

# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-many-lines
# pylint: disable=too-many-positional-arguments


def test_get_node_stats() -> None:
    """Test of get_node_stats() method of nodeStats class."""


@pytest.mark.skip(reason="Awaiting test to be written 2024-10-15")
def test_check_node_errorless() -> None:
    """Test of check_node_errorless() method of nodeStats class."""


@pytest.mark.skip(reason="Awaiting test to be written 2024-10-15")
def test_skeleton_image_to_graph() -> None:
    """Test of skeleton_image_to_graph() method of nodeStats class."""


@pytest.mark.skip(reason="Awaiting test to be written 2024-10-15")
def test_graph_to_skeleton_image() -> None:
    """Test of graph_to_skeleton_image() method of nodeStats class."""


@pytest.mark.skip(reason="Awaiting test to be written 2024-10-15")
def test_tidy_branches() -> None:
    """Test of tidy_branches() method of nodeStats class."""


@pytest.mark.skip(reason="Awaiting test to be written 2024-10-15")
def test_keep_biggest_object() -> None:
    """Test of keep_biggest_object() method of nodeStats class."""


@pytest.mark.skip(reason="Awaiting test to be written 2024-10-15")
def test_connect_close_nodes() -> None:
    """Test of connect_close_nodes() method of nodeStats class."""


@pytest.mark.skip(reason="Awaiting test to be written 2024-10-15")
def test_highlight_node_centres() -> None:
    """Test of highlight_node_centres() method of nodeStats class."""


@pytest.mark.skip(reason="Awaiting test to be written 2024-10-15")
def test_connect_extended_nodes() -> None:
    """Test of connect_extended_nodes() method of nodeStats class."""


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
        pixel_to_nm_scaling=np.float64(1.0),
        n_grain=0,
        node_joining_length=0.0,
        node_extend_dist=14.0,
        branch_pairing_length=20.0,
        pair_odd_branches=True,
    )
    nodestats.whole_skel_graph = nodestats.skeleton_image_to_graph(nodestats.skeleton)
    result = nodestats.connect_extended_nodes_nearest(connected_nodes, node_extend_dist=8.0)

    assert np.array_equal(result, expected_nodes)


@pytest.mark.skip(reason="Awaiting test to be written 2024-10-15")
def test_find_branch_starts() -> None:
    """Test of find_branch_starts() method of nodeStats class."""


# Create nodestats class using the cats image - will allow running the code for diagnostics
def test_analyse_nodes(nodestats_catenane: nodeStats, snapshot) -> None:
    """Test of analyse_nodes() method of nodeStats class."""
    nodestats_catenane.analyse_nodes(max_branch_length=20)

    node_dict_result = nodestats_catenane.node_dicts
    image_dict_result = nodestats_catenane.image_dict

    assert node_dict_result == snapshot
    assert image_dict_result == snapshot
    all_connected_nodes = nodestats_catenane.all_connected_nodes
    # ns-rse: syrupy doesn't yet support numpy arrays so we convert to string
    #         https://github.com/syrupy-project/syrupy/issues/887
    assert np.array2string(all_connected_nodes) == snapshot


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
            id="catenane",
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
    snapshot,
) -> None:
    """Test of add_branches_to_labelled_image() method of nodeStats class."""
    # Load the matched branches
    with Path(NODESTATS_RESOURCES / matched_branches_filename).open("rb") as f:
        matched_branches: dict[int, dict[str, npt.NDArray[np.number]]] = pickle.load(f)

    # Load the masked image
    with Path(NODESTATS_RESOURCES / masked_image_filename).open("rb") as f:
        masked_image: dict[int, dict[str, npt.NDArray[np.bool_]]] = pickle.load(f)

    # Load the ordered branches
    with Path(NODESTATS_RESOURCES / ordered_branches_filename).open("rb") as f:
        ordered_branches: list[npt.NDArray[np.int32]] = pickle.load(f)

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

    # ns-rse: syrupy doesn't yet support numpy arrays so we convert to string
    #         https://github.com/syrupy-project/syrupy/issues/887
    assert np.array2string(result_branch_image) == snapshot
    assert np.array2string(result_average_image, precision=9) == snapshot


# FIXME Need a test for not pairing odd branches. Will need a test image with 3-nodes.
@pytest.mark.parametrize(
    (
        "pixel_to_nm_scaling",
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
            "catenane_image",
            True,
            (np.int32(280), np.int32(353)),
            True,
            "catenane_test_image",
            np.float64(1000 / 512),
            np.array([(1, 3), (2, 0)]),
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
    pixel_to_nm_scaling: float,
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
    expected_branch_under_over_order: npt.NDArray[np.int32],
    expected_conf: float,
    expected_singlet_branch_vectors: list[npt.NDArray[np.int32]],
    request,
    snapshot,
) -> None:
    """Test of analyse_node_branches() method of nodeStats class."""
    # Load the fixtures
    image = request.getfixturevalue(image)

    # Load the reduced node area
    reduced_node_area = np.load(NODESTATS_RESOURCES / reduced_node_area_filename)

    # Load the reduced skeleton graph
    with Path(NODESTATS_RESOURCES / reduced_skeleton_graph_filename).open("rb") as f:
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
        p_to_nm=np.float64(pixel_to_nm_scaling),
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

    # ns-rse : Could potentially replace the `expected_` with == snapshot ?
    np.testing.assert_equal(result_pairs, expected_pairs)
    np.testing.assert_equal(result_branch_idx_order, expected_branch_under_over_order)
    np.testing.assert_almost_equal(result_conf, expected_conf, decimal=6)
    np.testing.assert_almost_equal(result_singlet_branch_vectors, expected_singlet_branch_vectors, decimal=6)
    assert result_matched_branches == snapshot
    assert result_ordered_branches == snapshot
    assert result_masked_image == snapshot


@pytest.mark.parametrize(
    (
        "pairs",
        "ordered_branches_filename",
        "reduced_skeleton_graph_filename",
        "image",
        "average_trace_advised",
        "node_coords",
        "filename",
    ),
    [
        pytest.param(
            np.array([[1, 3], [2, 0]]),
            "catenane_node_0_ordered_branches.pkl",
            "catenane_node_0_reduced_skeleton_graph.pkl",
            "catenane_image",
            True,
            (280, 353),
            "catenane_test_image",
            id="node 0",
        ),
        pytest.param(
            np.array([[0, 3], [2, 1]]),
            "catenane_node_1_ordered_branches.pkl",
            "catenane_node_1_reduced_skeleton_graph.pkl",
            "catenane_image",
            True,
            (312, 237),
            "catenane_test_image",
            id="node 1",
        ),
        pytest.param(
            np.array([[3, 1], [0, 2]]),
            "catenane_node_2_ordered_branches.pkl",
            "catenane_node_2_reduced_skeleton_graph.pkl",
            "catenane_image",
            True,
            (407, 438),
            "catenane_test_image",
            id="node 2",
        ),
        pytest.param(
            np.array([[1, 3], [2, 0]]),
            "catenane_node_3_ordered_branches.pkl",
            "catenane_node_3_reduced_skeleton_graph.pkl",
            "catenane_image",
            True,
            (451, 224),
            "catenane_test_image",
            id="node 3",
        ),
        pytest.param(
            np.array([[1, 3], [2, 0]]),
            "catenane_node_4_ordered_branches.pkl",
            "catenane_node_4_reduced_skeleton_graph.pkl",
            "catenane_image",
            True,
            (558, 194),
            "catenane_test_image",
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
    request,
    snapshot,
) -> None:
    """Test of join_matching_branches_through_node() method of nodeStats class."""
    # Load the fixtures
    image = request.getfixturevalue(image)

    # Load the ordered branches
    with Path(NODESTATS_RESOURCES / ordered_branches_filename).open("rb") as f:
        ordered_branches = pickle.load(f)

    # Load the reduced skeleton graph
    with Path(NODESTATS_RESOURCES / reduced_skeleton_graph_filename).open("rb") as f:
        reduced_skeleton_graph = pickle.load(f)

    result_matched_branches, result_masked_image = nodeStats.join_matching_branches_through_node(
        pairs=pairs,
        ordered_branches=ordered_branches,
        reduced_skeleton_graph=reduced_skeleton_graph,
        image=image,
        average_trace_advised=average_trace_advised,
        node_coords=node_coords,
        filename=filename,
    )

    assert result_matched_branches == snapshot
    assert result_masked_image == snapshot


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


@pytest.mark.skip(reason="Awaiting test to be written 2024-10-15")
def test_tri() -> None:
    """Test of tri() method of nodeStats class."""


@pytest.mark.skip(reason="Awaiting test to be written 2024-10-15")
def test_auc() -> None:
    """Test of auc() method of nodeStats class."""


@pytest.mark.skip(reason="Awaiting test to be written 2024-10-15")
def test_cross_confidence() -> None:
    """Test of cross_confidence() method of nodeStats class."""


@pytest.mark.skip(reason="Awaiting test to be written 2024-10-15")
def test_recip() -> None:
    """Test of recip() method of nodeStats class."""


@pytest.mark.skip(reason="Awaiting test to be written 2024-10-15")
def test_per_diff() -> None:
    """Test of per_diff() method of nodeStats class."""


@pytest.mark.skip(reason="Awaiting test to be written 2024-10-15")
def test_detect_ridges() -> None:
    """Test of detect_ridges() method of nodeStats class."""


@pytest.mark.skip(reason="Awaiting test to be written 2024-10-15")
def test_get_box_lims() -> None:
    """Test of get_box_lims() method of nodeStats class."""


@pytest.mark.skip(reason="Awaiting test to be written 2024-10-15")
def test_order_branch() -> None:
    """Test of order_branch() method of nodeStats class."""


@pytest.mark.skip(reason="Awaiting test to be written 2024-10-15")
def test_order_branch_from_start() -> None:
    """Test of order_branch_from_start() method of nodeStats class."""


@pytest.mark.skip(reason="Awaiting test to be written 2024-10-15")
def test_local_area_sum() -> None:
    """Test of local_area_sum() method of nodeStats class."""


@pytest.mark.skip(reason="Awaiting test to be written 2024-10-15")
def test_get_vector() -> None:
    """Test of get_vector() method of nodeStats class."""


@pytest.mark.skip(reason="Awaiting test to be written 2024-10-15")
def test_calc_angles() -> None:
    """Test of calc_angles() method of nodeStats class."""


@pytest.mark.skip(reason="Awaiting test to be written 2024-10-15")
def test_pair_vectors() -> None:
    """Test of pair_vectors() method of nodeStats class."""


@pytest.mark.skip(reason="Awaiting test to be written 2024-10-15")
def test_best_matches() -> None:
    """Test of best_matches() method of nodeStats class."""


@pytest.mark.skip(reason="Awaiting test to be written 2024-10-15")
def test_create_weighted_graph() -> None:
    """Test of create_weighted_graph() method of nodeStats class."""


@pytest.mark.skip(reason="Awaiting test to be written 2024-10-15")
def test_pair_angles() -> None:
    """Test of pair_angles() method of nodeStats class."""


@pytest.mark.skip(reason="Awaiting test to be written 2024-10-15")
def test_gaussian() -> None:
    """Test of gaussian() method of nodeStats class."""


@pytest.mark.skip(reason="Awaiting test to be written 2024-10-15")
def test_fwhm() -> None:
    """Test of fwhm() method of nodeStats class."""


@pytest.mark.skip(reason="Awaiting test to be written 2024-10-15")
def test_fwhm2() -> None:
    """Test of fwhm2() method of nodeStats class."""


@pytest.mark.skip(reason="Awaiting test to be written 2024-10-15")
def test_peak_height() -> None:
    """Test of peak_height() method of nodeStats class."""


@pytest.mark.skip(reason="Awaiting test to be written 2024-10-15")
def test_lin_interp() -> None:
    """Test of lin_interp() method of nodeStats class."""


@pytest.mark.skip(reason="Awaiting test to be written 2024-10-15")
def test_close_coords() -> None:
    """Test of close_coords() method of nodeStats class."""


@pytest.mark.skip(reason="Awaiting test to be written 2024-10-15")
def test_order_branches() -> None:
    """Test of order_branches() method of nodeStats class."""


@pytest.mark.skip(reason="Awaiting test to be written 2024-10-15")
def test_binary_line() -> None:
    """Test of binary_line() method of nodeStats class."""


@pytest.mark.skip(reason="Awaiting test to be written 2024-10-15")
def test_coord_dist() -> None:
    """Test of coord_dist() method of nodeStats class."""


@pytest.mark.skip(reason="Awaiting test to be written 2024-10-15")
def test_coord_dist_rad() -> None:
    """Test of coord_dist_rad() method of nodeStats class."""


@pytest.mark.skip(reason="Awaiting test to be written 2024-10-15")
def test_above_below_value_idx() -> None:
    """Test of above_below_value_idx() method of nodeStats class."""


@pytest.mark.skip(reason="Awaiting test to be written 2024-10-15")
def test_average_height_trace() -> None:
    """Test of average_height_trace() method of nodeStats class."""


@pytest.mark.skip(reason="Awaiting test to be written 2024-10-15")
def test_fill_holes() -> None:
    """Test of fill_holes() method of nodeStats class."""


@pytest.mark.skip(reason="Awaiting test to be written 2024-10-15")
def test_remove_re_entering_branches() -> None:
    """Test of remove_re_entering_branches() method of nodeStats class."""


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


@pytest.mark.skip(reason="Awaiting test to be written 2024-10-15")
def test_compile_trace() -> None:
    """Test of compile_trace() method of nodeStats class."""


@pytest.mark.skip(reason="Awaiting test to be written 2024-10-15")
def test_get_minus_img() -> None:
    """Test of get_minus_img() method of nodeStats class."""


@pytest.mark.skip(reason="Awaiting test to be written 2024-10-15")
def test_remove_common_values() -> None:
    """Test of remove_common_values() method of nodeStats class."""


@pytest.mark.skip(reason="Awaiting test to be written 2024-10-15")
def test_trace() -> None:
    """Test of trace() method of nodeStats class."""


@pytest.mark.skip(reason="Awaiting test to be written 2024-10-15")
def test_reduce_rows() -> None:
    """Test of reduce_rows() method of nodeStats class."""


@pytest.mark.skip(reason="Awaiting test to be written 2024-10-15")
def test_get_trace_segment() -> None:
    """Test of get_trace_segment() method of nodeStats class."""


@pytest.mark.skip(reason="Awaiting test to be written 2024-10-15")
def test_comb_xyzs() -> None:
    """Test of comb_xyzs() method of nodeStats class."""


@pytest.mark.skip(reason="Awaiting test to be written 2024-10-15")
def test_remove_duplicates() -> None:
    """Test of remove_duplicates() method of nodeStats class."""


@pytest.mark.skip(reason="Awaiting test to be written 2024-10-15")
def test_order_from_end() -> None:
    """Test of order_from_end() method of nodeStats class."""


@pytest.mark.skip(reason="Awaiting test to be written 2024-10-15")
def test_get_trace_idxs() -> None:
    """Test of get_trace_idxs() method of nodeStats class."""


@pytest.mark.skip(reason="Awaiting test to be written 2024-10-15")
def test_get_visual_img() -> None:
    """Test of get_visual_img() method of nodeStats class."""


@pytest.mark.skip(reason="Awaiting test to be written 2024-10-15")
def test_average_crossing_confs() -> None:
    """Test of average_crossing_confs() method of nodeStats class."""


@pytest.mark.skip(reason="Awaiting test to be written 2024-10-15")
def test_minimum_crossing_confs() -> None:
    """Test minimum_crossing_confs() method of nodeStats class."""


@pytest.mark.parametrize(
    (
        "image_filename",
        "pixel_to_nm_scaling",
        "disordered_tracing_crop_data_filename",
        "node_joining_length",
        "node_extend_dist",
        "branch_pairing_length",
        "pair_odd_branches",
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
    snapshot,
) -> None:
    """Test of nodestats_image() method of nodeStats class."""
    # Load the image
    image = np.load(RESOURCES / image_filename)
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
    )

    # # DEBUGGING (For viewing images)
    # convolved_skeletons = result_all_images["convolved_skeletons"]
    # node_centres = result_all_images["node_centres"]
    # connected_nodes = result_all_images["connected_nodes"]

    assert result_nodestats_data == snapshot
    # ns-rse: syrupy doesn't yet support Pandas DataFrames so we convert to string
    #         https://github.com/syrupy-project/syrupy/issues/887
    assert result_nodestats_grainstats.to_string() == snapshot
    assert result_nodestats_all_images == snapshot
    assert result_nodestats_branch_images == snapshot
