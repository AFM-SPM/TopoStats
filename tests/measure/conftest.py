"""Fixtures for testing the measure module."""

from __future__ import annotations

import networkx as nx
import numpy as np
import numpy.typing as npt
import pytest
from skimage import draw
from skimage.morphology import label

from topostats.tracing.nodestats import nodeStats

# pylint: disable=redefined-outer-name


@pytest.fixture()
def network_array_representation_figure_8() -> npt.NDArray[np.int32]:
    """Fixture for the network array representation of the figure 8 test molecule."""
    return np.array(
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
    )


def tiny_circle() -> npt.NDArray:
    """Tiny circle."""
    tiny_circle = np.zeros((3, 3), dtype=np.uint8)
    rr, cc = draw.circle_perimeter(1, 1, 1)
    tiny_circle[rr, cc] = 1
    return tiny_circle


@pytest.fixture()
def small_circle() -> npt.NDArray:
    """Small circle."""
    small_circle = np.zeros((5, 5), dtype=np.uint8)
    rr, cc = draw.circle_perimeter(2, 2, 2)
    small_circle[rr, cc] = 1
    return small_circle


@pytest.fixture()
def tiny_quadrilateral() -> npt.NDArray:
    """Tiny quadrilateral."""
    return np.asarray(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )


@pytest.fixture()
def labelled_nodes_figure_8(network_array_representation_figure_8) -> npt.NDArray[np.int32]:
    """Fixture for the labelled nodes of the figure 8 test molecule."""
    # Grab the nodes from the network array representation.
    just_nodes = np.where(network_array_representation_figure_8 == 3, 1, 0)
    # Adding the 1 and 0 in the np.where call above makes the nodes 1 and the rest 0.
    return label(just_nodes)


@pytest.fixture()
def labelled_branches_figure_8(network_array_representation_figure_8) -> npt.NDArray[np.int32]:
    """Fixture for the labelled branches of the figure 8 test molecule."""
    # Grab the branches from the network array representation.
    just_branches = np.where(network_array_representation_figure_8 == 1, 1, 0)
    # Adding the 1 and 0 in the np.where call above makes the branches 1 and the rest 0.
    return label(just_branches)


# fixture for just the skeleton
@pytest.fixture()
def skeleton_figure_8(network_array_representation_figure_8) -> npt.NDArray[np.bool_]:
    """Fixture for the skeleton of the figure 8 test molecule."""
    return network_array_representation_figure_8.astype(bool)


@pytest.fixture()
def whole_skeleton_graph_figure_8(skeleton_figure_8) -> nx.classes.graph.Graph:
    """Fixture for the whole skeleton graph of the figure 8 test molecule."""
    # Create graph of just skeleton from the skeleton (just 1s)
    return nodeStats.skeleton_image_to_graph(skeleton_figure_8)


@pytest.fixture()
def expected_network_array_representation_figure_8() -> npt.NDArray[np.int32]:
    """Fixture for the expected network array representation of the figure 8 test molecule."""
    return np.array(
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
    )


def tiny_square() -> npt.NDArray:
    """Tiny square."""
    return np.asarray([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]], dtype=np.uint8)


@pytest.fixture()
def tiny_triangle() -> npt.NDArray:
    """Tiny triangle."""
    return np.asarray([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]], dtype=np.uint8)


@pytest.fixture()
def arbitrary_triangle() -> npt.NDArray:
    """Arbitrary triangle."""
    return np.array([[1.18727719, 2.96140198], [4.14262995, 3.17983179], [6.92472119, 0.64147496]])


@pytest.fixture()
def tiny_rectangle() -> npt.NDArray:
    """Tiny rectangle."""
    return np.asarray([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]], dtype=np.uint8)


@pytest.fixture()
def tiny_ellipse() -> npt.NDArray:
    """Tiny ellipse."""
    return np.asarray(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )


@pytest.fixture()
def holo_circle() -> npt.NDArray:
    """Holo circle."""
    holo_circle = np.zeros((7, 7), dtype=np.uint8)
    rr, cc = draw.circle_perimeter(3, 3, 2)
    holo_circle[rr, cc] = 1
    return holo_circle


@pytest.fixture()
def holo_ellipse_vertical() -> npt.NDArray:
    """Holo ellipse (vertical)."""
    holo_ellipse_vertical = np.zeros((11, 9), dtype=np.uint8)
    rr, cc = draw.ellipse_perimeter(5, 4, 4, 3)
    holo_ellipse_vertical[rr, cc] = 1
    return holo_ellipse_vertical


@pytest.fixture()
def holo_ellipse_horizontal() -> npt.NDArray:
    """Holo ellipse (horizontal)."""
    holo_ellipse_horizontal = np.zeros((9, 11), dtype=np.uint8)
    rr, cc = draw.ellipse_perimeter(4, 5, 3, 4)
    holo_ellipse_horizontal[rr, cc] = 1
    return holo_ellipse_horizontal


@pytest.fixture()
def holo_ellipse_angled() -> npt.NDArray:
    """Holo ellipse (angled)."""
    holo_ellipse_angled = np.zeros((8, 10), dtype=np.uint8)
    rr, cc = draw.ellipse_perimeter(4, 5, 1, 3, orientation=np.deg2rad(30))
    holo_ellipse_angled[rr, cc] = 1
    return holo_ellipse_angled


@pytest.fixture()
def curved_line() -> npt.NDArray:
    """Curved line."""
    curved_line = np.zeros((10, 10), dtype=np.uint8)
    rr, cc = draw.bezier_curve(1, 5, 5, -2, 8, 8, 2)
    curved_line[rr, cc] = 1
    return curved_line


@pytest.fixture()
def filled_circle() -> npt.NDArray:
    """Circle."""
    filled_circle = np.zeros((9, 9), dtype=np.uint8)
    rr, cc = draw.disk((4, 4), 4)
    filled_circle[rr, cc] = 1
    return filled_circle


@pytest.fixture()
def filled_ellipse_vertical() -> npt.NDArray:
    """Ellipse (vertical)."""
    filled_ellipse_vertical = np.zeros((9, 7), dtype=np.uint8)
    rr, cc = draw.ellipse(4, 3, 4, 3)
    filled_ellipse_vertical[rr, cc] = 1
    return filled_ellipse_vertical


@pytest.fixture()
def filled_ellipse_horizontal() -> npt.NDArray:
    """Ellipse (horizontal)."""
    filled_ellipse_horizontal = np.zeros((7, 9), dtype=np.uint8)
    rr, cc = draw.ellipse(3, 4, 3, 4)
    filled_ellipse_horizontal[rr, cc] = 1
    return filled_ellipse_horizontal


@pytest.fixture()
def filled_ellipse_angled() -> npt.NDArray:
    """Ellipse (angled)."""
    filled_ellipse_angled = np.zeros((9, 11), dtype=np.uint8)
    rr, cc = draw.ellipse(4, 5, 3, 5, rotation=np.deg2rad(30))
    filled_ellipse_angled[rr, cc] = 1
    return filled_ellipse_angled
