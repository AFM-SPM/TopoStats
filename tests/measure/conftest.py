"""Fixtures for testing the measure module."""

import numpy as np
import pytest
from skimage.morphology import label

from topostats.tracing.nodestats import nodeStats

# pylint: disable=redefined-outer-name


@pytest.fixture()
def network_array_representation_figure_8():
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


@pytest.fixture()
def labelled_nodes_figure_8(network_array_representation_figure_8):
    """Fixture for the labelled nodes of the figure 8 test molecule."""
    # Grab the nodes from the network array representation.
    just_nodes = np.where(network_array_representation_figure_8 == 3, 1, 0)
    # Adding the 1 and 0 in the np.where call above makes the nodes 1 and the rest 0.
    return label(just_nodes)


@pytest.fixture()
def labelled_branches_figure_8(network_array_representation_figure_8):
    """Fixture for the labelled branches of the figure 8 test molecule."""
    # Grab the branches from the network array representation.
    just_branches = np.where(network_array_representation_figure_8 == 1, 1, 0)
    # Adding the 1 and 0 in the np.where call above makes the branches 1 and the rest 0.
    return label(just_branches)


# fixture for just the skeleton
@pytest.fixture()
def skeleton_figure_8(network_array_representation_figure_8):
    """Fixture for the skeleton of the figure 8 test molecule."""
    return network_array_representation_figure_8.astype(bool)


@pytest.fixture()
def whole_skeleton_graph_figure_8(skeleton_figure_8):
    """Fixture for the whole skeleton graph of the figure 8 test molecule."""
    # Create graph of just skeleton from the skeleton (just 1s)
    return nodeStats.skeleton_image_to_graph(skeleton_figure_8)


@pytest.fixture()
def expected_network_array_representation_figure_8():
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
