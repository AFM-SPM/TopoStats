"""Fixtures for the tracing tests."""

import pickle
from pathlib import Path
from typing import TypedDict

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from topostats.filters import Filters
from topostats.grains import Grains
from topostats.tracing.dnatracing import dnaTrace
from topostats.tracing.nodestats import nodeStats
from topostats.tracing.skeletonize import getSkeleton, topostatsSkeletonize

# This is required because of the inheritance used throughout
# pylint: disable=redefined-outer-name

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"

RNG = np.random.default_rng(seed=1000)

# Derive fixtures for DNA Tracing
GRAINS = np.array(
    [
        [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 2],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 2],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 2],
        [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
        [0, 0, 3, 3, 3, 3, 3, 0, 0, 0, 2],
        [0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 2],
        [0, 0, 3, 3, 3, 3, 3, 0, 0, 0, 2],
        [0, 0, 4, 4, 4, 4, 4, 0, 0, 0, 2],
    ]
)
FULL_IMAGE = RNG.random((GRAINS.shape[0], GRAINS.shape[1]))


@pytest.fixture()
def test_dnatracing() -> dnaTrace:
    """Instantiate a dnaTrace object."""
    return dnaTrace(image=FULL_IMAGE, grain=GRAINS, filename="Test", pixel_to_nm_scaling=1.0)


@pytest.fixture()
def minicircle_dnatracing(
    minicircle_grain_gaussian_filter: Filters,
    minicircle_grain_coloured: Grains,
    dnatracing_config: dict,
) -> dnaTrace:
    """DnaTrace object instantiated with minicircle data."""  # noqa: D403
    dnatracing_config.pop("pad_width")
    dna_traces = dnaTrace(
        image=minicircle_grain_coloured.image.T,
        grain=minicircle_grain_coloured.directions["above"]["labelled_regions_02"],
        filename=minicircle_grain_gaussian_filter.filename,
        pixel_to_nm_scaling=minicircle_grain_gaussian_filter.pixel_to_nm_scaling,
        **dnatracing_config,
    )
    dna_traces.trace_dna()
    return dna_traces


# DNA Tracing Fixtures
@pytest.fixture()
def minicircle_all_statistics() -> pd.DataFrame:
    """Statistics for minicricle."""
    return pd.read_csv(RESOURCES / "minicircle_default_all_statistics.csv", header=0)


# Skeletonizing Fixtures
@pytest.fixture()
def skeletonize_get_skeleton() -> getSkeleton:
    """Instantiate a getSkeleton object."""
    return getSkeleton(image=None, mask=None)


@pytest.fixture()
def skeletonize_circular() -> np.ndarray:
    """Circular molecule for testing skeletonizing."""
    return np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0],
            [0, 0, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1, 0, 0],
            [0, 0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 2, 1, 0, 0],
            [0, 0, 1, 2, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 2, 1, 0, 0],
            [0, 0, 1, 2, 3, 4, 3, 2, 2, 2, 2, 2, 2, 2, 3, 4, 3, 2, 1, 0, 0],
            [0, 0, 1, 2, 3, 4, 3, 2, 1, 1, 1, 1, 1, 2, 3, 4, 3, 2, 1, 0, 0],
            [0, 0, 1, 2, 3, 4, 3, 2, 1, 0, 0, 0, 1, 2, 3, 4, 3, 2, 1, 0, 0],
            [0, 0, 1, 2, 3, 4, 3, 2, 1, 0, 0, 0, 1, 2, 3, 4, 3, 2, 1, 0, 0],
            [0, 0, 1, 2, 3, 4, 3, 2, 1, 0, 0, 0, 1, 2, 3, 4, 3, 2, 1, 0, 0],
            [0, 0, 1, 2, 3, 4, 3, 2, 1, 1, 1, 1, 1, 2, 3, 4, 3, 2, 1, 0, 0],
            [0, 0, 1, 2, 3, 4, 3, 2, 2, 2, 2, 2, 2, 2, 3, 4, 3, 2, 1, 0, 0],
            [0, 0, 1, 2, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 2, 1, 0, 0],
            [0, 0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 2, 1, 0, 0],
            [0, 0, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1, 0, 0],
            [0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )


@pytest.fixture()
def skeletonize_circular_bool_int(skeletonize_circular: np.ndarray) -> np.ndarray:
    """Circular molecule for testing skeletonizing as a boolean integer array."""
    return np.array(skeletonize_circular, dtype="bool").astype(int)


@pytest.fixture()
def skeletonize_linear() -> np.ndarray:
    """Linear molecule for testing skeletonizing."""
    return np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 2, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 3, 3, 4, 3, 2, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 3, 3, 4, 4, 3, 2, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 4, 4, 3, 3, 2, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 3, 3, 2, 2, 1, 0, 0],
            [0, 0, 0, 0, 1, 2, 2, 2, 3, 3, 3, 4, 4, 3, 2, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 2, 3, 3, 3, 4, 4, 4, 3, 3, 2, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 3, 3, 3, 2, 2, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 2, 3, 4, 4, 3, 3, 2, 2, 2, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 2, 3, 4, 3, 3, 2, 2, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 2, 3, 4, 3, 3, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 2, 3, 4, 3, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 2, 2, 2, 3, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 2, 3, 3, 3, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 2, 3, 4, 4, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 2, 3, 3, 3, 3, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )


@pytest.fixture()
def skeletonize_linear_bool_int(skeletonize_linear) -> np.ndarray:
    """Linear molecule for testing skeletonizing as a boolean integer array."""
    return np.array(skeletonize_linear, dtype="bool").astype(int)


@pytest.fixture()
def topostats_skeletonise(skeletonize_circular, skeletonize_circular_bool_int):
    """TopostatsSkeletonise for testing individual functions."""
    return topostatsSkeletonize(skeletonize_circular, skeletonize_circular_bool_int, 0.6)


# class CatenaneImageType(TypedDict):
#     """Type for the catenane image dictionary fixture."""

#     image: npt.NDArray[np.number]
#     mask: npt.NDArray[np.bool_]
#     p_to_nm: np.float32


# @pytest.fixture()
# def catenane_image() -> CatenaneImageType:
#     """Image, mask and pixel to nanometre scaling of a catenane molecule."""
#     with h5py.File(RESOURCES / "catenane_image.topostats", "r") as file:
#         image: npt.NDArray[np.number] = file["image"][:]
#         mask: npt.NDArray[np.bool_] = file["grain_masks"]["above"][:]
#         p_to_nm: np.float32 = file["pixel_to_nm_scaling"][()]

#     return {"image": image, "mask": mask, "p_to_nm": p_to_nm}


@pytest.fixture()
def catenane_image() -> npt.NDArray[np.number]:
    """Image of a catenane molecule."""
    return np.load(RESOURCES / "catenane_image.npy")


@pytest.fixture()
def catenane_skeleton() -> npt.NDArray[np.bool_]:
    """Skeleton of the catenane test image."""
    return np.load(RESOURCES / "catenane_skeleton.npy")


@pytest.fixture()
def catenane_smoothed_mask() -> npt.NDArray[np.bool_]:
    """Smoothed mask of the catenane test image."""
    return np.load(RESOURCES / "catenane_smoothed_mask.npy")


# @pytest.fixture()
# def catenane_pruned_skeleton() -> npt.NDArray[np.bool_]:
#     """Pruned skeleton of the catenane test image."""
#     return np.load(RESOURCES / "catenane_pruned_skeleton.npy")


@pytest.fixture()
def catenane_node_centre_mask() -> npt.NDArray[np.int32]:
    """
    Catenane node centre mask.

    Effectively just the skeleton, but
    with the nodes set to 2 while the skeleton is 1 and background is 0.
    """
    return np.load(RESOURCES / "catenane_node_centre_mask.npy")


@pytest.fixture()
def catenane_connected_nodes() -> npt.NDArray[np.int32]:
    """
    Return connected nodes of the catenane test image.

    Effectively just the skeleton, but with the extended nodes
    set to 2 while the skeleton is 1 and background is 0.
    """
    return np.load(RESOURCES / "catenane_connected_nodes.npy")


@pytest.fixture()
def nodestats_catenane(
    catenane_image: npt.NDArray[np.number],
    catenane_smoothed_mask: npt.NDArray[np.bool_],
    catenane_skeleton: npt.NDArray[np.bool_],
    # catenane_pruned_skeleton: npt.NDArray[np.int32],
    catenane_node_centre_mask: npt.NDArray[np.int32],
    catenane_connected_nodes: npt.NDArray[np.int32],
) -> nodeStats:
    """Fixture for the nodeStats object for a catenated molecule, to be used in analyse_nodes."""
    # Create a nodestats object
    nodestats = nodeStats(
        filename="test_catenane",
        image=catenane_image,
        mask=catenane_smoothed_mask,
        smoothed_mask=catenane_smoothed_mask,
        skeleton=catenane_skeleton,
        px_2_nm=np.float64(0.18124609375),
        n_grain=1,
        node_joining_length=7,
        node_extend_dist=14.0,
        branch_pairing_length=20.0,
    )

    nodestats.node_centre_mask = catenane_node_centre_mask
    nodestats.connected_nodes = catenane_connected_nodes
    nodestats.skeleton = catenane_skeleton

    return nodestats


# pylint: disable=unspecified-encoding
@pytest.fixture()
def nodestats_catenane_node_dict() -> dict:
    """Node dictionary for the catenane test image."""
    with Path.open(RESOURCES / "catenane_node_dict.pkl", "rb") as file:
        return pickle.load(file)  # noqa: S301 - Pickles unsafe but we don't care


# pylint: disable=unspecified-encoding
@pytest.fixture()
def nodestats_catenane_image_dict() -> dict:
    """Image dictionary for the catenane test image."""
    with Path.open(RESOURCES / "catenane_image_dict.pkl", "rb") as file:
        return pickle.load(file)  # noqa: S301 - Pickles unsafe but we don't care


@pytest.fixture()
def nodestats_catenane_all_connected_nodes() -> npt.NDArray[np.int32]:
    """All connected nodes for the catenane test image."""
    return np.load(RESOURCES / "catenane_all_connected_nodes.npy")
