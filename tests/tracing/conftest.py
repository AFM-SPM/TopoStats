"""Fixtures for the tracing tests."""

import pickle as pkl
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from topostats.classes import GrainCrop, MatchedBranch, Molecule, OrderedTrace, TopoStats
from topostats.tracing.disordered_tracing import disorderedTrace
from topostats.tracing.skeletonize import getSkeleton, topostatsSkeletonize

# This is required because of the inheritance used throughout
# pylint: disable=redefined-outer-name,unspecified-encoding

# ruff: noqa: S301
BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"
TRACING_RESOURCES = RESOURCES / "tracing"
NODESTATS_RESOURCES = TRACING_RESOURCES / "nodestats"
ORDERED_TRACING_RESOURCES = TRACING_RESOURCES / "ordered_tracing"

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


@pytest.fixture()
def disordered_trace(skeletonize_circular, skeletonize_circular_bool_int) -> disorderedTrace:
    """Minimal disorderedTrace class object."""
    return disorderedTrace(
        image=skeletonize_circular, mask=skeletonize_circular_bool_int, filename=None, pixel_to_nm_scaling=1
    )


@pytest.fixture()
def catenane_image() -> npt.NDArray[np.number]:
    """Image of a catenane molecule."""
    return np.load(NODESTATS_RESOURCES / "catenane_image.npy")


@pytest.fixture()
def catenane_node_centre_mask() -> npt.NDArray[np.int32]:
    """
    Catenane node centre mask.

    Effectively just the skeleton, but
    with the nodes set to 2 while the skeleton is 1 and background is 0.
    """
    return np.load(NODESTATS_RESOURCES / "catenane_node_centre_mask.npy")


@pytest.fixture()
def catenane_connected_nodes() -> npt.NDArray[np.int32]:
    """
    Return connected nodes of the catenane test image.

    Effectively just the skeleton, but with the extended nodes
    set to 2 while the skeleton is 1 and background is 0.
    """
    return np.load(NODESTATS_RESOURCES / "catenane_connected_nodes.npy")


@pytest.fixture()
def catenane_smoothed_mask() -> npt.NDArray[np.bool_]:
    """Catenane smoothed mask."""
    return np.load(NODESTATS_RESOURCES / "catenane_smoothed_mask.npy")


@pytest.fixture()
def catenane_skeleton() -> npt.NDArray[np.bool_]:
    """Catenane smoothed mask."""
    return np.load(NODESTATS_RESOURCES / "catenane_skeleton.npy")


# @pytest.fixture()
# def nodestats_minicircle_small(
#     graincrop_minicircle_small: GrainCrop,
#     minicircle_small_post_disordered_trace: TopoStats,
# ) -> nodeStats:
#     """Fixture for the nodeStats object for Minicircle Small, to be used in analyse_nodes."""
#     return nodeStats(
#         grain_crop=graincrop_minicircle_small,
#         n_grain=0,
#         node_joining_length=7,
#         node_extend_dist=14.0,
#         branch_pairing_length=20.0,
#         pair_odd_branches=True,
#     )

# @pytest.fixture()
# def nodestats_catenane(
#     graincrop_catenane: GrainCrop,
#     catenane_post_disordered_trace: TopoStats,
# ) -> nodeStats:
#     """Fixture for the nodeStats object for a Catenane molecule, to be used in analyse_nodes."""
#     return nodeStats(
#         grain_crop=graincrop_catenane,
#         n_grain=0,
#         node_joining_length=7,
#         node_extend_dist=14.0,
#         branch_pairing_length=20.0,
#         pair_odd_branches=True,
#     )

# @pytest.fixture()
# def nodestats_rep_int(
#     graincrop_rep_int: GrainCrop,
#     rep_int_post_disordered_trace: TopoStats,
# ) -> nodeStats:
#     """Fixture for the nodeStats object for a Rep Int molecule, to be used in analyse_nodes."""
#     return nodeStats(
#         grain_crop=graincrop_rep_int,
#         n_grain=0,
#         node_joining_length=7,
#         node_extend_dist=14.0,
#         branch_pairing_length=20.0,
#         pair_odd_branches=True,
#     )


@pytest.fixture()
def grain_crop_curved_line() -> GrainCrop:
    """GrainCrop of a simple curved line."""
    return GrainCrop(
        image=np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 2, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 2, 1, 0, 0, 0, 0],
                [0, 0, 1, 1, 2, 1, 0, 0, 0, 0],
                [0, 0, 1, 2, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 2, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 2, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 2, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 2, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        ),
        mask=np.stack(
            arrays=[
                np.array(
                    [
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                        [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ]
                ),
                np.array(
                    [
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                        [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ]
                ),
            ],
            axis=-1,
        ),
        padding=1,
        bbox=(0, 0, 10, 10),
        pixel_to_nm_scaling=1,
        filename="simple slightly curved line",
        thresholds=None,
    )


@pytest.fixture()
def catenane_node0_matched_branches() -> MatchedBranch:
    """Catenane ``MatchedBranch`` dictionary."""
    with Path(NODESTATS_RESOURCES / "catenane_node_0_matched_branches_analyse_node_branches.pkl").open("rb") as f:
        matched_branches: dict[int, dict[str, npt.NDArray[np.number]]] = pkl.load(f)
    return {
        key: MatchedBranch(
            ordered_coords=value["ordered_coords"],
            heights=value["heights"],
            distances=value["distances"],
            fwhm=value["fwhm"]["fwhm"],
            fwhm_half_maxs=value["fwhm"]["half_maxs"],
            fwhm_peaks=value["fwhm"]["peaks"],
            angles=value["angles"],
        )
        for key, value in matched_branches.items()
    }


@pytest.fixture()
def catenane_node0_masked_images() -> dict[int, dict[str, npt.NDArray[np.bool]]]:
    """Averaged masked images for catenane."""
    with Path(NODESTATS_RESOURCES / "catenane_node_0_masked_image.pkl").open("rb") as f:
        return pkl.load(f)


@pytest.fixture()
def catenane_node0_ordered_branches() -> list[npt.NDArray[np.int32]]:
    """Ordered branches for catenane (node 0)."""
    with Path(NODESTATS_RESOURCES / "catenane_node_0_ordered_branches.pkl").open("rb") as f:
        return pkl.load(f)


@pytest.fixture()
def catenane_splining() -> TopoStats:
    """Catenane TopoStats object with necessary data for testing splining."""
    # Load the data
    image = np.load(TRACING_RESOURCES / "example_catenanes.npy")

    # Load the ordered tracing direction data
    with Path.open(ORDERED_TRACING_RESOURCES / "catenanes_ordered_tracing_data.pkl", "rb") as file:
        ordered_tracing_direction_data = pkl.load(file)
    # Construct TopoStats objects
    grain_crops = {}
    for grain_no, grain_data in ordered_tracing_direction_data.items():
        molecules = {}
        for mol_no, molecule_data in grain_data.items():
            # data = ordered_tracing_direction_data[grain_no][key]
            molecules[mol_no[-1:]] = Molecule(
                circular=molecule_data["mol_stats"]["circular"],
                topology=molecule_data["mol_stats"]["topology"],
                topology_flip=molecule_data["mol_stats"]["topology_flip"],
                ordered_coords=molecule_data["ordered_coords"],
                heights=molecule_data["heights"],
                distances=molecule_data["distances"],
                bbox=molecule_data["bbox"],
            )
        ordered_trace = OrderedTrace(molecule_data=molecules)
        grain_crops[grain_no[-1:]] = GrainCrop(
            image=np.zeros_like(image),
            mask=np.stack([np.zeros_like(image), np.zeros_like(image)], axis=2),
            padding=1,
            thresholds=None,
            pixel_to_nm_scaling=1.0,
            filename="catenane",
            ordered_trace=ordered_trace,
            # This is wrong as it comes from the molecule not the grain but loaded data
            # doesn't have bbox for the whole grain so we use this and ignore it.
            bbox=molecule_data["bbox"],  # pylint: disable=undefined-loop-variable
        )
    return TopoStats(
        image=image,
        filename="catenane",
        pixel_to_nm_scaling=1.0,
        grain_crops=grain_crops,
    )


@pytest.fixture()
def rep_int_splining() -> TopoStats:
    """Catenane TopoStats object with necessary data for testing splining."""
    # Load the data
    image = np.load(TRACING_RESOURCES / "example_rep_int.npy")

    # Load the ordered tracing direction data
    with Path.open(ORDERED_TRACING_RESOURCES / "rep_int_ordered_tracing_data.pkl", "rb") as file:
        ordered_tracing_direction_data = pkl.load(file)
    # Construct TopoStats objects
    grain_crops = {}
    for grain_no, grain_data in ordered_tracing_direction_data.items():
        molecules = {}
        for mol_no, molecule_data in grain_data.items():
            # data = ordered_tracing_direction_data[grain_no][key]
            molecules[mol_no[-1:]] = Molecule(
                circular=molecule_data["mol_stats"]["circular"],
                topology=molecule_data["mol_stats"]["topology"],
                topology_flip=molecule_data["mol_stats"]["topology_flip"],
                ordered_coords=molecule_data["ordered_coords"],
                heights=molecule_data["heights"],
                distances=molecule_data["distances"],
                bbox=molecule_data["bbox"],
            )
        ordered_trace = OrderedTrace(molecule_data=molecules)
        grain_crops[grain_no[-1:]] = GrainCrop(
            image=np.zeros_like(image),
            mask=np.stack([np.zeros_like(image), np.zeros_like(image)], axis=2),
            padding=1,
            thresholds=None,
            pixel_to_nm_scaling=1.0,
            filename="replication_intermediate",
            ordered_trace=ordered_trace,
            # This is wrong as it comes from the molecule not the grain but loaded data
            # doesn't have bbox for the whole grain so we use this and ignore it.
            bbox=molecule_data["bbox"],  # pylint: disable=undefined-loop-variable
        )
    return TopoStats(
        image=image,
        filename="replication_intermediate",
        pixel_to_nm_scaling=1.0,
        grain_crops=grain_crops,
    )
