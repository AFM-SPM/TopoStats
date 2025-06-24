"""Fixtures for the tracing tests."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from topostats.classes import GrainCrop, GrainCropsDirection, ImageGrainCrops, TopoStats
from topostats.tracing.disordered_tracing import disorderedTrace, trace_image_disordered
from topostats.tracing.nodestats import nodeStats
from topostats.tracing.skeletonize import getSkeleton, topostatsSkeletonize

# This is required because of the inheritance used throughout
# pylint: disable=redefined-outer-name

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"
TRACING_RESOURCES = RESOURCES / "tracing"
NODESTATS_RESOURCES = TRACING_RESOURCES / "nodestats"

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


@pytest.fixture()
def topostats_catenane(catenane_image: npt.NDArray[np.number], disordered_tracing_config: dict) -> GrainCrop:
    """TopoStats of Catenane post disordered tracing."""
    # The catenane image isn't square and currently GrainCrop only plays ball with square crops so we pad it with zeros
    # on one side
    catenane_image = np.pad(
        catenane_image, pad_width=((0, 0), (0, catenane_image.shape[0] - catenane_image.shape[1])), mode="constant"
    )
    catenane_mask = catenane_image.astype(np.bool).astype(int)
    catenane_tensor = np.stack([catenane_mask, catenane_mask], axis=-1)
    # Add the graincrop to a TopoStats object as this is the unit passed around on which trace_image_disordered and
    # nodestats runs
    topostats_object = TopoStats(
        image_grain_crops=ImageGrainCrops(
            above=GrainCropsDirection(
                full_mask_tensor=catenane_tensor,
                crops={
                    0: GrainCrop(
                        image=catenane_image,
                        mask=catenane_tensor,
                        filename="test_catenane",
                        padding=1,
                        bbox=(0, 0, 10, 10),
                        pixel_to_nm_scaling=1,
                    )
                },
            ),
            below=None,
        ),
        filename="test_catenane",
        img_path=None,
        # pixel_to_nm_scaling=1,
    )
    print(f"BEFORE\n{topostats_object.image_grain_crops.above.crops[0].disordered_trace=}\n")
    print(f"\n{disordered_tracing_config=}\n")
    (_, _, _, _) = trace_image_disordered(
        topostats_object=topostats_object,
        direction="above",
        **disordered_tracing_config,
    )
    # trace_image_disordered(topostats_object=topostats_object, direction="above", **disordered_tracing_config)
    print("WE SHOULD HAVE DISORDERED TRACES!!!!")
    print(f"AFTER\n{topostats_object.image_grain_crops.above.crops[0].disordered_trace=}\n")
    return topostats_object


@pytest.fixture()
def graincrop_catenane(catenane_image: npt.NDArray[np.number]) -> GrainCrop:
    """GrainCrop of Catenane post disordered tracing."""
    # The catenane image isn't square and currently GrainCrop only plays ball with square crops so we pad it with zeros
    # on one side
    catenane_image = np.pad(
        catenane_image, pad_width=((0, 0), (0, catenane_image.shape[0] - catenane_image.shape[1])), mode="constant"
    )
    catenane_mask = catenane_image.astype(np.bool).astype(int)
    catenane_tensor = np.stack([catenane_mask, catenane_mask], axis=-1)
    return GrainCrop(
        image=catenane_image,
        mask=catenane_tensor,
        filename="test_catenane",
        padding=1,
        bbox=(0, 0, 10, 10),
        pixel_to_nm_scaling=1,
    )


@pytest.fixture()
def nodestats_catenane(
    # catenane_image: npt.NDArray[np.number],
    # catenane_smoothed_mask: npt.NDArray[np.bool_],
    # catenane_skeleton: npt.NDArray[np.bool_],
    topostats_catenane: TopoStats,
    catenane_node_centre_mask: npt.NDArray[np.int32],
    catenane_connected_nodes: npt.NDArray[np.int32],
) -> nodeStats:
    """Fixture for the nodeStats object for a catenated molecule, to be used in analyse_nodes."""
    # Pull the catenane GrainCrop out of the TopoStats topostats_catenane object
    print(f"\nTOPOSTATS\n{topostats_catenane.__dict__=}")
    graincrop_catenane = topostats_catenane.image_grain_crops.above.crops[0]
    print(f"\nGRAINCROP\n{graincrop_catenane.__dict__=}\n")
    # Create a nodestats object
    nodestats = nodeStats(
        # filename="test_catenane",
        # image=catenane_image,
        # mask=catenane_smoothed_mask,
        # smoothed_mask=catenane_smoothed_mask,
        # skeleton=catenane_skeleton,
        # pixel_to_nm_scaling=np.float64(0.18124609375),
        # n_grain=1,
        graincrop=graincrop_catenane,
        node_joining_length=7,
        node_extend_dist=14.0,
        branch_pairing_length=20.0,
        pair_odd_branches=True,
    )

    nodestats.node_centre_mask = catenane_node_centre_mask
    nodestats.connected_nodes = catenane_connected_nodes
    nodestats.skeleton = catenane_skeleton

    return nodestats


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
    )
