"""Fixtures for the tracing tests."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from skimage import draw, filters
from skimage.morphology import skeletonize

from topostats.filters import Filters
from topostats.grains import Grains
from topostats.tracing.dnatracing import dnaTrace
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


# Skeletons are generated by...
#
# 1. Generate random boolean images using scikit-image.
# 2. Skeletonize these shapes (gives boolean skeletons), these are our targets that tracing should recreate.
# 3. Scale the skeletons by a factor (100)
# 4. Apply Gaussian filter to blur the heights and give an example original image of heights and a known skeleton to aim
#    for.


def _generate_heights(skeleton: npt.NDArray, scale: float = 100, sigma: float = 5.0, cval: float = 20.0) -> npt.NDArray:
    """Generate heights from skeletons by scaling image and applying Gaussian blurring.

    Uses scikit-image 'skimage.filters.gaussian()' to generate heights from skeletons.

    Parameters
    ----------
    skeleton : npt.NDArray
        Binary array of skeleton.
    scale : float
        Factor to scale heights by. Boolean arrays are 0/1 and so the factor will be the height of the skeleton ridge.
    sigma : float
        Standard deviation for Gaussian kernel passed to `skimage.filters.gaussian()'.
    cval : float
        Value to fill past edges of input, passed to `skimage.filters.gaussian()'.

    Returns
    -------
    npt.NDArray
        Array with heights of image based on skeleton which will be the backbone and target.
    """
    return filters.gaussian(skeleton * scale, sigma=sigma, cval=cval)


def _generate_random_skeleton(**extra_kwargs):
    """Generate random skeletons and heights using skimage.draw's random_shapes()."""
    kwargs = {
        "image_shape": (128, 128),
        "max_shapes": 20,
        "channel_axis": None,
        "shape": None,
        "allow_overlap": True,
    }
    heights = {"scale": 100, "sigma": 5.0, "cval": 20.0}
    random_image, _ = draw.random_shapes(**kwargs, **extra_kwargs)
    mask = random_image != 255
    skeleton = skeletonize(mask)
    return {"img": _generate_heights(skeleton, **heights), "skeleton": skeleton}


@pytest.fixture()
# def pruning_skeleton_loop1(heights=heights) -> dict:
def pruning_skeleton_loop1() -> dict:
    """Skeleton with loop to be retained and side-branches."""
    return _generate_random_skeleton(rng=1, min_size=20)


@pytest.fixture()
def pruning_skeleton_loop2() -> dict:
    """Skeleton with loop to be retained and side-branches."""
    return _generate_random_skeleton(rng=165103, min_size=60)


@pytest.fixture()
def pruning_skeleton_linear1() -> dict:
    """Linear skeleton with lots of large side-branches, some forked."""
    return _generate_random_skeleton(rng=13588686514, min_size=20)


@pytest.fixture()
def pruning_skeleton_linear2() -> dict:
    """Linear Skeleton with simple fork at one end."""
    return _generate_random_skeleton(rng=21, min_size=20)


@pytest.fixture()
def pruning_skeleton_linear3() -> dict:
    """Linear Skeletons (i.e. multiple) with branches."""
    return _generate_random_skeleton(rng=894632511, min_size=20)


## Helper functions for skeletons and heights


# def pruned_plot(gen_shape: dict) -> None:
#     """Plot the original skeleton, its derived height and the pruned skeleton."""
#     img_skeleton = gen_shape()
#     pruned = topostatsPrune(
#         img_skeleton["heights"],
#         img_skeleton["skeleton"],
#         max_length=-1,
#         height_threshold=90,
#         method_values="min",
#         method_outlier="abs",
#     )
#     pruned_skeleton = pruned._prune_by_length(pruned.skeleton, pruned.max_length)
#     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
#     ax1.imshow(img_skeleton["skeleton"])
#     ax2.imshow(img_skeleton["heights"])
#     ax3.imshow(pruned_skeleton)
#     plt.show()


# pruned_plot(pruning_skeleton_loop1)
# pruned_plot(pruning_skeleton_loop2)
# pruned_plot(pruning_skeleton_linear1)
# pruned_plot(pruning_skeleton_linear2)
# pruned_plot(pruning_skeleton_linear3)
