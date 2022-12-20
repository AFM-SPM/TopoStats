"""Test the skeletonize module."""
import numpy as np
import pytest

# pytest: disable=
# pytest: disable=import-error
from topostats.tracing.skeletonize import getSkeleton, joeSkeletonize

CIRCULAR_TARGET = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
)


def test_skeletonize_method(skeletonize_circular_bool_int: np.ndarray) -> None:
    """Test unsupported method raises the appropriate error."""
    with pytest.raises(ValueError):
        getSkeleton(None, skeletonize_circular_bool_int).get_skeleton(method="nonsense")


def test_skeletonize_circular_zhang(skeletonize_circular_bool_int: np.ndarray) -> None:
    """Test the Zhang method of skeletionzation on a circular object."""
    test = getSkeleton(None, skeletonize_circular_bool_int).get_skeleton(method="zhang")
    assert isinstance(test, np.ndarray)
    assert test.ndim == 2
    assert test.shape == (21, 21)
    assert test.sum() == 36
    np.testing.assert_array_equal(test, CIRCULAR_TARGET)


def test_skeletonize_circular_lee(skeletonize_circular_bool_int: np.ndarray) -> None:
    """Test the Lee method of skeletonization on a circular object."""
    test = getSkeleton(None, skeletonize_circular_bool_int).get_skeleton(method="lee").astype(int)
    assert isinstance(test, np.ndarray)
    assert test.ndim == 2
    assert test.shape == (21, 21)
    assert test.sum() == 36
    np.testing.assert_array_equal(test, CIRCULAR_TARGET)


def test_skeletonize_circular_medial_axis(skeletonize_circular_bool_int: np.ndarray) -> None:
    """Test the medial axis method of skeletonization on a circular object."""
    CIRCULAR_MEDIAL_AXIS_TARGET = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    test = getSkeleton(None, skeletonize_circular_bool_int).get_skeleton(method="medial_axis").astype(int)
    assert isinstance(test, np.ndarray)
    assert test.ndim == 2
    assert test.shape == (21, 21)
    assert test.sum() == 48
    np.testing.assert_array_equal(test, CIRCULAR_MEDIAL_AXIS_TARGET)


def test_skeletonize_circular_thin(skeletonize_circular_bool_int: np.ndarray) -> None:
    """Test the thin method of skeletonization on a circular object."""
    CIRCULAR_THIN_TARGET = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    test = getSkeleton(None, skeletonize_circular_bool_int).get_skeleton(method="thin").astype(int)
    assert isinstance(test, np.ndarray)
    assert test.ndim == 2
    assert test.shape == (21, 21)
    assert test.sum() == 28
    np.testing.assert_array_equal(test, CIRCULAR_THIN_TARGET)


def test_skeletonize_linear_zha(skeletonize_linear_bool_int: np.ndarray) -> None:
    """Test the Zhang method of skeletonization on a linear object."""
    LINEAR_ZHANG_TARGET = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    test = getSkeleton(None, skeletonize_linear_bool_int).get_skeleton(method="zhang").astype(int)
    assert isinstance(test, np.ndarray)
    assert test.ndim == 2
    assert test.shape == (24, 20)
    assert test.sum() == 20
    np.testing.assert_array_equal(test, LINEAR_ZHANG_TARGET)


def test_skeletonize_linear_lee(skeletonize_linear_bool_int: np.ndarray) -> None:
    """Test the Lee method of skeletonization on a linear object."""
    LINEAR_LEE_TARGET = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    test = getSkeleton(None, skeletonize_linear_bool_int).get_skeleton(method="lee").astype(int)
    assert isinstance(test, np.ndarray)
    assert test.ndim == 2
    assert test.shape == (24, 20)
    assert test.sum() == 17
    np.testing.assert_array_equal(test, LINEAR_LEE_TARGET)


def test_skeletonize_linear_medial_axis(skeletonize_linear_bool_int: np.ndarray) -> None:
    """Test the medial axis method of skeletonization on a linear object."""
    LINEAR_MEDIAL_AXIS_TARGET = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    test = getSkeleton(None, skeletonize_linear_bool_int).get_skeleton(method="medial_axis").astype(int)
    assert isinstance(test, np.ndarray)
    assert test.ndim == 2
    assert test.shape == (24, 20)
    assert test.sum() == 35
    np.testing.assert_array_equal(test, LINEAR_MEDIAL_AXIS_TARGET)


def test_skeletonize_linear_thin(skeletonize_linear_bool_int: np.ndarray) -> None:
    """Test the thin method of skeletonization on a linear object."""
    LINEAR_THIN_TARGET = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    test = getSkeleton(None, skeletonize_linear_bool_int).get_skeleton(method="thin").astype(int)
    assert isinstance(test, np.ndarray)
    assert test.ndim == 2
    assert test.shape == (24, 20)
    assert test.sum() == 11
    np.testing.assert_array_equal(test, LINEAR_THIN_TARGET)


def test_joeSkeletonize_get_local_pixels_binary() -> None:
    """Test the get_local_pixels_binary method of the joeSkeletonize class."""

    image = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
    )

    local_pixels = joeSkeletonize.get_local_pixels_binary(binary_map=image, x=1, y=1)
    assert (local_pixels == np.array([1, 2, 3, 4, 6, 7, 8, 9])).all()

def test_joeSkeletonize_binary_thin_check_a(joeSkeletonize_dummy: joeSkeletonize) -> None:
    """Test the binary_thin_check_a method of the joeSkeletonize class."""
    # pylint: disable=protected-access

    bmap1 = np.array([
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ])

    bmap2 = np.array([
        [1, 0, 0],
        [0, 0, 0],
        [0, 1, 0],
    ])

    bmap3 = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 0, 1],
    ])

    j = joeSkeletonize_dummy

    j.p7, j.p8, j.p9, j.p6, j.p2, j.p5, j.p4, j.p3 = joeSkeletonize.get_local_pixels_binary(
        binary_map=bmap1, x=1, y=1
    )
    t1 = j._binary_thin_check_a()

    j.p7, j.p8, j.p9, j.p6, j.p2, j.p5, j.p4, j.p3 = joeSkeletonize.get_local_pixels_binary(
        binary_map=bmap2, x=1, y=1
    )
    t2 = j._binary_thin_check_a()

    j.p7, j.p8, j.p9, j.p6, j.p2, j.p5, j.p4, j.p3 = joeSkeletonize.get_local_pixels_binary(
        binary_map=bmap3, x=1, y=1
    )
    t3 = j._binary_thin_check_a()

    assert not t1
    assert t2
    assert not t3

def test_joeSkeletonize_binary_thin_check_b_returncount(joeSkeletonize_dummy: joeSkeletonize) -> None:
    """Test the binary_thin_check_b_returncount method of the joeSkeletonize class."""
    # pylint: disable=protected-access

    bmap = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, 0, 0],
    ])

    j = joeSkeletonize_dummy

    j.p7, j.p8, j.p9, j.p6, j.p2, j.p5, j.p4, j.p3 = joeSkeletonize.get_local_pixels_binary(
        binary_map=bmap, x=1, y=1
    )
    t = j._binary_thin_check_b_returncount()

    assert t == 2


def test_joeSkeletonize_binary_thin_check_c(joeSkeletonize_dummy: joeSkeletonize) -> None:
    """Test the binary_thin_check_c method of the joeSkeletonize class."""
    # pylint: disable=protected-access

    bmap1 = np.array([
        [0, 0, 0],
        [1, 0, 1],
        [0, 1, 0],
    ])

    bmap2 = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
    ])

    j = joeSkeletonize_dummy

    j.p7, j.p8, j.p9, j.p6, j.p2, j.p5, j.p4, j.p3 = joeSkeletonize.get_local_pixels_binary(
        binary_map=bmap1, x=1, y=1
    )
    t1 = j._binary_thin_check_c()

    j.p7, j.p8, j.p9, j.p6, j.p2, j.p5, j.p4, j.p3 = joeSkeletonize.get_local_pixels_binary(
        binary_map=bmap2, x=1, y=1
    )
    t2 = j._binary_thin_check_c()

    assert not t1
    assert t2

def test_joeSkeletonize_binary_thin_check_d(joeSkeletonize_dummy: joeSkeletonize) -> None:
    """Test the binary_thin_check_d method of the joeSkeletonize class."""
    # pylint: disable=protected-access

    bmap1 = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 1, 0],
    ])

    bmap2 = np.array([
        [0, 1, 0],
        [0, 0, 0],
        [0, 1, 0],
    ])

    j = joeSkeletonize_dummy

    j.p7, j.p8, j.p9, j.p6, j.p2, j.p5, j.p4, j.p3 = joeSkeletonize.get_local_pixels_binary(
        binary_map=bmap1, x=1, y=1
    )
    t1 = j._binary_thin_check_d()

    j.p7, j.p8, j.p9, j.p6, j.p2, j.p5, j.p4, j.p3 = joeSkeletonize.get_local_pixels_binary(
        binary_map=bmap2, x=1, y=1
    )
    t2 = j._binary_thin_check_d()

    assert not t1
    assert t2

def test_joeSkeletonize_binary_thin_check_csharp(joeSkeletonize_dummy: joeSkeletonize) -> None:
    """Test the binary_thin_check_csharp method of the joeSkeletonize class."""
    # pylint: disable=protected-access

    bmap1 = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 1, 0],
    ])

    bmap2 = np.array([
        [0, 1, 0],
        [0, 0, 0],
        [0, 1, 0],
    ])

    j = joeSkeletonize_dummy

    j.p7, j.p8, j.p9, j.p6, j.p2, j.p5, j.p4, j.p3 = joeSkeletonize.get_local_pixels_binary(
        binary_map=bmap1, x=1, y=1
    )
    t1 = j._binary_thin_check_csharp()

    j.p7, j.p8, j.p9, j.p6, j.p2, j.p5, j.p4, j.p3 = joeSkeletonize.get_local_pixels_binary(
        binary_map=bmap2, x=1, y=1
    )
    t2 = j._binary_thin_check_csharp()

    assert not t1
    assert t2

def test_joeSkeletonize_binary_thin_check_dsharp(joeSkeletonize_dummy: joeSkeletonize) -> None:
    """Test the binary_thin_check_dsharp method of the joeSkeletonize class."""
    # pylint: disable=protected-access

    bmap1 = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 0, 0],
    ])

    bmap2 = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ])

    j = joeSkeletonize_dummy

    j.p7, j.p8, j.p9, j.p6, j.p2, j.p5, j.p4, j.p3 = joeSkeletonize.get_local_pixels_binary(
        binary_map=bmap1, x=1, y=1
    )
    t1 = j._binary_thin_check_dsharp()

    j.p7, j.p8, j.p9, j.p6, j.p2, j.p5, j.p4, j.p3 = joeSkeletonize.get_local_pixels_binary(
        binary_map=bmap2, x=1, y=1
    )
    t2 = j._binary_thin_check_dsharp()

    assert not t1
    assert t2

def test_joeSkeletonize_binary_final_thin_check_a(joeSkeletonize_dummy: joeSkeletonize) -> None:
    """Test the binary_final_thin_check_a method of the joeSkeletonize class."""
    # pylint: disable=protected-access

    bmap1 = np.array([
        [1, 0, 0],
        [1, 0, 1],
        [0, 0, 0],
    ])

    bmap2 = np.array([
        [1, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ])

    j = joeSkeletonize_dummy

    j.p7, j.p8, j.p9, j.p6, j.p2, j.p5, j.p4, j.p3 = joeSkeletonize.get_local_pixels_binary(
        binary_map=bmap1, x=1, y=1
    )
    t1 = j._binary_final_thin_check_a()

    j.p7, j.p8, j.p9, j.p6, j.p2, j.p5, j.p4, j.p3 = joeSkeletonize.get_local_pixels_binary(
        binary_map=bmap2, x=1, y=1
    )
    t2 = j._binary_final_thin_check_a()

    assert not t1
    assert t2

def test_joeSkeletonize_binary_final_thin_check_b(joeSkeletonize_dummy: joeSkeletonize) -> None:
    """Test the binary_final_thin_check_b method of the joeSkeletonize class."""
    # pylint: disable=protected-access

    bmap1 = np.array([
        [1, 0, 0],
        [1, 0, 1],
        [0, 0, 1],
    ])

    bmap2 = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [0, 0, 0],
    ])

    j = joeSkeletonize_dummy

    j.p7, j.p8, j.p9, j.p6, j.p2, j.p5, j.p4, j.p3 = joeSkeletonize.get_local_pixels_binary(
        binary_map=bmap1, x=1, y=1
    )
    t1 = j._binary_final_thin_check_b()

    j.p7, j.p8, j.p9, j.p6, j.p2, j.p5, j.p4, j.p3 = joeSkeletonize.get_local_pixels_binary(
        binary_map=bmap2, x=1, y=1
    )
    t2 = j._binary_final_thin_check_b()

    assert not t1
    assert t2

def test_joeSkeletonize_delete_pixel_subit1() -> None:
    """Test the delete_pixel_subit1 method of the joeSkeletonize class."""
    # pylint: disable=protected-access

    image = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ])

    mask1 = np.array([
        [0, 0, 1],
        [0, 1, 1],
        [0, 1, 1],
    ])

    mask2 = np.array([
        [1, 0, 1],
        [0, 1, 1],
        [0, 1, 1],
    ])

    mask3 = np.array([
        [0, 0, 1],
        [1, 1, 1],
        [1, 1, 1],
    ])

    mask4 = np.array([
        [0, 1, 0],
        [1, 1, 0],
        [1, 1, 1],
    ])

    # Successful.
    j = joeSkeletonize(image=image, mask=mask1)
    t1 = j._delete_pixel_subit1([1, 1])

    # Failure due to more than one [0, 1] in surrounding ring of pixels.
    j = joeSkeletonize(image=image, mask=mask2)
    t2 = j._delete_pixel_subit1([1, 1])

    # Failure due to none of p2, p4, p6 being zero.
    j = joeSkeletonize(image=image, mask=mask3)
    t3 = j._delete_pixel_subit1([1, 1])

    # Failure due to none of p4, p6, p8 being zero.
    j = joeSkeletonize(image=image, mask=mask4)
    t4 = j._delete_pixel_subit1([1, 1])


    assert t1
    assert not t2
    assert not t3
    assert not t4

def test_joeSkeletonize_delete_pixel_subit2() -> None:

    assert False

def test_joeSkeletonize_do_skeletonising() -> None:
    """Test the do_skeletonising method of the joeSkeletonize class."""

    assert False

def test_joeSkeletonize_do_skeletonizing_iteration() -> None:
    """Test the do_skeletonising_iteration method of the joeSkeletonize class."""

    assert False

def test_joeSkeletonize_final_skeletonization_iteration() -> None:

    assert False
