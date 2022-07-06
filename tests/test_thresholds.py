"""Test of thresholds"""
import pytest
import numpy as np
from skimage.filters import threshold_mean, threshold_minimum, threshold_otsu, threshold_yen, threshold_triangle

from topostats.thresholds import threshold

OPTIONS = {
    'nbins': 10,
}


def test_threshold_invalid_method(image_random: np.array) -> None:
    """Test exceptions is raised when invalid method supplied.

    Parameters
    ----------
    image_random : np.array
        Numpy array representing an image.
    """
    with pytest.raises(ValueError):
        threshold_invalid = threshold(image_random, method='shoes')


def test_threshold_otsu(image_random: np.array) -> None:
    """Test the Otsu threshold method.

    Parameters
    ----------
    image_random : np.array
        Numpy array representing an image.
    """
    _threshold = threshold(image_random, method='otsu', otsu_threshold_multiplier=1.7)

    assert isinstance(_threshold, float)
    assert _threshold == threshold_otsu(image_random)


def test_threshold_otsu_keywords(image_random: np.array) -> None:
    """Test the Otsu threshold method with additional keywords.

    Parameters
    ----------
    image_random : np.array
        Numpy array representing an image.
    """
    _threshold = threshold(image_random, method='otsu', otsu_threshold_multiplier=1.7, **OPTIONS)

    assert isinstance(_threshold, float)
    assert _threshold == threshold_otsu(image_random, **OPTIONS)


def test_threshold_minimum(image_random: np.array) -> None:
    """Test the Minimum threshold method.

    Parameters
    ----------
    image_random : np.array
        Numpy array representing an image.
    """
    _threshold = threshold(image_random, method='minimum')

    assert isinstance(_threshold, float)
    assert _threshold == threshold_minimum(image_random)


# def test__threshold_keywords(image_random: np.array) -> None:
#     """Test the Minimum threshold method with additional keywords.

#     Parameters
#     ----------
#     image_random : np.array
#         Numpy array representing an image.
#     """
#     _threshold = threshold(image_random, method='minimum', **OPTIONS)

#     assert isinstance(_threshold, float)
#     assert _threshold == threshold_minimum(image_random, **OPTIONS)


def test_threshold_mean(image_random: np.array) -> None:
    """Test the Mean threshold method.

    Parameters
    ----------
    image_random : np.array
        Numpy array representing an image.
    """
    _threshold = threshold(image_random, method='mean')

    assert isinstance(_threshold, float)
    assert _threshold == threshold_mean(image_random)


def test_threshold_yen(image_random: np.array) -> None:
    """Test the Yen threshold method.

    Parameters
    ----------
    image_random : np.array
        Numpy array representing an image.
    """
    _threshold = threshold(image_random, method='yen')

    assert isinstance(_threshold, float)
    assert _threshold == threshold_yen(image_random)


def test_threshold_yen_keywords(image_random: np.array) -> None:
    """Test the Yen threshold method with additional keywords.

    Parameters
    ----------
    image_random : np.array
        Numpy array representing an image.
    """
    _threshold = threshold(image_random, method='yen', **OPTIONS)

    assert isinstance(_threshold, float)
    assert _threshold == threshold_yen(image_random, **OPTIONS)


def test_threshold_triangle(image_random: np.array) -> None:
    """Test the Triangle threshold method.

    Parameters
    ----------
    image_random : np.array
        Numpy array representing an image.
    """
    _threshold = threshold(image_random, method='triangle')

    assert isinstance(_threshold, float)
    assert _threshold == threshold_triangle(image_random)


def test_threshold_triangle_keywords(image_random: np.array) -> None:
    """Test the Triangle threshold method with additional keywords.

    Parameters
    ----------
    image_random : np.array
        Numpy array representing an image.
    """
    _threshold = threshold(image_random, method='triangle', **OPTIONS)

    assert isinstance(_threshold, float)
    assert _threshold == threshold_triangle(image_random, **OPTIONS)
