"""Tests of the filters module."""
#+ pylint: disable=no-name-in-module
#+ pylint: disable=invalid-name
import numpy as np

from skimage.filters import gaussian
from topostats.filters import (amplify, row_col_quantiles, align_rows, remove_x_y_tilt, get_threshold, get_mask,
                               average_background)
from topostats.find_grains import quadratic

# Specify the absolute and relattive tolerance for floating point comparison
TOLERANCE = {'atol': 1e-07, 'rtol': 1e-07}


def test_amplify(image_random: np.array) -> None:
    """Test amplification filter."""
    filtered = amplify(image_random, 1.5)
    target = image_random * 1.5

    np.testing.assert_array_equal(filtered, target)


def test_row_col_quantiles_small_array_no_mask(small_array: np.array) -> None:
    """Test generation of quantiles for rows and columns without masking."""
    row_quantiles, col_quantiles = row_col_quantiles(small_array, mask=None)
    expected_rows = np.quantile(small_array, [0.25, 0.5, 0.75], axis=1).T
    expected_cols = np.quantile(small_array, [0.25, 0.5, 0.75], axis=0).T

    np.testing.assert_array_equal(row_quantiles, expected_rows)
    np.testing.assert_array_equal(col_quantiles, expected_cols)


def test_row_col_quantiles_small_array_mask(small_array: np.array, small_mask: np.array) -> None:
    """Test generation of quantiles for rows and columns."""
    row_quantiles, col_quantiles = row_col_quantiles(small_array, mask=small_mask)
    small_array_masked = np.ma.masked_array(small_array, mask=small_mask, fill_value=np.nan)
    expected_rows = np.quantile(small_array_masked, [0.25, 0.5, 0.75], axis=1).T
    expected_cols = np.quantile(small_array_masked, [0.25, 0.5, 0.75], axis=0).T

    np.testing.assert_array_equal(row_quantiles, expected_rows)
    np.testing.assert_array_equal(col_quantiles, expected_cols)

    expected_rows = np.nanpercentile(small_array_masked, (25, 50, 75), axis=1).T
    expected_cols = np.nanpercentile(small_array_masked, (25, 50, 75), axis=0).T
    np.testing.assert_array_equal(row_quantiles, expected_rows)
    np.testing.assert_array_equal(col_quantiles, expected_cols)


def test_row_col_quantiles(image_random: np.array, image_random_row_quantiles: np.array,
                           image_random_col_quantiles: np.array) -> None:
    """Test generation of quantiles for rows and columns.
    """
    row_quantiles, col_quantiles = row_col_quantiles(image_random, mask=None)

    np.testing.assert_array_equal(row_quantiles, image_random_row_quantiles)
    np.testing.assert_array_equal(col_quantiles, image_random_col_quantiles)


def test_align_rows(image_random: np.array, image_random_aligned_rows: np.array) -> None:
    """Test aligning of rows by median height.
    """
    aligned_rows = align_rows(image_random, mask=None)

    np.testing.assert_allclose(aligned_rows, image_random_aligned_rows, **TOLERANCE)


def test_remove_x_y_tilt(image_random: np.array, image_random_remove_x_y_tilt: np.array) -> None:
    """Test removal of linear plane slant."""
    x_y_tilt_removed = remove_x_y_tilt(image_random, mask=None)

    np.testing.assert_allclose(x_y_tilt_removed, image_random_remove_x_y_tilt, **TOLERANCE)


def test_get_threshold(image_random: np.array):
    """Test calculation of threshold."""
    threshold = get_threshold(image_random)
    expected_threshold = 0.4980470463263117

    assert threshold == expected_threshold


def test_get_mask(image_random: np.array, image_random_mask: np.array):
    """Test generation of mask"""
    threshold = get_threshold(image_random)
    mask = get_mask(image_random, threshold)

    np.testing.assert_array_equal(mask, image_random_mask)


def test_row_col_quantiles_with_mask(image_random: np.array, image_random_mask: np.array,
                                     image_random_row_quantiles_masked: np.array,
                                     image_random_col_quantiles_masked: np.array) -> None:
    """Test generation of quantiles for rows and columns.
    """
    row_quantiles, col_quantiles = row_col_quantiles(image_random, mask=image_random_mask)
    row_quantiles = np.ma.getdata(row_quantiles)
    col_quantiles = np.ma.getdata(col_quantiles)

    np.testing.assert_array_equal(row_quantiles.data, image_random_row_quantiles_masked)
    np.testing.assert_array_equal(col_quantiles, image_random_col_quantiles_masked)


def test_quadratic(small_array) -> None:
    """Test quadratic function."""
    values = {'a': 2, 'b': 3, 'c': 4}
    small_array_quadratic = quadratic(small_array, values['a'], values['b'], values['c'])
    target = (values['a'] * small_array**2) + (values['b'] * small_array) + values['c']

    np.testing.assert_array_equal(target, small_array_quadratic)


def test_average_background(image_random: np.array, image_random_mask: np.array) -> None:
    """Test averaging of background."""
    background_averaged = average_background(image_random, image_random_mask)
    row_quantiles, _ = row_col_quantiles(image_random, mask=image_random_mask)
    target = image_random - np.array(row_quantiles[:, 1], ndmin=2).T

    np.testing.assert_array_equal(target, background_averaged)


# def test_curve_fit(small_array:np.array) -> None:
#     """Test fitting of curve."""
#     fitted_curve = curve_fit(np.mean(small_array, axis=1))
#     target = curve_fit(f=quadratic,
#                        xdata=np.arange(0, small_array.shape[0]),
#                        ydata=np.mean(small_array, axis=1))
#     print(f'np.mean(small_array, axis=1) :\n{np.mean(small_array, axis=1)}')
#     print(f'fited_curve                  :\n{fitted_curve}')
#     print(f'target                       :\n{target}')
#     np.testing.assert_array_equal(target, fitted_curve)
#     assert False

# def test_remove_x_bowing(image_random: np.array,
#                          image_random_mask: np.array) -> None:
#     """Test removal of x bowing."""
#     image_random_masked = np.ma.masked_array(image_random,
#                                              mask=image_random_mask,
#                                              fill_value=np.nan)
#     assert True
