"""Tests of the filters module."""
import numpy as np

from topostats.filters import amplify, row_col_quantiles, align_rows, remove_x_y_tilt, get_threshold

# Specify the absolute and relattive tolerance for floating point comparison
TOLERANCE = {'atol': 1e-07, 'rtol': 1e-07}


def test_amplify(image_random: np.array) -> None:
    """Test amplification filter."""
    filtered = amplify(image_random, 1.5)

    target = image_random * 1.5

    np.testing.assert_array_equal(filtered, target)


def test_row_col_quantiles(image_random: np.array,
                           image_random_row_quantiles: np.array,
                           image_random_col_quantiles: np.array) -> None:
    """Test generation of quantiles for rows and columns.
    """
    row_quantiles, col_quantiles = row_col_quantiles(image_random)

    np.testing.assert_array_equal(row_quantiles, image_random_row_quantiles)
    np.testing.assert_array_equal(col_quantiles, image_random_col_quantiles)


def test_align_rows(image_random: np.array,
                    image_random_aligned_rows: np.array) -> None:
    """Test aligning of rows by median height.
    """
    aligned_rows = align_rows(image_random)

    np.testing.assert_allclose(aligned_rows, image_random_aligned_rows,
                               **TOLERANCE)


def test_remove_x_y_tilt(image_random: np.array,
                         image_random_remove_x_y_tilt: np.array) -> None:
    """Test removal of linear plane slant.
    """
    x_y_tilt_removed = remove_x_y_tilt(image_random)

    np.testing.assert_allclose(x_y_tilt_removed, image_random_remove_x_y_tilt,
                               **TOLERANCE)


def test_get_threshold(image_random: np.array):
    """Test calculation of threshold
    """
    image_threshold = get_threshold(image_random)
    expected_threshold = 0.4980470463263117

    assert image_threshold == expected_threshold
