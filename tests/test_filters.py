"""Tests of the filters module."""
import numpy as np

from topostats.filters import amplify, row_col_quantiles, align_rows, remove_x_y_tilt, get_threshold, get_mask

# Specify the absolute and relattive tolerance for floating point comparison
TOLERANCE = {'atol': 1e-07, 'rtol': 1e-07}


def test_amplify(image_random: np.array) -> None:
    """Test amplification filter."""
    filtered = amplify(image_random, 1.5)

    target = image_random * 1.5

    np.testing.assert_array_equal(filtered, target)

# FIXME : Add tests with masking
def test_row_col_quantiles(image_random: np.array,
                           image_random_row_quantiles: np.array,
                           image_random_col_quantiles: np.array) -> None:
    """Test generation of quantiles for rows and columns.
    """
    row_quantiles, col_quantiles = row_col_quantiles(image_random, mask=None)

    np.testing.assert_array_equal(row_quantiles, image_random_row_quantiles)
    np.testing.assert_array_equal(col_quantiles, image_random_col_quantiles)


def test_align_rows(image_random: np.array,
                    image_random_aligned_rows: np.array) -> None:
    """Test aligning of rows by median height.
    """
    aligned_rows = align_rows(image_random, mask=None)

    np.testing.assert_allclose(aligned_rows, image_random_aligned_rows,
                               **TOLERANCE)


def test_remove_x_y_tilt(image_random: np.array,
                         image_random_remove_x_y_tilt: np.array) -> None:
    """Test removal of linear plane slant."""
    x_y_tilt_removed = remove_x_y_tilt(image_random, mask=None)

    np.testing.assert_allclose(x_y_tilt_removed, image_random_remove_x_y_tilt,
                               **TOLERANCE)


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

def test_row_col_quantiles_with_mask(image_random: np.array,
                                     image_random_mask: np.array,
                                     image_random_row_quantiles_masked: np.array,
                                     image_random_col_quantiles_masked: np.array) -> None:
    """Test generation of quantiles for rows and columns.
    """
    row_quantiles, col_quantiles = row_col_quantiles(image_random, mask=image_random_mask)
    print('########## ROW ')
    print(row_quantiles.data)
    print(f'row_quantiles.data : \n{row_quantiles.data}')
    print('\n\n\n###### FROM FILE')
    print(f'image_random_row_quantiles_masked : \n{image_random_row_quantiles_masked}')
    # Remove masked values for comparison
    row_quantiles = row_quantiles.data
    col_quantiles = col_quantiles.data
    print(row_quantiles)

    np.testing.assert_array_equal(row_quantiles.data, image_random_row_quantiles_masked)
    np.testing.assert_array_equal(col_quantiles, image_random_col_quantiles_masked)
    assert False
