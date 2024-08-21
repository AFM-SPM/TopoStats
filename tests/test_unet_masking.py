"""Test unet masking methods."""

import pytest

from topostats.unet_masking import make_crop_square, pad_bounding_box


@pytest.mark.parametrize(
    ("crop_min_row", "crop_min_col", "crop_max_row", "crop_max_col", "image_shape", "expected_indices"),
    [
        pytest.param(0, 0, 100, 100, (100, 100), (0, 0, 100, 100), id="not a crop"),
        pytest.param(3, 4, 8, 8, (10, 10), (3, 4, 8, 9), id="free space single min col decrease"),
        pytest.param(4, 3, 8, 8, (10, 10), (4, 3, 9, 8), id="free space single min row decrease"),
        pytest.param(4, 4, 7, 8, (10, 10), (4, 4, 8, 8), id="free space single max col increase"),
        pytest.param(4, 4, 8, 7, (10, 10), (4, 4, 8, 8), id="free space single max row increase"),
        pytest.param(4, 2, 8, 8, (10, 10), (3, 2, 9, 8), id="free space double min col decrease"),
        pytest.param(2, 4, 8, 8, (10, 10), (2, 3, 8, 9), id="free space double min row decrease"),
        pytest.param(4, 4, 8, 6, (10, 10), (4, 3, 8, 7), id="free space double max col increase"),
        pytest.param(4, 4, 6, 8, (10, 10), (3, 4, 7, 8), id="free space double max row increase"),
        pytest.param(1, 1, 6, 2, (10, 10), (1, 1, 6, 6), id="constrained left"),
        pytest.param(1, 6, 7, 8, (10, 10), (1, 2, 7, 8), id="constrained right"),
        pytest.param(1, 1, 2, 6, (10, 10), (1, 1, 6, 6), id="constrained top"),
        pytest.param(6, 1, 8, 7, (10, 10), (2, 1, 8, 7), id="constrained bottom"),
    ],
)
def test_make_crop_square(
    crop_min_row: int,
    crop_min_col: int,
    crop_max_row: int,
    crop_max_col: int,
    image_shape: tuple[int, int],
    expected_indices: tuple[int, int, int, int],
):
    """Test the crop square method."""
    result = make_crop_square(crop_min_row, crop_min_col, crop_max_row, crop_max_col, image_shape)
    assert result == expected_indices


@pytest.mark.parametrize(
    ("crop_min_row", "crop_min_col", "crop_max_row", "crop_max_col", "image_shape", "padding", "expected_indices"),
    [
        pytest.param(2, 2, 6, 6, (10, 10), 0, (2, 2, 6, 6), id="no padding"),
        pytest.param(0, 0, 10, 10, (10, 10), 5, (0, 0, 10, 10), id="crop too big to be padded"),
        pytest.param(2, 2, 6, 6, (10, 10), 2, (0, 0, 8, 8), id="padding 2, no obstruction"),
        pytest.param(0, 4, 6, 8, (10, 10), 2, (0, 2, 8, 10), id="padding 2, obstruction top"),
        pytest.param(4, 0, 8, 6, (10, 10), 2, (2, 0, 10, 8), id="padding 2, obstruction left"),
        pytest.param(4, 4, 10, 6, (10, 10), 2, (2, 2, 10, 8), id="padding 2, obstruction bottom"),
        pytest.param(4, 4, 6, 10, (10, 10), 2, (2, 2, 8, 10), id="padding 2, obstruction right"),
    ],
)
def test_pad_bounding_box(
    crop_min_row: int,
    crop_min_col: int,
    crop_max_row: int,
    crop_max_col: int,
    image_shape: tuple[int, int],
    padding,
    expected_indices,
):
    """Test the pad bounding box method."""
    result = pad_bounding_box(crop_min_row, crop_min_col, crop_max_row, crop_max_col, image_shape, padding)
    assert result == expected_indices
