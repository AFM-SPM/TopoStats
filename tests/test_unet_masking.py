"""Test unet masking methods."""

import pytest

from topostats.unet_masking import make_crop_square


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
def test_make_crop_square(crop_min_row, crop_min_col, crop_max_row, crop_max_col, image_shape, expected_indices):
    """Test the crop square method."""
    result = make_crop_square(crop_min_row, crop_min_col, crop_max_row, crop_max_col, image_shape)
    assert result == expected_indices
