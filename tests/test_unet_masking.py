"""Test unet masking methods."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pytest

from topostats.unet_masking import dice_loss, iou_loss, make_bounding_box_square, pad_bounding_box, predict_unet


@pytest.mark.parametrize(
    ("y_true", "y_pred", "smooth", "expected_loss"),
    [
        pytest.param(
            np.array([[0, 0], [0, 0]]), np.array([[0, 0], [0, 0]]).astype(np.float32), 1e-5, 0.0, id="perfect match"
        ),
        pytest.param(
            np.array([[1, 1], [1, 1]]), np.array([[0, 0], [0, 0]]).astype(np.float32), 1e-5, 1.0, id="complete mismatch"
        ),
        pytest.param(
            np.array([[1, 0], [0, 0]]).astype(np.float32),
            np.array([[0.5, 0], [0, 0]]).astype(np.float32),
            1e-5,
            0.2,
            id="partial match, perfect location, weak intensity",
        ),
        pytest.param(
            np.array([[1, 0], [0, 0]]).astype(np.float32),
            np.array([[1, 0.5], [0, 0]]).astype(float),
            1e-5,
            0.111111,
            id="partial match, perfect intensity, weak location (spillage)",
        ),
        pytest.param(
            np.array([[1, 0], [0, 0]]).astype(np.float32),
            np.array([[0.5, 1], [0, 0]]).astype(np.float32),
            1e-5,
            0.555555,
            id="mostly mismatched, poor location, poor intensity",
        ),
    ],
)
def test_dice_loss(
    y_true: npt.NDArray[np.float32], y_pred: npt.NDArray[np.float32], smooth: float, expected_loss: float
):
    """Test the dice_loss function."""
    y_true = np.expand_dims(y_true, axis=0)
    y_pred = np.expand_dims(y_pred, axis=0)
    result = dice_loss(y_true, y_pred, smooth)
    np.testing.assert_allclose(result, expected_loss, atol=1e-5)


@pytest.mark.parametrize(
    ("y_true", "y_pred", "smooth", "expected_loss"),
    [
        pytest.param(np.array([[0, 0], [0, 0]]), np.array([[0, 0], [0, 0]]), 1e-5, 0.0, id="perfect match"),
        pytest.param(np.array([[1, 1], [1, 1]]), np.array([[0, 0], [0, 0]]), 1e-5, 1.0, id="complete mismatch"),
        pytest.param(
            np.array([[1, 0], [0, 0]]),
            np.array([[0.5, 0], [0, 0]]),
            1e-5,
            0.333333,
            id="partial match, perfect location, weak intensity",
        ),
        pytest.param(
            np.array([[1, 0], [0, 0]]),
            np.array([[1, 0.5], [0, 0]]),
            1e-5,
            0.2,
            id="partial match, perfect intensity, weak location (spillage)",
        ),
        pytest.param(
            np.array([[1, 0], [0, 0]]),
            np.array([[0.5, 1], [0, 0]]),
            1e-5,
            0.714282,
            id="mostly mismatched, poor location, poor intensity",
        ),
    ],
)
def test_iou_loss(
    y_true: npt.NDArray[np.float32], y_pred: npt.NDArray[np.float32], smooth: float, expected_loss: float
):
    """Test the iou_loss function."""
    y_true = np.expand_dims(y_true, axis=0)
    y_pred = np.expand_dims(y_pred, axis=0)
    result = iou_loss(y_true, y_pred, smooth)
    np.testing.assert_allclose(result, expected_loss, atol=1e-5)


def test_predict_unet(mock_model_5_by_5_single_class: MagicMock) -> None:
    """Test the predict_unet method."""
    image = np.array(
        [
            [0.1, 0.2, 0.1, 0.2, 0.1],
            [0.1, 1.0, 1.0, 1.0, 0.1],
            [0.2, 1.0, 1.0, 1.0, 0.2],
            [0.1, 1.0, 1.0, 1.0, 0.1],
            [0.1, 0.1, 0.2, 0.2, 0.1],
        ]
    )
    confidence = 0.5
    model_input_shape = (None, 5, 5, 1)
    upper_norm_bound = 1.0
    lower_norm_bound = 0.0

    predicted_mask = predict_unet(
        image=image,
        model=mock_model_5_by_5_single_class,
        confidence=confidence,
        model_input_shape=model_input_shape,
        upper_norm_bound=upper_norm_bound,
        lower_norm_bound=lower_norm_bound,
    )

    assert predicted_mask.shape == (5, 5, 1)
    assert isinstance(predicted_mask, np.ndarray)
    assert predicted_mask.dtype == np.bool_
    np.testing.assert_array_equal(
        predicted_mask,
        np.array(
            [
                [
                    [0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 1, 0, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0],
                ]
            ]
        ).reshape((5, 5, 1)),
    )


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
def test_make_bounding_box_square(
    crop_min_row: int,
    crop_min_col: int,
    crop_max_row: int,
    crop_max_col: int,
    image_shape: tuple[int, int],
    expected_indices: tuple[int, int, int, int],
) -> None:
    """Test the make_bounding_box_square method."""
    result = make_bounding_box_square(crop_min_row, crop_min_col, crop_max_row, crop_max_col, image_shape)
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
) -> None:
    """Test the pad_bounding_box method."""
    result = pad_bounding_box(crop_min_row, crop_min_col, crop_max_row, crop_max_col, image_shape, padding)
    assert result == expected_indices
