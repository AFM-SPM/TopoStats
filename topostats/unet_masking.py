"""Segment grains using a U-Net model."""

import logging

import keras
import numpy as np
import numpy.typing as npt
import tensorflow as tf
from PIL import Image

from topostats.logs.logs import LOGGER_NAME

LOGGER = logging.getLogger(LOGGER_NAME)


# DICE Loss
def dice_loss(y_true: npt.NDArray[np.float32], y_pred: npt.NDArray[np.float32], smooth: float = 1e-5) -> tf.Tensor:
    """
    DICE loss function.

    Expects y_true and y_pred to be of shape (batch_size, height, width, 1).

    Parameters
    ----------
    y_true : npt.NDArray[np.float32]
        True values.
    y_pred : npt.NDArray[np.float32]
        Predicted values.
    smooth : float
        Smoothing factor to prevent division by zero.

    Returns
    -------
    tf.Tensor
        The DICE loss.
    """
    # Ensure the tensors are of the same shape
    y_true = tf.squeeze(y_true, axis=-1) if y_true.shape[-1] == 1 else y_true
    y_pred = tf.squeeze(y_pred, axis=-1) if y_pred.shape[-1] == 1 else y_pred
    # Ensure floats not bool
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2))
    sum_of_squares_pred = tf.reduce_sum(tf.square(y_pred), axis=(1, 2))
    sum_of_squares_true = tf.reduce_sum(tf.square(y_true), axis=(1, 2))
    return 1 - (2 * intersection + smooth) / (sum_of_squares_pred + sum_of_squares_true + smooth)


# IoU Loss
def iou_loss(y_true: npt.NDArray[np.float32], y_pred: npt.NDArray[np.float32], smooth: float = 1e-5) -> tf.Tensor:
    """
    Intersection over Union loss function.

    Expects y_true and y_pred to be of shape (batch_size, height, width, 1).

    Parameters
    ----------
    y_true : npt.NDArray[np.float32]
        True values.
    y_pred : npt.NDArray[np.float32]
        Predicted values.
    smooth : float
        Smoothing factor to prevent division by zero.

    Returns
    -------
    tf.Tensor
        The IoU loss.
    """
    # Ensure the tensors are of the same shape
    y_true = tf.squeeze(y_true, axis=-1) if y_true.shape[-1] == 1 else y_true
    y_pred = tf.squeeze(y_pred, axis=-1) if y_pred.shape[-1] == 1 else y_pred
    # Ensure floats not bool
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2))
    sum_of_squares_pred = tf.reduce_sum(tf.square(y_pred), axis=(1, 2))
    sum_of_squares_true = tf.reduce_sum(tf.square(y_true), axis=(1, 2))
    return 1 - (intersection + smooth) / (sum_of_squares_pred + sum_of_squares_true - intersection + smooth)


def predict_unet(
    image: npt.NDArray[np.float32],
    model: keras.Model,
    confidence: float,
    model_input_shape: tuple[int | None, int, int, int],
    upper_norm_bound: float,
    lower_norm_bound: float,
) -> npt.NDArray[np.bool_]:
    """
    Predict cats segmentation from a flattened image.

    Parameters
    ----------
    image : npt.NDArray[np.float32]
        The image to predict the mask for.
    model : keras.Model
        The U-Net model.
    confidence : float
        The confidence threshold for the mask.
    model_input_shape : tuple[int | None, int, int, int]
        The shape of the model input, including the batch and channel dimensions.
    upper_norm_bound : float
        The upper bound for normalising the image.
    lower_norm_bound : float
        The lower bound for normalising the image.

    Returns
    -------
    npt.NDArray[np.bool_]
        The predicted mask.
    """
    # Strip the batch dimension from the model input shape
    image_shape: tuple[int, int] = model_input_shape[1:3]
    LOGGER.info(f"Model input shape: {model_input_shape}")

    # Make a copy of the original image
    original_image = image.copy()

    # Run the model on a single image
    LOGGER.info("Preprocessing image for Unet prediction...")

    # Normalise the image
    image = np.clip(image, lower_norm_bound, upper_norm_bound)
    image = image - lower_norm_bound
    image = image / (upper_norm_bound - lower_norm_bound)

    # Resize the image to the model input shape
    image_resized = Image.fromarray(image)
    image_resized = image_resized.resize(image_shape)
    image_resized_np: npt.NDArray[np.float32] = np.array(image_resized)

    # Predict the mask
    LOGGER.info("Running Unet & predicting mask")
    prediction: npt.NDArray[np.float32] = model.predict(np.expand_dims(image_resized_np, axis=(0, 3)))
    LOGGER.info("Unet finished predicted mask.")

    # Threshold the predicted mask
    predicted_mask: npt.NDArray[np.bool_] = prediction > confidence
    # Remove the batch and channel dimensions
    predicted_mask = predicted_mask[0, :, :, 0]

    # Resize the predicted mask back to the original image size
    predicted_mask_PIL = Image.fromarray(predicted_mask)
    predicted_mask_PIL = predicted_mask_PIL.resize((original_image.shape[0], original_image.shape[1]))

    return np.array(predicted_mask_PIL)


def make_bounding_box_square(
    crop_min_row: int, crop_min_col: int, crop_max_row: int, crop_max_col: int, image_shape: tuple[int, int]
) -> tuple[int, int, int, int]:
    """
    Make a bounding box square.

    Parameters
    ----------
    crop_min_row : int
        The minimum row index of the crop.
    crop_min_col : int
        The minimum column index of the crop.
    crop_max_row : int
        The maximum row index of the crop.
    crop_max_col : int
        The maximum column index of the crop.
    image_shape : tuple[int, int]
        The shape of the image.

    Returns
    -------
    tuple[int, int, int, int]
        The new crop indices.
    """
    crop_height = crop_max_row - crop_min_row
    crop_width = crop_max_col - crop_min_col

    diff: int
    new_crop_min_row: int
    new_crop_min_col: int
    new_crop_max_row: int
    new_crop_max_col: int

    if crop_height > crop_width:
        # The crop is taller than it is wide
        diff = crop_height - crop_width
        # Check if we can expand equally in each direction
        if crop_min_col - diff // 2 >= 0 and crop_max_col + diff - diff // 2 < image_shape[1]:
            new_crop_min_col = crop_min_col - diff // 2
            new_crop_max_col = crop_max_col + diff - diff // 2
        # If we can't expand uniformly, expand in just one direction
        # Check if we can expand right
        elif crop_max_col + diff - diff // 2 < image_shape[1]:
            # We can expand right
            new_crop_min_col = crop_min_col
            new_crop_max_col = crop_max_col + diff
        elif crop_min_col - diff // 2 >= 0:
            # We can expand left
            new_crop_min_col = crop_min_col - diff
            new_crop_max_col = crop_max_col
        # Set the new crop height to the original crop height since we are just updating the width
        new_crop_min_row = crop_min_row
        new_crop_max_row = crop_max_row
    elif crop_width > crop_height:
        # The crop is wider than it is tall
        diff = crop_width - crop_height
        # Check if we can expand equally in each direction
        if crop_min_row - diff // 2 >= 0 and crop_max_row + diff - diff // 2 < image_shape[0]:
            new_crop_min_row = crop_min_row - diff // 2
            new_crop_max_row = crop_max_row + diff - diff // 2
        # If we can't expand uniformly, expand in just one direction
        # Check if we can expand down
        elif crop_max_row + diff - diff // 2 < image_shape[0]:
            # We can expand down
            new_crop_min_row = crop_min_row
            new_crop_max_row = crop_max_row + diff
        elif crop_min_row - diff // 2 >= 0:
            # We can expand up
            new_crop_min_row = crop_min_row - diff
            new_crop_max_row = crop_max_row
        # Set the new crop width to the original crop width since we are just updating the height
        new_crop_min_col = crop_min_col
        new_crop_max_col = crop_max_col
    else:
        # If the crop is already square, return the original crop
        new_crop_min_row = crop_min_row
        new_crop_min_col = crop_min_col
        new_crop_max_row = crop_max_row
        new_crop_max_col = crop_max_col

    return new_crop_min_row, new_crop_min_col, new_crop_max_row, new_crop_max_col


def pad_bounding_box(
    crop_min_row: int,
    crop_min_col: int,
    crop_max_row: int,
    crop_max_col: int,
    image_shape: tuple[int, int],
    padding: int,
) -> tuple[int, int, int, int]:
    """
    Pad a bounding box.

    Parameters
    ----------
    crop_min_row : int
        The minimum row index of the crop.
    crop_min_col : int
        The minimum column index of the crop.
    crop_max_row : int
        The maximum row index of the crop.
    crop_max_col : int
        The maximum column index of the crop.
    image_shape : tuple[int, int]
        The shape of the image.
    padding : int
        The padding to apply to the bounding box.

    Returns
    -------
    tuple[int, int, int, int]
        The new crop indices.
    """
    new_crop_min_row: int = max(0, crop_min_row - padding)
    new_crop_min_col: int = max(0, crop_min_col - padding)
    new_crop_max_row: int = min(image_shape[0], crop_max_row + padding)
    new_crop_max_col: int = min(image_shape[1], crop_max_col + padding)

    return new_crop_min_row, new_crop_min_col, new_crop_max_row, new_crop_max_col