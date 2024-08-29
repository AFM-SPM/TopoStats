"""Segment grains using a U-Net model."""

from __future__ import annotations

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


def mean_iou(y_true: npt.NDArray[np.float32], y_pred: npt.NDArray[np.float32]):
    """
    Mean Intersection Over Union metric, ignoring the background class.

    Parameters
    ----------
    y_true : npt.NDArray[np.float32]
        True values.
    y_pred : npt.NDArray[np.float32]
        Predicted values.

    Returns
    -------
    tf.Tensor
        The mean IoU.
    """
    # Ensure the tensors are of the same shape, and ignore the background class
    # The [-1] here is to flatten the tensor into a 1D array, allowing for the calculation of the IoU
    # The 1: is to use all channels except channel 0. Since this would be the background class and if
    # we include it, the IoU would be very low since it is going to be mostly zero and so highly accurate?
    y_true_f = tf.reshape(y_true[:, :, :, 1:], [-1])  # ignore background class
    y_pred_f = tf.round(tf.reshape(y_pred[:, :, :, 1:], [-1]))  # ignore background class

    # Calculate the IoU, using all channels except the background class
    intersect = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersect
    smooth = tf.ones(tf.shape(intersect))  # Smoothing factor to prevent division by zero
    return tf.reduce_mean((intersect + smooth) / (union - intersect + smooth))


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
    LOGGER.info(f"Unet finished predicted mask. Prediction shape: {prediction.shape}")

    # Threshold the predicted mask
    predicted_mask: npt.NDArray[np.bool_] = prediction > confidence

    # Remove the batch dimension since we are predicting single images at a time
    predicted_mask = predicted_mask[0]

    # Note that this predicted mask can have any number of channels, depending on the number of classes for the model

    # Check if the output is a single channel mask and convert it to a two-channel mask since the output is
    # designed to be categorical, where even the background has a channel
    if predicted_mask.shape[2] == 1:
        predicted_mask = np.concatenate((1 - predicted_mask, predicted_mask), axis=2)

    assert len(predicted_mask.shape) == 3, f"Predicted mask shape is not 3D: {predicted_mask.shape}"
    assert (
        predicted_mask.shape[2] >= 2
    ), f"Predicted mask has less than 2 channels: {predicted_mask.shape[2]}, needs separate background channel"

    # Resize each channel of the predicted mask to the original image size, also remove the batch
    # dimension but keep channel
    resized_predicted_mask: npt.NDArray[np.bool_] = np.zeros(
        (original_image.shape[0], original_image.shape[1], predicted_mask.shape[2])
    ).astype(bool)
    for channel_index in range(predicted_mask.shape[2]):
        # Note that uint8 is required to allow PIL to load the array into an image
        channel_mask = predicted_mask[:, :, channel_index].astype(np.uint8)
        channel_mask_PIL = Image.fromarray(channel_mask)
        # Resize the channel mask to the original image size, but we want boolean so use nearest neighbour
        channel_mask_PIL = channel_mask_PIL.resize((original_image.shape[0], original_image.shape[1]), Image.NEAREST)
        resized_predicted_mask[:, :, channel_index] = np.array(channel_mask_PIL).astype(bool)

    return resized_predicted_mask


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
