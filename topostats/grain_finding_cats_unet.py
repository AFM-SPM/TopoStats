"""For segmenting cats grains using a specifically trained unet"""


from pathlib import Path
import logging

import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

from topostats.logs.logs import LOGGER_NAME

LOGGER = logging.getLogger(LOGGER_NAME)


def test_GPU():
    """Ensure that the GPU is working correctly"""
    # for key, value in os.environ.items():
    #     LOGGER.info(f"{key} : {value}")
    LOGGER.info("============= GPU TEST =============")
    tf.test.gpu_device_name()
    LOGGER.info("============= GPU TEST DONE =============")


def normalise_image(image: np.ndarray) -> np.ndarray:
    """Normalise image"""
    image = image - np.min(image)
    image = image / np.max(image)
    return image


def detect_ridges(gray, sigma=1.0):
    """Detect ridges in an image"""
    H_elems = hessian_matrix(gray, sigma=sigma, order="rc")
    maxima_ridges, minima_ridges = hessian_matrix_eigvals(H_elems)
    return maxima_ridges, minima_ridges


# DICE Loss
def dice_loss(y_true, y_pred, smooth=1e-5):
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2))
    sum_of_squares_pred = tf.reduce_sum(tf.square(y_pred), axis=(1, 2))
    sum_of_squares_true = tf.reduce_sum(tf.square(y_true), axis=(1, 2))
    dice = 1 - (2 * intersection + smooth) / (sum_of_squares_pred + sum_of_squares_true + smooth)
    return dice


# IoU Loss
def iou_loss(y_true, y_pred, smooth=1e-5):
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2))
    sum_of_squares_pred = tf.reduce_sum(tf.square(y_pred), axis=(1, 2))
    sum_of_squares_true = tf.reduce_sum(tf.square(y_true), axis=(1, 2))
    iou = (intersection + smooth) / (sum_of_squares_pred + sum_of_squares_true - intersection + smooth)
    return iou


def load_model(model_path: Path, custom_objects=None):
    """Load a keras unet model"""
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects)


def predict_unet(
    image: np.ndarray,
    confidence: float,
    model_image_size: int,
    image_output_dir: Path,
    filename: str,
) -> np.ndarray:
    """Predict cats segmentation from a flattened image."""

    # Make a copy of the original image
    original_image = image.copy()

    # Get the path to a file in the topostats package
    #model_path = Path(__file__).parent / "./catsnet_cropped_2023-11-10_17-45-07.h5"
    model_path = Path(__file__).parent / "./catsnet_2024-04-12_12-17-55.h5"

    LOGGER.info("Loading Unet model")
    model = load_model(model_path=model_path, custom_objects={"dice_loss": dice_loss, "iou_loss": iou_loss})
    LOGGER.info("Loaded Unet model")

    # Run the model on a single image
    LOGGER.info("Preprocessing image for Unet prediction...")

    # Normalise the image
    low_bound, up_bound = (-1, 7)
    LOGGER.info("normalising image")
    image = np.clip(image, low_bound, up_bound)
    image = image - low_bound
    image = image / (up_bound - low_bound)

    # Resize the image to 512x512
    image = Image.fromarray(image)
    image = image.resize((512, 512))
    image = np.array(image)

    # Predict the mask
    LOGGER.info("Running Unet & predicting mask")
    prediction = model.predict(np.expand_dims(image, axis=0))
    LOGGER.info("Unet finished, predicted mask. saving...")

    # Threshold the predicted mask
    predicted_mask = prediction > confidence
    # Remove the batch dimension
    predicted_mask = predicted_mask[0, :, :, 0]

    # Resize the predicted mask back to the original image size
    predicted_mask = Image.fromarray(predicted_mask)
    predicted_mask = predicted_mask.resize(original_image.shape)
    predicted_mask = np.array(predicted_mask)

    # Save the predicted mask
    fig, ax = plt.subplots(1, 3, figsize=(20, 7))
    ax[1].imshow(original_image)
    ax[1].set_title("normalised image")
    ax[0].imshow(original_image)
    ax[0].set_title("original image")
    ax[2].imshow(predicted_mask)
    ax[2].set_title("predicted mask")
    fig.suptitle(f"strictness: {confidence}")
    fig.tight_layout()
    #plt.savefig(image_output_dir / f"{filename}_prediction.png")

    return predicted_mask


if __name__ == "__main__":
    test_GPU()
