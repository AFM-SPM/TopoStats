"""For segmenting cats grains using a specifically trained unet"""


from pathlib import Path

import os
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from skimage.feature import hessian_matrix, hessian_matrix_eigvals


def test_GPU():
    """Ensure that the GPU is working correctly"""
    # for key, value in os.environ.items():
    #     print(f"{key} : {value}")
    tf.test.gpu_device_name()
    print("GPU TEST OVER")


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


def load_model(model_path: Path):
    """Load a keras unet model"""
    return tf.keras.models.load_model(model_path)


def predict_unet(
    image: np.ndarray,
    confidence: float,
    model_image_size: int,
    image_output_dir: Path,
    filename: str,
) -> np.ndarray:
    """Predict cats segmentation from a flattened image."""

    original_image = image.copy()


    # Scale the image to ensure there are no messy negatives etc
    image = image - np.min(image)
    image = image / (np.max(image) / 255)

    # invert the image since it was trained with binary colour mapping which inverts it
    # when loaded via cv2.
    image = 255 - image

    # Detect ridges on the image
    maxima, minima = detect_ridges(image.copy(), sigma=4.0)

    # Plot maxima and minima
    # fig, ax = plt.subplots(1, 2)
    # im1 = ax[0].imshow(maxima, cmap="binary")
    # ax[0].set_title("maxima")
    # im2 = ax[1].imshow(minima, cmap="binary")
    # ax[1].set_title("minima")
    # fig.colorbar(im1)
    # fig.colorbar(im2)
    # fig.savefig("./maxima_minima.png")
    
    # Use maxima
    ridges = maxima

    to_predict = ridges.copy()

    # Get the image into the correct format for prediction
    to_predict = normalise_image(to_predict)
    print(f"normalised image shape: {ridges.shape}")
    # Resize image - this is really janky
    to_predict = Image.fromarray(to_predict)
    to_predict = to_predict.resize((model_image_size, model_image_size))
    to_predict = np.array(to_predict)
    to_predict = [to_predict]
    to_predict = np.array(to_predict)
    to_predict = np.expand_dims(to_predict, 3)
    to_predict = to_predict[0]
    to_predict = to_predict[:, :, 0][:, :, None]
    to_predict = np.expand_dims(to_predict, 0)

    # Predict segmentation
    # model = load_model(model_path="./models/20230817_10-26-23_cats_512_b4_e50_hessian_upper_4_0.hdf5")
    model = load_model(
        model_path="/Users/sylvi/Documents/TopoStats/catsnet/catsnet/saved_models/20230817_10-26-23_cats_512_b4_e50_hessian_upper_4_0.hdf5"
    )
    prediction = (model.predict(to_predict)[0, :, :, 0] > confidence).astype(np.uint8)

    fig, ax = plt.subplots(1, 3, figsize=(20, 7))
    ax[1].imshow(ridges)
    ax[1].set_title("ridges")
    ax[0].imshow(original_image)
    ax[0].set_title("original image")
    ax[2].imshow(prediction)
    ax[2].set_title("prediction")
    fig.suptitle(f"strictness: {confidence}")
    fig.tight_layout()
    plt.savefig(image_output_dir / f"{filename}_prediction.png")

    # Resize prediction back up to original size
    prediction = Image.fromarray(prediction)
    prediction = prediction.resize(original_image.shape)
    prediction = np.array(prediction)

    return prediction


if __name__ == "__main__":
    test_GPU()
