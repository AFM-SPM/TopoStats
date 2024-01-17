from grain_finding_haribo_unet import predict_unet_multiclass_and_get_angle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from topostats.plottingfuncs import Colormap

colormap = Colormap()
cmap = colormap.get_cmap()


def process_grain(
    cropped_image: np.ndarray,
    model,
    grain_number: int = 0,
    IMAGE_SAVE_DIR: Path = Path("./"),
    filename: str = "test",
):
    """Take a cropped protein-bound-molecule image and return the angle between the vectors and the predicted masks for the gem and ring."""

    model_image_size = 256
    model_to_original_image_size_factor = cropped_image.shape[0] / model_image_size

    # Run the UNet on the region
    try:
        (
            combined_predicted_mask,
            angle,
            plotting_info,
        ) = predict_unet_multiclass_and_get_angle(
            image=cropped_image,
            model=model,
            confidence=0.5,
            model_image_size=model_image_size,
            image_output_dir=Path("./"),
            filename=filename + f"_grain_{grain_number}",
            IMAGE_SAVE_DIR=IMAGE_SAVE_DIR,
            image_index=grain_number,
        )
    except ValueError as e:
        # Check if "found array witn 0 sample(s)" in error message
        if "Found array with 0 sample(s)" in str(e):
            # If so, skip this grain
            print(f"Angle calculation failed: k means 0 samples. Skipping grain {grain_number}")
            raise ValueError(f"Angle calculation failed: k means 0 samples. Skipping grain {grain_number}") from e
        elif "KMEANS" in str(e):
            print(f"Angle calculation failed: k means (too few touching coordinates). Skipping grain {grain_number}")
            raise ValueError(
                f"Angle calculation failed: k means (too few touching coordinates). Skipping grain {grain_number}"
            ) from e
        else:
            raise e

    # Ignore grains where the gem and ring masks are too small
    gem_min_size = 100
    ring_min_size = 200
    num_gem_pixels = np.sum(combined_predicted_mask == 2)
    num_ring_pixels = np.sum(combined_predicted_mask == 1)
    if num_gem_pixels < gem_min_size:
        print(f"Gem mask too small: {num_gem_pixels}. Skipping grain {grain_number}")
        raise ValueError(f"Gem mask too small: {num_gem_pixels}. Skipping grain {grain_number}")
    if num_ring_pixels < ring_min_size:
        print(f"Ring mask too small: {num_ring_pixels}. Skipping grain {grain_number}")
        raise ValueError(f"Ring mask too small: {num_ring_pixels}. Skipping grain {grain_number}")

    # Plot the angle visualisation
    path = plotting_info["path"]
    vector_visualisation_start_x = plotting_info["vector_visualisation_start_x"]
    vector_visualisation_start_y = plotting_info["vector_visualisation_start_y"]
    vector_visualisation_end_x = plotting_info["vector_visualisation_end_x"]
    vector_visualisation_end_y = plotting_info["vector_visualisation_end_y"]

    # Plot the vectors
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    axs[1, 1].imshow(cropped_image, cmap=cmap, vmin=-4, vmax=8)
    axs[1, 1].scatter(
        vector_visualisation_start_x[0] * model_to_original_image_size_factor,
        vector_visualisation_start_y[0] * model_to_original_image_size_factor,
        c="red",
    )
    axs[1, 1].scatter(
        vector_visualisation_end_x[1] * model_to_original_image_size_factor,
        vector_visualisation_end_y[1] * model_to_original_image_size_factor,
        c="blue",
    )
    axs[1, 1].plot(
        path[1] * model_to_original_image_size_factor,
        path[0] * model_to_original_image_size_factor,
        linewidth=4,
        c="pink",
    )
    # Plot the average vectors
    axs[1, 1].plot(
        [
            vector_visualisation_start_x[0] * model_to_original_image_size_factor,
            vector_visualisation_start_x[1] * model_to_original_image_size_factor,
        ],
        [
            vector_visualisation_start_y[0] * model_to_original_image_size_factor,
            vector_visualisation_start_y[1] * model_to_original_image_size_factor,
        ],
        linewidth=5,
        c="red",
    )
    axs[1, 1].plot(
        [
            vector_visualisation_end_x[0] * model_to_original_image_size_factor,
            vector_visualisation_end_x[1] * model_to_original_image_size_factor,
        ],
        [
            vector_visualisation_end_y[0] * model_to_original_image_size_factor,
            vector_visualisation_end_y[1] * model_to_original_image_size_factor,
        ],
        linewidth=5,
        c="blue",
    )
    # Round the angle to 2 decimal places
    angle_between_vectors_degrees = np.round(angle, 2)
    axs[1, 1].set_title("Path and Vectors, $\\alpha$ = " + str(angle_between_vectors_degrees))
    # Set title fond size
    axs[1, 1].title.set_fontsize(50)

    # Remove ticks
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])

    # plt.savefig(IMAGE_SAVE_DIR / f"path_and_vectors_{filename}_{grain_number}.png")

    print(f"Combined predicted mask shape: {combined_predicted_mask.shape}")

    # Plot region image and predicted mask
    axs[0, 0].imshow(cropped_image)
    axs[0, 0].set_title("region image")
    axs[0, 1].imshow(combined_predicted_mask)
    axs[0, 1].set_title("predicted mask")
    fig.tight_layout()
    plt.savefig(IMAGE_SAVE_DIR / f"{filename}_grain_{grain_number}_predicted_mask_and_vectors.png")

    predicted_ring_mask = combined_predicted_mask == 1

    return angle, predicted_ring_mask
