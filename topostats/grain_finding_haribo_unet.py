"""For segmenting cats grains using a specifically trained unet"""


from pathlib import Path
from typing import Tuple
import logging

import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation
from skimage.color import label2rgb
from sklearn.cluster import KMeans
from scipy.ndimage import distance_transform_edt
from skimage.graph import route_through_array

from topostats.logs.logs import LOGGER_NAME
from topostats.plottingfuncs import Colormap

cmap = Colormap()
cmap = cmap.get_cmap()

LOGGER = logging.getLogger(LOGGER_NAME)


def test_GPU():
    """Ensure that the GPU is working correctly"""
    # for key, value in os.environ.items():
    #     LOGGER.info(f"{key} : {value}")
    LOGGER.info("============= GPU TEST =============")
    tf.test.gpu_device_name()
    LOGGER.info("============= GPU TEST DONE =============")


def mean_iou(y_true, y_pred):
    """Mean Intersection Over Union metric, ignoring the background class."""
    y_true_f = tf.reshape(y_true[:, :, :, 1:], [-1])  # ignore background class
    y_pred_f = tf.round(tf.reshape(y_pred[:, :, :, 1:], [-1]))  # ignore background class
    intersect = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersect
    smooth = tf.ones(tf.shape(intersect))
    return tf.reduce_mean((intersect + smooth) / (union - intersect + smooth))


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


def load_model(model_path: Path, custom_objects=None):
    """Load a keras unet model"""
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects)


def turn_small_gem_regions_into_ring(combined_predicted_mask: np.ndarray, image: np.ndarray):
    # Make copy
    combined_predicted_mask_copy = combined_predicted_mask.copy()
    gem_mask = combined_predicted_mask_copy == 2

    # Find the largest gem region
    gem_labels = label(gem_mask)
    gem_regions = regionprops(gem_labels)
    gem_areas = [region.area for region in gem_regions]

    if len(gem_areas) == 0:
        return combined_predicted_mask

    largest_gem_region = gem_regions[np.argmax(gem_areas)]

    # For all the other regions, check if they touch a ring region
    for region in gem_regions:
        if region.label == largest_gem_region.label:
            continue
        # Get only the pixels in the region
        region_mask = gem_labels == region.label
        # Dilate the region
        small_gem_dilation_strength = 5
        dilated_region_mask = region_mask
        for i in range(small_gem_dilation_strength):
            dilated_region_mask = binary_dilation(dilated_region_mask)
        # Get the intersection with the ring mask
        predicted_ring_mask = combined_predicted_mask == 1
        intersection = dilated_region_mask & predicted_ring_mask
        # If there is an intersection, then it is a ring
        if np.any(intersection):
            combined_predicted_mask[dilated_region_mask] = 1

    return combined_predicted_mask


def remove_small_ring_regions(combined_predicted_mask):
    # Keep only the largest ring region
    ring_mask = combined_predicted_mask == 1
    ring_mask_labelled = label(ring_mask)
    ring_props = regionprops(ring_mask_labelled)
    if len(ring_props) == 0:
        return combined_predicted_mask
    ring_areas = [region.area for region in ring_props]
    largest_ring_index = np.argmax(ring_areas)
    largest_ring_region = ring_props[largest_ring_index]
    largest_ring_mask = ring_mask_labelled == largest_ring_region.label

    combined_predicted_mask[combined_predicted_mask == 1] = 0
    combined_predicted_mask[largest_ring_mask] = 1

    return combined_predicted_mask


def remove_small_background_regions(combined_predicted_mask):
    # Remove small regions of background
    background_mask = combined_predicted_mask == 0

    # fig, ax = plt.subplots(figsize=(12, 12))
    # ax.imshow(background_mask)
    # ax.set_title("Background Mask")
    # plt.show()

    # Turn any regions that are smaller than a certain size into 1s
    background_mask_components = label(background_mask)
    background_mask_components_coloured = label2rgb(background_mask_components)
    # fig, ax = plt.subplots(figsize=(12, 12))
    # ax.imshow(background_mask_components_coloured)
    # ax.set_title("Background Mask Components")
    # plt.show()

    # Make a copy of the combined mask
    combined_mask_ring_expanded = np.copy(combined_predicted_mask)

    # Find the two largest background regions
    indexes_to_keep = []
    background_mask_properties = regionprops(background_mask_components)
    background_areas = [region.area for region in background_mask_properties]
    background_areas_sorted = np.sort(background_areas)
    # Get indexes of the two largest background regions
    # and add them to the indexes to keep
    indexes_to_keep.append(background_areas.index(background_areas_sorted[-1]))
    if len(background_areas_sorted) >= 2:
        indexes_to_keep.append(background_areas.index(background_areas_sorted[-2]))

    background_mask_properties = regionprops(background_mask_components)
    for index, region in enumerate(background_mask_properties):
        # print(f"Region: {index}, Area: {region.area}")

        # Only remove regions that are smaller than 200 pixels
        # and are not the two largest background regions
        if region.area < 200 and index not in indexes_to_keep:
            # If the region is touching any pixels that are in the ring mask, then set that region to be in the ring mask
            # Get the outline of the region by dilating the region and then subtracting the original region
            region_mask = background_mask_components == region.label
            region_mask_dilated = binary_dilation(region_mask)
            region_outline = np.logical_and(region_mask_dilated, np.invert(region_mask))
            # Get the pixels in the outline from the combined mask
            region_outline_pixel_values = combined_predicted_mask[region_outline]
            # print(f'region outline pixel values: {region_outline_pixel_values[0:10]}')
            # If any of the pixels in the outline are in the ring mask (ie are 1s) then set the region to be in the ring mask
            # Check if there are any pixels with value 1 in the outline
            if np.any(region_outline_pixel_values == 1):
                # Set the region to be in the ring mask
                combined_mask_ring_expanded[background_mask_components == region.label] = 1
            else:
                # Set the region to be in the gem mask
                combined_mask_ring_expanded[background_mask_components == region.label] = 2

    return combined_mask_ring_expanded


def find_connecting_regions(combined_predicted_mask):
    # Take the gem mask and dilate it
    gem_mask = combined_predicted_mask == 2
    ring_mask = combined_predicted_mask == 1
    near_gem_ring_dilation_strength = 5
    gem_mask_dilated = np.copy(gem_mask)
    for i in range(near_gem_ring_dilation_strength):
        gem_mask_dilated = binary_dilation(gem_mask_dilated)

    # Find the pixels in the dilation that are also in the ring mask
    near_gem_and_ring = np.logical_and(gem_mask_dilated, ring_mask)

    return near_gem_and_ring


def find_centroids_of_connecting_regions(near_gem_and_ring):
    ## Find the midpoint of each of the two intersection regions

    # Cluster the points into two groups, one for each intersection region
    # Use the k-means algorithm to cluster the points into two groups

    # Get the coordinates of the points
    coordinates = np.argwhere(near_gem_and_ring)

    if len(coordinates) < 2:
        raise ValueError("KMEANS: Not enough points to find centroids")

    # Cluster the points
    kmeans = KMeans(n_clusters=2, random_state=0).fit(coordinates)
    # Get mask of each cluster
    cluster1_mask = kmeans.labels_ == 0
    cluster2_mask = kmeans.labels_ == 1
    # # Plot the clusters
    # plt.scatter(coordinates[cluster1_mask, 0], coordinates[cluster1_mask, 1], c="red")
    # plt.scatter(coordinates[cluster2_mask, 0], coordinates[cluster2_mask, 1], c="blue")
    # plt.title("K-Means Clustering")
    # plt.show()

    # For each cluster:
    centroid_1 = np.mean(coordinates[cluster1_mask, :], axis=0)
    centroid_2 = np.mean(coordinates[cluster2_mask, :], axis=0)

    # Find the closest point in the other cluster to each centroid
    # Get the coordinates of each cluster
    cluster1_coordinates = coordinates[cluster1_mask, :]
    # Get the distance between each point in cluster 1 and the centroid of cluster 1
    distances_1 = np.linalg.norm(cluster1_coordinates - centroid_1, axis=1)
    # Get the index of the closest point
    closest_point_index_1 = np.argmin(distances_1)
    # Get the coordinates of the closest point
    closest_point_1 = cluster1_coordinates[closest_point_index_1, :]
    centroid_1 = closest_point_1

    # Get the coordinates of each cluster
    cluster2_coordinates = coordinates[cluster2_mask, :]
    # Get the distance between each point in cluster 2 and the centroid of cluster 2
    distances_2 = np.linalg.norm(cluster2_coordinates - centroid_2, axis=1)
    # Get the index of the closest point
    closest_point_index_2 = np.argmin(distances_2)
    # Get the coordinates of the closest point
    closest_point_2 = cluster2_coordinates[closest_point_index_2, :]
    centroid_2 = closest_point_2

    # # Plot the centroids with the cluster
    # plt.imshow(near_gem_and_ring)
    # plt.scatter(centroid_1[1], centroid_1[0], c="red")
    # plt.scatter(centroid_2[1], centroid_2[0], c="blue")
    # plt.title("Centroid")
    # plt.show()

    return centroid_1, centroid_2


def create_pathdinging_weighting(
    image_512,
    combined_predicted_mask,
):
    ring_mask = combined_predicted_mask == 1

    # Get the pixels in the image that are in the mask
    ring_pixels = image_512[ring_mask]

    # Create an array of zeros the same size as the image
    ring_pixel_image = np.zeros_like(image_512)
    # Set the pixels in the mask to be the pixels from the image
    ring_pixel_image[ring_mask] = ring_pixels

    print(f"Ring Pixel Image Shape: {ring_pixel_image.shape}")
    print(f"Maximum Pixel Value: {np.max(ring_pixel_image)} Minimum Pixel Value: {np.min(ring_pixel_image)}")

    ring_pixel_image = ring_pixel_image - np.min(ring_pixel_image)

    # Increase the contrast of the image
    ring_pixel_image = ring_pixel_image**10

    # Normalise the image
    ring_pixel_image = ring_pixel_image - np.min(ring_pixel_image)
    ring_pixel_image = ring_pixel_image / np.max(ring_pixel_image)

    # Create a distance transform of the ring mask
    distance_transform = distance_transform_edt(ring_mask)

    # Normalise the distance transform
    distance_transform = distance_transform - np.min(distance_transform)
    distance_transform = distance_transform / np.max(distance_transform)

    # Combine the distance transform and the ring pixel image to get a final weighting
    distance_weighting_factor = 1.0

    final_weighting = (
        distance_weighting_factor * distance_transform + (1 - distance_weighting_factor) * ring_pixel_image
    )

    # Normalise
    final_weighting = final_weighting - np.min(final_weighting)
    final_weighting = final_weighting / np.max(final_weighting)

    # Invert the weighting
    final_weighting = 1 - final_weighting

    final_weighting_high_outside = np.copy(final_weighting)
    final_weighting_high_outside[combined_predicted_mask != 1] = 1000

    return final_weighting, final_weighting_high_outside


def find_path(image_512, centroid_1, centroid_2, final_weighting):
    # Use a weighted pathfinding algorithm to find the best path between the two centroids,
    # with the distance transform as the weight map

    # Get the path
    path, weight = route_through_array(final_weighting, centroid_1, centroid_2, fully_connected=True, geometric=True)

    path = np.stack(path, axis=-1)

    print(f"weight: {weight}")

    # # Plot the path
    # fig, axs = plt.subplots(1, 3, figsize=(30, 16))
    # axs[0].imshow(final_weighting)
    # axs[0].scatter(centroid_1[1], centroid_1[0], c="red")
    # axs[0].scatter(centroid_2[1], centroid_2[0], c="blue")
    # axs[0].plot(path[1], path[0], linewidth=2, c="pink")
    # axs[0].set_title("Pathfinding weighting")
    # axs[1].imshow(combined_predicted_mask)
    # axs[1].scatter(centroid_1[1], centroid_1[0], c="red")
    # axs[1].scatter(centroid_2[1], centroid_2[0], c="blue")
    # axs[1].plot(path[1], path[0], linewidth=2, c="pink")
    # axs[1].set_title("Combined Mask")
    # axs[2].imshow(image_512)
    # axs[2].plot(path[1], path[0], linewidth=2, c="pink")
    # axs[2].set_title("Original Image")
    # plt.show()

    # Create binary image of path
    path_image = np.zeros_like(image_512)
    path_image[path[0], path[1]] = 1

    return path_image, path


def calculate_vectors_and_angle(original_image, image_512, path, IMAGE_SAVE_DIR: Path, image_index=None):
    path_t = path.T

    # print(f"Path: \n{path_t[0:10]}")

    # Find the ends of the path
    path_start = path_t[0]
    # print(f"Path Start: {path_start}")
    path_end = path_t[-1]
    # print(f"Path End: {path_end}")

    # print(f"Centroid 1: {centroid_1}")
    # print(f"Centroid 2: {centroid_2}")

    # Get the vectors for each end to a nearby point on the path

    # Placeholder for the scaling factor since we don't know it for the training images
    pixel_to_nm_scaling = 0.5
    # Nm distance to take vectors until

    max_point_distance_nm = 20
    # Convert to pixels
    max_point_distance_pixels = int(max_point_distance_nm / pixel_to_nm_scaling)
    # print(f"Max Point Distance: {max_point_distance_pixels} pixels")

    # From the start point:
    # Calculate mean and std dev for vectors from start point until max_point_distance_pixels
    points_to_calculate_start_vectors = path_t[0:max_point_distance_pixels]
    # print(f"Points to calculate start vectors: {points_to_calculate_start_vectors}")
    start_vectors = points_to_calculate_start_vectors - path_start
    # print(f"Start Vectors: \n{start_vectors[0:10]}")
    # Calculate angles of vectors
    start_vectors_angles = np.arctan2(start_vectors[:, 0], start_vectors[:, 1])
    # print(f"Start Vectors Angles: \n{start_vectors_angles[0:10]}")
    # Calculate the mean and std dev of the angles
    start_vectors_angles_mean = np.mean(start_vectors_angles)
    start_vectors_angles_std = np.std(start_vectors_angles)
    # print(f"Start Vectors Angles Mean: {start_vectors_angles_mean}, Std: {start_vectors_angles_std}")
    # Calculate the mean vector
    start_vectors_mean = np.mean(start_vectors, axis=0)
    # print(f"Start Vectors Mean: {start_vectors_mean}")

    # From the end point:
    # Calculate mean and std dev for vectors from end point until max_point_distance_pixels
    points_to_calculate_end_vectors = path_t[-max_point_distance_pixels:]
    # print(f"Points to calculate end vectors: {points_to_calculate_end_vectors}")
    end_vectors = points_to_calculate_end_vectors - path_end
    # print(f"End Vectors: \n{end_vectors[0:10]}")
    # Calculate angles of vectors
    end_vectors_angles = np.arctan2(end_vectors[:, 0], end_vectors[:, 1])
    # print(f"End Vectors Angles: \n{end_vectors_angles[0:10]}")
    # Calculate the mean and std dev of the angles
    end_vectors_angles_mean = np.mean(end_vectors_angles)
    end_vectors_angles_std = np.std(end_vectors_angles)
    # print(f"End Vectors Angles Mean: {end_vectors_angles_mean}, Std: {end_vectors_angles_std}")
    # Calculate the mean vector
    end_vectors_mean = np.mean(end_vectors, axis=0)
    # print(f"End Vectors Mean: {end_vectors_mean}")

    # Get the angle between the two vectors in radians and degrees
    angle_between_vectors = np.arccos(
        np.dot(start_vectors_mean, end_vectors_mean)
        / np.linalg.norm(start_vectors_mean)
        / np.linalg.norm(end_vectors_mean)
    )
    angle_between_vectors_degrees = np.degrees(angle_between_vectors)
    # print(
    #     f"Angle between vectors: {angle_between_vectors} radians, {angle_between_vectors_degrees} degrees"
    # )

    # # Plot the vectors
    # fig, axs = plt.subplots(1, 1, figsize=(20, 20))
    # axs.imshow(original_image, cmap=cmap, vmin=-4, vmax=8)
    # axs.scatter(path_start[1], path_start[0], c="red")
    # axs.scatter(path_end[1], path_end[0], c="blue")
    # axs.plot(path[1], path[0], linewidth=4, c="pink")
    # # Plot the average vectors
    # axs.plot(
    #     [path_start[1], path_start[1] + start_vectors_mean[1]],
    #     [path_start[0], path_start[0] + start_vectors_mean[0]],
    #     linewidth=5,
    #     c="red",
    # )
    # axs.plot(
    #     [path_end[1], path_end[1] + end_vectors_mean[1]],
    #     [path_end[0], path_end[0] + end_vectors_mean[0]],
    #     linewidth=5,
    #     c="blue",
    # )
    # # Round the angle to 2 decimal places
    # angle_between_vectors_degrees = np.round(angle_between_vectors_degrees, 2)
    # axs.set_title("Path and Vectors, $\\alpha$ = " + str(angle_between_vectors_degrees))
    # # Set title fond size
    # axs.title.set_fontsize(50)

    # # Remove ticks
    # axs.set_xticks([])
    # axs.set_yticks([])

    # if image_index is not None:
    #     plt.savefig(IMAGE_SAVE_DIR / f"path_and_vectors_{image_index}.png")
    # else:
    #     plt.show()

    vector_visualisation_start_x = (path_start[1], path_start[1] + start_vectors_mean[1])
    vector_visualisation_start_y = (path_start[0], path_start[0] + start_vectors_mean[0])
    vector_visualisation_end_x = (path_end[1], path_end[1] + end_vectors_mean[1])
    vector_visualisation_end_y = (path_end[0], path_end[0] + end_vectors_mean[0])

    plotting_info = {
        "path": path,
        "vector_visualisation_start_x": vector_visualisation_start_x,
        "vector_visualisation_start_y": vector_visualisation_start_y,
        "vector_visualisation_end_x": vector_visualisation_end_x,
        "vector_visualisation_end_y": vector_visualisation_end_y,
    }

    return (
        angle_between_vectors_degrees,
        plotting_info,
    )


def find_angle(
    original_image: np.ndarray,
    image: np.ndarray,
    combined_predicted_mask: np.ndarray,
    IMAGE_SAVE_DIR: Path,
    image_index: int,
):
    # Turn small gem regions into ring regions
    combined_predicted_mask = turn_small_gem_regions_into_ring(combined_predicted_mask, image)

    # Remove small ring regions
    combined_predicted_mask = remove_small_ring_regions(combined_predicted_mask)

    # Remove small background regions
    combined_predicted_mask = remove_small_background_regions(combined_predicted_mask)

    # Find the regions that are near the ring
    near_gem_and_ring = find_connecting_regions(combined_predicted_mask)

    # Find the centroids of the two regions
    centroid_1, centroid_2 = find_centroids_of_connecting_regions(near_gem_and_ring)

    # Create a weighting for pathfinding
    final_weighting, final_weighting_high_outside = create_pathdinging_weighting(image, combined_predicted_mask)

    # Find the path
    path_image, path = find_path(image, centroid_1, centroid_2, final_weighting)

    # Calculate the angle
    angle, plotting_info = calculate_vectors_and_angle(
        original_image, image, path, IMAGE_SAVE_DIR, image_index=image_index
    )

    return angle, plotting_info


def predict_unet(
    image: np.ndarray,
    model: tf.keras.models.Model,
    confidence: float,
    model_image_size: int,
    image_output_dir: Path,
    filename: str,
) -> np.ndarray:
    """Predict cats segmentation from a flattened image."""

    # Make a copy of the original image
    original_image = image.copy()

    # Run the model on a single image
    LOGGER.info("Preprocessing image for Unet prediction...")

    # Normalise the image
    LOGGER.info("normalising image")
    image = image - np.min(image)
    image = image / np.max(image)

    # Resize the image
    image = Image.fromarray(image)
    image = image.resize((model_image_size, model_image_size))
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

    return predicted_mask


def predict_unet_multiclass_and_get_angle(
    image: np.ndarray,
    model: tf.keras.models.Model,
    confidence: float,
    model_image_size: int,
    image_output_dir: Path,
    filename: str,
    IMAGE_SAVE_DIR: Path,
    image_index: int,
) -> Tuple[np.ndarray, float]:
    """Predict cats segmentation from a flattened image."""

    # Make a copy of the original image
    original_image = image.copy()
    original_image_resized = original_image.copy()
    original_image_resized = Image.fromarray(original_image_resized)
    original_image_resized = original_image_resized.resize((model_image_size, model_image_size))
    original_image_resized = np.array(original_image_resized)

    # # Print summary
    # LOGGER.info(
    #     "===============================================================================\n\n"
    # )
    # model.summary()
    # LOGGER.info(
    #     "===============================================================================\n\n"
    # )

    # Run the model on a single image
    LOGGER.info("Preprocessing image for Unet prediction...")

    # Resize the image
    image = Image.fromarray(image)
    image = image.resize((model_image_size, model_image_size))
    image = np.array(image)

    # Normalise the image
    LOGGER.info("normalising image")
    # image = image - np.min(image)
    # image = image / np.max(image)
    LOWER_LIMIT = -1
    UPPER_LIMIT = 8
    image = np.clip(image, LOWER_LIMIT, UPPER_LIMIT)
    image = image - LOWER_LIMIT
    image = image / (UPPER_LIMIT - LOWER_LIMIT)

    # Predict the mask
    LOGGER.info("Running Unet & predicting mask")
    prediction = model.predict(np.expand_dims(image, axis=0))[0]
    LOGGER.info("Unet finished, predicted mask. saving...")

    # Remove the batch dimension
    # Gem
    predicted_gem_mask = prediction[:, :, 2] > confidence
    predicted_ring_mask = prediction[:, :, 1] > confidence
    predicted_background_mask = prediction[:, :, 0] > confidence

    combined_predicted_mask = np.zeros_like(predicted_gem_mask).astype(np.uint8)
    combined_predicted_mask[predicted_gem_mask] = 2
    combined_predicted_mask[predicted_ring_mask] = 1

    # Do all the processing for finding the angle
    angle, plotting_info = find_angle(
        original_image_resized,
        image,
        combined_predicted_mask,
        IMAGE_SAVE_DIR,
        image_index=image_index,
    )

    # Use the ring mask as the predicted mask
    # predicted_mask = combined_predicted_mask == 1

    # Resize the predicted mask back to the original image size
    # predicted_mask = Image.fromarray(predicted_mask)
    # predicted_mask = predicted_mask.resize(original_image.shape)
    # predicted_mask = np.array(predicted_mask)

    combined_predicted_mask = Image.fromarray(combined_predicted_mask)
    combined_predicted_mask = combined_predicted_mask.resize(original_image.shape)
    combined_predicted_mask = np.array(combined_predicted_mask)

    return combined_predicted_mask, angle, plotting_info


def predict_unet_multiclass(
    image: np.ndarray,
    model: tf.keras.models.Model,
    confidence: float,
    model_image_size: int,
    image_output_dir: Path,
    filename: str,
    IMAGE_SAVE_DIR: Path,
    image_index: int,
) -> Tuple[np.ndarray, float]:
    """Predict cats segmentation from a flattened image."""

    # Make a copy of the original image
    original_image = image.copy()
    original_image_resized = original_image.copy()
    original_image_resized = Image.fromarray(original_image_resized)
    original_image_resized = original_image_resized.resize((model_image_size, model_image_size))
    original_image_resized = np.array(original_image_resized)

    # # Print summary
    # LOGGER.info(
    #     "===============================================================================\n\n"
    # )
    # model.summary()
    # LOGGER.info(
    #     "===============================================================================\n\n"
    # )

    # Run the model on a single image
    print("Preprocessing image for Unet prediction...")

    # Resize the image
    image = Image.fromarray(image)
    image = image.resize((model_image_size, model_image_size))
    image = np.array(image)

    # Normalise the image
    print("normalising image")
    # image = image - np.min(image)
    # image = image / np.max(image)
    LOWER_LIMIT = -1
    UPPER_LIMIT = 8
    image = np.clip(image, LOWER_LIMIT, UPPER_LIMIT)
    image = image - LOWER_LIMIT
    image = image / (UPPER_LIMIT - LOWER_LIMIT)

    # Predict the mask
    print("Running Unet & predicting mask")
    prediction = model.predict(np.expand_dims(image, axis=0))[0]
    print("Unet finished, predicted mask. saving...")

    # Remove the batch dimension
    # Gem
    predicted_gem_mask = prediction[:, :, 2] > confidence
    predicted_ring_mask = prediction[:, :, 1] > confidence
    predicted_background_mask = prediction[:, :, 0] > confidence

    combined_predicted_mask = np.zeros_like(predicted_gem_mask).astype(np.uint8)
    combined_predicted_mask[predicted_gem_mask] = 2
    combined_predicted_mask[predicted_ring_mask] = 1

    combined_predicted_mask = Image.fromarray(combined_predicted_mask)
    combined_predicted_mask = combined_predicted_mask.resize(original_image.shape)
    combined_predicted_mask = np.array(combined_predicted_mask)

    return combined_predicted_mask


if __name__ == "__main__":
    test_GPU()
