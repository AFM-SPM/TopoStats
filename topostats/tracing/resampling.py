"""functions for trace smoothing"""

import numpy.typing as npt
import numpy as np
from topostats.io import LoadScans
import matplotlib.pyplot as plt
from topostats.unet_masking import make_bounding_box_square, pad_bounding_box


def interpolate_between_two_points(point1, point2, distance):
    """Interpolate between two points to create a new point at a proportional distance."""
    distance_between_points = np.linalg.norm(point2 - point1)
    if distance_between_points < distance:
        raise ValueError("distance between points is less than the desired interval")
    proportion = distance / distance_between_points
    new_point = point1 + proportion * (point2 - point1)
    return new_point


def resample_points_regular_interval(points: npt.NDArray, interval: float):
    """Resample a set of points to be at regular intervals"""

    resampled_points = []
    resampled_points.append(points[0])
    current_point_index = 1
    while True:
        current_point = resampled_points[-1]
        next_original_point = points[current_point_index]
        distance_to_next_splined_point = np.linalg.norm(next_original_point - current_point)
        # if the distance to the next splined point is less than the interval, then skip to the next point
        if distance_to_next_splined_point < interval:
            current_point_index += 1
            if current_point_index >= len(points):
                break
            continue
        new_interpolated_point = interpolate_between_two_points(current_point, next_original_point, interval)
        resampled_points.append(new_interpolated_point)

    # if the first and last points are less than 0.5 * the interval apart, then remove the last point
    if np.linalg.norm(resampled_points[0] - resampled_points[-1]) < 0.5 * interval:
        resampled_points = resampled_points[:-1]

    resampled_points_numpy = np.array(resampled_points)

    return resampled_points_numpy


def construct_grains_dictionary(file_list: list, plot: bool = False):
    grains_dictionary: dict[any] = {}

    loadscans = LoadScans(file_list, channel="dummy")
    loadscans.get_data()
    img_dict = loadscans.img_dict

    bbox_padding = 10
    grain_index = 0
    for filename, file_data in img_dict.items():
        try:

            try:
                nodestats_data = file_data["nodestats"]["above"]["stats"]
            except KeyError:
                nodestats_data = None

            # print(f"getting data from {filename}")
            image = file_data["image"]
            ordered_trace_data = file_data["ordered_traces"]["above"]
            for current_grain_index, grain_ordered_trace_data in ordered_trace_data.items():
                # print(f"  grain {current_grain_index}")
                grains_dictionary[grain_index] = {}
                grains_dictionary[grain_index]["molecule_data"] = {}
                for current_molecule_index, molecule_ordered_trace_data in grain_ordered_trace_data.items():
                    molecule_data = {}
                    molecule_data["ordered_coords"] = molecule_ordered_trace_data["ordered_coords"]
                    molecule_data["heights"] = molecule_ordered_trace_data["heights"]
                    molecule_data["distances"] = molecule_ordered_trace_data["distances"]
                    bbox = molecule_ordered_trace_data["bbox"]
                    grains_dictionary[grain_index]["molecule_data"][current_molecule_index] = molecule_data

                    splining_coords = file_data["splining"]["above"][current_grain_index][current_molecule_index][
                        "spline_coords"
                    ]
                    molecule_data["spline_coords"] = splining_coords

                    # print(molecule_ordered_trace_data.keys())
                bbox_square = make_bounding_box_square(bbox[0], bbox[1], bbox[2], bbox[3], image.shape)
                bbox_padded = pad_bounding_box(
                    bbox_square[0], bbox_square[1], bbox_square[2], bbox_square[3], image.shape, padding=bbox_padding
                )
                added_left = bbox_padded[1] - bbox[1]
                added_top = bbox_padded[0] - bbox[0]

                image_crop = image[
                    bbox_padded[0] : bbox_padded[2],
                    bbox_padded[1] : bbox_padded[3],
                ]
                full_grain_mask = file_data["grain_masks"]["above"]
                grains_dictionary[grain_index]["image"] = image_crop
                grains_dictionary[grain_index]["full_image"] = image
                grains_dictionary[grain_index]["bbox"] = bbox_padded
                grains_dictionary[grain_index]["added_left"] = added_left
                grains_dictionary[grain_index]["added_top"] = added_top
                grains_dictionary[grain_index]["padding"] = bbox_padding
                mask_crop = full_grain_mask[
                    bbox_padded[0] : bbox_padded[2],
                    bbox_padded[1] : bbox_padded[3],
                ]
                grains_dictionary[grain_index]["mask"] = mask_crop
                grains_dictionary[grain_index]["filename"] = file_data["filename"]
                grains_dictionary[grain_index]["pixel_to_nm_scaling"] = file_data["pixel_to_nm_scaling"]

                # grab node coordinates
                all_node_coords = []
                if nodestats_data is not None:
                    try:
                        grain_nodestats_data = nodestats_data[current_grain_index]
                        for _node_index, node_data in grain_nodestats_data.items():
                            node_coords = node_data["node_coords"]
                            for node_coord in node_coords:
                                all_node_coords.append(node_coord)
                    except KeyError as e:
                        if "grain_" in str(e):
                            # grain has no nodestats data here, skip
                            pass

                grains_dictionary[grain_index]["node_coords"] = np.array(all_node_coords)

                grain_index += 1
        except KeyError as e:
            if "ordered_traces" in str(e):
                print(f"no ordered traces found in {filename}")
                continue
            raise e

    if plot:
        for grain_index, grain_data in grains_dictionary.items():
            print(f"grain {grain_index}")
            print(grain_data["filename"])
            print(grain_data["pixel_to_nm_scaling"])
            image = grain_data["image"]
            plt.imshow(image)
            for molecule_index, molecule_data in grain_data["molecule_data"].items():
                ordered_coords = molecule_data["ordered_coords"]
                plt.plot(ordered_coords[:, 1], ordered_coords[:, 0], "r")
            all_node_coords = grain_data["node_coords"]
            if all_node_coords.size > 0:
                plt.plot(all_node_coords[:, 1], all_node_coords[:, 0], "b.")
            plt.show()

            mask = grain_data["mask"][:, :, 1]
            plt.imshow(mask)
            plt.show()

    print(f"found {len(grains_dictionary)} grains in {len(file_list)} images")

    return grains_dictionary
