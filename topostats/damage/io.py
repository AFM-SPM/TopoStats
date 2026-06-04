"""IO functions for damage data."""

import pickle
import re
from datetime import datetime
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from topostats.damage.classes import (
    UnanalysedGrain,
    UnanalysedGrainCollection,
    UnanalysedMoleculeData,
    UnanalysedMoleculeDataCollection,
    combine_unanalysed_grain_collections,
)
from topostats.io import LoadScans
from topostats.unet_masking import make_bounding_box_square, pad_bounding_box_cutting_off_at_image_bounds


def get_dose_from_folder_name(folder_name: str) -> float:
    """Get the dose for a sample from the folder name string."""
    if "control" in folder_name.lower():
        return 0.0
    match = re.search(r"(\d+)_percent_damage", folder_name)
    if match:
        return float(match.group(1))
    raise ValueError(f"Could not extract dose from folder name: {folder_name}")


def get_sample_type_from_folder_name(folder_name: str) -> str:
    """Get the sample type from the folder name string."""
    # Take as the first string before a /, or the whole string if there is no /.
    return folder_name.split("/")[0]


def get_modified_file_time_hash(file_path: Path) -> str:
    """Get a hash of the modified time of the file at the given path."""
    modified_time_timestamp = file_path.stat().st_mtime
    modified_time_iso = datetime.fromtimestamp(modified_time_timestamp).isoformat()
    return sha256(modified_time_iso.encode()).hexdigest()


# Load the files
def load_grain_models_from_topo_files(  # noqa: C901
    topo_files: list[str | Path],
    df_grain_stats: pd.DataFrame,
    bbox_padding: int,
    folder_name: str,
) -> UnanalysedGrainCollection:
    """Load grain models from the given TopoStats files and grain statistics dataframe."""
    grain_model_collection = UnanalysedGrainCollection(unanalysed_grains={})
    loadscans = LoadScans(img_paths=topo_files, config={"loading": {"channel": "dummy"}})
    loadscans.get_data()
    loadscans_img_dict = loadscans.img_dict

    # Group the dataframe by image, since we will want to load all the grains from each image at once to avoid loading
    # the same image multiple times

    unique_dir_file_combinations: set = set(zip(df_grain_stats["basename"], df_grain_stats["image"]))

    print(f"unique directory and file combinations: {unique_dir_file_combinations}")

    for basename, filename in unique_dir_file_combinations:
        filename_with_ext = f"{filename}.topostats"
        # locate the corresponding row in the dataframe
        df_grain_stats_image = df_grain_stats[
            (df_grain_stats["image"] == filename) & (df_grain_stats["basename"] == basename)
        ]
        assert isinstance(df_grain_stats_image, pd.DataFrame)
        if len(df_grain_stats_image) == 0:
            raise ValueError(
                f"could not find any rows in the grain stats dataframe for image {filename} and"
                f"basename {basename}. this should not happen, debug this!"
            )

        # load the corresponding image file
        try:
            topostats_obj = loadscans_img_dict[filename_with_ext]
        except KeyError as e:
            print(f"keys: {list(loadscans_img_dict.keys())}")
            raise KeyError(f"could not find file data for image {filename} in loaded scans. debug this!") from e

        full_image = np.array(topostats_obj.image)
        full_mask = np.array(topostats_obj.full_mask_tensor)[:, :, 1]
        pixel_to_nm_scaling = topostats_obj.require_pixel_to_nm_scaling()
        grain_crops = topostats_obj.require_grain_crops()

        # grab individual grain data, based on each row of the dataframe

        # grab the grain indexes (local) ie, 0-N for the file
        grain_indexes_from_df: list[int] = df_grain_stats_image["grain_number"].unique()
        grain_indexes_from_file = topostats_obj.require_grain_crops().keys()

        if not set(grain_indexes_from_df) == grain_indexes_from_file:
            print(
                f"WARN: grain indexes from dataframe and file mismatch. extra indexes in dataframe: "
                f"{set(grain_indexes_from_df) - grain_indexes_from_file} extra indexes in "
                f"file: {grain_indexes_from_file - set(grain_indexes_from_df)}"
            )

        # Now we can confidently grab the grain data based on the dataframe
        for grain_index in grain_indexes_from_df:
            grain_crop = grain_crops[grain_index]
            df_grain_stats_grain = df_grain_stats_image[df_grain_stats_image["grain_number"] == grain_index]
            assert (
                len(df_grain_stats_grain) == 1
            ), f"expected exactly one row in the grain stats dataframe for grain {grain_index} in"
            f"image {filename}, but found {len(df_grain_stats_grain)}. DEBUG THIS!"

            dose_percentage = get_dose_from_folder_name(folder_name)
            sample_type = get_sample_type_from_folder_name(folder_name)
            aspect_ratio = df_grain_stats_grain["aspect_ratio"].values[0]
            total_contour_length = df_grain_stats_grain["total_contour_length"].values[0]
            num_crossings = df_grain_stats_grain["num_crossings"].values[0]
            smallest_bounding_area = df_grain_stats_grain["smallest_bounding_area"].values[0]

            # get the bounding box for the grain, stored as the same value in each molecule's data so just grab from
            # the first one
            grain_bbox = grain_crop.bbox

            # make the bounding box square and add some padding. bbox will be the same for all molecules so this is okay
            bbox_square = make_bounding_box_square(
                crop_min_row=grain_bbox[0],
                crop_min_col=grain_bbox[1],
                crop_max_row=grain_bbox[2],
                crop_max_col=grain_bbox[3],
                image_shape=(full_image.shape[0], full_image.shape[1]),
            )
            bbox_padded = pad_bounding_box_cutting_off_at_image_bounds(
                crop_min_row=bbox_square[0],
                crop_min_col=bbox_square[1],
                crop_max_row=bbox_square[2],
                crop_max_col=bbox_square[3],
                image_shape=(full_image.shape[0], full_image.shape[1]),
                padding=bbox_padding,
            )
            # keep track of how much padding we have added to the original bbox on each side so we can adjust
            # coordinates accordingly later.
            bbox_added_left = bbox_padded[1] - grain_bbox[1]
            bbox_added_top = bbox_padded[0] - grain_bbox[0]

            # create crops of the mask and image based on the new square bounding box + padding
            mask = full_mask[
                bbox_padded[0] : bbox_padded[2],
                bbox_padded[1] : bbox_padded[3],
            ]

            image = full_image[
                bbox_padded[0] : bbox_padded[2],
                bbox_padded[1] : bbox_padded[3],
            ]

            # get the molecule data
            grain_molecule_data = grain_crop.ordered_trace.require_molecule_data()
            molecule_data_collection = UnanalysedMoleculeDataCollection(molecules={})
            for molecule_id, molecule_ordered_trace_data in grain_molecule_data.items():
                molecule_data_ordered_coords = molecule_ordered_trace_data.ordered_coords
                assert molecule_data_ordered_coords is not None
                # adjust the ordered coords to account for padding
                molecule_data_ordered_coords[:, 0] -= bbox_added_top
                molecule_data_ordered_coords[:, 1] -= bbox_added_left
                molecule_data_ordered_coords_heights = molecule_ordered_trace_data.heights
                assert molecule_data_ordered_coords_heights is not None
                molecule_data_distances = molecule_ordered_trace_data.distances
                assert molecule_data_distances is not None
                molecule_data_circular = molecule_ordered_trace_data.circular
                assert molecule_data_circular is not None
                splining_coords = molecule_ordered_trace_data.require_splined_coords()
                # adjust the splining coords to account for padding
                splining_coords[:, 0] -= bbox_added_top
                splining_coords[:, 1] -= bbox_added_left
                molecule_data_spline_coords_px = splining_coords
                # get the splining heights
                molecule_data_spline_coords_heights = np.zeros(splining_coords.shape[0], dtype=np.float64)
                for point_index in range(splining_coords.shape[0]):
                    coord_int = tuple(splining_coords[point_index].astype(int))
                    molecule_data_spline_coords_heights[point_index] = image[coord_int[0], coord_int[1]]
                # get curvature information (pertaining to the splines)
                molecule_curvature_stats = molecule_ordered_trace_data.require_curvature_stats()

                molecule_data = UnanalysedMoleculeData(
                    molecule_id=molecule_id,
                    ordered_coords_heights=molecule_data_ordered_coords_heights,
                    spline_coords_heights=molecule_data_spline_coords_heights,
                    distances=molecule_data_distances,
                    circular=molecule_data_circular,
                    curvature_data=molecule_curvature_stats,
                    spline_coords_px=molecule_data_spline_coords_px,
                    ordered_coords=molecule_data_ordered_coords,
                    pixel_to_nm_scaling=pixel_to_nm_scaling,
                )
                molecule_data_collection.add_molecule(molecule_data)

            grain_model = UnanalysedGrain(
                file_grain_id=grain_index,
                filename=filename,
                pixel_to_nm_scaling=pixel_to_nm_scaling,
                folder=str(folder_name),
                percent_damage=dose_percentage,
                sample_type=sample_type,
                bbox=bbox_padded,
                image=image,
                mask=mask,
                aspect_ratio=aspect_ratio,
                total_contour_length=total_contour_length,
                num_crossings=num_crossings,
                molecule_data_collection=molecule_data_collection,
                added_left=bbox_added_left,
                added_top=bbox_added_top,
                padding=bbox_padding,
                smallest_bounding_area=smallest_bounding_area,
            )

            grain_model_collection.add_grain(grain_model)
    return grain_model_collection


def construct_grains_collection_from_topostats_files(
    dir_to_save_cache_files: Path,
    dir_processed_data: Path,
    df_grain_stats: pd.DataFrame,
    bbox_padding: int,
    force_reload: bool = False,
) -> UnanalysedGrainCollection:
    """
    Construct a grain model collection from the TopoStats files and grain stats dataframe.

    Parameters
    ----------
    dir_to_save_cache_files: Path
        Directory where cache files will be saved.
    dir_processed_data: Path
        Directory where the processed TopoStats files are located.
    df_grain_stats: pd.DataFrame
        Dataframe containing the grain statistics to be aligned with the data in the TopoStats files.
    bbox_padding: int
        Amount of padding to add to the bounding boxes when loading the grain data from the TopoStats files.
    force_reload: bool
        Whether to force reload the data from the TopoStats files even if the hash matches the previous hash.
    """
    grain_model_collection = UnanalysedGrainCollection(unanalysed_grains={})
    dir_loaded_datasets = dir_to_save_cache_files / "loaded_datasets"
    assert dir_loaded_datasets.exists(), f"could not find dataset hash directory at {dir_loaded_datasets}"

    # iterate through each topostats image file that remains in the grain stats dataframe and extract the grain data
    # from the topostats files and store them.
    unique_file_folder_combinations = df_grain_stats[["image", "basename"]].drop_duplicates()
    print(f"found {len(unique_file_folder_combinations)} unique topostats files in the grain stats dataframe")

    # Split the files by folder
    unique_file_folder_combinations_grouped = unique_file_folder_combinations.groupby("basename")
    print(unique_file_folder_combinations_grouped.groups.keys())

    # calculate a hash for each folder by combining the hashes of files in that folder.
    for local_folder, group in unique_file_folder_combinations_grouped:
        sample_type = str(local_folder)
        sample_type = str(sample_type).replace("../all_data/", "")
        sample_type = str(sample_type).replace("../test_subset_data/", "")
        dir_loaded_sample_type = dir_loaded_datasets / sample_type
        print(f"checking folder {sample_type} with {len(group)} files")
        print("calculating hashes for topostats files...")
        file_paths_and_hashes_topostats: dict[Path, str] = {}
        # construct the file paths for each unique combination of image and folder and check if it exists.
        for _, row in group.iterrows():
            filename = str(row["image"])
            basename = str(row["basename"])
            # reconstruct the path to the file using the basename, image name and structure of directories.
            if "all_data" in str(basename):
                dir_topo_file = Path(str(basename).replace("../all_data/", ""))
            if "test_subset_data" in str(basename):
                dir_topo_file = Path(str(basename).replace("../test_subset_data/", ""))
            dir_topo_file = dir_processed_data / dir_topo_file / "processed"
            assert dir_topo_file.exists(), f"could not find folder at {dir_topo_file}"
            file_topostats = dir_topo_file / f"{filename}.topostats"
            assert file_topostats.exists(), f"could not find topostats file at {file_topostats}"

            file_topostats_timestamp_hash = get_modified_file_time_hash(file_topostats)

            file_paths_and_hashes_topostats[file_topostats] = file_topostats_timestamp_hash

        # create hash for the folder by combining the hashes of the files in the folder
        combined_hash_string = "".join(sorted(file_paths_and_hashes_topostats.values()))
        folder_hash = sha256(combined_hash_string.encode()).hexdigest()

        # load the previous hash from text file if it exists
        previous_hash_file = dir_loaded_sample_type / "hash.txt"
        print(f"checking previous hash for folder {sample_type} at {previous_hash_file}")
        previous_hash = None
        if previous_hash_file.exists():
            previous_hash = previous_hash_file.read_text()

        print(
            f"folder {sample_type} hash calculated: {folder_hash}. previous hash: {previous_hash} force reload: {force_reload}"
        )

        if previous_hash == folder_hash and not force_reload:
            # check if the saved data for this folder exists and if it does, load it instead of loading the data
            # from the topostats files
            file_previous_loaded_data = dir_loaded_sample_type / "data.pkl"
            if file_previous_loaded_data.exists():
                print(
                    f"folder {sample_type} has not changed since last load, skipping loading it and using previous saved data"
                )
                with Path.open(file_previous_loaded_data, "rb") as f:
                    grain_model_collection_folder: UnanalysedGrainCollection = pickle.load(f)  # noqa: S301
                    # Combine the grain model collection for this folder with the main grain model collection
                    grain_model_collection = combine_unanalysed_grain_collections(
                        [grain_model_collection, grain_model_collection_folder]
                    )
            else:
                print(
                    f"hash for folder {sample_type} matched previous, but could not locate "
                    f"saved data: {file_previous_loaded_data}"
                )
        else:
            if not force_reload:
                print(
                    f"folder {sample_type} hash has changed since last load, loading the data from the topostats files"
                )
            else:
                print(f"forcing reload, loading the data from the topostats files for folder {sample_type}")

            # calculate a subset of the dataframe for just this folder
            df_grain_stats_folder = df_grain_stats[df_grain_stats["basename"] == local_folder]
            grain_model_collection_folder = load_grain_models_from_topo_files(
                topo_files=list(file_paths_and_hashes_topostats.keys()),
                df_grain_stats=df_grain_stats_folder,
                bbox_padding=bbox_padding,
                folder_name=sample_type,
            )

            print(
                "loaded topostats file data into grain model collection."
                "Saving loaded data to .pkl file and saving hash."
            )

            # after loading all the grains for the folder, save the model to a pickle and save the hash
            # for the folder
            file_to_save = dir_loaded_sample_type / "data.pkl"
            # ensure the parent folder is created
            file_to_save.parent.mkdir(parents=True, exist_ok=True)
            with Path.open(file_to_save, "wb") as f:
                pickle.dump(grain_model_collection_folder, f)
            # save the hash for the folder
            print(f"saving hash for folder {sample_type} to {previous_hash_file}")
            previous_hash_file.parent.mkdir(parents=True, exist_ok=True)
            with Path.open(previous_hash_file, "w") as f:
                f.write(folder_hash)

            # combine the grain model collection for this folder with the main grain model collection
            grain_model_collection = combine_unanalysed_grain_collections(
                [grain_model_collection, grain_model_collection_folder]
            )

    return grain_model_collection
