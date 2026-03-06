"""IO functions for damage data."""

import pickle
import re
from datetime import datetime
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from damage import UnanalysedGrain, UnanalysedGrainCollection, UnanalysedMoleculeData, UnanalysedMoleculeDataCollection

from topostats.damage.damage import combine_unanalysed_grain_collections
from topostats.io import LoadScans
from topostats.unet_masking import make_bounding_box_square, pad_bounding_box_cutting_off_at_image_bounds


def get_dose_from_sample_type(sample_type: str) -> float:
    """Get the dose for a sample from the sample type string."""
    if "control" in sample_type.lower():
        return 0.0
    match = re.search(r"(\d+)_percent_damage", sample_type)
    if match:
        return float(match.group(1))
    raise ValueError(f"Could not extract dose from sample type: {sample_type}")


def get_modified_file_time_hash(file_path: Path) -> str:
    """Get a hash of the modified time of the file at the given path."""
    modified_time_timestamp = file_path.stat().st_mtime
    modified_time_iso = datetime.fromtimestamp(modified_time_timestamp).isoformat()
    return sha256(modified_time_iso.encode()).hexdigest()


# Load the files
def load_grain_models_from_topo_files(  # noqa: C901
    topo_files: list[Path],
    df_grain_stats: pd.DataFrame,
    bbox_padding: int,
    sample_type: str,
) -> UnanalysedGrainCollection:
    """Load grain models from the given TopoStats files and grain statistics dataframe."""
    grain_model_collection = UnanalysedGrainCollection(unanalysed_grains={})
    loadscans = LoadScans(img_paths=topo_files, channel="dummy")
    loadscans.get_data()
    loadscans_img_dict = loadscans.img_dict

    # Group the dataframe by image, since we will want to load all the grains from each image at once to avoid loading
    # the same image multiple times

    unique_dir_file_combinations: set = set(zip(df_grain_stats["basename"], df_grain_stats["image"]))

    print(f"unique directory and file combinations: {unique_dir_file_combinations}")

    for basename, filename in unique_dir_file_combinations:
        print(f"extracting data for file {filename} in folder {basename}")
        # locate the corresponding row in the dataframe
        df_grain_stats_image = df_grain_stats[
            (df_grain_stats["image"] == filename) & (df_grain_stats["basename"] == basename)
        ]
        if len(df_grain_stats_image) == 0:
            raise ValueError(
                f"could not find any rows in the grain stats dataframe for image {filename} and"
                f"basename {basename}. this should not happen, debug this!"
            )

        # load the corresponding image file
        try:
            file_data = loadscans_img_dict[filename]
        except KeyError as e:
            print(f"keys: {list(loadscans_img_dict.keys())}")
            raise KeyError(f"could not find file data for image {filename} in loaded scans. debug this!") from e

        full_image = file_data["image"]
        full_mask = file_data["grain_tensors"]["above"]
        pixel_to_nm_scaling = file_data["pixel_to_nm_scaling"]
        ordered_trace_data = file_data["ordered_traces"]["above"]

        # Grab nodestats data
        try:
            nodestats_data = file_data["nodestats"]["above"]["stats"]
        except KeyError:
            nodestats_data = None

        # grab individual grain data, based on each row of the dataframe

        # grab the grain indexes (local) ie, 0-N for the file
        grain_indexes_from_df = df_grain_stats_image["grain_number"].unique()
        grain_indexes_from_file = {int(grain_id.replace("grain_", "")) for grain_id in ordered_trace_data.keys()}

        if not set(grain_indexes_from_df) == grain_indexes_from_file:
            print(
                f"WARN: grain indexes from dataframe and file mismatch. extra indexes in dataframe: "
                f"{set(grain_indexes_from_df) - grain_indexes_from_file} extra indexes in "
                f"file: {grain_indexes_from_file - set(grain_indexes_from_df)}"
            )

        # Now we can confidently grab the grain data based on the dataframe
        for grain_index in grain_indexes_from_df:
            grain_id_str = f"grain_{grain_index}"
            if grain_id_str not in ordered_trace_data:
                print(
                    f"WARN: grain id {grain_id_str} from dataframe not found in file data. THIS SHOULD NOT HAPPEN, DEBUG THIS!"
                )
                continue
            grain_ordered_trace_data = ordered_trace_data[grain_id_str]
            df_grain_stats_grain = df_grain_stats_image[df_grain_stats_image["grain_number"] == grain_index]
            assert (
                len(df_grain_stats_grain) == 1
            ), f"expected exactly one row in the grain stats dataframe for grain {grain_index} in"
            f"image {filename}, but found {len(df_grain_stats_grain)}. DEBUG THIS!"

            dose_percentage = get_dose_from_sample_type(sample_type)
            aspect_ratio = df_grain_stats_grain["aspect_ratio"].values[0]
            total_contour_length = df_grain_stats_grain["total_contour_length"].values[0]
            num_crossings = df_grain_stats_grain["num_crossings"].values[0]
            smallest_bounding_area = df_grain_stats_grain["smallest_bounding_area"].values[0]

            # get the bounding box for the grain, stored as the same value in each molecule's data so just grab from
            # the first one
            grain_bbox = list(grain_ordered_trace_data.values())[0]["bbox"]

            # make the bounding box square and add some padding. bbox will be the same for all molecules so this is okay
            bbox_square = make_bounding_box_square(
                crop_min_row=grain_bbox[0],
                crop_min_col=grain_bbox[1],
                crop_max_row=grain_bbox[2],
                crop_max_col=grain_bbox[3],
                image_shape=full_image.shape,
            )
            bbox_padded = pad_bounding_box_cutting_off_at_image_bounds(
                crop_min_row=bbox_square[0],
                crop_min_col=bbox_square[1],
                crop_max_row=bbox_square[2],
                crop_max_col=bbox_square[3],
                image_shape=full_image.shape,
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
            molecule_data_collection = UnanalysedMoleculeDataCollection(molecules={})
            for current_molecule_id_str, molecule_ordered_trace_data in grain_ordered_trace_data.items():
                print(f"-- processing molecule {current_molecule_id_str}")
                molecule_id = int(re.sub(r"mol_", "", current_molecule_id_str))

                ordered_coords = molecule_ordered_trace_data["ordered_coords"]
                # adjust the ordered coords to account for padding
                ordered_coords[:, 0] -= bbox_added_top
                ordered_coords[:, 1] -= bbox_added_left
                molecule_data_ordered_coords = ordered_coords

                molecule_data_ordered_coords_heights = molecule_ordered_trace_data["heights"]
                molecule_data_distances = molecule_ordered_trace_data["distances"]
                molecule_data_circular = molecule_ordered_trace_data["mol_stats"]["circular"]

                splining_coords = file_data["splining"]["above"][grain_id_str][current_molecule_id_str]["spline_coords"]
                # adjust the splining coords to account for padding
                splining_coords[:, 0] -= bbox_added_top
                splining_coords[:, 1] -= bbox_added_left
                molecule_data_spline_coords = splining_coords

                # get the splining heights
                molecule_data_spline_coords_heights = np.zeros(splining_coords.shape[0], dtype=np.float64)
                for point_index in range(splining_coords.shape[0]):
                    coord_int = tuple(splining_coords[point_index].astype(int))
                    molecule_data_spline_coords_heights[point_index] = image[coord_int[0], coord_int[1]]

                # get curvature information (pertaining to the splines)
                try:
                    curvature_data_grains = file_data["grain_curvature_stats"]["above"]["grains"]
                    curvature_data_grain = curvature_data_grains[grain_id_str]
                    curvature_data_molecules = curvature_data_grain["molecules"]
                    curvature_data_molecule = curvature_data_molecules[current_molecule_id_str]
                    molecule_data_curvature_data = curvature_data_molecule
                except KeyError:
                    print(
                        f"could not find curvature data for grain {grain_id_str}"
                        f"molecule {current_molecule_id_str}), setting to None. file keys: {list(file_data.keys())}"
                    )
                    molecule_data_curvature_data = None

                molecule_data = UnanalysedMoleculeData(
                    molecule_id=molecule_id,
                    ordered_coords_heights=molecule_data_ordered_coords_heights,
                    spline_coords_heights=molecule_data_spline_coords_heights,
                    distances=molecule_data_distances,
                    circular=molecule_data_circular,
                    curvature_data=molecule_data_curvature_data,
                    spline_coords=molecule_data_spline_coords,
                    ordered_coords=molecule_data_ordered_coords,
                )
                molecule_data_collection.add_molecule(molecule_data)

            # get nodestats data for the grain. messy but I don't want to refactor how topostats stores this atm.
            all_node_coords = []
            num_nodes: int = 0
            if nodestats_data is not None:
                try:
                    grain_nodestats_data = nodestats_data[grain_id_str]
                    num_nodes = len(grain_nodestats_data)
                    for _node_index, node_data in grain_nodestats_data.items():
                        node_coords = node_data["coords"]
                        # adjust the node coords to account for padding
                        node_coords[:, 0] -= bbox_added_top
                        node_coords[:, 1] -= bbox_added_left
                        for node_coord in node_coords:
                            all_node_coords.append(node_coord)
                except KeyError as e:
                    if "grain_" in str(e):
                        # grain has no nodestats data here, skip
                        pass
            all_node_coords_array = np.array(all_node_coords)

            grain_model = UnanalysedGrain(
                file_grain_id=grain_index,
                filename=filename,
                pixel_to_nm_scaling=pixel_to_nm_scaling,
                folder=str(sample_type),
                percent_damage=dose_percentage,
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
                node_coords=all_node_coords_array,
                num_nodes=num_nodes,
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
        sample_type = local_folder.replace("../all_data/", "").replace("../test_subset_data/", "")
        dir_loaded_sample_type = dir_loaded_datasets / sample_type
        print(f"checking folder {sample_type} with {len(group)} files")
        print("calculating hashes for topostats files...")
        file_paths_and_hashes_topostats: dict[Path, str] = {}
        # construct the file paths for each unique combination of image and folder and check if it exists.
        for _, row in group.iterrows():
            filename = row["image"]
            basename = row["basename"]
            # reconstruct the path to the file using the basename, image name and structure of directories.
            if "all_data" in basename:
                dir_topo_file = Path(str(basename).replace("../all_data/", ""))
            if "test_subset_data" in basename:
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
                sample_type=sample_type,
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
