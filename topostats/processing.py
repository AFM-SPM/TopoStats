"""Functions for processing data."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from topostats import __version__
from topostats.filters import Filters
from topostats.grains import Grains
from topostats.grainstats import GrainStats
from topostats.dna_protein_analysis import multiClassObjects
from topostats.io import get_out_path, save_topostats_file
from topostats.logs.logs import LOGGER_NAME, setup_logger
from topostats.plottingfuncs import Images, add_pixel_to_nm_to_plotting_config
from topostats.statistics import image_statistics
from topostats.tracing.dnatracing import trace_image
from topostats.utils import create_empty_dataframe

# pylint: disable=broad-except
# pylint: disable=line-too-long
# pylint: disable=too-many-arguments
# pylint: disable=too-many-branches
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=too-many-nested-blocks
# pylint: disable=unnecessary-dict-index-lookup

LOGGER = setup_logger(LOGGER_NAME)


def run_filters(
    unprocessed_image: np.ndarray,
    pixel_to_nm_scaling: float,
    filename: str,
    filter_out_path: Path,
    core_out_path: Path,
    filter_config: dict,
    plotting_config: dict,
) -> np.ndarray | None:
    """
    Filter and flatten an image. Optionally plots the results, returning the flattened image.

    Parameters
    ----------
    unprocessed_image : np.ndarray
        Image to be flattened.
    pixel_to_nm_scaling : float
        Scaling factor for converting pixel length scales to nanometres.
        ie the number of pixels per nanometre.
    filename : str
        File name for the image.
    filter_out_path : Path
        Output directory for step-by-step flattening plots.
    core_out_path : Path
        General output directory for outputs such as the flattened image.
    filter_config : dict
        Dictionary of configuration for the Filters class to use when initialised.
    plotting_config : dict
        Dictionary of configuration for plotting output images.

    Returns
    -------
    Union[np.ndarray, None]
        Either a numpy array of the flattened image, or None if an error occurs or
        flattening is disabled in the configuration.
    """
    if filter_config["run"]:
        filter_config.pop("run")
        LOGGER.info(f"[{filename}] Image dimensions: {unprocessed_image.shape}")
        LOGGER.info(f"[{filename}] : *** Filtering ***")
        filters = Filters(
            image=unprocessed_image,
            filename=filename,
            pixel_to_nm_scaling=pixel_to_nm_scaling,
            **filter_config,
        )
        filters.filter_image()
        # Optionally plot filter stage
        if plotting_config["run"]:
            plotting_config.pop("run")
            LOGGER.info(f"[{filename}] : Plotting Filtering Images")
            if plotting_config["image_set"] == "all":
                filter_out_path.mkdir(parents=True, exist_ok=True)
                LOGGER.debug(f"[{filename}] : Target filter directory created : {filter_out_path}")
            # Generate plots
            for plot_name, array in filters.images.items():
                if plot_name not in ["scan_raw"]:
                    if plot_name == "extracted_channel":
                        array = np.flipud(array.pixels)
                    plotting_config["plot_dict"][plot_name]["output_dir"] = (
                        core_out_path if plotting_config["plot_dict"][plot_name]["core_set"] else filter_out_path
                    )
                    try:
                        Images(array, **plotting_config["plot_dict"][plot_name]).plot_and_save()
                        Images(array, **plotting_config["plot_dict"][plot_name]).plot_histogram_and_save()
                    except AttributeError:
                        LOGGER.info(f"[{filename}] Unable to generate plot : {plot_name}")
            plotting_config["run"] = True
        # Always want the 'z_threshed' plot (aka "Height Thresholded") but in the core_out_path
        plot_name = "z_threshed"
        plotting_config["plot_dict"][plot_name]["output_dir"] = core_out_path
        Images(
            filters.images["gaussian_filtered"],
            filename=filename,
            **plotting_config["plot_dict"][plot_name],
        ).plot_and_save()

        return filters.images["gaussian_filtered"]

    # Otherwise, return None and warn that initial processing is disabled.
    LOGGER.error(
        "You have not included running the initial filter stage. This is required for all subsequent "
        "stages of processing. Please check your configuration file."
    )

    return None


def run_grains(
    image: np.ndarray,
    pixel_to_nm_scaling: float,
    filename: str,
    grain_out_path: Path,
    core_out_path: Path,
    plotting_config: dict,
    grains_config: dict,
):
    """
    Identify grains (molecules) and optionally plots the results.
    """
    if grains_config.get("run"):
        grains_config.pop("run")

        LOGGER.info(f"[{filename}] : *** Grain Finding ***")
        grains = Grains(
            image=image,
            filename=filename,
            pixel_to_nm_scaling=pixel_to_nm_scaling,
            **grains_config,
        )
        grains.find_grains()

        for direction, region_props in grains.region_properties.items():
            num_grains = len(region_props)
            LOGGER.info(f"[{filename}] : Grains found for direction {direction} : {num_grains}")

            if num_grains == 0:
                LOGGER.warning(f"[{filename}] : No grains found for direction {direction}")

            # Plotting logic if grains were found
            if plotting_config.get("run"):
                LOGGER.info(f"[{filename}] : Plotting Grain Finding Images for direction {direction}")
                grain_out_path_direction = grain_out_path / direction
                grain_out_path_direction.mkdir(parents=True, exist_ok=True)
                LOGGER.debug(f"[{filename}] : Target grain directory created : {grain_out_path_direction}")

                for plot_name, array in grains.directions[direction].items():
                    if len(array.shape) == 3 and "mask" not in plot_name:  # Skip masks for individual plotting
                        array = array[:, :, 1]  # Assuming this selects the correct layer
                    LOGGER.info(f"[{filename}] : Plotting {plot_name} image")
                    
                    # Ensure output_dir is set before plotting
                    plot_dict = plotting_config["plot_dict"][plot_name]
                    plot_dict["output_dir"] = grain_out_path_direction
                    
                    Images(array, **plot_dict).plot_and_save()

                # Plot bounding boxes and colored regions
                bounding_box_dict = plotting_config["plot_dict"]["bounding_boxes"]
                bounding_box_dict["output_dir"] = grain_out_path_direction
                Images(
                    grains.directions[direction]["coloured_regions"],
                    **bounding_box_dict,
                    region_properties=grains.region_properties[direction],
                ).plot_and_save()

                # Iterate over all classes for the labeled regions
                for unet_class in range(1, grains.directions[direction]["labelled_regions_02"].shape[2]):
                    class_mask = grains.directions[direction]["labelled_regions_02"][:, :, unet_class]
                    class_mask = class_mask.astype(np.float32)  # Ensure correct data type
                    LOGGER.debug(f"[{filename}] : Class mask shape: {class_mask.shape}, dtype: {class_mask.dtype}")

                    class_box_dict = plotting_config["plot_dict"]["coloured_boxes"]
                    class_box_dict["output_dir"] = grain_out_path_direction
                    LOGGER.info(f"[{filename}] : Plotting class {unet_class} image")
                    
                    Images(
                        class_mask,
                        **class_box_dict,
                        region_properties=grains.region_properties[direction],
                    ).plot_and_save()

                # Overlay the mask in core_out_path with a unique filename for each class
                for unet_class in range(1, grains.directions[direction]["labelled_regions_02"].shape[2]):
                    mask_overlay_dict = plotting_config["plot_dict"]["mask_overlay"]
                    mask_overlay_dict["output_dir"] = core_out_path
                    masked_array = grains.directions[direction]["removed_small_objects"][:, :, unet_class].astype(np.float32)
                    
                    Images(
                        image,
                        filename=f"{filename}_{direction}_class_{unet_class}_masked",
                        masked_array=masked_array,
                        **mask_overlay_dict,
                    ).plot_and_save()

        # Prepare grain masks for return
        grain_masks = {direction: grains.directions[direction]["labelled_regions_02"] for direction in grains.directions}
        return grain_masks

    LOGGER.info(f"[{filename}] Detection of grains disabled, returning empty data frame.")
    return None



"""
def run_dna_protein_analysis(
    image: np.ndarray,
    grain_masks: np.ndarray,
    filename: str,
    pixel_to_nm_scaling: float,
) -> pd.DataFrame:
    
    for direction, _ in grain_masks.items():
        dna_protein = multiClassObjects(
            image=image,
            grain=grain_masks[direction],
            filename=filename,
            pixel_to_nm_scaling=pixel_to_nm_scaling
        )

        dna_or_protein_regions = dna_protein.isolate_connected_classes()

        for region_id, masks in dna_or_protein_regions.items():
            dna_only_mask = masks['dna_only']
            protein_only_mask = masks['protein_only']

            dna_filename = f"/Users/laura/Desktop/dna_only_region_{region_id}.npy"
            protein_filename = f"/Users/laura/Desktop/protein_only_region_{region_id}.npy"

            np.save(dna_filename, dna_only_mask)
            np.save(protein_filename, protein_only_mask)

    return None
"""

def run_grainstats(
    image: np.ndarray,
    pixel_to_nm_scaling: float,
    grain_masks: dict,
    filename: str,
    grainstats_config: dict,
    plotting_config: dict,
    grain_out_path: Path,
):
    """
    Calculate grain statistics for an image and optionally plot the results.
    """
    if grainstats_config.get("run"):
        grainstats_config.pop("run")
        try:
            LOGGER.info(f"[{filename}] : *** Grain Statistics ***")
            grain_plot_dict = {
                key: value
                for key, value in plotting_config["plot_dict"].items()
                if key in ["grain_image", "grain_mask", "grain_mask_image"]
            }
            grainstats_dict = {}
            height_profiles_dict = {}

            for direction in grain_masks.keys():
                LOGGER.info(f"[{filename}] : Full Mask dimensions: {grain_masks[direction].shape}")
                assert len(grain_masks[direction].shape) == 3
                dna_class_mask = grain_masks[direction][:, :, grain_masks[direction].shape[2]-1]
                LOGGER.info(f"[{filename}] : DNA Mask dimensions: {dna_class_mask.shape}")

                if np.max(dna_class_mask) == 0:
                    LOGGER.warning(f"[{filename}] : No grains exist for {direction}. Skipping.")
                    grainstats_dict[direction] = create_empty_dataframe()
                    continue

                unique_molecules = np.unique(dna_class_mask)[1:]  # Skip background (0)
                molecule_map = {molecule: idx + 1 for idx, molecule in enumerate(unique_molecules)}

                grainstats_dfs = []

                for unet_class in range(1, grain_masks[direction].shape[2]):
                    class_mask = grain_masks[direction][:, :, unet_class]
                    overlap_mask = np.where(class_mask, dna_class_mask, 0)

                    grains_plot_data = []  # Initialize for this class


                    for molecule, molecule_number in molecule_map.items():
                        relabeled_mask = np.where(overlap_mask == molecule, molecule_number, 0)
                        relabeled_mask = relabeled_mask.astype(np.uint8)
                        labeled_array, num_features = label(relabeled_mask)
                        LOGGER.info(f"[{filename}] : Found {num_features} distinct regions for molecule {molecule_number} in direction {direction}, class {unet_class}.")

                        for region_idx in range(1, num_features + 1):

                            component_mask = np.where(labeled_array == region_idx, molecule_number, 0)

                            if np.max(component_mask) == 0:
                                LOGGER.warning(f"[{filename}] : Skipping empty region {region_idx} for molecule {molecule_number}.")
                                continue

                            LOGGER.info(f"[{filename}] : Processing region {region_idx} for molecule {molecule_number} in direction {direction}, class {unet_class}.")

                            grainstats_calculator = GrainStats(
                                data=image,
                                labelled_data=component_mask,
                                pixel_to_nanometre_scaling=pixel_to_nm_scaling,
                                direction=direction,
                                base_output_dir=grain_out_path,
                                image_name=filename,
                                plot_opts=grain_plot_dict,
                                **grainstats_config,
                            )
                            grainstats, grains_plot_data_part, height_profiles_dict[direction] = (
                                grainstats_calculator.calculate_stats()
                            )

                            grainstats["threshold"] = direction
                            grainstats["class"] = unet_class
                            grainstats["molecule_number"] = molecule_number
                            grainstats["object_number"] = region_idx  # Unique region identifier

                            grainstats_dfs.append(grainstats)

                            # Ensure molecule_number and object_number are part of the plot data
                            for plot_data in grains_plot_data_part:
                                plot_data["molecule_number"] = molecule_number
                                plot_data["object_number"] = region_idx

                            grains_plot_data.extend(grains_plot_data_part)

                    # Plot grains for this class if required
                    if plotting_config["image_set"] == "all":
                        LOGGER.info(f"[{filename}] : Plotting grain images for class: {unet_class}, direction: {direction}.")
                        for plot_data in grains_plot_data:
                            class_label = unet_class
                            molecule_label = plot_data.get("molecule_number", "unknown_molecule")
                            object_label = plot_data.get("object_number", "unknown_object")
                            plot_filename = f"{filename}_{plot_data['name']}_molecule{molecule_label}_class{class_label}_object{object_label}"

                            LOGGER.info(
                                f"[{filename}] : Plotting grain image {plot_filename} for direction: {direction}."
                            )

                            Images(
                                data=plot_data["data"],
                                output_dir=plot_data["output_dir"],
                                filename=plot_filename,
                                **plotting_config["plot_dict"][plot_data["name"]],
                            ).plot_and_save()

                # Combine stats for current direction
                if grainstats_dfs:
                    grainstats_dict[direction] = pd.concat(
                        [df for df in grainstats_dfs if not df.empty],
                        ignore_index=True
                    )
                else:
                    grainstats_dict[direction] = create_empty_dataframe()

            # Combine results from both directions
            grainstats_df = create_empty_dataframe()
            if "above" in grainstats_dict and "below" in grainstats_dict:
                grainstats_df = pd.concat([grainstats_dict["below"], grainstats_dict["above"]], ignore_index=True)
            elif "above" in grainstats_dict:
                grainstats_df = grainstats_dict["above"]
            elif "below" in grainstats_dict:
                grainstats_df = grainstats_dict["below"]

            # Reset index to avoid keeping the old index
            grainstats_df = grainstats_df.reset_index(drop=True)

            return grainstats_df, height_profiles_dict

        except Exception as e:
            LOGGER.error(f"[{filename}] : Errors occurred whilst calculating grain statistics: {e}")
            return create_empty_dataframe(), {}

    else:
        LOGGER.info(f"[{filename}] : Calculation of grainstats disabled, returning empty dataframe.")
        return create_empty_dataframe(), {}

from scipy.ndimage import label

def run_dnatracing(  # noqa: C901
    image: np.ndarray,
    grain_masks: dict,
    pixel_to_nm_scaling: float,
    image_path: Path,
    filename: str,
    core_out_path: Path,
    grain_out_path: Path,
    dnatracing_config: dict,
    plotting_config: dict,
    results_df: pd.DataFrame = None,
):
    """
    Trace DNA molecule for the supplied grains, adding results to statistics DataFrames and optionally plot results.
    """
    # Create empty dataframe if none is passed
    if results_df is None:
        results_df = create_empty_dataframe()

    try:
        grain_trace_data = None
        if dnatracing_config.get("run"):
            dnatracing_config.pop("run")
            LOGGER.info(f"[{filename}] : *** DNA Tracing ***")
            tracing_stats = defaultdict(list)  # Store stats per direction
            grain_trace_data = defaultdict()

            # Iterate over directions in grain_masks
            for direction in grain_masks.keys():
                LOGGER.info(f"[{filename}] : Mask dimensions: {grain_masks[direction].shape}")
                assert len(grain_masks[direction].shape) == 3, "Grain masks should be 3D tensors"

                # Extract the dna_class_mask and create molecule map
                dna_class_mask = grain_masks[direction][:, :, grain_masks[direction].shape[2]-1]  # Assuming UNET class 3 is relevant here
                unique_molecules = np.unique(dna_class_mask)[1:]  # Skip background (0)
                molecule_map = {molecule: idx + 1 for idx, molecule in enumerate(unique_molecules)}

                tracing_stats_dfs = []

                # Iterate over the UNET classes and process each
                for unet_class in range(1, grain_masks[direction].shape[2]):
                    class_mask = grain_masks[direction][:, :, unet_class]
                    overlap_mask = np.where(class_mask, dna_class_mask, 0)

                    # Initialize empty arrays to accumulate traces for the class
                    combined_spline_trace = np.zeros_like(image)

                    # Iterate over molecules
                    for molecule, molecule_number in molecule_map.items():
                        # Relabeling based on overlap with the molecule
                        relabeled_mask = np.where(overlap_mask == molecule, molecule_number, 0)

                        # Use the label function to identify distinct connected components (grains)
                        labeled_array, num_features = label(relabeled_mask)

                        for region_idx in range(1, num_features + 1):
                            # Create a mask for each connected component (region)
                            component_mask = np.where(labeled_array == region_idx, molecule_number, 0)

                            if np.max(component_mask) == 0:
                                LOGGER.warning(f"[{filename}] : Skipping empty region {region_idx} for molecule {molecule_number}.")
                                continue

                            LOGGER.info(f"[{filename}] : Processing region {region_idx} for molecule {molecule_number}")

                            # Trace the DNA in this component (grain)
                            tracing_results = trace_image(
                                image=image,
                                grains_mask=component_mask,
                                filename=filename,
                                pixel_to_nm_scaling=pixel_to_nm_scaling,
                                **dnatracing_config,
                            )

                            # Assign object numbers and other metadata
                            tracing_results["statistics"]["threshold"] = direction
                            tracing_results["statistics"]["class"] = unet_class
                            tracing_results["statistics"]["object_number"] = region_idx
                            tracing_results["statistics"]["molecule_number"] = molecule_number
                            tracing_results["statistics"]["image"] = filename  # Ensure image name is stored

                            # Accumulate traces for the class
                            combined_spline_trace += tracing_results["image_spline_trace"]

                            ordered_traces = tracing_results["all_ordered_traces"]
                            cropped_images: dict[int, np.ndarray] = tracing_results["all_cropped_images"]

                            grain_trace_data[direction] = {
                                "ordered_traces": ordered_traces,
                                "cropped_images": cropped_images,
                                "ordered_trace_heights": tracing_results["all_ordered_trace_heights"],
                                "ordered_trace_cumulative_distances": tracing_results["all_ordered_trace_cumulative_distances"],
                                "splined_traces": tracing_results["all_splined_traces"],
                            }

                            # Append the statistics for this region
                            tracing_stats_dfs.append(tracing_results["statistics"])

                    # Once all regions in the class have been processed, generate a single image per class
                    class_trace_filename = f"{filename}_class_{unet_class}_combined_traces"

                    # Plot combined trace for the whole class
                    Images(
                        image,
                        output_dir=core_out_path,
                        filename=class_trace_filename,
                        masked_array=combined_spline_trace,
                        **plotting_config["plot_dict"]["all_molecule_traces"],
                    ).plot_and_save()

                    # Plot traces for each unique molecule-class combo
                    if plotting_config["image_set"] == "all":
                        combined_trace_mask = np.zeros_like(cropped_images[0])  # Initialize a mask for the combined traces

                        for grain_index, grain_trace in ordered_traces.items():
                            cropped_image = cropped_images[grain_index]
                            
                            # Check if the grain_trace is valid
                            if grain_trace is not None:
                                for coordinate in grain_trace:
                                    combined_trace_mask[coordinate[0], coordinate[1]] = 1  # Aggregate traces into the combined mask

                        # Use the naming convention for the combined image
                        plot_filename = f"{filename}_trace_molecule{molecule_number}_class{unet_class}"

                        Images(
                            data=cropped_image,
                            output_dir=grain_out_path / direction,
                            filename=plot_filename,
                            masked_array=combined_trace_mask,
                            **plotting_config["plot_dict"]["single_molecule_trace"],
                        ).plot_and_save()

                # Concatenate the tracing stats DataFrames for the current direction
                if len(tracing_stats_dfs) > 0:
                    tracing_stats[direction] = pd.concat(tracing_stats_dfs)

            # Combine tracing stats for 'above' and 'below' thresholds
            if "above" in tracing_stats and "below" in tracing_stats:
                tracing_stats_df = pd.concat([tracing_stats["below"], tracing_stats["above"]])
            elif "above" in tracing_stats:
                tracing_stats_df = tracing_stats["above"]
            elif "below" in tracing_stats:
                tracing_stats_df = tracing_stats["below"]
            else:
                tracing_stats_df = create_empty_dataframe()

            # Reset index and merge with the original results dataframe
            tracing_stats_df = tracing_stats_df.reset_index(drop=True)
            LOGGER.info(f"[{filename}] : Combining {list(tracing_stats.keys())} grain statistics and DNA tracing statistics")
            results = results_df.merge(tracing_stats_df, on=["image", "threshold", "class", "object_number", "molecule_number"], how="left")
            results["basename"] = image_path.parent

            return results, grain_trace_data

        # If DNA tracing is disabled, return the original DataFrame
        LOGGER.info(f"[{filename}] : DNA Tracing is disabled. Returning grainstats data frame.")
        results = results_df
        results["basename"] = image_path.parent

        return results, grain_trace_data

    except Exception as e:
        # Handle errors and return the original DataFrame if something goes wrong
        LOGGER.warning(f"[{filename}] : Errors occurred while calculating DNA tracing statistics: {e}. Returning grain statistics.")
        results = results_df
        results["basename"] = image_path.parent
        grain_trace_data = None
        return results, grain_trace_data




def get_out_paths(image_path: Path, base_dir: Path, output_dir: Path, filename: str, plotting_config: dict):
    """
    Determine components of output paths for a given image and plotting config.

    Parameters
    ----------
    image_path : Path
        Path of the image being processed.
    base_dir : Path
        Path of the data folder.
    output_dir : Path
        Base output directory for output data.
    filename : str
        Name of the image being processed.
    plotting_config : dict
        Dictionary of configuration for plotting images.

    Returns
    -------
    tuple
        Core output path for general file outputs, filter output path for flattening related files and
        grain output path for grain finding related files.
    """
    LOGGER.info(f"Processing : {filename}")
    core_out_path = get_out_path(image_path, base_dir, output_dir).parent / "processed"
    core_out_path.mkdir(parents=True, exist_ok=True)
    filter_out_path = core_out_path / filename / "filters"
    grain_out_path = core_out_path / filename / "grains"
    if plotting_config["image_set"] == "all":
        filter_out_path.mkdir(exist_ok=True, parents=True)
        Path.mkdir(grain_out_path / "above", parents=True, exist_ok=True)
        Path.mkdir(grain_out_path / "below", parents=True, exist_ok=True)

    return core_out_path, filter_out_path, grain_out_path


def process_scan(
    topostats_object: dict,
    base_dir: str | Path,
    filter_config: dict,
    grains_config: dict,
    grainstats_config: dict,
    dnatracing_config: dict,
    plotting_config: dict,
    output_dir: str | Path = "output",
) -> tuple[dict, pd.DataFrame, dict]:
    """
    Process a single image, filtering, finding grains and calculating their statistics.

    Parameters
    ----------
    topostats_object : dict[str, Union[np.ndarray, Path, float]]
        A dictionary with keys 'image', 'img_path' and 'px_2_nm' containing a file or frames' image, it's path and it's
        pixel to namometre scaling value.
    base_dir : Union[str, Path]
        Directory to recursively search for files, if not specified the current directory is scanned.
    filter_config : dict
        Dictionary of configuration options for running the Filter stage.
    grains_config : dict
        Dictionary of configuration options for running the Grain detection stage.
    grainstats_config : dict
        Dictionary of configuration options for running the Grain Statistics stage.
    dnatracing_config : dict
        Dictionary of configuration options for running the DNA Tracing stage.
    plotting_config : dict
        Dictionary of configuration options for plotting figures.
    output_dir : Union[str, Path]
        Directory to save output to, it will be created if it does not exist. If it already exists then it is possible
        that output will be over-written.

    Returns
    -------
    tuple[dict, pd.DataFrame, dict]
        TopoStats dictionary object, DataFrame containing grain statistics and dna tracing statistics,
        and dictionary containing general image statistics.
    """
    core_out_path, filter_out_path, grain_out_path = get_out_paths(
        image_path=topostats_object["img_path"],
        base_dir=base_dir,
        output_dir=output_dir,
        filename=topostats_object["filename"],
        plotting_config=plotting_config,
    )

    plotting_config = add_pixel_to_nm_to_plotting_config(plotting_config, topostats_object["pixel_to_nm_scaling"])

    # Flatten Image
    image_flattened = run_filters(
        unprocessed_image=topostats_object["image_original"],
        pixel_to_nm_scaling=topostats_object["pixel_to_nm_scaling"],
        filename=topostats_object["filename"],
        filter_out_path=filter_out_path,
        core_out_path=core_out_path,
        filter_config=filter_config,
        plotting_config=plotting_config,
    )
    # Use flattened image if one is returned, else use original image
    topostats_object["image_flattened"] = (
        image_flattened if image_flattened is not None else topostats_object["image_original"]
    )

    # Find Grains :
    grain_masks = run_grains(
        image=topostats_object["image_flattened"],
        pixel_to_nm_scaling=topostats_object["pixel_to_nm_scaling"],
        filename=topostats_object["filename"],
        grain_out_path=grain_out_path,
        core_out_path=core_out_path,
        plotting_config=plotting_config,
        grains_config=grains_config,
    )

    # Update grain masks if new grain masks are returned. Else keep old grain masks. Topostats object's "grain_masks"
    # defaults to an empty dictionary so this is safe.
    topostats_object["grain_masks"] = grain_masks if grain_masks is not None else topostats_object["grain_masks"]

    if "above" in topostats_object["grain_masks"].keys() or "below" in topostats_object["grain_masks"].keys():
        # Grainstats :
        results_df, height_profiles = run_grainstats(
            image=topostats_object["image_flattened"],
            pixel_to_nm_scaling=topostats_object["pixel_to_nm_scaling"],
            grain_masks=topostats_object["grain_masks"],
            filename=topostats_object["filename"],
            grainstats_config=grainstats_config,
            plotting_config=plotting_config,
            grain_out_path=grain_out_path,
        )

        # DNAtracing
        results_df, grain_trace_data = run_dnatracing(
            image=topostats_object["image_flattened"],
            pixel_to_nm_scaling=topostats_object["pixel_to_nm_scaling"],
            grain_masks=topostats_object["grain_masks"],
            filename=topostats_object["filename"],
            core_out_path=core_out_path,
            grain_out_path=grain_out_path,
            image_path=topostats_object["img_path"],
            plotting_config=plotting_config,
            dnatracing_config=dnatracing_config,
            results_df=results_df,
        )

        """
        # DNAprotein analysis
        run_dna_protein_analysis(
            image=topostats_object["image_flattened"],
            pixel_to_nm_scaling=topostats_object["pixel_to_nm_scaling"],
            grain_masks=topostats_object["grain_masks"],
            filename=topostats_object["filename"],)
        """
        

        # Add grain trace data and height profiles to topostats object
        topostats_object["grain_trace_data"] = grain_trace_data
        topostats_object["height_profiles"] = height_profiles

    else:
        results_df = create_empty_dataframe()
        height_profiles = {}

    # Get image statistics
    LOGGER.info(f"[{topostats_object['filename']}] : *** Image Statistics ***")
    # Provide the raw image if image has not been flattened, else provide the flattened image.
    if topostats_object["image_flattened"] is not None:
        image_for_image_stats = topostats_object["image_flattened"]
    else:
        image_for_image_stats = topostats_object["image_original"]
    # Calculate image statistics - returns a dictionary
    image_stats = image_statistics(
        image=image_for_image_stats,
        filename=topostats_object["filename"],
        results_df=results_df,
        pixel_to_nm_scaling=topostats_object["pixel_to_nm_scaling"],
    )

    # Save the topostats dictionary object to .topostats file.
    save_topostats_file(
        output_dir=core_out_path, filename=str(topostats_object["filename"]), topostats_object=topostats_object
    )

    return topostats_object["img_path"], results_df, image_stats, height_profiles


def check_run_steps(filter_run: bool, grains_run: bool, grainstats_run: bool, dnatracing_run: bool) -> None:
    """
    Check options for running steps (Filter, Grain, Grainstats and DNA tracing) are logically consistent.

    This checks that earlier steps required are enabled.

    Parameters
    ----------
    filter_run : bool
        Flag for running Filtering.
    grains_run : bool
        Flag for running Grains.
    grainstats_run : bool
        Flag for running GrainStats.
    dnatracing_run : bool
        Flag for running DNA Tracing.
    """
    if dnatracing_run:
        if grainstats_run is False:
            LOGGER.error("DNA tracing enabled but Grainstats disabled. Please check your configuration file.")
        elif grains_run is False:
            LOGGER.error("DNA tracing enabled but Grains disabled. Please check your configuration file.")
        elif filter_run is False:
            LOGGER.error("DNA tracing enabled but Filters disabled. Please check your configuration file.")
        else:
            LOGGER.info("Configuration run options are consistent, processing can proceed.")
    elif grainstats_run:
        if grains_run is False:
            LOGGER.error("Grainstats enabled but Grains disabled. Please check your configuration file.")
        elif filter_run is False:
            LOGGER.error("Grainstats enabled but Filters disabled. Please check your configuration file.")
        else:
            LOGGER.info("Configuration run options are consistent, processing can proceed.")
    elif grains_run:
        if filter_run is False:
            LOGGER.error("Grains enabled but Filters disabled. Please check your configuration file.")
        else:
            LOGGER.info("Configuration run options are consistent, processing can proceed.")
    else:
        LOGGER.info("Configuration run options are consistent, processing can proceed.")


def completion_message(config: dict, img_files: list, summary_config: dict, images_processed: int) -> None:
    """
    Print a completion message summarising images processed.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    img_files : list()
        List of found image paths.
    summary_config : dict(
        Configuration for plotting summary statistics.
    images_processed : int
        Pandas DataFrame of results.
    """
    if summary_config is not None:
        distribution_plots_message = str(summary_config["output_dir"])
    else:
        distribution_plots_message = "Disabled. Enable in config 'summary_stats/run' if needed."
    LOGGER.info(
        f"\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ COMPLETE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n"
        f"  TopoStats Version           : {__version__}\n"
        f"  Base Directory              : {config['base_dir']}\n"
        f"  File Extension              : {config['file_ext']}\n"
        f"  Files Found                 : {len(img_files)}\n"
        f"  Successfully Processed^1    : {images_processed} ({(images_processed * 100) / len(img_files)}%)\n"
        f"  All statistics              : {str(config['output_dir'])}/all_statistics.csv\n"
        f"  Distribution Plots          : {distribution_plots_message}\n\n"
        f"  Configuration               : {config['output_dir']}/config.yaml\n\n"
        f"  Email                       : topostats@sheffield.ac.uk\n"
        f"  Documentation               : https://afm-spm.github.io/topostats/\n"
        f"  Source Code                 : https://github.com/AFM-SPM/TopoStats/\n"
        f"  Bug Reports/Feature Request : https://github.com/AFM-SPM/TopoStats/issues/new/choose\n"
        f"  Citation File Format        : https://github.com/AFM-SPM/TopoStats/blob/main/CITATION.cff\n\n"
        f"  ^1 Successful processing of an image is detection of grains and calculation of at least\n"
        f"     grain statistics. If these have been disabled the percentage will be 0.\n\n"
        f"  If you encounter bugs/issues or have feature requests please report them at the above URL\n"
        f"  or email us.\n\n"
        f"  If you have found TopoStats useful please consider citing it. A Citation File Format is\n"
        f"  linked above and available from the Source Code page.\n"
        f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n"
    )
