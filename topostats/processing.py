"""Functions for procesing data."""
from collections import defaultdict
from pathlib import Path
from typing import Dict, Union, List

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation

from topostats import __version__
from topostats.filters import Filters
from topostats.grains import Grains
from topostats.grainstats import GrainStats
from topostats.io import get_out_path, save_array
from topostats.logs.logs import setup_logger, LOGGER_NAME
from topostats.plottingfuncs import Images
from topostats.plotting import plot_crossing_linetrace_halfmax
from topostats.tracing.dnatracing import trace_image, dnaTrace
from topostats.utils import create_empty_dataframe, NpEncoder

# pylint: disable=broad-except
# pylint: disable=line-too-long
# pylint: disable=too-many-arguments
# pylint: disable=too-many-branches
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=too-many-nested-blocks
# pylint: disable=unnecessary-dict-index-lookup

LOGGER = setup_logger(LOGGER_NAME)


def process_scan(
    img_path_px2nm: Dict[str, Union[np.ndarray, Path, float]],
    base_dir: Union[str, Path],
    filter_config: dict,
    grains_config: dict,
    grainstats_config: dict,
    dnatracing_config: dict,
    plotting_config: dict,
    output_dir: Union[str, Path] = "output",
) -> None:
    """Process a single image, filtering, finding grains and calculating their statistics.

    Parameters
    ----------
    img_path_px2nm : Dict[str, Union[np.ndarray, Path, float]]
        A dictionary with keys 'image', 'img_path' and 'px_2_nm' containing a file or frames' image, it's path and it's pixel to namometre scaling value.
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


    Results
    -------
    None

    Results are written to CSV and images produced in configuration options request them.
    """

    image = img_path_px2nm["image"]
    image_path = img_path_px2nm["img_path"]
    pixel_to_nm_scaling = img_path_px2nm["px_2_nm"]
    filename = image_path.name

    LOGGER.info(f"Processing : {filename}")
    core_out_path = get_out_path(image_path, base_dir, output_dir).parent / "processed"
    core_out_path.mkdir(parents=True, exist_ok=True)
    filter_out_path = core_out_path / filename / "filters"
    grain_out_path = core_out_path / filename / "grains"
    dna_tracing_out_path = core_out_path / filename / "dnatracing"
    if plotting_config["image_set"] == "all":
        filter_out_path.mkdir(exist_ok=True, parents=True)
        Path.mkdir(grain_out_path / "above", parents=True, exist_ok=True)
        Path.mkdir(grain_out_path / "below", parents=True, exist_ok=True)
        Path.mkdir(dna_tracing_out_path / "above" / "nodes", parents=True, exist_ok=True)
        Path.mkdir(dna_tracing_out_path / "below" / "nodes", parents=True, exist_ok=True)

    # Filter Image
    if filter_config["run"]:
        filter_config.pop("run")
        LOGGER.info(f"[{filename}] : Image dimensions: {image.shape}")
        LOGGER.info(f"[{filename}] : *** Filtering ***")
        filtered_image = Filters(
            image,
            filename,
            pixel_to_nm_scaling,
            **filter_config,
        )
        filtered_image.filter_image()

        # Optionally plot filter stage
        if plotting_config["run"]:
            plotting_config.pop("run")
            LOGGER.info(f"[{filename}] : Plotting Filtering Images")
            # Update PLOT_DICT with pixel_to_nm_scaling (can't add _output_dir since it changes)
            plot_opts = {"pixel_to_nm_scaling": pixel_to_nm_scaling}
            for image, options in plotting_config["plot_dict"].items():
                plotting_config["plot_dict"][image] = {**options, **plot_opts}
            # Generate plots
            for plot_name, array in filtered_image.images.items():
                if plot_name not in ["scan_raw"]:
                    if plot_name == "extracted_channel":
                        array = np.flipud(array.pixels)
                    plotting_config["plot_dict"][plot_name]["output_dir"] = filter_out_path
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
            filtered_image.images["gaussian_filtered"],
            filename=filename,
            **plotting_config["plot_dict"][plot_name],
        ).plot_and_save()
        # Save the z_threshed image (aka "Height_Thresholded") numpy array
        save_array(
            array=filtered_image.images["gaussian_filtered"],
            outpath=core_out_path,
            filename=filename,
            array_type="height_thresholded",
        )

    else:
        LOGGER.error(
            "You have not included running the initial filter stage. This is required for all subsequent "
            "stages of processing. Please check your configuration file."
        )

    # Find Grains :
    if grains_config["run"]:
        grains_config.pop("run")
        try:
            LOGGER.info(f"[{filename}] : *** Grain Finding ***")
            grains = Grains(
                image=filtered_image.images["gaussian_filtered"],
                filename=filename,
                pixel_to_nm_scaling=pixel_to_nm_scaling,
                **grains_config,
            )
            grains.find_grains()
            for direction, _ in grains.directions.items():
                LOGGER.info(
                    f"[{filename}] : Grains found for direction {direction} : {len(grains.region_properties[direction])}"
                )
                if len(grains.region_properties[direction]) == 0:
                    LOGGER.warning(f"[{filename}] : No grains found for direction {direction}")
        except Exception as e:
            LOGGER.error(f"[{filename}] : An error occured during grain finding, skipping grainstats and dnatracing.")
            LOGGER.error(f"[{filename}] : The error: {e}")
            results = create_empty_dataframe()
        else:
            for direction, region_props in grains.region_properties.items():
                if len(region_props) == 0:
                    LOGGER.warning(f"[{filename}] : No grains found for the {direction} direction.")
            # Optionally plot grain finding stage if we have found grains and plotting is required
            if plotting_config["run"]:
                plotting_config.pop("run")
                LOGGER.info(f"[{filename}] : Plotting Grain Finding Images")
                for direction, image_arrays in grains.directions.items():
                    LOGGER.info(f"[{filename}] : Plotting {direction} Grain Finding Images")
                    for plot_name, array in image_arrays.items():
                        LOGGER.info(f"[{filename}] : Plotting {plot_name} image")
                        plotting_config["plot_dict"][plot_name]["output_dir"] = grain_out_path / f"{direction}"
                        Images(array, **plotting_config["plot_dict"][plot_name]).plot_and_save()
                    # Make a plot of coloured regions with bounding boxes
                    plotting_config["plot_dict"]["bounding_boxes"]["output_dir"] = grain_out_path / f"{direction}"
                    Images(
                        grains.directions[direction]["coloured_regions"],
                        **plotting_config["plot_dict"]["bounding_boxes"],
                        region_properties=grains.region_properties[direction],
                    ).plot_and_save()
                    plotting_config["plot_dict"]["coloured_boxes"]["output_dir"] = grain_out_path / f"{direction}"
                    Images(
                        grains.directions[direction]["labelled_regions_02"],
                        **plotting_config["plot_dict"]["coloured_boxes"],
                        region_properties=grains.region_properties[direction],
                    ).plot_and_save()
                    # Always want mask_overlay (aka "Height Thresholded with Mask") but in core_out_path
                    plot_name = "mask_overlay"
                    plotting_config["plot_dict"][plot_name]["output_dir"] = core_out_path
                    Images(
                        filtered_image.images["gaussian_filtered"],
                        filename=f"{filename}_{direction}_masked",
                        masked_array=grains.directions[direction]["removed_small_objects"],
                        **plotting_config["plot_dict"][plot_name],
                    ).plot_and_save()

                plotting_config["run"] = True
            else:
                LOGGER.info(f"[{filename}] : Plotting disabled for Grain Finding Images")

            # Grainstats :
            # Calculate statistics if required
            if grainstats_config["run"]:
                grainstats_config.pop("run")
                # Grain Statistics :
                try:
                    LOGGER.info(f"[{filename}] : *** Grain Statistics ***")
                    grain_plot_dict = {
                        key: value
                        for key, value in plotting_config["plot_dict"].items()
                        if key in ["grain_image", "grain_mask", "grain_mask_image"]
                    }
                    grainstats = {}
                    # There are two layers to process those above the given threshold and those below
                    for direction, _ in grains.directions.items():
                        if len(grains.region_properties[direction]) == 0:
                            LOGGER.warning(
                                f"[{filename}] : No grains exist for the {direction} direction. Skipping grainstats and DNAtracing."
                            )
                            grainstats[direction] = create_empty_dataframe()
                        else:
                            grainstats[direction], grains_plot_data = GrainStats(
                                data=filtered_image.images["gaussian_filtered"],
                                labelled_data=grains.directions[direction]["labelled_regions_02"],
                                pixel_to_nanometre_scaling=pixel_to_nm_scaling,
                                direction=direction,
                                base_output_dir=grain_out_path,
                                image_name=filename,
                                plot_opts=grain_plot_dict,
                                **grainstats_config,
                            ).calculate_stats()
                            grainstats[direction]["threshold"] = direction
                            # Plot grains
                            if plotting_config["image_set"] == "all":
                                LOGGER.info(f"[{filename}] : Plotting grain images for direction: {direction}.")
                                for plot_data in grains_plot_data:
                                    LOGGER.info(
                                        f"[{filename}] : Plotting grain image {plot_data['filename']} for direction: {direction}."
                                    )
                                    Images(
                                        data=plot_data["data"],
                                        output_dir=plot_data["output_dir"],
                                        filename=plot_data["filename"],
                                        **plotting_config["plot_dict"][plot_data["name"]],
                                    ).plot_and_save()
                    # Set grainstats_df in light of direction
                    if grains_config["direction"] == "both":
                        grainstats_df = pd.concat([grainstats["below"], grainstats["above"]])
                    elif grains_config["direction"] == "above":
                        grainstats_df = grainstats["above"]
                    elif grains_config["direction"] == "below":
                        grainstats_df = grainstats["below"]
                except Exception:
                    LOGGER.info(
                        f"[{filename}] : Errors occurred whilst calculating grain statistics. Skipping DNAtracing."
                    )
                    results = create_empty_dataframe()
                else:
                    # Run dnatracing
                    #try:
                    if dnatracing_config["run"]:
                        dnatracing_config.pop("run")
                        LOGGER.info(f"[{filename}] : *** DNA Tracing ***")
                        tracing_stats = defaultdict()
                        node_stats = defaultdict()
                        images = defaultdict()
                        for direction, _ in grainstats.items():
                            tracing_stats[direction], node_stats[direction], images[direction] = trace_image(
                                image=filtered_image.images["gaussian_filtered"],
                                grains_mask=grains.directions[direction]["labelled_regions_02"],
                                filename=filename,
                                pixel_to_nm_scaling=pixel_to_nm_scaling,
                                **dnatracing_config,
                            )
                            tracing_stats[direction]["threshold"] = direction
                        # Set tracing_stats_df in light of direction
                        if grains_config["direction"] == "both":
                            tracing_stats_df = pd.concat([tracing_stats["below"], tracing_stats["above"]])
                        elif grains_config["direction"] == "above":
                            tracing_stats_df = tracing_stats["above"]
                        elif grains_config["direction"] == "below":
                            tracing_stats_df = tracing_stats["below"]
                        LOGGER.info(
                            f"[{filename}] : Combining {direction} grain statistics and dnatracing statistics"
                        )
                        # NB - Merge on image, molecule and threshold because we may have above and below molecules which
                        #      gives duplicate molecule numbers as they are processed separately, if tracing stats
                        #      are not available (because skeleton was too small), grainstats are still retained.
                        results = grainstats_df.merge(
                            tracing_stats_df, on=["image", "threshold", "molecule_number"], how="left"
                        )
                        results["basename"] = image_path.parent
                        
                        # Plot dnatracing images
                        LOGGER.info(f"[{filename}] : Plotting DNA Tracing Images")
                        output_dir = Path(dna_tracing_out_path / f"{direction}")

                        plot_names = ["orig_grains", "smoothed_grains", "orig_skeletons", "pruned_skeletons", "nodes"]#, "fitted_trace", "ordered_trace", "splined_trace"]
                        data2s = [
                            images[direction]["grain"],
                            images[direction]["smoothed_grain"],
                            images[direction]["skeleton"],
                            images[direction]["prunted_skeleton"],
                            images[direction]["node_img"]
                            #images[direction]["fitted_trace"],
                            #images[direction]["ordered_trace"],
                            #images[direction]["splined_trace"],
                        ]
                        for i, plot_name in enumerate(plot_names):
                            plotting_config["plot_dict"][plot_name]["output_dir"] = output_dir
                            plotting_config["plot_dict"][plot_name]["mask_cmap"] = "green_black"
                            print(f"{plot_name} image size: {data2s[i].shape}")
                            print(np.unique(data2s[i], return_counts=True))
                            print("Image size: ", images[direction]["image"].shape)
                            Images(
                                images[direction]["image"],
                                masked_array=data2s[i],
                                **plotting_config["plot_dict"][plot_name],
                            ).save_figure_black(
                                background=images[direction]["grain"],
                            )
                        #np.savetxt(f"{core_out_path}_{filename}_skel.txt", dna_traces[direction].skeletons)
                        #np.savetxt(f"{core_out_path}_{filename}_connect.txt", nodes.all_connected_nodes)

                        # plot nodes and line traces
                        for mol_no, mol_stats in node_stats[direction].items():
                            for node_no, single_node_stats in mol_stats.items():
                                plotting_config["plot_dict"]["zoom_node"]["mask_cmap"] = "green_black"
                                plotting_config["plot_dict"]["crossings"]["mask_cmap"] = "green_green"
                                plotting_config["plot_dict"]["tripple_crossings"]["mask_cmap"] = "green_green"
                                # plot node + skeleton
                                Images(
                                    single_node_stats["node_stats"]["node_area_image"],
                                    masked_array=single_node_stats["node_stats"]["node_area_skeleton"],
                                    filename=f"mol_{mol_no}_node_{node_no}_node_area",
                                    output_dir=output_dir / "nodes",
                                    **plotting_config["plot_dict"]["zoom_node"],
                                ).save_figure_black(background=single_node_stats["node_stats"]["node_area_grain"])
                                # plot branch mask
                                Images(
                                    single_node_stats["node_stats"]["node_area_image"],
                                    masked_array=single_node_stats["node_stats"]["node_branch_mask"],
                                    filename=f"mol_{mol_no}_node_{node_no}_crossings",
                                    output_dir=output_dir / "nodes",
                                    **plotting_config["plot_dict"]["crossings"],
                                ).save_figure_black(background=single_node_stats["node_stats"]["node_area_grain"])
                                # plot avg branch mask
                                if single_node_stats["node_stats"]["node_avg_mask"] is not None:
                                    Images(
                                        single_node_stats["node_stats"]["node_area_image"],
                                        masked_array=single_node_stats["node_stats"]["node_avg_mask"],
                                        filename=f"mol_{mol_no}_node_{node_no}_average_crossings",
                                        output_dir=output_dir / "nodes",
                                        **plotting_config["plot_dict"]["tripple_crossings"],
                                    ).save_figure_black(background=single_node_stats["node_stats"]["node_area_grain"])
                                # Plot crossing height linetrace
                                if not single_node_stats["error"]:
                                    plotting_config["plot_dict"]["line_trace"] = {
                                        "title": "Heights of Crossing",
                                        "cmap": "blu_purp",
                                    }
                                    fig, _ = plot_crossing_linetrace_halfmax(
                                        single_node_stats["branch_stats"],
                                        **plotting_config["plot_dict"]["line_trace"],
                                    )
                                    fig.savefig(
                                        output_dir / "nodes" / f"mol_{mol_no}_node_{node_no}_linetrace_halfmax.svg",
                                        format="svg",
                                    )
                            """
                            # plot the molecules on their own
                            print("LEN: ",len(nodes.mol_coords))
                            if len(nodes.mol_coords[mol_no]) > 1:
                                for inner_mol_no, coords in enumerate(nodes.mol_coords[mol_no]):
                                    single_mol = np.zeros_like(dna_traces[direction].full_image_data)
                                    single_mol[coords[:, 0], coords[:, 1]] = 1
                                    single_mol = binary_dilation(single_mol)
                                    Images(
                                        dna_traces[direction].full_image_data,
                                        masked_array=single_mol,
                                        output_dir=output_dir,
                                        filename=f"Grain_{mol_no}_separated_mol_{inner_mol_no}",
                                        zrange=[0, 3.5e-9],
                                        **plotting_config["plot_dict"]["single_mol"],
                                    ).save_figure_black(background=grains.directions[direction]["removed_small_objects"])
                        
                        # plot the visual image for the whole image
                        visual = nodes.all_visuals_img
                        if visual is not None:
                            Images(
                                dna_traces[direction].full_image_data,
                                masked_array=visual,
                                output_dir=output_dir,
                                zrange=[0, 3.5e-9],
                                **plotting_config["plot_dict"]["visual"],
                            ).save_figure_black(background=grains.directions[direction]["removed_small_objects"])
                        """
                        # ------- branch vector img -------
                        """
                        vectors = nodes.test2
                        plot_name = "test"
                        plotting_config["plot_dict"][plot_name]["output_dir"] = output_dir
                        fig, ax = Images(
                            data=node_stats[direction][1][1]["node_stats"]["node_area_image"],
                            masked_array=nodes.test,
                            mask_cmap="viridis",
                            filename="branch_vectors.tiff",
                            **plotting_config["plot_dict"][plot_name],
                        ).save_figure_black(background=node_stats[direction][1][1]["node_stats"]["node_area_grain"])

                        col = ["m", "b", "g", "y"]
                        for i, vector in enumerate(np.asarray(vectors)):  # [:,::-1]):
                            ax.arrow(10.25, 10.5, vector[1] * 4, vector[0] * -4, width=0.3, color=col[i])
                        fig.savefig("cats2/vector_img.tiff")

                        # ------- branch vector + angles fig -------
                        vectors = nodes.test4
                        angles = nodes.test5
                        plot_name = "test"
                        plotting_config["plot_dict"][plot_name]["output_dir"] = output_dir
                        fig, ax = Images(
                            data=node_stats[direction][1][1]["node_stats"]["node_area_image"],
                            masked_array=node_stats[direction][1][1]["node_stats"]["node_branch_mask"],
                            mask_cmap="blu_purp",
                            filename="test",
                            **plotting_config["plot_dict"][plot_name],
                        ).save_figure_black(background=node_stats[direction][1][1]["node_stats"]["node_area_grain"])

                        col = ["b", "m"]
                        ax.arrow(10.25 - 4, 10.75, vectors[0][1] * 8, vectors[0][0] * -8, width=0.3, color=col[0])
                        ax.arrow(10.25, 10.75 + 4, vectors[1][1] * 8, vectors[1][0] * -8, width=0.3, color=col[1])

                        arc = Arc((10.05, 10.50), 7.2, 7.2, -2, 0, angles[1], lw=10, color="white")
                        ax.add_patch(arc)
                        ax.text(
                            12.5, 14, "%0.2f" % float(angles[1]) + "\u00b0", fontsize=40, weight="bold", color="white"
                        )  #'xx-large
                        fig.savefig("cats2/vector_angle_img.tiff")
                        """
                        
                    else:
                        LOGGER.info(
                            f"[{filename}] Calculation of DNA Tracing disabled, returning grainstats data frame."
                        )
                        results = grainstats_df
                        results["basename"] = image_path.parent
                    """
                    except Exception:
                        # If no results we need a dummy dataframe to return.
                        LOGGER.warning(
                            f"[{filename}] : Errors occurred whilst calculating DNA tracing statistics, "
                            "returning grain statistics"
                        )
                        results = grainstats_df
                        results["basename"] = image_path.parent
                    """
            else:
                LOGGER.info(f"[{filename}] Calculation of grainstats disabled, returning empty data frame.")
                results = create_empty_dataframe()
    else:
        LOGGER.info(f"[{filename}] Detection of grains disabled, returning empty data frame.")
        results = create_empty_dataframe()
        results["Image Name"] = filename
        results["Basename"] = image_path.parent
        node_stats = {"upper": None, "lower": None}
    return image_path, results, node_stats


def check_run_steps(filter_run: bool, grains_run: bool, grainstats_run: bool, dnatracing_run: bool) -> None:
    """Check options for running steps (Filter, Grain, Grainstats and DNA tracing) are logically consistent.

    This checks that earlier steps required are enabled.

    Parameters
    ----------
    filter_run: bool
        Flag for running Filtering.
    grains_run: bool
        Flag for running Grains.
    grainstats_run: bool
        Flag for running GrainStats.
    dnatracing_run: bool
        Flag for running DNA Tracing.

    Returns
    -------
    None
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


def completion_message(config: Dict, img_files: List, summary_config: Dict, images_processed: int) -> None:
    """Print a completion message summarising images processed.

    Parameters
    ----------
    config: dict
        Configuration dictionary.
    img_files: list()
        List of found image paths.
    summary_config: dict(
        Configuration for plotting summary statistics.
    images_processed: int
        Pandas DataFrame of results.

    Results
    -------
    None
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
        f"  Configuration               : {config['output_dir']}/config.yaml\n"
        f"  All statistics              : {str(config['output_dir'])}/all_statistics.csv\n"
        f"  Distribution Plots          : {distribution_plots_message}\n\n"
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
