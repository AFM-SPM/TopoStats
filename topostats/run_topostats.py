"""Run TopoStats

This provides an entry point for running TopoStats as a command line programme.
"""
import argparse as arg
from collections import defaultdict
from functools import partial
import importlib.resources as pkg_resources
import json
from multiprocessing import Pool
from pathlib import Path
import sys
from typing import Union, Dict
import yaml

import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.ndimage import binary_dilation
from matplotlib.patches import Arc

from topostats.filters import Filters
from topostats.grains import Grains
from topostats.grainstats import GrainStats
from topostats.io import find_images, read_yaml, write_yaml, get_out_path, LoadScans
from topostats.logs.logs import setup_logger, LOGGER_NAME
from topostats.plottingfuncs import Images
from topostats.plotting import plot_crossing_linetrace_gauss, plot_crossing_linetrace_halfmax
from topostats.tracing.dnatracing import dnaTrace, traceStats, nodeStats
from topostats.utils import (
    update_config,
    create_empty_dataframe,
    folder_grainstats,
)
from topostats.validation import validate_config, DEFAULT_CONFIG_SCHEMA, PLOTTING_SCHEMA

LOGGER = setup_logger(LOGGER_NAME)


# pylint: disable=broad-except
# pylint: disable=line-too-long
# pylint: disable=too-many-arguments
# pylint: disable=too-many-branches
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=unnecessary-dict-index-lookup


def create_parser() -> arg.ArgumentParser:
    """Create a parser for reading options."""
    parser = arg.ArgumentParser(
        description="Process AFM images. Additional arguments over-ride those in the configuration file."
    )
    parser.add_argument(
        "-c",
        "--config_file",
        dest="config_file",
        required=False,
        help="Path to a YAML configuration file.",
    )
    parser.add_argument(
        "--create-config-file",
        dest="create_config_file",
        type=str,
        required=False,
        help="Filename to write a sample YAML configuration file to (should end in '.yaml').",
    )
    parser.add_argument(
        "-b",
        "--base_dir",
        dest="base_dir",
        type=str,
        required=False,
        help="Base directory to scan for images.",
    )
    parser.add_argument(
        "-j",
        "--cores",
        dest="cores",
        type=int,
        required=False,
        help="Number of CPU cores to use when processing.",
    )
    parser.add_argument(
        "-f",
        "--file_ext",
        dest="file_ext",
        type=str,
        required=False,
        help="File extension to scan for.",
    )
    parser.add_argument(
        "--channel",
        dest="channel",
        type=str,
        required=False,
        help="Channel to extract.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        dest="output_dir",
        type=str,
        required=False,
        help="Output directory to write results to.",
    )
    parser.add_argument(
        "-s",
        "--save_plots",
        dest="save_plots",
        type=bool,
        required=False,
        help="Whether to save plots.",
    )
    parser.add_argument(
        "-t", "--threshold_method", dest="threshold_method", required=False, help="Method used for thresholding."
    )
    parser.add_argument(
        "--otsu_threshold_multiplier",
        dest="otsu_threshold_multiplier",
        required=False,
        help="Factor to scale threshold during grain finding.",
    )
    parser.add_argument("-m", "--mask", dest="mask", type=bool, required=False, help="Mask the image.")
    parser.add_argument("-q", "--quiet", dest="quiet", type=bool, required=False, help="Toggle verbosity.")
    parser.add_argument(
        "-w",
        "--warnings",
        dest="warnings",
        type=bool,
        required=False,
        help="Whether to ignore warnings.",
    )
    return parser


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


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
    filter_out_path.mkdir(exist_ok=True, parents=True)
    grain_out_path = core_out_path / filename / "grains"
    Path.mkdir(grain_out_path / "upper", parents=True, exist_ok=True)
    Path.mkdir(grain_out_path / "lower", parents=True, exist_ok=True)
    dna_tracing_out_path = core_out_path / filename / "dna_tracing"
    Path.mkdir(dna_tracing_out_path / "upper", parents=True, exist_ok=True)
    Path.mkdir(dna_tracing_out_path / "lower", parents=True, exist_ok=True)
    Path.mkdir(dna_tracing_out_path / "upper" / "nodes", parents=True, exist_ok=True)
    Path.mkdir(dna_tracing_out_path / "lower" / "nodes", parents=True, exist_ok=True)

    # Filter Image :
    if filter_config["run"]:
        filter_config.pop("run")
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
        except IndexError:
            LOGGER.info(f"[{filename}] : No grains were detected, skipping Grain Statistics and DNA Tracing.")
        except ValueError:
            LOGGER.info(f"[{filename}] : No image, it is all masked.")
            results = create_empty_dataframe()
        if grains.region_properties is None:
            results = create_empty_dataframe()
        # Optionally plot grain finding stage
        if plotting_config["run"] and grains.region_properties is not None:
            plotting_config.pop("run")
            LOGGER.info(f"[{filename}] : Plotting Grain Finding Images")
            for direction, image_arrays in grains.directions.items():
                for plot_name, array in image_arrays.items():
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
                binary_mask = grains.directions[direction]["removed_small_objects"].copy()
                binary_mask[binary_mask != 0] = 1
                Images(
                    filtered_image.images["gaussian_filtered"],
                    filename=f"{filename}_{direction}_masked",
                    data2=binary_mask,
                    mask_cmap="green_black",
                    **plotting_config["plot_dict"][plot_name],
                ).plot_and_save()

            plotting_config["run"] = True

        # Grainstats :
        #
        # There are two layers to process those above the given threshold and those below, use dictionary comprehension
        # to pass over these.
        if grainstats_config["run"] and grains.region_properties is not None:
            grainstats_config.pop("run")
            # Grain Statistics :
            # try:
            LOGGER.info(f"[{filename}] : *** Grain Statistics ***")
            grain_plot_dict = {
                key: value
                for key, value in plotting_config["plot_dict"].items()
                if key in ["grain_image", "grain_mask", "grain_mask_image"]
            }
            grainstats = {}
            for direction in grains.directions.keys():
                grainstats[direction] = GrainStats(
                    data=filtered_image.images["gaussian_filtered"],
                    labelled_data=grains.directions[direction]["labelled_regions_02"],
                    pixel_to_nanometre_scaling=pixel_to_nm_scaling,
                    direction=direction,
                    base_output_dir=grain_out_path,
                    image_name=filename,
                    plot_opts=grain_plot_dict,
                    **grainstats_config,
                ).calculate_stats()
                grainstats[direction]["statistics"]["threshold"] = direction
            # Set tracing_stats_df in light of direction
            if grains_config["direction"] == "both":
                grainstats_df = pd.concat([grainstats["lower"]["statistics"], grainstats["upper"]["statistics"]])
            elif grains_config["direction"] == "upper":
                grainstats_df = grainstats["upper"]["statistics"]
            elif grains_config["direction"] == "lower":
                grainstats_df = grainstats["lower"]["statistics"]

            # Run dnatracing
            if dnatracing_config["run"]:
                dnatracing_config.pop("run")
                LOGGER.info(f"[{filename}] : *** DNA Tracing ***")
                dna_traces = defaultdict()
                tracing_stats = defaultdict()
                node_stats = defaultdict()
                for direction, _ in grainstats.items():
                    dna_traces[direction] = dnaTrace(
                        full_image_data=filtered_image.images["gaussian_filtered"],
                        grains=grains.directions[direction]["labelled_regions_02"],
                        filename=filename,
                        pixel_size=pixel_to_nm_scaling,
                        **dnatracing_config,
                    )
                    dna_traces[direction].trace_dna()

                    """
                    Images(
                        filtered_image.images["gaussian_filtered"],
                        filename=f"{filename}_{direction}_smooth_masked",
                        data2=dna_traces[direction].smoothed_grains,
                        mask_cmap="green_black",
                        **plotting_config["plot_dict"][plot_name],
                    ).plot_and_save()
                    """

                    tracing_stats[direction] = traceStats(trace_object=dna_traces[direction], image_path=image_path)
                    tracing_stats[direction].df["threshold"] = direction

                    nodes = nodeStats(
                        image=dna_traces[direction].full_image_data,
                        grains=grains.directions[direction]["removed_small_objects"],
                        skeletons=dna_traces[direction].skeletons,
                        px_2_nm=pixel_to_nm_scaling,
                    )
                    nodes.get_node_stats()
                    node_stats[direction] = nodes.full_dict

                    # Plot dnatracing images
                    LOGGER.info(f"[{filename}] : Plotting DNA Tracing Images")
                    output_dir = Path(dna_tracing_out_path / f"{direction}")

                    plot_names = ["orig_grains", "smoothed_grains", "orig_skeletons", "pruned_skeletons", "nodes"]
                    data2s = [
                        dna_traces[direction].grains_orig,
                        dna_traces[direction].smoothed_grains,
                        dna_traces[direction].orig_skeletons,
                        dna_traces[direction].skeletons,
                        nodes.all_connected_nodes,
                    ]
                    for i, plot_name in enumerate(plot_names):
                        plotting_config["plot_dict"][plot_name]["output_dir"] = output_dir
                        Images(
                            filtered_image.images["gaussian_filtered"],
                            data2=data2s[i],
                            zrange=[0, 3.5],
                            **plotting_config["plot_dict"][plot_name],
                        ).save_figure_black(
                            background=grains.directions[direction]["removed_small_objects"],
                        )

                    # plot nodes and line traces
                    for mol_no, mol_stats in node_stats[direction].items():
                        for node_no, single_node_stats in mol_stats.items():
                            # plot node + skeleton
                            Images(
                                single_node_stats["node_stats"]["node_area_image"],
                                data2=single_node_stats["node_stats"]["node_area_skeleton"],
                                filename=f"mol_{mol_no}_node_{node_no}_node_area",
                                output_dir=output_dir / "nodes",
                                zrange=[0, 3.5e-9],
                                **plotting_config["plot_dict"]["zoom_node"],
                            ).save_figure_black(background=single_node_stats["node_stats"]["node_area_grain"])
                            # plot branch mask
                            Images(
                                single_node_stats["node_stats"]["node_area_image"],
                                data2=single_node_stats["node_stats"]["node_branch_mask"],
                                filename=f"mol_{mol_no}_node_{node_no}_crossings",
                                output_dir=output_dir / "nodes",
                                zrange=[0, 3.5e-9],
                                **plotting_config["plot_dict"]["crossings"],
                            ).save_figure_black(background=single_node_stats["node_stats"]["node_area_grain"])
                            # plot avg branch mask
                            if single_node_stats["node_stats"]["node_avg_mask"] is not None:
                                Images(
                                    single_node_stats["node_stats"]["node_area_image"],
                                    data2=single_node_stats["node_stats"]["node_avg_mask"],
                                    filename=f"mol_{mol_no}_node_{node_no}_average_crossings",
                                    output_dir=output_dir / "nodes",
                                    zrange=[0, 3.5e-9],
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
                        # plot the molecules on their own
                        if len(nodes.mol_coords[mol_no]) > 1:
                            for inner_mol_no, coords in enumerate(nodes.mol_coords[mol_no]):
                                single_mol = np.zeros_like(dna_traces[direction].full_image_data)
                                single_mol[coords[:, 0], coords[:, 1]] = 1
                                single_mol = binary_dilation(single_mol)
                                Images(
                                    dna_traces[direction].full_image_data,
                                    data2=single_mol,
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
                            data2=visual,
                            output_dir=output_dir,
                            zrange=[0, 3.5e-9],
                            **plotting_config["plot_dict"]["visual"],
                        ).save_figure_black(background=grains.directions[direction]["removed_small_objects"])

                    # ------- branch vector img -------
                    """
                    vectors = nodes.test2
                    plot_name = "test"
                    plotting_config["plot_dict"][plot_name]["output_dir"] = output_dir
                    fig, ax = Images(
                        data=node_stats[direction][1][1]["node_stats"]["node_area_image"],
                        data2=nodes.test,
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
                        data2=node_stats[direction][1][1]["node_stats"]["node_branch_mask"],
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

                # Set tracing_stats_df in light of direction
                if grains_config["direction"] == "both":
                    tracing_stats_df = pd.concat([tracing_stats["lower"].df, tracing_stats["upper"].df])
                elif grains_config["direction"] == "upper":
                    tracing_stats_df = tracing_stats["upper"].df
                elif grains_config["direction"] == "lower":
                    tracing_stats_df = tracing_stats["lower"].df
                LOGGER.info(f"[{filename}] : Combining {direction} grain statistics and dnatracing statistics")
                results = grainstats_df.merge(tracing_stats_df, on=["Molecule Number", "threshold"])
            else:
                results = grainstats_df
                results["Image Name"] = filename
                results["Basename"] = image_path.parent

            """except Exception:
                # If no results we need a dummy dataframe to return.
                LOGGER.info(
                    f"[{filename}] : Errors occurred whilst calculating grain statistics and DNA tracing statistics."
                )
                results = create_empty_dataframe()
            """
        else:
            results = create_empty_dataframe()
            results["Image Name"] = filename
            results["Basename"] = image_path.parent
            node_stats = {"upper": None, "lower": None}

    return image_path, results, node_stats


def main(args=None):
    """Find and process all files."""

    # Parse command line options, load config (or default) and update with command line options
    parser = create_parser()
    args = parser.parse_args() if args is None else parser.parse_args(args)
    if args.config_file is not None:
        config = read_yaml(args.config_file)
    else:
        default_config = pkg_resources.open_text(__package__, "default_config.yaml")
        config = yaml.safe_load(default_config.read())
    config = update_config(config, args)

    # Validate configuration
    validate_config(config, schema=DEFAULT_CONFIG_SCHEMA, config_type="YAML configuration file")

    config["output_dir"].mkdir(parents=True, exist_ok=True)

    # Write sample configuration if asked to do so and exit
    if args.create_config_file:
        write_yaml(
            config,
            output_dir="./",
            config_file=args.create_config_file,
            header_message="Sample configuration file auto-generated",
        )
        LOGGER.info(f"A sample configuration has been written to : ./{args.create_config_file}")
        LOGGER.info(
            "Please refer to the documentation on how to use the configuration file : \n\n"
            "https://afm-spm.github.io/TopoStats/usage.html#configuring-topostats\n"
            "https://afm-spm.github.io/TopoStats/configuration.html"
        )
        sys.exit()
    # Load plotting_dictionary and validate
    plotting_dictionary = pkg_resources.open_text(__package__, "plotting_dictionary.yaml")
    config["plotting"]["plot_dict"] = yaml.safe_load(plotting_dictionary.read())
    validate_config(
        config["plotting"]["plot_dict"], schema=PLOTTING_SCHEMA, config_type="YAML plotting configuration file"
    )

    # FIXME : Make this a function and from topostats.utils import update_plot_dict and write tests
    # Update the config["plotting"]["plot_dict"] with plotting options
    for image, options in config["plotting"]["plot_dict"].items():
        config["plotting"]["plot_dict"][image] = {
            **options,
            "save_format": config["plotting"]["save_format"],
            "image_set": config["plotting"]["image_set"],
            "colorbar": config["plotting"]["colorbar"],
            "axes": config["plotting"]["axes"],
            "cmap": config["plotting"]["cmap"],
            "zrange": config["plotting"]["zrange"],
            "histogram_log_axis": config["plotting"]["histogram_log_axis"],
        }
        if image not in ["z_threshed", "mask_overlay", "grain_image", "grain_mask_image"]:
            config["plotting"]["plot_dict"][image].pop("zrange")

    LOGGER.info(f"Configuration file loaded from      : {args.config_file}")
    LOGGER.info(f"Scanning for images in              : {config['base_dir']}")
    LOGGER.info(f"Output directory                    : {str(config['output_dir'])}")
    LOGGER.info(f"Looking for images with extension   : {config['file_ext']}")
    img_files = find_images(config["base_dir"], file_ext=config["file_ext"])
    LOGGER.info(f"Images with extension {config['file_ext']} in {config['base_dir']} : {len(img_files)}")
    if len(img_files) == 0:
        LOGGER.error(f"No images with extension {config['file_ext']} in {config['base_dir']}")
        LOGGER.error("Please check your configuration and directories.")
        sys.exit()
    LOGGER.info(f'Thresholding method (Filtering)     : {config["filter"]["threshold_method"]}')
    LOGGER.info(f'Thresholding method (Grains)        : {config["grains"]["threshold_method"]}')

    if config["quiet"]:
        LOGGER.setLevel("ERROR")

    processing_function = partial(
        process_scan,
        base_dir=config["base_dir"],
        filter_config=config["filter"],
        grains_config=config["grains"],
        grainstats_config=config["grainstats"],
        dnatracing_config=config["dnatracing"],
        plotting_config=config["plotting"],
        output_dir=config["output_dir"],
    )

    all_scan_data = LoadScans(img_files, **config["loading"])
    all_scan_data.get_data()
    scan_data_dict = all_scan_data.img_dic

    with Pool(processes=config["cores"]) as pool:
        results = defaultdict()
        node_results = defaultdict()
        with tqdm(
            total=len(img_files),
            desc=f"Processing images from {config['base_dir']}, results are under {config['output_dir']}",
        ) as pbar:
            for img, result, node_result in pool.imap_unordered(
                processing_function,
                scan_data_dict.values(),
            ):
                results[str(img)] = result
                node_results[str(img)] = node_result
                pbar.update()
    results = pd.concat(results.values())
    results.reset_index()
    results.to_csv(config["output_dir"] / "all_statistics.csv", index=False)
    with open(config["output_dir"] / "all_node_stats.json", "w", encoding="utf8") as json_file:
        json.dump(node_results, json_file, cls=NpEncoder)
    folder_grainstats(config["output_dir"], config["base_dir"], results)
    # Write config to file
    config["plotting"].pop("plot_dict")
    write_yaml(config, output_dir=config["output_dir"])
    images_processed = len(results["Image Name"].unique())
    LOGGER.info(
        (
            f"\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ COMPLETE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n"
            f"  Base Directory              : {config['base_dir']}\n"
            f"  File Extension              : {config['file_ext']}\n"
            f"  Files Found                 : {len(img_files)}\n"
            f"  Successfully Processed      : {images_processed} ({(images_processed * 100) / len(img_files)}%)\n"
            f"  Configuration               : {config['output_dir']}/config.yaml\n"
            f"  All statistics              : {str(config['output_dir'])}/all_statistics.csv\n\n"
            f"  Email                       : topostats@sheffield.ac.uk\n"
            f"  Documentation               : https://afm-spm.github.io/topostats/\n"
            f"  Source Code                 : https://github.com/AFM-SPM/TopoStats/\n"
            f"  Bug Reports/Feature Request : https://github.com/AFM-SPM/TopoStats/issues/new/choose\n"
            f"  Citation File Format        : https://github.com/AFM-SPM/TopoStats/blob/main/CITATION.cff\n\n"
            f"  If you encounter bugs/issues or have feature requests please report them at the above URL\n"
            f"  or email us.\n\n"
            f"  If you have found TopoStats useful please consider citing it. A citation file format is\n"
            f"  included and there are links on the Source Code page.\n"
            f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n"
        )
    )


if __name__ == "__main__":
    main()
