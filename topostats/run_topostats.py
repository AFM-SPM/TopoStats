"""Topotracing"""
import argparse as arg
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Union
import pandas as pd
import numpy as np

from tqdm import tqdm

from topostats.filters import Filters
from topostats.grains import Grains
from topostats.grainstats import GrainStats
from topostats.io import read_yaml, write_yaml
from topostats.logs.logs import setup_logger, LOGGER_NAME
from topostats.plottingfuncs import plot_and_save
from topostats.tracing.dnatracing import dnaTrace, traceStats
from topostats.utils import find_images, update_config, convert_path, create_empty_dataframe

LOGGER = setup_logger(LOGGER_NAME)


# pylint: disable=broad-except
# pylint: disable=line-too-long
# pylint: disable=too-many-arguments
# pylint: disable=too-many-branches
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=unnecessary-dict-index-lookup

PLOT_DICT = {
    "extracted_channel": {"filename": "00-raw_heightmap.png", "title": "Raw Height", "type": "non-binary"},
    "pixels": {"filename": "01-pixels.png", "title": "Pixels", "type": "non-binary"},
    "initial_align": {"filename": "02-initial_align_unmasked.png", "title": "Initial Alignment (Unmasked)", "type": "non-binary"},
    "initial_tilt_removal": {
        "filename": "03-initial_tilt_removal_unmasked.png",
        "title": "Initial Tilt Removal (Unmasked)",
        "type": "non-binary",
    },
    "mask": {"filename": "04-binary_mask.png", "title": "Binary Mask", "type": "binary"},
    "masked_align": {"filename": "05-secondary_align_masked.png", "title": "Secondary Alignment (Masked)", "type": "non-binary"},
    "masked_tilt_removal": {
        "filename": "06-secondary_tilt_removal_masked.png",
        "title": "Secondary Tilt Removal (Masked)",
        "type": "non-binary",
    },
    "zero_averaged_background": {"filename": "07-zero_average_background.png", "title": "Zero Average Background", "type": "non-binary"},
    "gaussian_filtered": {"filename": "08-gaussian_filtered.png", "title": "Gaussian Filtered", "type": "non-binary"},
    "z_threshed": {"filename": "08_5-z_thresholded.png", "title": "Height Thresholded", "type": "non-binary"},
    "mask_grains": {"filename": "09-mask_grains.png", "title": "Mask for Grains", "type": "binary"},
    "tidied_border": {"filename": "10-tidy_borders.png", "title": "Tidied Borders", "type": "binary"},
    "removed_noise": {"filename": "11-noise_removed.png", "title": "Noise removed", "type": "binary"},
    "labelled_regions_01": {"filename": "12-labelled_regions.png", "title": "Labelled Regions", "type": "binary"},
    "removed_small_objects": {"filename": "13-small_objects_removed.png", "title": "Small Objects Removed", "type": "binary"},
    "mask_overlay": {"filename": "13_5-mask_overlay.png", "title": "Hight Thresholded with Molecule Mask", "type": "non-binary"},
    "labelled_regions_02": {"filename": "14-labelled_regions.png", "title": "Labelled Regions", "type": "binary"},
    "coloured_regions": {"filename": "15-coloured_regions.png", "title": "Coloured Regions", "type": "binary"},
    "bounding_boxes": {"filename": "16-bounding_boxes.png", "title": "Bounding Boxes", "type": "binary"},
    "coloured_boxes": {"filename": "17-labelled_image_bboxes.png", "title": "Labelled Image with Bounding Boxes", "type": "binary"},
}


def create_parser() -> arg.ArgumentParser:
    """Create a parser for reading options."""
    parser = arg.ArgumentParser(
        description="Process AFM images. Additional arguments over-ride those in the configuration file."
    )
    parser.add_argument(
        "-c",
        "--config_file",
        dest="config_file",
        required=True,
        help="Path to a YAML configuration file.",
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
        "-a",
        "--amplify_level",
        dest="amplify_level",
        type=float,
        required=False,
        help="Amplify signals by the given factor.",
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
        dest="quiet",
        type=str,
        required=False,
        help="Whether to ignore warnings.",
    )
    return parser


def process_scan(
    image_path: Union[str, Path] = None,
    channel: str = "Height",
    amplify_level: float = 1.0,
    filter_threshold_method: str = "otsu",
    filter_otsu_threshold_multiplier: Union[int, float] = 1.7,
    filter_threshold_std_dev=1.0,
    filter_threshold_abs_lower=None,
    filter_threshold_abs_upper=None,
    gaussian_size: Union[int, float] = 2,
    gaussian_mode: str = "nearest",
    absolute_smallest_grain_size=None,
    background: float = 0.0,
    grains_threshold_method: str = "otsu",
    grains_otsu_threshold_multiplier: Union[int, float] = 1.7,
    grains_threshold_std_dev=1.0,
    grains_threshold_abs_lower=None,
    grains_threshold_abs_upper=None,
    zrange = None,
    mask_direction = None,
    save_plots: bool = True,
    colorbar: bool = True,
    output_dir: Union[str, Path] = "output",
) -> None:
    """Process a single image, filtering, finding grains and calculating their statistics.

    Parameters
    ----------
    image_path : Union[str, Path]
        Path to image to process.
    channel : str
        Channel to extract and process, default 'height'.
    amplify_level : float
        Level to amplify image prior to processing by.
    threshold_method: str
        Method for determining threshold to mask values, default is 'otsu'.
    gaussian_size : Union[int, float]
        Minimum grain size in nanometers (nm).
    gaussian_mode : str
        Mode for filtering (default is 'nearest').
    otsu_threshold_multiplier : Union[int, float]
        Factor by which lower threshold is to be scaled prior to masking.
    background : float
        The value to average the background around.
    zrange : list
        Lower and upper limits for the Z-range.
    save_plots : bool
        Flag as to whether to save plots to PNG files.
    colorbar : bool
        Flag as to whether to include a colorbar scale in scan plots.
    output_dir : Union[str, Path]
        Path to output directory for saving results.

    Examples
    --------

    from topostats.topotracing import process_scan

    process_scan(image_path='minicircle.spm',
                 save_plots=True,
                 output_dir='output/')

    """
    LOGGER.info(f"Processing : {image_path}")

    _output_dir = output_dir
    _output_dir.mkdir(parents=True, exist_ok=True)
    # Filter Image :
    #
    # The Filters class has a convenience method that runs the instantiated class in full.
    filtered_image = Filters(
        image_path,
        threshold_method=filter_threshold_method,
        otsu_threshold_multiplier=filter_otsu_threshold_multiplier,
        threshold_std_dev=filter_threshold_std_dev,
        threshold_absolute_lower=filter_threshold_abs_lower,
        threshold_absolute_upper=filter_threshold_abs_upper,
        channel=channel,
        amplify_level=amplify_level,
        output_dir=_output_dir / image_path.stem / "filters",
    )
    filtered_image.filter_image()
    Path.mkdir(_output_dir / filtered_image.filename / "upper", parents=True, exist_ok=True)
    Path.mkdir(_output_dir / filtered_image.filename / "lower", parents=True, exist_ok=True)

    # Find Grains :
    # The Grains class also has a convenience method that runs the instantiated class in full.
    try:
        LOGGER.info(f"[{filtered_image.filename}] : *** Grain Finding ***")
        grains = Grains(
            image=filtered_image.images["zero_averaged_background"],
            filename=filtered_image.filename,
            pixel_to_nm_scaling=filtered_image.pixel_to_nm_scaling,
            gaussian_size=gaussian_size,
            gaussian_mode=gaussian_mode,
            threshold_method=grains_threshold_method,
            otsu_threshold_multiplier=grains_otsu_threshold_multiplier,
            threshold_std_dev=grains_threshold_std_dev,
            threshold_absolute_lower=grains_threshold_abs_lower,
            threshold_absolute_upper=grains_threshold_abs_upper,
            absolute_smallest_grain_size=absolute_smallest_grain_size,
            background=background,
            base_output_dir=_output_dir / filtered_image.filename / "grains",
            zrange=zrange,
        )
        grains.find_grains()
    except IndexError:
        LOGGER.info(
            f"[{filtered_image.filename}] : No grains were detected, skipping Grain Statistics and DNA Tracing."
        )
    except ValueError:
        LOGGER.info(f"[{filtered_image.filename}] : No image, it is all masked.")
        results = create_empty_dataframe()

    # Grainstats :
    #
    # There are two layers to process those above the given threshold and those below, use dictionary comprehension
    # to pass over these.
    if grains.region_properties is not None:
        # Grain Statistics :
        try:
            LOGGER.info(f"[{filtered_image.filename}] : *** Grain Statistics ***")
            grainstats = {
                direction: GrainStats(
                    data=grains.images["gaussian_filtered"],
                    labelled_data=grains.directions[direction]["labelled_regions_02"],
                    pixel_to_nanometre_scaling=filtered_image.pixel_to_nm_scaling,
                    direction=f"{direction}",
                    base_output_dir=_output_dir / filtered_image.filename,
                    image_name=filtered_image.filename,
                ).calculate_stats()
                for direction in grains.directions
            }
            # If there are both upper and lower grainstats, then join them, else otherwise we always have upper
            if "lower" in grainstats.keys():
                grainstats["lower"]["statistics"]["threshold"] = "lower"
                grainstats["upper"]["statistics"]["threshold"] = "upper"
                grainstats_df = pd.concat([grainstats["lower"]["statistics"], grainstats["upper"]["statistics"]])
            else:
                grainstats_df = grainstats["upper"]["statistics"]
            grainstats_df.to_csv(_output_dir / filtered_image.filename / "grainstats.csv")

            # Run dnatracing
            LOGGER.info(f"[{filtered_image.filename}] : *** DNA Tracing ***")
            dna_traces = defaultdict()
            tracing_stats = defaultdict()
            for direction, grainstat in grainstats.items():
                dna_traces[direction] = dnaTrace(
                    full_image_data=grains.images["gaussian_filtered"].T,
                    grains=grains.directions[direction]["labelled_regions_02"],
                    filename=filtered_image.filename,
                    pixel_size=filtered_image.pixel_to_nm_scaling,
                )
                dna_traces[direction].trace_dna()
                tracing_stats[direction] = traceStats(trace_object=dna_traces[direction], image_path=image_path)
                tracing_stats[direction].save_trace_stats(_output_dir / filtered_image.filename / direction)

                LOGGER.info(
                    f"[{filtered_image.filename}] : Combining {direction} grain statistics and dnatracing statistics"
                )
                results = grainstat["statistics"].merge(tracing_stats[direction].df, on="Molecule Number")
                results.to_csv(_output_dir / filtered_image.filename / direction / "all_statistics.csv")
                LOGGER.info(
                    f"[{filtered_image.filename}] : Combined statistics saved to {str(_output_dir)}/{filtered_image.filename}/{direction}/all_statistics.csv"
                )

        except Exception:
            # If no results we need a dummy dataframe to return.
            LOGGER.info(
                f"[{filtered_image.filename}] : Errors occurred attempting to calculate grain statistics and DNA tracing statistics."
            )
            results = create_empty_dataframe()

    # Optionally plot all stages
    if save_plots:
        LOGGER.info(f"[{filtered_image.filename}] : Plotting Filtering Images")
        # Update PLOT_DICT with pixel_to_nm_scaling (can't add _output_dir since it changes)
        plot_opts = {"pixel_to_nm_scaling_factor": filtered_image.pixel_to_nm_scaling, "colorbar": colorbar}
        for image, options in PLOT_DICT.items():
            PLOT_DICT[image] = {**options, **plot_opts}

        # Filtering stage
        for plot_name, array in filtered_image.images.items():
            if plot_name not in ["scan_raw"]:
                if plot_name == "extracted_channel":
                    array = np.flipud(array.pixels)
                PLOT_DICT[plot_name]["output_dir"] = Path(_output_dir) / filtered_image.filename
                try:
                    plot_and_save(array, **PLOT_DICT[plot_name])
                except AttributeError:
                    LOGGER.info(f"[{filtered_image.filename}] Unable to generate plot : {plot_name}")

        # Grain stage - only if we have grains
        if grains.region_properties is not None:
            LOGGER.info(f"[{filtered_image.filename}] : Plotting Grain Images")
            plot_name = "gaussian_filtered"
            PLOT_DICT[plot_name]["output_dir"] = Path(_output_dir) / filtered_image.filename
            plot_and_save(grains.images["gaussian_filtered"], **PLOT_DICT[plot_name])

            if zrange is not None:
                plot_name = "z_threshed"
                PLOT_DICT[plot_name]["output_dir"] = Path(_output_dir) / filtered_image.filename
                plot_and_save(grains.images["z_threshed"], **PLOT_DICT[plot_name])

            for direction, image_arrays in grains.directions.items():
                output_dir = Path(_output_dir) / filtered_image.filename / f"{direction}"
                for plot_name, array in image_arrays.items():
                    PLOT_DICT[plot_name]["output_dir"] = output_dir
                    plot_and_save(array, **PLOT_DICT[plot_name])
                # Make a plot of coloured regions with bounding boxes
                PLOT_DICT["bounding_boxes"]["output_dir"] = output_dir
                plot_and_save(
                    grains.directions[direction]["coloured_regions"],
                    **PLOT_DICT["bounding_boxes"],
                    region_properties=grains.region_properties[direction],
                )
                PLOT_DICT["coloured_boxes"]["output_dir"] = output_dir
                plot_and_save(
                    grains.directions[direction]["labelled_regions_02"],
                    **PLOT_DICT["coloured_boxes"],
                    region_properties=grains.region_properties[direction],
                )

            plot_name = "mask_overlay"
            PLOT_DICT[plot_name]["output_dir"] = Path(_output_dir) / filtered_image.filename
            plot_and_save(
                grains.images["z_threshed"], 
                data2=grains.directions[mask_direction]["removed_small_objects"], 
                **PLOT_DICT[plot_name]
            )

    return image_path, results


def main():
    """Run processing."""

    # Parse command line options, load config and update with command line options
    parser = create_parser()
    args = parser.parse_args()
    config = read_yaml(args.config_file)
    config = update_config(config, args)
    config["output_dir"] = convert_path(config["output_dir"])
    config["output_dir"].mkdir(parents=True, exist_ok=True)

    # Update the PLOT_DICT with plotting options
    for image, options in PLOT_DICT.items():
        PLOT_DICT[image] = {**options, **config["plotting"]}

    LOGGER.info(f"Configuration file loaded from      : {args.config_file}")
    LOGGER.info(f'Scanning for images in              : {config["base_dir"]}')
    LOGGER.info(f'Output directory                    : {str(config["output_dir"])}')
    LOGGER.info(f'Looking for images with extension   : {config["file_ext"]}')
    img_files = find_images(config["base_dir"])
    LOGGER.info(f'Images with extension {config["file_ext"]} in {config["base_dir"]} : {len(img_files)}')
    LOGGER.info(f'Thresholding method (Filtering)     : {config["filter"]["threshold"]["method"]}')
    LOGGER.info(f'Thresholding method (Grains)        : {config["grains"]["threshold"]["method"]}')

    if config["quiet"]:
        LOGGER.setLevel("ERROR")
    # For debugging (as Pool makes it hard to find things when they go wrong)
    # for x in img_files:
    #     process_scan(
    #         image_path=x,
    #         channel=config["channel"],
    #         amplify_level=config["amplify_level"],
    #         threshold_method=config["threshold_method"],
    #         otsu_threshold_multiplier=config["grains"]["otsu_threshold_multiplier"],

    #         gaussian_size=config["grains"]["gaussian_size"],
    #         gaussian_mode=config["grains"]["gaussian_mode"],
    #         background=config["grains"]["background"],
    #         save_plots=config["plotting"]["save"],
    #         colorbar=config["plotting"]["colorbar"],
    #         output_dir=config["output_dir"],
    #     )
    processing_function = partial(
        process_scan,
        channel=config["channel"],
        amplify_level=config["amplify_level"],
        filter_threshold_method=config["filter"]["threshold"]["method"],
        filter_otsu_threshold_multiplier=config["filter"]["threshold"]["otsu_multiplier"],
        filter_threshold_std_dev=config["filter"]["threshold"]["std_dev"],
        filter_threshold_abs_lower=config["filter"]["threshold"]["absolute"][0],
        filter_threshold_abs_upper=config["filter"]["threshold"]["absolute"][1],
        gaussian_size=config["grains"]["gaussian_size"],
        gaussian_mode=config["grains"]["gaussian_mode"],
        absolute_smallest_grain_size=config["grains"]["absolute_smallest_grain_size"],
        background=config["grains"]["background"],
        zrange=config["grains"]["zrange"],
        mask_direction=config["grains"]["mask_direction"],
        save_plots=config["plotting"]["save"],
        colorbar=config["plotting"]["colorbar"],
        output_dir=config["output_dir"],
        grains_threshold_method=config["grains"]["threshold"]["method"],
        grains_otsu_threshold_multiplier=config["grains"]["threshold"]["otsu_multiplier"],
        grains_threshold_std_dev=config["grains"]["threshold"]["std_dev"],
        grains_threshold_abs_lower=config["grains"]["threshold"]["absolute"][0],
        grains_threshold_abs_upper=config["grains"]["threshold"]["absolute"][1],
    )

    with Pool(processes=config["cores"]) as pool:
        results = defaultdict()
        with tqdm(
            total=len(img_files),
            desc=f'Processing images from {config["base_dir"]}, results are under {config["output_dir"]}',
        ) as pbar:
            for img, result in pool.imap_unordered(processing_function, img_files):
                results[str(img)] = result
                pbar.update()

    results = pd.concat(results.values())
    results.reset_index()
    results.to_csv(config["output_dir"] / "all_statistics.csv", index=False)
    LOGGER.info(
        f"All statistics combined for {len(img_files)} images(s) are saved to : {str(config['output_dir'] / 'all_statistics.csv')}"
    )

    # Write config to file
    LOGGER.info(f"Writing configuration to : {config['output_dir']}/config.yaml")
    write_yaml(config, output_dir=config["output_dir"])


if __name__ == "__main__":
    main()
