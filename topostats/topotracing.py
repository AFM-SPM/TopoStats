"""Topotracing"""
import argparse as arg
from asyncio.log import logger
from collections import defaultdict

# from email.policy import default
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Union, Dict

import pandas as pd
import numpy as np
from tqdm import tqdm

from topostats.filters import Filters
from topostats.grains import Grains
from topostats.grainstats import GrainStats, get_grainstats
from topostats.io import read_yaml
from topostats.logs.logs import setup_logger, LOGGER_NAME
from topostats.plottingfuncs import plot_and_save
from topostats.thresholds import threshold
from topostats.utils import convert_path, find_images, update_config

LOGGER = setup_logger(LOGGER_NAME)

# PLOT_DICT = {
#     "pixels": {"filename": "01-raw_heightmap.png", "title": "Raw Height"},
#     "initial_align": {
#         "filename": "02-initial_align_unmasked.png",
#         "title": "Initial Alignment (Unmasked)",
#     },
#     "initial_tilt_removal": {
#         "filename": "03-initial_tilt_removal_unmasked.png",
#         "title": "Initial Tilt Removal (Unmasked)",
#     },
#     "mask": {"filename": "04-binary_mask.png", "title": "Binary Mask"},
#     "masked_align": {
#         "filename": "05-secondary_align_masked.png",
#         "title": "Secondary Alignment (Masked)",
#     },
#     "masked_tilt_removal": {
#         "filename": "06-secondary_tilt_removal_masked.png",
#         "title": "Secondary Tilt Removal (Masked)",
#     },
#     "zero_averaged_background": {
#         "filename": "07-zero_average_background.png",
#         "title": "Zero Average Background",
#     },
#     "gaussian_filtered": {
#         "filename": "08-gaussian_filtered.png",
#         "title": "Gaussian Filtered",
#     },
#     "mask_grains": {"filename": "09-mask_grains.png", "title": "Mask for Grains"},
#     "tidied_border": {"filename": "10-tidy_borders.png", "title": "Tidied Borders"},
#     "tiny_objects_removed": {
#         "filename": "11a-tiny_objects_removed.png",
#         "title": "Tiny objects removed",
#     },
#     "objects_removed": {
#         "filename": "11-small_objects_removed.png",
#         "title": "Small Objects Removed",
#     },
#     "labelled_regions": {
#         "filename": "12-labelled_regions.png",
#         "title": "Labelled Regions",
#     },
#     "coloured_regions": {
#         "filename": "13-coloured_regions.png",
#         "title": "Coloured Regions",
#     },
#     "bounding_boxes": {"filename": "14-bounding_boxes.png", "title": "Bounding Boxes"},
#     "coloured_boxes": {
#         "filename": "15-labelled_image_bboxes.png",
#         "title": "Labelled Image with Bounding Boxes",
#     },
# }


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
        "-t",
        "--threshold_method",
        dest="threshold_method",
        required=False,
        help="Method used for thresholding.",
    )
    parser.add_argument(
        "--threshold_multiplier",
        dest="threshold_multiplier",
        required=False,
        help="Factor to scale threshold during grain finding.",
    )
    parser.add_argument("-m", "--mask", dest="mask", type=bool, required=False, help="Mask the image.")
    parser.add_argument(
        "-q",
        "--quiet",
        dest="quiet",
        type=bool,
        required=False,
        help="Toggle verbosity.",
    )
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
    filter_threshold_multiplier: Union[int, float] = 1.7,
    filter_threshold_std_dev=1.0,
    filter_threshold_abs_lower=None,
    filter_threshold_abs_upper=None,
    gaussian_size: Union[int, float] = 2,
    gaussian_mode: str = "nearest",
    absolute_smallest_grain_size=None,
    background: float = 0.0,
    grains_threshold_method: str = "otsu",
    grains_threshold_multiplier: Union[int, float] = 1.7,
    grains_threshold_std_dev=1.0,
    grains_threshold_abs_lower=None,
    grains_threshold_abs_upper=None,
    save_plots: bool = True,
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
    mode : str
        Mode for filtering (default is 'nearest').
    threshold_multiplier : Union[int, float]
        Factor by which lower threshold is to be scaled prior to masking.
    background : float
    save_plots : bool
        Flag as to whether to save plots to PNG files.
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
    # Filter Image :
    #
    # The Filters class has a convenience method that runs the instantiated class in full.
    filtered_image = Filters(
        image_path,
        threshold_method=filter_threshold_method,
        threshold_std_dev=filter_threshold_std_dev,
        threshold_absolute_lower=filter_threshold_abs_lower,
        threshold_absolute_upper=filter_threshold_abs_upper,
        channel=channel,
        amplify_level=amplify_level,
        output_dir=output_dir,
    )
    filtered_image.filter_image()

    # Find Grains :
    #
    # The Grains class also has a convenience method that runs the instantiated class in full.
    grains = Grains(
        image=filtered_image.images["zero_averaged_background"],
        filename=filtered_image.filename,
        pixel_to_nm_scaling=filtered_image.pixel_to_nm_scaling,
        gaussian_size=gaussian_size,
        gaussian_mode=gaussian_mode,
        threshold_method=grains_threshold_method,
        threshold_std_dev=grains_threshold_std_dev,
        threshold_absolute_lower=grains_threshold_abs_lower,
        threshold_absolute_upper=grains_threshold_abs_upper,
        absolute_smallest_grain_size=absolute_smallest_grain_size,
        background=background,
        output_dir=output_dir,
    )
    grains.find_grains()

    # Grainstats :
    #
    # There are two layers to process those above the given threshold and those below, use dictionary comprehension
    # to pass over these.
    grainstats = {
        direction: GrainStats(
            data=grains.images["gaussian_filtered"],
            labelled_data=grains.directions[direction]["labelled_regions_02"],
            pixel_to_nanometre_scaling=filtered_image.pixel_to_nm_scaling,
            img_name=f"{filtered_image.filename}/{direction}",
            output_dir=output_dir,
        ).calculate_stats()
        for direction in grains.directions
    }
    grainstats["lower"]["statistics"]["threshold"] = "lower"
    grainstats["upper"]["statistics"]["threshold"] = "upper"
    grainstats_df = pd.concat([grainstats["lower"]["statistics"], grainstats["upper"]["statistics"]])
    grainstats_df.to_csv(output_dir / filtered_image.filename / "grainstats.csv")


def main():
    """Run processing."""

    # Parse command line options, load config and update with command line options
    parser = create_parser()
    args = parser.parse_args()
    config = read_yaml(args.config_file)
    config = update_config(config, args)

    LOGGER.info(f"Configuration file loaded from    : {args.config_file}")
    LOGGER.info(f'Scanning for images in            : {config["base_dir"]}')
    LOGGER.info(f'Output directory                  : {config["output_dir"]}')
    LOGGER.info(f'Looking for images with extension : {config["file_ext"]}')
    img_files = find_images(config["base_dir"])
    LOGGER.info(f'Images with extension {config["file_ext"]} in {config["base_dir"]} : {len(img_files)}')

    if config["quiet"]:
        LOGGER.setLevel("ERROR")
    # For debugging (as Pool makes it hard to find things when they go wrong)
    # for x in img_files:
    #     process_scan(
    #         image_path=x,
    #         channel=config["channel"],
    #         amplify_level=config["amplify_level"],
    #         threshold_method=config["threshold_method"],
    #         threshold_multiplier=config["grains"]["threshold_multiplier"],
    #         gaussian_size=config["grains"]["gaussian_size"],
    #         gaussian_mode=config["grains"]["gaussian_mode"],
    #         background=config["grains"]["background"],
    #         output_dir=config["output_dir"],
    #     )
    processing_function = partial(
        process_scan,
        channel=config["channel"],
        amplify_level=config["amplify_level"],
        filter_threshold_method=config["filter"]["threshold"]["method"],
        filter_threshold_multiplier=config["filter"]["threshold"]["multiplier"],
        filter_threshold_std_dev=config["filter"]["threshold"]["std_dev"],
        filter_threshold_abs_lower=config["filter"]["threshold"]["absolute"][0],
        filter_threshold_abs_upper=config["filter"]["threshold"]["absolute"][1],
        gaussian_size=config["grains"]["gaussian_size"],
        gaussian_mode=config["grains"]["gaussian_mode"],
        absolute_smallest_grain_size=config["grains"]["absolute_smallest_grain_size"],
        background=config["grains"]["background"],
        output_dir=config["output_dir"],
        grains_threshold_method=config["grains"]["threshold"]["method"],
        grains_threshold_multiplier=config["grains"]["threshold"]["multiplier"],
        grains_threshold_std_dev=config["grains"]["threshold"]["std_dev"],
        grains_threshold_abs_lower=config["grains"]["threshold"]["absolute"][0],
        grains_threshold_abs_upper=config["grains"]["threshold"]["absolute"][1],
    )

    with Pool(processes=config["cores"]) as pool:
        with tqdm(
            total=len(img_files),
            desc=f'Processing images from {config["base_dir"]}, results are under {config["output_dir"]}',
        ) as pbar:
            for _ in pool.imap_unordered(processing_function, img_files):
                pbar.update()


if __name__ == "__main__":
    main()
