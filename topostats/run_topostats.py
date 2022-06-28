"""Topotracing"""
import argparse as arg
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from topostats.filters import Filters
from topostats.grains import Grains
from topostats.grainstats import GrainStats
from topostats.io import read_yaml, write_yaml
from topostats.logs.logs import setup_logger, LOGGER_NAME
from topostats.plottingfuncs import plot_and_save
from topostats.tracing.dnatracing import dnaTrace, traceStats
from topostats.utils import find_images, update_config

LOGGER = setup_logger(LOGGER_NAME)

PLOT_DICT = {
    "extracted_channel": {"filename": "00-raw_heightmap.png", "title": "Raw Height"},
    "pixels": {"filename": "01-pixels.png", "title": "Pixels"},
    "initial_align": {"filename": "02-initial_align_unmasked.png", "title": "Initial Alignment (Unmasked)"},
    "initial_tilt_removal": {
        "filename": "03-initial_tilt_removal_unmasked.png",
        "title": "Initial Tilt Removal (Unmasked)",
    },
    "mask": {"filename": "04-binary_mask.png", "title": "Binary Mask"},
    "masked_align": {"filename": "05-secondary_align_masked.png", "title": "Secondary Alignment (Masked)"},
    "masked_tilt_removal": {
        "filename": "06-secondary_tilt_removal_masked.png",
        "title": "Secondary Tilt Removal (Masked)",
    },
    "zero_averaged_background": {"filename": "07-zero_average_background.png", "title": "Zero Average Background"},
    "gaussian_filtered": {"filename": "08-gaussian_filtered.png", "title": "Gaussian Filtered"},
    "mask_grains": {"filename": "09-mask_grains.png", "title": "Mask for Grains"},
    "tidied_border": {"filename": "10-tidy_borders.png", "title": "Tidied Borders"},
    "objects_removed": {"filename": "11-small_objects_removed.png", "title": "Small Objects Removed"},
    "labelled_regions": {"filename": "12-labelled_regions.png", "title": "Labelled Regions"},
    "coloured_regions": {"filename": "13-coloured_regions.png", "title": "Coloured Regions"},
    "bounding_boxes": {"filename": "14-bounding_boxes.png", "title": "Bounding Boxes"},
    "coloured_boxes": {"filename": "15-labelled_image_bboxes.png", "title": "Labelled Image with Bounding Boxes"},
}
ALL_STATISTICS_COLUMNS = [
    "Molecule Number",
    "centre_x",
    "centre_y",
    "radius_min",
    "radius_max",
    "radius_mean",
    "radius_median",
    "height_min",
    "height_max",
    "height_median",
    "height_mean",
    "volume",
    "area",
    "area_cartesian_bbox",
    "smallest_bounding_width",
    "smallest_bounding_length",
    "smallest_bounding_area",
    "aspect_ratio",
    "Contour Lengths",
    "Circular",
    "End to End Distance",
    "Image Name",
    "Basename",
]


def create_parser() -> arg.ArgumentParser:
    """Create a parser for reading options."""
    parser = arg.ArgumentParser(
        description="Process AFM images. Additional arguments over-ride those in the configuration file."
    )
    parser.add_argument(
        "-c", "--config_file", dest="config_file", required=True, help="Path to a YAML configuration file."
    )
    parser.add_argument(
        "-b", "--base_dir", dest="base_dir", type=str, required=False, help="Base directory to scan for images."
    )
    parser.add_argument(
        "-f", "--file_ext", dest="file_ext", type=str, required=False, help="File extension to scan for."
    )
    parser.add_argument("--channel", dest="channel", type=str, required=False, help="Channel to extract.")
    parser.add_argument(
        "-o", "--output_dir", dest="output_dir", type=str, required=False, help="Output directory to write results to."
    )
    parser.add_argument(
        "-s", "--save_plots", dest="save_plots", type=bool, required=False, help="Whether to save plots."
    )
    parser.add_argument(
        "--type_of_sample",
        dest="type_of_sample",
        type=str,
        required=False,
        help="Type of sample being processed.",
    )
    parser.add_argument(
        "--minimum_grain_size",
        dest="minimum_grain_size",
        type=str,
        required=False,
        help="Minimum size of grains.",
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
        "--threshold_multiplier",
        dest="threshold_multiplier",
        required=False,
        help="Factor to scale threshold during grain finding.",
    )
    parser.add_argument("-m", "--mask", dest="mask", type=bool, required=False, help="Mask the image.")
    parser.add_argument("-q", "--quiet", dest="quiet", type=bool, required=False, help="Toggle verbosity.")
    parser.add_argument("-w", "--warnings", dest="quiet", type=str, required=False, help="Whether to ignore warnings.")
    return parser


def process_scan(
    image_path: Union[str, Path] = None,
    channel: str = "Height",
    amplify_level: float = 1.0,
    minimum_grain_size: float = 500,
    threshold_method: str = "otsu",
    threshold_multiplier: Union[int, float] = 1.7,
    gaussian_size: Union[int, float] = 2,
    gaussian_mode: str = "nearest",
    background: float = 0.0,
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
    gaussian_mode : str
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
    filtered_image = Filters(image_path, channel, amplify_level, threshold_method, output_dir)
    filtered_image.filter_image()

    # Find Grains :
    try:
        LOGGER.info(f"[{filtered_image.filename}] : Grain Finding")
        grains = Grains(
            image=filtered_image.images["zero_averaged_background"],
            filename=filtered_image.filename,
            pixel_to_nm_scaling=filtered_image.pixel_to_nm_scaling,
            threshold_method=threshold_method,
            gaussian_size=gaussian_size,
            gaussian_mode=gaussian_mode,
            threshold_multiplier=threshold_multiplier,
            background=background,
        )
        grains.find_grains()
    except IndexError as index_exception:
        LOGGER.info(
            f"[{filtered_image.filename}] : No grains were detected, skipping Grain Statistics and DNA Tracing."
        )

    if len(grains.region_properties) > 0:
        # Grain Statistics :
        try:
            LOGGER.info(f"[{filtered_image.filename}] : Grain Statistics")
            grainstats = GrainStats(
                data=grains.images["gaussian_filtered"],
                labelled_data=grains.images["labelled_regions"],
                pixel_to_nanometre_scaling=filtered_image.pixel_to_nm_scaling,
                img_name=filtered_image.filename,
                output_dir=output_dir / filtered_image.filename,
            )
            grain_statistics = grainstats.calculate_stats()

            # Run dnatracing
            LOGGER.info(f"[{filtered_image.filename}] : DNA Tracing")
            dna_traces = dnaTrace(
                full_image_data=grains.images["gaussian_filtered"].T,
                grains=grains.images["labelled_regions"],
                filename=filtered_image.filename,
                pixel_size=filtered_image.pixel_to_nm_scaling,
            )
            dna_traces.trace_dna()

            tracing_stats = traceStats(trace_object=dna_traces, image_path=image_path)
            tracing_stats.save_trace_stats(Path(output_dir) / filtered_image.filename)

            # Combine grainstats and tracingstats
            LOGGER.info(f"[{filtered_image.filename}] : Combining grain statistics and dnatracing statistics")
            results = grain_statistics["statistics"].merge(tracing_stats.df, on="Molecule Number")
            results.to_csv(output_dir / filtered_image.filename / "all_statistics.csv")
        # Possible exceptions
        # ValueError : Arises when lots of small grains with small areas
        except Exception as broad_exception:
            # If no results we need a dummy dataframe to return.
            LOGGER.info("Errors occurred attempting to calculate grain statistics and dnnatracing statistics.")
            results = pd.DataFrame(columns=ALL_STATISTICS_COLUMNS)

    # Optionally plot all stages
    if save_plots:
        LOGGER.info(f"[{filtered_image.filename}] : Plotting Filtering Images")
        # Filtering stage
        for plot_name, array in filtered_image.images.items():
            if plot_name not in ["scan_raw"]:
                if plot_name == "extracted_channel":
                    array = np.flipud(array.pixels)
                PLOT_DICT[plot_name]["output_dir"] = Path(output_dir) / filtered_image.filename
                try:
                    plot_and_save(array, **PLOT_DICT[plot_name])
                except AttributeError:
                    pass
        # Grain stage - only if we have grains
        if len(grains.region_properties) > 0:
            LOGGER.info(f"[{filtered_image.filename}] : Plotting Grain Images")
            for plot_name, array in grains.images.items():
                PLOT_DICT[plot_name]["output_dir"] = Path(output_dir) / filtered_image.filename
                plot_and_save(array, **PLOT_DICT[plot_name])
            # Make a plot of coloured regions with bounding boxes
            plot_and_save(
                grains.images["coloured_regions"],
                Path(output_dir) / filtered_image.filename,
                **PLOT_DICT["bounding_boxes"],
                region_properties=grains.region_properties,
            )
            plot_and_save(
                grains.images["labelled_regions"],
                Path(output_dir) / filtered_image.filename,
                **PLOT_DICT["coloured_boxes"],
                region_properties=grains.region_properties,
            )

    return image_path, results


def main():
    """Run processing."""

    # Parse command line options, load config and update with command line options
    parser = create_parser()
    args = parser.parse_args()
    config = read_yaml(args.config_file)
    config = update_config(config, args)

    LOGGER.info(f"Configuration file loaded from      : {args.config_file}")
    LOGGER.info(f'Scanning for images in              : {config["base_dir"]}')
    LOGGER.info(f'Output directory                    : {config["output_dir"]}')
    LOGGER.info(f'Looking for images with extension   : {config["file_ext"]}')
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
        threshold_method=config["threshold_method"],
        threshold_multiplier=config["grains"]["threshold_multiplier"],
        gaussian_size=config["grains"]["gaussian_size"],
        gaussian_mode=config["grains"]["gaussian_mode"],
        background=config["grains"]["background"],
        output_dir=config["output_dir"],
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
