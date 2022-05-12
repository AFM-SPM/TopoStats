"""Run Topotracing at the command line."""
import argparse as arg
from functools import partial
import logging
from multiprocessing import Pool
from pathlib import Path
from typing import Union, Dict
import warnings

from tqdm import tqdm

from topostats.filters import (
    extract_img_name,
    extract_channel,
    extract_pixels,
    extract_pixel_to_nm_scaling,
    amplify,
    align_rows,
    remove_x_y_tilt,
    average_background,
)
from topostats.find_grains import (
    gaussian_filter,
    tidy_border,
    remove_objects,
    calc_minimum_grain_size,
    label_regions,
    colour_regions,
    region_properties,
    get_bounding_boxes,
    save_region_stats,
)
from topostats.io import read_yaml, load_scan
from topostats.plottingfuncs import plot_and_save
from topostats.logs.logs import LOGGER_NAME
from topostats.utils import convert_path, find_images, update_config, get_mask, get_threshold

LOGGER = logging.getLogger(LOGGER_NAME)


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
        "-o", "--output_dir", dest="output_dir", type=str, required=False, help="Output directory to write results to."
    )
    parser.add_argument("-f", "--file_ext", dest="file_ext", help="File extension to scan for")
    parser.add_argument(
        "-a",
        "--amplify_level",
        dest="amplify_level",
        type=float,
        required=False,
        help="Amplify signals by the given factor.",
    )
    parser.add_argument("-m", "--mask", dest="mask", type=bool, required=False, help="Mask the image.")
    parser.add_argument(
        "-w",
        "--warnings",
        dest="warnings",
        type=str,
        required=False,
        help="Whether to ignore warnings (ignore (default); deprecation).",
    )
    parser.add_argument("-q", "--quiet", dest="quiet", type=bool, required=False, help="Toggle verbosity.")
    return parser


def process_scan(
    image_path: Union[str, Path] = None,
    channel: str = "Height",
    amplify_level: float = 1.0,
    gaussian_size: Union[int, float] = 2,
    mode: str = "nearest",
    threshold_multiplier: Union[int, float] = 1.7,
    background: float = 0.0,
    save_plots: bool = True,
    output_dir: Union[str, Path] = "output",
) -> None:
    """Process a single image, filtering and finding grains.

    Parameters
    ----------
    image_path : Union[str, Path]
        Path to image to process.
    channel : str
        Channel to extract and process, default 'height'.
    amplify_level : float
        Level to amplify image prior to processing by.
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

    # Create output directory
    img_name = extract_img_name(image_path)
    output_dir = Path(output_dir) / f"{img_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(f"Created output directory : {output_dir}")

    # Load image, extract channel and pixel to nm scaling
    image = load_scan(image_path)
    LOGGER.info(f"[{img_name}] : Loaded image.")
    extracted_channel = extract_channel(image, channel)
    LOGGER.info(f"[{img_name}] : Extracted {channel}.")
    pixels = extract_pixels(extracted_channel)
    LOGGER.info(f"[{img_name}] : Pixels extracted.")
    pixel_nm_scaling = extract_pixel_to_nm_scaling(extracted_channel)
    LOGGER.info(f"[{img_name}] : Pixel to nanometre scaling extracted from image : {pixel_nm_scaling}")

    # Amplify image
    if amplify_level != 1.0:
        pixels = amplify(pixels, amplify_level)
        LOGGER.info(f"[{img_name}] : Image amplified by {amplify_level}.")

    # First pass filtering (no mask)
    initial_align = align_rows(pixels, mask=None)
    LOGGER.info(f"[{img_name}] : Initial alignment (unmasked) complete.")
    initial_tilt_removal = remove_x_y_tilt(initial_align, mask=None)
    LOGGER.info(f"[{img_name}] : Initial tilt removal (unmasked) complete.")

    # Create mask
    threshold = get_threshold(initial_tilt_removal)
    mask = get_mask(initial_tilt_removal, threshold)
    LOGGER.info(f"[{img_name}] : Mask created.")

    # Second pass filtering (with mask based on threshold)
    second_align = align_rows(initial_tilt_removal, mask=mask)
    LOGGER.info(f"[{img_name}] : Secondary alignent (masked) complete.")
    second_tilt_removal = remove_x_y_tilt(second_align, mask=mask)
    LOGGER.info(f"[{img_name}] : Secondary tilt removal (masked) complete.")

    # Average background
    averaged_background = average_background(second_tilt_removal, mask=mask)
    LOGGER.info(f"[{img_name}] : Background zero-averaged.")

    # Get threshold
    lower_threshold = get_threshold(averaged_background) * threshold_multiplier

    # Find grains
    # Apply a gaussian filter (using pySPM derived pixel_nm_scaling)
    gaussian_filtered = gaussian_filter(
        averaged_background, gaussian_size=gaussian_size, pixel_to_nm_scaling=pixel_nm_scaling, mode=mode
    )
    LOGGER.info(
        f"[{img_name}] : Gaussian filter applied (size : {gaussian_size}; : Pixel to Nanometer Scaling {pixel_nm_scaling}; mode : {mode})"
    )

    # Create a boolean image
    boolean_image_mask = get_mask(gaussian_filtered, threshold=lower_threshold)

    # Tidy borders
    tidied_borders = tidy_border(boolean_image_mask)
    LOGGER.info(f"[{img_name}] : Borders tidied.")

    # Ge threshold for small objects, need to first label all regions
    # Calculate minimum grain size in pixels
    labelled_regions = label_regions(tidied_borders)
    minimum_grain_size = calc_minimum_grain_size(labelled_regions, background=background)
    LOGGER.info(f"[{img_name}] : Minimum grain size in pixels calculated : {minimum_grain_size}.")

    # Remove objects
    small_objects_removed = remove_objects(
        tidied_borders, minimum_grain_size_pixels=minimum_grain_size, pixel_to_nm_scaling=pixel_nm_scaling
    )
    LOGGER.info(f"[{img_name}] : Small objects (< {minimum_grain_size} pixels) removed.")

    # Label regions after cleaning
    regions_labelled = label_regions(small_objects_removed, background=background)
    LOGGER.info(f"[{img_name}] : Regions labelled.")

    # Colour regions after cleaning
    coloured_regions = colour_regions(regions_labelled)
    LOGGER.info(f"[{img_name}] : Regions coloured.")

    # Extract region properties after cleaning
    image_region_properties = region_properties(regions_labelled)
    LOGGER.info(f"[{img_name}] : Properties extracted for regions.")

    # Derive bounding boxes and save statistics
    bounding_boxes = get_bounding_boxes(image_region_properties)
    LOGGER.info(f"[{img_name}] : Extracted bounding boxes")
    save_region_stats(bounding_boxes, output_dir=output_dir)
    LOGGER.info(f'[{img_name}] : Saved region statistics to {str(output_dir / "grainstats.csv")}')

    # Optionally save images of each stage of processing.
    # Could perhaps improve to make plots either individual or a faceted single image.
    # Also saving arrays to a dictionary and having an associated dictionary with the same keys and values containing
    # filename and title would make this considerably less code.
    if save_plots:
        plot_and_save(pixels, output_dir / "01-raw_heightmap.png", title="Raw Height")
        plot_and_save(initial_align, output_dir / "02-initial_align_unmasked.png", title="Initial Alignment (Unmasked)")
        plot_and_save(
            initial_tilt_removal,
            output_dir / "03-initial_tilt_removal_unmasked.png",
            title="Initial Tilt Removal (Unmasked)",
        )
        plot_and_save(mask, output_dir / "04-binary_mask.png", title="Binary Mask")
        plot_and_save(second_align, output_dir / "05-secondary_align_masked.png", title="Secondary Alignment (Masked)")
        plot_and_save(
            second_tilt_removal,
            output_dir / "06-secondary_tilt_removal_masked.png",
            title="Secondary Tilt Removal (Masked)",
        )
        plot_and_save(
            averaged_background, output_dir / "07-zero_average_background.png", title="Zero Average Background"
        )
        plot_and_save(gaussian_filtered, output_dir / "08-gaussian_filtered.png", title="Gaussian Filtered")
        plot_and_save(boolean_image_mask, output_dir / "09-boolean.png", title="Boolean Mask")
        plot_and_save(tidied_borders, output_dir / "10-tidy_borders.png", title="Tidied Borders")
        plot_and_save(small_objects_removed, output_dir / "11-small_objects_removed.png", title="Small Objects Removed")
        plot_and_save(regions_labelled, output_dir / "12-labelled_regions.png", title="Labelled Regions")
        plot_and_save(coloured_regions, output_dir / "13-coloured_regions.png", title="Coloured Regions")
        plot_and_save(
            coloured_regions,
            output_dir / "14-bounding_boxes.png",
            region_properties=image_region_properties,
            title="Bounding Boxes",
        )


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
    LOGGER.debug(f"Configuration : {config}")

    # Optionally ignore all warnings or just show deprecation warnings
    if config["warnings"] == "ignore":
        warnings.filterwarnings("ignore")
        LOGGER.info("NB : All warnings have been turned off for this run.")
    elif config["warnings"] == "deprecated":

        def fxn():
            warnings.warn("deprecated", DeprecationWarning)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fxn()
    if config["quiet"]:
        LOGGER.setLevel("ERROR")

    # For debugging (as Pool makes it hard to find things when they go wrong)
    # for x in img_files:
    #     process_scan(image_path=x,
    #                  amplify_level=config['amplify_level'],
    #                  channel=config['channel'],
    #                  gaussian_size=config['grains']['gaussian_size'],
    #                  dx=config['grains']['dx'],
    #                  mode=config['grains']['mode'],
    #                  lower_threshold_otsu_multiplier=config['grains']['lower_threshold_otsu_multiplier'],
    #                  minimum_grain_size=config['grains']['minimum_grain_size'],
    #                  background=config['grains']['background'],
    #                  save_plots=config['save_plots'],
    #                  output_dir=config['output_dir'])

    # Process all images found in parallel (constrainged by 'cores' option in config).
    processing_function = partial(
        process_scan,
        amplify_level=config["amplify_level"],
        channel=config["channel"],
        gaussian_size=config["grains"]["gaussian_size"],
        mode=config["grains"]["mode"],
        threshold_multiplier=config["grains"]["threshold_multiplier"],
        background=config["grains"]["background"],
        save_plots=config["save_plots"],
        output_dir=config["output_dir"],
    )

    with Pool(processes=config["cores"]) as pool:
        with tqdm(
            total=len(img_files),
            desc=f'Processing {len(img_files)} images from {config["base_dir"]}, results are under {config["output_dir"]}',
        ) as pbar:
            for _ in pool.imap_unordered(processing_function, img_files):
                pbar.update()


if __name__ == "__main__":
    main()
