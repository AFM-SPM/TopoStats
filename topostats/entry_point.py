"""
Entry point for all TopoStats programs.

Parses command-line arguments and passes input on to the relevant functions / modules.
"""

import argparse as arg
import sys
from pathlib import Path

from topostats import __version__, run_modules
from topostats.io import write_config_with_comments
from topostats.plotting import run_toposum

# pylint: disable=too-many-lines
# pylint: disable=too-many-statements


def create_parser() -> arg.ArgumentParser:
    """
    Create a parser for reading options.

    Creates a parser, with multiple sub-parsers for reading options to run 'topostats'.

    Returns
    -------
    arg.ArgumentParser
        Argument parser.
    """
    parser = arg.ArgumentParser(
        description="Run various programs relating to AFM data. Add the name of the program you wish to run."
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"Installed version of TopoStats: {__version__}",
        help="Report the current version of TopoStats that is installed",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        dest="config_file",
        type=Path,
        required=False,
        help="Path to a YAML configuration file.",
    )
    parser.add_argument(
        "-s",
        "--summary-config",
        dest="summary_config",
        required=False,
        help="Path to a YAML configuration file for summary plots and statistics.",
    )
    parser.add_argument(
        "--matplotlibrc",
        dest="matplotlibrc",
        type=Path,
        required=False,
        help="Path to a matplotlibrc file.",
    )
    parser.add_argument(
        "-b",
        "--base-dir",
        dest="base_dir",
        type=Path,
        required=False,
        help="Base directory to scan for images.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        type=Path,
        required=False,
        help="Output directory to write results to.",
    )
    parser.add_argument(
        "-l",
        "--log-level",
        dest="log_level",
        type=str,
        required=False,
        help="Logging level to use, default is 'info' for verbose output use 'debug'.",
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
        "--file-ext",
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
        "--image-set",
        dest="image_set",
        type=str,
        required=False,
        help="Image set to generate, default is 'core' other option is 'all'.",
    )

    subparsers = parser.add_subparsers(title="program", description="Available programs, listed below:", dest="program")

    # Create a sub-parsers for different stages of processing and tasks
    process_parser = subparsers.add_parser(
        "process",
        description="Process AFM images. Additional arguments over-ride defaults or those in the configuration file.",
        help="Process AFM images. Additional arguments over-ride defaults or those in the configuration file.",
    )
    # Filter options
    process_parser.add_argument(
        "--filter-row-alignment-quantile",
        dest="filter_row_alignment_quantile",
        type=float,
        required=False,
        help="Lower values may improve flattening of larger features.",
    )
    process_parser.add_argument(
        "--filter-threshold-method",
        dest="filter_threshold_method",
        type=str,
        required=False,
        help="Method for thresholding Filtering. Options are otsu, std_dev, absolute.",
    )
    process_parser.add_argument(
        "--filter-otsu-threshold-multiplier",
        dest="filter_otsu_threshold_multiplier",
        type=float,
        required=False,
        help="Factor for scaling the Otsu threshold during Filtering.",
    )
    process_parser.add_argument(
        "--filter-threshold-std-dev-below",
        dest="filter_threshold_std_dev_below",
        type=float,
        required=False,
        help="Threshold for data below the image background for std dev method during Filtering.",
    )
    process_parser.add_argument(
        "--filter-threshold-std-dev-above",
        dest="filter_threshold_std_dev_above",
        type=float,
        required=False,
        help="Threshold for data above the image background for std dev method during Filtering.",
    )
    process_parser.add_argument(
        "--filter-threshold-absolute-below",
        dest="filter_threshold_absolute_below",
        type=float,
        required=False,
        help="Threshold for data below the image bacnground dor absolute method during Filtering",
    )
    process_parser.add_argument(
        "--filter-threshold-absolute-above",
        dest="filter_threshold_absolute_above",
        type=float,
        required=False,
        help="Threshold for data above the image bacnground dor absolute method during Filtering",
    )
    process_parser.add_argument(
        "--filter-gaussian-size",
        dest="filter_gaussian_size",
        type=float,
        required=False,
        help="Gaussian blur intensity in pixels.",
    )
    process_parser.add_argument(
        "--filter-gaussian-mode",
        dest="filter_gaussian_mode",
        type=str,
        required=False,
        help="Gaussian blur method. Options are 'nearest' (default), 'reflect', 'constant', 'mirror' or 'wrap'.",
    )
    process_parser.add_argument(
        "--filter-remove-scars", dest="filter_scars_run", type=bool, required=False, help="Whether to remove scars."
    )
    process_parser.add_argument(
        "--filter-scars-removal-iterations",
        dest="filter_scars_removal_iterations",
        type=int,
        required=False,
        help="Number of times to run scar removal",
    )
    process_parser.add_argument(
        "--filter-scars-threshold-low",
        dest="filter_scars_threshold_low",
        type=float,
        required=False,
        help="Lower values make scar removal more sensitive",
    )
    process_parser.add_argument(
        "--filter-scars-threshold-high",
        dest="filter_scars_threshold_high",
        type=float,
        required=False,
        help="Lower values make scar removal more sensitive",
    )
    process_parser.add_argument(
        "--filter-scars-max-scar-width",
        dest="filter_scars_max_scar_width",
        type=int,
        required=False,
        help="Maximum thickness of scars in pixels",
    )
    process_parser.add_argument(
        "--filter-scars-max-scar-length",
        dest="filter_scars_max_scar_length",
        type=int,
        required=False,
        help="Maximum length of scars in pixels",
    )

    # Grains
    process_parser.add_argument(
        "--grains-threshold-method",
        dest="grains_threshold_method",
        type=str,
        required=False,
        help="Method for thresholding Grain finding. Options are otsu, std_dev, absolute.",
    )
    process_parser.add_argument(
        "--grains-otsu-threshold-multiplier",
        dest="grains_otsu_threshold_multiplier",
        type=float,
        required=False,
        help="Factor for scaling the Otsu threshold during Grain finding.",
    )
    process_parser.add_argument(
        "--grains-threshold-std-dev-below",
        dest="grains_threshold_std_dev_below",
        type=float,
        required=False,
        help="Threshold for data below the image background for std dev method during Grain finding.",
    )
    process_parser.add_argument(
        "--grains-threshold-std-dev-above",
        dest="grains_threshold_std_dev_above",
        type=float,
        required=False,
        help="Threshold for data above the image background for std dev method during Grain finding.",
    )
    process_parser.add_argument(
        "--grains-threshold-absolute-below",
        dest="grains_threshold_absolute_below",
        type=float,
        required=False,
        help="Threshold for data below the image bacnground dor absolute method during Grain finding",
    )
    process_parser.add_argument(
        "--grains-threshold-absolute-above",
        dest="grains_threshold_absolute_above",
        type=float,
        required=False,
        help="Threshold for data above the image bacnground dor absolute method during Grain finding",
    )
    process_parser.add_argument(
        "--grains-direction",
        dest="grains_direction",
        type=str,
        required=False,
        help="Whether to look for grains 'above' pr 'below' thresholds of 'both'",
    )
    process_parser.add_argument(
        "--grains-smallest-grain-size-nm2",
        dest="grains_smallest_grain_size_nm2",
        type=float,
        required=False,
        help="Size in nm^2 of tiny grains/blobs to remove, must be > 0.0",
    )
    process_parser.add_argument(
        "--grains-absolute-area-threshold-above",
        dest="grains_absolute_area_threshold_above",
        type=float,
        required=False,
        nargs=2,
        help="Above surface (low, high) in nm^2, both low and high should be specified",
    )
    process_parser.add_argument(
        "--grains-absolute-area-threshold-below",
        dest="grains_absolute_area_threshold_below",
        type=float,
        required=False,
        nargs=2,
        help="Below surface (low, high) in nm^2, both low and high should be specified",
    )
    process_parser.add_argument(
        "--grains-remove-edge-intersecting-grains",
        dest="grains_remove_edge_intersecting_grains",
        type=bool,
        required=False,
        help="Whether or not to remove grains that touch the image border",
    )
    # Unet
    process_parser.add_argument(
        "--unet-model-path", dest="unet_model_path", type=Path, required=False, help="Path to a trained U-net model"
    )
    process_parser.add_argument(
        "--unet-grain-crop-padding",
        dest="unet_grain_crop_padding",
        type=int,
        required=False,
        help="Padding to apply to the grain crop bounding box",
    )
    process_parser.add_argument(
        "--unet-upper-norm-bound",
        dest="unet_upper_norm_bound",
        type=float,
        required=False,
        help="Upper bound for normalisation of input data. This should be slightly higher than the maximum desired"
        "/expected height of grains",
    )
    process_parser.add_argument(
        "--unet-lower-norm-bound",
        dest="unet_lower_norm_bound",
        type=float,
        required=False,
        help="Lower bound for normalisation of input data. This should be slightly lower than the minimum desired"
        "/expected height of the background",
    )

    # Grainstats
    process_parser.add_argument(
        "--grainstats-edge-detection-method",
        dest="grainstats_edge_detection_method",
        type=str,
        required=False,
        help="Method of edge detection, do NOT change this unless you are sure of what it will do. Options 'canny' and"
        "'binary_erosion (default)",
    )
    process_parser.add_argument(
        "--grainstats-cropped-size",
        dest="grainstats_cropped_size",
        type=float,
        required=False,
        help="Length (in nm) of square cropped images (can take -1 for grain-sized box)",
    )
    process_parser.add_argument(
        "--grainstats-extract-height-profile",
        dest="grainstats_extract_height_profile",
        type=bool,
        required=False,
        help="Extract height profiles along maximum feret of molecules",
    )

    # Disordered Tracing
    process_parser.add_argument(
        "--disordered-min-skeleton-size",
        dest="disordered_min_skeleton_size",
        type=float,
        required=False,
        help="Minimum number of pixels in a skeleton for it to be retained",
    )
    process_parser.add_argument(
        "--disordered-pad-width",
        dest="disordered_pad_width",
        type=int,
        required=False,
        help="Pixels to pad grains by when tracing",
    )
    process_parser.add_argument(
        "--disordered-mask-smoothing-params-gaussian-sigma",
        dest="disordered_mask_smoothing_params_gaussian_sigma",
        type=float,
        required=False,
        help="Gaussian smoothing parameter 'sigma' in pixels",
    )
    process_parser.add_argument(
        "--disordered-mask-smoothing-params-dilation-iterations",
        dest="disordered_mask_smoothing_params_dilation_iterations",
        type=int,
        required=False,
        help="Number of dilation iterations to use for grain smoothing",
    )
    process_parser.add_argument(
        "--disordered-mask-smoothing-params-holearea-min-max",
        dest="disordered_mask_smoothing_params_holearea_min_max",
        type=float,
        required=False,
        nargs=2,
        help="Range (min, max) of a hole area in nm to refil in the smoothed masks",
    )
    process_parser.add_argument(
        "--disordered-skeletonisation-params-method",
        dest="disordered_skeletonisation_params_method",
        type=str,
        required=False,
        help="Skeletonisation method. Options : zhang | lee | thin | topostats",
    )
    process_parser.add_argument(
        "--disordered-skeletonisation-height-bias",
        dest="disordered_skeletonisation_height_bias",
        type=float,
        required=False,
        help="Percentage of lowest pixels to remove each skeletonisation iteration. 1.0 equates to Zhang method",
    )
    process_parser.add_argument(
        "--disordered-pruning-params-method",
        dest="disordered_pruning_params_method",
        type=str,
        required=False,
        help="Method to clean branches from the skeleton: Options : 'topostats'",
    )
    process_parser.add_argument(
        "--disordered-pruning-params-max-length",
        dest="disordered_pruning_params_max_length",
        type=float,
        required=False,
        help="Maximum length in nm to remove a branch containing an endpoint",
    )
    process_parser.add_argument(
        "--disordered-pruning-params-height-threshold",
        dest="disordered_pruning_params_height_threshold",
        type=float,
        required=False,
        help="The height to remove branches below",
    )
    process_parser.add_argument(
        "--disordered-pruning-params-method-values",
        dest="disordered_pruning_params_method_values",
        type=str,
        required=False,
        help="Method to obtain a branch's height for pruning. Options : 'min' | 'median', 'mid'",
    )
    process_parser.add_argument(
        "--disordered-pruning-params-method-outlier",
        dest="disordered_pruning_params_method_outlier",
        type=str,
        required=False,
        help="Method to prune branches based on height. Options : 'abs' | 'mean_abs' | 'iqr'",
    )

    # Nodestats
    process_parser.add_argument(
        "--nodestats-node-joining-length",
        dest="nodestats_node_joining_length",
        type=float,
        required=False,
        help="The distance over which to join nearby crossing points",
    )
    process_parser.add_argument(
        "--nodestats-node-extend-dist",
        dest="nodestats_node_extend_dist",
        type=float,
        required=False,
        help="The distance over which to join nearby odd-branched nodes",
    )
    process_parser.add_argument(
        "--nodestats-branch-pairing-length",
        dest="nodestats_branch_pairing_length",
        type=float,
        required=False,
        help="The length from the crossing point to pair and trace, obtaining FWHM's",
    )
    process_parser.add_argument(
        "--nodestats-pair-odd-branches",
        dest="nodestats_pair_odd_branches",
        type=bool,
        required=False,
        help="Whether to try and pair odd-branched nodes",
    )
    process_parser.add_argument(
        "--nodestats-pad-width",
        dest="nodestats_pad_width",
        type=int,
        required=False,
        help="Pixels to pad grains by when tracing (should be the same as --disordered-pad-width)",
    )

    # Ordered Tracing
    process_parser.add_argument(
        "--ordered-ordering-method",
        dest="ordered_ordering_method",
        type=str,
        required=False,
        help="Ordering method for ordering disordered traces. Option 'nodestats'",
    )
    process_parser.add_argument(
        "--ordered-pad-width",
        dest="ordered_pad_width",
        type=int,
        required=False,
        help="Pixels to pad grains by when tracing (should be the same as --disordered-pad-width)",
    )

    # Splining
    process_parser.add_argument(
        "--splining-method",
        dest="splining_method",
        type=str,
        required=False,
        help="Method for splining. Options 'spline' or 'rolling_window",
    )
    process_parser.add_argument(
        "--splining-window-size",
        dest="splining_window_size",
        type=float,
        required=False,
        help="Size in nm of the rolling window",
    )
    process_parser.add_argument(
        "--splining-step-size",
        dest="splining_step_size",
        type=float,
        required=False,
        help="The sampling rate of the spline in metres",
    )
    process_parser.add_argument(
        "--splining-linear-smoothing",
        dest="splining_linear_smoothing",
        type=float,
        required=False,
        help="The amount of smoothing to apply to linear features",
    )
    process_parser.add_argument(
        "--splining-circular-smoothing",
        dest="splining_circular_smoothing",
        type=float,
        required=False,
        help="The amount of smoothing to apply to circular features",
    )
    process_parser.add_argument(
        "--splining-degree",
        dest="splining_degree",
        type=int,
        required=False,
        help="The polynomial degree of the spline",
    )

    # Plotting
    process_parser.add_argument(
        "--save-plots",
        dest="save_plots",
        type=bool,
        required=False,
        help="Whether to save plots.",
    )
    process_parser.add_argument(
        "--savefig-format",
        dest="savefig_format",
        type=str,
        required=False,
        help="Format for saving figures to, options are 'png', 'svg', or other valid Matplotlib supported formats.",
    )
    process_parser.add_argument(
        "--savefig-dpi",
        dest="savefig_dpi",
        type=int,
        required=False,
        help="Dots Per Inch for plots, should be integer for dots per inch.",
    )
    process_parser.add_argument(
        "--cmap",
        dest="cmap",
        type=str,
        required=False,
        help="Colormap to use, options include 'nanoscope', 'afmhot' or any valid Matplotlib colormap.",
    )
    process_parser.add_argument("-m", "--mask", dest="mask", type=bool, required=False, help="Mask the image.")
    process_parser.add_argument(
        "-w",
        "--warnings",
        dest="warnings",
        type=bool,
        required=False,
        help="Whether to ignore warnings.",
    )
    # Run the relevant function with the arguments
    process_parser.set_defaults(func=run_modules.process)

    load_parser = subparsers.add_parser(
        "load",
        description="Load and save all images as .topostats files for subsequent processing.",
        help="Load and save all images as .topostats files for subsequent processing.",
    )
    # Run the relevant function with the arguments
    load_parser.set_defaults(func=run_modules.process)

    # Filter
    filter_parser = subparsers.add_parser(
        "filter",
        description="WIP DO NOT USE - Load and filter images, saving as .topostats files for subsequent processing.",
        help="WIP DO NOT USE - Load and filter images, saving as .topostats files for subsequent processing.",
    )
    filter_parser.add_argument(
        "--row-alignment-quantile",
        dest="row_alignment_quantile",
        type=float,
        required=False,
        help="Lower values may improve flattening of larger features.",
    )
    filter_parser.add_argument(
        "--threshold-method",
        dest="threshold_method",
        type=str,
        required=False,
        help="Method for thresholding Filtering. Options are otsu, std_dev, absolute.",
    )
    filter_parser.add_argument(
        "--otsu-threshold-multiplier",
        dest="otsu_threshold_multiplier",
        type=float,
        required=False,
        help="Factor for scaling the Otsu threshold during Filtering.",
    )
    filter_parser.add_argument(
        "--threshold-std-dev-below",
        dest="threshold_std_dev_below",
        type=float,
        required=False,
        help="Threshold for data below the image background for std dev method during Filtering.",
    )
    filter_parser.add_argument(
        "--threshold-std-dev-above",
        dest="threshold_std_dev_above",
        type=float,
        required=False,
        help="Threshold for data above the image background for std dev method during Filtering.",
    )
    filter_parser.add_argument(
        "--threshold-absolute-below",
        dest="threshold_absolute_below",
        type=float,
        required=False,
        help="Threshold for data below the image bacnground dor absolute method during Filtering",
    )
    filter_parser.add_argument(
        "--threshold-absolute-above",
        dest="threshold_absolute_above",
        type=float,
        required=False,
        help="Threshold for data above the image bacnground dor absolute method during Filtering",
    )
    filter_parser.add_argument(
        "--gaussian-size", dest="gaussian_size", type=float, required=False, help="Gaussian blur intensity in pixels."
    )
    filter_parser.add_argument(
        "--gaussian-mode",
        dest="gaussian_mode",
        type=str,
        required=False,
        help="Gaussian blur method. Options are 'nearest' (default), 'reflect', 'constant', 'mirror' or 'wrap'.",
    )
    filter_parser.add_argument(
        "--remove-scars", dest="scars_run", type=bool, required=False, help="Whether to remove scars."
    )
    filter_parser.add_argument(
        "--scars-removal-iterations",
        dest="scars_removal_iterations",
        type=int,
        required=False,
        help="Number of times to run scar removal",
    )
    filter_parser.add_argument(
        "--scars-threshold-low",
        dest="scars_threshold_low",
        type=float,
        required=False,
        help="Lower values make scar removal more sensitive",
    )
    filter_parser.add_argument(
        "--scars-threshold-high",
        dest="scars_threshold_high",
        type=float,
        required=False,
        help="Lower values make scar removal more sensitive",
    )
    filter_parser.add_argument(
        "--scars-max-scar-width",
        dest="scars_max_scar_width",
        type=int,
        required=False,
        help="Maximum thickness of scars in pixels",
    )
    filter_parser.add_argument(
        "--scars-max-scar-length",
        dest="scars_max_scar_length",
        type=int,
        required=False,
        help="Maximum length of scars in pixels",
    )
    # Run the relevant function with the arguments
    filter_parser.set_defaults(func=run_modules.filters)

    grains_parser = subparsers.add_parser(
        "grains",
        description="WIP DO NOT USE - Load filtered images from '.topostats' files and detect grains.",
        help="WIP DO NOT USE - Load filtered images from '.topostats' files and detect grains.",
    )
    grains_parser.add_argument(
        "--threshold-method",
        dest="threshold_method",
        type=str,
        required=False,
        help="Method for thresholding Grain finding. Options are otsu, std_dev, absolute.",
    )
    grains_parser.add_argument(
        "--otsu-threshold-multiplier",
        dest="otsu_threshold_multiplier",
        type=float,
        required=False,
        help="Factor for scaling the Otsu threshold during Grain finding.",
    )
    grains_parser.add_argument(
        "--threshold-std-dev-below",
        dest="threshold_std_dev_below",
        type=float,
        required=False,
        help="Threshold for data below the image background for std dev method during Grain finding.",
    )
    grains_parser.add_argument(
        "--threshold-std-dev-above",
        dest="threshold_std_dev_above",
        type=float,
        required=False,
        help="Threshold for data above the image background for std dev method during Grain finding.",
    )
    grains_parser.add_argument(
        "--threshold-absolute-below",
        dest="threshold_absolute_below",
        type=float,
        required=False,
        help="Threshold for data below the image bacnground dor absolute method during Grain finding",
    )
    grains_parser.add_argument(
        "--threshold-absolute-above",
        dest="threshold_absolute_above",
        type=float,
        required=False,
        help="Threshold for data above the image bacnground dor absolute method during Grain finding",
    )
    grains_parser.add_argument(
        "--direction",
        dest="direction",
        type=str,
        required=False,
        help="Whether to look for grains 'above' pr 'below' thresholds of 'both'",
    )
    grains_parser.add_argument(
        "--smallest-grain-size-nm2",
        dest="smallest_grain_size_nm2",
        type=float,
        required=False,
        help="Size in nm^2 of tiny grains/blobs to remove, must be > 0.0",
    )
    grains_parser.add_argument(
        "--absolute-area-threshold-above",
        dest="absolute_area_threshold_above",
        type=float,
        required=False,
        nargs=2,
        help="Above surface (low, high) in nm^2, both low and high should be specified",
    )
    grains_parser.add_argument(
        "--absolute-area-threshold-below",
        dest="absolute_area_threshold_below",
        type=float,
        required=False,
        nargs=2,
        help="Below surface (low, high) in nm^2, both low and high should be specified",
    )
    grains_parser.add_argument(
        "--remove-edge-intersecting-grains",
        dest="remove_edge_intersecting_grains",
        type=bool,
        required=False,
        help="Whether or not to remove grains that touch the image border",
    )
    # Unet
    grains_parser.add_argument(
        "--unet-model-path", dest="unet_model_path", type=Path, required=False, help="Path to a trained U-net model"
    )
    grains_parser.add_argument(
        "--unet-grain-crop-padding",
        dest="unet_grain_crop_padding",
        type=int,
        required=False,
        help="Padding to apply to the grain crop bounding box",
    )
    grains_parser.add_argument(
        "--unet-upper-norm-bound",
        dest="unet_upper_norm_bound",
        type=float,
        required=False,
        help="Upper bound for normalisation of input data. This should be slightly higher than the maximum desired"
        "/expected height of grains",
    )
    grains_parser.add_argument(
        "--unet-lower-norm-bound",
        dest="unet_lower_norm_bound",
        type=float,
        required=False,
        help="Lower bound for normalisation of input data. This should be slightly lower than the minimum desired"
        "/expected height of the background",
    )
    # Run the relevant function with the arguments
    grains_parser.set_defaults(func=run_modules.grains)

    grainstats_parser = subparsers.add_parser(
        "grainstats",
        description="WIP DO NOT USE - Load images with grains from '.topostats' files and calculate statistics.",
        help="WIP DO NOT USE - Load images with grains from '.topostats' files and calculate statistics.",
    )
    grainstats_parser.add_argument(
        "--edge-detection-method",
        dest="edge_detection_method",
        type=str,
        required=False,
        help="Method of edge detection, do NOT change this unless you are sure of what it will do. Options 'canny' and"
        "'binary_erosion (default)",
    )
    grainstats_parser.add_argument(
        "--cropped-size",
        dest="cropped_size",
        type=float,
        required=False,
        help="Length (in nm) of square cropped images (can take -1 for grain-sized box)",
    )
    grainstats_parser.add_argument(
        "--extract-height-profile",
        dest="extract_height_profile",
        type=bool,
        required=False,
        help="Extract height profiles along maximum feret of molecules",
    )
    # Run the relevant function with the arguments
    grainstats_parser.set_defaults(func=run_modules.grainstats)

    # Disordered
    disordered_tracing_parser = subparsers.add_parser(
        "disordered-tracing",
        description="WIP DO NOT USE - Skeletonise and prune objects to disordered traces.",
        help="WIP DO NOT USE - Skeletonise and prune objects to disordered traces.",
    )
    disordered_tracing_parser.add_argument(
        "--min-skeleton-size",
        dest="min_skeleton_size",
        type=float,
        required=False,
        help="Minimum number of pixels in a skeleton for it to be retained",
    )
    disordered_tracing_parser.add_argument(
        "--pad-width",
        dest="pad_width",
        type=int,
        required=False,
        help="Pixels to pad grains by when tracing",
    )
    disordered_tracing_parser.add_argument(
        "--mask-smoothing-params-gaussian-sigma",
        dest="mask_smoothing_params_gaussian_sigma",
        type=float,
        required=False,
        help="Gaussian smoothing parameter 'sigma' in pixels",
    )
    disordered_tracing_parser.add_argument(
        "--mask-smoothing-params-dilation-iterations",
        dest="mask_smoothing_params_dilation_iterations",
        type=int,
        required=False,
        help="Number of dilation iterations to use for grain smoothing",
    )
    disordered_tracing_parser.add_argument(
        "--mask-smoothing-params-holearea-min-max",
        dest="mask_smoothing_params_holearea_min_max",
        type=float,
        required=False,
        nargs=2,
        help="Range (min, max) of a hole area in nm to refil in the smoothed masks",
    )
    disordered_tracing_parser.add_argument(
        "--skeletonisation-params-method",
        dest="skeletonisation_params_method",
        type=str,
        required=False,
        help="Skeletonisation method. Options : zhang | lee | thin | topostats",
    )
    disordered_tracing_parser.add_argument(
        "--skeletonisation-height-bias",
        dest="skeletonisation_height_bias",
        type=float,
        required=False,
        help="Percentage of lowest pixels to remove each skeletonisation iteration. 1.0 equates to Zhang method",
    )
    disordered_tracing_parser.add_argument(
        "--pruning-params-method",
        dest="pruning_params_method",
        type=str,
        required=False,
        help="Method to clean branches from the skeleton: Options : 'topostats'",
    )
    disordered_tracing_parser.add_argument(
        "--pruning-params-max-length",
        dest="pruning_params_max_length",
        type=float,
        required=False,
        help="Maximum length in nm to remove a branch containing an endpoint",
    )
    disordered_tracing_parser.add_argument(
        "--pruning-params-height-threshold",
        dest="pruning_params_height_threshold",
        type=float,
        required=False,
        help="The height to remove branches below",
    )
    disordered_tracing_parser.add_argument(
        "--pruning-params-method-values",
        dest="pruning_params_method_values",
        type=str,
        required=False,
        help="Method to obtain a branch's height for pruning. Options : 'min' | 'median' | 'mid'",
    )
    disordered_tracing_parser.add_argument(
        "--pruning-params-method-outlier",
        dest="pruning_params_method_outlier",
        type=str,
        required=False,
        help="Method to prune branches based on height. Options : 'abs' | 'mean_abs' | 'iqr'",
    )
    # Run the relevant function with the arguments
    disordered_tracing_parser.set_defaults(func=run_modules.disordered_tracing)

    # Nodestats
    nodestats_parser = subparsers.add_parser(
        "nodestats",
        description="WIP DO NOT USE - Calculate node statistics and disentangle molecules.",
        help="WIP DO NOT USE - Calculate node statistics and disentangle molecules.",
    )
    nodestats_parser.add_argument(
        "--node-joining-length",
        dest="node_joining_length",
        type=float,
        required=False,
        help="The distance over which to join nearby crossing points",
    )
    nodestats_parser.add_argument(
        "--node-extend-dist",
        dest="node_extend_dist",
        type=float,
        required=False,
        help="The distance over which to join nearby odd-branched nodes",
    )
    nodestats_parser.add_argument(
        "--branch-pairing-length",
        dest="branch_pairing_length",
        type=float,
        required=False,
        help="The length from the crossing point to pair and trace, obtaining FWHM's",
    )
    nodestats_parser.add_argument(
        "--pair-odd-branches",
        dest="pair_odd_branches",
        type=bool,
        required=False,
        help="Whether to try and pair odd-branched nodes",
    )
    nodestats_parser.add_argument(
        "--pad-width",
        dest="pad_width",
        type=int,
        required=False,
        help="Pixels to pad grains by when tracing (should be the same as --disordered-pad-width)",
    )
    # Run the relevant function with the arguments
    nodestats_parser.set_defaults(func=run_modules.nodestats)

    # Ordered Tracing
    ordered_tracing_parser = subparsers.add_parser(
        "ordered-tracing",
        description="WIP DO NOT USE - Ordered traces of pruned skeletons.",
        help="WIP DO NOT USE - Ordered traces of pruned skeletons.",
    )
    ordered_tracing_parser.add_argument(
        "--ordering-method",
        dest="ordering_method",
        type=str,
        required=False,
        help="Ordering method for ordering disordered traces. Option 'nodestats'",
    )
    ordered_tracing_parser.add_argument(
        "--pad-width",
        dest="pad_width",
        type=int,
        required=False,
        help="Pixels to pad grains by when tracing (should be the same as --disordered-pad-width)",
    )
    # Run the relevant function with the arguments
    ordered_tracing_parser.set_defaults(func=run_modules.ordered_tracing)

    # Splining
    splining_parser = subparsers.add_parser(
        "splining",
        description="WIP DO NOT USE - Splining of traced molecules to produce smooth curves.",
        help="WIP DO NOT USE - Splining of traced molecules to produce smooth curves.",
    )
    splining_parser.add_argument(
        "--method",
        dest="method",
        type=str,
        required=False,
        help="Method for splining. Options 'spline' or 'rolling_window",
    )
    splining_parser.add_argument(
        "--window-size",
        dest="window_size",
        type=float,
        required=False,
        help="Size in nm of the rolling window",
    )
    splining_parser.add_argument(
        "--step-size",
        dest="step_size",
        type=float,
        required=False,
        help="The sampling rate of the spline in metres",
    )
    splining_parser.add_argument(
        "--linear-smoothing",
        dest="linear_smoothing",
        type=float,
        required=False,
        help="The amount of smoothing to apply to linear features",
    )
    splining_parser.add_argument(
        "--circular-smoothing",
        dest="circular_smoothing",
        type=float,
        required=False,
        help="The amount of smoothing to apply to circular features",
    )
    splining_parser.add_argument(
        "--degree",
        dest="degree",
        type=int,
        required=False,
        help="The polynomial degree of the spline",
    )
    # Run the relevant function with the arguments
    splining_parser.set_defaults(func=run_modules.splining)

    summary_parser = subparsers.add_parser(
        "summary",
        description="Plotting and summary of TopoStats output statistics.",
        help="Plotting and summary of TopoStats output statistics.",
    )
    summary_parser.add_argument(
        "--input-csv",
        dest="input_csv",
        type=Path,
        required=False,
        help="Path to CSV file to plot.",
    )
    summary_parser.add_argument(
        "--config-file",
        dest="config_file",
        type=Path,
        required=False,
        help="Path to a YAML plotting dictionary that maps variable names to labels.",
    )
    summary_parser.add_argument(
        "--var-to-label",
        dest="var_to_label",
        type=Path,
        required=False,
        help="Path to a YAML plotting dictionary that maps variable names to labels.",
    )
    summary_parser.add_argument(
        "--create-config-file",
        dest="create_config_file",
        type=Path,
        required=False,
        help="Filename to write a sample YAML configuration file to (should end in '.yaml').",
    )
    summary_parser.add_argument(
        "--create-label-file",
        dest="create_label_file",
        type=Path,
        required=False,
        help="Filename to write a sample YAML label file to (should end in '.yaml').",
    )
    summary_parser.add_argument(
        "--savefig-format",
        dest="savefig_format",
        type=str,
        required=False,
        help="Format for saving figures to, options are 'png', 'svg', or other valid Matplotlib supported formats.",
    )
    summary_parser.set_defaults(func=run_toposum)

    create_config_parser = subparsers.add_parser(
        "create-config",
        description="Create a configuration file using the defaults.",
        help="Create a configuration file using the defaults.",
    )
    create_config_parser.add_argument(
        "-f",
        "--filename",
        dest="filename",
        type=Path,
        required=False,
        default="config.yaml",
        help="Name of YAML file to save configuration to (default 'config.yaml').",
    )
    create_config_parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        type=Path,
        required=False,
        default="./",
        help="Path to where the YAML file should be saved (default './' the current directory).",
    )
    create_config_parser.add_argument(
        "-c",
        "--config",
        dest="config",
        type=str,
        default=None,
        help="Configuration to use, currently only one is supported, the 'default'.",
    )
    create_config_parser.add_argument(
        "-s",
        "--simple",
        dest="simple",
        action="store_true",
        help="Create a simple configuration file with only the most common options.",
    )
    create_config_parser.set_defaults(func=write_config_with_comments)

    create_matplotlibrc_parser = subparsers.add_parser(
        "create-matplotlibrc",
        description="Create a Matplotlibrc parameters file.",
        help="Create a Matplotlibrc parameters file using the defaults.",
    )
    create_matplotlibrc_parser.add_argument(
        "-f",
        "--filename",
        dest="filename",
        type=Path,
        required=False,
        default="topostats.mplstyle",
        help="Name of file to save Matplotlibrc configuration to (default 'topostats.mplstyle').",
    )
    create_matplotlibrc_parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        type=Path,
        required=False,
        default="./",
        help="Path to where the YAML file should be saved (default './' the current directory).",
    )
    create_matplotlibrc_parser.add_argument(
        "-c",
        "--config",
        dest="config",
        default="topostats.mplstyle",
        help="Matplotlibrc style file to use, currently only one is supported, the 'topostats.mplstyle'.",
    )
    create_matplotlibrc_parser.set_defaults(func=write_config_with_comments)

    return parser


def entry_point(manually_provided_args=None, testing=False) -> None:
    """
    Entry point for all TopoStats programs.

    Main entry point for running 'topostats' which allows the different processing steps ('process', 'filter',
    'create_config' etc.) to be run.

    Parameters
    ----------
    manually_provided_args : None
        Manually provided arguments.
    testing : bool
        Whether testing is being carried out.

    Returns
    -------
    None
        Does not return anything.
    """
    # Parse command line options, load config (or default) and update with command line options
    parser = create_parser()
    args = parser.parse_args() if manually_provided_args is None else parser.parse_args(manually_provided_args)

    # No program specified, print help and exit
    if not args.program:
        parser.print_help()
        sys.exit()

    if testing:
        return args

    # call the relevant function
    args.func(args)

    return None
