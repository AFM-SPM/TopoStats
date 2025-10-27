"""Validation of configuration."""

import logging
import os
from pathlib import Path

from schema import And, Optional, Or, Schema, SchemaError

from topostats.logs.logs import LOGGER_NAME

LOGGER = logging.getLogger(LOGGER_NAME)

# pylint: disable=line-too-long
# pylint: disable=too-many-lines


def validate_config(config: dict, schema: Schema, config_type: str) -> None:
    """
    Validate configuration.

    Parameters
    ----------
    config : dict
        Config dictionary imported by read_yaml() and parsed through clean_config().
    schema : Schema
        A schema against which the configuration is to be compared.
    config_type : str
        Description of of configuration being validated.
    """
    try:
        schema.validate(config)
        LOGGER.info(f"The {config_type} is valid.")
    except SchemaError as schema_error:
        raise SchemaError(
            f"There is an error in your {config_type} configuration. "
            "Please refer to the first error message above for details"
        ) from schema_error


DEFAULT_CONFIG_SCHEMA = Schema(
    {
        "base_dir": Path,
        "output_dir": Path,
        "log_level": Or(
            "debug",
            "info",
            "warning",
            "error",
            error="Invalid value in config for 'log_level', valid values are 'info' (default), 'debug', 'error' or 'warning",
        ),
        "cores": lambda n: 1 <= n <= os.cpu_count(),
        "file_ext": Or(
            ".spm",
            ".asd",
            ".jpk",
            ".ibw",
            ".gwy",
            ".topostats",
            ".stp",
            ".top",
            error="Invalid value in config for 'file_ext', valid values are '.spm', '.jpk', '.ibw', '.gwy', '.topostats', or '.asd'.",
        ),
        "output_stats": Or(
            "basic", "full", error="Invalid value for 'output_stats', valid values are 'full' or 'basic'."
        ),
        "loading": {
            "channel": str,
            "extract": Or(
                "all",
                "raw",
                "filters",
                error="Invalid value in config for 'extract', valid values are 'all', 'raw' or 'filters'",
            ),
        },
        "filter": {
            "run": Or(
                True,
                False,
                error="Invalid value in config for 'filter.run', valid values are 'True' or 'False'",
            ),
            "row_alignment_quantile": lambda n: 0.0 <= n <= 1.0,
            "threshold_method": Or(
                "absolute",
                "otsu",
                "std_dev",
                error=(
                    "Invalid value in config for 'filter.threshold_method', valid values "
                    "are 'absolute', 'otsu' or 'std_dev'"
                ),
            ),
            "otsu_threshold_multiplier": float,
            "threshold_std_dev": {
                "below": lambda n: n > 0,
                "above": lambda n: n > 0,
            },
            "threshold_absolute": {
                "below": Or(
                    int,
                    float,
                    error=("Invalid value in config for filter.threshold.absolute.below should be type int or float"),
                ),
                "above": Or(
                    int,
                    float,
                    error=("Invalid value in config for filter.threshold.absolute.below should be type int or float"),
                ),
            },
            "gaussian_size": float,
            "gaussian_mode": Or(
                "nearest",
                error="Invalid value in config for 'filter.gaussian_mode', valid values are 'nearest'",
            ),
            "remove_scars": {
                "run": bool,
                "removal_iterations": lambda n: 0 <= n < 10,
                "threshold_low": lambda n: n > 0,
                "threshold_high": lambda n: n > 0,
                "max_scar_width": lambda n: n >= 1,
                "min_scar_length": lambda n: n >= 1,
            },
        },
        "grains": {
            "run": Or(
                True,
                False,
                error="Invalid value in config for grains.run, valid values are 'True' or 'False'",
            ),
            "grain_crop_padding": int,
            "threshold_method": Or(
                "absolute",
                "otsu",
                "std_dev",
                error=(
                    "Invalid value in config for 'grains.threshold_method', valid values "
                    "are 'absolute', 'otsu' or 'std_dev'"
                ),
            ),
            "otsu_threshold_multiplier": float,
            "threshold_std_dev": {
                "below": [lambda n: n > 0],
                "above": [
                    lambda n: n > 0,
                ],
            },
            "threshold_absolute": {
                "below": [
                    Or(
                        int,
                        float,
                        error=(
                            "Invalid value in config for grains.threshold.absolute.below should be type int or float"
                        ),
                    )
                ],
                "above": [
                    Or(
                        int,
                        float,
                        error=(
                            "Invalid value in config for grains.threshold.absolute.above should be type int or float"
                        ),
                    )
                ],
            },
            "area_thresholds": {
                "above": [
                    Or(
                        int,
                        None,
                        error=(
                            "Invalid value in config for 'grains.area_thresholds.above', valid values are int or null"
                        ),
                    )
                ],
                "below": [
                    Or(
                        int,
                        None,
                        error=(
                            "Invalid value in config for 'grains.area_thresholds.below', valid values are int or null"
                        ),
                    )
                ],
            },
            "direction": Or(
                "both",
                "below",
                "above",
                error="Invalid direction for grains.direction valid values are 'both', 'below' or 'above",
            ),
            "remove_edge_intersecting_grains": Or(
                True,
                False,
                error="Invalid value in config for 'grains.remove_edge_intersecting_grains', valid values are 'True' or 'False'",
            ),
            "unet_config": {
                "model_path": Or(None, str),
                "upper_norm_bound": float,
                "lower_norm_bound": float,
                "remove_disconnected_grains": bool,
                "confidence": float,
            },
            "vetting": {
                "whole_grain_size_thresholds": Or(
                    None,
                    [lambda n: n > 0, lambda n: n > 0],
                    error="Invalid value in config for 'grainstats.vetting.whole_grain_size_thresholds', this needs to"
                    "be a list of two positive floats",
                ),
                "class_conversion_size_thresholds": Or(
                    None,
                    # List of lists of 3 integers and 2 integers
                    [[[int, int, int], [int, int]]],
                    error="Invalid value in config for 'grainstats.vetting.class_conversion_size_thresholds', this "
                    "needs to be a list of lists of 3 integers and 2 integers - [class number, class to convert to "
                    "if too small, class to convert to if too big] and [minimum size, maximum size]",
                ),
                "class_region_number_thresholds": Or(
                    None,
                    # List of lists of 3 integers
                    [[int, int, int]],
                    error="Invalid value in config for 'grainstats.vetting.class_region_number_thresholds', this needs"
                    "to be a list of lists of 3 integers - class number, minimum region number, maximum region number",
                ),
                "class_size_thresholds": Or(
                    None,
                    # List of lists of 3 integers
                    [[int, int, int]],
                    error="Invalid value in config for 'grainstats.vetting.class_size_thresholds', this needs to be a"
                    "list of lists of 3 integers - class number, minimum size, maximum size",
                ),
                "nearby_conversion_classes_to_convert": Or(
                    None,
                    # List of tuples of two integers
                    [[int, int]],
                    error="Invalid value in config for 'grainstats.vetting.nearby_conversion_classes_to_convert', this"
                    "needs to be a list of tuples of two integers",
                ),
                "class_touching_threshold": lambda n: n >= 0,
                "keep_largest_labelled_regions_classes": Or(
                    None,
                    # List of integers
                    [int],
                    error="Invalid value in config for 'grainstats.vetting.keep_largest_labelled_regions_classes', this"
                    "needs to be a list of integers",
                ),
                "class_connection_point_thresholds": Or(
                    None,
                    # List of tuples: list[tuple[[tuple[int, int], tuple[int, int]]]
                    [[[int, int], [int, int]]],
                    error="Invalid value in config for 'grainstats.vetting.class_connection_point_thresholds', this"
                    "needs to be a list of tuples of two tuples of two integers. Eg [((1, 2), (3, 4))]",
                ),
            },
            "classes_to_merge": Or(
                None,
                [int],
                error="Invalid value in config for 'grains.classes_to_merge', this needs to be a list of integers.",
            ),
        },
        "grainstats": {
            "run": Or(
                True,
                False,
                error="Invalid value in config for 'grainstats.run', valid values are 'True' or 'False'",
            ),
            "edge_detection_method": Or(
                "binary_erosion",
                "canny",
            ),
            "extract_height_profile": Or(
                True,
                False,
                error="Invalid value in config for 'grainstats.extract_height_profile',"
                "valid values are 'True' or 'False'",
            ),
            "class_names": [str],
        },
        "disordered_tracing": {
            "run": Or(
                True,
                False,
                error="Invalid value in config for 'disordered_tracing.run', valid values are 'True' or 'False'",
            ),
            "class_index": int,
            "min_skeleton_size": lambda n: n > 0.0,
            "mask_smoothing_params": {
                "gaussian_sigma": Or(
                    float,
                    int,
                    None,
                ),
                "dilation_iterations": Or(
                    int,
                    None,
                ),
                "holearea_min_max": [
                    Or(
                        int,
                        float,
                        None,
                        error=(
                            "Invalid value in config for 'disordered_tracing.mask_smoothing_params.holearea_min_max', valid values "
                            "are int, float or null"
                        ),
                    ),
                ],
            },
            "skeletonisation_params": {
                "method": Or(
                    "zhang",
                    "lee",
                    "thin",
                    "medial_axis",
                    "topostats",
                    error="Invalid value in config for 'disordered_tracing.skeletonisation_method',"
                    "valid values are 'zhang', 'lee', 'thin', 'medial_axis', 'topostats'",
                ),
                "height_bias": lambda n: 0 < n <= 1,
            },
            "pruning_params": {
                "method": Or(
                    "topostats",
                    error="Invalid value in config for 'disordered_tracing.pruning_method', valid values are 'topostats'",
                ),
                "max_length": lambda n: n >= 0,
                "method_values": Or("min", "median", "mid"),
                "method_outlier": Or("abs", "mean_abs", "iqr"),
                "height_threshold": Or(int, float, None),
                "only_height_prune_endpoints": Or(
                    True,
                    False,
                    error="Invalid value in config for 'disordered_tracing.pruning_params.run', valid values are 'True' or 'False'",
                ),
            },
        },
        "nodestats": {
            "run": Or(
                True,
                False,
                error="Invalid value in config for 'nodestats.run', valid values are 'True' or 'False'",
            ),
            "node_joining_length": float,
            "node_extend_dist": float,
            "branch_pairing_length": float,
            "pair_odd_branches": bool,
        },
        "ordered_tracing": {
            "run": Or(
                True,
                False,
                error="Invalid value in config for 'ordered_tracing.run', valid values are 'True' or 'False'",
            ),
            "ordering_method": Or(
                "nodestats",
                "original",
                error="Invalid value in config for 'ordered_tracing.ordering_method', valid values are 'nodestats' or 'original'",
            ),
        },
        "splining": {
            "run": Or(
                True,
                False,
                error="Invalid value in config for 'splining.run', valid values are 'True' or 'False'",
            ),
            "method": Or(
                "spline",
                "rolling_window",
                error="Invalid value in config for 'splining.method', valid values are 'spline' or 'rolling_window'",
            ),
            "rolling_window_size": lambda n: n > 0.0,
            "rolling_window_resampling": bool,
            "rolling_window_resample_regular_spatial_interval": lambda n: n > 0.0,
            "spline_step_size": lambda n: n > 0.0,
            "spline_linear_smoothing": lambda n: n >= 0.0,
            "spline_circular_smoothing": lambda n: n >= 0.0,
            "spline_degree": int,
            # "cores": lambda n: n > 0.0,
        },
        "curvature": {
            "run": Or(
                True,
                False,
                error="Invalid value in config for 'curvature.run', valid values are 'True' or 'False'",
            ),
            "colourmap_normalisation_bounds": [
                Or(
                    float,
                    int,
                    error="Invalid value in config for 'curvature.colourmap_normalisation_bounds', valid values"
                    "are float or int",
                )
            ],
        },
        "plotting": {
            "run": Or(
                True,
                False,
                error="Invalid value in config for 'plotting.run', valid values are 'True' or 'False'",
            ),
            "style": And(
                str,
                Or(
                    "topostats.mplstyle",
                    str,
                    Path,
                    None,
                    error="Invalid value in config for 'plotting.style', valid values are 'topostats.mplstyle' or None",
                ),
            ),
            "savefig_format": Or(
                None,
                str,
                error="Invalid value in config for plotting.savefig_formatmust be a value supported by Matplotlib.",
            ),
            "savefig_dpi": Or(
                None,
                "figure",
                lambda n: n > 0,
                error="Invalid value in config for plotting.savefig_dpi, validvalues are 'figure' or floats",
            ),
            "image_set": [
                Or(
                    "all",
                    "core",
                    "filters",
                    "grains",
                    "grain_crops",
                    "disordered_tracing",
                    "nodestats",
                    "ordered_tracing",
                    "splining",
                    error="Invalid value in config for 'plotting.image_set', valid values are 'all' or 'core'",
                )
            ],
            "pixel_interpolation": Or(
                None,
                "none",
                "bessel",
                "bicubic",
                "bilinear",
                "catrom",
                "gaussian",
                "hamming",
                "hanning",
                "hermite",
                "kaiser",
                "lanczos",
                "mitchell",
                "nearest",
                "quadric",
                "sinc",
                "spline16",
                "spline36",
                error="Invalid interpolation value. See https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html for options.",
            ),
            "grain_crop_plot_size_nm": Or(
                int,
                float,
                error=("Invalid value in config for 'grains.grain_crop_plot_size_nm', valid values are int or float"),
            ),
            "zrange": [float, int, None],
            "colorbar": Or(
                True,
                False,
                error="Invalid value in config for 'plotting.colorbar', valid values are 'True' or 'False'",
            ),
            "axes": Or(
                True,
                False,
                error="Invalid value in config plotting.for 'axes', valid values are 'True' or 'False'",
            ),
            "num_ticks": Or(
                [None, And(int, lambda n: n > 1)],
                error="Invalid value in config plotting.for 'num_ticks', valid values are 'null' or integers > 1",
            ),
            "cmap": Or(
                None,
                str,
                error="Invalid value in config for 'plotting.cmap', valid values are 'afmhot', 'nanoscope', "
                "'gwyddion' or values supported by Matplotlib",
            ),
            "mask_cmap": str,
            "histogram_log_axis": Or(
                True,
                False,
                error=(
                    "Invalid value in config plotting histogram. For 'log_y_axis', valid values are 'True' or 'False'"
                ),
            ),
            "number_grains": Or(
                True,
                False,
                error=("Invalid value in config for 'plotting.number_grain_plots', valid values are 'True' or 'False'"),
            ),
        },
        "summary_stats": {
            "run": Or(
                True,
                False,
                error="Invalid value in config for summary_stats.run, valid values are 'True' or 'False'",
            ),
            "config": Or(
                None,
                str,
                error=(
                    "Invalid value in config for summary_stats.config, valid values are 'None' or a path to a "
                    "config file."
                ),
            ),
        },
    }
)


PLOTTING_SCHEMA = Schema(
    {
        "extracted_channel": {
            "filename": str,
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=(
                    "Invalid value in config 'extracted_channel.image_type', valid values are 'binary' or 'non-binary'"
                ),
            ),
            "core_set": bool,
            "savefig_dpi": Or(
                lambda n: n > 0,
                "figure",
                error="Invalid value in config for 'dpi', valid values are 'figure' or > 0.",
            ),
            "module": Or(
                "filters",
                "grains",
                "grain_crops",
                "disordered_tracing",
                "nodestats",
                "ordered_tracing",
                "splining",
                error=(
                    "Invalid value in config 'extracted_channel.module', valid values "
                    "are 'filters', 'grains', 'grain_crops', 'disordered_tracing', "
                    "'nodestats', 'ordered_tracing', or 'splining'"
                ),
            ),
        },
        "pixels": {
            "filename": str,
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error="Invalid value in config 'pixels.image_type', valid values are 'binary' or 'non-binary'",
            ),
            "core_set": bool,
            "savefig_dpi": Or(
                lambda n: n > 0,
                "figure",
                error="Invalid value in config for 'dpi', valid values are 'figure' or > 0.",
            ),
            "module": Or(
                "filters",
                "grains",
                "grain_crops",
                "disordered_tracing",
                "nodestats",
                "ordered_tracing",
                "splining",
                error=(
                    "Invalid value in config 'pixels.module', valid values "
                    "are 'filters', 'grains', 'grain_crops', 'disordered_tracing', "
                    "'nodestats', 'ordered_tracing', or 'splining'"
                ),
            ),
        },
        "initial_median_flatten": {
            "filename": str,
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=(
                    "Invalid value in config 'initial_median_flatten.image_type', valid values "
                    "are 'binary' or 'non-binary'"
                ),
            ),
            "core_set": bool,
            "savefig_dpi": Or(
                lambda n: n > 0,
                "figure",
                error="Invalid value in config for 'dpi', valid values are 'figure' or > 0.",
            ),
            "module": Or(
                "filters",
                "grains",
                "grain_crops",
                "disordered_tracing",
                "nodestats",
                "ordered_tracing",
                "splining",
                error=(
                    "Invalid value in config 'initial_median_flatten.module', valid values "
                    "are 'filters', 'grains', 'grain_crops', 'disordered_tracing', "
                    "'nodestats', 'ordered_tracing', or 'splining'"
                ),
            ),
        },
        "initial_tilt_removal": {
            "filename": str,
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=(
                    "Invalid value in config 'initial_tilt_removal.image_type', valid values "
                    "are 'binary' or 'non-binary'"
                ),
            ),
            "core_set": bool,
            "savefig_dpi": Or(
                lambda n: n > 0,
                "figure",
                error="Invalid value in config for 'dpi', valid values are 'figure' or > 0.",
            ),
            "module": Or(
                "filters",
                "grains",
                "grain_crops",
                "disordered_tracing",
                "nodestats",
                "ordered_tracing",
                "splining",
                error=(
                    "Invalid value in config 'initial_tilt_removal.module', valid values "
                    "are 'filters', 'grains', 'grain_crops', 'disordered_tracing', "
                    "'nodestats', 'ordered_tracing', or 'splining'"
                ),
            ),
        },
        "initial_quadratic_removal": {
            "filename": str,
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=(
                    "Invalid value in config 'initial_quadratic_removal.image_type', valid values "
                    "are 'binary' or 'non-binary'"
                ),
            ),
            "core_set": bool,
            "savefig_dpi": Or(
                lambda n: n > 0,
                "figure",
                error="Invalid value in config for 'dpi', valid values are 'figure' or > 0.",
            ),
            "module": "filters",
        },
        "initial_scar_removal": {
            "filename": str,
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=(
                    "Invalid value in config 'initial_scar_removal.image_type', valid values "
                    "are 'binary' or 'non-binary'"
                ),
            ),
            "core_set": bool,
            "savefig_dpi": Or(
                lambda n: n > 0,
                "figure",
                error="Invalid value in config for 'dpi', valid values are 'figure' or > 0.",
            ),
            "module": "filters",
        },
        "initial_zero_average_background": {
            "filename": str,
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=(
                    "Invalid value in config 'initial_zero_average_background.image_type', valid values "
                    "are 'binary' or 'non-binary'"
                ),
            ),
            "core_set": bool,
            "savefig_dpi": Or(
                lambda n: n > 0,
                "figure",
                error="Invalid value in config for 'dpi', valid values are 'figure' or > 0.",
            ),
            "module": "filters",
        },
        "initial_nonlinear_polynomial_removal": {
            "filename": str,
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=(
                    "Invalid value in config 'initial_nonlinear_polynomial_removal.image_type', valid values "
                    "are 'binary' or 'non-binary'"
                ),
            ),
            "core_set": bool,
            "savefig_dpi": Or(
                lambda n: n > 0,
                "figure",
                error="Invalid value in config for 'dpi', valid values are 'figure' or > 0.",
            ),
            "module": "filters",
        },
        "mask": {
            "filename": str,
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error="Invalid value in config 'mask.image_type', valid values are 'binary' or 'non-binary'",
            ),
            "core_set": bool,
            "savefig_dpi": Or(
                lambda n: n > 0,
                "figure",
                error="Invalid value in config for 'dpi', valid values are 'figure' or > 0.",
            ),
            "module": "filters",
        },
        "masked_median_flatten": {
            "filename": str,
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=(
                    "Invalid value in config 'masked_median_flatten.image_type', valid values "
                    "are 'binary' or 'non-binary'"
                ),
            ),
            "core_set": bool,
            "savefig_dpi": Or(
                lambda n: n > 0,
                "figure",
                error="Invalid value in config for 'dpi', valid values are 'figure' or > 0.",
            ),
            "module": "filters",
        },
        "masked_tilt_removal": {
            "filename": str,
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=(
                    "Invalid value in config 'masked_tilt_removal.image_type', valid values "
                    "are 'binary' or 'non-binary'"
                ),
            ),
            "core_set": bool,
            "savefig_dpi": Or(
                lambda n: n > 0,
                "figure",
                error="Invalid value in config for 'dpi', valid values are 'figure' or > 0.",
            ),
            "module": "filters",
        },
        "masked_quadratic_removal": {
            "filename": str,
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=(
                    "Invalid value in config 'masked_quadratic_removal.image_type', valid values "
                    "are 'binary' or 'non-binary'"
                ),
            ),
            "core_set": bool,
            "savefig_dpi": Or(
                lambda n: n > 0,
                "figure",
                error="Invalid value in config for 'dpi', valid values are 'figure' or > 0.",
            ),
            "module": "filters",
        },
        "masked_nonlinear_polynomial_removal": {
            "filename": str,
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=(
                    "Invalid value in config 'masked_nonlinear_polynomial_removal.image_type', valid values "
                    "are 'binary' or 'non-binary'"
                ),
            ),
            "core_set": bool,
            "savefig_dpi": Or(
                lambda n: n > 0,
                "figure",
                error="Invalid value in config for 'dpi', valid values are 'figure' or > 0.",
            ),
            "module": "filters",
        },
        "secondary_scar_removal": {
            "filename": str,
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=(
                    "Invalid value in config 'masked_quadratic_removal.image_type', valid values "
                    "are 'binary' or 'non-binary'"
                ),
            ),
            "core_set": bool,
            "savefig_dpi": Or(
                lambda n: n > 0,
                "figure",
                error="Invalid value in config for 'dpi', valid values are 'figure' or > 0.",
            ),
            "module": "filters",
        },
        "scar_mask": {
            "filename": str,
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=(
                    "Invalid value in config 'masked_quadratic_removal.image_type', valid values "
                    "are 'binary' or 'non-binary'"
                ),
            ),
            "core_set": bool,
            "savefig_dpi": Or(
                lambda n: n > 0,
                "figure",
                error="Invalid value in config for 'dpi', valid values are 'figure' or > 0.",
            ),
            "module": "filters",
        },
        "final_zero_average_background": {
            "filename": str,
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=(
                    "Invalid value in config 'final_zero_average_background.image_type', valid values "
                    "are 'binary' or 'non-binary'"
                ),
            ),
            "core_set": bool,
            "savefig_dpi": Or(
                lambda n: n > 0,
                "figure",
                error="Invalid value in config for 'dpi', valid values are 'figure' or > 0.",
            ),
            "module": "filters",
        },
        "gaussian_filtered": {
            "filename": str,
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=(
                    "Invalid value in config 'gaussian_filtered.image_type', valid values are 'binary' or 'non-binary'"
                ),
            ),
            "core_set": bool,
            "savefig_dpi": Or(
                lambda n: n > 0,
                "figure",
                error="Invalid value in config for 'dpi', valid values are 'figure' or > 0.",
            ),
            "module": "filters",
        },
        "z_threshed": {
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=("Invalid value in config 'z_threshold.image_type', valid values are 'binary' or 'non-binary'"),
            ),
            "core_set": True,
            "savefig_dpi": Or(
                lambda n: n > 0,
                "figure",
                error="Invalid value in config for 'dpi', valid values are 'figure' or > 0.",
            ),
            "module": "filters",
        },
        "thresholded_grains": {
            "filename": str,
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=("Invalid value in config 'mask_grains.image_type', valid values are 'binary' or 'non-binary'"),
            ),
            "core_set": bool,
            "savefig_dpi": Or(
                lambda n: n > 0,
                "figure",
                error="Invalid value in config for 'dpi', valid values are 'figure' or > 0.",
            ),
            "mask_cmap": str,
            "module": "grains",
        },
        "tidied_border": {
            "filename": str,
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=("Invalid value in config 'tidied_border.image_type', valid values are 'binary' or 'non-binary'"),
            ),
            "core_set": bool,
            "savefig_dpi": Or(
                lambda n: n > 0,
                "figure",
                error="Invalid value in config for 'dpi', valid values are 'figure' or > 0.",
            ),
            "mask_cmap": str,
            "module": "grains",
        },
        "removed_objects_too_small_to_process": {
            "filename": str,
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=(
                    "Invalid value in config 'removed_small_objects.image_type', valid values "
                    "are 'binary' or 'non-binary'"
                ),
            ),
            "mask_cmap": str,
            "core_set": bool,
            "savefig_dpi": Or(
                lambda n: n > 0,
                "figure",
                error="Invalid value in config for 'dpi', valid values are 'figure' or > 0.",
            ),
            "module": "grains",
        },
        "area_thresholded": {
            "filename": str,
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=(
                    "Invalid value in config 'removed_objects_too_small_to_process.image_type', valid values "
                    "are 'binary' or 'non-binary'"
                ),
            ),
            "mask_cmap": str,
            "core_set": bool,
            "savefig_dpi": Or(
                lambda n: n > 0,
                "figure",
                error="Invalid value in config for 'dpi', valid values are 'figure' or > 0.",
            ),
            "module": "grains",
        },
        "unet": {
            "filename": str,
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=("Invalid value in config 'unet_tensor.image_type', valid values are 'binary' or 'non-binary'"),
            ),
            "core_set": bool,
            "savefig_dpi": Or(
                lambda n: n > 0,
                "figure",
                error="Invalid value in config for 'dpi', valid values are 'figure' or > 0.",
            ),
            "module": "grains",
        },
        "vetted": {
            "filename": str,
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=("Invalid value in config 'vetted_tensor.image_type', valid values are 'binary' or 'non-binary'"),
            ),
            "core_set": bool,
            "savefig_dpi": Or(
                lambda n: n > 0,
                "figure",
                error="Invalid value in config for 'dpi', valid values are 'figure' or > 0.",
            ),
            "module": "grains",
        },
        "merged_classes": {
            "filename": str,
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=(
                    "Invalid value in config 'merged_classes_tensor.image_type', valid values "
                    "are 'binary' or 'non-binary'"
                ),
            ),
            "core_set": bool,
            "savefig_dpi": Or(
                lambda n: n > 0,
                "figure",
                error="Invalid value in config for 'dpi', valid values are 'figure' or > 0.",
            ),
            "module": "grains",
        },
        "mask_overlay": {
            "filename": str,
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=("Invalid value in config 'mask_overlay.image_type', valid values are 'binary' or 'non-binary'"),
            ),
            "core_set": True,
            "savefig_dpi": Or(
                lambda n: n > 0,
                "figure",
                error="Invalid value in config for 'dpi', valid values are 'figure' or > 0.",
            ),
            "module": "grains",
        },
        "bounding_boxes": {
            "filename": str,
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=(
                    "Invalid value in config 'bounding_boxes.image_type', valid values are 'binary' or 'non-binary'"
                ),
            ),
            "core_set": bool,
            "savefig_dpi": Or(
                lambda n: n > 0,
                "figure",
                error="Invalid value in config for 'dpi', valid values are 'figure' or > 0.",
            ),
            "module": "grains",
        },
        "grain_image": {
            "image_type": Or(
                "binary",
                "non-binary",
                error=("Invalid value in config 'grain_image.image_type', valid values are 'binary' or 'non-binary'"),
            ),
            "core_set": False,
            "savefig_dpi": Or(
                lambda n: n > 0,
                "figure",
                error="Invalid value in config for 'dpi', valid values are 'figure' or > 0.",
            ),
            "module": "grain_crops",
        },
        "grain_mask": {
            "image_type": Or(
                "binary",
                "non-binary",
                error=("Invalid value in config 'grain_mask.image_type', valid values are 'binary' or 'non-binary'"),
            ),
            "core_set": bool,
            "savefig_dpi": Or(
                lambda n: n > 0,
                "figure",
                error="Invalid value in config for 'dpi', valid values are 'figure' or > 0.",
            ),
            "module": "grain_crops",
        },
        "grain_mask_image": {
            "image_type": Or(
                "binary",
                "non-binary",
                error=(
                    "Invalid value in config 'grain_mask_image.image_type', valid values are 'binary' or 'non-binary'"
                ),
            ),
            "core_set": bool,
            "savefig_dpi": Or(
                lambda n: n > 0,
                "figure",
                error="Invalid value in config for 'dpi', valid values are 'figure' or > 0.",
            ),
            "module": "grain_crops",
        },
        "orig_grain": {
            "filename": str,
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=(
                    "Invalid value in config 'coloured_boxes.image_type', valid values are 'binary' or 'non-binary'"
                ),
            ),
            "mask_cmap": str,
            "core_set": bool,
            "module": "disordered_tracing",
        },
        "smoothed_grain": {
            "filename": str,
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=(
                    "Invalid value in config 'coloured_boxes.image_type', valid values are 'binary' or 'non-binary'"
                ),
            ),
            "mask_cmap": str,
            "core_set": bool,
            "module": "disordered_tracing",
        },
        "skeleton": {
            "filename": str,
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=(
                    "Invalid value in config 'coloured_boxes.image_type', valid values are 'binary' or 'non-binary'"
                ),
            ),
            "mask_cmap": str,
            "core_set": bool,
            "savefig_dpi": int,
            "module": "disordered_tracing",
        },
        "pruned_skeleton": {
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=(
                    "Invalid value in config 'coloured_boxes.image_type', valid values are 'binary' or 'non-binary'"
                ),
            ),
            "mask_cmap": str,
            "core_set": bool,
            "savefig_dpi": int,
            "module": "disordered_tracing",
        },
        "branch_indexes": {
            "filename": str,
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=(
                    "Invalid value in config 'branch_indexes.image_type', valid values are 'binary' or 'non-binary'"
                ),
            ),
            "mask_cmap": str,
            "core_set": bool,
            "savefig_dpi": int,
            "module": "disordered_tracing",
        },
        "branch_types": {
            "filename": str,
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=(
                    "Invalid value in config 'coloured_boxes.image_type', valid values are 'binary' or 'non-binary'"
                ),
            ),
            "mask_cmap": str,
            "core_set": bool,
            "savefig_dpi": int,
            "module": "disordered_tracing",
        },
        "convolved_skeletons": {
            "filename": str,
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=(
                    "Invalid value in config 'convolved_skeleton.image_type', valid values are 'binary' or 'non-binary'"
                ),
            ),
            "mask_cmap": str,
            "core_set": bool,
            "savefig_dpi": int,
            "module": "nodestats",
        },
        "node_centres": {
            "filename": str,
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=("Invalid value in config 'node_centres.image_type', valid values are 'binary' or 'non-binary'"),
            ),
            "mask_cmap": str,
            "core_set": bool,
            "savefig_dpi": int,
            "module": "nodestats",
        },
        "connected_nodes": {
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=(
                    "Invalid value in config 'node_branch_mask.image_type', valid values are 'binary' or 'non-binary'"
                ),
            ),
            "mask_cmap": str,
            "core_set": bool,
            "savefig_dpi": int,
            "module": "nodestats",
        },
        "node_area_skeleton": {
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=(
                    "Invalid value in config 'node_area_skeleton.image_type', valid values are 'binary' or 'non-binary'"
                ),
            ),
            "mask_cmap": str,
            "core_set": bool,
            "savefig_dpi": int,
            "module": "nodestats",
        },
        "node_branch_mask": {
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=(
                    "Invalid value in config 'node_branch_mask.image_type', valid values are 'binary' or 'non-binary'"
                ),
            ),
            "mask_cmap": str,
            "core_set": bool,
            "savefig_dpi": int,
            "module": "nodestats",
        },
        "node_avg_mask": {
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=("Invalid value in config 'node_avg_mask.image_type', valid values are 'binary' or 'non-binary'"),
            ),
            "mask_cmap": str,
            "core_set": bool,
            "module": "nodestats",
        },
        "node_line_trace": {
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=(
                    "Invalid value in config 'coloured_boxes.image_type', valid values are 'binary' or 'non-binary'"
                ),
            ),
            "mask_cmap": str,
            "core_set": bool,
            "module": "nodestats",
        },
        "ordered_traces": {
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=(
                    "Invalid value in config 'all_molecule_traces.image_type', valid values "
                    "are 'binary' or 'non-binary'"
                ),
            ),
            "mask_cmap": str,
            "core_set": bool,
            "savefig_dpi": Or(
                lambda n: n > 0,
                "figure",
                error="Invalid value in config for 'dpi', valid values are 'figure' or > 0.",
            ),
            "module": "ordered_tracing",
        },
        "trace_segments": {
            "filename": str,
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=(
                    "Invalid value in config 'all_molecule_traces.image_type', valid values "
                    "are 'binary' or 'non-binary'"
                ),
            ),
            "mask_cmap": str,
            "core_set": bool,
            "savefig_dpi": Or(
                lambda n: n > 0,
                "figure",
                error="Invalid value in config for 'dpi', valid values are 'figure' or > 0.",
            ),
            "module": "ordered_tracing",
        },
        "over_under": {
            "filename": str,
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=(
                    "Invalid value in config 'all_molecule_traces.image_type', valid values "
                    "are 'binary' or 'non-binary'"
                ),
            ),
            "mask_cmap": str,
            "core_set": bool,
            "savefig_dpi": Or(
                lambda n: n > 0,
                "figure",
                error="Invalid value in config for 'dpi', valid values are 'figure' or > 0.",
            ),
            "module": "ordered_tracing",
        },
        "all_molecules": {
            "filename": str,
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=(
                    "Invalid value in config 'coloured_boxes.image_type', valid values are 'binary' or 'non-binary'"
                ),
            ),
            "mask_cmap": str,
            "core_set": bool,
            "savefig_dpi": int,
            "module": "ordered_tracing",
        },
        "fitted_trace": {
            "filename": str,
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=(
                    "Invalid value in config 'coloured_boxes.image_type', valid values are 'binary' or 'non-binary'"
                ),
            ),
            "mask_cmap": str,
            "core_set": bool,
            "savefig_dpi": int,
            "module": "ordered_tracing",
        },
        "splined_trace": {
            "image_type": Or(
                "binary",
                "non-binary",
                error=("Invalid value in config 'splined_trace.image_type', valid values are 'binary' or 'non-binary'"),
            ),
            "title": str,
            "core_set": bool,
            "savefig_dpi": Or(
                lambda n: n > 0,
                "figure",
                error="Invalid value in config for 'dpi', valid values are 'figure' or > 0.",
            ),
            "module": "splining",
        },
        "curvature": {
            "image_type": Or(
                "binary",
                "non-binary",
                error=("Invalid value in config 'curvature.image_type', valid values are 'binary' or 'non-binary'"),
            ),
            "title": str,
            "core_set": bool,
            "savefig_dpi": Or(
                lambda n: n > 0,
                "figure",
                error="Invalid value in config for 'dpi', valid values are 'figure' or > 0.",
            ),
            "module": "splining",
        },
        "curvature_individual_grains": {
            "filename": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=(
                    "Invalid value in config 'curvature_individual_grains.image_type', valid values "
                    "are 'binary' or 'non-binary'"
                ),
            ),
            "title": str,
            "core_set": bool,
            "savefig_dpi": Or(
                lambda n: n > 0,
                "figure",
                error="Invalid value in config for 'dpi', valid values are 'figure' or > 0.",
            ),
            "module": "splining",
        },
    }
)

SUMMARY_SCHEMA = Schema(
    {
        "base_dir": Path,
        "output_dir": Path,
        "csv_file": str,
        "savefig_format": Or(
            "png",
            "pdf",
            "svg",
            "tiff",
            "tif",
            error=("Invalid value in config 'savefig_format', valid values are 'png', 'pdf', 'svg', 'tiff' or 'tif'"),
        ),
        "var_to_label": Or(
            None,
            str,
            error="Invalid value in config for 'var_to_label', valid values are 'None' or a str",
        ),
        "image_id": str,
        "molecule_id": str,
        "hist": Or(
            True,
            False,
            error="Invalid value in config for 'hist', valid values are 'True' or 'False'",
        ),
        "bins": lambda n: n > 0,
        "stat": Or(
            "count",
            "frequency",
            "probability",
            "percent",
            "density",
            error=(
                "Invalid value in config 'stat', valid values are 'count', 'frequency', "
                "'probability', 'percent' or 'density'"
            ),
        ),
        "kde": Or(
            True,
            False,
            error="Invalid value in config for 'kde', valid values are 'True' or 'False'",
        ),
        "violin": Or(
            True,
            False,
            error="Invalid value in config for 'violin', valid values are 'True' or 'False'",
        ),
        "figsize": [lambda n: n > 0],
        "alpha": lambda n: n > 0,
        "palette": Or(
            "colorblind",
            "deep",
            "muted",
            "pastel",
            "bright",
            "dark",
            "Spectral",
            "Set2",
            error=(
                "Invalid value in config 'palette', valid values are 'colorblind', 'deep', "
                "'muted', 'pastel', 'bright', 'dark', 'Spectral' or 'Set2'"
            ),
        ),
        "stats_to_sum": [
            Optional("area"),
            Optional("area_cartesian_bbox"),
            Optional("aspect_ratio"),
            Optional("bending_angle"),
            Optional("total_contour_length"),
            Optional("average_end_to_end_distance"),
            Optional("height_max"),
            Optional("height_mean"),
            Optional("height_median"),
            Optional("height_min"),
            Optional("max_feret"),
            Optional("min_feret"),
            Optional("radius_max"),
            Optional("radius_mean"),
            Optional("radius_median"),
            Optional("radius_min"),
            Optional("smallest_bounding_area"),
            Optional("smallest_bounding_length"),
            Optional("smallest_bounding_width"),
            Optional("volume"),
        ],
    }
)
