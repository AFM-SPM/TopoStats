"""Validation of configuration."""
import logging
import os
from pathlib import Path
from schema import Optional, Or, Schema, SchemaError

from topostats.logs.logs import LOGGER_NAME

LOGGER = logging.getLogger(LOGGER_NAME)

# pylint: disable=line-too-long


def validate_config(config: dict, schema: Schema, config_type: str) -> None:
    """Validate configuration.

    Parameters
    ----------
    config: dict
        Config dictionary imported by read_yaml() and parsed through clean_config().
    schema: Schema
        A schema against which the configuration is to be compared.
    config_type: str
    """

    try:
        schema.validate(config)
        LOGGER.info(f"The {config_type} configuration is valid.")
    except SchemaError as schema_error:
        raise SchemaError(
            f"There is an error in your {config_type} configuration. "
            "Please refer to the first error message above for details"
        ) from schema_error


DEFAULT_CONFIG_SCHEMA = Schema(
    {
        "base_dir": Path,
        "output_dir": Path,
        "warnings": Or("ignore", error="Invalid value in config for 'warnings', valid values are 'ignore'"),
        "cores": lambda n: 1 <= n <= os.cpu_count(),
        "quiet": Or(True, False, error="Invalid value in config for 'quiet', valid values are 'True' or 'False'"),
        "file_ext": Or(
            ".spm",
            ".asd",
            ".jpk",
            ".ibw",
            error="Invalid value in config for 'file_ext', valid values are '.spm', '.jpk', '.ibw' or '.asd'.",
        ),
        "loading": {
            "channel": Or(
                "ZSensor",
                "",
                "Stiffness",
                "LogStiffness",
                "Adhesion",
                "Deformation",
                "Dissipation",
                "Height Sensor",
                "Height",  # end of spm channels
                "HeightTracee",
                "HeightRetrace",
                "ZSensorTrace",
                "ZSensorRetrace",
                "UserIn0Trace",
                "UserIn0Retrace",
                "UserIn1Trace",
                "UserIn1Retrace",  # end of ibw channels
                "topography",
                "phase",  # end of asd channels
                "height_retrace",
                "measuredHeight_retrace",
                "amplitude_retrace",
                "phase_retrace",
                "error_retrace",
                "height_trace",
                "measuredHeight_trace",
                "amplitude_trace",
                "phase_trace",
                "error_trace",  # end of jpk channels
                error="Invalid value in config file for 'channel', all possible"
                "image channels are seen in the above error message.",
            )
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
                "lower": lambda n: n > 0,
                "upper": lambda n: n > 0,
            },
            "threshold_absolute": {
                "lower": Or(
                    int,
                    float,
                    error=(
                        "Invalid value in config for filter.threshold.absolute.lower " "should be type int or float"
                    ),
                ),
                "upper": Or(
                    int,
                    float,
                    error=(
                        "Invalid value in config for filter.threshold.absolute.lower " "should be type int or float"
                    ),
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
            "run": Or(True, False, error="Invalid value in config for grains.run, valid values are 'True' or 'False'"),
            "smallest_grain_size_nm2": lambda n: n > 0.0,
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
                "lower": lambda n: n > 0,
                "upper": lambda n: n > 0,
            },
            "threshold_absolute": {
                "lower": Or(
                    int,
                    float,
                    error=(
                        "Invalid value in config for grains.threshold.absolute.lower " "should be type int or float"
                    ),
                ),
                "upper": Or(
                    int,
                    float,
                    error=(
                        "Invalid value in config for grains.threshold.absolute.lower " "should be type int or float"
                    ),
                ),
            },
            "absolute_area_threshold": {
                "upper": [
                    Or(
                        int,
                        None,
                        error=(
                            "Invalid value in config for 'grains.absolute_area_threshold.upper', valid values "
                            "are int or null"
                        ),
                    )
                ],
                "lower": [
                    Or(
                        int,
                        None,
                        error=(
                            "Invalid value in config for 'grains.absolute_area_threshold.lower', valid values "
                            "are int or null"
                        ),
                    )
                ],
            },
            "direction": Or(
                "both",
                "lower",
                "upper",
                error="Invalid direction for grains.direction valid values are 'both', 'lower' or 'upper",
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
            "cropped_size": Or(
                float,
                int,
            ),
        },
        "dnatracing": {
            "run": Or(
                True,
                False,
                error="Invalid value in config for 'filter.run', valid values are 'True' or 'False'",
            )
        },
        "plotting": {
            "run": Or(
                True,
                False,
                error="Invalid value in config for 'plotting.run', valid values are 'True' or 'False'",
            ),
            "save_format": str,
            "image_set": Or(
                "all",
                "core",
                error="Invalid value in config for 'plotting.image_set', valid values " "are 'all' or 'core'",
            ),
            "zrange": list,
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
            "cmap": Or(
                "afmhot",
                "nanoscope",
                "gwyddion",
                error="Invalid value in config for 'plotting.cmap', valid values are 'afmhot', 'nanoscope' or 'gwyddion'",
            ),
            "mask_cmap": str,
            "histogram_log_axis": Or(
                True,
                False,
                error=(
                    "Invalid value in config plotting histogram. For 'log_y_axis', valid values are 'True' or "
                    "'False'"
                ),
            ),
            "histogram_bins": lambda n: n > 0,
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
                    "Invalid value in config 'extracted_channel.image_type', valid values "
                    "are 'binary' or 'non-binary'"
                ),
            ),
            "core_set": bool,
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
        },
        "zero_average_background": {
            "filename": str,
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=(
                    "Invalid value in config 'gaussian_filtered.image_type', valid values "
                    "are 'binary' or 'non-binary'"
                ),
            ),
            "core_set": bool,
        },
        "gaussian_filtered": {
            "filename": str,
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=(
                    "Invalid value in config 'gaussian_filtered.image_type', valid values "
                    "are 'binary' or 'non-binary'"
                ),
            ),
            "core_set": bool,
        },
        "z_threshed": {
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=(
                    "Invalid value in config 'z_threshold.image_type', valid values " "are 'binary' or 'non-binary'"
                ),
            ),
            "core_set": True,
        },
        "mask_grains": {
            "filename": str,
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=(
                    "Invalid value in config 'mask_grains.image_type', valid values " "are 'binary' or 'non-binary'"
                ),
            ),
            "core_set": bool,
        },
        "labelled_regions_01": {
            "filename": str,
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=(
                    "Invalid value in config 'labelled_regions_01.image_type', valid values "
                    "are 'binary' or 'non-binary'"
                ),
            ),
            "core_set": bool,
        },
        "tidied_border": {
            "filename": str,
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=(
                    "Invalid value in config 'tidied_border.image_type', valid values " "are 'binary' or 'non-binary'"
                ),
            ),
            "core_set": bool,
        },
        "removed_noise": {
            "filename": str,
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=(
                    "Invalid value in config 'removed_noise.image_type', valid values " "are 'binary' or 'non-binary'"
                ),
            ),
            "core_set": bool,
        },
        "removed_small_objects": {
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
            "core_set": bool,
        },
        "mask_overlay": {
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=(
                    "Invalid value in config 'mask_overlay.image_type', valid values " "are 'binary' or 'non-binary'"
                ),
            ),
            "core_set": True,
        },
        "labelled_regions_02": {
            "filename": str,
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=(
                    "Invalid value in config 'labelled_regions_02.image_type', valid values "
                    "are 'binary' or 'non-binary'"
                ),
            ),
            "core_set": bool,
        },
        "coloured_regions": {
            "filename": str,
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=(
                    "Invalid value in config 'coloured_regions.image_type', valid values "
                    "are 'binary' or 'non-binary'"
                ),
            ),
            "core_set": bool,
        },
        "bounding_boxes": {
            "filename": str,
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=(
                    "Invalid value in config 'bounding_boxes.image_type', valid values " "are 'binary' or 'non-binary'"
                ),
            ),
            "core_set": bool,
        },
        "coloured_boxes": {
            "filename": str,
            "title": str,
            "image_type": Or(
                "binary",
                "non-binary",
                error=(
                    "Invalid value in config 'coloured_boxes.image_type', valid values " "are 'binary' or 'non-binary'"
                ),
            ),
            "core_set": bool,
        },
        "grain_image": {
            "image_type": Or(
                "binary",
                "non-binary",
                error=(
                    "Invalid value in config 'grain_image.image_type', valid values " "are 'binary' or 'non-binary'"
                ),
            ),
            "core_set": False,
        },
        "grain_mask": {
            "image_type": Or(
                "binary",
                "non-binary",
                error=("Invalid value in config 'grain_mask.image_type', valid values " "are 'binary' or 'non-binary'"),
            ),
            "core_set": bool,
        },
        "grain_mask_image": {
            "image_type": Or(
                "binary",
                "non-binary",
                error=(
                    "Invalid value in config 'grain_mask_image.image_type', valid values "
                    "are 'binary' or 'non-binary'"
                ),
            ),
            "core_set": bool,
        },
    }
)

SUMMARY_SCHEMA = Schema(
    {
        "output_dir": Path,
        "csv_file": str,
        "file_ext": Or(
            "png",
            "pdf",
            "svg",
            error=("Invalid value in config 'file_ext', valid values are 'png', 'pdf' or 'svg' "),
        ),
        "pickle_plots": Or(
            True, False, error="Invalid value in config for 'pickle_plots', valid values are 'True' or 'False'"
        ),
        "var_to_label": Or(
            None, str, error="Invalid value in config for 'var_to_label', valid values are 'None' or a str"
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
            Optional("contour_lengths"),
            Optional("end_to_end_distance"),
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
