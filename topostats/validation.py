"""Validation of configuration."""
import logging
import os
from pathlib import Path
from schema import Or, Schema, SchemaError

from topostats.logs.logs import LOGGER_NAME

LOGGER = logging.getLogger(LOGGER_NAME)

# pylint: disable=line-too-long


def validate_config(config: dict):
    """Validate configuration.

    Parameters
    ----------
    config: dict
        Config dictionary imported by read_yaml() and parsed through clean_config().
    """
    config_schema = Schema(
        {
            "base_dir": Path,
            "output_dir": Path,
            "warnings": Or("ignore", error="Invalid value in config for 'warnings', valid values are 'ignore'"),
            "cores": lambda n: 1 <= n <= os.cpu_count(),
            "quiet": Or(True, False, error="Invalid value in config for 'quiet', valid values are 'True' or 'False'"),
            "file_ext": Or(
                ".spm",
                ".jpk",
                ".ibw",
                error="Invalid value in config for 'file_ext', valid values are '.spm', '.jpk' or '.ibw'",
            ),
            "loading": {
                "channel": str,
            },
            "filter": {
                "run": Or(
                    True,
                    False,
                    error="Invalid value in config for 'filter.run', valid values are 'True' or 'False'",
                ),
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
                "threshold_std_dev": lambda n: 0 < n <= 6,
                "threshold_absolute_lower": float,
                "threshold_absolute_upper": float,
                "gaussian_size": float,
                "gaussian_mode": Or(
                    "nearest",
                    error="Invalid value in config for 'filter.gaussian_mode', valid values are 'nearest'",
                ),
            },
            "grains": {
                "run": Or(
                    True, False, error="Invalid value in config for grains.run, valid values are 'True' or 'False'"
                ),
                "absolute_smallest_grain_size": int,
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
                "threshold_std_dev": lambda n: 0 < n <= 6,
                "threshold_absolute_lower": float,
                "threshold_absolute_upper": float,
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
                "cropped_size": float,
                "save_cropped_grains": Or(
                    True,
                    False,
                    error=(
                        "Invalid value in config for 'grainstats.save_cropped_grains, valid values "
                        "are 'True' or 'False'"
                    ),
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
                    error="Invalid value in config for 'plotting.cmap', valid values are 'afmhot' or 'nanoscope",
                ),
            },
        }
    )

    try:
        config_schema.validate(config)
        LOGGER.info("Configuration is valid.")
    except SchemaError as schema_error:
        raise SchemaError(
            "There is an error in your configuration. Please refer to the first error message above for details"
        ) from schema_error


def validate_plotting(config: dict) -> None:
    """Validate configuration.

    Parameters
    ----------
    config: dict
        Config dictionary imported by read_yaml() and parsed through clean_config().
    """
    config_schema = Schema(
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
            "initial_align": {
                "filename": str,
                "title": str,
                "image_type": Or(
                    "binary",
                    "non-binary",
                    error=(
                        "Invalid value in config 'initial_align.image_type', valid values "
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
            "masked_align": {
                "filename": str,
                "title": str,
                "image_type": Or(
                    "binary",
                    "non-binary",
                    error=(
                        "Invalid value in config 'masked_align.image_type', valid values "
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
            "zero_averaged_background": {
                "filename": str,
                "title": str,
                "image_type": Or(
                    "binary",
                    "non-binary",
                    error=(
                        "Invalid value in config 'zero_averaged_bacground.image_type', valid values "
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
                        "Invalid value in config 'tidied_border.image_type', valid values "
                        "are 'binary' or 'non-binary'"
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
                        "Invalid value in config 'removed_noise.image_type', valid values "
                        "are 'binary' or 'non-binary'"
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
                        "Invalid value in config 'mask_overlay.image_type', valid values "
                        "are 'binary' or 'non-binary'"
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
                        "Invalid value in config 'bounding_boxes.image_type', valid values "
                        "are 'binary' or 'non-binary'"
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
                        "Invalid value in config 'coloured_boxes.image_type', valid values "
                        "are 'binary' or 'non-binary'"
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
                "core_set": True,
            },
            "grain_mask": {
                "image_type": Or(
                    "binary",
                    "non-binary",
                    error=(
                        "Invalid value in config 'grain_mask.image_type', valid values " "are 'binary' or 'non-binary'"
                    ),
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
    try:
        config_schema.validate(config)
        LOGGER.info("Plotting configuration is valid.")
    except SchemaError as schema_error:
        raise SchemaError(
            "There is an error in your configuration. Please refer to the first error message above for details"
        ) from schema_error
