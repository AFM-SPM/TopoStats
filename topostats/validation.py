"""Validation of configuration."""
import logging
import os
from pathlib import Path
from schema import Or, Schema, SchemaError

from topostats.logs.logs import LOGGER_NAME

LOGGER = logging.getLogger(LOGGER_NAME)


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
                ".spm", ".jpk", error="Invalid value in config for 'file_ext', valid values are '.spm' or '.jpk'"
            ),
            "loading": {
                "channel": Or("Height"),
            },
            "filter": {
                "run": Or(
                    True, False, error="Invalid value in config for 'filter.run', valid values are 'True' or 'False'"
                ),
                "channel": Or("Height"),
                "threshold_method": Or(
                    "absolute",
                    "otsu",
                    "std_dev",
                    error="Invalid value in config for 'filter.threshold_method', valid values are 'absolute', 'otsu' or 'std_dev'",  # pylint: disable=line-too-long
                ),
                "otsu_threshold_multiplier": float,
                "threshold_std_dev": lambda n: 0 < n <= 6,
                "threshold_absolute_lower": float,
                "threshold_absolute_upper": float,
                "gaussian_size": float,
                "gaussian_mode": Or(
                    "nearest", error="Invalid value in config for 'filter.gaussian_mode', valid values are 'nearest'"
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
                    error="Invalid value in config for 'grains.threshold_method', valid values are 'absolute', 'otsu' or 'std_dev'",  # pylint: disable=line-too-long
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
                            error="Invalid value in config for 'grains.absolute_area_threshold.upper', valid values are int or null",  # pylint: disable=line-too-long
                        )
                    ],
                    "lower": [
                        Or(
                            int,
                            None,
                            error="Invalid value in config for 'grains.absolute_area_threshold.lower', valid values are int or null",  # pylint: disable=line-too-long
                        )
                    ],
                },
                "direction": Or(
                    "both",
                    "lower",
                    "upper",
                    error="Invalid direction for grains.direction valid values are 'both', 'lower' or 'upper",
                ),
                "background": float,
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
                    error="Invalid value in config for 'grainstats.save_cropped_grains, valid values are 'True' or 'False'",  # pylint: disable=line-too-long
                ),
            },
            "dnatracing": {
                "run": Or(
                    True, False, error="Invalid value in config for 'filter.run', valid values are 'True' or 'False'"
                )
            },
            "plotting": {
                "run": Or(
                    True, False, error="Invalid value in config for 'plotting.run', valid values are 'True' or 'False'"
                ),
                "save_format": str,
                "image_set": Or(
                    "all",
                    "core",
                    error="Invalid value in config for 'plotting.image_set', valid values are 'all' or 'core'",
                ),
                "zrange": list,
                "colorbar": Or(
                    True,
                    False,
                    error="Invalid value in config for 'plotting.colorbar', valid values are 'True' or 'False'",
                ),
                "axes": Or(
                    True, False, error="Invalid value in config plotting.for 'axes', valid values are 'True' or 'False'"
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
    except SchemaError as se:
        raise SchemaError(
            "There is an error in your configuration. Please refer to the first error message above for details"
        ) from se
