"""Utilities"""
from argparse import Namespace
import logging
from pathlib import Path
from typing import Union, List, Dict

import numpy as np
from skimage.filters import threshold_otsu

from topostats.logs.logs import LOGGER_NAME

LOGGER = logging.getLogger(LOGGER_NAME)


def convert_path(path: Union[str, Path]) -> Path:
    """Ensure path is Path object.

    Parameters
    ----------
    path: Union[str, Path]
        Path to be converted.

    Returns
    -------
    Path
        pathlib Path
    """
    return Path().cwd() if path == './' else Path(path)


def find_images(base_dir: Union[str, Path] = None, file_ext: str = '.spm') -> List:
    """Scan the specified directory for images with the given file extension.

    Parameters
    ----------
    base_dir: Union[str, Path]
        Directory to recursively search for files, if not specified the current directory is scanned.
    file_ext: str
        File extension to search for.

    Returns
    -------
    List
        List of files found with the extension in the given directory.
    """
    base_dir = Path('./') if base_dir is None else Path(base_dir)
    return list(base_dir.glob('**/*' + file_ext))


def update_config(config: dict, args: Union[dict, Namespace]) -> Dict:
    """Update the configuration with any arguments

    Parameters
    ----------
    config: dict
        Dictionary of configuration (typically read from YAML file specified with '-c/--config <filename>')
    args: Namespace
        Command line arguments
    Returns
    -------
    Dict
        Dictionary updated with command arguments.
    """
    args = vars(args) if isinstance(args, Namespace) else args

    config_keys = config.keys()
    for arg_key, arg_value in args.items():
        if arg_key in config_keys and arg_value is not None:
            original_value = config[arg_key]
            config[arg_key] = arg_value
            LOGGER.info(f'Updated config config[{arg_key}] : {original_value} > {arg_value} ')
    config['base_dir'] = convert_path(config['base_dir'])
    config['output_dir'] = convert_path(config['output_dir'])
    return config


def get_mask(image: np.array, threshold: float) -> np.array:
    """Calculate a mask for pixels that exceed the threshold

    Parameters
    ----------
    image: np.array
        Numpy array representing image.
    threshold: float
        Factor for defining threshold.

    Returns
    -------
    np.array
        Numpy array of image with objects coloured.
    """
    LOGGER.info('[get_mask] Deriving mask.')
    return image > threshold


def get_threshold(image: np.array) -> float:
    """Returns a threshold value separating the background and foreground of a 2D heightmap.

    Parameters
    ----------
    image: np.array
        Numpy array representing image.

    Returns
    -------
    float
        Otsu threshold.

    Notes
    -----

    The `Otsu method <https://en.wikipedia.org/wiki/Otsu%27s_method>`_ is used for threshold derivation.
    """
    LOGGER.info('[get_threshold] Calculating Otsu threshold.')
    return threshold_otsu(image)
