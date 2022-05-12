"""Functions for reading and writing data."""
import logging
from pathlib import Path
from typing import Union, Dict

from pySPM.Bruker import Bruker
import ruamel.yaml
from ruamel.yaml import YAML, YAMLError
from topostats.logs.logs import LOGGER_NAME

LOGGER = logging.getLogger(LOGGER_NAME)


def read_yaml(filename: Union[str, Path]) -> Dict:
    """Read a YAML file.

    Parameters
    ----------
    filename: Union[str, Path]
        YAML file to read.

    Returns
    -------
    Dict
        Dictionary of the file."""

    with Path(filename).open() as f:
        try:
            yaml_file = YAML(typ='safe')
            return yaml_file.load(f)
        except YAMLError as exception:
            LOGGER.error(exception)
            return {}


def load_scan(img_path: Union[str, Path]) -> Bruker:
    """Load the image from file.

    Parameters
    ----------
    img_path : Union[str, Path]
        Path to image that needs loading.

    Returns
    -------
    Bruker

    Examples
    --------
    FIXME: Add docs.

    """
    LOGGER.info(f'Loading image from : {img_path}')
    return Bruker(Path(img_path))
