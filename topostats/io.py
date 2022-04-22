"""Functions for reading and writing data."""
import logging
from pathlib import Path
from typing import Union, Dict

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
