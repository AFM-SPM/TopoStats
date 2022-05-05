"""Tests of IO"""
from pathlib import Path
from unittest import TestCase

from pySPM.Bruker import Bruker

from topostats.io import read_yaml

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / 'tests' / 'resources'

CONFIG = {
    "this": "is",
    "a": "test",
    "yaml": "file",
    "numbers": 123,
    "logical": True,
    "nested": {
        "something": "else"
    },
    "a_list": [1, 2, 3]
}


def test_read_yaml() -> None:
    """Test reading of YAML file."""
    sample_config = read_yaml(RESOURCES / 'test.yaml')

    TestCase().assertDictEqual(sample_config, CONFIG)


def test_load_scan(minicircle) -> None:
    """Test loading of image"""

    assert isinstance(minicircle, Bruker)
