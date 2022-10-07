"""Tests of IO"""
from pathlib import Path
from unittest import TestCase

import numpy as np
import pytest

from topostats.io import read_yaml, LoadScan

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"


CONFIG = {
    "this": "is",
    "a": "test",
    "yaml": "file",
    "numbers": 123,
    "logical": True,
    "nested": {"something": "else"},
    "a_list": [1, 2, 3],
}


def test_read_yaml() -> None:
    """Test reading of YAML file."""
    sample_config = read_yaml(RESOURCES / "test.yaml")

    TestCase().assertDictEqual(sample_config, CONFIG)


def test_load_scan_spm(load_scan: LoadScan) -> None:
    """Test loading of Bruker spm image"""
    image, px_to_nm_scaling = load_scan.load_spm()
    assert isinstance(image, np.ndarray)
    assert image.shape == (1024, 1024)
    assert image.sum() == 30695369.188316286
    assert isinstance(px_to_nm_scaling, float)
    assert px_to_nm_scaling == 0.4940029296875


def test_load_scan_get_data(load_scan: LoadScan) -> None:
    """Test the LoadScan.get_data() method."""
    load_scan.get_data()
    assert isinstance(load_scan.filename, str)
    assert load_scan.filename == "minicircle"
    assert isinstance(load_scan.suffix, str)
    assert load_scan.suffix == ".spm"


def test_load_scan_load_jpk() -> None:
    """Test loading of JPK image."""
    assert True


def test_load_scan_extract_jpk() -> None:
    """Test extraction of data from loaded JPK image."""
    assert True
