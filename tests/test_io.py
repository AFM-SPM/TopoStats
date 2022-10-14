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
    image, px_to_nm_scaling = load_scan._load_spm()
    assert isinstance(image, np.ndarray)
    assert image.shape == (1024, 1024)
    assert image.sum() == 30695369.188316286
    assert isinstance(px_to_nm_scaling, float)
    assert px_to_nm_scaling == 0.4940029296875


# FIXME : Get this test working
# @pytest.mark.parametrize(
#     "unit, x, y, expected",
#     [
#         ("um", 100, 100, 97.65625),
#         ("nm", 50, 50, 0.048828125),
#     ],
# )
# def test_extract_pixel_to_nm_scaling(load_scan: LoadScan, unit, x, y, expected) -> None:
#     """Test extraction of pixels to nanometer scaling."""
#     load_scan.load_spm()
#     load_scan._spm_pixel_to_nm_scaling() {"unit": unit, "x": x, "y": y}
#     test_filters_random.extract_pixel_to_nm_scaling()
#     assert test_filters_random.pixel_to_nm_scaling == expected


@pytest.mark.parametrize(
    "load_scan_object, filename, suffix", [
        ("load_scan", "minicircle", ".spm"),
        ("load_scan_ibw", "minicircle2", ".ibw")
    ]
)
def test_load_scan_get_data(load_scan_object: LoadScan, filename: str, suffix: str, request) -> None:
    """Test the LoadScan.get_data() method."""
    scan = request.getfixturevalue(load_scan_object)
    scan.get_data()
    assert isinstance(scan.filename, str)
    assert scan.filename == filename
    assert isinstance(scan.suffix, str)
    assert scan.suffix == suffix


def test_load_scan_ibw(load_scan_ibw: LoadScan) -> None:
    image, px_to_nm_scaling = load_scan_ibw._load_ibw()
    assert isinstance(image, np.ndarray)
    assert image.shape == (512, 512)
    assert image.sum() == -218091520.0
    assert isinstance(px_to_nm_scaling, float)
    assert px_to_nm_scaling == 1.5625


# FIXME : Get this test working
# @pytest.mark.parametrize(
#     "px_x, px_y, re_x, re_y, expected",
#     [
#         (100, 100, 1e-7, 1e-7, 1),
#         (50, 50, 1e-7, 1e-7, 2),
#     ],
# )
# def test_extract_pixel_to_nm_scaling(load_scan_ibw: LoadScan, unit, x, y, expected) -> None:
#     """Test extraction of pixels to nanometer scaling."""
#     load_scan_ibw.load_spm()
#     px_2_nm = load_scan_ibw._ibw_pixel_to_nm_scaling() {
# "unit": unit, "px_x": px_x, "px_y": py_y, "re_x": re_x, "re_y": re_y
# }
#     assert px_2_nm == expected



def test_load_scan_load_jpk() -> None:
    """Test loading of JPK image."""
    assert True


def test_load_scan_extract_jpk() -> None:
    """Test extraction of data from loaded JPK image."""
    assert True
