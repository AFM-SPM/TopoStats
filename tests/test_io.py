"""Tests of IO"""
from pathlib import Path
from unittest import TestCase

import numpy as np
import pytest

from topostats.io import read_yaml, LoadScans

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


def test_load_scan_spm(load_scan: LoadScans) -> None:
    """Test loading of Bruker spm image"""
    load_scan.img_path = load_scan.img_paths[0]
    load_scan.filename = load_scan.img_paths[0].stem
    image, px_to_nm_scaling = load_scan.load_spm()
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
# def test_extract_pixel_to_nm_scaling(load_scan: LoadScans, unit, x, y, expected) -> None:
#     """Test extraction of pixels to nanometer scaling."""
#     load_scan.load_spm()
#     load_scan._spm_pixel_to_nm_scaling() {"unit": unit, "x": x, "y": y}
#     test_filters_random.extract_pixel_to_nm_scaling()
#     assert test_filters_random.pixel_to_nm_scaling == expected


def test_load_scan_asd(load_scan_asd: LoadScans) -> None:
    """Test loading of high-speed asd image"""
    load_scan_asd.img_path = str(load_scan_asd.img_paths[0])
    load_scan_asd.filename = load_scan_asd.img_paths[0].stem
    images, px_to_nm_scaling = load_scan_asd.load_asd()
    image = images[40]  # some frames are just black so pick a middle one
    assert len(images) == 64
    assert isinstance(image, np.ndarray)
    assert image.shape == (256, 256)
    assert image.sum() == 5958870.556640625
    assert isinstance(px_to_nm_scaling, float)
    assert px_to_nm_scaling == 1.953125


def test_load_scan_get_data(load_scan_data: LoadScans) -> None:
    """Test the LoadScans.get_data() method."""
    assert isinstance(load_scan_data.filename, str)
    assert load_scan_data.filename == "minicircle"
    assert isinstance(load_scan_data.suffix, str)
    assert load_scan_data.suffix == ".spm"
    assert isinstance(load_scan_data.img_dic, dict)
    assert len(load_scan_data.img_dic["images"]) == 1
    assert len(load_scan_data.img_dic["img_paths"]) == 1
    assert len(load_scan_data.img_dic["px_2_nms"]) == 1


def test_load_scan_load_jpk() -> None:
    """Test loading of JPK image."""
    assert True


def test_load_scan_extract_jpk() -> None:
    """Test extraction of data from loaded JPK image."""
    assert True
