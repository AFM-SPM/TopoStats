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
    """Test loading of Bruker .spm file"""
    load_scan.img_path = load_scan.img_paths[0]
    load_scan.filename = load_scan.img_paths[0].stem
    image, px_to_nm_scaling = load_scan.load_spm()
    assert isinstance(image, np.ndarray)
    assert image.shape == (1024, 1024)
    assert image.sum() == 30695369.188316286
    assert isinstance(px_to_nm_scaling, float)
    assert px_to_nm_scaling == 0.4940029296875


def test_load_scan_asd(load_scan_asd: LoadScans) -> None:
    """Test loading of high-speed .asd file"""
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


def test_load_scan_ibw(load_scan_ibw: LoadScans) -> None:
    """Test loading of Igor binarywave .ibw file"""
    load_scan_ibw.img_path = load_scan_ibw.img_paths[0]
    load_scan_ibw.filename = load_scan_ibw.img_paths[0].stem
    image, px_to_nm_scaling = load_scan_ibw.load_ibw()
    assert isinstance(image, np.ndarray)
    assert image.shape == (512, 512)
    assert image.sum() == -218091520.0
    assert isinstance(px_to_nm_scaling, float)
    assert px_to_nm_scaling == 1.5625


def test_load_scan_jpk(load_scan_jpk: LoadScans) -> None:
    """Test loading of JPK Instruments .jpk file."""
    load_scan_jpk.img_path = load_scan_jpk.img_paths[0]
    load_scan_jpk.filename = load_scan_jpk.img_paths[0].stem
    image, px_to_nm_scaling = load_scan_jpk.load_jpk()
    print(image.shape)
    print(image.sum())
    print(px_to_nm_scaling)
    assert isinstance(image, np.ndarray)
    assert image.shape == (256, 256)
    assert image.sum() == 286598232.9308627
    assert isinstance(px_to_nm_scaling, float)
    assert px_to_nm_scaling == 1.2770176335964876


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


@pytest.mark.parametrize(
    "load_scan_object, length, image_shape, image_sum, filename, pixel_to_nm_scaling",
    [
        ("load_scan", 1, (1024, 1024), 30695369.188316286, "minicircle", 0.4940029296875),
        ("load_scan_ibw", 1, (512, 512), -218091520.0, "minicircle2", 1.5625),
        ("load_scan_jpk", 1, (256, 256), 286598232.9308627, "file", 1.2770176335964876),
        ("load_scan_asd", 64, (256, 256), 5958870.556640625, "minicircles_frame_40", 1.953125),
    ],
)
def test_load_scan_get_data(
    load_scan_object: LoadScans,
    length: int,
    image_shape: tuple,
    image_sum: float,
    filename: str,
    pixel_to_nm_scaling: float,
    request,
) -> None:
    """Test the LoadScan.get_data() method."""
    scan = request.getfixturevalue(load_scan_object)
    scan.get_data()
    assert len(scan.img_dict) == length
    assert isinstance(scan.img_dict[filename]["image"], np.ndarray)
    assert scan.img_dict[filename]["image"].shape == image_shape
    assert scan.img_dict[filename]["image"].sum() == image_sum
    assert isinstance(scan.img_dict[filename]["img_path"], Path)
    assert scan.img_dict[filename]["img_path"] == RESOURCES / filename
    assert isinstance(scan.img_dict[filename]["px_2_nm"], float)
    assert scan.img_dict[filename]["px_2_nm"] == pixel_to_nm_scaling
