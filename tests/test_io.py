"""Tests of IO."""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
import pySPM
import pytest

from topostats.io import (
    LoadScans,
    convert_basename_to_relative_paths,
    dict_to_hdf5,
    find_files,
    get_date_time,
    get_out_path,
    get_relative_paths,
    hdf5_to_dict,
    load_array,
    load_pkl,
    path_to_str,
    read_64d,
    read_char,
    read_gwy_component_dtype,
    read_null_terminated_string,
    read_u32i,
    read_yaml,
    save_array,
    save_folder_grainstats,
    save_pkl,
    save_topostats_file,
    write_config_with_comments,
    write_yaml,
)
from topostats.logs.logs import LOGGER_NAME

LOGGER = logging.getLogger(LOGGER_NAME)

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

# pylint: disable=protected-access
# pylint: disable=too-many-lines


def dict_almost_equal(dict1, dict2, abs_tol=1e-9):
    """Recursively check if two dictionaries are almost equal with a given absolute tolerance.

    Parameters
    ----------
    dict1: dict
        First dictionary to compare.
    dict2: dict
        Second dictionary to compare.
    abs_tol: float
        Absolute tolerance to check for equality.

    Returns
    -------
    bool
        True if the dictionaries are almost equal, False otherwise.
    """
    if dict1.keys() != dict2.keys():
        return False

    LOGGER.info("Comparing dictionaries")

    for key in dict1:
        LOGGER.info(f"Comparing key {key}")
        if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            if not dict_almost_equal(dict1[key], dict2[key], abs_tol=abs_tol):
                return False
        elif isinstance(dict1[key], np.ndarray) and isinstance(dict2[key], np.ndarray):
            if not np.allclose(dict1[key], dict2[key], atol=abs_tol):
                LOGGER.info(f"Key {key} type: {type(dict1[key])} not equal: {dict1[key]} != {dict2[key]}")
                return False
        elif isinstance(dict1[key], float) and isinstance(dict2[key], float):
            if not np.isclose(dict1[key], dict2[key], atol=abs_tol):
                LOGGER.info(f"Key {key} type: {type(dict1[key])} not equal: {dict1[key]} != {dict2[key]}")
                return False
        elif dict1[key] != dict2[key]:
            LOGGER.info(f"Key {key} not equal: {dict1[key]} != {dict2[key]}")
            return False

    return True


def test_get_date_time() -> None:
    """Test the fetching of a formatted date and time string."""
    assert datetime.strptime(get_date_time(), "%Y-%m-%d %H:%M:%S")


def test_read_yaml() -> None:
    """Test reading of YAML file."""
    sample_config = read_yaml(RESOURCES / "test.yaml")

    assert sample_config == CONFIG


@pytest.mark.parametrize(
    ("filename", "config", "expected_filename"),
    [
        ("test_config_with_comments.yaml", None, "test_config_with_comments.yaml"),
        ("test_config_with_comments", None, "test_config_with_comments.yaml"),
        (None, "default", "config.yaml"),
        (None, None, "config.yaml"),
        # Example of how to test `dna_config.yaml`
        # (None, "dna", "dna_config.yaml")
    ],
)
def test_write_config_with_comments(tmp_path: Path, filename: str, config: str, expected_filename: str) -> None:
    """Test writing of config file with comments.

    If and when specific configurations for different sample types are introduced then the parametrisation can be
    extended to allow these adding their names under "config" and introducing specific parameters that may differe
    between the configuration files.
    """
    # Setup argparse.Namespace with the tests parameters
    args = argparse.Namespace()
    args.filename = filename
    args.output_dir = tmp_path
    args.config = config

    # Write default config with comments to file
    write_config_with_comments(args)

    # Read the written config
    with Path.open(tmp_path / expected_filename, encoding="utf-8") as f:
        written_config = f.read()

    # Validate that the written config has comments in it
    assert "Config file generated" in written_config
    assert "For more information on configuration and how to use it" in written_config
    # Validate some of the parameters are present
    assert "loading:" in written_config
    assert "gaussian_mode: nearest" in written_config
    assert "style: topostats.mplstyle" in written_config
    assert "pixel_interpolation: null" in written_config


def test_write_config_with_comments_user_warning(tmp_path: Path) -> None:
    """Tests a user warning is raised if an attempt is made to request a configuration file type that does not exist."""
    args = argparse.Namespace()
    args.filename = "config.yaml"
    args.output_dir = tmp_path
    args.config = "nonsense"

    with pytest.raises(UserWarning):
        write_config_with_comments(args)


def test_write_yaml(tmp_path: Path) -> None:
    """Test writing of dictionary to YAML."""
    write_yaml(
        config=CONFIG,
        output_dir=tmp_path,
        config_file="test.yaml",
        header_message="This is a test YAML configuration file",
    )
    outfile = tmp_path / "test.yaml"
    assert outfile.is_file()


def test_path_to_str(tmp_path: Path) -> None:
    """Test that Path objects are converted to strings."""
    CONFIG_PATH = {
        "this": "is",
        "a": "test",
        "with": tmp_path,
        "and": {"nested": tmp_path / "nested"},
    }
    CONFIG_STR = path_to_str(CONFIG_PATH)

    assert isinstance(CONFIG_STR, dict)
    assert isinstance(CONFIG_STR["with"], str)
    assert CONFIG_STR["with"] == str(tmp_path)
    assert isinstance(CONFIG_STR["and"]["nested"], str)
    assert CONFIG_STR["and"]["nested"] == str(tmp_path / "nested")


def test_save_array(synthetic_scars_image: np.ndarray, tmp_path: Path) -> None:
    """Test saving Numpy arrays."""
    save_array(array=synthetic_scars_image, outpath=tmp_path, filename="test", array_type="synthetic")

    outfile = tmp_path / "test_synthetic.npy"
    assert outfile.is_file()


def test_load_array() -> None:
    """Test loading Numpy arrays."""
    target = load_array(RESOURCES / "test_scars_synthetic_scar_image.npy")
    expected = np.load(RESOURCES / "test_scars_synthetic_scar_image.npy")

    np.testing.assert_array_equal(target, expected)


@pytest.mark.parametrize(
    ("dict1", "dict2", "tolerance", "expected"),
    [
        pytest.param(
            {"a": "test", "b": np.array([1.0, 2.0, 3.0])},
            {"c": "test", "d": np.array([1.0, 2.0, 3.0])},
            0.00001,
            False,
            id="keys not equal",
        ),
        pytest.param(
            {"a": "test", "b": np.array([1.0, 2.0, 3.0])},
            {"a": "test", "b": np.array([1.0, 2.0, 3.0])},
            0.00001,
            True,
            id="string equal",
        ),
        pytest.param(
            {"a": "test", "b": np.array([1.0, 2.0, 3.0])},
            {"a": "WRONG", "b": np.array([1.0, 2.0, 3.0])},
            0.00001,
            False,
            id="string not equal",
        ),
        pytest.param(
            {
                "a": "test",
                "b": np.array([1.00001, 2.00002, 3.00005]),
                "c": {"d": np.array([1.00001, 2.00002, 3.00005])},
            },
            {"a": "test", "b": np.array([1.0, 2.0, 3.0]), "c": {"d": np.array([1.0, 2.0, 3.0])}},
            0.0001,
            True,
            id="generous tolerance",
        ),
        pytest.param(
            {
                "a": "test",
                "b": np.array([1.00001, 2.00002, 3.00005]),
                "c": {"d": np.array([1.00001, 2.00002, 3.00005])},
            },
            {"a": "test", "b": np.array([1.0, 2.0, 3.0]), "c": {"d": np.array([1.0, 2.0, 3.0])}},
            0.000001,
            False,
            id="strict tolerance",
        ),
        pytest.param(
            {
                "a": "test",
                "b": np.array([1.0, 2.0, 3.0]),
                "c": {"d": np.array([1.0, 2.0, 3.0])},
            },
            {"a": "test", "b": np.array([1.0, 2.0, 3.0]), "c": {"d": np.array([1.0, 2.0, 9.0])}},
            0.0001,
            False,
            id="nested dict not equal",
        ),
        pytest.param(
            {"a": 0.01},
            {"a": 0.02},
            0.1,
            True,
            id="float equal",
        ),
        pytest.param(
            {"a": 0.01},
            {"a": 0.02},
            0.001,
            False,
            id="float not equal",
        ),
    ],
)
def test_dict_almost_equal(dict1: dict, dict2: dict, tolerance: float, expected: bool) -> None:
    """Test that two dictionaries are almost equal."""
    assert dict_almost_equal(dict1, dict2, tolerance) == expected


@pytest.mark.parametrize("non_existant_file", [("does_not_exist.npy"), ("does_not_exist.np"), ("does_not_exist.csv")])
def test_load_array_file_not_found(non_existant_file: str) -> None:
    """Test exceptions when trying to load arrays that don't exist."""
    with pytest.raises(FileNotFoundError):
        assert load_array(non_existant_file)


def test_find_files() -> None:
    """Test finding images."""
    found_images = find_files(base_dir="tests/", file_ext=".spm")

    assert isinstance(found_images, list)
    assert len(found_images) == 1
    assert isinstance(found_images[0], Path)
    assert "minicircle.spm" in str(found_images[0])


def test_read_null_terminated_string() -> None:
    """Test reading a null terminated string from a binary file."""
    with Path.open(RESOURCES / "IO_binary_file.bin", "rb") as open_binary_file:  # pylint: disable=unspecified-encoding
        value = read_null_terminated_string(open_binary_file)
        assert isinstance(value, str)
        assert value == "test"


def test_read_u32i() -> None:
    """Test reading an unsigned 32 bit integer from a binary file."""
    with Path.open(RESOURCES / "IO_binary_file.bin", "rb") as open_binary_file:  # pylint: disable=unspecified-encoding
        open_binary_file.seek(5)
        value = read_u32i(open_binary_file)
        assert isinstance(value, int)
        assert value == 32


def test_read_64d() -> None:
    """Test reading a 64-bit double from an open binary file."""
    with Path.open(RESOURCES / "IO_binary_file.bin", "rb") as open_binary_file:  # pylint: disable=unspecified-encoding
        open_binary_file.seek(9)
        value = read_64d(open_binary_file)
        assert isinstance(value, float)
        assert value == 3.141592653589793


def test_read_char() -> None:
    """Test reading a character from an open binary file."""
    with Path.open(RESOURCES / "IO_binary_file.bin", "rb") as open_binary_file:  # pylint: disable=unspecified-encoding
        open_binary_file.seek(17)
        value = read_char(open_binary_file)
        assert isinstance(value, str)
        assert value == "Z"


def test_read_gwy_component_dtype() -> None:
    """Test reading a data type of a `.gwy` file component from an open binary file."""
    with Path.open(RESOURCES / "IO_binary_file.bin", "rb") as open_binary_file:  # pylint: disable=unspecified-encoding
        open_binary_file.seek(18)
        value = read_gwy_component_dtype(open_binary_file)
        assert isinstance(value, str)
        assert value == "D"


@pytest.mark.parametrize(
    ("input_paths", "expected_paths"),
    [
        (
            [Path("a/b/c/d"), Path("a/b/e/f"), Path("a/b/g"), Path("a/b/h")],
            ["c/d", "e/f", "g", "h"],
        ),
        (["a/b/c/d", "a/b/e/f", "a/b/g", "a/b/h"], ["c/d", "e/f", "g", "h"]),
        (["g", "a/b/e/f", "a/b/g", "a/b/h"], ["g", "a/b/e/f", "a/b/g", "a/b/h"]),
        (["a/b/c/d"], ["a/b/c/d"]),
        (["a/b/c/d", "a/b/c/d"], ["a/b/c/d", "a/b/c/d"]),
    ],
)
def test_get_relative_paths(input_paths: list, expected_paths: list):
    """Test the get_paths_relative_to_deepest_common_path function."""
    relative_paths = get_relative_paths(input_paths)

    assert relative_paths == expected_paths


def test_convert_basename_to_relative_paths():
    """Test the convert_basename_to_relative_paths function."""
    input_df = {
        "Image": ["im1", "im2", "im3", "im4"],
        "threshold": ["above", "above", "above", "above"],
        "molecule_number": [0, 0, 0, 0],
        "basename": ["super/sub1", "super/sub2", "super/sub3", "super/sub3/sub4"],
        "area": [10, 20, 30, 40],
    }

    input_df = pd.DataFrame(input_df)

    result = convert_basename_to_relative_paths(input_df)

    expected = {
        "Image": ["im1", "im2", "im3", "im4"],
        "threshold": ["above", "above", "above", "above"],
        "molecule_number": [0, 0, 0, 0],
        "basename": ["sub1", "sub2", "sub3", "sub3/sub4"],
        "area": [10, 20, 30, 40],
    }

    expected = pd.DataFrame(expected)

    pd.testing.assert_frame_equal(expected, result)


@pytest.mark.parametrize(
    ("base_dir", "image_path", "output_dir", "expected"),
    [
        # Absolute path, nested under base_dir, with file suffix
        (
            Path("/some/random/path"),
            Path("/some/random/path/images/test.spm"),
            Path("output/here"),
            Path("output/here/images/test/"),
        ),
        # Absolute path, nested under base_dir, with file suffix and multiple periods
        (
            Path("/some/random/path"),
            Path("/some/random/path/images/to.at.spm"),
            Path("output/here"),
            Path("output/here/images/to.at/"),
        ),
        # Absolute path, nested under base_dir, with file suffix
        (
            Path("/some/random/path"),
            Path("/some/random/path/images/today/test.spm"),
            Path("output/here"),
            Path("output/here/images/today/test/"),
        ),
        # Relative path, nested under base_dir, with file_suffix
        (
            Path("/some/random/path"),
            Path("images/test.spm"),
            Path("output/here"),
            Path("output/here/images/test"),
        ),
        # Relative path, nested (two deep) under base_dir, with file_suffix
        (
            Path("/some/random/path"),
            Path("images/today/test.spm"),
            Path("output/here"),
            Path("output/here/images/today/test"),
        ),
        # Relative path, nested under base_dir, no file suffix
        (
            Path("/some/random/path"),
            Path("images/"),
            Path("output/here"),
            Path("output/here/images/"),
        ),
        # Absolute path, nested under base_dir, output not nested under base_dir, with file_suffix
        (
            Path("/some/random/path"),
            Path("/some/random/path/images/test.spm"),
            Path("/different/absolute/path"),
            Path("/different/absolute/path/images/test"),
        ),
        # Absolute path, nested under base_dir, output not nested under base_dir, no file file_suffix
        (
            Path("/some/random/path"),
            Path("/some/random/path/images/"),
            Path("/different/absolute/path"),
            Path("/different/absolute/path/images/"),
        ),
        # Relative path, nested under base_dir, output not nested under base_dir, with file_suffix
        (
            Path("/some/random/path"),
            Path("images/test.spm"),
            Path("/an/absolute/path"),
            Path("/an/absolute/path/images/test"),
        ),
    ],
)
def test_get_out_path(image_path: Path, base_dir: Path, output_dir: Path, expected: Path) -> None:
    """Test output directories."""
    out_path = get_out_path(image_path, base_dir, output_dir)
    assert isinstance(out_path, Path)
    assert out_path == expected


def test_get_out_path_attributeerror() -> None:
    """Test get_out_path() raises AttribteError when passed a string instead of a Path() for image_path."""
    with pytest.raises(AttributeError):
        get_out_path(
            image_path="images/test.spm",
            base_dir=Path("/some/random/path"),
            output_dir=Path("output/here"),
        )


def test_save_folder_grainstats(tmp_path: Path) -> None:
    """Test a folder-wide grainstats file is made."""
    test_df = pd.DataFrame({"dummy1": [1, 2, 3], "dummy2": ["a", "b", "c"]})
    input_path = tmp_path / "minicircle"
    test_df["basename"] = input_path
    out_path = tmp_path / "subfolder"
    Path.mkdir(out_path, parents=True)
    save_folder_grainstats(out_path, input_path, test_df)
    assert Path(out_path / "processed" / "folder_grainstats.csv").exists()


def test_load_scan_spm(load_scan_spm: LoadScans) -> None:
    """Test loading of Bruker .spm file."""
    load_scan_spm.img_path = load_scan_spm.img_paths[0]
    load_scan_spm.filename = load_scan_spm.img_paths[0].stem
    image, px_to_nm_scaling = load_scan_spm.load_spm()
    assert isinstance(image, np.ndarray)
    assert image.shape == (1024, 1024)
    assert image.sum() == 30695369.188316286
    assert isinstance(px_to_nm_scaling, float)
    assert px_to_nm_scaling == 0.4940029296875


def test_load_scan_ibw(load_scan_ibw: LoadScans) -> None:
    """Test loading of Igor binarywave .ibw file."""
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
    assert isinstance(image, np.ndarray)
    assert image.shape == (256, 256)
    assert image.sum() == 286598232.9308627
    assert isinstance(px_to_nm_scaling, float)
    assert px_to_nm_scaling == 1.2770176335964876


def test_load_scan_gwy(load_scan_gwy: LoadScans) -> None:
    """Test loading of a .gwy file."""
    load_scan_gwy.img_path = load_scan_gwy.img_paths[0]
    load_scan_gwy.filename = load_scan_gwy.img_paths[0].stem
    image, px_to_nm_scaling = load_scan_gwy.load_gwy()
    assert isinstance(image, np.ndarray)
    assert image.shape == (512, 512)
    assert image.sum() == 33836850.232917726
    assert isinstance(px_to_nm_scaling, float)
    assert px_to_nm_scaling == 0.8468632812499975


def test_load_scan_asd_file_not_found() -> None:
    """Test file not found exception is raised when loading non existent .ASD file."""
    load_scan_asd = LoadScans([Path("file_does_not_exist.asd")], channel="TP")
    load_scan_asd.img_path = load_scan_asd.img_paths[0]
    load_scan_asd.filename = load_scan_asd.img_paths[0].stem
    with pytest.raises(FileNotFoundError):
        load_scan_asd.load_asd()


def test_load_scan_asd(load_scan_asd: LoadScans) -> None:
    """Test loading of a .asd file."""
    load_scan_asd.img_path = load_scan_asd.img_paths[0]
    load_scan_asd.filename = load_scan_asd.img_paths[0].stem
    frames, px_to_nm_scaling = load_scan_asd.load_asd()
    assert isinstance(frames, np.ndarray)
    assert frames.shape == (197, 200, 200)
    assert frames.sum() == -1368044348.3393068
    assert isinstance(px_to_nm_scaling, float)
    assert px_to_nm_scaling == 2.0


def test_load_scan_topostats(load_scan_topostats: LoadScans) -> None:
    """Test loading of a .topostats file."""
    load_scan_topostats.img_path = load_scan_topostats.img_paths[0]
    load_scan_topostats.filename = load_scan_topostats.img_paths[0].stem
    image, px_to_nm_scaling = load_scan_topostats.load_topostats()
    grain_masks = load_scan_topostats.grain_masks
    above_grain_mask = grain_masks["above"]
    grain_trace_data = load_scan_topostats.grain_trace_data
    assert isinstance(image, np.ndarray)
    assert image.shape == (1024, 1024)
    assert image.sum() == 184140.8593819073
    assert isinstance(px_to_nm_scaling, float)
    assert px_to_nm_scaling == 0.4940029296875
    # Check that the grain mask is loaded correctly
    assert isinstance(above_grain_mask, np.ndarray)
    assert above_grain_mask.sum() == 633746
    assert isinstance(grain_trace_data, dict)
    assert grain_trace_data.keys() == {"above"}


def test_gwy_read_object(load_scan_dummy: LoadScans) -> None:
    """Test reading an object of a `.gwy` file object from an open binary file."""
    with Path.open(RESOURCES / "IO_binary_file.bin", "rb") as open_binary_file:  # pylint: disable=unspecified-encoding
        open_binary_file.seek(19)
        test_dict = {}
        load_scan_dummy._gwy_read_object(open_file=open_binary_file, data_dict=test_dict)

        assert list(test_dict.keys()) == ["test component", "test object component"]
        assert list(test_dict.values()) == [500, {"test nested component": 3}]


def test_gwy_read_component(load_scan_dummy: LoadScans) -> None:
    """Tests reading a component of a `.gwy` file object from an open binary file."""
    with Path.open(RESOURCES / "IO_binary_file.bin", "rb") as open_binary_file:  # pylint: disable=unspecified-encoding
        open_binary_file.seek(55)
        test_dict = {}
        byte_size = load_scan_dummy._gwy_read_component(
            initial_byte_pos=55, open_file=open_binary_file, data_dict=test_dict
        )
        print(test_dict.items())
        print(test_dict.values())
        assert byte_size == 73
        assert list(test_dict.keys()) == ["test object component"]
        assert list(test_dict.values()) == [{"test nested component": 3}]


@patch("pySPM.SPM.SPM_image.pxs")
@pytest.mark.parametrize(
    ("unit", "x", "y", "expected_px2nm"),
    [
        pytest.param("mm", 0.01, 0.01, 10000, id="mm units; square"),
        pytest.param("um", 1.5, 1.5, 1500, id="um units; square"),
        pytest.param("nm", 50, 50, 50, id="nm units; square"),
        pytest.param("pm", 233, 233, 0.233, id="pm units; square"),
        pytest.param("pm", 1, 512, 0.001, id="pm units; rectangular (thin)"),
        pytest.param("pm", 512, 1, 0.512, id="pm units; rectangular (tall)"),
    ],
)
def test__spm_pixel_to_nm_scaling(
    mock_pxs,
    load_scan_spm: LoadScans,
    spm_channel_data: pySPM.SPM.SPM_image,
    unit: str,
    x: int,
    y: int,
    expected_px2nm: float,
) -> None:
    """Test extraction of pixels to nanometer scaling."""
    mock_pxs.return_value = [(x, unit), (y, unit)]  # issue is that pxs is a func that returns the data
    result = load_scan_spm._spm_pixel_to_nm_scaling(spm_channel_data)
    assert result == expected_px2nm


@pytest.mark.parametrize(
    ("load_scan_object", "length", "image_shape", "image_sum", "filename", "pixel_to_nm_scaling"),
    [
        ("load_scan_spm", 1, (1024, 1024), 30695369.188316286, "minicircle", 0.4940029296875),
        ("load_scan_ibw", 1, (512, 512), -218091520.0, "minicircle2", 1.5625),
        ("load_scan_jpk", 1, (256, 256), 286598232.9308627, "file", 1.2770176335964876),
        ("load_scan_gwy", 1, (512, 512), 33836850.232917726, "file", 0.8468632812499975),
        ("load_scan_topostats", 1, (1024, 1024), 184140.8593819073, "file", 0.4940029296875),
        ("load_scan_asd", 197, (200, 200), -12843725.967220962, "file_122", 2.0),
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
    assert isinstance(scan.img_dict[filename]["image_original"], np.ndarray)
    assert scan.img_dict[filename]["image_original"].shape == image_shape
    assert scan.img_dict[filename]["image_original"].sum() == image_sum
    assert isinstance(scan.img_dict[filename]["img_path"], Path)
    assert scan.img_dict[filename]["img_path"] == RESOURCES / filename
    assert isinstance(scan.img_dict[filename]["pixel_to_nm_scaling"], float)
    assert scan.img_dict[filename]["pixel_to_nm_scaling"] == pixel_to_nm_scaling


@pytest.mark.parametrize(
    ("x", "y", "log_msg"),
    [
        (100, 100, "Image added to processing"),
        (9, 100, "Skipping, image too small"),
        (100, 9, "Skipping, image too small"),
    ],
)
def test_load_scan_get_data_check_image_size_and_add_to_dict(
    load_scan_spm: LoadScans, x: int, y: int, log_msg: str, caplog, tmp_path
) -> None:
    """Test errors are raised when images that are too small are passed."""
    load_scan_spm.filename = "minicircle"
    load_scan_spm.img_path = tmp_path
    load_scan_spm.image = np.ndarray((x, y))
    load_scan_spm._check_image_size_and_add_to_dict(image=load_scan_spm.image, filename=load_scan_spm.filename)
    assert log_msg in caplog.text


def test_save_pkl(summary_config: dict, tmp_path) -> None:
    """Test saving a pickle."""
    outfile = tmp_path / "test.pkl"
    save_pkl(outfile=outfile, to_pkl=summary_config)
    assert outfile.exists()


def test_load_pkl() -> None:
    """Test loading a pickle."""
    infile = RESOURCES / "test.pkl"
    small_dictionary = load_pkl(infile)
    assert isinstance(small_dictionary, dict)


def test_dict_to_hdf5_all_together_group_path_default(tmp_path: Path) -> None:
    """Test saving a nested dictionary with arrays to HDF5 format with group path as default."""
    to_save = {
        "a": 1,
        "b": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        "c": "test",
        "d": {"e": 1, "f": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), "g": "test"},
        "h": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    }

    expected = {
        "a": 1,
        "b": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        "c": "test",
        "d": {"e": 1, "f": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), "g": "test"},
        "h": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
    }

    group_path = "/"

    with h5py.File(tmp_path / "hdf5_file_nested_with_arrays_group_path_standard.hdf5", "w") as f:
        dict_to_hdf5(open_hdf5_file=f, group_path=group_path, dictionary=to_save)

    # Load it back in and check if the dictionary is the same
    with h5py.File(tmp_path / "hdf5_file_nested_with_arrays_group_path_standard.hdf5", "r") as f:
        # Check keys are the same
        assert list(f.keys()) == list(expected.keys())
        assert f["a"][()] == expected["a"]
        np.testing.assert_array_equal(f["b"][()], expected["b"])
        # pylint thinks that f["c"] is a group but it is a bytes object that can be decoded
        # pylint: disable=no-member
        assert f["c"][()].decode("utf-8") == expected["c"]
        assert f["d"]["e"][()] == expected["d"]["e"]
        np.testing.assert_array_equal(f["d"]["f"][()], expected["d"]["f"])
        assert f["d"]["g"][()].decode("utf-8") == expected["d"]["g"]
        np.testing.assert_array_equal(f["h"][()], expected["h"])


def test_dict_to_hdf5_all_together_group_path_non_standard(tmp_path: Path) -> None:
    """Test saving a nested dictionary with arrays to HDF5 format with a non-standard group path."""
    to_save = {
        "a": 1,
        "b": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        "c": "test",
        "d": {"e": 1, "f": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), "g": "test"},
        "h": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    }

    expected = {
        "d": {
            "a": 1,
            "b": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            "c": "test",
            "d": {"e": 1, "f": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), "g": "test"},
            "h": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        }
    }

    group_path = "/d/"

    with h5py.File(tmp_path / "hdf5_file_all_together_group_path_nonstandard.hdf5", "w") as f:
        dict_to_hdf5(open_hdf5_file=f, group_path=group_path, dictionary=to_save)

    # Load it back in and check if the dictionary is the same
    with h5py.File(tmp_path / "hdf5_file_all_together_group_path_nonstandard.hdf5", "r") as f:
        # Check keys are the same
        assert list(f.keys()) == list(expected.keys())
        assert f["d"]["a"][()] == expected["d"]["a"]
        np.testing.assert_array_equal(f["d"]["b"][()], expected["d"]["b"])
        assert f["d"]["c"][()].decode("utf-8") == expected["d"]["c"]
        assert f["d"]["d"]["e"][()] == expected["d"]["d"]["e"]
        np.testing.assert_array_equal(f["d"]["d"]["f"][()], expected["d"]["d"]["f"])
        assert f["d"]["d"]["g"][()].decode("utf-8") == expected["d"]["d"]["g"]
        np.testing.assert_array_equal(f["d"]["h"][()], expected["d"]["h"])


def test_dict_to_hdf5_int(tmp_path: Path) -> None:
    """Test saving a dictionary with an integer to HDF5 format."""
    to_save = {"a": 1, "b": 2}
    expected = {"a": 1, "b": 2}
    group_path = "/"

    with h5py.File(tmp_path / "hdf5_file_int.hdf5", "w") as f:
        dict_to_hdf5(open_hdf5_file=f, group_path=group_path, dictionary=to_save)

    # Load it back in and check if the dictionary is the same
    with h5py.File(tmp_path / "hdf5_file_int.hdf5", "r") as f:
        # Check keys are the same
        assert list(f.keys()) == list(expected.keys())
        assert f["a"][()] == expected["a"]
        assert f["b"][()] == expected["b"]


def test_dict_to_hdf5_float(tmp_path: Path) -> None:
    """Test saving a dictionary with a float to HDF5 format."""
    to_save = {"a": 0.01, "b": 0.02}
    expected = {"a": 0.01, "b": 0.02}
    group_path = "/"

    with h5py.File(tmp_path / "hdf5_file_float.hdf5", "w") as f:
        dict_to_hdf5(open_hdf5_file=f, group_path=group_path, dictionary=to_save)

    # Load it back in and check if the dictionary is the same
    with h5py.File(tmp_path / "hdf5_file_float.hdf5", "r") as f:
        # Check keys are the same
        assert list(f.keys()) == list(expected.keys())
        assert f["a"][()] == expected["a"]
        assert f["b"][()] == expected["b"]


def test_dict_to_hdf5_str(tmp_path: Path) -> None:
    """Test saving a dictionary with a string to HDF5 format."""
    to_save = {"a": "test", "b": "test2"}
    expected = {"a": "test", "b": "test2"}
    group_path = "/"

    with h5py.File(tmp_path / "hdf5_file_str.hdf5", "w") as f:
        dict_to_hdf5(open_hdf5_file=f, group_path=group_path, dictionary=to_save)

    # Load it back in and check if the dictionary is the same
    with h5py.File(tmp_path / "hdf5_file_str.hdf5", "r") as f:
        # Check keys are the same
        assert list(f.keys()) == list(expected.keys())
        # pylint thinks that f["a"] is a group but it is a bytes object that can be decoded
        # pylint: disable=no-member
        assert f["a"][()].decode("utf-8") == expected["a"]
        # pylint thinks that f["b"] is a group but it is a bytes object that can be decoded
        # pylint: disable=no-member
        assert f["b"][()].decode("utf-8") == expected["b"]


def test_dict_to_hdf5_nested_lists(tmp_path: Path) -> None:
    """Test saving a nested dictionary with lists to HDF5 format."""
    to_save = {
        "list": [1, 2, 3],
        "2d list": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    }

    expected = {
        "list": np.array([1, 2, 3]),
        "2d list": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
    }

    group_path = "/"

    with h5py.File(tmp_path / "hdf5_file_nested_lists.hdf5", "w") as f:
        dict_to_hdf5(open_hdf5_file=f, group_path=group_path, dictionary=to_save)

    # Load it back in and check if the dictionary is the same
    with h5py.File(tmp_path / "hdf5_file_nested_lists.hdf5", "r") as f:
        # Check keys are the same
        assert sorted(f.keys()) == sorted(expected.keys())
        np.testing.assert_array_equal(f["list"][()], expected["list"])
        np.testing.assert_array_equal(f["2d list"][()], expected["2d list"])


def test_dict_to_hdf5_nested_dict(tmp_path: Path) -> None:
    """Test saving a nested dictionary to HDF5 format."""
    to_save = {
        "a": 1,
        "b": 2,
        "c": {"d": 3, "e": 4},
    }

    expected = {
        "a": 1,
        "b": 2,
        "c": {
            "d": 3,
            "e": 4,
        },
    }

    group_path = "/"

    with h5py.File(tmp_path / "hdf5_file_nested_dict.hdf5", "w") as f:
        dict_to_hdf5(open_hdf5_file=f, group_path=group_path, dictionary=to_save)

    # Load it back in and check if the dictionary is the same
    with h5py.File(tmp_path / "hdf5_file_nested_dict.hdf5", "r") as f:
        # Check keys are the same
        assert sorted(f.keys()) == sorted(expected.keys())
        assert f["a"][()] == expected["a"]
        assert f["b"][()] == expected["b"]
        assert sorted(f["c"].keys()) == sorted(expected["c"].keys())
        assert f["c"]["d"][()] == expected["c"]["d"]
        assert f["c"]["e"][()] == expected["c"]["e"]


def test_dict_to_hdf5_nested_dict_group_path(tmp_path: Path) -> None:
    """Test saving a nested dictionary to HDF5 format with a non-standard group path."""
    to_save = {
        "a": 1,
        "b": 2,
        "c": {"d": 3, "e": 4},
    }

    expected = {
        "nested": {
            "a": 1,
            "b": 2,
            "c": {
                "d": 3,
                "e": 4,
            },
        }
    }

    group_path = "/nested/"

    with h5py.File(tmp_path / "hdf5_file_nested_dict_group_path.hdf5", "w") as f:
        dict_to_hdf5(open_hdf5_file=f, group_path=group_path, dictionary=to_save)

    # Load it back in and check if the dictionary is the same
    with h5py.File(tmp_path / "hdf5_file_nested_dict_group_path.hdf5", "r") as f:
        # Check keys are the same
        assert sorted(f.keys()) == sorted(expected.keys())
        assert f["nested"]["a"][()] == expected["nested"]["a"]
        assert f["nested"]["b"][()] == expected["nested"]["b"]
        assert sorted(f["nested"]["c"].keys()) == sorted(expected["nested"]["c"].keys())
        assert f["nested"]["c"]["d"][()] == expected["nested"]["c"]["d"]
        assert f["nested"]["c"]["e"][()] == expected["nested"]["c"]["e"]


def test_dict_to_hdf5_list(tmp_path: Path) -> None:
    """Test saving a dictionary with a list to HDF5 format."""
    to_save = {"list": [1, 2, 3]}
    expected = {"list": np.array([1, 2, 3])}
    group_path = "/"

    with h5py.File(tmp_path / "hdf5_file_list.hdf5", "w") as f:
        dict_to_hdf5(open_hdf5_file=f, group_path=group_path, dictionary=to_save)

    # Load it back in and check if the dictionary is the same
    with h5py.File(tmp_path / "hdf5_file_list.hdf5", "r") as f:
        # Check keys are the same
        assert sorted(f.keys()) == sorted(expected.keys())
        np.testing.assert_array_equal(f["list"][()], expected["list"])


def test_hdf5_to_dict_all_together_group_path_default(tmp_path: Path) -> None:
    """Test loading a nested dictionary with arrays from HDF5 format with group path as default."""
    to_save = {
        "a": 1,
        "b": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        "c": "test",
        "d": {"e": 1, "f": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), "g": "test"},
    }

    group_path = "/"

    # Manually save the dictionary to HDF5 format
    with h5py.File(tmp_path / "hdf5_file_nested_with_arrays_group_path_standard.hdf5", "w") as f:
        # Write the datasets and groups to the file without using the dict_to_hdf5 function
        f.create_dataset("a", data=to_save["a"])
        f.create_dataset("b", data=to_save["b"])
        f.create_dataset("c", data=to_save["c"])
        d = f.create_group("d")
        d.create_dataset("e", data=to_save["d"]["e"])
        d.create_dataset("f", data=to_save["d"]["f"])
        d.create_dataset("g", data=to_save["d"]["g"])

    # Load it back in and check if the dictionary is the same
    with h5py.File(tmp_path / "hdf5_file_nested_with_arrays_group_path_standard.hdf5", "r") as f:
        result = hdf5_to_dict(open_hdf5_file=f, group_path=group_path)

    np.testing.assert_equal(result, to_save)


def test_hdf5_to_dict_all_together_group_path_non_standard(tmp_path: Path) -> None:
    """Test loading a nested dictionary with arrays from HDF5 format with a non-standard group path."""
    to_save = {
        "a": 1,
        "b": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        "c": "test",
        "d": {"e": 1, "f": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), "g": "test"},
    }

    group_path = "/d/"

    expected = {
        "e": 1,
        "f": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        "g": "test",
    }

    # Manually save the dictionary to HDF5 format
    with h5py.File(tmp_path / "hdf5_file_all_together_group_path_nonstandard.hdf5", "w") as f:
        # Write the datasets and groups to the file without using the dict_to_hdf5 function
        f.create_dataset("a", data=to_save["a"])
        f.create_dataset("b", data=to_save["b"])
        f.create_dataset("c", data=to_save["c"])
        d = f.create_group("d")
        d.create_dataset("e", data=to_save["d"]["e"])
        d.create_dataset("f", data=to_save["d"]["f"])
        d.create_dataset("g", data=to_save["d"]["g"])

    # Load it back in and check if the dictionary is the same
    with h5py.File(tmp_path / "hdf5_file_all_together_group_path_nonstandard.hdf5", "r") as f:
        result = hdf5_to_dict(open_hdf5_file=f, group_path=group_path)

    np.testing.assert_equal(result, expected)


def test_hdf5_to_dict_int(tmp_path: Path) -> None:
    """Test loading a dictionary with an integer from HDF5 format."""
    to_save = {"a": 1, "b": 2}

    group_path = "/"

    # Manually save the dictionary to HDF5 format
    with h5py.File(tmp_path / "hdf5_file_int.hdf5", "w") as f:
        # Write the datasets and groups to the file without using the dict_to_hdf5 function
        f.create_dataset("a", data=to_save["a"])
        f.create_dataset("b", data=to_save["b"])

    # Load it back in and check if the dictionary is the same
    with h5py.File(tmp_path / "hdf5_file_int.hdf5", "r") as f:
        result = hdf5_to_dict(open_hdf5_file=f, group_path=group_path)

    np.testing.assert_equal(result, to_save)


def test_hdf5_to_dict_float(tmp_path: Path) -> None:
    """Test loading a dictionary with a float from HDF5 format."""
    to_save = {"a": 0.01, "b": 0.02}

    group_path = "/"

    # Manually save the dictionary to HDF5 format
    with h5py.File(tmp_path / "hdf5_file_float.hdf5", "w") as f:
        # Write the datasets and groups to the file without using the dict_to_hdf5 function
        f.create_dataset("a", data=to_save["a"])
        f.create_dataset("b", data=to_save["b"])

    # Load it back in and check if the dictionary is the same
    with h5py.File(tmp_path / "hdf5_file_float.hdf5", "r") as f:
        result = hdf5_to_dict(open_hdf5_file=f, group_path=group_path)

    np.testing.assert_equal(result, to_save)


def test_hdf5_to_dict_str(tmp_path: Path) -> None:
    """Test loading a dictionary with a string from HDF5 format."""
    to_save = {"a": "test", "b": "test2"}

    group_path = "/"

    # Manually save the dictionary to HDF5 format
    with h5py.File(tmp_path / "hdf5_file_str.hdf5", "w") as f:
        # Write the datasets and groups to the file without using the dict_to_hdf5 function
        f.create_dataset("a", data=to_save["a"])
        f.create_dataset("b", data=to_save["b"])

    # Load it back in and check if the dictionary is the same
    with h5py.File(tmp_path / "hdf5_file_str.hdf5", "r") as f:
        result = hdf5_to_dict(open_hdf5_file=f, group_path=group_path)

    np.testing.assert_equal(result, to_save)


def test_hdf5_to_dict_dict_nested_dict(tmp_path: Path) -> None:
    """Test loading a nested dictionary from HDF5 format."""
    to_save = {
        "a": 1,
        "b": 2,
        "c": {"d": 3, "e": 4},
    }

    group_path = "/"

    # Manually save the dictionary to HDF5 format
    with h5py.File(tmp_path / "hdf5_file_nested_dict.hdf5", "w") as f:
        # Write the datasets and groups to the file without using the dict_to_hdf5 function
        f.create_dataset("a", data=to_save["a"])
        f.create_dataset("b", data=to_save["b"])
        c = f.create_group("c")
        c.create_dataset("d", data=to_save["c"]["d"])
        c.create_dataset("e", data=to_save["c"]["e"])

    # Load it back in and check if the dictionary is the same
    with h5py.File(tmp_path / "hdf5_file_nested_dict.hdf5", "r") as f:
        result = hdf5_to_dict(open_hdf5_file=f, group_path=group_path)

    np.testing.assert_equal(result, to_save)


def test_hdf5_to_dict_nested_dict_group_path(tmp_path: Path) -> None:
    """Test loading a nested dictionary from HDF5 format with a non-standard group path."""
    to_save = {
        "a": 1,
        "b": 2,
        "c": {"d": 3, "e": 4},
    }

    group_path = "/c/"

    expected = {
        "d": 3,
        "e": 4,
    }

    # Manually save the dictionary to HDF5 format
    with h5py.File(tmp_path / "hdf5_file_nested_dict_group_path.hdf5", "w") as f:
        # Write the datasets and groups to the file without using the dict_to_hdf5 function
        f.create_dataset("a", data=to_save["a"])
        f.create_dataset("b", data=to_save["b"])
        c = f.create_group("c")
        c.create_dataset("d", data=to_save["c"]["d"])
        c.create_dataset("e", data=to_save["c"]["e"])

    # Load it back in and check if the dictionary is the same
    with h5py.File(tmp_path / "hdf5_file_nested_dict_group_path.hdf5", "r") as f:
        result = hdf5_to_dict(open_hdf5_file=f, group_path=group_path)

    np.testing.assert_equal(result, expected)


@pytest.mark.parametrize(
    ("image", "pixel_to_nm_scaling", "grain_mask_above", "grain_mask_below", "grain_trace_data"),
    [
        pytest.param(
            np.arange(0, 100).reshape(10, 10),
            3.14159265,
            None,
            np.zeros((10, 10)),
            {
                "above": {
                    "ordered_traces": {
                        "0": np.array(
                            [
                                [0, 1],
                                [1, 0],
                                [2, 2],
                            ]
                        ),
                        "1": np.array(
                            [
                                [0, 0],
                                [2, 1],
                                [3, 0],
                            ]
                        ),
                    },
                    "cropped_images": {
                        "0": np.array([[0, 1, 2], [1, 2, 3], [2, 2, 1]]),
                        "1": np.array([[0, 1, 3], [2, 2, 4], [3, 4, 5]]),
                    },
                    "ordered_trace_heights": {
                        "0": np.array([5, 2, 3]),
                        "1": np.array([5, 7, 10]),
                    },
                    "ordered_trace_cumulative_distances": {
                        "0": np.array([0, 1.41, 2.41]),
                        "1": np.array([0, 1, 2]),
                    },
                    "splined_traces": {
                        "0": np.array(
                            [
                                [0, 1],
                                [1, 0],
                                [2, 2],
                            ]
                        ),
                        "1": np.array(
                            [
                                [0, 0],
                                [2, 1],
                                [3, 0],
                            ]
                        ),
                    },
                },
                "below": {
                    "ordered_traces": {
                        "0": np.array(
                            [
                                [0, 1],
                                [1, 0],
                                [2, 2],
                            ]
                        ),
                        "1": np.array(
                            [
                                [0, 0],
                                [2, 1],
                                [3, 0],
                            ]
                        ),
                    },
                    "cropped_images": {
                        "0": np.array([[0, 1, 2], [1, 2, 3], [2, 2, 1]]),
                        "1": np.array([[0, 1, 3], [2, 2, 4], [3, 4, 5]]),
                    },
                    "ordered_trace_heights": {
                        "0": np.array([5, 2, 3]),
                        "1": np.array([5, 7, 10]),
                    },
                    "ordered_trace_cumulative_distances": {
                        "0": np.array([0, 1.41, 2.41]),
                        "1": np.array([0, 1, 2]),
                    },
                    "splined_traces": {
                        "0": np.array(
                            [
                                [0, 1],
                                [1, 0],
                                [2, 2],
                            ]
                        ),
                        "1": np.array(
                            [
                                [0, 0],
                                [2, 1],
                                [3, 0],
                            ]
                        ),
                    },
                },
            },
            id="below_grain_mask_with_grain_trace_data",
        ),
        pytest.param(
            np.arange(0, 100).reshape(10, 10),
            3.14159265,
            np.zeros((10, 10)),
            None,
            None,
            id="above_grain_mask_without_grain_trace_data",
        ),
        pytest.param(
            np.arange(0, 100).reshape(10, 10),
            3.14159265,
            np.zeros((10, 10)),
            np.zeros((10, 10)),
            None,
            id="above_and_below_grain_masks_without_grain_trace_data",
        ),
    ],
)
def test_save_and_load_topostats_file(
    load_scan_topostats_test_file: LoadScans,
    tmp_path: Path,
    image: np.ndarray,
    pixel_to_nm_scaling: float,
    grain_mask_above: np.ndarray,
    grain_mask_below: np.ndarray,
    grain_trace_data: dict,
) -> None:
    """Test saving a .topostats file."""
    topostats_object = {
        "image_flattened": image,
        "pixel_to_nm_scaling": pixel_to_nm_scaling,
        "grain_masks": {"above": grain_mask_above, "below": grain_mask_below},
        "grain_trace_data": grain_trace_data,
    }

    save_topostats_file(
        output_dir=tmp_path,
        filename="topostats_file_test.topostats",
        topostats_object=topostats_object,
    )

    # Load the saved .topostats file using LoadScans
    loadscans = load_scan_topostats_test_file
    loadscans.get_data()
    topostats_data = loadscans.img_dict["topostats_file_test"]

    assert set(topostats_data.keys()) == {
        "image_original",
        "img_path",
        "filename",
        "grain_masks",
        "grain_trace_data",
        "image_flattened",
        "pixel_to_nm_scaling",
    }

    np.testing.assert_array_equal(image, topostats_data["image_original"])
    assert pixel_to_nm_scaling == topostats_data["pixel_to_nm_scaling"]
    if grain_mask_above is not None:
        np.testing.assert_array_equal(grain_mask_above, topostats_data["grain_masks"]["above"])
    if grain_mask_below is not None:
        np.testing.assert_array_equal(grain_mask_below, topostats_data["grain_masks"]["below"])
    if grain_trace_data is not None:
        np.testing.assert_equal(grain_trace_data, topostats_data["grain_trace_data"])
