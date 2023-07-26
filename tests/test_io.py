"""Tests of IO"""
from pathlib import Path
from unittest import TestCase

from datetime import datetime
import numpy as np
import pandas as pd
import pytest
import h5py

from topostats.io import (
    read_yaml,
    get_date_time,
    write_config_with_comments,
    write_yaml,
    save_array,
    load_array,
    find_files,
    get_out_path,
    path_to_str,
    save_folder_grainstats,
    LoadScans,
    save_pkl,
    load_pkl,
    read_null_terminated_string,
    read_u32i,
    read_64d,
    read_gwy_component_dtype,
    read_char,
    get_relative_paths,
    convert_basename_to_relative_paths,
    save_topostats_file,
)

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


def test_get_date_time() -> None:
    """Test the fetching of a formatted date and time string."""

    assert datetime.strptime(get_date_time(), "%Y-%m-%d %H:%M:%S")


def test_read_yaml() -> None:
    """Test reading of YAML file."""
    sample_config = read_yaml(RESOURCES / "test.yaml")

    TestCase().assertDictEqual(sample_config, CONFIG)


def test_write_config_with_comments(tmp_path: Path) -> None:
    """
    Test that the function write_yaml_with_comments successfully writes the default
    config with comments to a file.
    """

    # Read default config with comments
    with open(BASE_DIR / "topostats" / "default_config.yaml", encoding="utf-8") as f:
        default_config_string = f.read()

    # Write default config with comments to file
    write_config_with_comments(config=default_config_string, output_dir=tmp_path, filename="test_config_with_comments")

    # Read the written config
    with open(tmp_path / "test_config_with_comments.yaml", encoding="utf-8") as f:
        written_config = f.read()

    # Validate that the written config has comments in it
    assert default_config_string in written_config
    assert "Config file generated" in written_config
    assert "For more information on configuration and how to use it" in written_config


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
    CONFIG_PATH = {"this": "is", "a": "test", "with": tmp_path, "and": {"nested": tmp_path / "nested"}}
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


@pytest.mark.parametrize("non_existant_file", [("does_not_exist.npy"), ("does_not_exist.np"), ("does_not_exist.csv")])
def test_load_array_file_not_found(non_existant_file: str) -> None:
    """Test exceptions when trying to load arrays that don't exist."""
    with pytest.raises(FileNotFoundError):
        assert load_array(non_existant_file)


def test_find_files() -> None:
    """Test finding images"""
    found_images = find_files(base_dir="tests/", file_ext=".spm")

    assert isinstance(found_images, list)
    assert len(found_images) == 1
    assert isinstance(found_images[0], Path)
    assert "minicircle.spm" in str(found_images[0])


def test_read_null_terminated_string() -> None:
    """Test reading a null terminated string from a binary file."""
    with open(RESOURCES / "IO_binary_file.bin", "rb") as open_binary_file:
        value = read_null_terminated_string(open_binary_file)
        assert isinstance(value, str)
        assert value == "test"


def test_read_u32i() -> None:
    """Test reading an unsigned 32 bit integer from a binary file."""
    with open(RESOURCES / "IO_binary_file.bin", "rb") as open_binary_file:
        open_binary_file.seek(5)
        value = read_u32i(open_binary_file)
        assert isinstance(value, int)
        assert value == 32


def test_read_64d() -> None:
    """Test reading a 64-bit double from an open binary file."""
    with open(RESOURCES / "IO_binary_file.bin", "rb") as open_binary_file:
        open_binary_file.seek(9)
        value = read_64d(open_binary_file)
        assert isinstance(value, float)
        assert value == 3.141592653589793


def test_read_char() -> None:
    """Test reading a character from an open binary file."""
    with open(RESOURCES / "IO_binary_file.bin", "rb") as open_binary_file:
        open_binary_file.seek(17)
        value = read_char(open_binary_file)
        assert isinstance(value, str)
        assert value == "Z"


def test_read_gwy_component_dtype() -> None:
    """Test reading a data type of a `.gwy` file component from an open binary file."""
    with open(RESOURCES / "IO_binary_file.bin", "rb") as open_binary_file:
        open_binary_file.seek(18)
        value = read_gwy_component_dtype(open_binary_file)
        assert isinstance(value, str)
        assert value == "D"


@pytest.mark.parametrize(
    "input_paths, expected_paths",
    [
        ([Path("a/b/c/d"), Path("a/b/e/f"), Path("a/b/g"), Path("a/b/h")], ["c/d", "e/f", "g", "h"]),
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
    "base_dir, image_path, output_dir, expected",
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
            Path("/some/random/path/images/te.st.spm"),
            Path("output/here"),
            Path("output/here/images/te.st/"),
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
        (Path("/some/random/path"), Path("images/"), Path("output/here"), Path("output/here/images/")),
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
    """Test output directories"""
    out_path = get_out_path(image_path, base_dir, output_dir)
    assert isinstance(out_path, Path)
    assert out_path == expected


def test_get_out_path_attributeerror() -> None:
    """Test get_out_path() raises AttribteError when passed a string instead of a Path() for image_path."""
    with pytest.raises(AttributeError):
        get_out_path(image_path="images/test.spm", base_dir=Path("/some/random/path"), output_dir=Path("output/here"))


def test_save_folder_grainstats(tmp_path: Path) -> None:
    """Test a folder-wide grainstats file is made"""
    test_df = pd.DataFrame({"dummy1": [1, 2, 3], "dummy2": ["a", "b", "c"]})
    input_path = tmp_path / "minicircle"
    test_df["basename"] = input_path
    out_path = tmp_path / "subfolder"
    Path.mkdir(out_path, parents=True)
    save_folder_grainstats(out_path, input_path, test_df)
    assert Path(out_path / "processed" / "folder_grainstats.csv").exists()


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


def test_load_scan_topostats(load_scan_topostats: LoadScans) -> None:
    """Test loading of a .topostats file."""
    load_scan_topostats.img_path = load_scan_topostats.img_paths[0]
    load_scan_topostats.filename = load_scan_topostats.img_paths[0].stem
    image, px_to_nm_scaling = load_scan_topostats.load_topostats()
    grain_masks = load_scan_topostats.grain_masks
    above_grain_mask = grain_masks["above"]
    assert isinstance(image, np.ndarray)
    assert image.shape == (1024, 1024)
    assert image.sum() == 182067.12616107278
    assert isinstance(px_to_nm_scaling, float)
    assert px_to_nm_scaling == 0.4940029296875
    # Check that the grain mask is loaded correctly
    assert isinstance(above_grain_mask, np.ndarray)
    assert above_grain_mask.sum() == 635628


def test_gwy_read_object(load_scan_dummy: LoadScans) -> None:
    """Test reading an object of a `.gwy` file object from an open binary file."""
    with open(RESOURCES / "IO_binary_file.bin", "rb") as open_binary_file:
        open_binary_file.seek(19)
        test_dict = {}
        load_scan_dummy._gwy_read_object(open_file=open_binary_file, data_dict=test_dict)

        assert list(test_dict.keys()) == ["test component", "test object component"]
        assert list(test_dict.values()) == [500, {"test nested component": 3}]


def test_gwy_read_component(load_scan_dummy: LoadScans) -> None:
    """Tests reading a component of a `.gwy` file object from an open binary file."""
    with open(RESOURCES / "IO_binary_file.bin", "rb") as open_binary_file:
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
    "x, y, log_msg",
    [
        (100, 100, "Image added to processing"),
        (9, 100, "Skipping, image too small"),
        (100, 9, "Skipping, image too small"),
    ],
)
def test_load_scan_get_data_check_image_size_and_add_to_dict(
    load_scan: LoadScans, x: int, y: int, log_msg: str, caplog, tmp_path
) -> None:
    """Test errors are raised when images that are too small are passed."""
    load_scan.filename = "minicircle"
    load_scan.img_path = tmp_path
    load_scan.image = np.ndarray((x, y))
    load_scan._check_image_size_and_add_to_dict()
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


@pytest.mark.parametrize(
    "image, pixel_to_nm_scaling, grain_mask_above, grain_mask_below",
    [
        (
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            3.14159265,
            None,
            np.array([[0, 0, 0], [0, 1, 1], [0, 1, 0]]),
        ),
        (np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), 3.14159265, np.array([[0, 0, 0], [0, 1, 1], [0, 1, 0]]), None),
        (
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            3.14159265,
            np.array([[0, 0, 0], [0, 1, 1], [0, 1, 0]]),
            np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),
        ),
    ],
)
def test_save_topostats_file(
    tmp_path: Path,
    image: np.ndarray,
    pixel_to_nm_scaling: float,
    grain_mask_above: np.ndarray,
    grain_mask_below: np.ndarray,
) -> None:
    """Test saving a .topostats file"""

    topostats_object = {
        "image_flattened": image,
        "pixel_to_nm_scaling": pixel_to_nm_scaling,
        "grain_masks": {"above": grain_mask_above, "below": grain_mask_below},
    }

    save_topostats_file(
        output_dir=tmp_path, filename="topostats_file_test.topostats", topostats_object=topostats_object
    )

    with h5py.File(f"{tmp_path}/topostats_file_test.topostats", "r") as f:
        hdf5_file_keys = list(f.keys())
        topostats_file_version_read = f["topostats_file_version"][()]
        image_read = f["image"][:]
        pixel_to_nm_scaling_read = f["pixel_to_nm_scaling"][()]
        if grain_mask_above is not None:
            grain_mask_above_read = f["grain_masks/above"][:]
        if grain_mask_below is not None:
            grain_mask_below_read = f["grain_masks/below"][:]

    assert hdf5_file_keys == ["grain_masks", "image", "pixel_to_nm_scaling", "topostats_file_version"]
    assert 0.1 == topostats_file_version_read
    np.testing.assert_array_equal(image, image_read)
    assert pixel_to_nm_scaling == pixel_to_nm_scaling_read
    if grain_mask_above is not None:
        np.testing.assert_array_equal(grain_mask_above, grain_mask_above_read)
    if grain_mask_below is not None:
        np.testing.assert_array_equal(grain_mask_below, grain_mask_below_read)
