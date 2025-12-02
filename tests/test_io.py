"""Tests of IO."""

import json
import logging
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from topostats import grains
from topostats.classes import (
    DisorderedTrace,
    GrainCrop,
    Node,
    OrderedTrace,
    TopoStats,
)
from topostats.io import (
    LoadScans,
    _find_old_bruker_files,
    convert_basename_to_relative_paths,
    dict_almost_equal,
    dict_to_hdf5,
    dict_to_json,
    dict_to_topostats,
    find_files,
    get_date_time,
    get_out_path,
    get_relative_paths,
    hdf5_to_dict,
    lists_almost_equal,
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
# pylint: disable=too-many-arguments
# pylint: disable=too-many-lines
# pylint: disable=too-many-positional-arguments


def test_get_date_time() -> None:
    """Test the fetching of a formatted date and time string."""
    assert datetime.strptime(get_date_time(), "%Y-%m-%d %H:%M:%S")


def test_read_yaml() -> None:
    """Test reading of YAML file."""
    sample_config = read_yaml(RESOURCES / "test.yaml")

    assert sample_config == CONFIG


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
        pytest.param(
            {"a": np.nan},
            {"a": np.nan},
            0.0001,
            True,
            id="nan equal",
        ),
        pytest.param(
            {"a": [1.01, 2.01]},
            {"a": [1.0, 2.0]},
            0.1,
            True,
            id="list equal within tolerance",
        ),
        pytest.param(
            {"a": [1.01, 2.01]},
            {"a": [1.0, 2.0]},
            0.0001,
            False,
            id="list not equal within strict tolerance",
        ),
        pytest.param(
            {"a": 5.0, "b": 10.0},
            {"a": {"c": 5.0}, "b": 10.0},
            0.0001,
            False,
            id="dict's matching keys are of different types.",
        ),
    ],
)
def test_dict_almost_equal(dict1: dict, dict2: dict, tolerance: float, expected: bool) -> None:
    """Test that two dictionaries are almost equal."""
    assert dict_almost_equal(dict1, dict2, tolerance) == expected


@pytest.mark.parametrize(
    ("list1", "list2", "tolerance", "expected"),
    [
        pytest.param(
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            0.00001,
            True,
            id="list exactly equal",
        ),
        pytest.param(
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 4.0],
            0.00001,
            False,
            id="list not equal: value difference",
        ),
        pytest.param(
            [1.0, 2.0, 3.0],
            [1.0, 2.0],
            0.00001,
            False,
            id="lists not equal: different lengths",
        ),
        pytest.param(
            [1.00001, 2.00002, 3.00005],
            [1.0, 2.0, 3.0],
            0.01,
            True,
            id="list equal within tolerance",
        ),
        pytest.param(
            [[1, 2],[2, 3]],
            [[1, 2],[2, 3]],
            0.00001,
            True,
            id="lists equal: nested lists",
        ),
        pytest.param(
            [[1, 2],[2, 3]],
            [[1, 2, 3],[2, 3]],
            0.00001,
            False,
            id="lists not equal: nested lists",
        ),
    ],
)
def test_lists_almost_equal(list1: list, list2: list, tolerance: float, expected: bool) -> None:
    """Test the lists_almost_equal function."""
    assert lists_almost_equal(list1, list2, tolerance) == expected


def test_lists_almost_equal_notimplemented_error() -> None:
    """Test that lists_almost_equal raises NotImplementedError for illegal types."""
    with pytest.raises(NotImplementedError):
        lists_almost_equal([1, 2, {3}], [1, 2, {3}], 0.00001)


@pytest.mark.parametrize("non_existant_file", [("does_not_exist.npy"), ("does_not_exist.np"), ("does_not_exist.csv")])
def test_load_array_file_not_found(non_existant_file: str) -> None:
    """Test exceptions when trying to load arrays that don't exist."""
    with pytest.raises(FileNotFoundError):
        assert load_array(non_existant_file)


@pytest.mark.parametrize(
    ("file_ext", "filenames"),
    [
        pytest.param(".spm", ["minicircle.spm", "old_bruker.002", "old_bruker.004", "plasmids.spm"], id="spm"),
        pytest.param(".asd", ["file.asd", "minicircles.asd"], id="asd"),
        pytest.param(".gwy", ["file.gwy"], id="gwy"),
        pytest.param(".ibw", ["minicircle2.ibw"], id="ibw"),
        pytest.param(".jpk", ["file.jpk"], id="jpk"),
        pytest.param(".jpk-qi-image", ["file.jpk-qi-image"], id="jpk-qi-image"),
        pytest.param(".stp", ["file.stp"], id="stp"),
        pytest.param(".top", ["file.top"], id="top"),
        pytest.param(
            ".topostats",
            [
                "file.topostats",
                "minicircle_small.topostats",
                "minicircle_240.topostats",
                "process_scan_topostats_file_regtest.topostats",
                "catenane_post_disordered_tracing.topostats",
                "notebook3_image.topostats",
            ],
            id="topostats",
        ),
    ],
)
def test_find_files(file_ext: str, filenames: str | list[str]) -> None:
    """Test finding images based on file extension."""
    found_images = find_files(base_dir=RESOURCES, file_ext=file_ext)
    assert isinstance(found_images, list)
    assert len(found_images) == len(filenames)
    for image in found_images:
        assert isinstance(image, Path)
    # Sort expected and found images (converted to str) for comparison
    found_images = [str(image.name) for image in found_images]
    found_images.sort()
    filenames.sort()
    assert filenames == found_images


def test_find_old_bruker_files() -> None:
    """Test ``_find_old_bruker_files()``."""
    found_images = _find_old_bruker_files(base_dir=RESOURCES)
    assert isinstance(found_images, list)
    assert len(found_images) == 2
    for image in found_images:
        assert isinstance(image, Path)
    found_images = [str(image.name) for image in found_images]
    found_images.sort()
    expected = ["old_bruker.002", "old_bruker.004"]
    expected.sort()
    assert expected == found_images


@pytest.mark.parametrize(
    ("string_start_position", "expected_string"),
    [pytest.param(0, "test", id="utf8 string"), pytest.param(5, "µ ", id="ISO 8859-1 character")],
)
def test_read_null_terminated_string(string_start_position: int, expected_string: str) -> None:
    """Test reading a null terminated string from a binary file."""
    with Path.open(RESOURCES / "IO_binary_file.bin", "rb") as open_binary_file:  # pylint: disable=unspecified-encoding
        open_binary_file.seek(string_start_position)
        value = read_null_terminated_string(open_binary_file)
        assert isinstance(value, str)
        assert value == expected_string


def test_read_u32i() -> None:
    """Test reading an unsigned 32 bit integer from a binary file."""
    with Path.open(RESOURCES / "IO_binary_file.bin", "rb") as open_binary_file:  # pylint: disable=unspecified-encoding
        open_binary_file.seek(6)
        value = read_u32i(open_binary_file)
        assert isinstance(value, int)
        assert value == 32


def test_read_64d() -> None:
    """Test reading a 64-bit double from an open binary file."""
    with Path.open(RESOURCES / "IO_binary_file.bin", "rb") as open_binary_file:  # pylint: disable=unspecified-encoding
        open_binary_file.seek(10)
        value = read_64d(open_binary_file)
        assert isinstance(value, float)
        assert value == 3.141592653589793


def test_read_char() -> None:
    """Test reading a character from an open binary file."""
    with Path.open(RESOURCES / "IO_binary_file.bin", "rb") as open_binary_file:  # pylint: disable=unspecified-encoding
        open_binary_file.seek(18)
        value = read_char(open_binary_file)
        assert isinstance(value, str)
        assert value == "Z"


def test_read_gwy_component_dtype() -> None:
    """Test reading a data type of a `.gwy` file component from an open binary file."""
    with Path.open(RESOURCES / "IO_binary_file.bin", "rb") as open_binary_file:  # pylint: disable=unspecified-encoding
        open_binary_file.seek(19)
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
    save_folder_grainstats(out_path, input_path, test_df, "grainstats")
    assert Path(out_path / "processed" / "folder_grainstats.csv").exists()


def test_load_scan_spm(load_scan_spm: LoadScans) -> None:
    """Test loading of Bruker .spm file."""
    load_scan_spm.img_path = load_scan_spm.img_paths[0]
    load_scan_spm.filename = load_scan_spm.img_paths[0].stem
    image, px_to_nm_scaling = load_scan_spm.load_spm()
    assert isinstance(image, np.ndarray)
    assert image.shape == (1024, 1024)
    assert image.sum() == pytest.approx(30695369.188316286)
    assert isinstance(px_to_nm_scaling, float)
    assert px_to_nm_scaling == pytest.approx(0.4940029296875)


def test_load_scan_ibw(load_scan_ibw: LoadScans) -> None:
    """Test loading of Igor binarywave .ibw file."""
    load_scan_ibw.img_path = load_scan_ibw.img_paths[0]
    load_scan_ibw.filename = load_scan_ibw.img_paths[0].stem
    image, px_to_nm_scaling = load_scan_ibw.load_ibw()
    assert isinstance(image, np.ndarray)
    assert image.shape == (512, 512)
    assert image.sum() == pytest.approx(-218091520.0)
    assert isinstance(px_to_nm_scaling, float)
    assert px_to_nm_scaling == pytest.approx(1.5625)


def test_load_scan_jpk(load_scan_jpk: LoadScans) -> None:
    """Test loading of JPK Instruments .jpk file."""
    load_scan_jpk.img_path = load_scan_jpk.img_paths[0]
    load_scan_jpk.filename = load_scan_jpk.img_paths[0].stem
    image, px_to_nm_scaling = load_scan_jpk.load_jpk()
    assert isinstance(image, np.ndarray)
    assert image.shape == (256, 256)
    assert image.sum() == pytest.approx(219242202.8256843)
    assert isinstance(px_to_nm_scaling, float)
    assert px_to_nm_scaling == pytest.approx(1.2770176335964876)


def test_load_scan_jpk_qi(load_scan_jpk_qi: LoadScans) -> None:
    """Test loading of JPK Instruments .jpk-qi-image file."""
    load_scan_jpk_qi.img_path = load_scan_jpk_qi.img_paths[0]
    load_scan_jpk_qi.filename = load_scan_jpk_qi.img_paths[0].stem
    image, px_to_nm_scaling = load_scan_jpk_qi.load_jpk()
    assert isinstance(image, np.ndarray)
    assert image.shape == (100, 100)
    assert image.sum() == pytest.approx(31593146.16051172)
    assert isinstance(px_to_nm_scaling, float)
    assert px_to_nm_scaling == pytest.approx(4.999999999999986)


def test_load_scan_gwy(load_scan_gwy: LoadScans) -> None:
    """Test loading of a .gwy file."""
    load_scan_gwy.img_path = load_scan_gwy.img_paths[0]
    load_scan_gwy.filename = load_scan_gwy.img_paths[0].stem
    image, px_to_nm_scaling = load_scan_gwy.load_gwy()
    assert isinstance(image, np.ndarray)
    assert image.shape == (512, 512)
    assert image.sum() == pytest.approx(33836850.232917726)
    assert isinstance(px_to_nm_scaling, float)
    assert px_to_nm_scaling == pytest.approx(0.8468632812499975)


def test_load_scan_stp(load_scan_stp: LoadScans) -> None:
    """Test loading of a .stp file."""
    load_scan_stp.img_path = load_scan_stp.img_paths[0]
    load_scan_stp.filename = load_scan_stp.img_paths[0].stem
    image, px_to_nm_scaling = load_scan_stp.load_stp()
    assert isinstance(image, np.ndarray)
    assert image.shape == (512, 512)
    assert image.sum() == pytest.approx(-15070620.440757688)
    assert isinstance(px_to_nm_scaling, float)
    assert px_to_nm_scaling == pytest.approx(0.9765625)


def test_load_scan_top(load_scan_top: LoadScans) -> None:
    """Test loading of a .top file."""
    load_scan_top.img_path = load_scan_top.img_paths[0]
    load_scan_top.filename = load_scan_top.img_paths[0].stem
    image, px_to_nm_scaling = load_scan_top.load_top()
    assert isinstance(image, np.ndarray)
    assert image.shape == (512, 512)
    assert image.sum() == pytest.approx(6034386.429246264)
    assert isinstance(px_to_nm_scaling, float)
    assert px_to_nm_scaling == pytest.approx(0.9765625)


@pytest.mark.parametrize(
    ("non_existent_file", "channel"),
    [
        pytest.param("file_does_not_exist.asd", "TP", id="non-existent .asd"),
        pytest.param("file_does_not_exist.gwy", "ZSensor", id="non-existent .gwy"),
        pytest.param(
            "file_does_not_exist.ibw",
            "HeightTrace",
            id="non-existent .ibw",
        ),
        pytest.param("file_does_not_exist.jpk", "height_trace", id="non-existent .jpk"),
        pytest.param("file_does_not_exist.spm", "Height", id="non-existent .spm"),
        pytest.param("file_does_not_exist.topostats", "dummy_channel", id="non-existent .topostats"),
    ],
)
def test_get_data_file_not_found(non_existent_file: str, channel: str) -> None:
    """Test file not found exceptions are raised by .load_*() methods called by get_data()."""
    load_scan = LoadScans([Path(non_existent_file)], channel=channel)
    with pytest.raises(FileNotFoundError):
        load_scan.get_data()


def test_load_scan_asd(load_scan_asd: LoadScans) -> None:
    """Test loading of a .asd file."""
    load_scan_asd.img_path = load_scan_asd.img_paths[0]
    load_scan_asd.filename = load_scan_asd.img_paths[0].stem
    frames, px_to_nm_scaling = load_scan_asd.load_asd()
    assert isinstance(frames, np.ndarray)
    assert frames.shape == (197, 200, 200)
    assert frames.sum() == pytest.approx(-1368044348.3393068)
    assert isinstance(px_to_nm_scaling, float)
    assert px_to_nm_scaling == pytest.approx(2.0)


@pytest.mark.skip(
    reason="REDUNDANT : only works with '.topostats' loading as dictionaries. AFMReader loads HDF5 to"
    "dictionary but we convert to TopoStats objects so existing images can be processed. Only retain core"
    "fields, no tracing data is retained."
)
def test_load_scan_topostats_all(load_scan_topostats: LoadScans) -> None:
    """Test loading all data from a .topostats file."""
    load_scan_topostats.img_path = load_scan_topostats.img_paths[0]
    load_scan_topostats.filename = load_scan_topostats.img_paths[0].stem
    data = load_scan_topostats.load_topostats()
    above_grain_mask = data["grain_masks"]["above"]
    grain_trace_data = data["grain_trace_data"]
    assert isinstance(data["image"], np.ndarray)
    assert data["image"].shape == (1024, 1024)
    assert data["image"].sum() == pytest.approx(184140.8593819073)
    assert isinstance(data["pixel_to_nm_scaling"], float)
    assert data["pixel_to_nm_scaling"] == pytest.approx(0.4940029296875)
    # Check that the grain mask is loaded correctly
    assert isinstance(above_grain_mask, np.ndarray)
    assert above_grain_mask.sum() == 633746
    assert isinstance(grain_trace_data, dict)
    assert grain_trace_data.keys() == {"above"}


@pytest.mark.skip(
    reason="REDUNDANT : Loading topostats objects aligns with all other files types. See next test "
    "(test_load_scan_get_data) which now covers this."
)
@pytest.mark.parametrize(
    ("scan_fixture", "image_sum", "image_original_sum", "pixel_to_nm_scaling"),
    [
        pytest.param(
            "load_scan_topostats",
            184140.8593819073,
            30695369.188316286,
            0.4940029296875,
            id="old topostats_file_version 0.2",
        ),
    ],
)
def test_load_scan_topostats(
    scan_fixture: str, image_sum: float, image_original_sum: float, pixel_to_nm_scaling: float, request
) -> None:
    """Test loading different components from a .topostats file."""
    load_scan_topostats = request.getfixturevalue(scan_fixture)
    load_scan_topostats.img_path = load_scan_topostats.img_paths[0]
    load_scan_topostats.filename = load_scan_topostats.img_paths[0].stem
    topostats_object = load_scan_topostats.load_topostats()
    assert isinstance(topostats_object, TopoStats)
    assert isinstance(topostats_object.image, np.ndarray)
    assert topostats_object.image.shape == (1024, 1024)
    assert topostats_object.image.sum() == image_sum
    assert isinstance(topostats_object.image_original, np.ndarray)
    assert topostats_object.image_original.shape == (1024, 1024)
    assert topostats_object.image_original.sum() == image_original_sum
    assert isinstance(topostats_object.pixel_to_nm_scaling, float)
    assert topostats_object.pixel_to_nm_scaling == pixel_to_nm_scaling


@pytest.mark.parametrize(
    ("load_scan_object", "length", "image_shape", "image_sum", "filename", "pixel_to_nm_scaling"),
    [
        pytest.param("load_scan_spm", 1, (1024, 1024), 30695369.188316286, "minicircle", 0.4940029296875, id="spm"),
        pytest.param("load_scan_ibw", 1, (512, 512), -218091520.0, "minicircle2", 1.5625, id="ibw"),
        pytest.param("load_scan_jpk", 1, (256, 256), 219242202.8256843, "file", 1.2770176335964876, id="jpk"),
        pytest.param("load_scan_gwy", 1, (512, 512), 33836850.232917726, "file", 0.8468632812499975, id="gwy"),
        pytest.param(
            "load_scan_topostats",
            1,
            (1024, 1024),
            30695369.188316286,
            "file",
            0.4940029296875,
            id="topostats",
        ),
        pytest.param("load_scan_asd", 197, (200, 200), -12843725.967220962, "file_122", 2.0, id="asd"),
        pytest.param(
            "load_scan_topostats_240",
            1,
            (1024, 1024),
            30695369.188316286,
            "minicircle_240",
            0.4940029296875,
            id="topostats (version 2.4.0)",
        ),
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
    assert isinstance(scan.img_dict[filename].image_original, np.ndarray)
    assert scan.img_dict[filename].image_original.shape == image_shape
    assert scan.img_dict[filename].image_original.sum() == image_sum
    # If we are loading minicircle_240 it has a flattened .image attribute we can check (note it differs from above
    # disabled test though)
    if filename == "minicircle_240":
        assert scan.img_dict[filename].image.shape == image_shape
        assert scan.img_dict[filename].image.sum() == 184140.85939149323
    assert isinstance(scan.img_dict[filename].img_path, Path)
    assert scan.img_dict[filename].img_path == RESOURCES / filename
    assert isinstance(scan.img_dict[filename].pixel_to_nm_scaling, float)
    assert scan.img_dict[filename].pixel_to_nm_scaling == pixel_to_nm_scaling


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
    caplog.set_level(logging.DEBUG, LOGGER_NAME)
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
        # pylint thinks that f["c"] is a group but it is a bytes object that can be decoded
        # pylint: disable=no-member
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


def test_dict_to_hdf5_graincrop(dummy_graincrops_dict: grains.GrainCrop, tmp_path: Path) -> None:
    """Test loading a GrainGrop object and writing to HDF5 file."""
    # Make a dictionary from dummy_graincrop
    expected = {
        "0": {
            "image": dummy_graincrops_dict[0].image,
            "mask": dummy_graincrops_dict[0].mask,
            "padding": dummy_graincrops_dict[0].padding,
            "bbox": dummy_graincrops_dict[0].bbox,
            "pixel_to_nm_scaling": dummy_graincrops_dict[0].pixel_to_nm_scaling,
            "filename": dummy_graincrops_dict[0].filename,
            "stats": dummy_graincrops_dict[0].stats,
            "height_profiles": dummy_graincrops_dict[0].height_profiles,
        }
    }
    with h5py.File(tmp_path / "hdf5_grain_crop.hdf5", "w") as f:
        dict_to_hdf5(open_hdf5_file=f, group_path="/", dictionary=dummy_graincrops_dict)
    # Load it back in and check if the dictionary is the same
    with h5py.File(tmp_path / "hdf5_grain_crop.hdf5", "r") as f:
        assert sorted(f.keys()) == sorted(expected.keys())
        np.testing.assert_array_equal(f["0"]["image"][()], expected["0"]["image"])
        np.testing.assert_array_equal(f["0"]["mask"][()], expected["0"]["mask"])
        assert f["0"]["padding"][()] == expected["0"]["padding"]
        np.testing.assert_array_equal(f["0"]["bbox"][()], expected["0"]["bbox"])
        assert f["0"]["pixel_to_nm_scaling"][()] == expected["0"]["pixel_to_nm_scaling"]
        assert f["0"]["filename"][()].decode("utf-8") == expected["0"]["filename"]  # pylint: disable=no-member
        for key, value in f["0"]["stats"]["1"]["0"].items():
            assert value[()] == expected["0"]["stats"][1][0][key]
        np.testing.assert_array_equal(f["0"]["height_profiles"]["1"]["0"][()], expected["0"]["height_profiles"][1][0])


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
    (
        "topostats_version",
        "image",
        "pixel_to_nm_scaling",
        "filename",
        "grain_crop",
        "img_path",
        "grain_mask_above",
        "grain_mask_below",
        "grain_trace_data",
    ),
    [
        pytest.param(
            "0.2",  # topotats_version
            np.arange(0, 100).reshape(10, 10),  # image
            3.14159265,  # pixel_to_nm_scaling
            "below_grain_mask_with_grain_trace_data",  # filename
            None,  # graincrop
            "./below_grain_mask_with_grain_trace_data.topostats",  # img_path
            np.zeros((10, 10)),
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
            marks=pytest.mark.skip("to be removed"),
        ),
        pytest.param(
            "0.2",
            np.arange(0, 100).reshape(10, 10),
            3.14159265,
            "above_grain_mask_without_grain_trace_data",
            None,
            "./above_grain_mask_without_grain_trace_data.topostats",
            np.zeros((10, 10)),
            np.zeros((10, 10)),
            None,
            id="above_grain_mask_without_grain_trace_data",
            marks=pytest.mark.skip("to be removed"),
        ),
        pytest.param(
            "0.2",
            np.arange(0, 100).reshape(10, 10),
            3.14159265,
            "above_and_below_grain_masks_without_grain_trace_data",
            "./above_and_below_grain_masks_without_grain_trace_data.topostats",
            None,
            np.zeros((10, 10)),
            np.zeros((10, 10)),
            None,
            id="above_and_below_grain_masks_without_grain_trace_data",
            marks=pytest.mark.skip("to be removed"),
        ),
        pytest.param(
            "2.4.0",
            "topostats_rep_int_2_4_0",
            None,
            "minicircles",
            None,  # grain_crop
            None,  # img_path
            None,  # grain_mask_above
            None,  # grain_mask_below
            None,  # grain_trace_data
            id="2.4.0 rep_int",
            marks=pytest.mark.skip("work in progress, may need AFMReader tweaks"),
        ),
        pytest.param(
            "2.4.0",
            "topostats_catenanes_2_4_0",
            None,
            "catenanes",
            None,  # grain_crop
            None,  # img_path
            None,  # grain_mask_above
            None,  # grain_mask_below
            None,  # grain_trace_data
            id="2.4.0 catenanes",
            marks=pytest.mark.skip("work in progress, may need AFMReader tweaks"),
        ),
    ],
)
def test_save_and_load_topostats(  # pylint: disable=unused-argument,too-many-locals
    load_scan_topostats_test_file: LoadScans,
    tmp_path: Path,
    topostats_version: str,
    image: np.ndarray,
    pixel_to_nm_scaling: float,
    filename: str,
    grain_crop: None,
    img_path: str,
    grain_mask_above: np.ndarray,
    grain_mask_below: np.ndarray,
    grain_trace_data: dict,
    request,
) -> None:
    """Test saving a .topostats file."""
    # Setup topostats_object conditional on version
    if topostats_version < "2.4.0":
        topostats_object = {
            "filename": filename,
            "img_path": img_path,
            "pixel_to_nm_scaling": pixel_to_nm_scaling,
            "image_original": image,
            "image": image,
            "grain_masks": {"above": grain_mask_above, "below": grain_mask_below},
            "grain_trace_data": grain_trace_data,
        }
    else:
        # If we are testing >= 2.4.0 then topostats_object should be TopoStats
        # @ns-rse 2025-10-30 : Probably need to switch to a fixture of TopoStats here
        topostats_object = request.getfixturevalue(image)
    # Save the file
    save_topostats_file(
        output_dir=tmp_path,
        filename="topostats_file_test.topostats",
        topostats_object=topostats_object,
        topostats_version=topostats_version,
    )
    # Conditionally test the returned object
    if topostats_version < "2.4.0":
        # Load the saved .topostats file using LoadScans
        loadscans = load_scan_topostats_test_file
        loadscans.get_data()
        np.testing.assert_array_equal(image, loadscans.img_dict["topostats_file_test"]["image_original"])
        assert pixel_to_nm_scaling == loadscans.img_dict["topostats_file_test"]["pixel_to_nm_scaling"]
        if grain_mask_above is not None:
            np.testing.assert_array_equal(
                grain_mask_above, loadscans.img_dict["topostats_file_test"]["grain_masks"]["above"]
            )
            if grain_mask_below is not None:
                np.testing.assert_array_equal(
                    grain_mask_below, loadscans.img_dict["topostats_file_test"]["grain_masks"]["below"]
                )
                if grain_trace_data is not None:
                    np.testing.assert_equal(
                        grain_trace_data, loadscans.img_dict["topostats_file_test"]["grain_trace_data"]
                    )
    else:
        # Load the saved .topostats file using LoadScans
        loadscans = load_scan_topostats_test_file
        loadscans.get_data()
        topostats_object_from_disk = loadscans.img_dict["topostats_file_test"]
        assert isinstance(topostats_object, TopoStats)
        assert topostats_object == topostats_object_from_disk


@pytest.mark.parametrize(
    ("dictionary", "target"),
    [
        pytest.param(
            {"above": {"a": [1, 2, 3], "b": [4, 5, 6]}},
            {"above": {"a": [1, 2, 3], "b": [4, 5, 6]}},
            id="dictionary and lists",
        ),
        pytest.param(
            {"above": {"a": np.asarray([1, 2, 3]), "b": np.asarray([4, 5, 6])}},
            {"above": {"a": [1, 2, 3], "b": [4, 5, 6]}},
            id="dictionary and numpy arrays",
        ),
    ],
)
def test_dict_to_json(dictionary: dict, target: dict, tmp_path: Path) -> None:
    """Test writing of dictionary to JSON file."""
    dict_to_json(data=dictionary, output_dir=tmp_path, filename="test.json")

    outfile = tmp_path / "test.json"
    assert outfile.is_file()

    with outfile.open("r", encoding="utf-8") as f:
        assert target == json.load(f)


@pytest.mark.parametrize(
    ("dictionary", "topostats_expected"),
    [
        pytest.param(
            {
                "grain_crops": None,
                "filename": "basic",
                "pixel_to_nm_scaling": 0.5,
                "img_path": Path("./"),
                "image": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                "image_original": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                "topostats_version": "42",
            },
            TopoStats(
                grain_crops=None,
                filename="basic",
                pixel_to_nm_scaling=0.5,
                img_path=Path("./"),
                image=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                image_original=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                topostats_version="42",
            ),
            id="basic",
        ),
        pytest.param(
            {
                "grain_crops": {
                    0: {
                        "image": np.array([[1, 2], [3, 4]]),
                        "mask": np.array([[[0, 1], [1, 0]], [[1, 0], [0, 1]]]),
                        "padding": 1,
                        "bbox": (0, 1, 0, 1),
                        "pixel_to_nm_scaling": 0.5,
                        "filename": "basic_graincrop",
                        "skeleton": np.array([[0, 1], [1, 0]]),
                        # "height_profiles": np.array([2, 3]),
                        "stats": None,
                    }
                },
                "full_mask_tensor": np.array(
                    [
                        [[0, 0], [0, 0]],
                        [[1, 0], [0, 1]],
                        [[1, 1], [1, 1]],
                    ]
                ),
                "filename": "basic_graincrop",
                "pixel_to_nm_scaling": 0.5,
                "img_path": Path("./"),
                "image": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                "image_original": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                "topostats_version": "42",
            },
            TopoStats(
                grain_crops={
                    0: GrainCrop(
                        image=np.array([[1, 2], [3, 4]]),
                        mask=np.array([[[0, 1], [1, 0]], [[1, 0], [0, 1]]]),
                        padding=1,
                        bbox=(0, 1, 0, 1),
                        pixel_to_nm_scaling=0.5,
                        filename="basic_graincrop",
                        skeleton=np.array([[0, 1], [1, 0]]),
                        # height_profiles=np.array([2, 3]),
                        thresholds=[0, 2],
                        stats=None,
                    )
                },
                full_mask_tensor=np.array(
                    [
                        [[0, 0], [0, 0]],
                        [[1, 0], [0, 1]],
                        [[1, 1], [1, 1]],
                    ]
                ),
                filename="basic_graincrop",
                pixel_to_nm_scaling=0.5,
                img_path=Path("./"),
                image=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                image_original=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                topostats_version="42",
            ),
            id="single crop, no tracing",
            marks=pytest.mark.xfail(reason="Failing because GrainCrop objects have different memory addresses."),
        ),
        pytest.param(
            {
                "grain_crops": {
                    0: {
                        "image": np.array([[1, 2], [3, 4]]),
                        "mask": np.array([[[0, 1], [1, 0]], [[1, 0], [0, 1]]]),
                        "padding": 1,
                        "bbox": (0, 1, 0, 1),
                        "pixel_to_nm_scaling": 0.5,
                        "filename": "basic_graincrop",
                        "skeleton": np.array([[0, 1], [1, 0]]),
                        # "height_profiles": np.array([2, 3]),
                        "stats": None,
                        "disordered_trace": {
                            "images": {"pruned_skeleton": np.array([[0, 1], [1, 0]])},
                            "grain_endpoints": 3,
                            "grain_junctions": 6,
                            "total_branch_length": 42,
                            "grain_width_mean": 8,
                        },
                        "nodes": {
                            0: {"error": False, "pixel_to_nm_scaling": 1, "confidence": 0.999},
                            1: {"error": True},
                        },
                        "ordered_trace": {"molecules": 2, "writhe": "+"},
                    }
                },
                "full_mask_tensor": np.array(
                    [
                        [[0, 0], [0, 0]],
                        [[1, 0], [0, 1]],
                        [[1, 1], [1, 1]],
                    ]
                ),
                "filename": "basic_graincrop",
                "pixel_to_nm_scaling": 0.5,
                "img_path": Path("./"),
                "image": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                "image_original": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                "topostats_version": "42",
            },
            TopoStats(
                grain_crops={
                    0: GrainCrop(
                        image=np.array([[1, 2], [3, 4]]),
                        mask=np.array([[[0, 1], [1, 0]], [[1, 0], [0, 1]]]),
                        padding=1,
                        bbox=(0, 1, 0, 1),
                        pixel_to_nm_scaling=0.5,
                        filename="basic_graincrop",
                        skeleton=np.array([[0, 1], [1, 0]]),
                        # height_profiles=np.array([2, 3]),
                        stats=None,
                        thresholds=[0, 2],
                        disordered_trace=DisorderedTrace(
                            images={"pruned_skeleton": np.array([[0, 1], [1, 0]])},
                            grain_endpoints=3,
                            grain_junctions=6,
                            total_branch_length=42,
                            grain_width_mean=8,
                        ),
                        nodes={
                            0: Node(error=False, pixel_to_nm_scaling=1, confidence=0.999),
                            1: Node(error=True),
                        },
                        ordered_trace=OrderedTrace(molecules=2, writhe="+"),
                    )
                },
                full_mask_tensor=np.array(
                    [
                        [[0, 0], [0, 0]],
                        [[1, 0], [0, 1]],
                        [[1, 1], [1, 1]],
                    ]
                ),
                filename="basic_graincrop",
                pixel_to_nm_scaling=0.5,
                img_path=Path("./"),
                image=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                image_original=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                topostats_version="42",
            ),
            id="single crop, tracing (partial)",
            marks=pytest.mark.xfail(reason="Failing because GrainCrop objects have different memory addresses."),
        ),
    ],
)
def test_dict_to_topostats(dictionary: dict, topostats_expected: TopoStats) -> None:
    """Test for dict_to_topostats()."""
    topostats_object = dict_to_topostats(dictionary)
    assert topostats_object == topostats_expected
