# Disable ruff 301 - pickle loading is unsafe, but we don't care for tests.
# ruff: noqa: S301
"""Test end-to-end running of topostats."""

import logging
import pickle
from pathlib import Path

import filetype
import h5py
import numpy as np
import pandas as pd
import pytest
from test_io import dict_almost_equal

from topostats.grains import GrainCrop, GrainCropsDirection, ImageGrainCrops
from topostats.io import LoadScans, hdf5_to_dict
from topostats.processing import (
    LOGGER_NAME,
    check_run_steps,
    process_scan,
    run_filters,
    run_grains,
    run_grainstats,
)
from topostats.utils import update_plotting_config

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests/resources"

# pylint: disable=too-many-lines
# pylint: disable=too-many-positional-arguments


def test_debug_process_file(tmp_path) -> None:
    """Debug processing a file."""
    from topostats.io import read_yaml
    import yaml
    import importlib.resources as pkg_resources
    import topostats
    import copy

    # config
    config_path = Path("/Users/sylvi/topo_data/bug-grain-crop-size/config.yaml")
    assert config_path.exists()
    original_config = read_yaml(config_path)
    plotting_dictionary = pkg_resources.open_text(topostats, "plotting_dictionary.yaml")
    original_config["plotting"]["plot_dict"] = yaml.safe_load(plotting_dictionary.read())

    # data
    datafiles = Path("/Users/sylvi/topo_data/bug-grain-crop-size/data").glob("*.spm")
    scans = LoadScans(list(datafiles), channel="Height", extract="all")
    scans.get_data()
    image_dict = scans.img_dict
    for filename, topo_image in image_dict.items():

        if filename != "20250501_5nMTRF1_1ngtel12_picoz_EE_freezer_nicl.0_00041":
            continue

        print()
        print(f"PROCESSING IMAGE {filename}")
        print()

        config = copy.deepcopy(original_config)

        _, _, _, _, _, _ = process_scan(
            topostats_object=topo_image,
            base_dir=BASE_DIR,
            filter_config=config["filter"],
            grains_config=config["grains"],
            grainstats_config=config["grainstats"],
            disordered_tracing_config=config["disordered_tracing"],
            nodestats_config=config["nodestats"],
            ordered_tracing_config=config["ordered_tracing"],
            splining_config=config["splining"],
            curvature_config=config["curvature"],
            plotting_config=config["plotting"],
            output_dir=tmp_path,
        )


# Can't see a way of parameterising with pytest-regtest as it writes to a file based on the file/function
# so instead we run three regression tests.
def test_process_scan_below(regtest, tmp_path, process_scan_config: dict, load_scan_data: LoadScans) -> None:
    """Regression test for checking the process_scan functions correctly."""
    # Ensure there are below grains
    process_scan_config["grains"]["threshold_std_dev"]["below"] = [0.8]
    process_scan_config["grains"]["area_thresholds"]["below"] = [10, 1000000000]
    process_scan_config["grains"]["direction"] = "below"
    # Make sure the pruning won't remove our only grain
    process_scan_config["disordered_tracing"]["pruning_params"]["max_length"] = None
    img_dic = load_scan_data.img_dict
    _, results, _, img_stats, _, _ = process_scan(
        topostats_object=img_dic["minicircle_small"],
        base_dir=BASE_DIR,
        filter_config=process_scan_config["filter"],
        grains_config=process_scan_config["grains"],
        grainstats_config=process_scan_config["grainstats"],
        disordered_tracing_config=process_scan_config["disordered_tracing"],
        nodestats_config=process_scan_config["nodestats"],
        ordered_tracing_config=process_scan_config["ordered_tracing"],
        splining_config=process_scan_config["splining"],
        curvature_config=process_scan_config["curvature"],
        plotting_config=process_scan_config["plotting"],
        output_dir=tmp_path,
    )
    # Remove the basename column as this differs on CI
    results.drop(["basename"], axis=1, inplace=True)
    print(img_stats.to_string(float_format="{:.4e}".format), file=regtest)  # noqa: T201
    print(results.to_string(float_format="{:.4e}".format), file=regtest)  # noqa: T201


def test_process_scan_below_height_profiles(tmp_path, process_scan_config: dict, load_scan_data: LoadScans) -> None:
    """Regression test for checking the process_scan functions correctly."""
    # Ensure there are below grains
    process_scan_config["grains"]["threshold_std_dev"]["below"] = 0.8
    process_scan_config["grains"]["area_thresholds"]["below"] = [10, 1000000000]

    process_scan_config["grains"]["direction"] = "below"
    img_dic = load_scan_data.img_dict
    _, _, height_profiles, _, _, _ = process_scan(
        topostats_object=img_dic["minicircle_small"],
        base_dir=BASE_DIR,
        filter_config=process_scan_config["filter"],
        grains_config=process_scan_config["grains"],
        grainstats_config=process_scan_config["grainstats"],
        disordered_tracing_config=process_scan_config["disordered_tracing"],
        nodestats_config=process_scan_config["nodestats"],
        ordered_tracing_config=process_scan_config["ordered_tracing"],
        splining_config=process_scan_config["splining"],
        curvature_config=process_scan_config["curvature"],
        plotting_config=process_scan_config["plotting"],
        output_dir=tmp_path,
    )

    # Save height profiles dictionary to pickle
    # with open(RESOURCES / "process_scan_expected_below_height_profiles.pkl", "wb") as f:
    #     pickle.dump(height_profiles, f)

    # Load expected height profiles dictionary from pickle
    # pylint wants an encoding but binary mode doesn't use one
    # pylint: disable=unspecified-encoding
    with Path.open(RESOURCES / "process_scan_expected_below_height_profiles.pkl", "rb") as f:
        expected_height_profiles = pickle.load(f)  # noqa: S301 - Pickles are unsafe but we don't care

    assert dict_almost_equal(height_profiles, expected_height_profiles, abs_tol=1e-11)


def test_process_scan_above(regtest, tmp_path, process_scan_config: dict, load_scan_data: LoadScans) -> None:
    """Regression test for checking the process_scan functions correctly."""
    # Ensure there are below grains
    process_scan_config["grains"]["area_thresholds"]["below"] = [10, 1000000000]

    img_dic = load_scan_data.img_dict
    _, results, _, img_stats, _, _ = process_scan(
        topostats_object=img_dic["minicircle_small"],
        base_dir=BASE_DIR,
        filter_config=process_scan_config["filter"],
        grains_config=process_scan_config["grains"],
        grainstats_config=process_scan_config["grainstats"],
        disordered_tracing_config=process_scan_config["disordered_tracing"],
        nodestats_config=process_scan_config["nodestats"],
        ordered_tracing_config=process_scan_config["ordered_tracing"],
        splining_config=process_scan_config["splining"],
        curvature_config=process_scan_config["curvature"],
        plotting_config=process_scan_config["plotting"],
        output_dir=tmp_path,
    )
    # Remove the Basename column as this differs on CI
    results.drop(["basename"], axis=1, inplace=True)
    print(img_stats.to_string(float_format="{:.4e}".format), file=regtest)  # noqa: T201
    print(results.to_string(float_format="{:.4e}".format), file=regtest)  # noqa: T201


def test_process_scan_above_height_profiles(tmp_path, process_scan_config: dict, load_scan_data: LoadScans) -> None:
    """Regression test for checking the process_scan functions correctly."""
    # Ensure there are below grains
    process_scan_config["grains"]["area_thresholds"]["below"] = [10, 1000000000]

    img_dic = load_scan_data.img_dict
    _, _, height_profiles, _, _, _ = process_scan(
        topostats_object=img_dic["minicircle_small"],
        base_dir=BASE_DIR,
        filter_config=process_scan_config["filter"],
        grains_config=process_scan_config["grains"],
        grainstats_config=process_scan_config["grainstats"],
        disordered_tracing_config=process_scan_config["disordered_tracing"],
        nodestats_config=process_scan_config["nodestats"],
        ordered_tracing_config=process_scan_config["ordered_tracing"],
        splining_config=process_scan_config["splining"],
        curvature_config=process_scan_config["curvature"],
        plotting_config=process_scan_config["plotting"],
        output_dir=tmp_path,
    )

    # Save height profiles dictionary to pickle
    # with open(RESOURCES / "process_scan_expected_above_height_profiles.pkl", "wb") as f:
    #     pickle.dump(height_profiles, f)

    # Load expected height profiles dictionary from pickle
    # pylint wants an encoding but binary mode doesn't use one
    # pylint: disable=unspecified-encoding
    with Path.open(RESOURCES / "process_scan_expected_above_height_profiles.pkl", "rb") as f:
        expected_height_profiles = pickle.load(f)  # noqa: S301 - Pickles are unsafe but we don't care

    assert dict_almost_equal(height_profiles, expected_height_profiles, abs_tol=1e-11)


def test_process_scan_both(regtest, tmp_path, process_scan_config: dict, load_scan_data: LoadScans) -> None:
    """Regression test for checking the process_scan functions correctly."""
    # Ensure there are below grains
    process_scan_config["grains"]["threshold_std_dev"]["below"] = 0.8
    process_scan_config["grains"]["area_thresholds"]["below"] = [10, 1000000000]

    process_scan_config["grains"]["direction"] = "both"
    img_dic = load_scan_data.img_dict
    _, results, _, img_stats, _, _ = process_scan(
        topostats_object=img_dic["minicircle_small"],
        base_dir=BASE_DIR,
        filter_config=process_scan_config["filter"],
        grains_config=process_scan_config["grains"],
        grainstats_config=process_scan_config["grainstats"],
        disordered_tracing_config=process_scan_config["disordered_tracing"],
        nodestats_config=process_scan_config["nodestats"],
        ordered_tracing_config=process_scan_config["ordered_tracing"],
        splining_config=process_scan_config["splining"],
        curvature_config=process_scan_config["curvature"],
        plotting_config=process_scan_config["plotting"],
        output_dir=tmp_path,
    )
    # Remove the Basename column as this differs on CI
    results.drop(["basename"], axis=1, inplace=True)
    print(img_stats.to_string(float_format="{:.4e}".format), file=regtest)  # noqa: T201
    print(results.to_string(float_format="{:.4e}".format), file=regtest)  # noqa: T201

    # Regtest for the topostats file
    assert Path.exists(tmp_path / "tests/resources/test_image/processed/minicircle_small.topostats")
    with h5py.File(RESOURCES / "process_scan_topostats_file_regtest.topostats", "r") as f:
        expected_topostats = hdf5_to_dict(f, group_path="/")
    with h5py.File(tmp_path / "tests/resources/test_image/processed/minicircle_small.topostats", "r") as f:
        saved_topostats = hdf5_to_dict(f, group_path="/")

    # Remove the image path as this differs on CI
    saved_topostats.pop("img_path")

    # Script for updating the regtest file
    # with h5py.File(RESOURCES / "process_scan_topostats_file_regtest.topostats", "w") as f:
    #     dict_to_hdf5(open_hdf5_file=f, group_path="/", dictionary=saved_topostats)

    # Check the keys, this will flag all new keys when adding output stats
    assert expected_topostats.keys() == saved_topostats.keys()
    # Check the data (we pop the file version as we are interested in comparing the underlying data)
    expected_topostats.pop("topostats_file_version")
    saved_topostats.pop("topostats_file_version")
    assert dict_almost_equal(expected_topostats, saved_topostats, abs_tol=1e-6)


@pytest.mark.parametrize(
    ("image_set", "expected"),
    [
        pytest.param(["core"], False, id="core"),
        pytest.param(["all"], True, id="all"),
        pytest.param(["filters"], False, id="filters"),
        pytest.param(["grain_crops"], True, id="grain crops"),
        pytest.param(["filters", "grain_crops", "disordered_tracing"], True, id="list with grain crops"),
        pytest.param(["filters", "grains", "disordered_tracing"], False, id="list without grain crops"),
    ],
)
def test_save_cropped_grains(
    tmp_path: Path, process_scan_config: dict, load_scan_data: LoadScans, image_set: list[str], expected: bool
) -> None:
    """Tests if cropped grains are saved only when ``image_set`` is 'all' or contains ``grain_crops``."""
    process_scan_config["plotting"]["image_set"] = image_set
    process_scan_config["plotting"] = update_plotting_config(process_scan_config["plotting"])
    process_scan_config["plotting"]["savefig_dpi"] = 50
    img_dic = load_scan_data.img_dict
    _, _, _, _, _, _ = process_scan(
        topostats_object=img_dic["minicircle_small"],
        base_dir=BASE_DIR,
        filter_config=process_scan_config["filter"],
        grains_config=process_scan_config["grains"],
        grainstats_config=process_scan_config["grainstats"],
        disordered_tracing_config=process_scan_config["disordered_tracing"],
        nodestats_config=process_scan_config["nodestats"],
        ordered_tracing_config=process_scan_config["ordered_tracing"],
        splining_config=process_scan_config["splining"],
        curvature_config=process_scan_config["curvature"],
        plotting_config=process_scan_config["plotting"],
        output_dir=tmp_path,
    )

    assert (
        Path.exists(
            tmp_path
            / "tests/resources/test_image/processed/minicircle_small/grains/above"
            / "minicircle_small_grain_0.png"
        )
        == expected
    )
    assert (
        Path.exists(
            tmp_path
            / "tests/resources/test_image/processed/minicircle_small/grains/above"
            / "minicircle_small_grain_mask_0_class_1.png"
        )
        == expected
    )


@pytest.mark.parametrize(
    ("image_set", "expected_image"),
    [
        pytest.param(
            ["core"],
            {
                "core": True,
                "filters": False,
                "grains": False,
                "grain_crop": False,
                "disordered_tracing": False,
                "nodestats": False,
                "ordered_tracing": False,
                "splining": False,
            },
            id="only core",
        ),
        pytest.param(
            ["all"],
            {
                "core": True,
                "filters": True,
                "grains": True,
                "grain_crop": True,
                "disordered_tracing": True,
                "nodestats": True,
                "ordered_tracing": True,
                "splining": True,
            },
            id="all",
        ),
        pytest.param(
            ["filters"],
            {
                "core": True,
                "filters": True,
                "grains": False,
                "grain_crop": False,
                "disordered_tracing": False,
                "nodestats": False,
                "ordered_tracing": False,
                "splining": False,
            },
            id="only filters",
        ),
        pytest.param(
            ["grains"],
            {
                "core": True,
                "filters": False,
                "grains": True,
                "grain_crop": False,
                "disordered_tracing": False,
                "nodestats": False,
                "ordered_tracing": False,
                "splining": False,
            },
            id="only grains",
        ),
        pytest.param(
            ["grain_crops"],
            {
                "core": True,
                "filters": False,
                "grains": False,
                "grain_crop": True,
                "disordered_tracing": False,
                "nodestats": False,
                "ordered_tracing": False,
                "splining": False,
            },
            id="only grain_crops",
        ),
        pytest.param(
            ["disordered_tracing"],
            {
                "core": True,
                "filters": False,
                "grains": False,
                "grain_crop": False,
                "disordered_tracing": True,
                "nodestats": False,
                "ordered_tracing": False,
                "splining": False,
            },
            id="only disordered_tracing",
        ),
        pytest.param(
            ["nodestats"],
            {
                "core": True,
                "filters": False,
                "grains": False,
                "grain_crop": False,
                "disordered_tracing": False,
                "nodestats": True,
                "ordered_tracing": False,
                "splining": False,
            },
            id="only nodestats",
        ),
        pytest.param(
            ["ordered_tracing"],
            {
                "core": True,
                "filters": False,
                "grains": False,
                "grain_crop": False,
                "disordered_tracing": False,
                "nodestats": False,
                "ordered_tracing": True,
                "splining": False,
            },
            id="only ordered_tracing",
        ),
        pytest.param(
            ["splining"],
            {
                "core": True,
                "filters": False,
                "grains": False,
                "grain_crop": False,
                "disordered_tracing": False,
                "nodestats": False,
                "ordered_tracing": False,
                "splining": True,
            },
            id="only splining",
        ),
        pytest.param(
            ["filters", "grain_crops", "disordered_tracing"],
            {
                "core": True,
                "filters": True,
                "grains": False,
                "grain_crop": True,
                "disordered_tracing": True,
                "nodestats": False,
                "ordered_tracing": False,
                "splining": False,
            },
            id="filters, grain_crops and disordered_tracing",
        ),
        pytest.param(
            ["grains", "nodestats", "ordered_tracing"],
            {
                "core": True,
                "filters": False,
                "grains": True,
                "grain_crop": False,
                "disordered_tracing": False,
                "nodestats": True,
                "ordered_tracing": True,
                "splining": False,
            },
            id="grains, nodestats, ordered_tracing",
        ),
        pytest.param(
            ["filters", "disordered_tracing", "splining"],
            {
                "core": True,
                "filters": True,
                "grains": False,
                "grain_crop": False,
                "disordered_tracing": True,
                "nodestats": False,
                "ordered_tracing": False,
                "splining": True,
            },
            id="filters, disordered_tracing, splining",
        ),
    ],
)
def test_image_set(
    tmp_path: Path,
    process_scan_config: dict,
    load_scan_data: LoadScans,
    image_set: list[str],
    expected_image: dict[str, bool],
) -> None:
    """Tests if specific diagnostic images are saved only when image set is 'all' rather than 'core'."""
    process_scan_config["plotting"]["image_set"] = image_set
    process_scan_config["plotting"] = update_plotting_config(process_scan_config["plotting"])
    process_scan_config["plotting"]["savefig_dpi"] = 50
    img_dic = load_scan_data.img_dict
    _, _, _, _, _, _ = process_scan(
        topostats_object=img_dic["minicircle_small"],
        base_dir=BASE_DIR,
        filter_config=process_scan_config["filter"],
        grains_config=process_scan_config["grains"],
        grainstats_config=process_scan_config["grainstats"],
        disordered_tracing_config=process_scan_config["disordered_tracing"],
        nodestats_config=process_scan_config["nodestats"],
        ordered_tracing_config=process_scan_config["ordered_tracing"],
        splining_config=process_scan_config["splining"],
        curvature_config=process_scan_config["curvature"],
        plotting_config=process_scan_config["plotting"],
        output_dir=tmp_path,
    )

    # expected image paths
    images = {
        "core": "minicircle_small_above_all_splines.png",
        "filters": "minicircle_small/filters/01-pixels.png",
        "grains": "minicircle_small/grains/above/24-area_thresholded_class_1.png",
        "grain_crop": "minicircle_small/grains/above/minicircle_small_grain_0.png",
        "disordered_tracing": "minicircle_small/dnatracing/above/22-original_skeletons.png",
        "nodestats": "minicircle_small/dnatracing/above/26-node_centres.png",
        "ordered_tracing": "minicircle_small/dnatracing/above/28-molecule_crossings.png",
        "splining": "minicircle_small/dnatracing/above/curvature/grain_0_curvature.png",
    }
    for key, img_path in images.items():
        assert Path.exists(tmp_path / "tests/resources/test_image/processed/" / img_path) == expected_image[key]


@pytest.mark.parametrize("extension", [("png"), ("tif")])
def test_save_format(process_scan_config: dict, load_scan_data: LoadScans, tmp_path: Path, extension: str):
    """Tests if save format applied to cropped images."""
    process_scan_config["plotting"]["image_set"] = "all"
    process_scan_config["plotting"]["savefig_format"] = extension
    process_scan_config["plotting"] = update_plotting_config(process_scan_config["plotting"])
    img_dic = load_scan_data.img_dict
    _, _, _, _, _, _ = process_scan(
        topostats_object=img_dic["minicircle_small"],
        base_dir=BASE_DIR,
        filter_config=process_scan_config["filter"],
        grains_config=process_scan_config["grains"],
        grainstats_config=process_scan_config["grainstats"],
        disordered_tracing_config=process_scan_config["disordered_tracing"],
        nodestats_config=process_scan_config["nodestats"],
        ordered_tracing_config=process_scan_config["ordered_tracing"],
        splining_config=process_scan_config["splining"],
        curvature_config=process_scan_config["curvature"],
        plotting_config=process_scan_config["plotting"],
        output_dir=tmp_path,
    )

    guess = filetype.guess(
        tmp_path
        / "tests/resources/test_image/processed/minicircle_small/grains/above"
        / f"minicircle_small_grain_mask_0_class_1.{extension}"
    )
    assert guess.extension == extension


# noqa: PLR0913
# pylint: disable=too-many-arguments
@pytest.mark.parametrize(
    (
        "filter_run",
        "grains_run",
        "grainstats_run",
        "disordered_tracing_run",
        "nodestats_run",
        "ordered_tracing_run",
        "splining_run",
        "log_msg",
    ),
    [
        pytest.param(
            True,
            True,
            True,
            True,
            False,
            True,
            True,
            "Splining enabled but NodeStats disabled. Tracing will use the 'old' method.",
            id="Splining, Ordered Tracing, Disordered Tracing, Grainstats, Grains and Filters no Nodestats",
        ),
        pytest.param(
            True,
            True,
            False,
            True,
            False,
            True,
            True,
            "Splining enabled but Grainstats disabled. Please check your configuration file.",
            id="Splining, Ordered Tracing, Disordered Tracing, Grains and Filters enabled but no Grainstats or "
            + "NodeStats",
        ),
        pytest.param(
            True,
            False,
            False,
            False,
            False,
            True,
            True,
            "Splining enabled but Disordered Tracing disabled. Please check your configuration file.",
            id="Splining, Ordered Tracing and Filters enabled but no NodeStats, Disordered Tracing or Grainstats",
        ),
        pytest.param(
            False,
            False,
            True,
            True,
            False,
            True,
            True,
            "Splining enabled but Grains disabled. Please check your configuration file.",
            id="Splining, Ordered Tracing, Disordered Tracing, and Grainstats enabled but no NodeStats, Grains or "
            + "Filters",
        ),
        pytest.param(
            False,
            False,
            False,
            False,
            True,
            False,
            False,
            "NodeStats enabled but Disordered Tracing disabled. Please check your configuration file.",
            id="Nodestats enabled but no Ordered Tracing, Disordered Tracing, Grainstats, Grains or Filters",
        ),
        pytest.param(
            False,
            False,
            False,
            True,
            True,
            False,
            False,
            "NodeStats enabled but Grainstats disabled. Please check your configuration file.",
            id="Nodestats, Disordered Tracing enabled but no Grainstats, Grains or Filters",
        ),
        pytest.param(
            False,
            False,
            True,
            True,
            True,
            False,
            False,
            "NodeStats enabled but Grains disabled. Please check your configuration file.",
            id="Nodestats, Disordered Tracing, Grainstats enabled but no Grains or Filters",
        ),
        pytest.param(
            False,
            True,
            True,
            True,
            True,
            False,
            False,
            "NodeStats enabled but Filters disabled. Please check your configuration file.",
            id="Nodestats, Disordered Tracing, Grainstats Grains enabled but no Filters",
        ),
        pytest.param(
            False,
            False,
            False,
            True,
            False,
            False,
            False,
            "Disordered Tracing enabled but Grainstats disabled. Please check your configuration file.",
            id="Disordered Tracing enabled but no Grainstats, Grains or Filters",
        ),
        pytest.param(
            False,
            False,
            True,
            True,
            False,
            False,
            False,
            "Disordered Tracing enabled but Grains disabled. Please check your configuration file.",
            id="Disordered tracing and Grainstats enabled but no Grains or Filters",
        ),
        pytest.param(
            False,
            True,
            True,
            True,
            False,
            False,
            False,
            "Disordered Tracing enabled but Filters disabled. Please check your configuration file.",
            id="Disordered tracing, Grains and Grainstats enabled but no Filters",
        ),
        pytest.param(
            False,
            False,
            True,
            False,
            False,
            False,
            False,
            "Grainstats enabled but Grains disabled. Please check your configuration file.",
            id="Grainstats enabled but no Grains or Filters",
        ),
        pytest.param(
            False,
            True,
            True,
            False,
            False,
            False,
            False,
            "Grainstats enabled but Filters disabled. Please check your configuration file.",
            id="Grains enabled and Grainstats but no Filters",
        ),
        pytest.param(
            False,
            True,
            False,
            False,
            False,
            False,
            False,
            "Grains enabled but Filters disabled. Please check your configuration file.",
            id="Grains enabled but not Filters",
        ),
        pytest.param(
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            "Configuration run options are consistent, processing can proceed.",
            id="Consistent configuration upto Filters",
        ),
        pytest.param(
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            "Configuration run options are consistent, processing can proceed.",
            id="Consistent configuration upto Grains",
        ),
        pytest.param(
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            "Configuration run options are consistent, processing can proceed.",
            id="Consistent configuration upto Grainstats",
        ),
        pytest.param(
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            "Configuration run options are consistent, processing can proceed.",
            id="Consistent configuration upto DNA tracing",
        ),
        pytest.param(
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            "Configuration run options are consistent, processing can proceed.",
            id="Consistent configuration upto Nodestats",
        ),
        pytest.param(
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            "Configuration run options are consistent, processing can proceed.",
            id="Consistent configuration upto Splining",
        ),
    ],
)
def test_check_run_steps(
    filter_run: bool,
    grains_run: bool,
    grainstats_run: bool,
    disordered_tracing_run: bool,
    nodestats_run: bool,
    ordered_tracing_run: bool,
    splining_run: bool,
    log_msg: str,
    caplog,
) -> None:
    """Test the logic which checks whether enabled processing options are consistent."""
    check_run_steps(
        filter_run,
        grains_run,
        grainstats_run,
        disordered_tracing_run,
        nodestats_run,
        ordered_tracing_run,
        splining_run,
    )
    assert log_msg in caplog.text


# pylint: disable=too-many-arguments
@pytest.mark.parametrize(
    (
        "filter_run",
        "grains_run",
        "grainstats_run",
        "disordered_tracing_run",
        "nodestats_run",
        "ordered_tracing_run",
        "splining_run",
        "curvature_run",
        "log_msg1",
        "log_msg2",
    ),
    [
        pytest.param(
            False,  # Filters
            False,  # Grains
            False,  # Grainstats
            False,  # Disordered tracing
            False,  # Nodestats
            False,  # Ordered tracing
            False,  # Splining
            False,  # Curvature
            "You have not included running the initial filter stage.",
            "Please check your configuration file.",
            id="All stages are disabled",
        ),
        pytest.param(
            True,  # Filters
            False,  # Grains
            False,  # Grainstats
            False,  # Disordered tracing
            False,  # Nodestats
            False,  # Ordered tracing
            False,  # Splining
            False,  # Curvature
            "Detection of grains disabled, GrainStats will not be run.",
            "",
            id="Only filtering enabled",
        ),
        pytest.param(
            True,  # Filters
            True,  # Grains
            False,  # Grainstats
            False,  # Disordered tracing
            False,  # Nodestats
            False,  # Ordered tracing
            False,  # Splining
            False,  # Curvature
            "Calculation of grainstats disabled, returning empty dataframe and empty height_profiles.",
            "",
            id="Filtering and Grain enabled",
        ),
        pytest.param(
            True,  # Filter
            True,  # Grains
            True,  # Grainstats
            False,  # Disordered Tracing
            False,  # Nodestats
            False,  # Ordered tracing
            False,  # Splining
            False,  # Curvature
            "Processing grain",
            "Calculation of Disordered Tracing disabled, returning empty dictionary.",
            id="Filtering, Grain and GrainStats enabled",
        ),
        pytest.param(
            True,  # Filter
            True,  # Grains
            True,  # Grainstats
            True,  # Disordered Tracing
            True,  # Nodestats
            True,  # Ordered tracing
            True,  # Splining
            False,  # Curvature
            "Processing grain",
            "Calculation of Curvature Stats disabled, returning None.",
            id="All but curvature enabled",
        ),
        # @ns-rse 2024-09-13 : Parameters need updating so test is performed.
        # pytest.param(
        #     True,
        #     True,
        #     True,
        #     True,
        #     "Traced grain 3 of 3",
        #     "Combining ['above'] grain statistics and dnatracing statistics",
        #     id="Filtering, Grain, GrainStats and DNA Tracing enabled",
        # ),
    ],
)
def test_process_stages(
    process_scan_config: dict,
    load_scan_data: LoadScans,
    tmp_path: Path,
    filter_run: bool,
    grains_run: bool,
    grainstats_run: bool,
    disordered_tracing_run: bool,
    nodestats_run: bool,
    ordered_tracing_run: bool,
    splining_run: bool,
    curvature_run: bool,
    log_msg1: str,
    log_msg2: str,
    caplog,
) -> None:
    """Regression test for checking the process_scan functions correctly when specific sections are disabled.

    Currently there is no test for having later stages (e.g. DNA Tracing or Grainstats) enabled when Filters and/or
    Grainstats are disabled. Whislt possible it is expected that users understand the need to run earlier stages before
    later stages can run and do not disable earlier stages.
    """
    caplog.set_level(logging.DEBUG, LOGGER_NAME)
    img_dic = load_scan_data.img_dict
    process_scan_config["filter"]["run"] = filter_run
    process_scan_config["grains"]["run"] = grains_run
    process_scan_config["grainstats"]["run"] = grainstats_run
    process_scan_config["disordered_tracing"]["run"] = disordered_tracing_run
    process_scan_config["nodestats"]["run"] = nodestats_run
    process_scan_config["ordered_tracing"]["run"] = ordered_tracing_run
    process_scan_config["splining"]["run"] = splining_run
    process_scan_config["curvature"]["run"] = curvature_run
    _, _, _, _, _, _ = process_scan(
        topostats_object=img_dic["minicircle_small"],
        base_dir=BASE_DIR,
        filter_config=process_scan_config["filter"],
        grains_config=process_scan_config["grains"],
        grainstats_config=process_scan_config["grainstats"],
        disordered_tracing_config=process_scan_config["disordered_tracing"],
        nodestats_config=process_scan_config["nodestats"],
        ordered_tracing_config=process_scan_config["ordered_tracing"],
        splining_config=process_scan_config["splining"],
        curvature_config=process_scan_config["curvature"],
        plotting_config=process_scan_config["plotting"],
        output_dir=tmp_path,
    )

    assert log_msg1 in caplog.text
    assert log_msg2 in caplog.text


def test_process_scan_no_grains(process_scan_config: dict, load_scan_data: LoadScans, tmp_path: Path, caplog) -> None:
    """Test handling no grains found during grains.find_grains()."""
    img_dic = load_scan_data.img_dict
    process_scan_config["grains"]["threshold_std_dev"]["above"] = 1000
    process_scan_config["filter"]["remove_scars"]["run"] = False
    _, _, _, _, _, _ = process_scan(
        topostats_object=img_dic["minicircle_small"],
        base_dir=BASE_DIR,
        filter_config=process_scan_config["filter"],
        grains_config=process_scan_config["grains"],
        grainstats_config=process_scan_config["grainstats"],
        disordered_tracing_config=process_scan_config["disordered_tracing"],
        nodestats_config=process_scan_config["nodestats"],
        ordered_tracing_config=process_scan_config["ordered_tracing"],
        splining_config=process_scan_config["splining"],
        curvature_config=process_scan_config["curvature"],
        plotting_config=process_scan_config["plotting"],
        output_dir=tmp_path,
    )
    assert "Grains found: 0 above, 0 below" in caplog.text
    assert "No grains found, skipping grainstats and tracing stages." in caplog.text


def test_run_filters(process_scan_config: dict, load_scan_data: LoadScans, tmp_path: Path) -> None:
    """Test the filter wrapper function of processing.py."""
    img_dict = load_scan_data.img_dict
    unprocessed_image = img_dict["minicircle_small"]["image_original"]
    pixel_to_nm_scaling = img_dict["minicircle_small"]["pixel_to_nm_scaling"]
    flattened_image = run_filters(
        unprocessed_image=unprocessed_image,
        pixel_to_nm_scaling=pixel_to_nm_scaling,
        filename="dummy filename",
        filter_out_path=tmp_path,
        core_out_path=tmp_path,
        filter_config=process_scan_config["filter"],
        plotting_config=process_scan_config["plotting"],
    )
    assert isinstance(flattened_image, np.ndarray)
    assert flattened_image.shape == (64, 64)
    assert np.sum(flattened_image) == pytest.approx(1172.6088236592373)


def test_run_grains(process_scan_config: dict, tmp_path: Path) -> None:
    """Test the grains wrapper function of processing.py."""
    flattened_image = np.load("./tests/resources/minicircle_cropped_flattened.npy")
    grains_config = process_scan_config["grains"]
    grains_config["threshold_method"] = "absolute"
    grains_config["direction"] = "both"
    grains_config["threshold_absolute"]["above"] = 1.0
    grains_config["threshold_absolute"]["below"] = -0.4
    grains_config["area_thresholds"]["above"] = [20, 10000000]
    grains_config["area_thresholds"]["below"] = [20, 10000000]

    imagegraincrops = run_grains(
        image=flattened_image,
        pixel_to_nm_scaling=0.4940029296875,
        filename="dummy filename",
        grain_out_path=tmp_path,
        core_out_path=tmp_path,
        grains_config=grains_config,
        plotting_config=process_scan_config["plotting"],
    )

    assert isinstance(imagegraincrops, ImageGrainCrops)
    assert isinstance(imagegraincrops.above, GrainCropsDirection)
    assert len(imagegraincrops.above.crops) == 6
    # Floating point errors mean that on different systems, different results are
    # produced for such generous thresholds. This is not an issue for more stringent
    # thresholds.
    assert isinstance(imagegraincrops.below, GrainCropsDirection)
    assert len(imagegraincrops.below.crops) == 2


def test_run_grainstats(process_scan_config: dict, tmp_path: Path) -> None:
    """Test the grainstats_wrapper function of processing.py."""
    with Path.open(  # pylint: disable=unspecified-encoding
        RESOURCES / "minicircle_cropped_imagegraincrops.pkl", "rb"
    ) as f:
        image_grain_crops = pickle.load(f)
    grainstats_df, _, grain_crops = run_grainstats(
        image_grain_crops=image_grain_crops,
        filename="dummy filename",
        basename=RESOURCES,
        grainstats_config=process_scan_config["grainstats"],
        plotting_config=process_scan_config["plotting"],
        grain_out_path=tmp_path,
    )

    GRAIN_CROP_ATTRIBUTES = [
        "bbox",
        "debug_locate_difference",
        "filename",
        "grain_crop_to_dict",
        "height_profiles",
        "image",
        "mask",
        "padding",
        "pixel_to_nm_scaling",
        "stats",
    ]
    assert isinstance(grainstats_df, pd.DataFrame)
    # Expect 6 grains in the above direction for cropped minicircle
    assert grainstats_df.shape[0] == 6
    assert len(grainstats_df.columns) == 26
    assert isinstance(grain_crops, dict)
    assert len(grain_crops) == 6
    for grain_crop in grain_crops.values():
        assert isinstance(grain_crop, GrainCrop)
        assert all(x in dir(grain_crop) for x in GRAIN_CROP_ATTRIBUTES)


# ns-rse 2024-09-11 : Test disabled as run_dnatracing() has been removed in refactoring, needs updating/replacing to
#                     reflect the revised workflow/functions.
# def test_run_dnatracing(process_scan_config: dict, tmp_path: Path) -> None:
#     """Test the dnatracing_wrapper function of processing.py."""
#     flattened_image = np.load("./tests/resources/minicircle_cropped_flattened.npy")
#     mask_above = np.load("./tests/resources/minicircle_cropped_masks_above.npy")
#     mask_below = np.load("./tests/resources/minicircle_cropped_masks_below.npy")
#     grain_masks = {"above": mask_above, "below": mask_below}

#     dnatracing_df, grain_trace_data = run_dnatracing(
#         image=flattened_image,
#         grain_masks=grain_masks,
#         pixel_to_nm_scaling=0.4940029296875,
#         image_path=tmp_path,
#         filename="dummy filename",
#         core_out_path=tmp_path,
#         grain_out_path=tmp_path,
#         dnatracing_config=process_scan_config["dnatracing"],
#         plotting_config=process_scan_config["plotting"],
#         results_df=pd.read_csv("./tests/resources/minicircle_cropped_grainstats.csv"),
#     )

#     assert isinstance(grain_trace_data, dict)
#     assert list(grain_trace_data.keys()) == ["above", "below"]
#     assert isinstance(dnatracing_df, pd.DataFrame)
#     assert dnatracing_df.shape[0] == 13
#     assert len(dnatracing_df.columns) == 26
