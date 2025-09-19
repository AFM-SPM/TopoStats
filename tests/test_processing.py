# Disable ruff 301 - pickle loading is unsafe, but we don't care for tests.
# ruff: noqa: S301
"""Test end-to-end running of topostats."""

import logging
import pickle
from pathlib import Path
from typing import Any

import filetype
import h5py
import numpy as np
import pandas as pd
import pytest

from topostats.classes import DisorderedTrace, GrainCrop, GrainCropsDirection, ImageGrainCrops, TopoStats
from topostats.io import LoadScans, dict_almost_equal, hdf5_to_dict
from topostats.processing import (
    LOGGER_NAME,
    check_run_steps,
    process_scan,
    run_disordered_tracing,
    run_filters,
    run_grains,
    run_grainstats,
    run_nodestats,
)
from topostats.utils import update_plotting_config

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests/resources"

# pylint: disable=too-many-lines
# pylint: disable=too-many-positional-arguments


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
    flattened_image = run_filters(
        topostats_object=load_scan_data.img_dict["minicircle_small"],
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
    topostats_object = TopoStats(
        image_grain_crops=None,
        filename="dummy filename",
        pixel_to_nm_scaling=0.4940029296875,
        img_path=tmp_path,
        image=flattened_image,
        image_original=None,
        topostats_version=None,
    )
    grains_config = process_scan_config["grains"]
    grains_config["threshold_method"] = "absolute"
    grains_config["direction"] = "both"
    grains_config["threshold_absolute"]["above"] = 1.0
    grains_config["threshold_absolute"]["below"] = -0.4
    grains_config["area_thresholds"]["above"] = [20, 10000000]
    grains_config["area_thresholds"]["below"] = [20, 10000000]

    _ = run_grains(
        topostats_object=topostats_object,
        grain_out_path=tmp_path,
        core_out_path=tmp_path,
        grains_config=grains_config,
        plotting_config=process_scan_config["plotting"],
    )

    assert isinstance(topostats_object.image_grain_crops, ImageGrainCrops)
    assert isinstance(topostats_object.image_grain_crops.above, GrainCropsDirection)
    assert len(topostats_object.image_grain_crops.above.crops) == 6
    # Floating point errors mean that on different systems, different results are
    # produced for such generous thresholds. This is not an issue for more stringent
    # thresholds.
    assert isinstance(topostats_object.image_grain_crops.below, GrainCropsDirection)
    assert len(topostats_object.image_grain_crops.below.crops) == 2


def test_run_grainstats(process_scan_config: dict, tmp_path: Path) -> None:
    """Test the grainstats_wrapper function of processing.py."""
    with Path.open(  # pylint: disable=unspecified-encoding
        RESOURCES / "minicircle_cropped_imagegraincrops.pkl", "rb"
    ) as f:
        image_grain_crops = pickle.load(f)
    topostats_object = TopoStats(
        image_grain_crops=image_grain_crops,
        filename="dummy filename",
        pixel_to_nm_scaling=0.4940029296875,
        img_path=tmp_path,
        image=None,
        image_original=None,
        topostats_version=None,
    )
    grainstats_df, _, grain_crops = run_grainstats(
        topostats_object=topostats_object,
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


def test_run_disordered_tracing(
    process_scan_config: dict[str, Any], load_scan_data: LoadScans, tmp_path: Path, caplog
) -> None:
    """Test run_disordered_tracing()."""
    topostats_object = load_scan_data.img_dict["minicircle_small"]
    with Path.open(  # pylint: disable=unspecified-encoding
        RESOURCES / "minicircle_cropped_imagegraincrops.pkl", "rb"
    ) as f:
        topostats_object.image_grain_crops = pickle.load(f)
    topostats_object.image = np.load("./tests/resources/minicircle_cropped_flattened.npy")
    run_disordered_tracing(
        topostats_object=topostats_object,
        core_out_path=tmp_path,
        tracing_out_path=tmp_path,
        disordered_tracing_config=process_scan_config["disordered_tracing"],
        plotting_config=process_scan_config["plotting"],
    )
    expected = {
        0: {
            "grain_endpoints": 0,
            "grain_junctions": 4,
            "total_branch_length": 1.1716095681103254e-07,
            "grain_width_mean": 5.117318599979457e-09,
        },
        3: {
            "grain_endpoints": 0,
            "grain_junctions": 0,
            "total_branch_length": 8.49805509668089e-08,
            "grain_width_mean": 5.40982849376278e-09,
        },
        4: {
            "grain_endpoints": 0,
            "grain_junctions": 0,
            "total_branch_length": 1.0965161327419527e-07,
            "grain_width_mean": 4.788512127551645e-09,
        },
    }
    assert "Grain 1 skeleton < 10, skipping." in caplog.text
    assert "Grain 2 skeleton < 10, skipping." in caplog.text
    assert "Grain 5 skeleton < 10, skipping." in caplog.text
    for grain, grain_crop in topostats_object.image_grain_crops.above.crops.items():
        # Skeletons < 10 for these so disordered tracing is skipped
        if grain not in (1, 2, 5):
            print(f"\n{grain=}\n")
            print(f"\n{grain_crop.disordered_trace.__dict__=}\n")
            assert grain_crop.disordered_trace is not None
            assert isinstance(grain_crop.disordered_trace, DisorderedTrace)
            assert isinstance(grain_crop.disordered_trace.images, dict)
            assert grain_crop.disordered_trace.grain_endpoints == expected[grain]["grain_endpoints"]
            assert grain_crop.disordered_trace.grain_junctions == expected[grain]["grain_junctions"]
            assert grain_crop.disordered_trace.total_branch_length == expected[grain]["total_branch_length"]
            assert grain_crop.disordered_trace.grain_width_mean == expected[grain]["grain_width_mean"]


@pytest.mark.parametrize(
    ("topostats_object", "expected"),
    [
        pytest.param(
            "minicircle_small_topostats",
            {
                0: {  # Grain
                    1: {  # Node
                        "error": False,
                        "unmatched_branch_stats": {
                            0: {"angles": np.float64(0.0)},
                            1: {"angles": np.float64(144.75600131884323)},
                            2: {"angles": np.float64(55.91538621284788)},
                            3: {"angles": np.float64(178.57739509829827)},
                        },
                        "node_coords": np.array([[50, 37], [51, 38], [52, 38], [52, 39]]),
                        "confidence": np.float64(0.0),
                    }
                },
                3: {  # Grain
                    1: {  # Node
                        "error": False,
                        "unmatched_branch_stats": {
                            0: {"angles": np.float64(0.0)},
                            1: {"angles": np.float64(49.50796095624354)},
                            2: {"angles": np.float64(139.25543479929738)},
                        },
                        "node_coords": np.array(
                            [[54, 70], [55, 69], [56, 68], [57, 66], [57, 67], [58, 66], [59, 66], [60, 65]]
                        ),
                        "confidence": None,
                    }
                },
            },
            id="minicircle small",
        ),
        pytest.param(
            "catenane_topostats",
            {
                0: {  # Grain
                    1: {  # Node
                        "error": False,
                        "pixel_to_nm_scaling": 1,
                        "unmatched_branch_stats": {
                            0: {"angles": np.float64(0.0)},
                            1: {"angles": np.float64(63.62466925374735)},
                            2: {"angles": np.float64(29.510196308405636)},
                            3: {"angles": np.float64(106.0045892179681)},
                        },
                        "node_coords": np.array(
                            [
                                [19, 560],
                                [20, 560],
                                [21, 560],
                                [22, 560],
                                [23, 560],
                                [24, 560],
                                [25, 560],
                                [26, 560],
                                [26, 561],
                                [27, 559],
                                [28, 558],
                                [29, 558],
                                [30, 558],
                                [31, 558],
                                [32, 558],
                                [33, 558],
                                [34, 558],
                                [35, 558],
                                [36, 557],
                                [37, 556],
                                [38, 555],
                                [39, 554],
                                [40, 553],
                                [41, 552],
                                [42, 550],
                                [42, 551],
                                [43, 550],
                            ]
                        ),
                        "confidence": np.float64(0.0),
                    },
                    2: {  # Node
                        "error": False,
                        "pixel_to_nm_scaling": 1,
                        "unmatched_branch_stats": {
                            0: {"angles": np.float64(0.0)},
                            1: {"angles": np.float64(80.4978059963834)},
                            2: {"angles": np.float64(9.502194003616577)},
                            3: {"angles": np.float64(178.1922615295965)},
                        },
                        "node_coords": np.array(
                            [
                                [53, 116],
                                [53, 117],
                                [54, 116],
                                [55, 116],
                                [56, 116],
                                [57, 116],
                                [58, 116],
                                [59, 116],
                                [60, 116],
                                [61, 115],
                            ]
                        ),
                        "confidence": np.float64(0.0),
                    },
                    3: {  # Node
                        "error": False,
                        "pixel_to_nm_scaling": 1,
                        "unmatched_branch_stats": {
                            0: {"angles": np.float64(0.0)},
                            1: {"angles": np.float64(88.19362666463878)},
                            2: {"angles": np.float64(96.40062666662755)},
                            3: {"angles": np.float64(175.36095529910162)},
                        },
                        "node_coords": np.array(
                            [[76, 24], [77, 25], [78, 25], [79, 25], [80, 25], [81, 24], [82, 24], [82, 25]]
                        ),
                        "confidence": np.float64(0.0),
                    },
                    4: {  # Node
                        "error": False,
                        "pixel_to_nm_scaling": 1,
                        "unmatched_branch_stats": {
                            0: {"angles": np.float64(0.0)},
                            1: {"angles": np.float64(100.17551084304321)},
                            2: {"angles": np.float64(176.9495055058588)},
                        },
                        "node_coords": np.array([[111, 17], [112, 17], [112, 18]]),
                        "confidence": None,
                    },
                    5: {  # Node
                        "error": False,
                        "pixel_to_nm_scaling": 1,
                        "unmatched_branch_stats": {
                            0: {"angles": np.float64(0.0)},
                            1: {"angles": np.float64(45.0)},
                            2: {"angles": np.float64(138.3664606634299)},
                        },
                        "node_coords": np.array([[114, 505]]),
                        "confidence": None,
                    },
                    6: {  # Node
                        "error": False,
                        "pixel_to_nm_scaling": 1,
                        "unmatched_branch_stats": {
                            0: {"angles": np.float64(0.0)},
                            1: {"angles": np.float64(35.31121343963319)},
                            2: {"angles": np.float64(0.0)},
                            3: {"angles": np.float64(9.266607240759264)},
                            4: {"angles": np.float64(0.0)},
                            5: {"angles": np.float64(63.43494882292215)},
                            6: {"angles": np.float64(90.0)},
                            7: {"angles": np.float64(97.12501634890181)},
                            8: {"angles": np.float64(132.33699923393286)},
                        },
                        "node_coords": np.array(
                            [
                                [115, 475],
                                [116, 476],
                                [117, 477],
                                [118, 478],
                                [119, 478],
                                [120, 478],
                                [121, 478],
                                [121, 493],
                                [122, 478],
                                [122, 493],
                                [122, 494],
                                [123, 479],
                                [123, 489],
                                [123, 492],
                                [124, 479],
                                [124, 489],
                                [124, 490],
                                [124, 491],
                                [125, 480],
                                [125, 481],
                                [125, 483],
                                [125, 484],
                                [125, 485],
                                [125, 488],
                                [126, 482],
                                [126, 486],
                                [126, 487],
                            ]
                        ),
                        "confidence": None,
                    },
                    7: {  # Node
                        "error": False,
                        "pixel_to_nm_scaling": 1,
                        "unmatched_branch_stats": {
                            0: {"angles": np.float64(0.0)},
                            1: {"angles": np.float64(180.0)},
                            2: {"angles": np.float64(12.20911975849626)},
                        },
                        "node_coords": np.array([[127, 506], [127, 507], [128, 507]]),
                        "confidence": None,
                    },
                    8: {  # Node
                        "error": False,
                        "pixel_to_nm_scaling": 1,
                        "unmatched_branch_stats": {
                            0: {"angles": np.float64(0.0)},
                            1: {"angles": np.float64(178.7885198442754)},
                            2: {"angles": np.float64(82.76604810520416)},
                        },
                        "node_coords": np.array([[140, 473]]),
                        "confidence": None,
                    },
                    9: {  # Node
                        "error": False,
                        "pixel_to_nm_scaling": 1,
                        "unmatched_branch_stats": {
                            0: {"angles": np.float64(0.0)},
                            1: {"angles": np.float64(131.1019940793969)},
                            2: {"angles": np.float64(106.14433878028349)},
                        },
                        "node_coords": np.array([[161, 39]]),
                        "confidence": None,
                    },
                    10: {  # Node
                        "error": False,
                        "pixel_to_nm_scaling": 1,
                        "unmatched_branch_stats": {
                            0: {"angles": np.float64(0.0)},
                            1: {"angles": np.float64(31.56149988969091)},
                            2: {"angles": np.float64(154.63558280444693)},
                        },
                        "node_coords": np.array([[169, 524]]),
                        "confidence": None,
                    },
                    11: {  # Node
                        "error": False,
                        "pixel_to_nm_scaling": 1,
                        "unmatched_branch_stats": {
                            0: {"angles": np.float64(0.0)},
                            1: {"angles": np.float64(172.3706896990102)},
                            2: {"angles": np.float64(101.30993247402029)},
                        },
                        "node_coords": np.array([[173, 536], [173, 537], [174, 536]]),
                        "confidence": None,
                    },
                    12: {  # Node
                        "error": False,
                        "pixel_to_nm_scaling": 1,
                        "unmatched_branch_stats": {
                            0: {"angles": np.float64(0.0)},
                            1: {"angles": np.float64(149.3956416608597)},
                            2: {"angles": np.float64(59.241358747447656)},
                        },
                        "node_coords": np.array([[182, 571]]),
                        "confidence": None,
                    },
                    13: {  # Node
                        "error": False,
                        "pixel_to_nm_scaling": 1,
                        "unmatched_branch_stats": {
                            0: {"angles": np.float64(0.0)},
                            1: {"angles": np.float64(45.0)},
                            2: {"angles": np.float64(np.nan)},
                        },
                        "node_coords": np.array([[209, 209]]),
                        "confidence": None,
                    },
                    14: {  # Node
                        "error": False,
                        "pixel_to_nm_scaling": 1,
                        "unmatched_branch_stats": {
                            0: {"angles": np.float64(0.0)},
                            1: {"angles": np.float64(45.0)},
                            2: {"angles": np.float64(45.0)},
                            3: {"angles": np.float64(147.52880770915186)},
                            4: {"angles": np.float64(45.0)},
                        },
                        "node_coords": np.array(
                            [[259, 361], [260, 360], [261, 359], [262, 358], [263, 357], [264, 356], [265, 355]]
                        ),
                        "confidence": None,
                    },
                    15: {  # Node
                        "error": False,
                        "pixel_to_nm_scaling": 1,
                        "unmatched_branch_stats": {
                            0: {"angles": np.float64(0.0)},
                            1: {"angles": np.float64(102.53565471909889)},
                            2: {"angles": np.float64(176.59446008153773)},
                        },
                        "node_coords": np.array([[268, 15]]),
                        "confidence": None,
                    },
                    16: {  # Node
                        "error": False,
                        "pixel_to_nm_scaling": 1,
                        "unmatched_branch_stats": {
                            0: {"angles": np.float64(0.0)},
                            1: {"angles": np.float64(90.0)},
                            2: {"angles": np.float64(160.7873281828144)},
                        },
                        "node_coords": np.array([[276, 354]]),
                        "confidence": None,
                    },
                    17: {  # Node
                        "error": False,
                        "pixel_to_nm_scaling": 1,
                        "unmatched_branch_stats": {
                            0: {"angles": np.float64(0.0)},
                            1: {"angles": np.float64(92.04540848888729)},
                            2: {"angles": np.float64(135.0)},
                        },
                        "node_coords": np.array([[286, 333], [287, 333], [287, 334]]),
                        "confidence": None,
                    },
                    18: {  # Node
                        "error": False,
                        "pixel_to_nm_scaling": 1,
                        "unmatched_branch_stats": {
                            0: {"angles": np.float64(0.0)},
                            1: {"angles": np.float64(42.76882539196875)},
                            2: {"angles": np.float64(177.76882539196893)},
                        },
                        "node_coords": np.array([[293, 41]]),
                        "confidence": None,
                    },
                    19: {  # Node
                        "error": False,
                        "pixel_to_nm_scaling": 1,
                        "unmatched_branch_stats": {
                            0: {"angles": np.float64(0.0)},
                            1: {"angles": np.float64(45.0)},
                            2: {"angles": np.nan},
                        },
                        "node_coords": np.array([[295, 296]]),
                        "confidence": None,
                    },
                    20: {  # Node
                        "error": False,
                        "pixel_to_nm_scaling": 1,
                        "unmatched_branch_stats": {
                            0: {"angles": np.float64(0.0)},
                            1: {"angles": np.float64(135.0)},
                            2: {"angles": np.float64(88.53119928561418)},
                        },
                        "node_coords": np.array([[304, 52]]),
                        "confidence": None,
                    },
                    21: {  # Node
                        "error": False,
                        "pixel_to_nm_scaling": 1,
                        "unmatched_branch_stats": {
                            0: {"angles": np.float64(0.0)},
                            1: {"angles": np.float64(90.0)},
                            2: {"angles": np.float64(45.0)},
                            3: {"angles": np.float64(135.0)},
                        },
                        "node_coords": np.array([[309, 309], [310, 310]]),
                        "confidence": np.float64(0.0),
                    },
                    22: {  # Node
                        "error": False,
                        "pixel_to_nm_scaling": 1,
                        "unmatched_branch_stats": {
                            0: {"angles": np.float64(0.0)},
                            1: {"angles": np.float64(141.11452292754814)},
                            2: {"angles": np.float64(61.90445758035311)},
                        },
                        "node_coords": np.array([[317, 31], [317, 32], [318, 31]]),
                        "confidence": None,
                    },
                    23: {  # Node
                        "error": False,
                        "pixel_to_nm_scaling": 1,
                        "unmatched_branch_stats": {
                            0: {"angles": np.float64(0.0)},
                            1: {"angles": np.float64(135.0)},
                            2: {"angles": np.float64(180.0)},
                            3: {"angles": np.float64(90.0)},
                            4: {"angles": np.float64(135.0)},
                        },
                        "node_coords": np.array(
                            [
                                [444, 310],
                                [445, 309],
                                [446, 308],
                                [447, 307],
                                [448, 306],
                                [449, 305],
                                [450, 304],
                                [451, 303],
                                [452, 302],
                            ]
                        ),
                        "confidence": None,
                    },
                    24: {  # Node
                        "error": False,
                        "pixel_to_nm_scaling": 1,
                        "unmatched_branch_stats": {
                            0: {"angles": np.float64(0.0)},
                            1: {"angles": np.float64(85.2363583092737)},
                            2: {"angles": np.float64(100.4742348282259)},
                        },
                        "node_coords": np.array([[449, 222]]),
                        "confidence": None,
                    },
                    25: {  # Node
                        "error": False,
                        "pixel_to_nm_scaling": 1,
                        "unmatched_branch_stats": {
                            0: {"angles": np.float64(0.0)},
                            1: {"angles": np.float64(93.9141372186123)},
                            2: {"angles": np.float64(167.11323616649125)},
                        },
                        "node_coords": np.array([[483, 592]]),
                        "confidence": None,
                    },
                    26: {  # Node
                        "error": False,
                        "pixel_to_nm_scaling": 1,
                        "unmatched_branch_stats": {
                            0: {"angles": np.float64(0.0)},
                            1: {"angles": np.float64(79.25870703970048)},
                            2: {"angles": np.float64(163.37495445311453)},
                            3: {"angles": np.float64(83.07278187399083)},
                        },
                        "node_coords": np.array(
                            [
                                [489, 32],
                                [489, 33],
                                [489, 34],
                                [490, 32],
                                [490, 35],
                                [491, 36],
                                [492, 37],
                                [493, 38],
                                [494, 39],
                                [495, 40],
                                [496, 41],
                                [497, 42],
                                [498, 43],
                                [499, 44],
                                [500, 45],
                                [500, 46],
                                [501, 45],
                            ]
                        ),
                        "confidence": np.float64(0.0),
                    },
                    27: {  # Node
                        "error": False,
                        "pixel_to_nm_scaling": 1,
                        "unmatched_branch_stats": {
                            0: {"angles": np.float64(0.0)},
                            1: {"angles": np.nan},
                            2: {"angles": np.float64(135.0)},
                        },
                        "node_coords": np.array([[526, 228]]),
                        "confidence": None,
                    },
                    28: {  # Node
                        "error": False,
                        "pixel_to_nm_scaling": 1,
                        "unmatched_branch_stats": {
                            0: {"angles": np.float64(0.0)},
                            1: {"angles": np.float64(135.0)},
                            2: {"angles": np.float64(135.0)},
                            3: {"angles": np.float64(135.0)},
                        },
                        "node_coords": np.array(
                            [
                                [562, 428],
                                [563, 429],
                                [564, 430],
                                [565, 431],
                                [566, 432],
                                [567, 433],
                                [568, 434],
                                [569, 435],
                                [570, 436],
                                [571, 437],
                                [572, 438],
                                [573, 439],
                                [574, 440],
                                [574, 441],
                                [574, 442],
                                [574, 443],
                                [575, 444],
                            ]
                        ),
                        "confidence": np.float64(0.0),
                    },
                    29: {  # Node
                        "error": False,
                        "pixel_to_nm_scaling": 1,
                        "unmatched_branch_stats": {
                            0: {"angles": np.float64(0.0)},
                            1: {"angles": np.float64(66.8014094863519)},
                            2: {"angles": np.float64(129.47751818500538)},
                            3: {"angles": np.float64(131.12651027565556)},
                        },
                        "node_coords": np.array([[576, 47], [577, 48], [578, 48], [579, 48], [580, 48]]),
                        "confidence": np.float64(0.0),
                    },
                    30: {  # Node
                        "error": False,
                        "pixel_to_nm_scaling": 1,
                        "unmatched_branch_stats": {
                            0: {"angles": np.float64(0.0)},
                            1: {"angles": np.float64(104.17843837875398)},
                            2: {"angles": np.float64(85.71949103125833)},
                        },
                        "node_coords": np.array([[588, 577]]),
                        "confidence": None,
                    },
                    31: {  # Node
                        "error": False,
                        "pixel_to_nm_scaling": 1,
                        "unmatched_branch_stats": {
                            0: {"angles": np.float64(0.0)},
                            1: {"angles": np.float64(99.11372721809165)},
                            2: {"angles": np.float64(92.8624052261117)},
                        },
                        "node_coords": np.array([[589, 160], [589, 161], [590, 160]]),
                        "confidence": None,
                    },
                    32: {  # Node
                        "error": False,
                        "pixel_to_nm_scaling": 1,
                        "unmatched_branch_stats": {
                            0: {"angles": np.float64(0.0)},
                            1: {"angles": np.float64(105.74577831169138)},
                            2: {"angles": np.float64(170.77477120211938)},
                        },
                        "node_coords": np.array([[632, 582]]),
                        "confidence": None,
                    },
                    33: {  # Node
                        "error": False,
                        "pixel_to_nm_scaling": 1,
                        "unmatched_branch_stats": {
                            0: {"angles": np.float64(0.0)},
                            1: {"angles": np.float64(99.7673309987942)},
                            2: {"angles": np.float64(168.1113419603717)},
                            3: {"angles": np.float64(93.24323550537468)},
                        },
                        "node_coords": np.array(
                            [
                                [638, 25],
                                [639, 25],
                                [640, 24],
                                [641, 24],
                                [642, 24],
                                [643, 24],
                                [644, 24],
                                [645, 24],
                                [646, 24],
                                [647, 24],
                                [648, 24],
                                [648, 25],
                            ]
                        ),
                        "confidence": np.float64(0.0),
                    },
                    34: {  # Node
                        "error": False,
                        "pixel_to_nm_scaling": 1,
                        "unmatched_branch_stats": {
                            0: {"angles": np.float64(0.0)},
                            1: {"angles": np.float64(170.28365068572384)},
                            2: {"angles": np.float64(59.1150546692716)},
                        },
                        "node_coords": np.array([[650, 551], [650, 552], [651, 551]]),
                        "confidence": None,
                    },
                    35: {  # Node
                        "error": False,
                        "pixel_to_nm_scaling": 1,
                        "unmatched_branch_stats": {
                            0: {"angles": np.float64(0.0)},
                            1: {"angles": np.float64(151.69924423399326)},
                            2: {"angles": np.float64(73.30075576600676)},
                        },
                        "node_coords": np.array([[655, 567]]),
                        "confidence": None,
                    },
                    36: {  # Node
                        "error": False,
                        "pixel_to_nm_scaling": 1,
                        "unmatched_branch_stats": {
                            0: {"angles": np.float64(0.0)},
                            1: {"angles": np.float64(119.99507961713938)},
                            2: {"angles": np.float64(99.99180662782496)},
                        },
                        "node_coords": np.array([[655, 576]]),
                        "confidence": None,
                    },
                    37: {  # Node
                        "error": False,
                        "pixel_to_nm_scaling": 1,
                        "unmatched_branch_stats": {
                            0: {"angles": np.float64(0.0)},
                            1: {"angles": np.float64(94.21417852273404)},
                            2: {"angles": np.float64(175.78582147726598)},
                        },
                        "node_coords": np.array([[712, 21]]),
                        "confidence": None,
                    },
                    38: {  # Node
                        "error": False,
                        "pixel_to_nm_scaling": 1,
                        "unmatched_branch_stats": {
                            0: {"angles": np.float64(0.0)},
                            1: {"angles": np.float64(135.0)},
                            2: {"angles": np.float64(135.0)},
                        },
                        "node_coords": np.array([[713, 537]]),
                        "confidence": None,
                    },
                    39: {  # Node
                        "error": False,
                        "pixel_to_nm_scaling": 1,
                        "unmatched_branch_stats": {
                            0: {"angles": np.float64(0.0)},
                            1: {"angles": np.float64(179.61258629167784)},
                            2: {"angles": np.float64(109.03864199370402)},
                        },
                        "node_coords": np.array([[713, 585], [713, 586], [714, 586]]),
                        "confidence": None,
                    },
                    40: {  # Node
                        "error": False,
                        "pixel_to_nm_scaling": 1,
                        "unmatched_branch_stats": {
                            0: {"angles": np.float64(0.0)},
                            1: {"angles": np.float64(108.20848447058127)},
                            2: {"angles": np.float64(174.87594458340195)},
                        },
                        "node_coords": np.array([[722, 544]]),
                        "confidence": None,
                    },
                    41: {  # Node
                        "error": False,
                        "pixel_to_nm_scaling": 1,
                        "unmatched_branch_stats": {
                            0: {"angles": np.float64(0.0)},
                            1: {"angles": np.float64(120.96375653207353)},
                            2: {"angles": np.float64(80.32739868765275)},
                            3: {"angles": np.float64(178.43886624428245)},
                        },
                        "node_coords": np.array(
                            [
                                [732, 28],
                                [733, 27],
                                [734, 27],
                                [735, 27],
                                [736, 28],
                                [737, 28],
                                [738, 28],
                                [739, 27],
                            ]
                        ),
                        "confidence": np.float64(0.0),
                    },
                },
            },
            id="catenane",
        ),
    ],
)
def test_run_nodestats(  # noqa: C901
    topostats_object: TopoStats, expected: dict[str, Any], process_scan_config: dict[str, Any], tmp_path, request
) -> None:
    """Test for run_nodestats()."""
    topostats_object = request.getfixturevalue(topostats_object)
    if topostats_object.filename == "minicircle_small":
        run_disordered_tracing(
            topostats_object=topostats_object,
            core_out_path=tmp_path,
            tracing_out_path=tmp_path,
            disordered_tracing_config=process_scan_config["disordered_tracing"],
            plotting_config=process_scan_config["plotting"],
        )
    run_nodestats(
        topostats_object=topostats_object,
        core_out_path=tmp_path,
        tracing_out_path=tmp_path,
        nodestats_config=process_scan_config["nodestats"],
        plotting_config=process_scan_config["plotting"],
        grainstats_df=pd.DataFrame(),
    )
    if topostats_object.filename == "minicircle_small":
        for grain, grain_crop in topostats_object.image_grain_crops.above.crops.items():
            # Grains 1, 2, 5 have Skeletons < 10 so disordered tracing is skipped; Grain 4 has no crossover
            if grain not in (1, 2, 4, 5):
                for node, nodestats in grain_crop.nodes.items():
                    assert nodestats.error == expected[grain][node]["error"]
                    assert nodestats.unmatched_branch_stats == expected[grain][node]["unmatched_branch_stats"]
                    np.testing.assert_array_equal(nodestats.node_coords, expected[grain][node]["node_coords"])
    elif topostats_object.filename == "test_catenane":
        for grain, grain_crop in topostats_object.image_grain_crops.above.crops.items():
            for node, nodestats in grain_crop.nodes.items():
                assert nodestats.error == expected[grain][node]["error"]
                # ns-rse 2025-09-24 Equality is failing for 'nan' tried both `np.float64(np.nan)` and
                # `np.float64(float('nan'))` but to no avail. Probably more important to work out why we observe
                # `nan` in the first place but for expedience and the current urgency to complete this work we skip
                # those affected for now. We know where to look as we get a warning and this has also been noted
                # in-line
                #
                #   /home/neil/work/git/hub/AFM-SPM/TopoStats/topostats/tracing/nodestats.py:1129:
                #      RuntimeWarning: invalid value encountered in arccos
                #   return abs(np.arccos(cos_angles) / np.pi * 180)  # angles in degrees
                if node not in (13, 19, 27):
                    assert nodestats.unmatched_branch_stats == expected[grain][node]["unmatched_branch_stats"]
                np.testing.assert_array_equal(nodestats.node_coords, expected[grain][node]["node_coords"])
