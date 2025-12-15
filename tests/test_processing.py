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
import pytest

from topostats.classes import DisorderedTrace, GrainCrop, TopoStats
from topostats.config import update_plotting_config
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

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests/resources"

# pylint: disable=too-many-lines
# pylint: disable=too-many-positional-arguments


def test_process_scan_both(regtest, tmp_path, process_scan_config: dict, load_scan_data: LoadScans) -> None:
    """Regression test for checking the process_scan functions correctly."""
    # Ensure there are below grains
    process_scan_config["grains"]["threshold_std_dev"]["below"] = 0.8
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
    # ns-rse 2025-11-16 : Switch to syrupy
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
    # ns-rse 2025-11-16 : Switch to syrupy
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
    process_scan(
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


# ns-rse 2025-11-26 : This seems a lot just to check files are output with the correct extension, I wonder if we could
# check this as part of another test?
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


@pytest.mark.skip(reason="ns-rse 2025-12-12 Need to remove ImageGrainCrops from pickle loaded by fixture.")
def test_run_filters(minicircle_small_topostats: TopoStats, tmp_path: Path) -> None:
    """Test the filter wrapper function of processing.py."""
    run_filters(
        topostats_object=minicircle_small_topostats,
        filter_out_path=tmp_path,
        core_out_path=tmp_path,
    )
    assert isinstance(minicircle_small_topostats.image, np.ndarray)
    assert minicircle_small_topostats.shape == (64, 64)
    assert np.sum(minicircle_small_topostats) == pytest.approx(1172.6088236592373)


def test_run_grains(process_scan_config: dict, tmp_path: Path) -> None:
    """Test the grains wrapper function of processing.py."""
    flattened_image = np.load("./tests/resources/minicircle_cropped_flattened.npy")
    topostats_object = TopoStats(
        grain_crops=None,
        filename="dummy filename",
        pixel_to_nm_scaling=0.4940029296875,
        img_path=tmp_path,
        image=flattened_image,
        image_original=None,
        topostats_version=None,
    )
    topostats_object.config = process_scan_config
    grains_config = process_scan_config["grains"]
    grains_config["threshold_method"] = "absolute"
    grains_config["threshold_absolute"]["above"] = 1.0
    grains_config["threshold_absolute"]["below"] = -0.4
    grains_config["area_thresholds"]["above"] = [20, 10000000]
    grains_config["area_thresholds"]["below"] = [20, 10000000]
    run_grains(
        topostats_object=topostats_object,
        grain_out_path=tmp_path,
        core_out_path=tmp_path,
    )
    assert isinstance(topostats_object.grain_crops, dict)
    # @ns-rse 2025-11-18 - Only getting six above, should be two below
    assert len(topostats_object.grain_crops) == 6
    for _, grain_crop in topostats_object.grain_crops.items():
        assert isinstance(grain_crop, GrainCrop)


@pytest.mark.skip(reason="ns-rse 2025-12-12 Need to remove ImageGrainCrops from pickle loaded by fixture.")
def test_run_grainstats(default_config: dict[str, Any], tmp_path: Path) -> None:
    """Test the grainstats_wrapper function of processing.py."""
    with Path.open(  # pylint: disable=unspecified-encoding
        RESOURCES / "minicircle_cropped_imagegraincrops.pkl", "rb"
    ) as f:
        image_grain_crops = pickle.load(f)
    topostats_object = TopoStats(
        grain_crops=image_grain_crops,
        filename="dummy filename",
        pixel_to_nm_scaling=0.4940029296875,
        img_path=tmp_path,
        image=None,
        image_original=None,
        config=default_config,
        topostats_version=None,
    )
    run_grainstats(
        topostats_object=topostats_object,
        core_out_path=tmp_path,
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
    # assert isinstance(grainstats_df, pd.DataFrame)
    # Expect 6 grains in the above direction for cropped minicircle
    # assert grainstats_df.shape[0] == 6
    # assert len(grainstats_df.columns) == 26
    assert isinstance(topostats_object.grain_crops, dict)
    assert len(topostats_object.grain_crops) == 6
    for _, grain_crop in topostats_object.grain_crops.values():
        assert isinstance(grain_crop, GrainCrop)
        assert all(x in dir(grain_crop) for x in GRAIN_CROP_ATTRIBUTES)


@pytest.mark.parametrize(
    ("topostats_object", "detected_grains", "log_messages", "expected"),
    [
        pytest.param(
            "minicircle_small_post_grainstats",
            [0, 3, 4],
            [
                "Grain 1 skeleton < 10, skipping.",
                "Grain 2 skeleton < 10, skipping.",
                "Grain 5 skeleton < 10, skipping.",
            ],
            {
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
            },
            id="minicircle small",
        ),
        pytest.param(
            "catenanes_post_grainstats",
            [0, 1],
            None,
            {
                0: {
                    "grain_endpoints": 0,
                    "grain_junctions": 14,
                    "total_branch_length": 5.759292550205616e-07,
                    "grain_width_mean": 4.406927419327375e-09,
                },
                # @ns-rse 2025-10-22 : Its strange that the values for these differ because the two grains in
                #                      catenanes are in essence identical being a mirror of each other.
                1: {
                    "grain_endpoints": 0,
                    "grain_junctions": 12,
                    "total_branch_length": 5.746673912389996e-07,
                    "grain_width_mean": 4.456449209835062e-09,
                },
            },
            id="catenanes",
        ),
        pytest.param(
            "rep_int_post_grainstats",
            [0],
            None,
            {
                0: {
                    "grain_endpoints": 0,
                    "grain_junctions": 13,
                    "total_branch_length": 9.685225788725929e-07,
                    "grain_width_mean": 3.031795147452633e-09,
                },
            },
            id="rep int",
        ),
    ],
)
def test_run_disordered_tracing(
    topostats_object: str,
    detected_grains: list[int],
    log_messages: list[str],
    expected: dict[int, Any],
    process_scan_config: dict[str, Any],
    tmp_path,
    caplog,
    request,
) -> None:
    """Test for run_grainstats()."""
    topostats_object: TopoStats = request.getfixturevalue(topostats_object)
    run_disordered_tracing(
        topostats_object=topostats_object,
        core_out_path=tmp_path,
        tracing_out_path=tmp_path,
        disordered_tracing_config=process_scan_config["disordered_tracing"],
        plotting_config=process_scan_config["plotting"],
    )
    # Check log messages
    if log_messages is not None:
        for msg in log_messages:
            assert msg in caplog.text
    # Check grains disordered_trace attribute against expected
    for grain, grain_crop in topostats_object.grain_crops.above.crops.items():
        if grain in detected_grains:
            assert grain_crop.disordered_trace is not None
            assert isinstance(grain_crop.disordered_trace, DisorderedTrace)
            assert isinstance(grain_crop.disordered_trace.images, dict)
            assert grain_crop.disordered_trace.grain_endpoints == expected[grain]["grain_endpoints"]
            assert grain_crop.disordered_trace.grain_junctions == expected[grain]["grain_junctions"]
            assert grain_crop.disordered_trace.total_branch_length == expected[grain]["total_branch_length"]
            assert grain_crop.disordered_trace.grain_width_mean == expected[grain]["grain_width_mean"]


@pytest.mark.parametrize(
    ("topostats_object", "detected_grains", "expected_nodes", "expected_matched_branch"),
    [
        pytest.param(
            "minicircle_small_post_disordered_tracing",
            [0, 3, 4],
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
            },
            None,
            id="minicircle small",
            marks=pytest.mark.skip(reason="disable whilst testing other params"),
        ),
        pytest.param(
            "catenanes_post_disordered_tracing",
            [0, 1],
            {
                0: {  # Grain
                    1: {  # Node
                        "error": False,
                        "pixel_to_nm_scaling": 1,
                        "unmatched_branch_stats": {
                            0: {"angles": np.float64(0.0)},
                            1: {"angles": np.float64(139.2976869956517)},
                            2: {"angles": np.float64(40.75146951858621)},
                            3: {"angles": np.float64(116.78086102349846)},
                        },
                        "node_coords": np.array([[38, 214], [39, 214], [39, 215], [40, 216], [40, 217]]),
                        "confidence": np.float64(0.41698778009744997),
                    },
                    2: {  # Node
                        "error": False,
                        "pixel_to_nm_scaling": 1,
                        "fwhm": {
                            "fwhm": np.float64(13.02537913683945),
                            "half_maxs": [
                                np.float64(-6.7760196954317475),
                                np.float64(6.249359441407702),
                                np.float64(3.379744597269503),
                            ],
                            "peaks": [np.int64(35), np.float64(1.4142135623730951), np.float64(4.242989227790521)],
                        },
                        "unmatched_branch_stats": {
                            0: {"angles": np.float64(0.0)},
                            1: {"angles": np.float64(125.28757605500637)},
                            2: {"angles": np.float64(81.95910195134098)},
                            3: {"angles": np.float64(164.48566237463905)},
                        },
                        "node_coords": np.array(
                            [
                                [41, 152],
                                [41, 153],
                                [42, 154],
                                [42, 161],
                                [42, 162],
                                [42, 163],
                                [42, 164],
                                [42, 165],
                                [43, 155],
                                [43, 156],
                                [43, 157],
                                [43, 158],
                                [43, 159],
                                [43, 160],
                                [43, 165],
                            ]
                        ),
                        "confidence": np.float64(0.24062584379143193),
                    },
                    # @ns-rse 2025-10-22 : There are 41 nodes, excessive to check them all.
                },
            },
            {  # Expected MatchedBranch (only test the first Matched Branch)
                0: {
                    "ordered_coords": np.array(
                        [
                            [35, 149],
                            [36, 149],
                            [37, 149],
                            [38, 150],
                            [39, 150],
                            [40, 151],
                            [41, 151],
                            [42, 151],
                            [43, 151],
                            [44, 151],
                            [45, 151],
                            [46, 152],
                            [47, 152],
                            [48, 152],
                            [49, 153],
                            [50, 154],
                            [51, 154],
                            [52, 154],
                            [53, 155],
                            [54, 155],
                            [55, 156],
                            [56, 156],
                            [57, 156],
                            [58, 156],
                            [59, 156],
                            [60, 157],
                            [61, 157],
                            [62, 157],
                            [63, 157],
                            [64, 157],
                            [65, 157],
                            [66, 157],
                            [67, 158],
                            [68, 158],
                            [69, 158],
                            [70, 158],
                            [71, 158],
                            [72, 158],
                            [73, 159],
                            [74, 158],
                            [75, 159],
                            [76, 160],
                            [77, 160],
                            [78, 160],
                            [79, 161],
                            [80, 161],
                            [81, 161],
                            [82, 162],
                            [83, 163],
                            [84, 164],
                            [85, 165],
                            [85, 166],
                            [85, 167],
                            [86, 168],
                            [86, 169],
                            [86, 170],
                            [86, 171],
                            [87, 172],
                            [88, 173],
                            [88, 174],
                            [89, 175],
                            [89, 176],
                            [90, 177],
                            [91, 178],
                            [91, 179],
                            [91, 180],
                            [92, 181],
                            [92, 182],
                            [92, 183],
                            [92, 184],
                            [93, 185],
                            [93, 186],
                            [93, 187],
                            [93, 188],
                            [93, 189],
                        ]
                    ),
                    "heights": np.array(
                        [
                            2.66007735,
                            2.73889072,
                            2.72317582,
                            2.66718301,
                            2.65305027,
                            2.63008874,
                            2.64981724,
                            2.62444184,
                            2.5744252,
                            2.55552232,
                            2.52431496,
                            2.52440314,
                            2.4412618,
                            2.14124924,
                            1.93387523,
                            1.87504463,
                            1.99976331,
                            2.24085686,
                            2.33089594,
                            2.36080805,
                            2.30181414,
                            2.3691871,
                            2.51991619,
                            2.57115865,
                            2.65555971,
                            2.90791223,
                            3.0944814,
                            3.44446235,
                            3.61127617,
                            3.6980007,
                            3.7981108,
                            3.99926775,
                            4.10041132,
                            4.14083975,
                            4.00956674,
                            3.66364719,
                            3.19393572,
                            3.01011084,
                            2.87818221,
                            2.73445944,
                            2.47502752,
                            2.38338454,
                            2.41278875,
                            2.44899607,
                            2.51602884,
                            2.56498187,
                            2.55347311,
                            2.51797001,
                            2.50513275,
                            2.5018767,
                            2.50242615,
                            2.51859131,
                            2.5566577,
                            2.62273121,
                            2.63212958,
                            2.64895348,
                            2.67723726,
                            2.66198982,
                            2.65403313,
                            2.76641648,
                            2.80110234,
                            2.74804339,
                            2.55559296,
                            2.5130926,
                            2.48379064,
                            2.49812208,
                            2.63811433,
                            2.84474954,
                            2.99323735,
                        ]
                    ),
                    "distances": np.array(
                        [
                            -31.78049716,
                            -31.04834939,
                            -30.3644529,
                            -29.73213749,
                            -29.15475947,
                            -28.23118843,
                            -27.73084925,
                            -26.83281573,
                            -26.41968963,
                            -26.07680962,
                            -25.23885893,
                            -25.0,
                            -23.60084744,
                            -22.20360331,
                            -21.40093456,
                            -20.61552813,
                            -19.84943324,
                            -18.43908891,
                            -17.69180601,
                            -16.2788206,
                            -14.86606875,
                            -14.14213562,
                            -12.72792206,
                            -11.3137085,
                            -9.89949494,
                            -8.48528137,
                            -7.81024968,
                            -6.40312424,
                            -5.0,
                            -3.60555128,
                            -2.82842712,
                            -1.41421356,
                            0.0,
                            1.0,
                            2.23606798,
                            3.60555128,
                            5.0,
                            5.65685425,
                            6.40312424,
                            7.21110255,
                            8.60232527,
                            9.43398113,
                            10.29563014,
                            11.18033989,
                            12.08304597,
                            13.0,
                            13.92838828,
                            14.86606875,
                            15.8113883,
                            16.76305461,
                            17.72004515,
                            18.68154169,
                            19.6468827,
                            20.61552813,
                            21.58703314,
                            22.36067977,
                            23.34523506,
                            24.33105012,
                            25.3179778,
                            26.17250466,
                            27.07397274,
                            28.0713377,
                            29.01723626,
                            30.01666204,
                            31.0,
                            32.0,
                            33.0,
                            34.0,
                            35.0142828,
                        ]
                    ),
                    "fwhm": {
                        "fwhm": np.float64(12.366939358000284),
                        "half_maxs": [
                            np.float64(-7.385393414608109),
                            np.float64(4.981545943392174),
                            np.float64(3.2001518598774576),
                        ],
                        "peaks": [np.int64(33), np.float64(1.0), np.float64(4.140839746574575)],
                    },
                    "angles": np.float64(0.0),
                },
            },
            id="catenane",
            # marks=pytest.mark.skip(reason="disable whilst testing other params"),
        ),
        pytest.param(
            "rep_int_post_disordered_tracing",
            [0],
            {
                0: {  # Grain
                    1: {  # Node
                        "error": False,
                        "pixel_to_nm_scaling": 0.488,
                        "unmatched_branch_stats": {
                            0: {"angles": np.float64(0.0)},
                            1: {"angles": np.float64(94.21885263314698)},
                            2: {"angles": np.float64(66.84778128958743)},
                            3: {"angles": np.float64(142.2590882489817)},
                        },
                        "node_coords": np.array([[73, 159], [74, 158]]),
                        "confidence": np.float64(0.03871449534836169),
                    },
                    # @ns-rse 2025-10-22 : There are multiple nodes, excessive to check them all.
                },
            },
            {  # Expected MatchedBranch (only test the first Matched Branch)
                0: {
                    "ordered_coords": np.array(
                        [
                            [35, 149],
                            [36, 149],
                            [37, 149],
                            [38, 150],
                            [39, 150],
                            [40, 151],
                            [41, 151],
                            [42, 151],
                            [43, 151],
                            [44, 151],
                            [45, 151],
                            [46, 152],
                            [47, 152],
                            [48, 152],
                            [49, 153],
                            [50, 154],
                            [51, 154],
                            [52, 154],
                            [53, 155],
                            [54, 155],
                            [55, 156],
                            [56, 156],
                            [57, 156],
                            [58, 156],
                            [59, 156],
                            [60, 157],
                            [61, 157],
                            [62, 157],
                            [63, 157],
                            [64, 157],
                            [65, 157],
                            [66, 157],
                            [67, 158],
                            [68, 158],
                            [69, 158],
                            [70, 158],
                            [71, 158],
                            [72, 158],
                            [73, 159],
                            [74, 158],
                            [75, 159],
                            [76, 160],
                            [77, 160],
                            [78, 160],
                            [79, 161],
                            [80, 161],
                            [81, 161],
                            [82, 162],
                            [83, 163],
                            [84, 164],
                            [85, 165],
                            [85, 166],
                            [85, 167],
                            [86, 168],
                            [86, 169],
                            [86, 170],
                            [86, 171],
                            [87, 172],
                            [88, 173],
                            [88, 174],
                            [89, 175],
                            [89, 176],
                            [90, 177],
                            [91, 178],
                            [91, 179],
                            [91, 180],
                            [92, 181],
                            [92, 182],
                            [92, 183],
                            [92, 184],
                            [93, 185],
                            [93, 186],
                            [93, 187],
                            [93, 188],
                            [93, 189],
                        ]
                    ),
                    "heights": np.array(
                        [
                            2.13170185,
                            2.16828175,
                            2.18915553,
                            2.33270282,
                            2.35610101,
                            2.09070047,
                            2.13967484,
                            2.20562769,
                            2.27269809,
                            2.27767211,
                            2.19568794,
                            2.1207163,
                            2.11777145,
                            2.02342038,
                            2.15461443,
                            2.13955196,
                            2.22808776,
                            2.18022929,
                            2.10849704,
                            2.16432768,
                            2.14639965,
                            2.16124483,
                            2.11018236,
                            2.08464181,
                            2.10248132,
                            2.12527932,
                            2.10657728,
                            2.08488174,
                            2.14284489,
                            2.18671382,
                            2.14539571,
                            2.079853,
                            2.15980134,
                            2.27324934,
                            2.33982509,
                            2.43301785,
                            2.62186505,
                            2.99968733,
                            3.26873387,
                            3.75706697,
                            3.25026302,
                            2.27529441,
                            2.03195061,
                            1.93491741,
                            1.75556747,
                            1.86329297,
                            1.89655767,
                            2.13985724,
                            2.16505325,
                            2.14428989,
                            2.08145202,
                            2.19329704,
                            2.20784933,
                            2.20108216,
                            2.28153367,
                            2.27541704,
                            2.19782075,
                            2.11942912,
                            1.81888196,
                            1.87162372,
                            1.7598468,
                            1.84672653,
                            2.08694429,
                            2.14235931,
                            2.23785505,
                            2.15870862,
                            2.26121916,
                            2.31478999,
                            2.26939355,
                            2.15618801,
                            2.06309372,
                            2.03224247,
                            2.00724186,
                            1.97382715,
                            1.92037258,
                        ]
                    ),
                    "distances": np.array(
                        [
                            -40.02499219,
                            -39.05124838,
                            -38.07886553,
                            -36.87817783,
                            -35.90264614,
                            -34.71310992,
                            -33.73425559,
                            -32.75667871,
                            -31.78049716,
                            -30.8058436,
                            -29.83286778,
                            -28.63564213,
                            -27.65863337,
                            -26.68332813,
                            -25.49509757,
                            -24.33105012,
                            -23.34523506,
                            -22.36067977,
                            -21.21320344,
                            -20.22374842,
                            -19.10497317,
                            -18.11077028,
                            -17.11724277,
                            -16.1245155,
                            -15.13274595,
                            -14.03566885,
                            -13.03840481,
                            -12.04159458,
                            -11.04536102,
                            -10.04987562,
                            -9.05538514,
                            -8.06225775,
                            -7.0,
                            -6.0,
                            -5.0,
                            -4.0,
                            -3.0,
                            -2.0,
                            -1.41421356,
                            0.0,
                            1.41421356,
                            2.82842712,
                            3.60555128,
                            4.47213595,
                            5.83095189,
                            6.70820393,
                            7.61577311,
                            8.94427191,
                            10.29563014,
                            11.66190379,
                            13.03840481,
                            13.60147051,
                            14.2126704,
                            15.62049935,
                            16.2788206,
                            16.97056275,
                            17.69180601,
                            19.10497317,
                            20.51828453,
                            21.26029163,
                            22.6715681,
                            23.43074903,
                            24.8394847,
                            26.2488095,
                            27.01851217,
                            27.80287755,
                            29.20616373,
                            30.0,
                            30.8058436,
                            31.6227766,
                            33.01514804,
                            33.83784863,
                            34.66987165,
                            35.51056181,
                            36.35931793,
                        ]
                    ),
                    "fwhm": {
                        "fwhm": np.float64(4.470980292896586),
                        "half_maxs": [
                            np.float64(-2.447857504061359),
                            np.float64(2.0231227888352277),
                            np.float64(2.8304767888297717),
                        ],
                        "peaks": [np.int64(39), np.float64(0.0), np.float64(3.757066972693363)],
                    },
                    "angles": np.float64(0.0),
                },
            },
            id="rep int",
            marks=pytest.mark.skip(reason="disable whilst testing other params"),
        ),
    ],
)
def test_run_nodestats(  # noqa: C901
    topostats_object: str,
    detected_grains: list[int],
    expected_nodes: dict[str, Any],
    expected_matched_branch: dict[str, Any],
    tmp_path,
    request,
) -> None:
    """Test for run_nodestats()."""
    fixture_name = topostats_object
    topostats_object: TopoStats = request.getfixturevalue(topostats_object)
    run_nodestats(
        topostats_object=topostats_object,
        core_out_path=tmp_path,
        tracing_out_path=tmp_path,
    )
    for grain, grain_crop in topostats_object.grain_crops.above.crops.items():
        if grain in detected_grains:
            # We only check the first grain
            if grain_crop.nodes is not None and grain == 0:
                for node, nodestats in grain_crop.nodes.items():
                    # print(f"\n{nodestats=}\n")
                    if node == 1:
                        assert nodestats.error == expected_nodes[grain][node]["error"]
                        assert nodestats.unmatched_branch_stats == expected_nodes[grain][node]["unmatched_branch_stats"]
                        np.testing.assert_array_equal(nodestats.node_coords, expected_nodes[grain][node]["node_coords"])
                    # Check Matched branch statistics for node 0 of catenanes/rep_int node of catenanes and rep_int
                    # (only unmatched node stats in minicricle so nothing to check)
                    if fixture_name != "minicircle_small_post_disordered_tracing" and node == 0:
                        assert dict_almost_equal(
                            nodestats.branch_stats[0].matched_branch_to_dict(), expected_matched_branch[0]
                        )


# @pytest.mark.skip(reason="in development")
# @pytest.mark.parametrize(
#     ("topostats_object", "detected_grains", "expected_ordered_trace"),
#     [
#         pytest.param(
#             "minicircle_post_nodestats",
#             [0, 3, 4],
#             {
#                 0: {
#                     "ordered_trace": {}
#                 },  # Grain
#             },
#             id="minicircle",
#         ),
#         pytest.param(
#             "catenane_post_nodestats",
#             [0, 1],
#             {
#                 0: {
#                     "ordered_trace": {}
#                 },  # Grain
#             },
#             id="catenane",
#         ),
#         pytest.param(
#             "rep_int_post_nodestats",
#             [0, 1],
#             {
#                 0: {
#                     "ordered_trace": {}
#                 },  # Grain
#             },
#             id="catenane",
#         ),
#     ],
# )
# def test_run_ordered_tracing(
#     topostats_object: str,
#     detected_grains: list[int],
#     expected_ordered_trace: dict[str, Any],
#     sample,
#     process_scan_config: dict[str, Any],
#     default_config: dict[str, Any],
#     tmp_path,
#     request,
# ) -> None:
#     """Test for run_ordered_tracing."""
#     fixture_name = topostats_object
#     topostats_object = request.getfixturevalue(topostats_object)
#     run_ordered_tracing(
#         topostats_object=topostats_object,
#         core_out_path=tmp_path,
#         tracing_out_path=tmp_path,
#         ordered_tracing_config=process_scan_config["ordered_tracing"],
#         plotting_config=process_scan_config["plotting"],
#     )
#     # print(f"\n{topostats_object.image_grain_crops.__dict__()=}\n")
#     # print(f"\n{topostats_object.image_grain_crops.above.__dict__()=}\n")
#     for grain, grain_crop in topostats_object.image_grain_crops.above.crops.items():
#         assert grain_crop.ordered_trace is not None
