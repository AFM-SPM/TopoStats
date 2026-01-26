# Disable ruff 301 - pickle loading is unsafe, but we don't care for tests.
# ruff: noqa: S301
"""Test end-to-end running of topostats."""

import logging
from pathlib import Path
from typing import Any

import filetype
import numpy as np
import numpy.typing as npt
import pytest
from AFMReader.topostats import load_topostats
from syrupy.matchers import path_type

from topostats.classes import DisorderedTrace, GrainCrop, TopoStats
from topostats.config import update_plotting_config
from topostats.io import LoadScans
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
# pylint: disable=too-many-locals
# pylint: disable=too-many-positional-arguments


def test_process_scan(tmp_path, process_scan_config: dict, load_scan_data: LoadScans, snapshot) -> None:
    """Regression test for checking the process_scan functions correctly."""
    img_dic = load_scan_data.img_dict
    _, grain_stats, _, img_stats, _, _, _ = process_scan(
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
    assert img_stats.to_string(float_format="{:.4e}".format) == snapshot
    # Have to drop 'basename' from dataframe as this varies between local machines and on CI
    grain_stats.drop(["basename"], axis=1, inplace=True)
    assert grain_stats.to_string(float_format="{:.4e}".format) == snapshot

    # Regtest for the topostats file
    assert Path.exists(tmp_path / "tests/resources/test_image/processed/minicircle_small.topostats")
    # Load the results, note that we use AFMReader.topostats.load_topostats() here which simply loads the data as
    # dictionaries and means it is easy to compare to syrupy snapshots
    saved_topostats = load_topostats(
        file_path=tmp_path / "tests/resources/test_image/processed/minicircle_small.topostats"
    )
    # Drop the config, img_path and topostats_version from top level of dictionary and and basename from
    # disorded_trace.stats_dict) as we don't want to compare configuration nor test absolute paths.
    saved_topostats.pop("config")
    saved_topostats.pop("img_path")
    saved_topostats.pop("topostats_version")
    # Precision issues on OSX and M$-Win mean we need to break out some variables and test with different precision
    disordered_stats = {}
    unmatched_branch_stats = {}
    grain_stats = {}
    volume_stats = {}
    for grain_number, grain_crop in saved_topostats["grain_crops"].items():
        disordered_stats[grain_number] = grain_crop["disordered_trace"].pop("stats_dict")
        grain_stats[grain_number] = grain_crop.pop("stats")
        volume_stats[grain_number] = {}
        for key1, data in grain_stats[grain_number].items():
            volume_stats[grain_number][key1] = {}
            for key2, stats in data.items():
                volume_stats[grain_number][key1][key2] = stats.pop("volume")
        for _, stats in disordered_stats[grain_number].items():
            stats.pop("basename")
        unmatched_branch_stats[grain_number] = {}
        if "nodes" in list(grain_crop.keys()):
            for node_number, node in grain_crop["nodes"].items():
                unmatched_branch_stats[grain_number][node_number] = node.pop("unmatched_branch_stats")
    rms_roughness = saved_topostats["image_statistics"].pop("rms_roughness")
    # There is one(!!!) curvature statistic difference cropping up on OSX (both Python 3.10 and 3.11)
    #
    # FAILED tests/test_processing.py::test_process_scan - assert [+ received] == [- snapshot]
    #     ......
    #        ...
    #            6.73909078e-01, 2.59792188e-14, 5.77315973e-15, 8.57926578e-01,
    #   -        8.21565038e-15, 1.06581410e-14, 6.21041537e-01, 1.15463195e-14,
    #   +        8.43769499e-15, 1.04360964e-14, 6.21041537e-01, 1.15463195e-14,
    #            1.15463195e-14, 3.04556842e-01, 7.54951657e-15, 7.54951657e-15,
    #        ...
    #     ......
    # = 1 failed, 965 passed
    #
    # We set this to None
    saved_topostats["grain_crops"]["0"]["ordered_trace"]["molecule_data"]["0"]["curvature_stats"] = None
    assert saved_topostats == snapshot
    assert disordered_stats == snapshot(matcher=path_type(types=(float,), replacer=lambda data, _: round(data, 12)))
    assert unmatched_branch_stats == snapshot(
        matcher=path_type(types=(float,), replacer=lambda data, _: round(data, 12))
    )
    assert grain_stats == snapshot(matcher=path_type(types=(float,), replacer=lambda data, _: round(data, 16)))
    assert volume_stats == snapshot(matcher=path_type(types=(float,), replacer=lambda data, _: round(data, 28)))
    assert rms_roughness == snapshot(matcher=path_type(types=(float,), replacer=lambda data, _: round(data, 20)))


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
    _, _, _, _, _, _, _ = process_scan(
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
            tmp_path / "tests/resources/test_image/processed/minicircle_small/grains/" / "minicircle_small_grain_0.png"
        )
        == expected
    )
    assert (
        Path.exists(
            tmp_path
            / "tests/resources/test_image/processed/minicircle_small/grains/"
            / "minicircle_small_grain_mask_0_class_1.png"
        )
        == expected
    )


# ns-rse 2025-12-17 It seems rather excessive to have so many options for which subset of images to output, would be
# much simpler if we just provided 'core' or 'all'
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
        "core": "minicircle_small_all_splines.png",
        "filters": "minicircle_small/filters/01-pixels.png",
        "grains": "minicircle_small/grains/24-area_thresholded_class_1.png",
        "grain_crop": "minicircle_small/grains/minicircle_small_grain_0.png",
        "disordered_tracing": "minicircle_small/dnatracing/disordered/22-original_skeletons.png",
        "nodestats": "minicircle_small/dnatracing/nodes/26-node_centres.png",
        "ordered_tracing": "minicircle_small/dnatracing/ordered/28-molecule_crossings.png",
        "splining": "minicircle_small/dnatracing/curvature/0_curvature.png",
    }
    for key, img_path in images.items():
        # Leaving in for debugging so we quickly know what fails
        print(f"\n{key=} : {img_path=}\n")
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
    _, _, _, _, _, _, _ = process_scan(
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
        / "tests/resources/test_image/processed/minicircle_small/grains/"
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
            "Your configuration disables running the initial filter stage.",
            "Please correct your configuration file.",
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
            "Calculation of grainstats disabled.",
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
            "Disordered Tracing disabled.",
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
            "Calculation of curvature statistics disabled.",
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
    topostats_object = load_scan_data.img_dict["minicircle_small"]
    # Remove existing data so its calculated anew
    topostats_object.grain_crops = None
    process_scan_config["filter"]["run"] = filter_run
    process_scan_config["grains"]["run"] = grains_run
    process_scan_config["grainstats"]["run"] = grainstats_run
    process_scan_config["disordered_tracing"]["run"] = disordered_tracing_run
    process_scan_config["nodestats"]["run"] = nodestats_run
    process_scan_config["ordered_tracing"]["run"] = ordered_tracing_run
    process_scan_config["splining"]["run"] = splining_run
    process_scan_config["curvature"]["run"] = curvature_run
    _, _, _, _, _, _, _ = process_scan(
        topostats_object=topostats_object,
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
    topostats_object = load_scan_data.img_dict["minicircle_small"]
    topostats_object.grain_crops = None
    process_scan_config["grains"]["threshold_std_dev"]["above"] = 1000
    process_scan_config["filter"]["remove_scars"]["run"] = False
    _, _, _, _, _, _, _ = process_scan(
        topostats_object=topostats_object,
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
    assert "Grains found 0" in caplog.text
    assert "No grains found, skipping grainstats and tracing stages." in caplog.text


def test_run_filters(minicircle_small_topostats: TopoStats, tmp_path: Path) -> None:
    """Test the filter wrapper function of processing.py."""
    run_filters(
        topostats_object=minicircle_small_topostats,
        filter_out_path=tmp_path,
        core_out_path=tmp_path,
    )
    assert isinstance(minicircle_small_topostats.image, np.ndarray)
    assert minicircle_small_topostats.image.shape == (64, 64)
    assert np.sum(minicircle_small_topostats.image) == pytest.approx(1172.6088236592373)


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


def test_run_grainstats(post_processing_minicircle_topostats_object: TopoStats, tmp_path: Path, snapshot) -> None:
    """Test the grainstats_wrapper function of processing.py."""
    # Remove results that are not required
    for _, grain_crop in post_processing_minicircle_topostats_object.grain_crops.items():
        grain_crop.convolved_skeleton = None
        grain_crop.disordered_trace = None
        grain_crop.height_profiles = None
        grain_crop.nodes = None
        grain_crop.ordered_trace = None
        # Importantly for this test reset the `stats` attribute
        grain_crop.stats = None
    run_grainstats(
        topostats_object=post_processing_minicircle_topostats_object,
        core_out_path=tmp_path,
        grain_out_path=tmp_path,
    )
    assert isinstance(post_processing_minicircle_topostats_object.grain_crops, dict)
    assert len(post_processing_minicircle_topostats_object.grain_crops) == 21
    # Build a dictionary of grain_crop.stats to compare to a snapshot
    grain_crop_statistics = {
        grain_number: grain_crop.stats
        for grain_number, grain_crop in post_processing_minicircle_topostats_object.grain_crops.items()
    }
    # Because of precision issues across OS's we split out the volume and area values to be tested with different
    # precision.
    grain_crop_areas = {}
    grain_crop_volume = {}
    grain_crop_aspect_ratio = {}
    for grain_number, grain_crop in grain_crop_statistics.items():
        grain_crop_areas[grain_number] = {
            "area": grain_crop[1][0].pop("area"),
            "area_cartesian_bbox": grain_crop[1][0].pop("area_cartesian_bbox"),
            "smallest_bounding_area": grain_crop[1][0].pop("smallest_bounding_area"),
        }
        grain_crop_volume[grain_number] = grain_crop[1][0].pop("volume")
        grain_crop_aspect_ratio[grain_number] = grain_crop[1][0].pop("aspect_ratio")
    assert grain_crop_statistics == snapshot(
        matcher=path_type(types=(float,), replacer=lambda data, _: round(data, 12))
    )
    assert grain_crop_areas == snapshot(matcher=path_type(types=(float,), replacer=lambda data, _: round(data, 24)))
    assert grain_crop_volume == snapshot(matcher=path_type(types=(float,), replacer=lambda data, _: round(data, 32)))
    assert grain_crop_aspect_ratio == snapshot(
        matcher=path_type(types=(float,), replacer=lambda data, _: round(data, 12))
    )


@pytest.mark.parametrize(
    ("topostats_object", "detected_grains", "log_messages", "expected"),
    [
        pytest.param(
            "minicircle_small_post_grainstats",
            [0, 3, 4],
            [
                "Disordered Tracing stage completed successfully.",
                "Disordered trace plotting completed successfully.",
            ],
            {
                0: {
                    "grain_endpoints": 0,
                    "grain_junctions": 2,
                    "total_branch_length": 8.818429896553074e-08,
                    "grain_width_mean": 7.793805389705788e-09,
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
                    "grain_junctions": 12,
                    "total_branch_length": 5.743815274574378e-07,
                    "grain_width_mean": 4.21578115152422e-09,
                },
                # @ns-rse 2025-10-22 : Its strange that the values for these differ because the two grains in
                #                      catenanes are in essence identical being a mirror of each other.
                1: {
                    "grain_endpoints": 0,
                    "grain_junctions": 14,
                    "total_branch_length": 5.730849825836855e-07,
                    "grain_width_mean": 4.2413316040868905e-09,
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
    for grain, grain_crop in topostats_object.grain_crops.items():
        if grain in detected_grains:
            assert grain_crop.disordered_trace is not None
            assert isinstance(grain_crop.disordered_trace, DisorderedTrace)
            assert isinstance(grain_crop.disordered_trace.images, dict)
            assert grain_crop.disordered_trace.grain_endpoints == expected[grain]["grain_endpoints"]
            assert grain_crop.disordered_trace.grain_junctions == expected[grain]["grain_junctions"]
            assert grain_crop.disordered_trace.total_branch_length == expected[grain]["total_branch_length"]
            assert grain_crop.disordered_trace.grain_width_mean == expected[grain]["grain_width_mean"]


@pytest.mark.parametrize(
    ("topostats_object", "detected_grains", "node_coords"),
    [
        pytest.param(
            "minicircle_small_post_disordered_tracing",
            [0, 3, 4],
            np.array([[50, 37], [51, 38], [52, 38], [52, 39]]),
            id="minicircle small",
        ),
        pytest.param(
            "catenanes_post_disordered_tracing",
            [0, 1],
            np.array(
                [
                    [41, 151],
                    [41, 152],
                    [42, 153],
                    [43, 154],
                    [43, 155],
                    [43, 156],
                    [43, 157],
                    [43, 158],
                    [43, 159],
                    [43, 160],
                    [43, 161],
                    [43, 162],
                    [43, 163],
                    [43, 164],
                ]
            ),
            id="catenane",
        ),
        pytest.param(
            "rep_int_post_disordered_tracing",
            [0],
            np.array([[110, 200], [111, 201], [112, 201], [112, 202]]),
            id="rep int",
        ),
    ],
)
def test_run_nodestats(  # noqa: C901
    topostats_object: str,
    detected_grains: list[int],
    node_coords: npt.NDArray,
    tmp_path,
    request,
    snapshot,
) -> None:
    """Test for run_nodestats()."""
    fixture_name = topostats_object
    topostats_object: TopoStats = request.getfixturevalue(topostats_object)
    run_nodestats(
        topostats_object=topostats_object,
        core_out_path=tmp_path,
        tracing_out_path=tmp_path,
    )
    for grain, grain_crop in topostats_object.grain_crops.items():
        if grain in detected_grains:
            # We only check the first grain
            if grain_crop.nodes is not None and grain == 0:
                for node, nodestats in grain_crop.nodes.items():
                    if node == 1:
                        assert nodestats.error == snapshot
                        assert nodestats.unmatched_branch_stats == snapshot
                        np.testing.assert_array_equal(nodestats.node_coords, node_coords)
                    # Check Matched branch statistics for node 0 of catenanes/rep_int node of catenanes and rep_int
                    # (only unmatched node stats in minicricle so nothing to check)
                    if fixture_name != "minicircle_small_post_disordered_tracing" and node == 0:
                        assert nodestats.branch_stats[0] == snapshot


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
#     for grain, grain_crop in topostats_object.image_grain_crops.above.crops.items():
#         assert grain_crop.ordered_trace is not None
