"""Test end-to-end running of topostats."""
from pathlib import Path

import filetype
import numpy as np
import pandas as pd
import pytest

from topostats.io import LoadScans
from topostats.processing import (
    check_run_steps,
    process_scan,
    run_filters,
    run_grains,
    run_grainstats,
    run_dnatracing,
)
from topostats.utils import update_plotting_config

BASE_DIR = Path.cwd()


# Can't see a way of paramterising with pytest-regtest as it writes to a file based on the file/function
# so instead we run three regression tests.
def test_process_scan_below(regtest, tmp_path, process_scan_config: dict, load_scan_data: LoadScans) -> None:
    """Regression test for checking the process_scan functions correctly."""
    # Ensure there are below grains
    process_scan_config["grains"]["threshold_std_dev"]["below"] = 0.8
    process_scan_config["grains"]["smallest_grain_size_nm2"] = 10
    process_scan_config["grains"]["absolute_area_threshold"]["below"] = [1, 1000000000]

    process_scan_config["grains"]["direction"] = "below"
    img_dic = load_scan_data.img_dict
    _, results, img_stats = process_scan(
        topostats_object=img_dic["minicircle_small"],
        base_dir=BASE_DIR,
        filter_config=process_scan_config["filter"],
        grains_config=process_scan_config["grains"],
        grainstats_config=process_scan_config["grainstats"],
        dnatracing_config=process_scan_config["dnatracing"],
        plotting_config=process_scan_config["plotting"],
        output_dir=tmp_path,
    )
    # Remove the basename column as this differs on CI
    results.drop(["basename"], axis=1, inplace=True)
    print(img_stats.to_string(float_format="{:.4e}".format), file=regtest)  # noqa: T201
    print(results.to_string(float_format="{:.4e}".format), file=regtest)  # noqa: T201


def test_process_scan_above(regtest, tmp_path, process_scan_config: dict, load_scan_data: LoadScans) -> None:
    """Regression test for checking the process_scan functions correctly."""
    # Ensure there are below grains
    process_scan_config["grains"]["smallest_grain_size_nm2"] = 10
    process_scan_config["grains"]["absolute_area_threshold"]["below"] = [1, 1000000000]

    img_dic = load_scan_data.img_dict
    _, results, img_stats = process_scan(
        topostats_object=img_dic["minicircle_small"],
        base_dir=BASE_DIR,
        filter_config=process_scan_config["filter"],
        grains_config=process_scan_config["grains"],
        grainstats_config=process_scan_config["grainstats"],
        dnatracing_config=process_scan_config["dnatracing"],
        plotting_config=process_scan_config["plotting"],
        output_dir=tmp_path,
    )
    # Remove the Basename column as this differs on CI
    results.drop(["basename"], axis=1, inplace=True)
    print(img_stats.to_string(float_format="{:.4e}".format), file=regtest)  # noqa: T201
    print(results.to_string(float_format="{:.4e}".format), file=regtest)  # noqa: T201


def test_process_scan_both(regtest, tmp_path, process_scan_config: dict, load_scan_data: LoadScans) -> None:
    """Regression test for checking the process_scan functions correctly."""
    # Ensure there are below grains
    process_scan_config["grains"]["threshold_std_dev"]["below"] = 0.8
    process_scan_config["grains"]["smallest_grain_size_nm2"] = 10
    process_scan_config["grains"]["absolute_area_threshold"]["below"] = [1, 1000000000]

    process_scan_config["grains"]["direction"] = "both"
    img_dic = load_scan_data.img_dict
    _, results, img_stats = process_scan(
        topostats_object=img_dic["minicircle_small"],
        base_dir=BASE_DIR,
        filter_config=process_scan_config["filter"],
        grains_config=process_scan_config["grains"],
        grainstats_config=process_scan_config["grainstats"],
        dnatracing_config=process_scan_config["dnatracing"],
        plotting_config=process_scan_config["plotting"],
        output_dir=tmp_path,
    )
    # Remove the Basename column as this differs on CI
    results.drop(["basename"], axis=1, inplace=True)
    print(img_stats.to_string(float_format="{:.4e}".format), file=regtest)  # noqa: T201
    print(results.to_string(float_format="{:.4e}".format), file=regtest)  # noqa: T201


@pytest.mark.parametrize(
    ("image_set", "expected"),
    [
        ("core", False),
        ("all", True),
    ],
)
def test_save_cropped_grains(
    tmp_path: Path, process_scan_config: dict, load_scan_data: LoadScans, image_set, expected
) -> None:
    """Tests if cropped grains are saved only when image set is 'all' rather than 'core'."""
    process_scan_config["plotting"]["image_set"] = image_set
    process_scan_config["plotting"] = update_plotting_config(process_scan_config["plotting"])
    process_scan_config["plotting"]["dpi"] = 50

    img_dic = load_scan_data.img_dict
    _, _, _ = process_scan(
        topostats_object=img_dic["minicircle_small"],
        base_dir=BASE_DIR,
        filter_config=process_scan_config["filter"],
        grains_config=process_scan_config["grains"],
        grainstats_config=process_scan_config["grainstats"],
        dnatracing_config=process_scan_config["dnatracing"],
        plotting_config=process_scan_config["plotting"],
        output_dir=tmp_path,
    )

    assert (
        Path.exists(
            tmp_path
            / "tests/resources/test_image/processed/minicircle_small/grains/above"
            / "minicircle_small_grain_image_0.png"
        )
        == expected
    )
    assert (
        Path.exists(
            tmp_path
            / "tests/resources/test_image/processed/minicircle_small/grains/above"
            / "minicircle_small_grain_mask_0.png"
        )
        == expected
    )
    assert (
        Path.exists(
            tmp_path
            / "tests/resources/test_image/processed/minicircle_small/grains/above"
            / "minicircle_small_grain_mask_image_0.png"
        )
        == expected
    )


@pytest.mark.parametrize("extension", [("png"), ("tif")])
def test_save_format(process_scan_config: dict, load_scan_data: LoadScans, tmp_path: Path, extension: str):
    """Tests if save format applied to cropped images."""
    process_scan_config["plotting"]["image_set"] = "all"
    process_scan_config["plotting"]["save_format"] = extension
    process_scan_config["plotting"] = update_plotting_config(process_scan_config["plotting"])

    img_dic = load_scan_data.img_dict
    _, _, _ = process_scan(
        topostats_object=img_dic["minicircle_small"],
        base_dir=BASE_DIR,
        filter_config=process_scan_config["filter"],
        grains_config=process_scan_config["grains"],
        grainstats_config=process_scan_config["grainstats"],
        dnatracing_config=process_scan_config["dnatracing"],
        plotting_config=process_scan_config["plotting"],
        output_dir=tmp_path,
    )

    guess = filetype.guess(
        tmp_path
        / "tests/resources/test_image/processed/minicircle_small/grains/above"
        / f"minicircle_small_grain_image_0.{extension}"
    )
    assert guess.extension == extension


@pytest.mark.parametrize(
    ("filter_run", "grains_run", "grainstats_run", "dnatracing_run", "log_msg"),
    [
        (
            False,
            False,
            False,
            True,
            "DNA tracing enabled but Grainstats disabled. Please check your configuration file.",
        ),
        (
            False,
            False,
            True,
            True,
            "DNA tracing enabled but Grains disabled. Please check your configuration file.",
        ),
        (
            False,
            True,
            True,
            True,
            "DNA tracing enabled but Filters disabled. Please check your configuration file.",
        ),
        (
            False,
            False,
            True,
            False,
            "Grainstats enabled but Grains disabled. Please check your configuration file.",
        ),
        (
            False,
            True,
            True,
            False,
            "Grainstats enabled but Filters disabled. Please check your configuration file.",
        ),
        (
            False,
            True,
            False,
            False,
            "Grains enabled but Filters disabled. Please check your configuration file.",
        ),
        (
            True,
            False,
            False,
            False,
            "Configuration run options are consistent, processing can proceed.",
        ),
        (
            True,
            True,
            False,
            False,
            "Configuration run options are consistent, processing can proceed.",
        ),
        (
            True,
            True,
            True,
            False,
            "Configuration run options are consistent, processing can proceed.",
        ),
        (
            True,
            True,
            True,
            True,
            "Configuration run options are consistent, processing can proceed.",
        ),
    ],
)
def test_check_run_steps(
    filter_run: bool,
    grains_run: bool,
    grainstats_run: bool,
    dnatracing_run: bool,
    log_msg: str,
    caplog,
) -> None:
    """Test the logic which checks whether enabled processing options are consistent."""
    check_run_steps(filter_run, grains_run, grainstats_run, dnatracing_run)
    assert log_msg in caplog.text


# noqa: disable=too-many-arguments
# pylint: disable=too-many-arguments
@pytest.mark.parametrize(
    ("filter_run", "grains_run", "grainstats_run", "dnatracing_run", "log_msg1", "log_msg2"),
    [
        (
            False,
            False,
            False,
            False,
            "You have not included running the initial filter stage.",
            "Please check your configuration file.",
        ),
        (
            True,
            False,
            False,
            False,
            "Detection of grains disabled, returning empty data frame.",
            "15-gaussian_filtered",
        ),
        (
            True,
            True,
            False,
            False,
            "Calculation of grainstats disabled, returning empty dataframe.",
            "24-labelled_image_bboxes",
        ),
        (
            True,
            True,
            True,
            False,
            "Processing grain",
            "Calculation of DNA Tracing disabled, returning grainstats data frame.",
        ),
        (
            True,
            True,
            True,
            True,
            "Traced grain 3 of 3",
            "Combining ['above'] grain statistics and dnatracing statistics",
        ),
    ],
)
def test_process_stages(
    process_scan_config: dict,
    load_scan_data: LoadScans,
    tmp_path: Path,
    filter_run: bool,
    grains_run: bool,
    grainstats_run: bool,
    dnatracing_run: bool,
    log_msg1: str,
    log_msg2: str,
    caplog,
) -> None:
    """Regression test for checking the process_scan functions correctly when specific sections are disabled.

    Currently there is no test for having later stages (e.g. DNA Tracing or Grainstats) enabled when Filters and/or
    Grainstats are disabled. Whislt possible it is expected that users understand the need to run earlier stages before
    later staged can run and do not disable earlier stages.
    """
    img_dic = load_scan_data.img_dict
    process_scan_config["filter"]["run"] = filter_run
    process_scan_config["grains"]["run"] = grains_run
    process_scan_config["grainstats"]["run"] = grainstats_run
    process_scan_config["dnatracing"]["run"] = dnatracing_run
    _, _, _ = process_scan(
        topostats_object=img_dic["minicircle_small"],
        base_dir=BASE_DIR,
        filter_config=process_scan_config["filter"],
        grains_config=process_scan_config["grains"],
        grainstats_config=process_scan_config["grainstats"],
        dnatracing_config=process_scan_config["dnatracing"],
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
    _, _, _ = process_scan(
        topostats_object=img_dic["minicircle_small"],
        base_dir=BASE_DIR,
        filter_config=process_scan_config["filter"],
        grains_config=process_scan_config["grains"],
        grainstats_config=process_scan_config["grainstats"],
        dnatracing_config=process_scan_config["dnatracing"],
        plotting_config=process_scan_config["plotting"],
        output_dir=tmp_path,
    )
    assert "Grains found for direction above : 0" in caplog.text
    assert "No grains exist for the above direction. Skipping grainstats for above." in caplog.text


def test_process_scan_align_grainstats_dnatracing(
    process_scan_config: dict, load_scan_data: LoadScans, tmp_path: Path
) -> None:
    """Ensure molecule numbers from dnatracing align with those from grainstats.

    Sometimes grains are removed from tracing due to small size, however we need to ensure that tracing statistics for
    those molecules that remain align with grain statistics.

    By setting processing parameters as below two molecules are purged for being too small after skeletonisation and so
    do not have DNA tracing statistics (but they do have Grain Statistics).
    """
    img_dic = load_scan_data.img_dict
    process_scan_config["filter"]["remove_scars"]["run"] = False
    process_scan_config["grains"]["absolute_area_threshold"]["above"] = [150, 3000]
    process_scan_config["dnatracing"]["min_skeleton_size"] = 50
    _, results, _ = process_scan(
        topostats_object=img_dic["minicircle_small"],
        base_dir=BASE_DIR,
        filter_config=process_scan_config["filter"],
        grains_config=process_scan_config["grains"],
        grainstats_config=process_scan_config["grainstats"],
        dnatracing_config=process_scan_config["dnatracing"],
        plotting_config=process_scan_config["plotting"],
        output_dir=tmp_path,
    )
    tracing_to_check = ["contour_length", "circular", "end_to_end_distance"]

    assert results.shape == (3, 25)
    assert np.isnan(results.loc[2, "contour_length"])
    assert np.isnan(sum(results.loc[2, tracing_to_check]))


def test_run_filters(process_scan_config: dict, load_scan_data: LoadScans, tmp_path: Path) -> None:
    """Test the filter_wrapper function of processing.py."""
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
    """Test the grains_wrapper function of processing.py."""
    flattened_image = np.load("./tests/resources/minicircle_cropped_flattened.npy")

    grains_config = process_scan_config["grains"]
    grains_config["threshold_method"] = "absolute"
    grains_config["direction"] = "both"
    grains_config["threshold_absolute"]["above"] = 1.0
    grains_config["threshold_absolute"]["below"] = -0.4
    grains_config["smallest_grain_size_nm2"] = 20
    grains_config["absolute_area_threshold"]["above"] = [20, 10000000]

    grains = run_grains(
        image=flattened_image,
        pixel_to_nm_scaling=0.4940029296875,
        filename="dummy filename",
        grain_out_path=tmp_path,
        core_out_path=tmp_path,
        grains_config=grains_config,
        plotting_config=process_scan_config["plotting"],
    )

    assert isinstance(grains, dict)
    assert list(grains.keys()) == ["above", "below"]
    assert isinstance(grains["above"], np.ndarray)
    assert np.max(grains["above"]) == 6
    # Floating point errors mean that on different systems, different results are
    # produced for such generous thresholds. This is not an issue for more stringent
    # thresholds.
    assert np.max(grains["below"]) > 0
    assert np.max(grains["above"]) < 10


def test_run_grainstats(process_scan_config: dict, tmp_path: Path) -> None:
    """Test the grainstats_wrapper function of processing.py."""
    flattened_image = np.load("./tests/resources/minicircle_cropped_flattened.npy")
    mask_above = np.load("./tests/resources/minicircle_cropped_masks_above.npy")
    mask_below = np.load("./tests/resources/minicircle_cropped_masks_below.npy")
    grain_masks = {"above": mask_above, "below": mask_below}

    grainstats_df = run_grainstats(
        image=flattened_image,
        pixel_to_nm_scaling=0.4940029296875,
        grain_masks=grain_masks,
        filename="dummy filename",
        grainstats_config=process_scan_config["grainstats"],
        plotting_config=process_scan_config["plotting"],
        grain_out_path=tmp_path,
    )

    assert isinstance(grainstats_df, pd.DataFrame)
    assert grainstats_df.shape[0] == 13
    assert len(grainstats_df.columns) == 21


def test_run_dnatracing(process_scan_config: dict, tmp_path: Path) -> None:
    """Test the dnatracing_wrapper function of processing.py."""
    flattened_image = np.load("./tests/resources/minicircle_cropped_flattened.npy")
    mask_above = np.load("./tests/resources/minicircle_cropped_masks_above.npy")
    mask_below = np.load("./tests/resources/minicircle_cropped_masks_below.npy")
    grain_masks = {"above": mask_above, "below": mask_below}

    dnatracing_df = run_dnatracing(
        image=flattened_image,
        grain_masks=grain_masks,
        pixel_to_nm_scaling=0.4940029296875,
        image_path=tmp_path,
        filename="dummy filename",
        core_out_path=tmp_path,
        grain_out_path=tmp_path,
        dnatracing_config=process_scan_config["dnatracing"],
        plotting_config=process_scan_config["plotting"],
        results_df=pd.read_csv("./tests/resources/minicircle_cropped_grainstats.csv"),
    )

    assert isinstance(dnatracing_df, pd.DataFrame)
    assert dnatracing_df.shape[0] == 13
    assert len(dnatracing_df.columns) == 23
