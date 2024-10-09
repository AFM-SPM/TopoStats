"""Test end-to-end running of topostats."""

import pickle
from pathlib import Path

import filetype
import h5py
import numpy as np
import pandas as pd
import pytest
from test_io import dict_almost_equal

from topostats.io import LoadScans, hdf5_to_dict
from topostats.processing import (
    check_run_steps,
    process_scan,
    run_filters,
    run_grains,
    run_grainstats,
)
from topostats.utils import update_plotting_config

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests/resources"


# Can't see a way of parameterising with pytest-regtest as it writes to a file based on the file/function
# so instead we run three regression tests.
def test_process_scan_below(regtest, tmp_path, process_scan_config: dict, load_scan_data: LoadScans) -> None:
    """Regression test for checking the process_scan functions correctly."""
    # Ensure there are below grains
    process_scan_config["grains"]["threshold_std_dev"]["below"] = 0.8
    process_scan_config["grains"]["smallest_grain_size_nm2"] = 10
    process_scan_config["grains"]["absolute_area_threshold"]["below"] = [1, 1000000000]
    process_scan_config["grains"]["direction"] = "below"
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
    process_scan_config["grains"]["smallest_grain_size_nm2"] = 10
    process_scan_config["grains"]["absolute_area_threshold"]["below"] = [1, 1000000000]

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
        plotting_config=process_scan_config["plotting"],
        output_dir=tmp_path,
    )

    # Save height profiles dictionary to pickle
    # with open(RESOURCES / "process_scan_expected_below_height_profiles.pickle", "wb") as f:
    #     pickle.dump(height_profiles, f)

    # Load expected height profiles dictionary from pickle
    # pylint wants an encoding but binary mode doesn't use one
    # pylint: disable=unspecified-encoding
    with Path.open(RESOURCES / "process_scan_expected_below_height_profiles.pickle", "rb") as f:
        expected_height_profiles = pickle.load(f)  # noqa: S301 - Pickles are unsafe but we don't care

    assert dict_almost_equal(height_profiles, expected_height_profiles, abs_tol=1e-11)


def test_process_scan_above(regtest, tmp_path, process_scan_config: dict, load_scan_data: LoadScans) -> None:
    """Regression test for checking the process_scan functions correctly."""
    # Ensure there are below grains
    process_scan_config["grains"]["smallest_grain_size_nm2"] = 10
    process_scan_config["grains"]["absolute_area_threshold"]["below"] = [1, 1000000000]

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
    process_scan_config["grains"]["smallest_grain_size_nm2"] = 10
    process_scan_config["grains"]["absolute_area_threshold"]["below"] = [1, 1000000000]

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
        plotting_config=process_scan_config["plotting"],
        output_dir=tmp_path,
    )

    # Save height profiles dictionary to pickle
    # with open(RESOURCES / "process_scan_expected_above_height_profiles.pickle", "wb") as f:
    #     pickle.dump(height_profiles, f)

    # Load expected height profiles dictionary from pickle
    # pylint wants an encoding but binary mode doesn't use one
    # pylint: disable=unspecified-encoding
    with Path.open(RESOURCES / "process_scan_expected_above_height_profiles.pickle", "rb") as f:
        expected_height_profiles = pickle.load(f)  # noqa: S301 - Pickles are unsafe but we don't care

    assert dict_almost_equal(height_profiles, expected_height_profiles, abs_tol=1e-11)


def test_process_scan_both(regtest, tmp_path, process_scan_config: dict, load_scan_data: LoadScans) -> None:
    """Regression test for checking the process_scan functions correctly."""
    # Ensure there are below grains
    process_scan_config["grains"]["threshold_std_dev"]["below"] = 0.8
    process_scan_config["grains"]["smallest_grain_size_nm2"] = 10
    process_scan_config["grains"]["absolute_area_threshold"]["below"] = [1, 1000000000]

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
    # Check the data
    assert dict_almost_equal(expected_topostats, saved_topostats)


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
        plotting_config=process_scan_config["plotting"],
        output_dir=tmp_path,
    )

    guess = filetype.guess(
        tmp_path
        / "tests/resources/test_image/processed/minicircle_small/grains/above"
        / f"minicircle_small_grain_image_0.{extension}"
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
            "Splining enabled but NodeStats disabled. Tracing will use the 'old' method.",
            id="Splining, Disordered Tracing, Grainstats, Grains and Filters no Nodestats",
        ),
        pytest.param(
            True,
            True,
            False,
            False,
            False,
            True,
            "Splining enabled but Grainstats disabled. Please check your configuration file.",
            id="Splining, Grains and Filters enabled but no Grainstats, Disordered Tracing or NodeStats",
        ),
        pytest.param(
            True,
            False,
            False,
            False,
            False,
            True,
            "Splining enabled but Grainstats disabled. Please check your configuration file.",
            id="Splining and Filters enabled but no NodeStats, Disordered Tracing or Grainstats",
        ),
        pytest.param(
            False,
            False,
            True,
            False,
            False,
            True,
            "Splining enabled but Grains disabled. Please check your configuration file.",
            id="Splining and Grainstats enabled but no Grains or Filters",
        ),
        pytest.param(
            False,
            False,
            False,
            False,
            True,
            False,
            "NodeStats enabled but Disordered Tracing disabled. Please check your configuration file.",
            id="Nodestats enabled but no Disordered Tracing, Grainstats, Grains or Filters",
        ),
        pytest.param(
            False,
            False,
            False,
            True,
            True,
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
    splining_run: bool,
    log_msg: str,
    caplog,
) -> None:
    """Test the logic which checks whether enabled processing options are consistent."""
    check_run_steps(filter_run, grains_run, grainstats_run, disordered_tracing_run, nodestats_run, splining_run)
    assert log_msg in caplog.text


# pylint: disable=too-many-arguments
@pytest.mark.parametrize(
    ("filter_run", "grains_run", "grainstats_run", "dnatracing_run", "log_msg1", "log_msg2"),
    [
        pytest.param(
            False,
            False,
            False,
            False,
            "You have not included running the initial filter stage.",
            "Please check your configuration file.",
            id="All stages are disabled",
        ),
        pytest.param(
            True,
            False,
            False,
            False,
            "Detection of grains disabled, GrainStats will not be run.",
            "",
            id="Only filtering enabled",
        ),
        pytest.param(
            True,
            True,
            False,
            False,
            "Calculation of grainstats disabled, returning empty dataframe and empty height_profiles.",
            "",
            id="Filtering and Grain enabled",
        ),
        pytest.param(
            True,
            True,
            True,
            False,
            "Processing grain",
            "Calculation of Disordered Tracing disabled, returning empty dictionary.",
            id="Filtering, Grain and GrainStats enabled",
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
    dnatracing_run: bool,
    log_msg1: str,
    log_msg2: str,
    caplog,
) -> None:
    """Regression test for checking the process_scan functions correctly when specific sections are disabled.

    Currently there is no test for having later stages (e.g. DNA Tracing or Grainstats) enabled when Filters and/or
    Grainstats are disabled. Whislt possible it is expected that users understand the need to run earlier stages before
    later stages can run and do not disable earlier stages.
    """
    img_dic = load_scan_data.img_dict
    process_scan_config["filter"]["run"] = filter_run
    process_scan_config["grains"]["run"] = grains_run
    process_scan_config["grainstats"]["run"] = grainstats_run
    process_scan_config["disordered_tracing"]["run"] = dnatracing_run
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
        plotting_config=process_scan_config["plotting"],
        output_dir=tmp_path,
    )
    assert "Grains found for direction above : 0" in caplog.text
    assert "No grains exist for the above direction. Skipping grainstats for above." in caplog.text


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
    mask_above_dna = np.load("./tests/resources/minicircle_cropped_masks_above.npy")
    mask_above_background = np.load("./tests/resources/minicircle_cropped_masks_above_background.npy")

    mask_above = np.stack([mask_above_background, mask_above_dna], axis=-1)
    assert mask_above.shape == (256, 256, 2)

    # Create inverted mask above
    # mask_above_background = np.logical_not(mask_above)
    # # Keep only the largest grain
    # from skimage.measure import label, regionprops

    # mask_above_labelled = label(mask_above)
    # regions = regionprops(mask_above_labelled)
    # areas = [region.area for region in regions]
    # mask_above[mask_above_labelled != np.argmax(areas) + 1] = False
    # # save the mask above
    # np.save("./tests/resources/minicircle_cropped_masks_above_background.npy", mask_above_background)

    mask_below_dna = np.load("./tests/resources/minicircle_cropped_masks_below.npy")
    mask_below_background = np.load("./tests/resources/minicircle_cropped_masks_below_background.npy")

    mask_below = np.stack([mask_below_background, mask_below_dna], axis=-1)
    assert mask_below.shape == (256, 256, 2)

    # Create inverted mask below
    # mask_below_background = np.logical_not(mask_below)
    # # Keep only the largest grain
    # mask_below_labelled = label(mask_below)
    # regions = regionprops(mask_below_labelled)
    # areas = [region.area for region in regions]
    # mask_below[mask_below_labelled != np.argmax(areas) + 1] = False
    # # save the mask below
    # np.save("./tests/resources/minicircle_cropped_masks_below_background.npy", mask_below_background)

    grain_masks = {"above": mask_above, "below": mask_below}

    grainstats_df, _ = run_grainstats(
        image=flattened_image,
        pixel_to_nm_scaling=0.4940029296875,
        grain_masks=grain_masks,
        filename="dummy filename",
        basename=RESOURCES,
        grainstats_config=process_scan_config["grainstats"],
        plotting_config=process_scan_config["plotting"],
        grain_out_path=tmp_path,
    )

    assert isinstance(grainstats_df, pd.DataFrame)
    assert grainstats_df.shape[0] == 13
    assert len(grainstats_df.columns) == 21


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
