"""Test end-to-end running of topostats."""
from pathlib import Path

# pylint: disable=deprecated-module
import imghdr
import pytest

from topostats.io import LoadScans
from topostats.processing import check_run_steps, process_scan
from topostats.utils import update_plotting_config

BASE_DIR = Path.cwd()


# Can't see a way of paramterising with pytest-regtest as it writes to a file based on the file/function
# so instead we run three regression tests.
def test_process_scan_below(regtest, tmp_path, process_scan_config: dict, load_scan_data: LoadScans) -> None:
    """Regression test for checking the process_scan functions correctly"""
    process_scan_config["grains"]["direction"] = "below"
    img_dic = load_scan_data.img_dict
    _, results = process_scan(
        img_path_px2nm=img_dic["minicircle"],
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
    print(results.to_string(), file=regtest)  # noqa: T201


def test_process_scan_above(regtest, tmp_path, process_scan_config: dict, load_scan_data: LoadScans) -> None:
    """Regression test for checking the process_scan functions correctly"""
    img_dic = load_scan_data.img_dict
    _, results = process_scan(
        img_path_px2nm=img_dic["minicircle"],
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
    print(results.to_string(), file=regtest)  # noqa: T201


def test_process_scan_both(regtest, tmp_path, process_scan_config: dict, load_scan_data: LoadScans) -> None:
    """Regression test for checking the process_scan functions correctly"""
    process_scan_config["grains"]["direction"] = "both"
    img_dic = load_scan_data.img_dict
    _, results = process_scan(
        img_path_px2nm=img_dic["minicircle"],
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
    print(results.to_string(), file=regtest)  # noqa: T201


@pytest.mark.parametrize(
    "image_set, expected",
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

    img_dic = load_scan_data.img_dict
    _, _ = process_scan(
        img_path_px2nm=img_dic["minicircle"],
        base_dir=BASE_DIR,
        filter_config=process_scan_config["filter"],
        grains_config=process_scan_config["grains"],
        grainstats_config=process_scan_config["grainstats"],
        dnatracing_config=process_scan_config["dnatracing"],
        plotting_config=process_scan_config["plotting"],
        output_dir=tmp_path,
    )

    assert (
        Path.exists(tmp_path / "tests/resources/processed/minicircle/grains/above" / "minicircle_grain_image_0.png")
        == expected
    )
    assert (
        Path.exists(tmp_path / "tests/resources/processed/minicircle/grains/above" / "minicircle_grain_mask_0.png")
        == expected
    )
    assert (
        Path.exists(
            tmp_path / "tests/resources/processed/minicircle/grains/above" / "minicircle_grain_mask_image_0.png"
        )
        == expected
    )


@pytest.mark.parametrize("extension", [("png"), ("tiff")])
def test_save_format(process_scan_config: dict, load_scan_data: LoadScans, tmp_path: Path, extension: str):
    """Tests if save format applied to cropped images"""
    process_scan_config["plotting"]["image_set"] = "all"
    process_scan_config["plotting"]["save_format"] = extension
    process_scan_config["plotting"] = update_plotting_config(process_scan_config["plotting"])

    img_dic = load_scan_data.img_dict
    _, _ = process_scan(
        img_path_px2nm=img_dic["minicircle"],
        base_dir=BASE_DIR,
        filter_config=process_scan_config["filter"],
        grains_config=process_scan_config["grains"],
        grainstats_config=process_scan_config["grainstats"],
        dnatracing_config=process_scan_config["dnatracing"],
        plotting_config=process_scan_config["plotting"],
        output_dir=tmp_path,
    )

    assert (
        imghdr.what(
            tmp_path / "tests/resources/processed/minicircle/grains/above" / f"minicircle_grain_image_0.{extension}"
        )
        == extension
    )


@pytest.mark.parametrize(
    "filter_run, grains_run, grainstats_run, dnatracing_run, log_msg",
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
    "filter_run, grains_run, grainstats_run, dnatracing_run, log_msg1, log_msg2",
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
            "13-gaussian_filtered",
        ),
        (
            True,
            True,
            False,
            False,
            "Calculation of grainstats disabled, returning empty data frame.",
            "22-labelled_image_bboxes",
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
            "There are 15 circular and 6 linear DNA molecules found in the image",
            "Combining above grain statistics and dnatracing statistics",
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
    later staged can run and do not disable earlier stages."""
    img_dic = load_scan_data.img_dict
    process_scan_config["filter"]["run"] = filter_run
    process_scan_config["grains"]["run"] = grains_run
    process_scan_config["grainstats"]["run"] = grainstats_run
    process_scan_config["dnatracing"]["run"] = dnatracing_run
    _, _ = process_scan(
        img_path_px2nm=img_dic["minicircle"],
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


def test_process_scan_region_properties_is_none(
    process_scan_config: dict, load_scan_data: LoadScans, tmp_path: Path, caplog
) -> None:
    """Test capturing disabling GrainStats where no grains have been found."""
    img_dic = load_scan_data.img_dict
    process_scan_config["grains"]["absolute_area_threshold"] = [4000, 9000]
    _, _ = process_scan(
        img_path_px2nm=img_dic["minicircle"],
        base_dir=BASE_DIR,
        filter_config=process_scan_config["filter"],
        grains_config=process_scan_config["grains"],
        grainstats_config=process_scan_config["grainstats"],
        dnatracing_config=process_scan_config["dnatracing"],
        plotting_config=process_scan_config["plotting"],
        output_dir=tmp_path,
    )
    assert "No grains to plot" in caplog.text
    assert "No grains detected skipping calculation of grain statistics and DNA tracing." in caplog.text
