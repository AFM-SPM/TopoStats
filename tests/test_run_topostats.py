"""Test end-to-end running of topostats."""
import logging
from pathlib import Path
import imghdr

import pytest

# from topostats import run_topostats
from topostats.run_topostats import process_scan, main as run_topostats_main
from topostats.io import LoadScans
from topostats.utils import update_plotting_config

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"


# Can't see a way of paramterising with pytest-regtest as it writes to a file based on the file/function
# so instead we run three regression tests.
def test_process_scan_lower(regtest, tmp_path, process_scan_config: dict, load_scan_data: LoadScans) -> None:
    """Regression test for checking the process_scan functions correctly"""
    process_scan_config["grains"]["direction"] = "lower"
    img_dic = load_scan_data.img_dic
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


def test_process_scan_upper(regtest, tmp_path, process_scan_config: dict, load_scan_data: LoadScans) -> None:
    """Regression test for checking the process_scan functions correctly"""
    img_dic = load_scan_data.img_dic
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
    img_dic = load_scan_data.img_dic
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


# @pytest.fixture
# def run_topostats_arguments() -> arg.ArgumentParser:


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

    print(f"image set: {process_scan_config['plotting']['image_set']}")

    img_dic = load_scan_data.img_dic
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
        Path.exists(tmp_path / "tests/resources/processed/minicircle/grains/upper" / "minicircle_grain_image_0.png")
        == expected
    )
    assert (
        Path.exists(tmp_path / "tests/resources/processed/minicircle/grains/upper" / "minicircle_grain_mask_0.png")
        == expected
    )
    assert (
        Path.exists(
            tmp_path / "tests/resources/processed/minicircle/grains/upper" / "minicircle_grain_mask_image_0.png"
        )
        == expected
    )


@pytest.mark.parametrize("extension", [("png"), ("tiff")])
def test_save_format(process_scan_config: dict, load_scan_data: LoadScans, tmp_path: Path, extension: str):
    "Tests if save format applied to cropped images"
    # minicircle_grainstats.save_cropped_grains = True
    # minicircle_grainstats.plot_opts["grain_image"]["save_format"] = extension
    # minicircle_grainstats.base_output_dir = tmp_path
    # minicircle_grainstats.calculate_stats()

    process_scan_config["plotting"]["image_set"] = "all"
    process_scan_config["plotting"]["save_format"] = extension
    process_scan_config["plotting"] = update_plotting_config(process_scan_config["plotting"])

    img_dic = load_scan_data.img_dic
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
            tmp_path / "tests/resources/processed/minicircle/grains/upper" / f"minicircle_grain_image_0.{extension}"
        )
        == extension
    )


@pytest.mark.parametrize("option", ("-h", "--help"))
def test_run_topostats_main_help(capsys, option) -> None:
    """Test the -h/--help flag to run_topostats."""
    try:
        run_topostats_main([option])
    except SystemExit:
        pass
    assert "Process AFM images." in capsys.readouterr().out


def test_run_topostats_process_all(caplog) -> None:
    """Test run_topostats completes without error when no arguments are given."""
    caplog.set_level(logging.INFO)
    # Explicitly force loading of topostats/default_config.yaml as I couldn't work out how to invoke process_all()
    # without any arguments as it defaults to 'sys.argv' as this is wrapped within pytest it picks up the arguments
    # pytest was invoked with (see thread on StackOverflow at https://stackoverflow.com/a/55260580/1444043)
    run_topostats_main(args=["--config", f"{BASE_DIR / 'topostats' / 'default_config.yaml'}"])
    assert "~~~~~~~~~~~~~~~~~~~~ COMPLETE ~~~~~~~~~~~~~~~~~~~~" in caplog.text
