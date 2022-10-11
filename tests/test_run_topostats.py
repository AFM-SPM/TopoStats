"""Test end-to-end running of topostats."""
import importlib.resources as pkg_resources
from pathlib import Path
import yaml
from topostats.run_topostats import process_scan
from topostats.io import read_yaml

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"


# Can't see a way of paramterising with pytest-regtest as it writes to a file based on the file/function
# so instead we run three regression tests.
def test_process_scan_lower(regtest, tmp_path, process_scan_config: dict) -> None:
    """Regression test for checking the process_scan functions correctly"""
    process_scan_config["grains"]["direction"] = "lower"
    _, results = process_scan(
        image_path=RESOURCES / "minicircle.spm",
        base_dir=BASE_DIR,
        filter_config=process_scan_config["filter"],
        grains_config=process_scan_config["grains"],
        grainstats_config=process_scan_config["grainstats"],
        dnatracing_config=process_scan_config["dnatracing"],
        plotting_config=process_scan_config["plotting"],
        output_dir=tmp_path,
    )
    # Remove the Basename column as this differs on CI
    results.drop(["Basename"], axis=1, inplace=True)
    print(results.to_string(), file=regtest)


def test_process_scan_upper(regtest, tmp_path, process_scan_config: dict) -> None:
    """Regression test for checking the process_scan functions correctly"""
    _, results = process_scan(
        image_path=RESOURCES / "minicircle.spm",
        base_dir=BASE_DIR,
        filter_config=process_scan_config["filter"],
        grains_config=process_scan_config["grains"],
        grainstats_config=process_scan_config["grainstats"],
        dnatracing_config=process_scan_config["dnatracing"],
        plotting_config=process_scan_config["plotting"],
        output_dir=tmp_path,
    )
    # Remove the Basename column as this differs on CI
    results.drop(["Basename"], axis=1, inplace=True)
    print(results.to_string(), file=regtest)


def test_process_scan_both(regtest, tmp_path, process_scan_config: dict) -> None:
    """Regression test for checking the process_scan functions correctly"""
    process_scan_config["grains"]["direction"] = "both"
    _, results = process_scan(
        image_path=RESOURCES / "minicircle.spm",
        base_dir=BASE_DIR,
        filter_config=process_scan_config["filter"],
        grains_config=process_scan_config["grains"],
        grainstats_config=process_scan_config["grainstats"],
        dnatracing_config=process_scan_config["dnatracing"],
        plotting_config=process_scan_config["plotting"],
        output_dir=tmp_path,
    )
    # Remove the Basename column as this differs on CI
    results.drop(["Basename"], axis=1, inplace=True)
    print(results.to_string(), file=regtest)
