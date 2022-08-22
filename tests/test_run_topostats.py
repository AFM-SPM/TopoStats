"""Test end-to-end running of topostats."""
from pathlib import Path

from topostats.run_topostats import process_scan
from topostats.io import read_yaml

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"


# Can't see a way of paramterising with pytest-regtest as it writes to a file based on the file/function
# so instead we run three regression tests.
def test_process_scan_lower(
    regtest,
    tmpdir,
    filter_config: dict,
    grains_config: dict,
    grainstats_config: dict,
    dnatracing_config: dict,
    plotting_config: dict,
) -> None:
    """Regression test for checking the process_scan functions correctly"""
    config = read_yaml(RESOURCES / "process_scan_config.yaml")
    config["grains"]["direction"] = "lower"
    _, results = process_scan(
        image_path=RESOURCES / "minicircle.spm",
        base_dir=BASE_DIR,
        filter_config=config["filter"],
        grains_config=config["grains"],
        grainstats_config=config["grainstats"],
        dnatracing_config=config["dnatracing"],
        plotting_config=config["plotting"],
        output_dir=tmpdir,
    )
    # Remove the Basename column as this differs on CI
    results.drop(["Basename"], axis=1, inplace=True)
    print(results.to_string(), file=regtest)


def test_process_scan_upper(
    regtest,
    tmpdir,
    filter_config: dict,
    grains_config: dict,
    grainstats_config: dict,
    dnatracing_config: dict,
    plotting_config: dict,
) -> None:
    """Regression test for checking the process_scan functions correctly"""
    config = read_yaml(RESOURCES / "process_scan_config.yaml")

    _, results = process_scan(
        image_path=RESOURCES / "minicircle.spm",
        base_dir=BASE_DIR,
        filter_config=config["filter"],
        grains_config=config["grains"],
        grainstats_config=config["grainstats"],
        dnatracing_config=config["dnatracing"],
        plotting_config=config["plotting"],
        output_dir=tmpdir,
    )
    # Remove the Basename column as this differs on CI
    results.drop(["Basename"], axis=1, inplace=True)
    print(results.to_string(), file=regtest)


def test_process_scan_both(
    regtest,
    tmpdir,
    filter_config: dict,
    grains_config: dict,
    grainstats_config: dict,
    dnatracing_config: dict,
    plotting_config: dict,
) -> None:
    """Regression test for checking the process_scan functions correctly"""
    config = read_yaml(RESOURCES / "process_scan_config.yaml")
    config["grains"]["direction"] = "both"
    _, results = process_scan(
        image_path=RESOURCES / "minicircle.spm",
        base_dir=BASE_DIR,
        filter_config=config["filter"],
        grains_config=config["grains"],
        grainstats_config=config["grainstats"],
        dnatracing_config=config["dnatracing"],
        plotting_config=config["plotting"],
        output_dir=tmpdir,
    )
    # Remove the Basename column as this differs on CI
    results.drop(["Basename"], axis=1, inplace=True)
    print(results.to_string(), file=regtest)
