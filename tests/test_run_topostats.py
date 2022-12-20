"""Test end-to-end running of topostats."""
from pathlib import Path

from topostats.run_topostats import process_scan
from topostats.io import LoadScans

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
        scars_config=process_scan_config["remove_scars"],
        grains_config=process_scan_config["grains"],
        grainstats_config=process_scan_config["grainstats"],
        dnatracing_config=process_scan_config["dnatracing"],
        plotting_config=process_scan_config["plotting"],
        output_dir=tmp_path,
    )
    # Remove the Basename column as this differs on CI
    results.drop(["Basename"], axis=1, inplace=True)
    print(results.to_string(), file=regtest)  # noqa: T201


def test_process_scan_upper(regtest, tmp_path, process_scan_config: dict, load_scan_data: LoadScans) -> None:
    """Regression test for checking the process_scan functions correctly"""
    img_dic = load_scan_data.img_dic
    _, results = process_scan(
        img_path_px2nm=img_dic["minicircle"],
        base_dir=BASE_DIR,
        filter_config=process_scan_config["filter"],
        scars_config=process_scan_config["remove_scars"],
        grains_config=process_scan_config["grains"],
        grainstats_config=process_scan_config["grainstats"],
        dnatracing_config=process_scan_config["dnatracing"],
        plotting_config=process_scan_config["plotting"],
        output_dir=tmp_path,
    )
    # Remove the Basename column as this differs on CI
    results.drop(["Basename"], axis=1, inplace=True)
    print(results.to_string(), file=regtest)  # noqa: T201


def test_process_scan_both(regtest, tmp_path, process_scan_config: dict, load_scan_data: LoadScans) -> None:
    """Regression test for checking the process_scan functions correctly"""
    process_scan_config["grains"]["direction"] = "both"
    img_dic = load_scan_data.img_dic
    _, results = process_scan(
        img_path_px2nm=img_dic["minicircle"],
        base_dir=BASE_DIR,
        filter_config=process_scan_config["filter"],
        scars_config=process_scan_config["remove_scars"],
        grains_config=process_scan_config["grains"],
        grainstats_config=process_scan_config["grainstats"],
        dnatracing_config=process_scan_config["dnatracing"],
        plotting_config=process_scan_config["plotting"],
        output_dir=tmp_path,
    )
    # Remove the Basename column as this differs on CI
    results.drop(["Basename"], axis=1, inplace=True)
    print(results.to_string(), file=regtest)  # noqa: T201
