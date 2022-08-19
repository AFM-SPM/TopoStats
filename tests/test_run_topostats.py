"""Test end-to-end running of topostats."""
from pathlib import Path

from topostats.run_topostats import process_scan
from topostats.io import read_yaml

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"


def test_process_scan(
    regtest,
    tmpdir,
    filter_config: dict,
    grains_config: dict,
    grainstats_config: dict,
    dnatracing_config: dict,
    plotting_config: dict,
) -> None:
    """Regression test the process_scan function correctly"""
    config = read_yaml(RESOURCES / "sample_config.yaml")
    # Tweak configuration
    config["filter"]["threshold_method"] = "std_dev"
    config["grains"]["threshold_method"] = "std_dev"
    config["grains"]["otsu_threshold_multiplier"] = 1.0

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
    print(results.to_string(), file=regtest)
