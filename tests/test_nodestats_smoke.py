"""Smoke test."""

import importlib.resources as pkg_resources
from pathlib import Path

import h5py
import yaml
from test_io import dict_almost_equal

import topostats
from topostats.io import LoadScans, hdf5_to_dict, read_yaml
from topostats.processing import (
    process_scan,
)

BASE_DIR = Path.cwd()
SMOKE = BASE_DIR / "tests/smoke"


def test_smoke(regtest, tmp_path, load_scan_spm: LoadScans) -> None:
    """Regression test for checking the process_scan functions correctly."""
    load_scan_spm.get_data()
    img_dic = load_scan_spm.img_dict
    # Ensure there are below grains
    process_scan_config = read_yaml(BASE_DIR / "topostats" / "default_config.yaml")
    plotting_dictionary = pkg_resources.open_text(topostats, "plotting_dictionary.yaml")
    process_scan_config["plotting"]["plot_dict"] = yaml.safe_load(plotting_dictionary.read())
    process_scan_config["grains"]["direction"] = "above"
    _, results, img_stats = process_scan(
        topostats_object=img_dic["minicircle"],
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

    # Regtest for the topostats file
    assert Path.exists(tmp_path / "tests/resources/processed/minicircle.topostats")
    with h5py.File(tmp_path / "tests/resources/processed/minicircle.topostats", "r") as f:
        saved_topostats = hdf5_to_dict(f, group_path="/")

    with h5py.File(SMOKE / "minicircle.topostats.out", "r") as f:
        expected_topostats = hdf5_to_dict(f, group_path="/")

    # Remove the image path as this differs on CI
    expected_topostats.pop("img_path")
    saved_topostats.pop("img_path")

    # Check the keys, this will flag all new keys when adding output stats
    assert expected_topostats.keys() == saved_topostats.keys()
    # Check the data
    assert dict_almost_equal(expected_topostats, saved_topostats)
