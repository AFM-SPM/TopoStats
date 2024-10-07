"""Regression test for nodestats tracing."""

from pathlib import Path

import h5py
from test_io import dict_almost_equal

from topostats.io import LoadScans, hdf5_to_dict
from topostats.processing import (
    process_scan,
)

BASE_DIR = Path.cwd()
REGRESSION = BASE_DIR / "tests/nodestats-regression"


def test_nodestats_tracing_results(default_config, regtest, tmp_path, load_scan_spm: LoadScans) -> None:
    """Regression test for checking the process_scan functions correctly."""
    load_scan_spm.get_data()
    img_dic = load_scan_spm.img_dict
    # Ensure there are below grains
    default_config["grains"]["direction"] = "above"
    _, results, img_stats = process_scan(
        topostats_object=img_dic["minicircle"],
        base_dir=BASE_DIR,
        filter_config=default_config["filter"],
        grains_config=default_config["grains"],
        grainstats_config=default_config["grainstats"],
        disordered_tracing_config=default_config["disordered_tracing"],
        dnatracing_config=default_config["dnatracing"],
        plotting_config=default_config["plotting"],
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

    with h5py.File(REGRESSION / "minicircle.topostats.out", "r") as f:
        expected_topostats = hdf5_to_dict(f, group_path="/")

    # Remove the image path as this differs on CI
    expected_topostats.pop("img_path")
    saved_topostats.pop("img_path")

    # Check the keys, this will flag all new keys when adding output stats
    assert expected_topostats.keys() == saved_topostats.keys()
    # Check the data
    assert dict_almost_equal(expected_topostats, saved_topostats)
