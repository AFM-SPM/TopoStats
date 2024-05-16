"""Tests for the plotting module."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import yaml
from matplotlib.figure import Figure

import topostats
from topostats.entry_point import entry_point
from topostats.plotting import TopoSum, _pad_array, plot_height_profiles, toposum

# pylint: disable=protected-access

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"


def test_melt_data():
    """Test the melt_data method of the TopoSum class."""
    df_to_melt = {
        "Image": ["im1", "im1", "im1", "im2", "im2", "im3", "im3"],
        "threshold": ["above", "above", "above", "below", "below", "above", "above"],
        "molecule_number": [0, 1, 2, 0, 1, 0, 1],
        "basename": ["super/sub1", "super/sub1", "super/sub1", "super/sub1", "super/sub1", "super/sub2", "super/sub2"],
        "area": [10, 20, 30, 40, 50, 60, 70],
    }

    df_to_melt = pd.DataFrame(df_to_melt)

    melted_data = TopoSum.melt_data(df=df_to_melt, stat_to_summarize="area", var_to_label={"area": "AREA"})

    expected = {
        "molecule_number": [0, 1, 2, 0, 1, 0, 1],
        "basename": ["super/sub1", "super/sub1", "super/sub1", "super/sub1", "super/sub1", "super/sub2", "super/sub2"],
        "variable": ["AREA", "AREA", "AREA", "AREA", "AREA", "AREA", "AREA"],
        "value": [10, 20, 30, 40, 50, 60, 70],
    }

    expected = pd.DataFrame(expected)

    pd.testing.assert_frame_equal(melted_data, expected)


def test_df_from_csv(toposum_object_multiple_directories: TopoSum) -> None:
    """Test loading of CSV file."""
    assert isinstance(toposum_object_multiple_directories.df, pd.DataFrame)
    expected = pd.read_csv(RESOURCES / "toposum_all_statistics_multiple_directories.csv", header=0)
    pd.testing.assert_frame_equal(expected, toposum_object_multiple_directories.df)


def test_toposum_class(toposum_object_multiple_directories: TopoSum) -> None:
    """Check the TopoSum class has been correctly instantiated."""
    assert isinstance(toposum_object_multiple_directories.df, pd.DataFrame)
    assert isinstance(toposum_object_multiple_directories.stat_to_sum, str)
    assert isinstance(toposum_object_multiple_directories.molecule_id, str)
    assert isinstance(toposum_object_multiple_directories.image_id, str)
    assert isinstance(toposum_object_multiple_directories.hist, bool)
    assert isinstance(toposum_object_multiple_directories.kde, bool)
    assert isinstance(toposum_object_multiple_directories.savefig_format, str)
    assert isinstance(toposum_object_multiple_directories.output_dir, Path)
    assert isinstance(toposum_object_multiple_directories.var_to_label, dict)


def test_outfile(toposum_object_multiple_directories: TopoSum) -> None:
    """Check fig and ax returned."""
    outfile = toposum_object_multiple_directories._outfile(plot_suffix="testing")
    assert isinstance(outfile, str)
    assert outfile == "area_testing"


def test_args_input_csv() -> None:
    """Test modifying the configuration value for the input CSV to be summarised."""
    assert True


def test_var_to_label_config(tmp_path: Path) -> None:
    """Test the var_to_label configuration file is created correctly."""
    with pytest.raises(SystemExit):
        entry_point(
            manually_provided_args=[
                "summary",
                "--create-label-file",
                f"{tmp_path / 'var_to_label_config.yaml'}",
                "--input_csv",
                f"{str(RESOURCES / 'minicircle_default_all_statistics.csv')}",
            ]
        )
    var_to_label_config = tmp_path / "var_to_label_config.yaml"
    with var_to_label_config.open("r", encoding="utf-8") as f:
        var_to_label_str = f.read()
    var_to_label = yaml.safe_load(var_to_label_str)
    plotting_yaml = (resources.files(topostats.__package__) / "var_to_label.yaml").read_text()
    expected_var_to_label = yaml.safe_load(plotting_yaml)

    assert var_to_label == expected_var_to_label


@pytest.mark.parametrize(
    ("var", "expected_label"),
    [
        ("contour_length", "Contour Length"),
        ("end_to_end_distance", "End to End Distance"),
        ("grain_bound_len", "Circumference"),
        ("grain_curvature1", "Smaller Curvature"),
    ],
)
def test_set_label(toposum_object_multiple_directories: TopoSum, var: str, expected_label: str) -> None:
    """Test labels are returned correctly."""
    toposum_object_multiple_directories._set_label(var)
    assert toposum_object_multiple_directories.label == expected_label


def test_set_label_keyerror(toposum_object_multiple_directories: TopoSum) -> None:
    """Test labels are returned correctly."""
    with pytest.raises(KeyError):
        toposum_object_multiple_directories._set_label("non_existent_stat")


def test_toposum(summary_config: dict) -> None:
    """Test the toposum function returns a dictionary with figures."""
    summary_config["csv_file"] = RESOURCES / "minicircle_default_all_statistics.csv"
    summary_config["df"] = pd.read_csv(summary_config["csv_file"])
    summary_config["violin"] = True
    summary_config["stats_to_sum"] = ["area"]
    summary_config["pickle_plots"] = True
    summary_config.pop("stat_to_sum")
    figures = toposum(summary_config)
    assert isinstance(figures, dict)
    assert "area" in figures.keys()
    assert isinstance(figures["area"]["dist"]["figure"], Figure)
    assert isinstance(figures["area"]["violin"]["figure"], Figure)


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/distributions/")
def test_plot_kde(toposum_object_single_directory: TopoSum) -> None:
    """Regression test for sns_plot() with a single KDE."""
    toposum_object_single_directory.hist = False
    fig, _ = toposum_object_single_directory.sns_plot()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/distributions/")
def test_plot_kde_multiple_directories(toposum_object_multiple_directories: TopoSum) -> None:
    """Regression test for sns_plot() with multiple KDE."""
    toposum_object_multiple_directories.hist = False
    fig, _ = toposum_object_multiple_directories.sns_plot()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/distributions/")
def test_plot_hist(toposum_object_single_directory: TopoSum) -> None:
    """Regression test for sns_plot() with a single histogram."""
    toposum_object_single_directory.kde = False
    fig, _ = toposum_object_single_directory.sns_plot()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/distributions/")
def test_plot_hist_multiple_directories(toposum_object_multiple_directories: TopoSum) -> None:
    """Regression test for sns_plot() with multiple overlaid histograms."""
    toposum_object_multiple_directories.kde = False
    fig, _ = toposum_object_multiple_directories.sns_plot()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/distributions/")
def test_plot_hist_kde(toposum_object_single_directory: TopoSum) -> None:
    """Test plotting Kernel Density Estimate and Histogram for area."""
    fig, _ = toposum_object_single_directory.sns_plot()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/distributions/")
def test_plot_hist_kde_multiple_directories(toposum_object_multiple_directories: TopoSum) -> None:
    """Test plotting Kernel Density Estimate and Histogram for area."""
    fig, _ = toposum_object_multiple_directories.sns_plot()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/distributions/")
def test_plot_violin(toposum_object_single_directory: TopoSum) -> None:
    """Test plotting Kernel Density Estimate and Histogram for area for a single image."""
    fig, _ = toposum_object_single_directory.sns_violinplot()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/distributions/")
def test_plot_violin_multiple_directories(toposum_object_multiple_directories: TopoSum) -> None:
    """Test plotting Kernel Density Estimate and Histogram for area with multiple images."""
    fig, _ = toposum_object_multiple_directories.sns_violinplot()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/height_profiles/")
@pytest.mark.parametrize(
    ("height_profile"),
    [
        pytest.param(np.asarray([0, 0, 0, 2, 3, 4, 4, 4, 3, 2, 0, 0, 0]), id="Single height profile"),
        pytest.param(
            [
                np.asarray([0, 0, 0, 2, 3, 4, 4, 4, 3, 2, 0, 0, 0]),
                np.asarray([0, 0, 0, 2, 4, 5, 5, 5, 4, 2, 0, 0, 0]),
            ],
            id="Two arrays of same length",
        ),
        pytest.param(
            [
                np.asarray([0, 0, 0, 2, 3, 4, 4, 4, 3, 2, 0, 0, 0]),
                np.asarray([0, 0, 2, 4, 5, 5, 5, 4, 2, 0, 0]),
            ],
            id="Two arrays of different length (diff in length is even)",
        ),
        pytest.param(
            [
                np.asarray([0, 0, 0, 2, 3, 4, 4, 4, 3, 2, 0, 0, 0]),
                np.asarray([0, 0, 2, 4, 5, 5, 5, 4, 2, 0, 0, 0]),
            ],
            id="Two arrays of different length (diff in length is odd)",
        ),
        pytest.param(
            [
                np.asarray([0, 0, 0, 2, 3, 4, 4, 4, 3, 2, 0, 0, 0]),
                np.asarray([0, 0, 2, 4, 5, 5, 5, 4, 2, 0, 0]),
                np.asarray([0, 0, 1, 5, 6, 7, 6, 5, 1, 0, 0, 0]),
            ],
            id="Three arrays of different length (one even, one odd)",
        ),
        pytest.param(
            [
                np.asarray([0, 0, 0, 2, 3, 4, 4, 4, 3, 2, 0, 0, 0]),
                np.asarray([0, 0, 2, 4, 5, 5, 5, 4, 2, 0, 0]),
                np.asarray([0, 0, 1, 5, 6, 7, 6, 5, 1, 0, 0, 0]),
                np.asarray([0, 0, 1, 4, 1, 0, 0]),
            ],
            id="Four arrays of different length (one even, two odd)",
        ),
    ],
)
def test_plot_height_profiles(height_profile: list | npt.NDArray) -> None:
    """Test plotting of height profiles."""
    fig, _ = plot_height_profiles(height_profile)
    return fig


@pytest.mark.parametrize(
    ("arrays", "max_array_length", "target"),
    [
        pytest.param(
            np.asarray([1, 1, 1]),
            3,
            np.asarray([1, 1, 1]),
            id="Array length 3, max array length 3",
        ),
        pytest.param(
            np.asarray([1, 1, 1]),
            5,
            np.asarray([0, 1, 1, 1, 0]),
            id="Array length 3, max array length 5",
        ),
        pytest.param(
            np.asarray([1, 1, 1]),
            4,
            np.asarray([1, 1, 1, 0]),
            id="Array length 3, max array length 4",
        ),
    ],
)
def test_pad_array(arrays: list, max_array_length: int, target: list) -> None:
    """Test padding of arrays."""
    np.testing.assert_array_equal(_pad_array(arrays, max_array_length), target)
