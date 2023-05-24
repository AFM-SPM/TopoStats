"""Tests for the plotting module."""
import importlib.resources as pkg_resources
from pathlib import Path
import yaml

from matplotlib.figure import Figure
import pandas as pd
import pytest

import topostats
from topostats.plotting import TopoSum, toposum
from topostats.plotting import main as plotting_main

# pylint: disable=protected-access

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"


def test_melt_data():
    """Test the melt_data method of the TopoSum class"""

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


@pytest.mark.parametrize(
    "input_paths, expected_paths",
    [
        ([Path("a/b/c/d"), Path("a/b/e/f"), Path("a/b/g"), Path("a/b/h")], ["c/d", "e/f", "g", "h"]),
        (["a/b/c/d", "a/b/e/f", "a/b/g", "a/b/h"], ["c/d", "e/f", "g", "h"]),
        (["g", "a/b/e/f", "a/b/g", "a/b/h"], ["g", "a/b/e/f", "a/b/g", "a/b/h"]),
        (["a/b/c/d"], ["a/b/c/d"]),
        (["a/b/c/d", "a/b/c/d"], ["a/b/c/d", "a/b/c/d"]),
    ],
)
def test_get_paths_relative_to_deepest_common_path(input_paths: list, expected_paths: list):
    """Test the get_paths_relative_to_deepest_common_path method of the TopoSum class."""

    relative_paths = TopoSum.get_paths_relative_to_deepest_common_path(input_paths)

    assert relative_paths == expected_paths


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
    assert isinstance(toposum_object_multiple_directories.file_ext, str)
    assert isinstance(toposum_object_multiple_directories.output_dir, Path)
    assert isinstance(toposum_object_multiple_directories.var_to_label, dict)


def test_outfile(toposum_object_multiple_directories: TopoSum) -> None:
    """Check fig and ax returned"""
    outfile = toposum_object_multiple_directories._outfile(plot_suffix="testing")
    assert isinstance(outfile, str)
    assert outfile == "area_testing"


def test_args_input_csv() -> None:
    """Test modifying the configuration value for the input CSV to be summarised."""
    assert True


def test_var_to_label_config(tmp_path: Path) -> None:
    """Test the var_to_label configuration file is created correctly."""
    with pytest.raises(SystemExit):
        plotting_main(
            args=[
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
    plotting_yaml = pkg_resources.open_text(topostats.__package__, "var_to_label.yaml")
    expected_var_to_label = yaml.safe_load(plotting_yaml.read())

    assert var_to_label == expected_var_to_label


@pytest.mark.parametrize(
    "var,expected_label",
    [
        ("contour_lengths", "Contour Lengths"),
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
def test_plot_kde_multiple_images(toposum_object_multiple_directories: TopoSum) -> None:
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
def test_plot_hist_multiple_images(toposum_object_multiple_directories: TopoSum) -> None:
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
def test_plot_hist_kde_multiple_images(toposum_object_multiple_directories: TopoSum) -> None:
    """Test plotting Kernel Density Estimate and Histogram for area."""
    fig, _ = toposum_object_multiple_directories.sns_plot()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/distributions/")
def test_plot_violin(toposum_object_single_directory: TopoSum) -> None:
    """Test plotting Kernel Density Estimate and Histogram for area for a single image."""
    fig, _ = toposum_object_single_directory.sns_violinplot()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/distributions/")
def test_plot_violin_multiple_images(toposum_object_multiple_directories: TopoSum) -> None:
    """Test plotting Kernel Density Estimate and Histogram for area with multiple images."""
    fig, _ = toposum_object_multiple_directories.sns_violinplot()
    return fig
