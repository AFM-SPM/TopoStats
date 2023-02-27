"""Tests for the plotting module."""
from pathlib import Path

from matplotlib.figure import Figure
import pandas as pd
import pytest

from topostats.plotting import TopoSum, toposum

# pylint: disable=protected-access

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"


def test_df_from_csv(minicircle_all_statistics: pd.DataFrame, toposum_object: TopoSum) -> None:
    """Test loading of CSV file."""
    assert isinstance(toposum_object.df, pd.DataFrame)
    pd.testing.assert_frame_equal(minicircle_all_statistics, toposum_object.df)


def test_toposum_class(toposum_object: TopoSum) -> None:
    """Check the TopoSum class has been correctly instantiated."""
    assert isinstance(toposum_object.df, pd.DataFrame)
    assert isinstance(toposum_object.stat_to_sum, str)
    assert isinstance(toposum_object.molecule_id, str)
    assert isinstance(toposum_object.image_id, str)
    assert isinstance(toposum_object.hist, bool)
    assert isinstance(toposum_object.kde, bool)
    assert isinstance(toposum_object.file_ext, str)
    assert isinstance(toposum_object.output_dir, Path)
    assert isinstance(toposum_object.var_to_label, dict)


def test_outfile(toposum_object: TopoSum) -> None:
    """Check fig and ax returned"""
    outfile = toposum_object._outfile(plot_suffix="testing")
    assert isinstance(outfile, str)
    assert outfile == "area_testing"


@pytest.mark.parametrize(
    "var,expected_label",
    [
        ("contour_lengths", "Contour Lengths"),
        ("end_to_end_distance", "End to End Distance"),
        ("grain_bound_len", "Circumference"),
        ("grain_curvature1", "Smaller Curvature"),
    ],
)
def test_set_label(toposum_object: TopoSum, var: str, expected_label: str) -> None:
    """Test labels are returned correctly."""
    toposum_object._set_label(var)
    assert toposum_object.label == expected_label


def test_set_label_keyerror(toposum_object: TopoSum) -> None:
    """Test labels are returned correctly."""
    with pytest.raises(KeyError):
        toposum_object._set_label("non_existent_stat")


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
def test_plot_kde(summary_config: dict) -> None:
    """Regression test for plotkde()."""
    summary_config["hist"] = False
    _toposum = TopoSum(csv_file=RESOURCES / "minicircle_default_all_statistics.csv", **summary_config)
    fig, _ = _toposum.sns_plot()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/distributions/")
def test_plot_kde_multiple_images(summary_config: dict, toposum_multiple_images: pd.DataFrame) -> None:
    """Regression test for plotkde()."""
    summary_config["hist"] = False
    _toposum = TopoSum(csv_file=RESOURCES / "minicircle_default_all_statistics.csv", **summary_config)
    _toposum.melted_data = toposum_multiple_images
    fig, _ = _toposum.sns_plot()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/distributions/")
def test_plot_hist(summary_config: dict) -> None:
    """Regression test for plotkde()."""
    summary_config["kde"] = False
    _toposum = TopoSum(csv_file=RESOURCES / "minicircle_default_all_statistics.csv", **summary_config)
    fig, _ = _toposum.sns_plot()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/distributions/")
def test_plot_hist_multiple_images(summary_config: dict, toposum_multiple_images: pd.DataFrame) -> None:
    """Regression test for plotkde()."""
    summary_config["kde"] = False
    _toposum = TopoSum(csv_file=RESOURCES / "minicircle_default_all_statistics.csv", **summary_config)
    _toposum.melted_data = toposum_multiple_images
    fig, _ = _toposum.sns_plot()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/distributions/")
def test_plot_hist_kde(summary_config: dict) -> None:
    """Test plotting Kernel Density Estimate and Histogram for area."""
    _toposum = TopoSum(csv_file=RESOURCES / "minicircle_default_all_statistics.csv", **summary_config)
    fig, _ = _toposum.sns_plot()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/distributions/")
def test_plot_hist_kde_multiple_images(summary_config: dict, toposum_multiple_images: pd.DataFrame) -> None:
    """Test plotting Kernel Density Estimate and Histogram for area."""
    _toposum = TopoSum(csv_file=RESOURCES / "minicircle_default_all_statistics.csv", **summary_config)
    _toposum.melted_data = toposum_multiple_images
    fig, _ = _toposum.sns_plot()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/distributions/")
def test_plot_violin(summary_config: dict) -> None:
    """Test plotting Kernel Density Estimate and Histogram for area."""
    _toposum = TopoSum(csv_file=RESOURCES / "minicircle_default_all_statistics.csv", **summary_config)
    fig, _ = _toposum.sns_violinplot()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/distributions/")
def test_plot_violin_multiple_images(summary_config: dict, toposum_multiple_images: pd.DataFrame) -> None:
    """Test plotting Kernel Density Estimate and Histogram for area."""
    _toposum = TopoSum(csv_file=RESOURCES / "minicircle_default_all_statistics.csv", **summary_config)
    _toposum.melted_data = toposum_multiple_images
    fig, _ = _toposum.sns_violinplot()
    return fig
