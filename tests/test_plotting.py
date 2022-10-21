"""Tests for the plotting module."""
from pathlib import Path

import pandas as pd
import pytest

from topostats.plotting import (
    importfromfile,
    savestats,
    pathman,
    labelunitconversion,
    dataunitconversion,
    plotkde,
    plotkde2var,
    plothist,
    plothist2var,
    plotdist,
    plotdist2var,
    plotviolin,
    plotjoint,
    plotLinearVsCircular,
    computeStats,
)

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"


def test_importfromfile(minicircle_all_statistics: pd.DataFrame) -> None:
    """Regression test for importfromfile()."""
    minicircle_df = importfromfile(RESOURCES / "minicircle_default_all_statistics.csv")
    assert isinstance(minicircle_df, pd.DataFrame)
    pd.testing.assert_frame_equal(minicircle_all_statistics, minicircle_df)


def test_savestats(regtest) -> None:
    """Regression test for savestats()."""
    assert True


# Not testing, path is contingent on system and changes with each test
# def test_pathman(tmpdir) -> None:
#     """Regression test for pathman()."""
#     plotname = pathman(str(tmpdir))
#     assert plotname == "/tmp/"


def test_labelunitconversion() -> None:
    """Regression test for labelunitconversion()."""
    assert True


def test_dataunitconversion() -> None:
    """Regression test for dataunitconversion()."""
    assert True


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/distributions/")
def test_plotkde(minicircle_all_statistics: pd.DataFrame, tmpdir) -> None:
    """Regression test for plotkde()."""
    fig = plotkde(df=minicircle_all_statistics, plotarg="area", nm=False, specpath=tmpdir)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/distributions/")
def test_plotkde_nm(minicircle_all_statistics: pd.DataFrame, tmpdir) -> None:
    """Regression test for plotkde()."""
    fig = plotkde(df=minicircle_all_statistics, plotarg="area", nm=True, specpath=tmpdir)
    return fig


# @pytest.mark.mpl_image_compare(baseline_dir="resources/img/distributions/")
# def test_plotkde2var() -> None:
#     """Regression test for plotkde2var()."""
#     assert True


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/distributions/")
def test_plothist(minicircle_all_statistics: pd.DataFrame, tmpdir) -> None:
    """Regression test for plothist()."""
    fig = plothist(df=minicircle_all_statistics, plotarg="area", nm=False, specpath=tmpdir)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/distributions/")
def test_plothist_nm(minicircle_all_statistics: pd.DataFrame, tmpdir) -> None:
    """Regression test for plothist()."""
    fig = plothist(df=minicircle_all_statistics, plotarg="area", nm=True, specpath=tmpdir)
    return fig


# @pytest.mark.mpl_image_compare(baseline_dir="resources/img/distributions/")
# def test_plothist2var() -> None:
#     """Regression test for plothist2var()."""
#     assert True


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/distributions/")
def test_plotdist(minicircle_all_statistics: pd.DataFrame, tmpdir) -> None:
    """Regression test for plotdist()."""
    fig = plotdist(df=minicircle_all_statistics, plotarg="area", nm=False, specpath=tmpdir)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="resources/img/distributions/")
def test_plotdist_nm(minicircle_all_statistics: pd.DataFrame, tmpdir) -> None:
    """Regression test for plotdist()."""
    fig = plotdist(df=minicircle_all_statistics, plotarg="area", nm=True, specpath=tmpdir)
    return fig


# @pytest.mark.mpl_image_compare(baseline_dir="resources/img/distributions/")
# def test_plotdist2var() -> None:
#     """Regression test for plotdist2var()."""
#     assert True


# @pytest.mark.mpl_image_compare(baseline_dir="resources/img/distributions/")
# def test_plotviolin() -> None:
#     """Regression test for plotviolin()."""
#     assert True


# @pytest.mark.mpl_image_compare(baseline_dir="resources/img/distributions/")
# def test_plotjoint() -> None:
#     """Regression test for plotjoint()."""
#     assert True


# @pytest.mark.mpl_image_compare(baseline_dir="resources/img/distributions/")
# def test_plotLinearVsCircular() -> None:
#     """Regression test for plotLinearVsCircular()."""
#     assert True


def test_computeStats(regtest, minicircle_all_statistics: pd.DataFrame) -> None:
    """Regression test for computeStats()."""
    data = [
        minicircle_all_statistics["aspect_ratio"],
        minicircle_all_statistics["area"],
        minicircle_all_statistics["volume"],
        minicircle_all_statistics["min_feret"],
        minicircle_all_statistics["max_feret"],
        minicircle_all_statistics["Contour Lengths"],
        minicircle_all_statistics["End to End Distance"],
    ]
    columns = ["aspect_ratio", "area", "volume", "min_feret", "max_feret", "Contour Lengths", "End to End Distance"]

    range = (40, 100)
    statistics_df = computeStats(data=data, columns=columns, min=range[0], max=range[1])
    print(statistics_df.to_string(), file=regtest)
