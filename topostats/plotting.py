"""Plotting and summary of Statistics"""
import os
import argparse as arg
from collections import defaultdict


import importlib.resources as pkg_resources
import logging
from pathlib import Path
import sys
from typing import Union, Dict, List
import yaml
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from topostats.io import read_yaml, save_pkl, write_yaml
from topostats.logs.logs import LOGGER_NAME
from topostats.utils import update_config

LOGGER = logging.getLogger(LOGGER_NAME)


# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-locals


def create_parser() -> arg.ArgumentParser:
    """Create a parser for reading options."""
    parser = arg.ArgumentParser(
        description="Summarise and plot histograms, kernel density estimates and scatter plots of TopoStats"
        "grain and DNA Tracing statistics."
    )
    parser.add_argument("-i", "--input_csv", dest="csv_file", required=False, help="Path to CSV file to plot.")
    parser.add_argument(
        "-c",
        "--config_file",
        dest="config_file",
        required=False,
        help="Path to a YAML configuration file.",
    )
    parser.add_argument(
        "-l",
        "--var_to_label",
        dest="var_to_label",
        required=False,
        help="Path to a YAML plotting dictionary that maps variable names to labels.",
    )
    parser.add_argument(
        "--create-config-file",
        dest="create_config_file",
        type=str,
        required=False,
        help="Filename to write a sample YAML configuration file to (should end in '.yaml').",
    )
    parser.add_argument(
        "--create-label-file",
        dest="create_label_file",
        type=str,
        required=False,
        help="Filename to write a sample YAML label file to (should end in '.yaml').",
    )
    return parser


class TopoSum:
    """Class for summarising grain statistics in plots."""

    def __init__(
        self,
        df: Union[pd.DataFrame] = None,
        base_dir: Union[str, Path] = None,
        csv_file: Union[str, Path] = None,
        stat_to_sum: str = None,
        molecule_id: str = "molecule_number",
        image_id: str = "image",
        hist: bool = True,
        stat: str = "count",
        bins: int = 12,
        kde: bool = True,
        cut: float = 20,
        figsize: tuple = (16, 9),
        alpha: float = 0.5,
        palette: str = "deep",
        file_ext: str = "png",
        output_dir: Union[str, Path] = ".",
        var_to_label: dict = None,
        hue: str = "basename",
    ) -> None:
        """Initialise the class.

        Parameters
        ==========
        df: Union[pd.DataFrame]
            Pandas data frame of data to be summarised.
        base_dir: Union[str, Path]
            Base directory from which all paths are relative to.
        csv_file: Union[str, Path]
            CSV file of data to be summarised.
        stat_to_sum: str
            Variable to summarise.
        molecule_id: str
            Variable that uniquely identifies molecules.
        image_id: str
            Variable that uniquely identifies images.
        hist: bool
            Whether to plot histograms.
        stat: str
            Statistic to plot on histogram 'count' (default), 'freq'.
        bins: int
            Number of bins to plot.
        kde: bool
            Whether to include a Kernel Density Estimate.
        cut: float = 20,
            Cut point for KDE.
        figsize: tuple
            Figure dimensions.
        alpha: float
            Opacity to use in plots.
        palette: str = "deep"
            Seaborn colour plot to use.
        file_ext: str
            File type to save plots as 'png' (default), 'pdf', 'svg'.
        output_dir: Union[str, Path]
            Location to save plots to.
        var_to_label: dict
            Variable to label dictionary for automatically adding titles to plots.
        hue: str
            Dataframe column to group plots by.

        Returns
        =======
        """
        self.df = df if df is not None else pd.read_csv(csv_file)
        self.base_dir = base_dir
        self.stat_to_sum = stat_to_sum
        self.molecule_id = molecule_id
        self.image_id = image_id
        self.hist = hist
        self.bins = bins
        self.stat = stat
        self.kde = kde
        self.cut = cut
        self.figsize = figsize
        self.alpha = alpha
        self.palette = palette
        self.file_ext = file_ext
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.var_to_label = var_to_label
        self.hue = hue
        self.melted_data = None
        self.summary_data = None
        self.label = None

        # melt the data given in the init method
        self.melted_data = self.melt_data(self.df, stat_to_summarize=self.stat_to_sum, var_to_label=self.var_to_label)
        TopoSum.convert_basename_to_relative_paths(df=self.melted_data)
        self.set_palette()
        self._set_label(self.stat_to_sum)

    def _setup_figure(self):
        """Setup Matplotlib figure and axes."""
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        return fig, ax

    def _outfile(self, plot_suffix: str) -> str:
        """Generate the output file name with the appropriate suffix.

        Parameters
        ----------
        plot_suffix: str
            The suffix to append to the output file.

        Returns
        -------
        str:
            Concanenated string of the outfile and plot_suffix.
        """
        return f"{self.stat_to_sum}_{plot_suffix}"

    def sns_plot(self) -> None:
        """Plot the distribution of one or more statistics as either histogram, kernel density estimates or both. Uses
        base Seaborn."""
        fig, ax = self._setup_figure()
        if self.hist and not self.kde:
            outfile = self._outfile("hist")
            sns.histplot(data=self.melted_data, x="value", bins=self.bins, stat=self.stat, hue=self.hue)
        if self.kde and not self.hist:
            outfile = self._outfile("kde")
            sns.kdeplot(data=self.melted_data, x="value", hue=self.hue)
        if self.hist and self.kde:
            outfile = self._outfile("hist_kde")
            sns.histplot(
                data=self.melted_data,
                x="value",
                bins=self.bins,
                stat=self.stat,
                hue=self.hue,
                kde=True,
                kde_kws={"cut": self.cut},
            )
        plt.ticklabel_format(axis="both", style="sci", scilimits=(-3, 3))
        plt.title(self.label)
        self.set_xlim()
        self.save_plot(outfile)
        return fig, ax

    def sns_violinplot(self) -> None:
        """Violin plot of data."""
        fig, ax = self._setup_figure()
        sns.violinplot(data=self.melted_data, x=self.hue, y="value", hue=self.hue, alpha=self.alpha)
        plt.title(self.label)
        plt.xlabel("directory")
        plt.ylabel(self.label)
        outfile = self._outfile("violin")
        self.save_plot(outfile)
        return fig, ax

    # def sns_jointplot(self, var1: str, var2: str) -> None:
    #     """Joint distribution of two variables."""
    #     fig, ax = self._setup_figure()
    #     sns.jointplot(data=self.df, x=var1, y=var2, kind="reg")
    #     outfile = f"{'_'.join(self.stats_to_sum.keys())}_jointplot"
    #     # outfile = self._outfile("jointplot")
    #     self.save_plot(outfile)
    #     return fig, ax

    @staticmethod
    def melt_data(df: pd.DataFrame, stat_to_summarize: str, var_to_label: dict) -> pd.DataFrame:
        """Melt a dataframe into long format for plotting with Seaborn."""
        melted_data = pd.melt(df.reset_index(), id_vars=["molecule_number", "basename"], value_vars=stat_to_summarize)
        melted_data["variable"] = melted_data["variable"].map(var_to_label)
        # melted_data.rename({"image": "Image"}, axis=1, inplace=True)
        # self.image_id = "Image"
        LOGGER.info("[plotting] Data has been melted to long format for plotting.")

        return melted_data

    def set_xlim(self, percent: float = 0.1) -> None:
        """Set the range of the x-axis.

        Parameters
        ----------
        percent: float
            Percentage of the observed range by which to extend the x-axis. Only used if supplied range is outside the
        observed values.
        """
        range_percent = percent * (self.melted_data["value"].max() - self.melted_data["value"].min())
        range_min = self.melted_data["value"].min()
        range_max = self.melted_data["value"].max()
        plt.xlim(range_min - range_percent, range_max + range_percent)
        LOGGER.info(f"[plotting] Setting x-axis range       : {range_min} - {range_max}")

    def set_palette(self):
        """Set the color palette."""
        sns.set_palette(self.palette)
        LOGGER.info(f"[plotting] Seaborn color palette : {self.palette}")

    def save_plot(self, outfile: Path) -> None:
        """Save the plot to the output_dir

        Parameters
        ----------
        outfile: str
            Output file name to save figure to.
        """
        plt.savefig(self.output_dir / f"{outfile}.{self.file_ext}")
        LOGGER.info(
            f"[plotting] Plotted {self.stat_to_sum} to : " f"{str(self.output_dir / f'{outfile}.{self.file_ext}')}"
        )

    def _set_label(self, var: str):
        """Get the label based on the column name(s).

        Parameters
        ----------
        var: str
            The variable for which a label is required.
        """
        self.label = self.var_to_label[var]
        LOGGER.debug(f"[plotting] self.label     : {self.label}")

    @staticmethod
    def get_relative_paths(paths: List[Path]) -> List[str]:
        """From a list of paths, create a list of these paths but where
        each path is relative to all path's closest common parent. For
        example, ['a/b/c', 'a/b/d', 'a/b/e/f'] would return ['c', 'd', 'e/f']

        Parameters
        ----------
        paths: list
            List of string or pathlib paths.

        Returns
        -------
        relative_paths: list
            List of string paths, relative to the common parent.
        """

        # Ensure paths are all pathlib paths, and not strings
        paths = [Path(path) for path in paths]

        # If the paths list consists of all the same path, then the relative path will
        # be '.', which we don't want. we want the relative path to be the full path probably.
        # len(set(my_list)) == 1 determines if all the elements in a list are the same.
        if len(set(paths)) == 1:
            return [str(path.as_posix()) for path in paths]

        deepest_common_path = os.path.commonpath(paths)
        # Have to convert to strings else the dataframe values will be slightly different
        # to what is expected.
        return [str(path.relative_to(deepest_common_path).as_posix()) for path in paths]

    @staticmethod
    def convert_basename_to_relative_paths(df: pd.DataFrame):
        """Converts the paths in the 'basename' column in a dataframe from being
        absolute paths, to paths relative to the deepest common parent. For example
        if the 'basename' column has the following paths: ['/usr/topo/data/a/b', '/usr
        /topo/data/c/d'], the output will be: ['a/b', 'c/d'].

        Parameters
        ----------
        df: pd.DataFrame
            A pandas dataframe containing a column 'basename' which contains the paths
            indicating the locations of the image data files.

        Returns
        -------
        df: pd.DataFrame
            A pandas dataframe where the 'basename' column has paths relative to a common
            parent.
        """

        paths = df["basename"].tolist()
        paths = [Path(path) for path in paths]
        relative_paths = TopoSum.get_relative_paths(paths=paths)
        df["basename"] = relative_paths

        return df


def toposum(config: dict) -> Dict:
    """Process plotting and summarisation of data.

    Parameters
    ----------
    config: dict
        Dictionary of summarisation options.

    Returns
    -------
    Dict
        Dictionary of nested dictionaries. Each variable has its own dictionary with keys 'dist' and 'violin' which
        contain distribution like plots and violin plots respectively (if the later are required). Each 'dist' and
       'violin' is itself a dictionary with two elements 'figures' and 'axes' which correspond to MatplotLib 'fig' and
       'ax' for that plot.
    """
    if "df" not in config.keys():
        config["df"] = pd.read_csv(config["csv_file"])
    if config["df"].isna().values.all():
        LOGGER.info("[plotting] No statistics in DataFrame. Exiting...")
        return None
    violin = config.pop("violin")
    all_stats_to_sum = config.pop("stats_to_sum")
    pickle_plots = config.pop("pickle_plots")
    figures = defaultdict()

    # Plot each variable on its own graph
    for var in all_stats_to_sum:
        if var in config["df"].columns:
            topo_sum = TopoSum(stat_to_sum=var, **config)
            figures[var] = {"dist": None, "violin": None}
            figures[var]["dist"] = defaultdict()
            figures[var]["dist"]["figure"], figures[var]["dist"]["axes"] = topo_sum.sns_plot()

            if violin:
                figures[var]["violin"] = defaultdict()
                figures[var]["violin"]["figure"], figures[var]["violin"]["axes"] = topo_sum.sns_violinplot()
        else:
            LOGGER.info(f"[plotting] Statistic is not in dataframe : {var}")
    if pickle_plots:
        outfile = Path(config["output_dir"]) / "distribution_plots.pkl"
        save_pkl(outfile=outfile, to_pkl=figures)
        LOGGER.info(f"[plotting] Images pickled to : {outfile}")

    return figures


def main(args=None):
    """Run Plotting"""

    parser = create_parser()
    args = parser.parse_args() if args is None else parser.parse_args(args)

    if args.config_file is not None:
        config = read_yaml(args.config_file)
        LOGGER.info(f"[plotting] Configuration file loaded from : {args.config_file}")
    else:
        summary_yaml = pkg_resources.open_text(__package__, "summary_config.yaml")
        config = yaml.safe_load(summary_yaml.read())
        LOGGER.info("[plotting] Default configuration file loaded.")
    config = update_config(config, args)
    if args.var_to_label is not None:
        config["var_to_label"] = read_yaml(args.var_to_label)
        LOGGER.info("[plotting] Variable to labels mapping loaded from : {args.var_to_label}")
    else:
        plotting_yaml = pkg_resources.open_text(__package__, "var_to_label.yaml")
        config["var_to_label"] = yaml.safe_load(plotting_yaml.read())
        LOGGER.info("[plotting] Default variable to labels mapping loaded.")
    if args.csv_file is not None:
        config["csv_file"] = args.csv_file

    # Write sample configuration if asked to do so and exit
    if args.create_config_file:
        write_yaml(
            config,
            output_dir="./",
            config_file=args.create_config_file,
            header_message="Sample configuration file auto-generated",
        )
        LOGGER.info(f"A sample configuration has been written to : ./{args.create_config_file}")
        LOGGER.info(
            "Please refer to the documentation on how to use the configuration file : \n\n"
            "https://afm-spm.github.io/TopoStats/usage.html#configuring-topostats\n"
            "https://afm-spm.github.io/TopoStats/configuration.html"
        )
        sys.exit()
    if args.create_label_file:
        write_yaml(
            config["var_to_label"],
            output_dir="./",
            config_file=args.create_label_file,
            header_message="Sample label file auto-generated",
        )
        LOGGER.info(f"A sample label file has been written to : ./{args.create_label_file}")
        LOGGER.info(
            "Please refer to the documentation on how to use the configuration file : \n\n"
            "https://afm-spm.github.io/TopoStats/usage.html#configuring-topostats\n"
            "https://afm-spm.github.io/TopoStats/configuration.html"
        )
        sys.exit()

    # Plot statistics
    toposum(config)


if __name__ == "__main__":
    main()
