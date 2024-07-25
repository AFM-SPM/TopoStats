"""Plotting and summary of TopoStats output statistics."""

from __future__ import annotations

from collections import defaultdict

import importlib.resources as pkg_resources
import logging
from pathlib import Path
import sys
import yaml
import matplotlib.pyplot as plt
import matplotlib.colors
import pandas as pd
import seaborn as sns
import numpy as np

from topostats.io import read_yaml, save_pkl, write_yaml, convert_basename_to_relative_paths
from topostats.logs.logs import LOGGER_NAME
from topostats.utils import update_config
from topostats.theme import Colormap

LOGGER = logging.getLogger(LOGGER_NAME)


# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-locals


class TopoSum:
    """
    Class for summarising grain statistics in plots.

    Parameters
    ----------
    df :  pd.DataFrame
        Pandas data frame of data to be summarised.
    base_dir :  str | Path
        Base directory from which all paths are relative to.
    csv_file :  str | Path
        CSV file of data to be summarised.
    stat_to_sum :  str
        Variable to summarise.
    molecule_id :  str
        Variable that uniquely identifies molecules.
    image_id :  str
        Variable that uniquely identifies images.
    hist :  bool
        Whether to plot histograms.
    stat :  str
        Statistic to plot on histogram 'count' (default), 'freq'.
    bins :  int
        Number of bins to plot.
    kde :  bool
        Whether to include a Kernel Density Estimate.
    cut :  float = 20,
        Cut point for KDE.
    figsize :  tuple
        Figure dimensions.
    alpha :  float
        Opacity to use in plots.
    palette :  str = "deep"
        Seaborn colour plot to use.
    savefig_format :  str
        File type to save plots as 'png' (default), 'pdf', 'svg'.
    output_dir :  str | Path
        Location to save plots to.
    var_to_label :  dict
        Variable to label dictionary for automatically adding titles to plots.
    hue :  str
        Dataframe column to group plots by.
    """

    def __init__(
        self,
        df: pd.DataFrame = None,
        base_dir: str | Path = None,
        csv_file: str | Path = None,
        stat_to_sum: str = None,
        molecule_id: str = "grain_number",
        image_id: str = "image",
        hist: bool = True,
        stat: str = "count",
        bins: int = 12,
        kde: bool = True,
        cut: float = 20,
        figsize: tuple = (16, 9),
        alpha: float = 0.5,
        palette: str = "deep",
        savefig_format: str = "png",
        output_dir: str | Path = ".",
        var_to_label: dict = None,
        hue: str = "basename",
    ) -> None:
        """
        Initialise the class.

        Parameters
        ----------
        df :  pd.DataFrame
            Pandas data frame of data to be summarised.
        base_dir :  str | Path
            Base directory from which all paths are relative to.
        csv_file :  str | Path
            CSV file of data to be summarised.
        stat_to_sum :  str
            Variable to summarise.
        molecule_id :  str
            Variable that uniquely identifies molecules.
        image_id :  str
            Variable that uniquely identifies images.
        hist :  bool
            Whether to plot histograms.
        stat :  str
            Statistic to plot on histogram 'count' (default), 'freq'.
        bins :  int
            Number of bins to plot.
        kde :  bool
            Whether to include a Kernel Density Estimate.
        cut :  float = 20,
            Cut point for KDE.
        figsize :  tuple
            Figure dimensions.
        alpha :  float
            Opacity to use in plots.
        palette :  str = "deep"
            Seaborn colour plot to use.
        savefig_format :  str
            File type to save plots as 'png' (default), 'pdf', 'svg'.
        output_dir :  str | Path
            Location to save plots to.
        var_to_label :  dict
            Variable to label dictionary for automatically adding titles to plots.
        hue :  str
            Dataframe column to group plots by.
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
        self.savefig_format = savefig_format
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.var_to_label = var_to_label
        self.hue = hue
        self.melted_data = None
        self.summary_data = None
        self.label = None

        # melt the data given in the init method
        self.melted_data = self.melt_data(self.df, stat_to_summarize=self.stat_to_sum, var_to_label=self.var_to_label)
        convert_basename_to_relative_paths(df=self.melted_data)
        self.set_palette()
        self._set_label(self.stat_to_sum)

    def _setup_figure(self):
        """
        Setup Matplotlib figure and axes.

        Returns
        -------
        fig, ax
            Matplotlib fig and ax objects.
        """
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        return fig, ax

    def _outfile(self, plot_suffix: str) -> str:
        """
        Generate the output file name with the appropriate suffix.

        Parameters
        ----------
        plot_suffix : str
            The suffix to append to the output file.

        Returns
        -------
        str:
            Concanenated string of the outfile and plot_suffix.
        """
        return f"{self.stat_to_sum}_{plot_suffix}"

    def sns_plot(self) -> tuple[plt.Figure, plt.Axes] | None:
        """
        Plot the distribution of one or more statistics as either histogram, kernel density estimates or both.

        Uses base Seaborn.

        Returns
        -------
        Optional[Union[Tuple[plt.Figure, plt.Axes], None]]
            Tuple of Matplotlib figure and axes if plotting is successful, None otherwise.
        """

        # Note: Plotting KDEs with Seaborn is not possible if all values are the same.
        # This is because the KDE is calculated using a Gaussian kernel and if all values
        # are the same, the standard deviation is 0 which results in a ZeroDivisionError with
        # is caught internally but then raises a numpy linalg error.
        # The try/catch is there to catch this error and skip plotting KDEs if all values are the same.

        fig, ax = self._setup_figure()

        # If histogram is requested but KDE is not, plot histogram
        if self.hist and not self.kde:
            outfile = self._outfile("hist")
            sns.histplot(data=self.melted_data, x="value", bins=self.bins, stat=self.stat, hue=self.hue)
        if self.kde and not self.hist:
            outfile = self._outfile("kde")
            try:
                sns.kdeplot(data=self.melted_data, x="value", hue=self.hue)
            except np.linalg.LinAlgError:
                LOGGER.info(
                    "[plotting] KDE plot error: Numpy linalg error encountered. This is a result of all values \
for KDE plot being the same. KDE plots cannot be made as there is no variance, skipping."
                )
                return None
        if self.hist and self.kde:
            outfile = self._outfile("hist_kde")
            try:
                sns.histplot(
                    data=self.melted_data,
                    x="value",
                    bins=self.bins,
                    stat=self.stat,
                    hue=self.hue,
                    kde=True,
                    kde_kws={"cut": self.cut},
                )
            except np.linalg.LinAlgError:
                LOGGER.info(
                    "[plotting] KDE plot error: Numpy linalg error encountered. This is a result of all values \
for KDE plot being the same. KDE plots cannot be made as there is no variance, skipping."
                )
                return None

        plt.ticklabel_format(axis="both", style="sci", scilimits=(-3, 3))
        plt.title(self.label)
        self.set_xlim()
        self.save_plot(outfile)

        return fig, ax

    def sns_violinplot(self) -> None:
        """
        Violin plot of data.

        Returns
        -------
        fig, ax
            Matplotlib fig and ax objects.
        """
        fig, ax = self._setup_figure()
        # Determine whether to draw a legend
        legend = "full" if len(self.melted_data[self.hue].unique()) > 1 else False
        sns.violinplot(
            data=self.melted_data,
            x=self.hue,
            y="value",
            hue=self.hue,
            alpha=self.alpha,
            legend=legend,
        )
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
        """
        Melt a dataframe into long format for plotting with Seaborn.

        Parameters
        ----------
        df : pd.DataFrame
            Statistics to melt.
        stat_to_summarize : str
            Statistics to summarise.
        var_to_label : dict
            Mapping of variable names to descriptions.

        Returns
        -------
        pd.DataFrame
            Data in long-format with descriptive variable names.
        """
        melted_data = pd.melt(df.reset_index(), id_vars=["grain_number", "basename"], value_vars=stat_to_summarize)
        melted_data["variable"] = melted_data["variable"].map(var_to_label)
        LOGGER.info("[plotting] Data has been melted to long format for plotting.")

        return melted_data

    def set_xlim(self, percent: float = 0.1) -> None:
        """
        Set the range of the x-axis.

        Parameters
        ----------
        percent : float
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
        """
        Save the plot to the output_dir.

        Parameters
        ----------
        outfile : str
            Output file name to save figure to.
        """
        plt.savefig(self.output_dir / f"{outfile}.{self.savefig_format}")
        LOGGER.info(
            f"[plotting] Plotted {self.stat_to_sum} to : "
            f"{str(self.output_dir / f'{outfile}.{self.savefig_format}')}"
        )

    def _set_label(self, var: str):
        """
        Get the label based on the column name(s).

        Parameters
        ----------
        var : str
            The variable for which a label is required.
        """
        self.label = self.var_to_label[var]
        LOGGER.debug(f"[plotting] self.label     : {self.label}")


def toposum(config: dict) -> dict:
    """
    Process plotting and summarisation of data.

    Parameters
    ----------
    config : dict
        Dictionary of summarisation options.

    Returns
    -------
    dict
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
            result_option: tuple | None = topo_sum.sns_plot()
            # Handle the Optional[Tuple]
            if result_option is not None:
                figures[var]["dist"]["figure"], figures[var]["dist"]["axes"] = result_option

            if violin:
                figures[var]["violin"] = defaultdict()
                (
                    figures[var]["violin"]["figure"],
                    figures[var]["violin"]["axes"],
                ) = topo_sum.sns_violinplot()
        else:
            LOGGER.info(f"[plotting] Statistic is not in dataframe : {var}")
    if pickle_plots:
        outfile = Path(config["output_dir"]) / "distribution_plots.pkl"
        save_pkl(outfile=outfile, to_pkl=figures)
        LOGGER.info(f"[plotting] Images pickled to : {outfile}")

    return figures


def run_toposum(args=None) -> None:
    """
    Run Plotting.

    Parameters
    ----------
    args : None
        Arguments to pass and update configuration.
    """

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


def plot_crossing_linetrace_halfmax(branch_stats_dict: dict, cmap: matplotlib.colors.Colormap, title: str) -> tuple:
    """Plots the heightmap lines traces of the branches found in the 'branch_stats' dictionary, and their meetings.

    Parameters:
    -----------
    branch_stats_dict: dict
        Dictionary containing branch height, distance and fwhm2 info.
    """
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    cmp = Colormap(cmap).get_cmap()
    total_branches = len(branch_stats_dict)
    # plot the highest first
    fwhms = []
    for branch_idx, values in branch_stats_dict.items():
        fwhms.append(values["fwhm2"][0])
    branch_idx_order = np.array(list(branch_stats_dict.keys()))[np.argsort(np.array(fwhms))]

    for i, branch_idx in enumerate(branch_idx_order):
        fwhm, hm_vals, m_vals = branch_stats_dict[branch_idx]["fwhm2"]
        if total_branches == 1:
            cmap_ratio = 0
        else:
            cmap_ratio = i / (total_branches - 1)
        heights = branch_stats_dict[branch_idx]["heights"]
        x = branch_stats_dict[branch_idx]["distances"]
        ax.plot(x, heights, c=cmp(cmap_ratio)) # label=f"Branch: {branch_idx}"

        # plot the high point lines
        plt.plot([-15, m_vals[1]], [m_vals[2], m_vals[2]], c=cmp(cmap_ratio), label=f"FWHM: {fwhm:.4f}")
        # plot the half max lines
        plt.plot([hm_vals[0], hm_vals[0]], [hm_vals[2], heights.min()], c=cmp(cmap_ratio))
        plt.plot([hm_vals[1], hm_vals[1]], [hm_vals[2], heights.min()], c=cmp(cmap_ratio))

    ax.tick_params(axis='both', labelsize=20)
    #ax.set_xlabel("Distance from Node (nm)", fontsize="22")
    #ax.set_ylabel("Height", fontsize="22")
    #ax.set_title(title, fontsize="20")
    ax.legend(fontsize="16")
    return fig, ax