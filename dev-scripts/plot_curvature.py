import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    from itertools import combinations
    from pathlib import Path

    import h5py
    import marimo as mo
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from scipy import stats

    from topostats.io import LoadScans
    from topostats.plottingfuncs import Colormap

    CMAP = Colormap().get_cmap()
    VMIN = -3
    VMAX = 4
    PLOTTINGARGS = {"cmap": CMAP, "vmin": VMIN, "vmax": VMAX}
    return (
        LoadScans,
        PLOTTINGARGS,
        Path,
        combinations,
        mo,
        mpl,
        np,
        pd,
        plt,
        sns,
        stats,
    )


@app.cell
def _(Path, plt):
    dir_base = Path(
        "/Volumes/shared/pyne_group/Shared/AFM_Data/Plasmids/pICoZ/20260105_20251031_20251107_combined_picoz_dataset/output"
    )
    assert dir_base.exists()
    dir_output_plots = Path(
        "/Volumes/shared/pyne_group/Shared/Papers/TopoStats v2/topostats_2/figures/fig-dataset-separation"
    )
    assert dir_output_plots.exists()

    plt.rcParams["font.family"] = "Arial"
    fig_axes_label_font_size = 20
    fig_axes_tick_font_size = 20
    fig_axes_legend_text_font_size = 16
    fig_axes_legend_title_font_size = 18

    sample_groups = [
        ["Supercoiled", "Relaxed"],
        ["AT rich insert", "Telomeric insert"],
    ]
    return (
        dir_base,
        dir_output_plots,
        fig_axes_label_font_size,
        fig_axes_legend_text_font_size,
        fig_axes_legend_title_font_size,
        fig_axes_tick_font_size,
        sample_groups,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### load and sanitise dataframe values
    """)
    return


@app.cell
def _(dir_base, pd, plt, sns):
    # load csv
    df_data = pd.read_csv(dir_base / "grain_statistics.csv")
    assert df_data is not None

    # define picoz versions
    picoz_species_to_name = {
        "SCpicoz": "Supercoiled",
        "nicked": "Relaxed",
        "3ATpicoz": "AT rich insert",
        "tel80picoz": "TEL80 insert",
        "TEL12": "Telomeric insert",
    }
    picoz_colors = ["#D81B60", "#1E88E5", "#FFC107", "#004D40", "#D35FB7"]
    picoz_excluded_species = ["TEL80 insert"]

    expected_length_nm = 408e-9
    length_cutoff_percentage = 0.90
    length_cutoff_nm = expected_length_nm * length_cutoff_percentage


    def find_species(image_name: str, picoz_species_to_name: dict[str, str]):
        for species_raw_name in list(picoz_species_to_name.keys()):
            if species_raw_name in str(image_name):
                return picoz_species_to_name[species_raw_name]
        return "other"


    df_data["species"] = df_data["image"].apply(
        find_species, picoz_species_to_name=picoz_species_to_name
    )

    print(f"columns:\n{df_data.columns}")

    # drop any grains with nan values for curvature
    df_data = df_data.dropna(subset=["total_contour_length", "curvature_mean"])
    # drop any excluded species
    df_data = df_data[~df_data["species"].isin(picoz_excluded_species)]

    # Update the contour length to be in nanometres
    df_data["total_contour_length_nm"] = df_data["total_contour_length"] * 1e9

    # Drop rows that have contour length lower than the minimum
    _fig, _ax = plt.subplots(2, 1)
    fig_min_x = df_data["total_contour_length"].min() * 0.9
    fig_max_x = df_data["total_contour_length"].max() * 1.1
    ns_before = df_data["species"].value_counts()
    sns.kdeplot(data=df_data, x="total_contour_length", ax=_ax[0])
    df_data = df_data[df_data["total_contour_length"] >= length_cutoff_nm]
    sns.kdeplot(data=df_data, x="total_contour_length", ax=_ax[1])
    _ax[0].set_xlim(fig_min_x, fig_max_x)
    _ax[1].set_xlim(fig_min_x, fig_max_x)
    _fig.tight_layout()
    plt.show()

    ns_after = df_data["species"].value_counts()

    print(
        f"ns before contour length filtering ({length_cutoff_nm / 1e-9:.2f} nm):\n{ns_before}"
    )
    print(
        f"ns after contour length filtering ({length_cutoff_nm / 1e-9:.2f} nm):\n{ns_after}"
    )
    return df_data, picoz_colors


@app.cell
def _(combinations, mpl, np, pd, stats):
    def normality_test(data: pd.Series) -> bool:
        """Return true if Shapiro-Wilk test indicates normality with 0.05 CI."""
        return stats.shapiro(data)[1] > 0.05


    def perform_t_test(
        df: pd.DataFrame, sample_type_1: str, sample_type_2: str, value_column: str
    ) -> tuple[str, float]:
        """Return test name and p-value for a t-test between two sample types."""
        data_per_sample = {
            sample_type_1: df[value_column][df["species"] == sample_type_1],
            sample_type_2: df[value_column][df["species"] == sample_type_2],
        }
        # Check for nans
        if any(np.isnan(data_per_sample[sample_type_1])) or any(
            np.isnan(data_per_sample[sample_type_2])
        ):
            raise ValueError("There are nans in the dataset.")
        # If either are not normal, use a non-parametric test, eg Mann-Whitney
        # Else, use a t-test
        if not normality_test(
            data_per_sample[sample_type_1]
        ) or not normality_test(data_per_sample[sample_type_2]):
            test_name = "Mann-Whitney"
            stat, p = stats.mannwhitneyu(
                data_per_sample[sample_type_1], data_per_sample[sample_type_2]
            )
        else:
            test_name = "t-test"
            stat, p = stats.ttest_ind(
                data_per_sample[sample_type_1], data_per_sample[sample_type_2]
            )
        return test_name, p


    def perform_group_test(
        df: pd.DataFrame, sample_types: list[str], value_column: str
    ) -> tuple[str, float]:
        """Return test name and p-value for multi-sample-type (>3) comparison."""
        assert len(sample_types) >= 3, (
            f"Can only perform group tests against groups with 3 or more sample types, provided: {sample_types}."
        )
        data_per_sample = {
            sample_type: df[value_column][df["species"] == sample_type]
            for sample_type in sample_types
        }
        # If any are not normal, use a non-parametric test, eg Kruskal-Wallis
        if not all(
            normality_test(data_per_sample[sample_type])
            for sample_type in sample_types
        ):
            test_name = "Kruskal-Wallis"
            stat, p = stats.kruskal(
                *[data_per_sample[sample_type] for sample_type in sample_types]
            )
        else:
            test_name = "ANOVA"
            stat, p = stats.f_oneway(
                *[data_per_sample[sample_type] for sample_type in sample_types]
            )
        return test_name, p


    def perform_group_pairwise_test(
        df: pd.DataFrame,
        sample_types: list[str],
        value_column: str,
    ) -> list[tuple[str, str, str, float]]:
        pairs = list(combinations(sample_types, r=2))
        results: list[tuple[str, str, str, float]] = []
        for sample_a, sample_b in pairs:
            data_a = df[df["species"] == sample_a][value_column]
            data_b = df[df["species"] == sample_b][value_column]
            if normality_test(data_a) and normality_test(data_b):
                test_name = "t-test"
                stat, p = stats.ttest_ind(data_a, data_b)
            else:
                test_name = "Mann-Whitney"
                stat, p = stats.mannwhitneyu(data_a, data_b)
            results.append((sample_a, sample_b, test_name, p))
        return results


    def add_significance_to_current_plot(
        ax: mpl.axes.Axes,
        df: pd.DataFrame,
        sample_types: list[str],
        value_column: str,
        # per_sample_y_offset_increment: float = 0.05,
        global_y_offset_modifier: float = 0.0,
        fontsize: float = 12,
    ) -> None:
        test_results = perform_group_pairwise_test(
            df=df,
            sample_types=sample_types,
            value_column=value_column,
        )
        ylim = ax.get_ylim()
        y_max = ylim[1]
        y_range = ylim[1] - ylim[0]
        per_sample_y_offset_increment: float = y_range / 7
        for i, (sample_a, sample_b, test_name, p) in enumerate(test_results):
            xpos_sample_a = sample_types.index(sample_a)
            xpos_sample_b = sample_types.index(sample_b)
            ypos = (
                y_max
                + per_sample_y_offset_increment * (i + 1)
                - global_y_offset_modifier
            )
            ax.plot(
                [xpos_sample_a, xpos_sample_a, xpos_sample_b, xpos_sample_b],
                [
                    ypos,
                    ypos + per_sample_y_offset_increment / 2,
                    ypos + per_sample_y_offset_increment / 2,
                    ypos,
                ],
                lw=1.5,
                c="k",
            )
            if p < 0.001:
                text = "***"
            elif p < 0.01:
                text = "**"
            elif p < 0.05:
                text = "*"
            else:
                text = "ns"
            ax.text(
                (xpos_sample_a + xpos_sample_b) / 2,
                ypos + per_sample_y_offset_increment / 2,
                text,
                ha="center",
                va="bottom",
                fontsize=fontsize,
            )

    return add_significance_to_current_plot, perform_group_test, perform_t_test


@app.cell
def _(
    df_data,
    dir_output_plots,
    fig_axes_label_font_size,
    fig_axes_tick_font_size,
    pd,
    plt,
    sample_groups,
    sns,
):
    def contour_lengths(
        df: pd.DataFrame,
        groups: list[list[str]],
    ) -> None:
        for group in groups:
            df_group = df[df["species"].isin(group)]
            fig, ax = plt.subplots(figsize=(6, 6))
            sns.boxplot(
                x="species",
                y="total_contour_length_nm",
                data=df_group,
                showfliers=False,
                linewidth=1.5,
                # palette=picoz_colors,
                # hue="species",
            )
            sns.stripplot(
                data=df_group,
                x="species",
                y="total_contour_length_nm",
                color="black",
                jitter=True,
                size=2,
                alpha=0.7,
            )

            # plt.xlabel("pICOz variant")
            plt.xlabel("")
            plt.ylabel("Measured length (nm)", fontsize=fig_axes_label_font_size)
            plt.ylim(320, 500)  # Adjust y-axis limits as needed
            # plt.title("Total Contour Length by pICOz Variant")
            axes_linewidth = 2
            ax.spines["top"].set_linewidth(axes_linewidth)
            ax.spines["right"].set_linewidth(axes_linewidth)
            ax.spines["left"].set_linewidth(axes_linewidth)
            ax.spines["bottom"].set_linewidth(axes_linewidth)
            ax.tick_params(
                axis="both",
                which="major",
                labelsize=fig_axes_tick_font_size,
            )
            ax.tick_params(
                axis="x",
                rotation=45,
            )

            sns.despine()
            fig.tight_layout()
            plt.savefig(dir_output_plots / f"contour-length_{group}.png")
            plt.show()


    contour_lengths(df=df_data, groups=sample_groups)
    return


@app.cell
def _(
    add_significance_to_current_plot,
    df_data,
    dir_base,
    dir_output_plots,
    fig_axes_label_font_size,
    fig_axes_tick_font_size,
    np,
    pd,
    picoz_colors,
    plt,
    sample_groups,
    sns,
):
    from matplotlib.pylab import ylim


    def plot_group_distributions(
        df: pd.DataFrame,
        metrics: list[tuple[str, str]],
        groups: list[list[str]],
        palette,
        save_dir: str = None,
        significance_global_y_offset_modifier: float = 0.0,
        figsize: tuple[int, int] = (20, 10),
        fontsize_significance: float = 12,
        fontsize_title: float = 12,
    ) -> None:
        for group in groups:
            data_group = df[df["species"].isin(group)]
            fig, axs = plt.subplots(
                len(metrics) // 2 + len(metrics) % 2,
                2,
                figsize=(figsize[0], figsize[1] * len(metrics) // 2),
            )
            for index, (parameter_name, plot_title) in enumerate(metrics):
                if len(axs.shape) > 1:
                    ax = axs[index // 2, index % 2]
                else:
                    ax = axs[index % 2]
                print(f"plotting {parameter_name}, titled: {plot_title}")
                sns.boxplot(
                    x="species",
                    y=parameter_name,
                    data=data_group,
                    # palette=palette,
                    # hue="species",
                    ax=ax,
                )
                sns.stripplot(
                    x="species",
                    y=parameter_name,
                    data=data_group,
                    color="black",
                    alpha=0.4,
                    jitter=True,
                    # palette=palette,
                    # hue="species",
                    ax=ax,
                )
                fig.tight_layout()
                add_significance_to_current_plot(
                    ax=ax,
                    df=df,
                    sample_types=group,
                    value_column=parameter_name,
                    global_y_offset_modifier=significance_global_y_offset_modifier,
                    fontsize=fontsize_significance,
                )
                ax.set_xlabel("pICOz variant")
                ax.set_ylabel(plot_title)
                ax.set_title(
                    f"Grain {plot_title} by species for {group}",
                    fontsize=fontsize_title,
                )
                sns.despine(ax=ax)
            if save_dir is not None:
                plt.savefig(save_dir / f"stats_plots_{group}.png")
            plt.show()


    metrics = [
        # ("curvature_mean", "Mean curvature"),
        # ("curvature_std", "Standard deviation of curvature"),
        # ("curvature_var", "Variance of curvature"),
        ("curvature_median", "Median curvature"),
        ("num_crossings", "Number of crossings"),
        ("curvature_iqr", "Interquartile range of curvature"),
        # ("curvature_total", "Total curvature"),
        # ("curvature_max", "Maximum curvature"),
        # ("curvature_min", "minimum_curvature"),
        ("curvature_90th", "90th percentile of curvature"),
        ("num_turns", "Number of turns"),
    ]

    plot_group_distributions(
        df=df_data,
        metrics=metrics,
        groups=sample_groups,
        palette=picoz_colors,
        save_dir=dir_base,
        figsize=(10, 5),
        fontsize_significance=8,
        fontsize_title=10,
    )


    # Plot individual figures
    def plot_individual_parameter_stripboxes(
        df: pd.DataFrame,
        groups: list[list[str]],
        metrics: list[tuple[str, str]],
        box_line_width: float = 1,
        fontsize_multiplier: float = 2,
        axes_linewidth: float = 1,
        significance_font_size: float = 16,
    ) -> None:
        _fig, _ax = plt.subplots()
        for group in groups:
            data_group = df_data[df_data["species"].isin(group)]
            for parameter_name, plot_title in metrics:
                for species in group:
                    data_species = data_group[data_group["species"] == species]
                    data_species_parameter = data_species[parameter_name]
                    print(f"median {parameter_name} for {species}: {np.median(data_species_parameter)}")
                fig, ax = plt.subplots(figsize=(5, 8))
                sns.boxplot(
                    x="species",
                    y=parameter_name,
                    data=data_group,
                    # palette=palette,
                    # hue="species",
                    ax=ax,
                    linewidth=box_line_width,
                )
                sns.stripplot(
                    x="species",
                    y=parameter_name,
                    data=data_group,
                    color="black",
                    alpha=0.4,
                    jitter=True,
                    # palette=palette,
                    # hue="species",
                    ax=ax,
                )
                ax.tick_params("x", rotation=45, labelsize=fig_axes_tick_font_size * fontsize_multiplier)
                ax.tick_params("y", labelsize=fig_axes_tick_font_size * fontsize_multiplier)
                ax.set_ylabel(plot_title, fontsize=fig_axes_label_font_size * fontsize_multiplier)
                ax.set_xlabel("")
                ax.spines["left"].set_linewidth(axes_linewidth)
                ax.spines["bottom"].set_linewidth(axes_linewidth)
                sns.despine(ax=ax)
                fig.tight_layout()
                add_significance_to_current_plot(
                    ax=ax,
                    df=df_data,
                    sample_types=group,
                    value_column=parameter_name,
                    global_y_offset_modifier=0,
                    fontsize=significance_font_size,
                )
                fig.savefig(dir_output_plots / f"{parameter_name}_{group}.png")
                plt.show()


    plot_individual_parameter_stripboxes(
        df=df_data,
        groups=sample_groups,
        metrics=[
            ("curvature_median", "Median curvature"),
            ("curvature_iqr", "Interquartile range of curvature"),
            ("curvature_90th", "90th percentile of curvature"),
        ],
        box_line_width=2,
        fontsize_multiplier=1.5,
        axes_linewidth=2,
        significance_font_size=30,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Stats tests
    """)
    return


@app.cell
def _(df_data, pd, perform_group_test, perform_t_test, sample_groups):
    def perform_stats_tests_on_groups(
        sample_groups: list[list[str]],
        df: pd.DataFrame,
        value_columns: str,
    ) -> None:

        for value_column in value_columns:
            for sample_group in sample_groups:
                if len(sample_group) == 2:
                    test, p = perform_t_test(
                        df=df,
                        sample_type_1=sample_group[0],
                        sample_type_2=sample_group[1],
                        value_column=value_column,
                    )
                    print(
                        f"t test for [{value_column}] for {sample_group}: {test}, {p:.3e}"
                    )
                else:
                    test, p = perform_group_test(
                        df=df,
                        sample_types=sample_group,
                        value_column=value_column,
                    )
                    print(
                        f"group test for [{value_column}] for {sample_group}: {test}, {p:.3e}"
                    )
                    print()


    perform_stats_tests_on_groups(
        sample_groups=sample_groups,
        df=df_data,
        value_columns=["curvature_iqr", "curvature_median"],
    )
    return


@app.cell
def _(
    df_data,
    dir_output_plots,
    fig_axes_label_font_size,
    fig_axes_legend_text_font_size,
    fig_axes_legend_title_font_size,
    fig_axes_tick_font_size,
    np,
    pd,
    plt,
    sample_groups,
    sns,
):
    # Stacked bar chart of number of crossings
    # stacked bar chart of % num_crossings rather than counts


    def stacked_bar_chart(
        df: pd.DataFrame,
        groups: list[list[str]],
    ):
        for group in sample_groups:
            df_data_filtered = df[df["species"].isin(group)]
            fig, ax = plt.subplots(figsize=(6, 6))
            # convert the dataframe to just be number of counts using groupby. groupby(x).size() returns a series with
            # multiple indices, where the first index is the groupby column and the second index is the value column
            # We then need to use unstack to convert the second index to columns, and fill_value=0 to fill in any missing
            # values with 0 since the series doesn't require each group to have all possible values
            # might want to consider using multiindex series elsewhere, since we often have data that doesn't have all values for
            # each group?
            df_num_crossings = df_data_filtered.groupby(
                ["species", "num_crossings"]
            ).size()
            # print(" --- Grouped counts --- ")
            # print(df_num_crossings)
            # print("--- Unstacked with fill_value=0 ---")
            df_counts_crossings = df_num_crossings.unstack(fill_value=0)
            # print(df_counts_crossings)
            # divide by the row sums to get percentages
            df_percents = df_counts_crossings.div(
                df_counts_crossings.sum(axis=1), axis=0
            )
            df_percents = df_percents.reindex(group)
            # df_percents = df_percents.reindex(order) * 100
            df_percents = df_percents * 100
            # print(f"\npercentages:\n {df_percents}")

            df_percents.plot.bar(stacked=True, ax=ax, width=0.7, colormap="Blues")
            # _ax.set_xticks(ticks=list(range(len(group) + 2)))
            ax.set_ylabel("Percentage", fontsize=fig_axes_label_font_size)
            ax.set_xlabel("", fontname="Arial")
            # line thickness for axes thicker
            axes_linewidth = 2
            ax.spines["top"].set_linewidth(axes_linewidth)
            ax.spines["right"].set_linewidth(axes_linewidth)
            ax.spines["left"].set_linewidth(axes_linewidth)
            ax.spines["bottom"].set_linewidth(axes_linewidth)
            # make y ticks be integers only
            ax.set_yticks(ticks=np.arange(0, 110, 10))
            # text size
            ax.tick_params(
                axis="both", which="major", labelsize=fig_axes_tick_font_size
            )
            ax.tick_params("x", rotation=45)
            # legend
            ax.legend(
                title="No. crossings",
                title_fontsize=fig_axes_legend_title_font_size,
                fontsize=fig_axes_legend_text_font_size,
                loc="upper left",
                frameon=False,
                bbox_to_anchor=(1, 1),
            )
            sns.despine()
            fig.tight_layout()
            plt.savefig(
                dir_output_plots / f"percentage_crossings_{str(group)}.png"
            )
            plt.show()


    stacked_bar_chart(
        df=df_data,
        groups=sample_groups,
    )
    return


@app.function
def pad_bounding_box_dynamically_at_limits(
    bbox: tuple[int, int, int, int],
    limits: tuple[int, int, int, int],
    padding: int,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """
    Pad a bounding box within limits. If the padding would exceed the limits bounds, pad in the other direction.

    Parameters
    ----------
    bbox : tuple[int, int, int, int]
        The bounding box to pad.
    limits : tuple[int, int, int, int]
        The region to limit the bounding box to in the form (min_row, min_col, max_row, max_col).
    padding : int
        The padding to apply to the bounding box.

    Returns
    -------
    tuple[[tuple[int, int, int, int], tuple[int, int, int, int]]
        The new bounding box indices and the amount they have been padded by.
    """
    # check that the padded size is smaller than the limits
    bbox_height = bbox[2] - bbox[0]
    bbox_width = bbox[3] - bbox[1]
    proposed_height = bbox_height + 2 * padding
    proposed_width = bbox_width + 2 * padding
    limits_height = limits[2] - limits[0]
    limits_width = limits[3] - limits[1]
    if proposed_height > limits_height or proposed_width > limits_width:
        raise ValueError(
            f"Proposed size {proposed_height}x{proposed_width} px = ({bbox_width}x{bbox_height}) + "
            f"({2 * padding}x{2 * padding}) px is larger than limits size "
            f"({limits_height}x{limits_width}) px. Cannot pad bounding box beyond limits."
        )
    pad_up_amount = padding
    pad_down_amount = padding
    pad_left_amount = padding
    pad_right_amount = padding
    # try padding up, check if hit the top of the limits
    if bbox[0] - padding < limits[0]:
        # if so, restrict up padding to the limits and add the remaining padding to the down padding
        pad_up_amount = bbox[0] - limits[0]
        # Can safely assume can increase down padding since we checked earlier that the proposed size is smaller than
        # limits
        pad_down_amount += padding - pad_up_amount
    # try padding down, check if hit the bottom of the limits
    elif bbox[2] + padding > limits[2]:
        # if so, restrict down padding to the limits and add the remaining padding to the up padding
        pad_down_amount = limits[2] - bbox[2]
        # Can safely assume can increase up padding since we checked earlier that the proposed size is smaller than
        # limits
        pad_up_amount += padding - pad_down_amount
    # try padding left, check if hit the left of the limits
    if bbox[1] - padding < limits[1]:
        # if so, restrict left padding to the limits and add the remaining padding to the right padding
        pad_left_amount = bbox[1] - limits[1]
        # Can safely assume can increase right padding since we checked earlier that the proposed size is smaller than
        # limits
        pad_right_amount += padding - pad_left_amount
    # try padding right, check if hit the right of the limits
    elif bbox[3] + padding > limits[3]:
        # if so, restrict right padding to the limits and add the remaining padding to the left padding
        pad_right_amount = limits[3] - bbox[3]
        # Can safely assume can increase left padding since we checked earlier that the proposed size is smaller than
        # limits
        pad_left_amount += padding - pad_right_amount
    # Return the new bounding box indices
    return (
        (
            bbox[0] - pad_up_amount,
            bbox[1] - pad_left_amount,
            bbox[2] + pad_down_amount,
            bbox[3] + pad_right_amount,
        ),
        (
            pad_up_amount,
            pad_left_amount,
            pad_down_amount,
            pad_right_amount,
        )
   )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # plot curvature zoom-ins
    """)
    return


@app.cell
def _(LoadScans, PLOTTINGARGS, Path, dir_base, dir_output_plots, mpl, np, plt):
    # find the files
    def sample_files(
        directory: Path,
        file_index: int,
    ) -> Path:
        all_matching_files = sorted(list(directory.glob("*.topostats")))
        return all_matching_files[file_index]

    sc_file = sample_files(dir_base / "sc" / "processed", file_index=0)
    nicked_file = sample_files(dir_base / "nicked" / "processed", file_index = 2)
    at_file = sample_files(dir_base / "3at" / "processed", file_index = 8)
    tel12_file = sample_files(dir_base / "20260310_TC_picoztel12" / "processed", file_index = 10)

    def plot_curvature(
        filepath: Path,
        linewidth: float,
        savepath: Path,
        colourmap_normalisation_bounds: tuple[float, float] | None,
        figsize: tuple[float, float] = (6, 6),
        crop_size_nm: int = 165
    ) -> None:
        loadscans = LoadScans(
            img_paths=[filepath],
            channel="dummy",
            extract="all"
        )
        loadscans.get_data()
        data_image = loadscans.img_dict[list(loadscans.img_dict.keys())[0]]
        print(data_image.keys())
        pixel_to_nm_scaling = data_image["pixel_to_nm_scaling"]
        curvatures_mol_0 = np.abs(data_image["grain_curvature_stats"]["above"]["grains"]["grain_0"]["molecules"]["mol_0"]["curvatures"])
        image = data_image["image"]
        mol_0_bbox = data_image["splining"]["above"]["grain_0"]["mol_0"]["bbox"]
        # pad the bbox
        mol_0_bbox_width = mol_0_bbox[2] - mol_0_bbox[0]
        to_pad = int((crop_size_nm / pixel_to_nm_scaling) - mol_0_bbox_width)
        print(f"to pad: {to_pad}")
        if to_pad < 0:
            raise ValueError(f"need to pad a negative amount: {to_pad} px. mol_0_bbox: {mol_0_bbox}, crop size nm: {crop_size_nm}")
        bbox_resized, pad_amounts = pad_bounding_box_dynamically_at_limits(
            mol_0_bbox,
            limits=[0, 0, image.shape[0], image.shape[1]],
            padding=to_pad//2,
        )
        splined_points_mol_0 = data_image["splining"]["above"]["grain_0"]["mol_0"]["spline_coords"]
        splined_points_mol_0 += (pad_amounts[0], pad_amounts[1])
        image_crop = image[bbox_resized[0]: bbox_resized[2], bbox_resized[1]: bbox_resized[3]]
        curvatures_mol_0_normalised = np.array(curvatures_mol_0)
        if colourmap_normalisation_bounds is not None:
            curvatures_mol_0_normalised = curvatures_mol_0_normalised - colourmap_normalisation_bounds[0]
            curvatures_mol_0_normalised = curvatures_mol_0_normalised / (
                colourmap_normalisation_bounds[1] - colourmap_normalisation_bounds[0]
            )

        fig, ax = plt.subplots(figsize=(figsize))
        image_crop = np.flipud(image_crop)
        print(f"crop is {image_crop.shape[0] * pixel_to_nm_scaling} nm")
        plt.imshow(image_crop, extent=(
                            0,
                            image_crop.shape[1] * pixel_to_nm_scaling,
                            0,
                            image_crop.shape[0] * pixel_to_nm_scaling,
                        ), **PLOTTINGARGS)
        # plt.plot(splined_points_mol_0[:, 1], splined_points_mol_0[:, 0])
        # plot the splined points with curvature

        cmap = mpl.cm.viridis
        for index, point in enumerate(splined_points_mol_0):
            colour = cmap(curvatures_mol_0_normalised[index])
            if index > 0:
                previous_point = splined_points_mol_0[index - 1]
                ax.plot(
                    [
                        previous_point[1] * pixel_to_nm_scaling,
                        point[1] * pixel_to_nm_scaling,
                    ],
                    [
                        previous_point[0] * pixel_to_nm_scaling,
                        point[0] * pixel_to_nm_scaling,
                    ],
                    color=colour,
                    linewidth=linewidth,
                )

        fig.savefig(savepath)
        plt.show()

    CURVATURE_NORM_BOUNDS = (0,0.3)
    plot_curvature(filepath=sc_file, linewidth=3, savepath=dir_output_plots / "curvatures_sc.png", colourmap_normalisation_bounds=CURVATURE_NORM_BOUNDS)
    plot_curvature(filepath=nicked_file, linewidth=3, savepath=dir_output_plots / "curvatures_nicked.png", colourmap_normalisation_bounds=CURVATURE_NORM_BOUNDS)
    plot_curvature(filepath=at_file, linewidth=3, savepath=dir_output_plots / "curvatures_at.png", colourmap_normalisation_bounds=CURVATURE_NORM_BOUNDS)
    plot_curvature(filepath=tel12_file, linewidth=3, savepath=dir_output_plots / "curvatures_tel12.png", colourmap_normalisation_bounds=CURVATURE_NORM_BOUNDS)
    return


if __name__ == "__main__":
    app.run()
