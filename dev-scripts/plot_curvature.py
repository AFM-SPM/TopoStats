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

    return Path, combinations, mo, mpl, np, pd, plt, sns, stats


@app.cell
def _(Path):
    dir_base = Path(
        "/Volumes/shared/pyne_group/Shared/AFM_Data/Plasmids/pICoZ/20260105_20251031_20251107_combined_picoz_dataset/output"
    )
    assert dir_base.exists()
    return (dir_base,)


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
def _(df_data, plt, sns):
    _fig, _ax = plt.subplots(figsize=(6, 6))
    sns.boxplot(
        x="species",
        y="total_contour_length_nm",
        data=df_data,
        showfliers=False,
        linewidth=1.5,
        # palette=picoz_colors,
        # hue="species",
    )
    sns.stripplot(
        data=df_data,
        x="species",
        y="total_contour_length_nm",
        color="black",
        jitter=True,
        size=2,
        alpha=0.7,
    )

    # plt.xlabel("pICOz variant")
    plt.xlabel("")
    plt.ylabel("Plasmid length (nm)", fontsize=16)
    plt.ylim(0, 600)  # Adjust y-axis limits as needed
    # plt.title("Total Contour Length by pICOz Variant")
    axes_linewidth = 2
    _ax.spines["top"].set_linewidth(axes_linewidth)
    _ax.spines["right"].set_linewidth(axes_linewidth)
    _ax.spines["left"].set_linewidth(axes_linewidth)
    _ax.spines["bottom"].set_linewidth(axes_linewidth)
    _ax.tick_params(axis="both", which="major", labelsize=16)
    sns.despine()
    plt.show()
    return (axes_linewidth,)


@app.cell
def _(
    add_significance_to_current_plot,
    df_data,
    dir_base,
    pd,
    picoz_colors,
    plt,
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

    groups = [
        ["Supercoiled", "Relaxed"],
        ["Supercoiled", "AT rich insert", "Telomeric insert"],
    ]

    plot_group_distributions(
        df=df_data,
        metrics=metrics,
        groups=groups,
        palette=picoz_colors,
        save_dir=dir_base,
        figsize=(10, 5),
        fontsize_significance=8,
        fontsize_title=10,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Stats tests
    """)
    return


@app.cell
def _(df_data, pd, perform_group_test, perform_t_test):
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
        sample_groups=[
            ["Supercoiled", "Relaxed"],
            ["Supercoiled", "AT rich insert", "Telomeric insert"],
        ],
        df=df_data,
        value_columns=["curvature_iqr", "curvature_median"],
    )
    return


@app.cell
def _(axes_linewidth, df_data, np, plt):
    # Stacked bar chart of number of crossings
    # stacked bar chart of % num_crossings rather than counts
    _groups = [
        ["Supercoiled", "Relaxed"],
        ["Supercoiled", "AT rich insert", "Telomeric insert"],
    ]

    for group in _groups:
        df_data_filtered = df_data[df_data["species"].isin(group)]
        _fig, _ax = plt.subplots(figsize=(6, 6))
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

        df_percents.plot.bar(stacked=True, ax=_ax, width=0.7, colormap="Blues")
        _ax.set_xticks(ticks=list(range(len(group) + 2)))
        _ax.set_ylabel("Percentage", fontsize=12)
        _ax.set_xlabel("", fontname="Arial")
        # line thickness for axes thicker
        _axes_linewidth = 2
        _ax.spines["top"].set_linewidth(axes_linewidth)
        _ax.spines["right"].set_linewidth(axes_linewidth)
        _ax.spines["left"].set_linewidth(axes_linewidth)
        _ax.spines["bottom"].set_linewidth(axes_linewidth)
        # make y ticks be integers only
        _ax.set_yticks(ticks=np.arange(0, 110, 10))
        # text size
        _ax.tick_params(axis="both", which="major", labelsize=12)
        # legend
        _ax.legend(
            title="No. crossings",
            title_fontsize=14,
            fontsize=12,
            loc="upper right",
            frameon=False,
        )
        plt.show()
    return


if __name__ == "__main__":
    app.run()
