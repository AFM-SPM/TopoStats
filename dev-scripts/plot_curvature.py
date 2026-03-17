import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    from itertools import combinations
    from pathlib import Path

    import h5py
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from scipy import stats

    return Path, mo, np, pd, plt, sns, stats


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
    curvature_data = pd.read_csv(dir_base / "grain_statistics.csv")
    assert curvature_data is not None

    # define picoz versions
    picoz_species = ["SCpicoz", "nicked", "3ATpicoz", "tel80picoz", "TEL12"]
    picoz_colors = ["#D81B60", "#1E88E5", "#FFC107", "#004D40", "#D35FB7"]
    picoz_excluded_species = ["tel80picoz"]

    expected_length_nm = 408e-9
    length_cutoff_percentage = 0.90
    length_cutoff_nm = expected_length_nm * length_cutoff_percentage


    def find_species(image_name: str, picoz_species: list[str]):
        for species in picoz_species:
            if species in str(image_name):
                return species
        return "other"


    curvature_data["species"] = curvature_data["image"].apply(
        find_species, picoz_species=picoz_species
    )
    # drop any grains with nan values for curvature
    curvature_data = curvature_data.dropna(
        subset=["total_contour_length", "curvature_mean"]
    )
    # drop any excluded species
    curvature_data = curvature_data[
        ~curvature_data["species"].isin(picoz_excluded_species)
    ]

    # Update the contour length to be in nanometres
    curvature_data["total_contour_length_nm"] = (
        curvature_data["total_contour_length"] * 1e9
    )

    # Drop rows that have contour length lower than the minimum
    fig, ax = plt.subplots(2, 1)
    fig_min_x = curvature_data["total_contour_length"].min() * 0.9
    fig_max_x = curvature_data["total_contour_length"].max() * 1.1
    ns_before = curvature_data["species"].value_counts()
    sns.kdeplot(data=curvature_data, x="total_contour_length", ax=ax[0])
    curvature_data = curvature_data[
        curvature_data["total_contour_length"] >= length_cutoff_nm
    ]
    sns.kdeplot(data=curvature_data, x="total_contour_length", ax=ax[1])
    ax[0].set_xlim(fig_min_x, fig_max_x)
    ax[1].set_xlim(fig_min_x, fig_max_x)
    fig.tight_layout()
    plt.show()

    ns_after = curvature_data["species"].value_counts()

    print(
        f"ns before contour length filtering ({length_cutoff_nm / 1e-9:.2f} nm):\n{ns_before}"
    )
    print(
        f"ns after contour length filtering ({length_cutoff_nm / 1e-9:.2f} nm):\n{ns_after}"
    )
    return curvature_data, picoz_colors


@app.cell
def _(np, pd, stats):
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

    return perform_group_test, perform_t_test


@app.cell
def _(curvature_data, picoz_colors, plt, sns):
    sns.boxplot(
        x="species",
        y="total_contour_length_nm",
        data=curvature_data,
        palette=picoz_colors,
        hue="species",
    )

    plt.xlabel("pICOz variant")
    plt.ylabel("Total Contour Length (nm)")
    plt.ylim(0, 800)  # Adjust y-axis limits as needed
    plt.title("Total Contour Length by pICOz Variant")

    sns.despine()
    plt.show()
    return


@app.cell
def _(curvature_data, dir_base, pd, picoz_colors, plt, sns):
    curvature_metrics = [
        ("curvature_mean", "Mean curvature"),
        ("curvature_std", "Standard deviation of curvature"),
        ("curvature_var", "Variance of curvature"),
        ("curvature_median", "Median curvature"),
        ("curvature_iqr", "Interquartile range of curvature"),
        ("curvature_total", "Total curvature"),
        ("curvature_max", "Maximum curvature"),
        ("curvature_min", "minimum_curvature"),
        ("curvature_90th", "90th percentile of curvature"),
    ]

    groups = [["SCpicoz", "nicked"], ["SCpicoz", "3ATpicoz", "TEL12"]]


    def plot_group_curvature_distributions(
        curvature_data_df: pd.DataFrame,
        curvature_metrics: list[tuple[str, str]],
        groups: list[list[str]],
        palette,
        save_dir: str = None,
    ) -> None:
        for group in groups:
            curvature_data_group = curvature_data_df[
                curvature_data_df["species"].isin(group)
            ]
            fig, axs = plt.subplots(
                len(curvature_metrics) // 2 + len(curvature_metrics) % 2,
                2,
                figsize=(20, 30),
            )
            for index, (parameter_name, plot_title) in enumerate(
                curvature_metrics
            ):
                ax = axs[index // 2, index % 2]
                print(f"plotting {parameter_name}, titled: {plot_title}")
                sns.boxplot(
                    x="species",
                    y=parameter_name,
                    data=curvature_data_group,
                    palette=palette,
                    hue="species",
                    ax=ax,
                )
                sns.stripplot(
                    x="species",
                    y=parameter_name,
                    data=curvature_data_group,
                    color="black",
                    alpha=0.4,
                    jitter=True,
                    palette=palette,
                    hue="species",
                    ax=ax,
                )
                ax.set_xlabel("pICOz variant")
                ax.set_ylabel(plot_title)
                ax.set_title(f"Grain curvature by species for {group}")
                sns.despine(ax=ax)
            fig.tight_layout()
            if save_dir is not None:
                plt.savefig(save_dir / f"curvature_stats_plots_{group}.png")
            plt.show()


    plot_group_curvature_distributions(
        curvature_data_df=curvature_data,
        curvature_metrics=curvature_metrics,
        groups=groups,
        palette=picoz_colors,
        save_dir=dir_base,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Stats tests
    """)
    return


@app.cell
def _(curvature_data, perform_group_test, perform_t_test):
    # Do test between SC and nicked
    stat = "curvature_var"

    test, p = perform_t_test(
        df=curvature_data,
        sample_type_1="SCpicoz",
        sample_type_2="nicked",
        value_column=stat,
    )
    print(f"t test for {stat} for SCpicoz & nicked: {test}, {p:.3e}")


    test, p = perform_group_test(
        df=curvature_data,
        sample_types=["SCpicoz", "3ATpicoz", "TEL12"],
        value_column=stat,
    )
    print(f"group test for {stat} for SCpicoz, 3ATpicoz, TEL12: {test}, {p:.3e}")
    return


if __name__ == "__main__":
    app.run()
