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
def _(dir_base, pd):
    # load csv
    curvature_data = pd.read_csv(dir_base / "grain_statistics.csv")
    assert curvature_data is not None

    # define picoz versions
    picoz_species = ["SCpicoz", "nicked", "3ATpicoz", "tel80picoz", "TEL12"]
    picoz_colors = ["#D81B60", "#1E88E5", "#FFC107", "#004D40", "#D35FB7"]
    picoz_excluded_species = ["tel80picoz"]


    def find_species(image_name: str, picoz_species: list[str]):
        for species in picoz_species:
            if species in str(image_name):
                return species
        return "other"


    curvature_data["species"] = curvature_data["image"].apply(
        find_species, picoz_species=picoz_species
    )
    # drop any grains with nan values for curvature
    curvature_data = curvature_data.dropna()
    # drop any excluded species
    curvature_data = curvature_data[
        ~curvature_data["species"].isin(picoz_excluded_species)
    ]

    print(curvature_data["species"].value_counts())

    # Update the contour length to be in nanometres
    curvature_data["total_contour_length_nm"] = (
        curvature_data["total_contour_length"] * 1e9
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
def _(curvature_data, picoz_colors, plt, sns):
    curvature_metrics = [
        ("curvature_mean", "Mean curvature"),
        ("curvature_std", "Standard deviation of curvature"),
        ("curvature_var", "Variance of curvature"),
        ("curvature_median", "Median curvature"),
        ("curvature_iqr", "Interquartile range of curvature"),
        ("curvature_total", "Total curvature"),
        ("curvature_max", "Maximum curvature"),
        ("curvature_min", "minimum_curvature"),
    ]

    for parameter_name, plot_title in curvature_metrics:
        print(f"plotting {parameter_name}, titled: {plot_title}")
        sns.boxplot(
            x="species",
            y=parameter_name,
            data=curvature_data,
            palette=picoz_colors,
            hue="species",
        )
        sns.stripplot(
            x="species",
            y=parameter_name,
            data=curvature_data,
            color="black",
            alpha=0.4,
            jitter=True,
            palette=picoz_colors,
            hue="species",
        )
        plt.xlabel("pICOz variant")
        plt.ylabel(plot_title)
        plt.title("Grain curvature by species")
        sns.despine()
        plt.show()
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

    test, p = perform_t_test(
        df=curvature_data,
        sample_type_1="SCpicoz",
        sample_type_2="nicked",
        value_column="curvature_var",
    )
    print(f"t test for SCpicoz: {test}, {p:.5f}")


    test, p = perform_group_test(
        df=curvature_data,
        sample_types=["SCpicoz", "3ATpicoz", "TEL12"],
        value_column="curvature_var",
    )
    print(f"group test for SCpicoz, 3ATpicoz, TEL12: {test}, {p:.5f}")
    return


if __name__ == "__main__":
    app.run()
