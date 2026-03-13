import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import h5py
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    import marimo as mo

    return Path, mo, pd, plt, sns


@app.cell
def _(Path):
    dir_base = Path(
        "/Volumes/shared/pyne_group/Shared/AFM_Data/Plasmids/pICoZ/20260105_20251031_20251107_combined_picoz_dataset/output"
    )
    assert dir_base.exists()
    return (dir_base,)


@app.cell
def _(dir_base, pd):
    curvature_data = pd.read_csv(dir_base / "grain_statistics.csv")
    assert curvature_data is not None
    return (curvature_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### sanitise dataframe values
    """)
    return


@app.cell
def _(curvature_data):
    # define picoz versions
    picoz_species = ["SCpicoz", "nicked", "3ATpicoz", "tel80picoz", "TEL12"]
    picoz_colors = ["#D81B60", "#1E88E5", "#FFC107", "#004D40", "#D35FB7"]


    def find_species(image_name: str, picoz_species: list[str]):
        for species in picoz_species:
            if species in str(image_name):
                return species
        return "other"


    curvature_data["species"] = curvature_data["image"].apply(
        find_species, picoz_species=picoz_species
    )

    print(curvature_data["species"].value_counts())

    # Update the contour length to be in nanometres
    curvature_data["total_contour_length_nm"] = (
        curvature_data["total_contour_length"] * 1e9
    )
    return (picoz_colors,)


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


if __name__ == "__main__":
    app.run()
