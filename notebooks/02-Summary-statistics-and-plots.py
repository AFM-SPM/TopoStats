import marimo

__generated_with = "0.14.17"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Summarising and Plotting Statistics

    After a successful run of `run_topostats` you will have a `all_statistics.csv` file that contains a summary of various
    statistics about the detected molecules across all image files that were processed. There is a class
    `topostats.plotting.TopoSum` that uses this file to generate plots automatically and a convenience command
    `toposum` which provides an entry point to re-run the plotting at the command line.

    Inevitably though there will be a point where you want to tweak plots for publication or otherwise in some manner that
    is not conducive to scripting in this manner because making every single option from
    [Seaborn](https://seaborn.pydata.org/) and [Matplotlib](https://matplotlib.org/) accessible via this class is a
    considerable amount of work writing [boilerplate code](https://en.wikipedia.org/wiki/Boilerplate_code). Instead the
    plots should be generated and tweaked interactively a notebook. This Notebook serves as a sample showing how to use the
    `TopoSum` class and some examples of creating plots directly using [Pandas](https://pandas.pydata.org/).

    If you are unfamiliar with these packages it is recommended that you read the documentation. It is worth bearing in mind
    that both Pandas and Seaborn build on the basic functionality that Matplotlib provides, providing easier methods for
    generating plots. If you are stuck doing something with either of these refer to Matplotlib for how to achieve what you
    are trying to do.

    * [Pandas](https://pandas.pydata.org/docs/)
    * [10 minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html)
    * [Chart visualization — pandas](https://pandas.pydata.org/docs/user_guide/visualization.html?highlight=plotting)
    * [seaborn: statistical data visualization](https://seaborn.pydata.org/index.html)
    * [An introduction to seaborn](https://seaborn.pydata.org/tutorial/introduction.html)
    * [Matplotlib — Visualization with Python](https://matplotlib.org/)
    * [Tutorials — Matplotlib](https://matplotlib.org/stable/tutorials/index)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Load Libraries""")
    return


@app.cell
def _():
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    return matplotlib, np, pd, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Load  `all_statistics.csv`

    You need to load your data to be able to work with it, this is best achieved by importing it using
    [Pandas](https://pandas.pydata.org/). Here we use the `tests/resources/minicircle_default_all_statistics.csv` that is
    part of the TopoStats repository and load it into the object called `df` (short for "Data Frame"). You will need to
    change this path to reflect your output. 

    Because `molecule_number` is unique to the `image` and `threshold` we set a multi-level index of these three
    """
    )
    return


@app.cell
def _(pd):
    df = pd.read_csv("../tests/resources/minicircle_default_all_statistics.csv")
    df.set_index(["image", "threshold", "grain_number"], inplace=True)
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Data Manipulation

    Sometimes it is desirable to extract further information from the CSV, for example sub-folder names. Pandas is an
    excellent tool for doing this, but it can be a bit overwhelming with working out where to start as there are so many
    options. This section contains some simple recipes for manipulating the data.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Splitting `basename`

    The `basename` variable contains the directory paths and at times it may be desirable to group distribution plots across
    images based on the directory from which they originate. The specific directory name is part of the longer string
    `basename` and so this needs splitting to access the directory components.

    **NB** The value for `pat` (the pattern on which the string is split) may vary depending on the operating system the
    images were processed on.
    """
    )
    return


@app.cell
def _(df):
    # Split and expand `basename` into a new dataframe
    basename_components_df = df["basename"].str.split("/", expand=True)
    basename_components_df.head()
    return (basename_components_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    You can now select which elements of `basename_components_df` to merge back into the original `df`. To just include both
    components of the split `basename` you would
    """
    )
    return


@app.cell
def _(basename_components_df, df):
    basename_components_df.columns = ['basename1', 'basename2']
    df_1 = df.merge(basename_components_df, left_index=True, right_index=True)
    df_1.head()
    return (df_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Plotting with Pandas""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Plotting Contour Lengths""")
    return


@app.cell
def _(df_1):
    df_1['contour_length'].plot.hist(figsize=(16, 9), bins=20, title='Contour Lengths', alpha=0.5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Plotting End to End Distance of non-Circular grains

    Circular grains are excluded since their end-to-end length is 0.0.
    """
    )
    return


@app.cell
def _(df_1):
    df_1[df_1['circular'] == False]['end_to_end_distance'].plot.hist(figsize=(16, 9), bins=20, title='End to End Distance', alpha=0.5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Multiple Images

    Often you will have processed multiple images and you will want to plot the distributions of metrics for each image
    separately.

    For this example we duplicate the data and append it, adjusting the values slightly
    """
    )
    return


@app.cell
def _(df_1, pd):
    def scale_df(df: pd.DataFrame, scale: float, image: str) -> pd.DataFrame:
        """Scale the numerical values of a data frame. Retains string variables and the index.

        Parameters
        ----------
        df: pd.DataFrame
            Pandas Dataframe to scale.
        scale: float
            Factor by which to scale the data.
        image: str
            Name for new (dummy) image.

        Returns
        -------
        pd.DataFrame
            Scaled data frame
        """
        _df = df[df.select_dtypes(include=['number']).columns] * scale
        _df.reset_index(inplace=True)
        _df['image'] = image
        _df = pd.concat([_df, df[['circular', 'basename']]], axis=1)
        _df.set_index(['image', 'threshold', 'grain_number'], inplace=True)
        return _df
    smaller = scale_df(df_1, scale=0.4, image='smaller')
    larger = scale_df(df_1, scale=1.5, image='larger')
    df_three_images = pd.concat([smaller, df_1, larger])
    return (df_three_images,)


@app.cell
def _(df_three_images):
    df_three_images.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Contour Length from Three Processed Images""")
    return


@app.cell
def _(df_three_images, plt):
    groups = list(df_three_images.groupby("image"))
    n = len(groups)

    fig1, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)

    for ax1, (name, group) in zip(axes, groups):
        group["contour_length"].plot.hist(
            bins=20,
            alpha=0.7,
            ax=ax1,
            title=f"Contour Lengths – {name}"
        )
        ax1.set_xlabel("Contour length")
        ax1.set_ylabel("Count")

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(df_three_images, np, plt):
    # Global min/max (padded by ±10)
    x_min = df_three_images["contour_length"].min() - 10
    x_max = df_three_images["contour_length"].max() + 10

    # Make shared bin edges so that histogram bars are all the same size (20 bins => 21 edges)
    n_bins = 20
    bins = np.linspace(x_min, x_max, n_bins + 1)

    fig2, ax2 = plt.subplots(figsize=(10, 6))

    for name2, group2 in df_three_images.groupby("image"):
        group2["contour_length"].plot.hist(
            bins=bins,
            alpha=0.5,
            ax=ax2,
            label=name2,
            linewidth=0.5
        )

    ax2.set_xlim(x_min, x_max)
    ax2.set_title("Contour Lengths – all images")
    ax2.set_xlabel("Contour length")
    ax2.set_ylabel("Count")
    ax2.legend(title="Image")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Ignoring Image

    It is possible to plot the distribution of summary statistics without regard to the image from which they are
    derived. Simply omit the `.groupby("image")` from the plotting command.

    We also manually set the `fontsize`.
    """
    )
    return


@app.cell
def _(df_three_images, matplotlib):
    matplotlib.rcParams.update({"font.size": 20})
    df_three_images["contour_length"].plot.hist(figsize=(16, 9), bins=20, title="Contour Lengths", alpha=0.5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Violin Plot of `max_feret` using Seaborn

    Pandas does not have built-in support for Violin Plots so we switch to using Seaborn. Here the `fig` and `ax` objects
    are created first and we use the `ax.text()` method to add a string (`text_str`) in a box to the plot.
    """
    )
    return


@app.cell
def _(df_three_images, plt, sns):
    # Reset dataframe index to make `image` readily available
    plot_df = df_three_images.reset_index()

    fig_violin, ax_violin = plt.subplots(1, 1, figsize=(16, 9))
    sns.violinplot(
        data=plot_df,
        x="image",
        y="max_feret",
        ax=ax_violin,
        cut=0,
        inner="quartile"
    )
    ax_violin.set_title("Maximum Feret")
    ax_violin.set_ylabel("Maximum Feret / nm")

    text_str = "\n".join([
        "Sodium Concentration    : 0.001 mM",
        "Scan Size               : 200 x 200",
        "More useful information : :-)"
    ])
    props = dict(boxstyle="round", alpha=0.5)
    ax_violin.text(
        0.5, 0.85, text_str,
        transform=ax_violin.transAxes,
        fontsize=12,
        ha="center",
        bbox=props,
    )

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Joint Plot
    [Joint  Plots](https://seaborn.pydata.org/generated/seaborn.jointplot.html) showing the relationship between two variables can be plotted easily.
    """
    )
    return


@app.cell
def _(df_1, sns):
    df_1.columns
    sns.jointplot(data=df_1, x='min_feret', y='max_feret', hue='circular')
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
