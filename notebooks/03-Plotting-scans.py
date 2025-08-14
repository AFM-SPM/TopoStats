import marimo

__generated_with = "0.14.17"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Plotting Scans

    This Notebook demonstrates how to plot cleaned scans that have been produced by `run_topostats`. There are a large
    number of options available when plotting, too many to cover in this Notebook, but the aim is to demonstrate some
    basics...

    * Loading NumPy Arrays
    * Plotting using the TopoStats `plot_and_save()` function.
    * Selecting a subset of a scan and plotting that.
    * Applying different colour maps.
    * Adding custom headings and axis labels.
    * Saving images in a range of publication quality formats.

    The [NumPy](https://numpy.org/) arrays are plotted using [Matplotlib](https://matplotlib.org/) which has excellent
    documentation. If you want to learn more then the [Tutorials and
    Examples](https://matplotlib.org/stable/users/index.html#tutorials-and-examples) are a good place to start learning from.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Setup

    The first step required is to import some Python libraries to load and plot the data. You should run this Notebook
    within a Conda/Virtual Environment into which you have installed TopoStats, ideally with the necessary Notebook
    extensions. The following command will install TopoStats from [PyPI](https://pypi.org/project/topostats/) with the
    requirements for running Notebooks.

    ```python
    pip install topostats[notebooks]
    ```

    You should have successfully processed images using `run_topostats` at least once, this will have saved processed scans
    to disk that we will load.
    """
    )
    return


@app.cell
def _():
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    # from topostats.io import load_array  # if you prefer TopoStats' loader

    # 1) Set your .npy path
    npy_path = Path("path/to/your/file.npy")  # <-- change this

    # 2) Load
    arr = np.load(npy_path)  # or: arr = load_array(npy_path)

    # 3) Use it (e.g., show image)
    plt.imshow(arr, cmap="gray")
    plt.title(npy_path.name)
    plt.axis("off")
    plt.show()
    return Path, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Load

    Before we can plot data we need to load the data. You need to know where this file is located and this will depend on
    the configuration you used when using `run_topostats`. It will be located in the `processed` directory of your output
    (but remember that it reflects the directory structure your files were stored in originally).
    """
    )
    return


@app.cell
def _(Path, load_array):
    outpath = Path("../tests/resources/")
    image_array = load_array(outpath / "minicircle_cropped_flattened.npy")
    return (image_array,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Configuration

    A configuration saved as a Python Dictionary is the easiest way to work with plotting and saves a lot of repetitive
    typing of options. A sample is provided below and is stored in the object `plotting_config`.

    We set the output directory to be the current working directory, if you wish to set this as something different then you
    should modify the following cell to something like

    ```
    outpath = Path("/path/you/want/to/save/images/to/")
    ```

    In the cell below the `outpath` is set to the location from which we load the array data.
    """
    )
    return


@app.cell
def _(Path):
    outpath_1 = Path('../tests/resources/')
    plotting_config = {'image_set': 'core', 'zrange': [None, None], 'colorbar': True, 'axes': True, 'cmap': 'nanoscope', 'mask_cmap': 'blue', 'histogram_log_axis': False, 'histogram_bins': 200, 'core_set': True, 'title': 'Height Thresholded', 'image_type': 'non-binary', 'save': True, 'output_dir': outpath_1}
    return outpath_1, plotting_config


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Plotting with TopoStats

    TopoStats includes a class `Image` which makes plotting easy. It requires a few arguments though, the array that is to
    be plotted (`image_array`), the image name (`test_image`) and a dictionary of options which we have defined above.

    This last argument, the dictionary of options is prefixed `**` which is known as _Python Keywords_. It means that the
    dictionary is "unpacked" and we have setup the dictionary so that every key is an argument to the `Image` class and the
    values of the dictionary are passed into `Image`. If interested in finding out more about this see the following
    articles...

    * [Dictionaries in Python – Real Python](https://realpython.com/python-dicts/)
    * [Python args and kwargs: Demystified – Real Python](https://realpython.com/python-kwargs-and-args/)

    The cell below "instantiates" an object (`image_plot`) of the class `Image`, it _won't_ produce any output....yet!
    """
    )
    return


@app.cell
def _(Images, image_array, plotting_config):
    image_plot = Images(image_array, filename="minicircle", **plotting_config)
    return (image_plot,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Classes such as `Image` have "methods" associated with them, these are what does the hard work and produces output. This
    means the instance of `Image` that is `image_plot` has a method called `.plot_and_save()` which plots and saves the
    file. The method returns two objects, a `figure` which is the actual plot and an `axes` which is the region or box into
    which the `figure` is drawn.  If we call it now we are told the image is saved and we can then display the figure in the
    Notebook by using the returned `figure`.

    In this example we have included all f the options from the dictionary relevant to this type of plot, such as
    `colorbar=True` and the `cmap="nanoscope"` (`cmap` is short for "colormap" and defines the colours used for plotting).
    """
    )
    return


@app.cell
def _(image_plot):
    _figure, _axes = image_plot.plot_and_save()
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Changing Properties

    If we want to change the properties we can either define a new dictionary, or we can modify the properties of the
    instantiated `Images` object `image_plot`. For example to change the colour map (`cmap`) and _not_ plot the `colorbar`
    we can set those values to `viridis` and `False` respectively. And if we want to change the title we can change the
    `title` property.
    """
    )
    return


@app.cell
def _(image_plot):
    image_plot.cmap = 'viridis'
    image_plot.colorbar = False
    image_plot.title = 'Minicircle : Height Thresholded...in Viridis!'
    _figure, _axes = image_plot.plot_and_save()
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Colormaps

    Another colormap (`cmap`) that is available is `afmhot`. We plot the same `minicircle` image using this colormap and
    reinstate the colorbar, giving a unique title.
    """
    )
    return


@app.cell
def _(image_plot):
    image_plot.cmap = 'afmhot'
    image_plot.colorbar = True
    image_plot.title = 'Hot Minicircles!'
    _figure, _axes = image_plot.plot_and_save()
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Internally `Image()` is using the colormap palette defined in the `topostats.theme.Colormap` class that has been
    imported, which defines the range of colours for both `nanoscope`, `gwyddion` and `blu` custom colormaps. We will use
    these later.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Plotting a Region

    We may be interested in plotting just a region, say the bottom right-hand corner with the cluster of five molecules. To
    do so we need to subset the original array. This requires a little understanding of how to index [Numpy
    arrays](https://numpy.org/doc/stable/user/basics.indexing.html).

    A Numpy array holding a TopoStats image is a 2-Dimensional array and each cell can be referenced by its `row` position
    (`y`) first and then its `col` (`x'`). Indexing in Python (and most programming languages) starts at zero (`0`) so to
    get the contents of the very first cell you would use `image_array[0,0]` as shown below which shows you the height
    measurement of that cell.
    """
    )
    return


@app.cell
def _(image_array):
    image_array[0, 0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    However, we want to plot a range of rows and columns corresponding to the bottom right hand corner, we can refer to a
    range of values using the notation `start:end` and we can do so both for the `x` dimension and the `y` dimension. To get
    the last 300 rows and the last 300 columns we would therefore use `[701:,701:]` we don't need to specify the end
    location of the columns, Python will just use up to the end of the rows and columns.
    """
    )
    return


@app.cell
def _(image_array):
    image_array[100:, 100:]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We can now plot the subset by instantiating a new object which we call `small_plot` of the class `Images`. Instead of
    passing in the full `image_array` though we take a subset of the last rows after `700` and the last columns `700`. We
    specify a new, unique filename `test_image_small` and reuse the `plotting_config` dictionary.
    """
    )
    return


@app.cell
def _(Images, image_array, plotting_config):
    small_plot = Images(image_array[100:, 100:], filename='test_image_small', **plotting_config)
    _figure, _axes = small_plot.plot_and_save()
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    You may notice the colours are brighter in this cropped image than the region as it appears in the full image plot. Read
    on for how to handle this so that they match the whole image.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Plot just the image

    Its possible that you may want _just_ the image, colorbar or title. This can be done without recourse to the `Image`
    class using Matplotlib directly. We first need to setup a `figure` and `axes` to hold one figure. This is done using
    `plt.subplots()` from Matplotlib.

    We use the `Colormap("nanoscope").get_cmap()` class and method to use the `nanoscope` colour map.
    """
    )
    return


@app.cell
def _(Colormap, image_array, plt):
    _figure, _axes = plt.subplots(1, 1, figsize=(8, 8))
    plt.imshow(image_array, cmap=Colormap('nanoscope').get_cmap())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""If you want to save the image then use `plt.imsave()` with the same arguments, but give a filename as the first argument.""")
    return


@app.cell
def _(Colormap, image_array, outpath_1, plt):
    plt.imsave(outpath_1 / 'image_without_scale_or_title.png', image_array, cmap=Colormap('nanoscope').get_cmap())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Images and Regions 

    Here we setup a `figure` and `axes` with `nrows=1` and `ncols=2`, this makes `axes` essentially an array with length of
    2, starting with an index of 0 and so we reference `axes[0]` for the first image, and `axes[1]` for the second and we
    can combine our two images.

    We use `plt.savefig()` to save the image to a unique filename under `outpath` location (which we set further back).
    """
    )
    return


@app.cell
def _(Colormap, image_array, outpath_1, plt):
    _figure, _axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    _axes[0].set_title('Full Image')
    _axes[0].imshow(image_array, cmap=Colormap('nanoscope').get_cmap())
    _axes[1].set_title('Cropped Region')
    _axes[1].imshow(image_array[100:, 100:], cmap=Colormap('nanoscope').get_cmap())
    plt.savefig(outpath_1 / 'double_image.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    You may notice that the colormap is _not_ the same across the two images, in the _Cropped Region_ the heights are now
    much brighter. In order to make these consistent there are two solutions...

    a) Obtain the minimum and maximum values from the full image.
    b) Obtain a normalised range from the full image.
    """
    )
    return


@app.cell
def _(Colormap, image_array, outpath_1, plt):
    vmin = image_array.min()
    vmax = image_array.max()
    _figure, _axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    _axes[0].set_title('Full Image')
    _axes[0].imshow(image_array, cmap=Colormap('nanoscope').get_cmap(), vmin=vmin, vmax=vmax)
    _axes[1].set_title('Cropped Region')
    _axes[1].imshow(image_array[100:, 100:], cmap=Colormap('nanoscope').get_cmap(), vmin=vmin, vmax=vmax)
    plt.savefig(outpath_1 / 'double_image_standardised_colour.png')
    return


@app.cell
def _(Colormap, image_array, mcolors, outpath_1, plt):
    _norm = mcolors.Normalize(vmin=image_array.min(), vmax=image_array.max())
    _figure, _axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    _axes[0].set_title('Full Image')
    _axes[0].imshow(image_array, cmap=Colormap('nanoscope').get_cmap(), norm=_norm)
    _axes[1].set_title('Cropped Region')
    _axes[1].imshow(image_array[100:, 100:], cmap=Colormap('nanoscope').get_cmap(), norm=_norm)
    plt.savefig(outpath_1 / 'double_image_normalised_colour.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    And of course you can extend this to plot more regions, here we set up a 2x2 grid by virtue of `nrows=2` and
    `ncols=2`. Because this is a 2-d array, as with Numpy arrays we need to index both dimensions, this is done with
    `axes[0,0]` for the first row and column, `axes[0,1]` for the first row and second column, then the second row has
    `axes[1,0]` for the first column and `axes[1,1]` for the second column.

    We select different regions for each cell and again normalise the colour scale.
    """
    )
    return


@app.cell
def _(Colormap, image_array, mcolors, outpath_1, plt):
    _norm = mcolors.Normalize(vmin=image_array.min(), vmax=image_array.max())
    _figure, _axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 16))
    _axes[0, 0].set_title('Full Image')
    _axes[0, 0].imshow(image_array, cmap=Colormap('nanoscope').get_cmap(), norm=_norm)
    _axes[0, 1].set_title('Cropped Region 1')
    _axes[0, 1].imshow(image_array[100:, 100:], cmap=Colormap('nanoscope').get_cmap(), norm=_norm)
    _axes[1, 0].set_title('Cropped Region 2')
    _axes[1, 0].imshow(image_array[:200, :200], cmap=Colormap('nanoscope').get_cmap(), norm=_norm)
    _axes[1, 1].set_title('Cropped Region 3')
    _axes[1, 1].imshow(image_array[150:500, 150:500], cmap=Colormap('nanoscope').get_cmap(), norm=_norm)
    for ax in _axes.flat:
        ax.set(xlabel='Nanometres', ylabel='Nanometres')
    plt.savefig(outpath_1 / 'double_image_normalised_colour.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Going Further

    This Notebook has been a short introduction to the vast array of options that are available for plotting your image scan
    data. There are a _lot_ of options and it is not practical to translate all of these options into configuration options
    to TopoStats, nor is repeatedly running scripts to generate the exact image you want.

    Hopefully the examples introduced above are useful to get you started. More documentation on plotting with Matplotlib
    are available at the following links. 

    * [Matplotlib — Visualization with Python](https://matplotlib.org/)
    * [Image tutorial — Matplotlib
      documentation](https://matplotlib.org/stable/tutorials/introductory/images.html#sphx-glr-tutorials-introductory-images-py)
    * [Creating multiple subplots using plt.subplots — Matplotlib
      documentation](https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html)
    * [StackOverflow - Matplotlib](https://stackoverflow.com/questions/tagged/matplotlib) A Q&A forum where a lot of
      questions about using Matplotlib have been asked.

    If you have questions please feel free to ask in the [Plotting
    Discussions](https://github.com/AFM-SPM/TopoStats/discussions/categories/plotting) section on GitHub.
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
