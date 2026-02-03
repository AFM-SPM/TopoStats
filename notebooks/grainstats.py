import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #TopoStats - grain stats walkthrough
    This marimo notebook will walk you through using TopoStats from processing an AFM image through to quantification and data visualisation.

    There are several different ways of installing TopoStats depending on your usage. The simplest is to install from PyPi under a virtual environment:

    ```
    pip install topostats
    ```

    or you could install directly from the TopoStats GitHub repository to access all of the latest features:

    ```
    pip install git+https://github.com/AFM-SPM/TopoStats.git@main
    ```

    For more information on installing TopoStats and setting up virtual environments, please refer to our installation instructions [here]([https://link-url-here.org](https://afm-spm.github.io/TopoStats/main/installation.html)).
    """)
    return


@app.cell
def _():
    """
    Grain statistics notebook.

    This Marimo notebook demonstrates how to extract and visualise single-molecule statistics
    using TopoStats
    """
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Loading libraries and modules
    TopoStats is written as a series of modules with various classes and functions. In order to use these interactively we
    need to `import` them.
    """)
    return


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import numpy.typing as npt
    import seaborn as sns

    from topostats.filters import Filters
    from topostats.grains import Grains
    from topostats.grainstats import GrainStats
    from topostats.io import LoadScans, find_files, read_yaml
    from topostats.plottingfuncs import Images

    return (
        Filters,
        GrainStats,
        Grains,
        Images,
        LoadScans,
        Path,
        find_files,
        json,
        mo,
        np,
        npt,
        plt,
        read_yaml,
        sns,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Finding files to process
    When run from the command line, TopoStats needs to find files to process and the `find_files` function helps with this. It requires the directory path (a folder to search in), and the file extension to look for (this example processes a `.spm` file). The function then returns a list of all files within the directory that have the required file extension.

    To use the `find_files` function within the code block below, it is required that your image files are placed within the same directory as this notebook!
    """)
    return


@app.cell
def _(Path, find_files):
    # Set the base directory to be current working directory of the Notebook
    BASE_DIR = Path().cwd()
    # Adjust the file extension appropriately.
    FILE_EXT = ".spm"
    # Search for *.spm files one directory level up from the current notebooks
    image_files = find_files(base_dir=BASE_DIR.parent / "tests", file_ext=FILE_EXT)
    return BASE_DIR, image_files


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The image files that were found using `find_files` are listed below:
    """)
    return


@app.cell
def _(image_files):
    print(image_files)
    my_filename = "minicircle"
    return (my_filename,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Loading a configuration file

    You can specify all options explicitly by hand when instantiating classes or calling methods/functions. However when run
    at the command line in batch mode TopoStats loads these options from a [YAML](https://yaml.org/) configuration file and it is worth
    understanding the structure of this file and how it is used.

    A trimmed version is shown below. The words that come before the colon `:` are the option, e.g. `file_ext:` is the file extension of the AFM files to be processed, what comes after is the value, in this case `.spm`.

    ```yaml
    base_dir: ./ # Directory in which to search for data files
    output_dir: ./output # Directory to output results to
    log_level: info # Verbosity of output. Options: warning, error, info, debug
    cores: 2 # Number of CPU cores to utilise for processing multiple files simultaneously.
    file_ext: .spm # File extension of the data files.
    loading:
      channel: Height # Channel to pull data from in the data files.
      extract: raw # Array to extract when loading .topostats files.
    filter:
      run: true # Options : true, false
      row_alignment_quantile: 0.5 # lower values may improve flattening of larger features
      threshold_method: std_dev # Options : otsu, std_dev, absolute
      otsu_threshold_multiplier: 1.0
      threshold_std_dev:
        below: 10.0 # Threshold for data below the image background
        above: 1.0 # Threshold for data above the image background
      threshold_absolute:
        below: -1.0 # Threshold for data below the image background
        above: 1.0 # Threshold for data above the image background
      gaussian_size: 1.0121397464510862 # Gaussian blur intensity in px
      gaussian_mode: nearest # Mode for Gaussian blurring. Options : nearest, reflect, constant, mirror, wrap
      # Scar removal parameters. Be careful with editing these as making the algorithm too sensitive may
      # result in ruining legitimate data.
      remove_scars:
        run: false
        removal_iterations: 2 # Number of times to run scar removal.
        threshold_low: 0.250 # lower values make scar removal more sensitive
        threshold_high: 0.666 # lower values make scar removal more sensitive
        max_scar_width: 4 # Maximum thickness of scars in pixels.
        min_scar_length: 16 # Minimum length of scars in pixels.
    grains:
      run: true # Options : true, false
      # Thresholding by height
      grain_crop_padding: 1 # Padding to apply to grains. Needs to be at least 1, more padding may help with unets.
      threshold_method: std_dev # Options : std_dev, otsu, absolute, unet
      otsu_threshold_multiplier: 1.0
      threshold_std_dev:
        below: [10.0] # Thresholds for grains below the image background. List[float].
        above: [1.0] # Thresholds for grains above the image background. List[float].
      threshold_absolute:
        below: [-1.0] # Thresholds for grains below the image background. List[float].
        above: [1.0] # Thresholds for grains above the image background. List[float].
      direction: above # Options: above, below, both (defines whether to look for grains above or below thresholds or both)
      area_thresholds:
        above: [300, 3000] # above surface [Low, High] in nm^2 (also takes null)
        below: [null, null] # below surface [Low, High] in nm^2 (also takes null)
      remove_edge_intersecting_grains: true # Whether or not to remove grains that touch the image border
    grainstats:
      run: true # Options : true, false
      edge_detection_method: binary_erosion # Options: canny, binary erosion. Do not change this unless you are sure of what this will do.
      extract_height_profile: true # Extract height profiles along maximum feret of molecules
      class_names: ["DNA", "Protein"] # The names corresponding to each class of a object identified, please specify merged classes after.
    plotting:
      run: true # Options : true, false
      style: topostats.mplstyle # Options : topostats.mplstyle or path to a matplotlibrc params file
      savefig_format: null # Options : null, png, svg or pdf. tif is also available although no metadata will be saved. (defaults to png) See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html
      savefig_dpi: 100 # Options : null (defaults to the value in topostats/plotting_dictionary.yaml), see https://afm-spm.github.io/TopoStats/main/configuration.html#further-customisation and https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html
      pixel_interpolation: null # Options : https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html
      grain_crop_plot_size_nm: -1 # Size in nm of the square cropped grain images if using the grains image set. If -1, will use the grain's default bounding box size.
      image_set: # Options : all, core, filters, grains, grain_crops, disordered_tracing, nodestats, ordered_tracing, splining. Uncomment to include
        # - all
        - core
        # - filters
        # - grains
        # - grain_crops
        # - disordered_tracing
        # - nodestats
        # - ordered_tracing
        # - splining
      zrange: [null, null] # low and high height range for core images (can take [null, null]). low <= high
      colorbar: true # Options : true, false
      axes: true # Options : true, false (due to off being a bool when parsed)
      num_ticks: [null, null] # Number of ticks to have along the x and y axes. Options : null (auto) or integer > 1
      cmap: null # Colormap/colourmap to use (default is 'nanoscope' which is used if null, other options are 'afmhot', 'viridis' etc.)
      mask_cmap: blue_purple_green # Options : blu, jet_r and any in matplotlib
      histogram_log_axis: false # Options : true, false
      number_grain_plots: true # Add grain numbers next to each grain mask in images with mask overlays. Options : true, false
    summary_stats:
      run: true # Whether to make summary plots for output data
      config: null
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To load the configuration file into Python we use the `read_yaml()` function. This saves the options as a dictionary and
    we can access values by the keys. The example below prints out the top-levels keys and then the keys for the `filter`
    configuration.

    Python dictionaries have keys which can be considered as the parameter that is to be configured and each key has
    an associated value which is the value you wish to set the parameter to.
    """)
    return


@app.cell
def _(BASE_DIR, read_yaml):
    config = read_yaml(BASE_DIR.parent / "topostats" / "default_config.yaml")
    print(f"Top level keys of config.yaml : \n\n {config.keys()}\n")
    print(f"Configuration options for Filter : \n\n {config['filter'].keys()}")
    return (config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To create a new config file we use the `write_config_with_comments()` function which takes the `args` namespace. There a 4 types of valid config (selected using `args.config`) that can be generated, which are as follows:
      1. `default` - The default config includes all the configuration options including some parameters which are not recommended to be changed unless confident on the effects.
      2. `simple` - A simple config includes a subset of the default configuration with only options which are likely to be adjusted by users.
      3. `mplstyle` - generates a [Matplotlib rc-file](https://matplotlib.org/stable/users/explain/customizing.html#customizing-with-matplotlibrc-files) which can be used for customising Matplotlib plots.
      4. `var_to_label` - generates a YAML file which maps variable names used in CSV output to descriptions which can be used when plotting data.
    These files are generated from the respective files that are part of the TopoStats package.
    Additional options that are used by the function is the filename (args.filename) the config will be saved as and the output directory (args.output_dir) it will be saved to.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can look at all of the options using the `json` package to "pretty" print the dictionary which makes it easier to
    read. Here we print the `filter` section. You can see the options map to those of the `Filter()` class with an
    additional `"run": true` which is used when running TopoStats at the command line.
    """)
    return


@app.cell
def _(config, json):
    print(json.dumps(config["filter"], indent=4))
    return


@app.cell
def _(np, npt, plt):
    def show_image(
        arr: npt.NDArray,
        cmap: str = "afmhot",
        size: tuple[int] = (8, 8),
        colorbar: bool = True,
        title: str | None = None,
    ) -> tuple:
        """
        Display a 2D NumPy array as an image.

        Parameters
        ----------
        arr : npt.NDArray
            2D array to plot.
        cmap : str
            Colormap to use in plot.
        size : tuple[int]
            Size to plot image as.
        colorbar : bool
            Whether to include a scale colorbar.
        title : str | None
            Title to include in plot.
        """
        arr = np.asarray(arr)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {arr.shape}")

        fig, ax = plt.subplots(figsize=size)
        im = ax.imshow(arr, cmap=cmap)
        if title:
            ax.set_title(title)
        if colorbar:
            fig.colorbar(im, ax=ax, shrink=0.7)
        fig.tight_layout()
        return fig

    return (show_image,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We will use the configuration options we have loaded in processing the `minicircle.spm` image. For convenience we save
    each set of options to their own dictionary and remove the `run` entry as this is not required when running TopoStats
    interactively.

    We also set the `plotting_config["image_set"]` to `all` so that all images can be plotted (there are some internal controls that determine whether images are plotted and returned).
    """)
    return


@app.cell
def _(config):
    filter_config = config["filter"]
    filter_config["remove_scars"]["run"] = True
    grain_config = config["grains"]
    grain_config.pop("run")
    grainstats_config = config["grainstats"]
    grainstats_config.pop("run")
    plotting_config = config["plotting"]
    plotting_config.pop("run")
    plotting_config["image_set"] = "all"
    return filter_config, grain_config, grainstats_config, plotting_config


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Load scans

    The first step before processing is to load an AFM scan. This extracts the image data into a numpy array along with the filepath and pixel to nm scaling parameter which is used throughout to scale the pixels within an image.
    """)
    return


@app.cell
def _(LoadScans, config, image_files, my_filename, show_image):
    all_scan_data = LoadScans(image_files, **config["loading"])
    all_scan_data.get_data()

    img = all_scan_data.img_dict[my_filename]["image_original"]

    # Plot the loaded scan in its raw format
    show_image(img, cmap="afmhot")
    return (all_scan_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Filtering an image

    Now that we have loaded some images, we can start processing using TopoStats. The first step is called filtering, and involves a sequence of processing steps to remove noise and AFM-specific imaging artifacts. There are a number of options that we need to specify to optimise filtering, and these are described in detail in the TopoStats [documentation](https://topostats.readthedocs.io/en/dev/topostats.filters.html).

    Once we setup a `Filters` object we can call the different methods that are available for it. There are lots of
    different methods that carry out the different steps but for convenience the `filter_image()` method runs all these.

    The following section instantiates ("sets up") an object called `filtered_image` of type `Filters` using the first file
    found (`image_files[0]`) and the various options from the `filter_config` dictionary.
    """)
    return


@app.cell
def _(Filters, all_scan_data, filter_config, my_filename):
    filter_config["remove_scars"]["run"] = True

    filtered_image = Filters(
        image=all_scan_data.img_dict[my_filename]["image_original"],
        filename=all_scan_data.img_dict[my_filename]["img_path"],
        pixel_to_nm_scaling=all_scan_data.img_dict[my_filename]["pixel_to_nm_scaling"],
        row_alignment_quantile=filter_config["row_alignment_quantile"],
        threshold_method=filter_config["threshold_method"],
        otsu_threshold_multiplier=filter_config["otsu_threshold_multiplier"],
        threshold_std_dev=filter_config["threshold_std_dev"],
        threshold_absolute=filter_config["threshold_absolute"],
        gaussian_size=filter_config["gaussian_size"],
        gaussian_mode=filter_config["gaussian_mode"],
        remove_scars=filter_config["remove_scars"],
    )

    filtered_image.filter_image()
    return (filtered_image,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `filtered_image` now has a a number of numpy arrays saved in the `.images` dictionary that can be accessed and plotted. To view
    the names of the images (technically the dictionary keys) you can print them with `filter_image.images.keys()`...
    """)
    return


@app.cell
def _(filtered_image):
    print(f"Available numpy arrays to plot in filter_image.images dictionary :\n\n{filtered_image.images.keys()}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can view the result of TopoStats filtering using the `show_image` function, the final result is stored in the `gaussian_filtered` key.
    """)
    return


@app.cell
def _(filtered_image, show_image):
    show_image(filtered_image.images["gaussian_filtered"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    TopoStats includes a custom plotting class `Images`  which formats plots in a more familiar manner.

    It has a number of options, please refer to the official documentation on
    [configuration](https://afm-spm.github.io/TopoStats/configuration.html) under the `plotting` entry for what these values
    are or the [API
    reference](https://afm-spm.github.io/TopoStats/topostats.plottingfuncs.html#module-topostats.plottingfuncs).

    The class requires a Numpy array, which we have just generated many of during the various filtering stages, and a number
    of options. Again for convenience we use the `**plotting_config` notation to unpack the key/value pairs stored in the
    `plotting_config` dictionary.
    """)
    return


@app.cell
def _(plotting_config):
    print(plotting_config)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here we plot the image after processing and zero-averaging the background but with the `viridis` palette and
    constraining the `zrange` to be between 0 and 3.
    """)
    return


@app.cell
def _(Images, filtered_image, plotting_config):
    current_cmap = plotting_config.pop("cmap")
    current_zrange = plotting_config.pop("zrange")
    fig, ax = Images(
        data=filtered_image.images["gaussian_filtered"],
        filename=filtered_image.filename,
        output_dir="img/",
        cmap="viridis",
        zrange=[0, 3],
        save=True,
        **plotting_config,
    ).plot_and_save()
    # Restore the value for cmap to the dictionary.
    plotting_config["cmap"] = current_cmap
    plotting_config["zrange"] = current_zrange
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Finding grains

    The next step in processing the image is to find grains - a.k.a the molecules we want to analyse. This is done using the `Grains` class and we have saved the
    configuration to the `grains_config` dictionary. For details of the arguments and their values please refer to the
    [configuration](https://afm-spm.github.io/TopoStats/configuration.html) and the [API
    reference](https://afm-spm.github.io/TopoStats/topostats.grains.html).

    The most important thing required for grain finding is the resulting image from the Filtering stage, however several
    other key variables are required. Again there is a one-to-one mapping between the options to the `Grains()` class and
    their values in the configuration file.

    The `pixel_to_nm_scaling` is one of several key variables, as is the `filename` and whilst you can specify these
    yourself explicitly they are available as properties of the `filtered_image` that we have processed. As with `Filters`
    the `Grains` class has a number of methods that carry out the grain finding, but there is a convenience method
    `find_grains()` which calls all these in the correct order.
    """)
    return


@app.cell
def _(Grains, filtered_image, grain_config):
    # NB - Again we can use the one-to-one mapping of configuration options in the grain_config we extracted.
    grains = Grains(
        image=filtered_image.images["final_zero_average_background"],
        filename=filtered_image.filename,
        pixel_to_nm_scaling=filtered_image.pixel_to_nm_scaling,
        **grain_config,
    )
    grains.find_grains()
    return (grains,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `grains` object now also contains a series of images that we can plot, these can be found as keys within grains.mask_images["above"]. We print the full contents of grains below.
    """)
    return


@app.cell
def _(grains):
    grains.__dict__
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To view the full grains mask, you can plot the `merged_classes` mask as below. The first plot shows the mask with background is 0 (black), and grains in 1 (white). The second plot shows the image and mask overlaid.
    """)
    return


@app.cell
def _(grains, show_image):
    # Extract grains mask and image
    mask = grains.mask_images["above"]["merged_classes"][:, :, 1].astype(bool)
    image = grains.image

    # Create overlay: background = 0, show image values where mask==1
    overlay = image * mask

    binary_mask = show_image(mask.astype(int), cmap="gray", title="Mask (0 = background, 1 = grains)", colorbar=False)
    mask_overlay = show_image(overlay, cmap="afmhot", title="Image with mask overlay")

    # Return both so they render one after the other
    binary_mask, mask_overlay
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The grains config enables users to play around with parameters to optimise masking for their individual use cases, it is printed below to show which parameters can be configured.

    The thresholds can be used in different ways based on the `direction` you want to detect grains. Typically for molecular
    imaging where the DNA or protein is raised above the background you will want to look for objects above the surface, using the `above`
    threshold. However, when imaging silicon, you may be interested in objects below the surface, using the `below` threshold. For convenience it is possible to look for grains that are `both` above the `above` threshold and `below` than the below threshold.

    This is controlled using the `config["grains"]["direction"]` configuration option which maps to the `Grains(direction=)`
    option and can take three values `below`, `above` or `both`.
    """)
    return


@app.cell
def _(config, json):
    print(json.dumps(config["grains"], indent=4))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    So far the thresholding method used has been `threshold_method="std_dev"` defined in the configuration file we
    loaded. This calculates the mean and standard deviation of height across the whole image and then determines the threshold by scaling the standard deviation by a given factor (defined by `threshold_std_dev`) and adding it to the mean to give the `above` threshold and/or subtracting if from the mean to give the `below` threshold.

    We could instead use `threshold_method="absolute"`. Absolute thresholding is a simple method of thresholding that uses a user-defined threshold value to separate the foreground and background pixels. This method is useful when you know the exact threshold value you want to use, for example if you know your DNA lies at 2nm above the surface you can set the threshold to 1.5nm to capture the DNA without capturing the background. Here we explicitly state the `below` and `above` height thresholds (although since we are only looking for objects above a given threshold only the `above` value will be used).

    Other thresholding methods that can be used within TopoStats are described in the documentation [here]("https://github.com/AFM-SPM/TopoStats/blob/main/docs/advanced/thresholding.md").

     In the code block below we set `threshold_method="absolute"` and show the resulting masks when two different absolute thresholds of 1.2nm and 2.0nm are used.
    """)
    return


@app.cell
def _(Grains, filtered_image, grain_config, grains, show_image):
    # Setting the absolute threshold to 2.0 rather than 1.2
    grain_config["threshold_method"] = "absolute"
    grain_config["threshold_absolute"]["above"] = [2.0]
    grains_newthreshold = Grains(
        image=filtered_image.images["final_zero_average_background"],
        filename=filtered_image.filename,
        pixel_to_nm_scaling=filtered_image.pixel_to_nm_scaling,
        **grain_config,
    )

    grains_newthreshold.find_grains()

    original_threshold = show_image(
        grains.mask_images["above"]["merged_classes"][:, :, 1].astype(bool).astype(int),
        cmap="gray",
        title="Absolute threshold = 1.2",
        colorbar=False,
    )

    new_threshold = show_image(
        grains_newthreshold.mask_images["above"]["merged_classes"][:, :, 1].astype(bool).astype(int),
        cmap="gray",
        title="Absolute threshold = 2.0",
        colorbar=False,
    )

    original_threshold, new_threshold
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Extracting grain statistics
    Now that the grains have been found we can calculate statistics for each. This is done using the `GrainStats()`
    class. Again the configuration options from the YAML file map to those of the class and there is a convenience method
    `calculate_stats()` which will run all steps of grain stats.

    For details of what the arguments are please refer to the [API
    reference](https://afm-spm.github.io/TopoStats/topostats.grainstats.html).

    The `GrainStats` class returns two objects, a Pandas `pd.DataFrame` of calculated statistics and a `list` of
    dictionaries containing the grain data to be plotted. We therefore instantiate ("set-up") the `grain_stats` dictionary
    to hold these results and unpack each to the keys `statistics` and `plots` respectively.
    """)
    return


@app.cell
def _(GrainStats, Grains, grains, grainstats_config, show_image):
    labelled_regions = Grains.label_regions(
        grains.mask_images["above"]["merged_classes"][:, :, 1].astype(bool).astype(int)
    )
    show_image(labelled_regions, cmap="viridis", colorbar=False)

    cfg = grainstats_config.copy()
    cfg.pop("class_names", None)

    grainstats = GrainStats(
        grain_crops=grains.image_grain_crops.above.crops,
        direction="above",
        base_output_dir="grains",
        **cfg,
    )

    _temp = grainstats.calculate_stats()
    grain_stats = {"statistics": _temp[0], "plots": _temp[1]}
    return (grain_stats,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `grain_stats["statistics"]` is a [Pandas
    DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html). We can print this out as
    shown below.
    """)
    return


@app.cell
def _(grain_stats):
    grain_stats["statistics"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Further we can summarise the dataframe for a subset of variables, here we show summary stats for `smallest_bounding_width` and `aspect_ratio` as an example.
    """)
    return


@app.cell
def _(grain_stats):
    grain_stats["statistics"][["smallest_bounding_width", "aspect_ratio"]].describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can use the `seaborn` package to produce plots of certain variables from grain_stats table. `seaborn` has a range of different options for data visualisation, below we show how you can create violin and KDE plots to explore the distribution of certain variables. Try playing around with the `col` definition to view plots of different variables.
    """)
    return


@app.cell
def _(grain_stats, plt, sns):
    col = "area"  # or whichever column you wish to plot

    # Violin plot
    plt.figure(figsize=(6, 4))
    sns.violinplot(y=grain_stats["statistics"][col], inner="box")  # y= for vertical violin
    plt.title(f"Violin plot of {col}")
    plt.ylabel(col)
    plt.show()

    # KDE plot
    plt.figure(figsize=(6, 4))
    sns.kdeplot(x=grain_stats["statistics"][col], fill=True)
    plt.title(f"KDE of {col}")
    plt.xlabel(col)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It could also be useful to compare whether two variables have any relationship with one another, and for this a scatter plot can be used to visualise correlation. Below we show an example of plotting `aspect_ratio` against `height_mean`, but these can be replaced with any other columns from the `grain_stats` table.
    """)
    return


@app.cell
def _(grain_stats, plt, sns):
    # Pick two variables
    xcol, ycol = "aspect_ratio", "height_mean"  # replace with whichever columns you want to plot

    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=grain_stats["statistics"], x=xcol, y=ycol)
    plt.title(f"Scatter plot: {ycol} vs {xcol}")
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.show()
    return


if __name__ == "__main__":
    app.run()
