import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #TopoStats - grain finding walkthrough
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
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ##Loading libraries and modules
    TopoStats is written as a series of modules with various classes and functions. In order to use these interactively we
    need to `import` them.
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    from topostats.io import LoadScans, find_files, read_yaml
    import json
    return Path, find_files, json, mo, read_yaml


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ##Finding files to process
    When run from the command line, TopoStats needs to find files to process and the `find_files` function helps with this. It requires the directory path (a folder to search in), and the file extension to look for (this example processes a `.spm` file). The function then returns a list of all files within the directory that have the required file extension. 

    To use the `find_files` function within the code block below, it is required that your image files are placed within the same directory as this notebook!
    """
    )
    return


@app.cell
def _(Path, find_files):
    # Set the base directory to be current working directory of the Notebook
    BASE_DIR = Path().cwd()
    # Alternatively if you know where your files are, comment the above line and uncomment the below adjust it for your use.
    # BASE_DIR = Path("/path/to/where/my/files/are")
    # Adjust the file extension appropriately.
    FILE_EXT = ".spm"
    # Search for *.spm files one directory level up from the current notebooks
    image_files = find_files(base_dir=BASE_DIR.parent / "tests", file_ext=FILE_EXT)
    return BASE_DIR, image_files


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The image files that were found using `find_files` are listed below:""")
    return


@app.cell
def _(image_files):
    image_files
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
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
    summary_stats:
      run: true # Whether to make summary plots for output data
      config: null
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    To load the configuration file into Python we use the `read_yaml()` function. This saves the options as a dictionary and
    we can access values by the keys. The example below prints out the top-levels keys and then the keys for the `filter`
    configuration.

    Python dictionaries have keys which can be considered as the parameter that is to be configured and each key has
    an associated value which is the value you wish to set the parameter to.
    """
    )
    return


@app.cell
def _(BASE_DIR, read_yaml):
    config = read_yaml(BASE_DIR.parent / "topostats" / "default_config.yaml")
    print(f"Top level keys of config.yaml : \n\n {config.keys()}\n")
    print(f"Configuration options for Filter : \n\n {config['filter'].keys()}")
    return (config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    You can look at all of the options using the `json` package to "pretty" print the dictionary which makes it easier to
    read. Here we print the `filter` section. You can see the options map to those of the `Filter()` class with an
    additional `"run": true` which is used when running TopoStats at the command line.
    """
    )
    return


@app.cell
def _(config, json):
    print(json.dumps(config["filter"], indent=4))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We will use the configuration options we have loaded in processing the `minicircle.spm` image. For convenience we save
    each set of options to their own dictionary and remove the `run` entry as this is not required when running TopoStats
    interactively.

    We also set the `plotting_config["image_set"]` to `all` so that all images can be plotted (there are some internal controls that determine whether images are plotted and returned).
    """
    )
    return


@app.cell
def _(config):
    loading_config = config["loading"]
    filter_config = config["filter"]
    grain_config = config["grains"]
    grainstats_config = config["grainstats"]
    plotting_config = config["plotting"]
    plotting_config["image_set"] = "all"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""" """)
    return


if __name__ == "__main__":
    app.run()
