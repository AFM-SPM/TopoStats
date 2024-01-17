# Configuration

Configuration for TopoStats is done using a [YAML](https://yaml.org/) configuration file that is specified on the
command line when invoking. If no configuration file is provided this default configuration is loaded automatically and
used.

The current configuration file is provided in the TopoStats repository at
[`topostats/default_config.yaml`](https://github.com/AFM-SPM/TopoStats/blob/main/topostats/default_config.yaml) but
please be aware this may not work with your installed version, particularly if you installed from PyPI.

## Generating a configuration

You can always generate a configuration file appropriate for the version you have installed (bar v2.0.0 as this option
was added afterwards). This writes the default configuration to the specified filename (i.e. it does not have to be
called `config.yaml` it could be called `spm-2023-02-20.yaml`). There are a few options available (use `topostats
create-config --help` for further details).

```bash
topostats create-config
```

## Using a custom configuration

If you have generated a configuration file you can modify and edit a configuration it to change the parameters (see
fields below). Once these changes have been saved, you can run TopoStats with this configuration file as shown below.

```bash
topostats process --config my_config.yaml
```

On completion a copy of the configuration that was used is written to the output directory so you have a record of the
parameters used to generate the results you have. This file can be used in subsequent runs of TopoStats.

## YAML Structure

YAML files have key and value pairs, the first word, e.g. `base_dir` is the key this is followed by a colon to separate
it from the value that it takes, by default `base_dir` takes the value `./` (which means the current directory) and so
the entry in the file is a single line with `base_dir: ./`. Other data structures are available in YAML files including
nested values and lists.

A list in YAML consists of a key (e.g. `above:`) followed by the values in square brackets separated by commas such as
`above: [ 500, 800 ]`. This means the `above` key is a list of the values `500` and `800`. Long lists can be split over
separate lines as shown below

```yaml
above:
  - 100
  - 200
  - 300
  - 400
```

## Fields

Aside from the comments in YAML file itself the fields are described below.

| Section         | Sub-Section                       | Data Type  | Default                     | Description                                                                                                                                                                                                                                                                                                                       |
| :-------------- | :-------------------------------- | :--------- | :-------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `base_dir`      |                                   | string     | `./`                        | Directory to recursively search for files within.[^1]                                                                                                                                                                                                                                                                             |
| `output_dir`    |                                   | string     | `./output`                  | Directory that output should be saved to.[^1]                                                                                                                                                                                                                                                                                     |
| `log_level`     |                                   | string     | `info`                      | Verbosity of logging, options are (in increasing order) `warning`, `error`, `info`, `debug`.                                                                                                                                                                                                                                      |
| `cores`         |                                   | integer    | `2`                         | Number of cores to run parallel processes on.                                                                                                                                                                                                                                                                                     |
| `file_ext`      |                                   | string     | `.spm`                      | File extensions to search for.                                                                                                                                                                                                                                                                                                    |
| `loading`       | `channel`                         | string     | `Height`                    | The channel of data to be processed, what this is will depend on the file-format you are processing and the channel you wish to process.                                                                                                                                                                                          |
| `filter`        | `run`                             | boolean    | `true`                      | Whether to run the filtering stage, without this other stages won't run so leave as `true`.                                                                                                                                                                                                                                       |
|                 | `threshold_method`                | str        | `std_dev`                   | Threshold method for filtering, options are `ostu`, `std_dev` or `absolute`.                                                                                                                                                                                                                                                      |
|                 | `otsu_threshold_multiplier`       | float      | `1.0`                       | Factor by which the derived Otsu Threshold should be scaled.                                                                                                                                                                                                                                                                      |
|                 | `threshold_std_dev`               | dictionary | `10.0, 1.0`                 | A pair of values that scale the standard deviation, after scaling the standard deviation `below` is subtracted from the image mean to give the below/lower threshold and the `above` is added to the image mean to give the above/upper threshold. These values should _always_ be positive.                                      |
|                 | `threshold_absolute`              | dictionary | `-1.0, 1.0`                 | Below (first) and above (second) absolute threshold for separating data from the image background.                                                                                                                                                                                                                                |
|                 | `gaussian_size`                   | float      | `0.5`                       | The number of standard deviations to build the Gaussian kernel and thus affects the degree of blurring. See [skimage.filters.gaussian](https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.gaussian) and `sigma` for more information.                                                                     |
|                 | `gaussian_mode`                   | string     | `nearest`                   |                                                                                                                                                                                                                                                                                                                                   |
| `grains`        | `run`                             | boolean    | `true`                      | Whether to run grain finding. Options `true`, `false`                                                                                                                                                                                                                                                                             |
|                 | `row_alignment_quantile`          | float      | `0.5`                       | Quantile (0.0 to 1.0) to be used to determine the average background for the image. below values may improve flattening of large features.                                                                                                                                                                                        |
|                 | `smallest_grain_size_nm2`         | int        | `100`                       | The smallest size of grains to be included (in nm^2), anything smaller than this is considered noise and removed. **NB** must be `> 0.0`.                                                                                                                                                                                         |
|                 | `threshold_method`                | float      | `std_dev`                   | Threshold method for grain finding. Options : `otsu`, `std_dev`, `absolute`                                                                                                                                                                                                                                                       |
|                 | `otsu_threshold_multiplier`       |            | `1.0`                       | Factor by which the derived Otsu Threshold should be scaled.                                                                                                                                                                                                                                                                      |
|                 | `threshold_std_dev`               | dictionary | `10.0, 1.0`                 | A pair of values that scale the standard deviation, after scaling the standard deviation `below` is subtracted from the image mean to give the below/lower threshold and the `above` is added to the image mean to give the above/upper threshold. These values should _always_ be positive.                                      |
|                 | `threshold_absolute`              | dictionary | `-1.0, 1.0`                 | Below (first), above (second) absolute threshold for separating grains from the image background.                                                                                                                                                                                                                                 |
|                 | `direction`                       |            | `above`                     | Defines whether to look for grains above or below thresholds or both. Options: `above`, `below`, `both`                                                                                                                                                                                                                           |
|                 | `smallest_grain_size`             | int        | `50`                        | Catch-all value for the minimum size of grains. Measured in nanometres squared. All grains with area below than this value are removed.                                                                                                                                                                                           |
|                 | `absolute_area_threshold`         | dictionary | `[300, 3000], [null, null]` | Area thresholds for above the image background (first) and below the image background (second), which grain sizes are permitted, measured in nanometres squared. All grains outside this area range are removed.                                                                                                                  |
|                 | `remove_edge_intersecting_grains` | boolean    | `true`                      | Whether to remove grains that intersect the image border. _Do not change this unless you know what you are doing_. This will ruin any statistics relating to grain size, shape and DNA traces.                                                                                                                                    |
| `grainstats`    | `run`                             | boolean    | `true`                      | Whether to calculate grain statistics. Options : `true`, `false`                                                                                                                                                                                                                                                                  |
|                 | `cropped_size`                    | float      | `40.0`                      | Force cropping of grains to this length (in nm) of square cropped images (can take `-1` for grain-sized box)                                                                                                                                                                                                                      |
|                 | `edge_detection_method`           | str        | `binary_erosion`            | Type of edge detection method to use when determining the edges of grain masks before calculating statistics on them. Options : `binary_erosion`, `canny`.                                                                                                                                                                        |
| `dnatracing`    | `run`                             | boolean    | `true`                      | Whether to run DNA Tracing. Options : true, false                                                                                                                                                                                                                                                                                 |
|                 | `min_skeleton_size`               | int        | `10`                        | The minimum number of pixels a skeleton should be for statistics to be calculated on it. Anything smaller than this is dropped but grain statistics are retained.                                                                                                                                                                 |
|                 | `skeletonisation_method`          | str        | `topostats`                 | Skeletonisation method to use, possible options are `zhang`, `lee`, `thin` (from [Scikit-image Morphology module](https://scikit-image.org/docs/stable/api/skimage.morphology.html)) or the original bespoke TopoStas method `topostats`.                                                                                         |
|                 | `spline_step_size`                | float      | `7.0e-9`                    | The sampling rate of the spline in metres. This is the frequency at which points are sampled from fitted traces to act as guide points for the splining process using scipy's splprep.                                                                                                                                            |
|                 | `spline_linear_smoothing`         | float      | `5.0`                       | The amount of smoothing to apply to splines of linear molecule traces.                                                                                                                                                                                                                                                            |
|                 | `spline_circular_smoothing`       | float      | `0.0`                       | The amount of smoothing to apply to splines of circular molecule traces.                                                                                                                                                                                                                                                          |
|                 | `pad_width`                       | int        | 10                          | Padding for individual grains when tracing. This is sometimes required if the bounding box around grains is too tight and they touch the edge of the image.                                                                                                                                                                       |
|                 | `cores`                           | int        | 1                           | Number of cores to use for tracing. **NB** Currently this is NOT used and should be left commented in the YAML file.                                                                                                                                                                                                              |
| `plotting`      | `run`                             | boolean    | `true`                      | Whether to run plotting. Options : `true`, `false`                                                                                                                                                                                                                                                                                |
|                 | `style`                           | str        | `topostats.mplstyle`        | The default loads a custom [matplotlibrc param file](https://matplotlib.org/stable/users/explain/customizing.html#the-matplotlibrc-file) that comes with TopoStats. Users can specify the path to their own style file as an alternative.                                                                                         |
|                 | `save_format`                     | string     | `png`                       | Format to save images in, see [matplotlib.pyplot.savefig](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html)                                                                                                                                                                                               |
|                 | `pixel_interpolation`             | string     | null                        | Interpolation method for image plots. Recommended default 'null' prevents banding that occurs in some images. If interpolation is needed, we recommend `gaussian`. See [matplotlib imshow interpolations documentation](https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html) for details. |
|                 | `image_set`                       | string     | `all`                       | Which images to plot. Options : `all`, `core`                                                                                                                                                                                                                                                                                     |
|                 | `zrange`                          | list       | `[0, 3]`                    | Low (first number) and high (second number) height range for core images (can take [null, null]). **NB** `low <= high` otherwise you will see a `ValueError: minvalue must be less than or equal to maxvalue` error.                                                                                                              |
|                 | `colorbar`                        | boolean    | `true`                      | Whether to include the colorbar scale in plots. Options `true`, `false`                                                                                                                                                                                                                                                           |
|                 | `axes`                            | boolean    | `true`                      | Whether to include the axes in the produced plots.                                                                                                                                                                                                                                                                                |
|                 | `num_ticks`                       | null / int | `null`                      | Number of ticks to have along the x and y axes. Options : `null` (auto) or an integer >1                                                                                                                                                                                                                                          |
|                 | `histogram_log_axis`              | boolean    | `false`                     | Whether to plot hisograms using a logarithmic scale or not. Options: `true`, `false`.                                                                                                                                                                                                                                             |
| `summary_stats` | `run`                             | boolean    | `true`                      | Whether to generate summary statistical plots of the distribution of different metrics grouped by the image that has been processed.                                                                                                                                                                                              |
|                 | `config`                          | str        | `null`                      | Path to a summary config YAML file that configures/controls how plotting is done. If one is not specified either the command line argument `--summary_config` value will be used or if that option is not invoked the default `topostats/summary_config.yaml` will be used.                                                       |

## Summary Configuration

Plots summarising the distribution of metrics are generated by default. The behaviour is controlled by a configuration
file. The default example can be found in `topostats/summary_config.yaml`. The fields of this file are described below.

| Section        | Sub-Section | Data Type | Default           | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| :------------- | :---------- | :-------- | :---------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `output_dir`   |             | `str`     | `./output/`       | Where output plots should be saved to.                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `csv_file`     |             | `str`     | `null`            | Where the results file should be loaded when running `toposum`                                                                                                                                                                                                                                                                                                                                                                                                     |
| `file_ext`     |             | `str`     | `png`             | File type to save images as.                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `pickle_plots` |             | `bool`    | True              | Whether to save images to a Python pickle.                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `var_to_label` |             | `str`     | `null`            | Optional YAML file that maps variable names to labels, uses `topostats/var_to_label.yaml` if null.                                                                                                                                                                                                                                                                                                                                                                 |
| `molecule_id`  |             | `str`     | `molecule_number` | Variable containing the molecule number.                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `image_id`     |             | `str`     | `image`           | Variable containing the image identifier.                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `hist`         |             | `bool`    | `True`            | Whether to plot a histogram of statistics.                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `bins`         |             | `int`     | `20`              | Number of bins to plot in histogram.                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `stat`         |             | `str`     | `count`           | What metric to plot on histogram valid values are `count` (default), `frequency`, `probability`, `percent`, `density`                                                                                                                                                                                                                                                                                                                                              |
| `kde`          |             | `bool`    | `True`            | Whether to include a Kernel Density Estimate on histograms. **NB** if both `hist` and `kde` are true they are overlaid.                                                                                                                                                                                                                                                                                                                                            |
| `violin`       |             | `bool`    | `True`            | Whether to generate [Violin Plots](https://en.wikipedia.org/wiki/Violin_plot).                                                                                                                                                                                                                                                                                                                                                                                     |
| `figsize`      |             | `list`    | `[16, 9]`         | Figure size (x then y dimensions).                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `alpha`        |             | `float`   | `0.5`             | Level of transparency to use when plotting.                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `palette`      |             | `str`     | `bright`          | Seaborn color palette. Options `colorblind`, `deep`, `muted`, `pastel`, `bright`, `dark`, `Spectral`, `Set2`                                                                                                                                                                                                                                                                                                                                                       |
| `stats_to_sum` |             | `list`    | `str`             | A list of strings of variables to plot, comment (placing a `#` at the start of the line) and uncomment as required. Possible values are `area`, `area_cartesian_bbox`, `aspect_ratio`, `banding_angle`, `contour_length`, `end_to_end_distance`, `height_max`, `height_mean`, `height_median`, `height_min`, `radius_max`, `radius_mean`, `radius_median`, `radius_min`, `smallest_bounding_area`, `smallest_bounding_length`, `smallest_bounding_width`, `volume` |

## Validation

Configuration files are validated against a schema to check that the values in the configuration file are within the
expected ranges or valid parameters. This helps capture problems early and should provide informative messages as to
what needs correcting if there are errors.

## Matplotlib Style

TopoStats generates a number of images of the scans at various steps in the processing. These are plotted using the
Python library [Matplotlib](matplotlib.org/stable/). A custom
[`matplotlibrc`](https://matplotlib.org/stable/users/explain/customizing.html#the-matplotlibrc-file) file is included in
TopoStats which defines the default parameters for generating images. This covers _all_ aspects of a plot that can be
customised, for example we define custom colour maps `nanoscope` and `afmhot` and by default the former is configured to
be used in this file. Other parameters that are customised are the `font.size` which affects axis labels and titles.

If you wish to modify the look of all images that are output you can generate a copy of the default configuration using
`topostats create-matplotlibrc` command which will write the output to `topostats.mplstyle` by default (**NB** there are
flags which allow you to specify the location and filename to write to, see `topostats create-matplotlibrc --help` for
further details).

You should read and understand this commented file in detail. Once changes have been made you can run TopoStats using
this custom file using the following command (substituting `my_custom_topostats.mplstyle` for whatever you have saved
your file as).

```bash
topostats process --matplotlibrc my_custom_topostats.mplstyle
```

**NB** Plotting with Matplotlib is highly configurable and there are a plethora of options that you may wish to
tweak. Before delving into customising `matplotlibrc` files it is recommended that you develop and build the style of
plot you wish to generate using Jupyter Notebooks and then translate them to the configuration file. Detailing all of
the possible options is beyond the scope of TopoStats but the [Matplotlib documentation](https://matplotlib.org/) is
comprehensive and there are some sample Jupyter Notebooks (see `notebooks/03-plotting-scans.ipynb`) that guide you
through the basics.

### Further customisation

Whilst the broad overall look of images is controlled in this manner there is one additional file that controls how
images are plotted in terms of filenames, titles and image types and whether an image is part of the `core` subset that
are always generated or not.

During development it was found that setting high DPI took a long time to generate and save some of the images which
slowed down the overall processing time. The solution we have implemented is a file `topostats/plotting_dictionary.yaml`
which sets these parameters on a per-image basis and these over-ride settings defined in default `topostats.mplstyle` or
any user generated document.

If you have to change these, for example if there is a particular image not included in the `core` set that you always
want produced or you wish to change the DPI (Dots Per Inch) of a particular image you will have to locate this file and
manually edit it. Where this is depends on how you have installed TopoStats, if it is from a clone of the Git repository
then it can be found in `TopoStats/topostats/plotting_dictionary.yaml`. If you have installed from PyPI using `pip
install topostats` then it will be under the virtual environment you have created
e.g. `~/.virtualenvs/topostats/lib/python3.11/site-packages/topostats/topostats/plotting_dictionary.yaml` if you are
using plain virtual environments or
`~/miniconda3/envs/topostats/lib/python3.11/site-packages/topostats/topostats/plotting_dictionary.yaml` if you are using
Conda environments and chose `~/miniconda3` as the base directory when installing Conda.

**NB** The exact location will be highly specific to your system so the above are just guides as to where to find
things.

[^1] When writing file paths you can use absolute or relative paths. On Windows systems absolute paths start with the
drive letter (e.g. `c:/`) on Linux and OSX systems they start with `/`. Relative paths are started either with a `./`
which denotes the current directory or one or more `../` which means the higher level directory from the current
directory. You can always find the current directory you are in using the `pwd` (`p`rint `w`orking `d`irectory). If
your work is in `/home/user/path/to/my/data` and `pwd` prints `/home/user` then the relative path to your data is
`./path/to/my/data`. The `cd` command is used to `c`hange `d`irectory.

```bash
pwd
/home/user/
# Two ways of changing directory using a relative path
cd ./path/to/my/data
pwd
/home/user/path/to/my/data
# Using an absolute path
cd /home/user/path/to/my/data
pwd
/home/user/path/to/my/data
```
