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

| Section           | Sub-Section                       | Data Type      | Default                     | Description                                                                                                                                                                                                                                                                                                                            |
| :---------------- | :-------------------------------- | :------------- | :-------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `base_dir`        |                                   | string         | `./`                        | Directory to recursively search for files within.[^1]                                                                                                                                                                                                                                                                                  |
| `output_dir`      |                                   | string         | `./output`                  | Directory that output should be saved to.[^1]                                                                                                                                                                                                                                                                                          |
| `log_level`       |                                   | string         | `info`                      | Verbosity of logging, options are (in increasing order) `warning`, `error`, `info`, `debug`.                                                                                                                                                                                                                                           |
| `cores`           |                                   | integer        | `2`                         | Number of cores to run parallel processes on.                                                                                                                                                                                                                                                                                          |
| `file_ext`        |                                   | string         | `.spm`                      | File extensions to search for.                                                                                                                                                                                                                                                                                                         |
| `loading`         | `channel`                         | string         | `Height`                    | The channel of data to be processed, what this is will depend on the file-format you are processing and the channel you wish to process.                                                                                                                                                                                               |
| `filter`          | `run`                             | boolean        | `true`                      | Whether to run the filtering stage, without this other stages won't run so leave as `true`.                                                                                                                                                                                                                                            |
|                   | `threshold_method`                | str            | `std_dev`                   | Threshold method for filtering, options are `ostu`, `std_dev` or `absolute`.                                                                                                                                                                                                                                                           |
|                   | `otsu_threshold_multiplier`       | float          | `1.0`                       | Factor by which the derived Otsu Threshold should be scaled.                                                                                                                                                                                                                                                                           |
|                   | `threshold_std_dev`               | dictionary     | `10.0, 1.0`                 | A pair of values that scale the standard deviation, after scaling the standard deviation `below` is subtracted from the image mean to give the below/lower threshold and the `above` is added to the image mean to give the above/upper threshold. These values should _always_ be positive.                                           |
|                   | `threshold_absolute`              | dictionary     | `-1.0, 1.0`                 | Below (first) and above (second) absolute threshold for separating data from the image background.                                                                                                                                                                                                                                     |
|                   | `gaussian_size`                   | float          | `0.5`                       | The number of standard deviations to build the Gaussian kernel and thus affects the degree of blurring. See [skimage.filters.gaussian](https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.gaussian) and `sigma` for more information.                                                                          |
|                   | `gaussian_mode`                   | string         | `nearest`                   |                                                                                                                                                                                                                                                                                                                                        |
| `grains`          | `run`                             | boolean        | `true`                      | Whether to run grain finding. Options `true`, `false`                                                                                                                                                                                                                                                                                  |
|                   | `row_alignment_quantile`          | float          | `0.5`                       | Quantile (0.0 to 1.0) to be used to determine the average background for the image. below values may improve flattening of large features.                                                                                                                                                                                             |
|                   | `smallest_grain_size_nm2`         | int            | `100`                       | The smallest size of grains to be included (in nm^2), anything smaller than this is considered noise and removed. **NB** must be `> 0.0`.                                                                                                                                                                                              |
|                   | `threshold_method`                | float          | `std_dev`                   | Threshold method for grain finding. Options : `otsu`, `std_dev`, `absolute`                                                                                                                                                                                                                                                            |
|                   | `otsu_threshold_multiplier`       |                | `1.0`                       | Factor by which the derived Otsu Threshold should be scaled.                                                                                                                                                                                                                                                                           |
|                   | `threshold_std_dev`               | dictionary     | `10.0, 1.0`                 | A pair of values that scale the standard deviation, after scaling the standard deviation `below` is subtracted from the image mean to give the below/lower threshold and the `above` is added to the image mean to give the above/upper threshold. These values should _always_ be positive.                                           |
|                   | `threshold_absolute`              | dictionary     | `-1.0, 1.0`                 | Below (first), above (second) absolute threshold for separating grains from the image background.                                                                                                                                                                                                                                      |
|                   | `direction`                       |                | `above`                     | Defines whether to look for grains above or below thresholds or both. Options: `above`, `below`, `both`                                                                                                                                                                                                                                |
|                   | `smallest_grain_size`             | int            | `50`                        | Catch-all value for the minimum size of grains. Measured in nanometres squared. All grains with area below than this value are removed.                                                                                                                                                                                                |
|                   | `absolute_area_threshold`         | dictionary     | `[300, 3000], [null, null]` | Area thresholds for above the image background (first) and below the image background (second), which grain sizes are permitted, measured in nanometres squared. All grains outside this area range are removed.                                                                                                                       |
|                   | `remove_edge_intersecting_grains` | boolean        | `true`                      | Whether to remove grains that intersect the image border. _Do not change this unless you know what you are doing_. This will ruin any statistics relating to grain size, shape and DNA traces.                                                                                                                                         |
| `grainstats`      | `run`                             | boolean        | `true`                      | Whether to calculate grain statistics. Options : `true`, `false`                                                                                                                                                                                                                                                                       |
|                   | `cropped_size`                    | float          | `40.0`                      | Force cropping of grains to this length (in nm) of square cropped images (can take `-1` for grain-sized box)                                                                                                                                                                                                                           |
|                   | `edge_detection_method`           | str            | `binary_erosion`            | Type of edge detection method to use when determining the edges of grain masks before calculating statistics on them. Options : `binary_erosion`, `canny`.                                                                                                                                                                             |
| `ordered_tracing` | `run`                             | boolean        | `true`                      | Whether to order the pruned skeletons of Disordered Traces. Options : true, false                                                                                                                                                                                                                                                      |
|                   | `ordering_method`                 | str            | `nodestats`                 | The method of ordering the disordered traces either using the nodestats output or solely the disordered traces. Options: `nodestats` or `topostats`.                                                                                                                                                                                   |
|                   | `pad_width`                       | int            | 10                          | Padding for individual grains when tracing. This is sometimes required if the bounding box around grains is too tight and they touch the edge of the image.                                                                                                                                                                            |
| `dnatracing`      | `run`                             | boolean        | `true`                      | Whether to run DNA Tracing. Options : true, false                                                                                                                                                                                                                                                                                      |
|                   | `min_skeleton_size`               | int            | `10`                        | The minimum number of pixels a skeleton should be for statistics to be calculated on it. Anything smaller than this is dropped but grain statistics are retained.                                                                                                                                                                      |
|                   | `skeletonisation_method`          | str            | `topostats`                 | Skeletonisation method to use, possible options are `zhang`, `lee`, `thin` (from [Scikit-image Morphology module](https://scikit-image.org/docs/stable/api/skimage.morphology.html)) or the original bespoke TopoStas method `topostats`.                                                                                              |
|                   | `spline_step_size`                | float          | `7.0e-9`                    | The sampling rate of the spline in metres. This is the frequency at which points are sampled from fitted traces to act as guide points for the splining process using scipy's splprep.                                                                                                                                                 |
|                   | `spline_linear_smoothing`         | float          | `5.0`                       | The amount of smoothing to apply to splines of linear molecule traces.                                                                                                                                                                                                                                                                 |
|                   | `spline_circular_smoothing`       | float          | `0.0`                       | The amount of smoothing to apply to splines of circular molecule traces.                                                                                                                                                                                                                                                               |
|                   | `pad_width`                       | int            | 10                          | Padding for individual grains when tracing. This is sometimes required if the bounding box around grains is too tight and they touch the edge of the image.                                                                                                                                                                            |
|                   | `cores`                           | int            | 1                           | Number of cores to use for tracing. **NB** Currently this is NOT used and should be left commented in the YAML file.                                                                                                                                                                                                                   |
| `plotting`        | `run`                             | boolean        | `true`                      | Whether to run plotting. Options : `true`, `false`                                                                                                                                                                                                                                                                                     |
|                   | `style`                           | str            | `topostats.mplstyle`        | The default loads a custom [matplotlibrc param file](https://matplotlib.org/stable/users/explain/customizing.html#the-matplotlibrc-file) that comes with TopoStats. Users can specify the path to their own style file as an alternative.                                                                                              |
|                   | `save_format`                     | string         | `null`                      | Format to save images in, `null` defaults to `png` see [matplotlib.pyplot.savefig](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html)                                                                                                                                                                           |
|                   | `savefig_dpi`                     | string / float | `null`                      | Dots Per Inch (DPI), if `null` then the value `figure` is used, for other values (typically integers) see [#further-customisation] and [Matplotlib](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html). Low DPI's improve processing time but can reduce the plotted trace (but not the actual trace) accuracy. |
|                   | `pixel_interpolation`             | string         | `null`                      | Interpolation method for image plots. Recommended default 'null' prevents banding that occurs in some images. If interpolation is needed, we recommend `gaussian`. See [matplotlib imshow interpolations documentation](https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html) for details.      |
|                   | `image_set`                       | string         | `all`                       | Which images to plot. Options : `all`, `core` (flattened image, grain mask overlay and trace overlay only).                                                                                                                                                                                                                            |
|                   | `zrange`                          | list           | `[0, 3]`                    | Low (first number) and high (second number) height range for core images (can take [null, null]). **NB** `low <= high` otherwise you will see a `ValueError: minvalue must be less than or equal to maxvalue` error.                                                                                                                   |
|                   | `colorbar`                        | boolean        | `true`                      | Whether to include the colorbar scale in plots. Options `true`, `false`                                                                                                                                                                                                                                                                |
|                   | `axes`                            | boolean        | `true`                      | Whether to include the axes in the produced plots.                                                                                                                                                                                                                                                                                     |
|                   | `num_ticks`                       | null / int     | `null`                      | Number of ticks to have along the x and y axes. Options : `null` (auto) or an integer >1                                                                                                                                                                                                                                               |
|                   | `cmap`                            | string         | `null`                      | Colormap/colourmap to use (defaults to 'nanoscope' if null (defined in `topostats/topostats.mplstyle`). Other options are 'afmhot', 'viridis' etc., see [Matplotlib : Choosing Colormaps](https://matplotlib.org/stable/users/explain/colors/colormaps.html).                                                                          |
|                   | `mask_cmap`                       | string         | `blu`                       | Color used when masking regions. Options `blu`, `jet_r` or any valid Matplotlib colour.                                                                                                                                                                                                                                                |
|                   | `histogram_log_axis`              | boolean        | `false`                     | Whether to plot hisograms using a logarithmic scale or not. Options: `true`, `false`.                                                                                                                                                                                                                                                  |
| `summary_stats`   | `run`                             | boolean        | `true`                      | Whether to generate summary statistical plots of the distribution of different metrics grouped by the image that has been processed.                                                                                                                                                                                                   |
|                   | `config`                          | str            | `null`                      | Path to a summary config YAML file that configures/controls how plotting is done. If one is not specified either the command line argument `--summary_config` value will be used or if that option is not invoked the default `topostats/summary_config.yaml` will be used.                                                            |

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
customised, for example we define custom colour maps `nanoscope` and `afmhot`. By default the former is configured to
be used. Other parameters that are customised are the `font.size` which affects axis labels and titles.

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

Whilst the overall look of images is controlled in this manner there is one additional file that controls how
images are plotted in terms of filenames, titles and image types and whether an image is part of the `core` subset
(flattened image, grain mask overlay and trace overlay) that are always generated or not.

This is the `topostats/plotting_dictionary.yaml` which for each image stage defines whether it is a component of the
`core` subset of images that are always generated, sets the `filename`, the `title` on the plot, the `image_type`
(whether it is a binary image), the `savefig_dpi` which controls the Dots Per Inch (essentially the resolution). Each
image has the following structure.

```yaml
z_threshed:
  title: "Height Thresholded"
  image_type: "non-binary"
  savefig_dpi: 100
  core_set: true
```

Whilst it is possible to edit this file it is not recommended to do so.

The following section describes how to override the DPI settings defined in this file and change the global `cmap`
(colormap/colourmap) used in plotting and output format.

#### DPI

During development it was found that setting high DPI globally for all images had a detrimental impact on processing
speeds, slowing down the overall processing time. The solution we have implemented is to use the
`topostats/plotting_dictionary.yaml` file and set the `savefig_dpi` parameter on a per-image basis.

If you wish to change the DPI there are two options, you can change the value for _all_ images by modifying the setting
in your a [custom configuration](#generating-a-configuration) by modifying the `savefig_dpi` from `null` to your desired
value. The example below shows a section of the configuration file you can generate and setting this value to `400`.

```yaml
plotting:
  run: true # Options : true, false
  style: topostats.mplstyle # Options : topostats.mplstyle or path to a matplotlibrc params file
  savefig_format: null # Options : null (defaults to png) or see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html
  savefig_dpi: 400 # Options : null (defaults to format) see https://afm-spm.github.io/TopoStats/main/configuration.html#further-customisation and https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html
```

The value in the configuration file (or the default if none is specified) can also be configured at run-time
using the `--savefig-dpi ###` option to the `topostats process`. This will over-ride both the default or any value
specified in a custom configuration you may have set. The following sets this to `400`

```bash
topostats process --savefig-dpi 400
```

**NB** Changing the DPI in this manner will apply to _all_ images and may significantly reduce processing speed as it
takes longer to write images with high DPI to disk.

If you wish to have fine grained control over the DPI on a per-image basis when batch processing then your only recourse
is to change the values in `topostats/plotting_dictionary.yaml`. Where this is depends on how you have installed
TopoStats, if it is from a clone of the Git repository then it can be found in
`TopoStats/topostats/plotting_dictionary.yaml`. If you have installed from PyPI using `pip install topostats` then it
will be under the virtual environment you have created
e.g. `~/.virtualenvs/topostats/lib/python3.11/site-packages/topostats/topostats/plotting_dictionary.yaml` if you are
using plain virtual environments or
`~/miniconda3/envs/topostats/lib/python3.11/site-packages/topostats/topostats/plotting_dictionary.yaml` if you are using
Conda environments and chose `~/miniconda3` as the base directory when installing Conda.

If you have installed TopoStats from the cloned Git repository the file will be under
`TopoStats/topostats/plotting_dictionary.yaml`.

**NB** The exact location will be highly specific to your system so the above are just guides as to where to find
things.

#### Colormap

The colormap used to plot images is set globally in `topostats/default_config.yaml`. TopoStats includes two custom
colormaps `nanoscope` and `afmhot` but any colormap recognised by Matplotlib can be used (see the [Matplotlib Colormap
reference](https://matplotlib.org/stable/gallery/color/colormap_reference.html) for choices).

If you want to modify the colormap that is used you have two options. Firstly you can [generate a
configuration](generating-a-configuration) file and modify the field `cmap` to your choice. The example below shows
changing this from `null` (which defaults to `nanoscope` as defined in `topostats.mplstyle`) to `rainbow`.

```yaml
plotting:
  ...
  cmap: rainbow # Colormap/colourmap to use (default is 'nanoscope' which is used if null, other options are 'afmhot', 'viridis' etc.)
```

Alternatively it is possible to specify the colormap that is used on the command line using the `--cmap` option to
`topostats process`. This will over-ride both the default or any value specified in a custom configuration you may have
set. The following sets this to `rainbow`.

```bash
topostats process --cmap rainbow
```

#### Saved Image format

Matplotlib, and by extension TopoStats, supports saving images in a range of different formats including `png`
([Portable Network Graphic](https://en.wikipedia.org/wiki/PNG)), `svg` ([Scalable Vector
Graphics](https://en.wikipedia.org/wiki/SVG)), `pdf` ([Portable Document
Format](https://en.wikipedia.org/wiki/PDF)), and `tif` ([Tag Image File
Format](https://en.wikipedia.org/wiki/TIFF)). The default is `png` but, as with both DPI and Colormap, these can be
easily changed via a custom configuration file or command line options to change these without having to edit the
[Matplotlib Style file](matplotlib-style). If using `tif` it is worth being aware that although the image will be saved,
this will be without metadata since this is not supported for `tif` files (see the note under `metadata` of [Matplotlib
savefig](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html)).

If you want to modify the output file format that is used you have two options. Firstly you can [generate a
configuration](generating-a-configuration) file and modify the field `savefig_format` to your choice. The example below
shows changing this from `null` (which defaults to `png` as defined in `topostats.mplstyle`) to `svg`.

```yaml
plotting:
  ...
  savefig_format: svg # Options : null (defaults to png) or see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html
```

Alternatively it is possible to specify the output image format that is used on the command line using the
`--savefig-format` option to `topostats process`. This will over-ride both the default or any value specified in a
custom configuration you may have set. The following sets this to `svg`.

```bash
topostats process --savefig-format svg
```

**NB** Note that these options are not mutually exclusive and can therefore be combined along with any of the other
options available to `topostats process`. The following would use a DPI of `400`, set the colormap to `rainbow` and the
output format to `svg` when running Topostats and would over-ride options in any custom configuration file or matplotlib
style file.

```bash
topostats process --savefig-dpi 400 --cmap rainbow --savefig-format svg
```

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
