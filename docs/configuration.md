# Configuration

Configuration for TopoStats is done using a [YAML](https://yaml.org/) configuration file that is specified on the command line when
invoking. The current configuration file is provided in the TopoStats repository at
[`topostats/default_config.yaml`](https://github.com/AFM-SPM/TopoStats/blob/main/topostats/default_config.yaml) but
please be aware this may not work with your installed version, particularly if you installed from PyPI.

## Generating a configuration

You can always [generate a configuration file](usage#generating-configuration-file) appropriate for the version you have
installed (bar v2.0.0 as this option was added afterwards).

``` bash
run_topostats --create-config-file config.yaml
```

This produces the file `config.yaml` which contains comments indicating valid values for many of the
fields. If no configuration file is provided this default configuration is loaded automatically and used.

## Using a custom configuration

You can modify and edit a configuration and, once saved, you can run TopoStats with this configuration file as shown below.

``` bash
run_topostats --config my_config.yaml
```

On completion a copy of the configuration that was used is written to the output directory.


## YAML Structure

YAML files have key and value pairs, the first word, e.g. `base_dir` is the key this is followed by a colon to separate
it from the value that it takes, by default `base_dir` takes the value `./` (which means the current directory) and so
the entry in the file is a single line with `base_dir: ./`. Other data structures are available in YAML files including
nested values and lists.

A list in YAML consists of a key (e.g. `upper:`) followed by the values in square brackets separated by commas such as
`upper: [ 500, 800 ]`. This means the `upper` key is a list of the values `500` and `800`. Long lists can be split over
separate lines as shown below

``` yaml
upper:
  - 100
  - 200
  - 300
  - 400
```


## Fields



Aside from the comments in YAML file itself the fields are described below.


| Section      | Sub-Section                    | Data Type  | Default        | Description                                                                                                                                                                                                                                                  |
|:-------------|:-------------------------------|:-----------|:---------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `base_dir`   |                                | string     | `./`           | Directory to recursively search for files within.[^1]                                                                                                                                                                                                            |
| `output_dir` |                                | string     | `./output`     | Directory that output should be saved to.[^1]                                                                                                                                                                                                                    |
| `warnings`   |                                | string     | `ignore`       | Turns off warnings being shown.                                                                                                                                                                                                                              |
| `cores`      |                                | integer    | `2`            | Number of cores to run parallel processes on.                                                                                                                                                                                                                |
| `quiet`      |                                | false      |                |                                                                                                                                                                                                                                                              |
| `file_ext`   |                                | string     | `.spm`         | File extensions to search for.                                                                                                                                                                                                                               |
| `loading`    | `channel`                      | string     | `Height`       | The channel of data to be processed, what this is will depend on the file-format you are processing and the channel you wish to process.                                                                                                                     |
| `filter`     | `run`                          | boolean    | `true`         | Whether to run the filtering stage, without this other stages won't run so leave as `true`.                                                                                                                                                                  |
|              | `threshold_method`             | str        | `std_dev`      | Threshold method for filtering, options are `ostu`, `std_dev` or `absolute`.                                                                                                                                                                                 |
|              | `otsu_threshold_multiplier`    | float      | `1.0`          |                                                                                                                                                                                                                                                              |
|              | `threshold_std_dev`            | float      | ` 1.0`         |                                                                                                                                                                                                                                                              |
|              | `threshold_absolute_lower`     | float      | `-1.0`         |                                                                                                                                                                                                                                                              |
|              | `threshold_absolute_upper`     | float      | `1.0`          |                                                                                                                                                                                                                                                              |
|              | `gaussian_size`                | float      | `0.5`          | The number of standard deviations to build the Gaussian kernel and thus affects the degree of blurring. See [skimage.filters.gaussian](https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.gaussian) and `sigma` for more information |
|              | `gaussian_mode`                | string     | `nearest`      |                                                                                                                                                                                                                                                              |
| `grains`     | `run`                          | boolean    | `true`         | Whether to run grain finding. Options `true`, `false`                                                                                                                                                                                                        |
|              | `smallest_grain_size_nm2` | int        | `100`          | The smallest size of grains to be included (in pixels), anything smaller than this is considered noise and removed.                                                                                                                                          |
|              | `threshold_method`             | float      | `std_dev`      | Threshold method for grain finding.  Options : `otsu`, `std_dev`, `absolute`                                                                                                                                                                                 |
|              | `otsu_threshold_multiplier`    |            | `1.0`          | Factor by which the derived Otsu Threshold should be scaled.                                                                                                                                                                                                 |
|              | `threshold_std_dev`            |            | `1.0`          |                                                                                                                                                                                                                                                              |
|              | `  threshold_absolute_lower`   |            | `1.0`          |                                                                                                                                                                                                                                                              |
|              | `  threshold_absolute_upper`   |            | `1.0`          |                                                                                                                                                                                                                                                              |
|              | `absolute_area_threshold`      | dictionary |                |                                                                                                                                                                                                                                                              |
|              | `...upper`                     | list       | `[500,800]`    | Height above surface [Low, High] in nm^2 (also takes null)                                                                                                                                                                                                   |
|              | `...lower`                     |            | `[null, null]` | Height below surface [Low, High] in nm^2 (also takes null)                                                                                                                                                                                                   |
|              | `direction`                    |            | `upper`        | Defines whether to look for grains above or below thresholds or both. Options: `upper`, `lower`, `both`                                                                                                                                                      |
|              | `background`                   | float      | `0.0`          |                                                                                                                                                                                                                                                              |
| `grainstats` | `run`                          | boolean    | `true`         | Whether to calculate grain statistics. Options : `true`, `false`                                                                                                                                                                                             |
|              | `cropped_size`                 | float      | `40.0`         | Force cropping of grains to this length (in nm) of square cropped images (can take `-1` for grain-sized box)                                                                                                                                                 |
|              | `save_cropped_grains`          | boolean    | `true`         | Options : true, false                                                                                                                                                                                                                                        |
| `dnatracing` | `run`                          | boolean    | `true`         | Whether to run DNA Tracing.  Options : true, false                                                                                                                                                                                                           |
| `plotting`   | `run`                          | boolean    | `true`         | Whether to run plotting. Options : `true`, `false`                                                                                                                                                                                                           |
|              | `save_format`                  | string     | `png`          | Format to save images in, see [matplotlib.pyplot.savefig](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html)                                                                                                                          |
|              | `image_set`                    | string     | `all`          | Which images to plot. Options : `all`, `core`                                                                                                                                                                                                                |
|              | `zrange`                       | list       | `[0, 3]`       | Low and high height range for core images (can take [null, null])                                                                                                                                                                                            |
|              | `colorbar`                     | boolean    | `true`         | Whether to include the colorbar scale in plots. Options `true`, `false`                                                                                                                                                                                      |
|              | `axes`                         | boolean    | `true`         | Wether to include the axes in the produced plots.                                                                                                                                                                                                            |
|              | `cmap`                         | string     | `nanoscope`    | Colormap to use in plotting. Options : `nanoscope`, `afmhot`                                                                                                                                                                                                 |
|              | `histogram_log_axis`           | boolean    | `false`        | Whether to plot hisograms using a logarithmic scale or not. Options: `true`, `false`.
|              | `histogram_bins`               | int        |200             | Number of bins to use for histograms
|

## Validation

Configuration files are validated against a schema to check that the values in the configuration file are within the
expected ranges or valid parameters. This helps capture problems early and should provide informative messages as to
what needs correcting if there are errors.

[^1] When writing file paths you can use absolute or relative paths. On Windows systems absolute paths start with the
drive letter (e.g. `c:/`) on Linux and OSX systems they start with `/`. Relative paths are started either with a `./`
which denotes the current directory or one or more `../` which means the higher level directory from the current
directory. You can always find the current directory you are in using the `pwd` (`p`rint `w`orking `d`irectory). If
your work is in `/home/user/path/to/my/data` and `pwd` prints `/home/user` then the relative path to your data is
`./path/to/my/data`. The `cd` command is used to `c`hange `d`irectory.


``` bash
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
