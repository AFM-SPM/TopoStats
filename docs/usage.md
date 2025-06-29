# Usage

After having [installed](installation) TopoStats you are ready to run it. For convenience TopoStats provides a command
line interface `topostats` that will load a default configuration file and process all images with reasonable
default configuration options.

However, because the location of your image files can not be known in advance you must make a copy of the default
configuration and modify it to work with your files. This guide will hopefully take you through the process of running
TopoStats and customising the configuration file with which it is run. If you encounter any problems please ask
questions in the [Discussions](https://github.com/AFM-SPM/TopoStats/discussions). If you think you have encountered a
bug or have a feature suggestion please create an [Issue](https://github.com/AFM-SPM/TopoStats/issues).

## Organising Scans

You should place all files you wish to batch process in a single directory. They can be nested in separate folders as
TopoStats will scan for all images within this directory but currently it will only process one scan type at a time
(i.e. `.spm` _or_ `.jpk` _or_ `.asd`). This may change in the future.

## Command Line Navigation

TopoStats currently runs as a command-line program. To use it you will have to use a "prompt" or "terminal" (they're
essentially the same thing). What you use will depend on your operating system, but the following are some simple
commands on navigation. If you use Windows then for consistency it is recommended to install and use
[PowerShell](https://learn.microsoft.com/en-us/powershell/scripting/install/installing-powershell).

At the command line you use `cd` to `c`hange `d`irectory to the location of your files. For example if your scans are on
the C-drive in `C:\User\me\work\spm\2022-12-08\scans` then you would

```bash
cd c:/User/me/work/spm/2022-12-08/scans
```

If you are on a Linux or OSX system then paths are not prefixed with letters and your files may be saved to
`/home/me/work/spm/2022-12-08/scans`. To change directory there you would...

```bash
cd /home/me/work/spm/2022-12-08/scans
```

**NB** - Always use a forward-slash (`/`) when typing directory paths. Windows will display back-slash (`\`) but
understands forward-slash. Under Linux and OSX they mean different things and so you should always use forward-slash
(`/`).

You can always find out what location you are at in the command line using the `pwd` command (`p`rint `w`orking
`d`irectory) and it will print out the directory you are currently at.

```bash
pwd
/home/me/work/spm/2022-12-08/scans
```

To navigate up one directory level use `cd ..`. These can be chained together and directories separated with `/`.

```bash
# Move up a single directory level
cd ..
pwd
/home/me/work/spm/2022-12-08
# Move up another two directory levels
cd ../../
pwd
/home/me/
```

You can list the files in a directory using the `ls` command.

```bash
ls
sample_image_scan_2022-12-08-1204.spm
```

To learn more about the command line see the [Introduction to the Command Line for
Genomics](https://datacarpentry.org/shell-genomics/).

## Running TopoStats

The default location that TopoStats looks for scans is the directory from which it is invoked. Once you start your
shell/terminal you will therefore need to do two things.

1. Navigate to the location of the scans you wish to process using `cd /path/to/where/scans/are/located`.
2. Activate the virtual environment under which you installed TopoStats (refer to [installed](installation) if unsure).

You can now run topostats by invoking `topostats process` and you should start to see some output similar to that below.

```bash
cd /path/to/where/scans/are/located
topostats process
[Tue, 15 Nov 2022 12:39:48] [INFO    ] [topostats] Configuration is valid.
[Tue, 15 Nov 2022 12:39:48] [INFO    ] [topostats] Plotting configuration is valid.
[Tue, 15 Nov 2022 12:39:48] [INFO    ] [topostats] Configuration file loaded from      : None
[Tue, 15 Nov 2022 12:39:48] [INFO    ] [topostats] Scanning for images in              : /home/neil/work/projects/topostats/TopoStats
[Tue, 15 Nov 2022 12:39:48] [INFO    ] [topostats] Output directory                    : output
[Tue, 15 Nov 2022 12:39:48] [INFO    ] [topostats] Looking for images with extension   : .spm
[Tue, 15 Nov 2022 12:39:48] [INFO    ] [topostats] Images with extension .spm in /home/neil/work/projects/topostats/TopoStats : 32
[Tue, 15 Nov 2022 12:39:48] [INFO    ] [topostats] Thresholding method (Filtering)     : std_dev
[Tue, 15 Nov 2022 12:39:48] [INFO    ] [topostats] Thresholding method (Grains)        : std_dev
...
```

On a successful completion you should see a message similar to the following which indicates various aspects of the run
along with information about how to give feedback, report bugs and cite the software.

```bash
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


  _______      _____      __ __       _____     ______    _______      _____      _______    ______
/\_______)\   ) ___ (    /_/\__/\    ) ___ (   / ____/\ /\_______)\   /\___/\   /\_______)\ / ____/\
\(___  __\/  / /\_/\ \   ) ) ) ) )  / /\_/\ \  ) ) __\/ \(___  __\/  / / _ \ \  \(___  __\/ ) ) __\/
  / / /     / /_/ (_\ \ /_/ /_/ /  / /_/ (_\ \  \ \ \     / / /      \ \(_)/ /    / / /      \ \ \
 ( ( (      \ \ )_/ / / \ \ \_\/   \ \ )_/ / /  _\ \ \   ( ( (       / / _ \ \   ( ( (       _\ \ \
  \ \ \      \ \/_\/ /   )_) )      \ \/_\/ /  )____) )   \ \ \     ( (_( )_) )   \ \ \     )____) )
  /_/_/       )_____(    \_\/        )_____(   \____\/    /_/_/      \/_/ \_\/    /_/_/     \____\/


[Thu, 14 Nov 2024 09:40:56] [INFO    ] [topostats]

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ COMPLETE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  Base Directory              : /home/neil/work/projects/topostats/TopoStats
  File Extension              : .spm
  Files Found                 : 1
  Successfully Processed      : 1 (100.0%)
  Configuration               : output/config.yaml
  All statistics              : output/all_statistics.csv
  Distribution Plots          : output/summary_distributions

  Email                       : topostats@sheffield.ac.uk
  Documentation               : https://afm-spm.github.io/topostats/
  Source Code                 : https://github.com/AFM-SPM/TopoStats/
  Bug Reports/Feature Request : https://github.com/AFM-SPM/TopoStats/issues/new/choose
  Citation File Format        : https://github.com/AFM-SPM/TopoStats/blob/main/CITATION.cff

  If you encounter bugs/issues or have feature requests please report them at the above URL
  or email us.

  If you have found TopoStats useful please consider citing it. A Citation File Format is
  linked above and available from the Source Code page.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```

## Help Options

The main `topostats` program has a number of flags which can be specified to change the behaviour of how the program
runs. You can view the possible options by supplying the `-h` or `--help` flag.

```bash
❱ topostats --help
usage: topostats [-h] [-v] [-c CONFIG_FILE] [-s SUMMARY_CONFIG] [--matplotlibrc MATPLOTLIBRC] [-b BASE_DIR] [-o OUTPUT_DIR] [-l LOG_LEVEL] [-j CORES] [-f FILE_EXT] [--channel CHANNEL] {process,load,filter,grains,grainstats,disordered-tracing,nodestats,ordered-tracing,splining,summary,create-config,create-matplotlibrc} ...

Run various programs relating to AFM data. Add the name of the program you wish to run.

options:
  -h, --help            show this help message and exit
  -v, --version         Report the current version of TopoStats that is installed
  -c CONFIG_FILE, --config-file CONFIG_FILE
                        Path to a YAML configuration file.
  -s SUMMARY_CONFIG, --summary-config SUMMARY_CONFIG
                        Path to a YAML configuration file for summary plots and statistics.
  --matplotlibrc MATPLOTLIBRC
                        Path to a matplotlibrc file.
  -b BASE_DIR, --base-dir BASE_DIR
                        Base directory to scan for images.
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Output directory to write results to.
  -l LOG_LEVEL, --log-level LOG_LEVEL
                        Logging level to use, default is 'info' for verbose output use 'debug'.
  -j CORES, --cores CORES
                        Number of CPU cores to use when processing.
  -f FILE_EXT, --file-ext FILE_EXT
                        File extension to scan for.
  --channel CHANNEL     Channel to extract.

program:
  Available programs, listed below:

  {process,load,filter,grains,grainstats,disordered-tracing,nodestats,ordered-tracing,splining,summary,create-config,create-matplotlibrc}
    process             Process AFM images. Additional arguments over-ride defaults or those in the configuration file.
    load                Load and save all images as .topostats files for subsequent processing.
    filter              Load and filter images, saving as .topostats files for subsequent processing.
    grains              Load filtered images from '.topostats' files and detect grains.
    grainstats          Load images with grains from '.topostats' files and calculate statistics.
    disordered-tracing  WIP DO NOT USE - Skeletonise and prune objects to disordered traces.
    nodestats           WIP DO NOT USE - Calculate node statistics and disentangle molecules.
    ordered-tracing     WIP DO NOT USE - Ordered traces of pruned skeletons.
    splining            WIP DO NOT USE - Splining of traced molecules to produce smooth curves.
    summary             Plotting and summary of TopoStats output statistics.
    create-config       Create a configuration file using the defaults.
    create-matplotlibrc
                        Create a Matplotlibrc parameters file using the defaults.
```

The global flags/options for modifying behaviour are listed. You then need to provide the name of the program you wish
to run which are listed at the bottom of the output along with a description.

Each sub-program has its own specific set of options too which can be specified to override the settings in the
configuration file that is loaded (either the default or the user specified configuration). To view these again use the
`-h` or `--help` flag. For a more detailed description of the options see the [configuration](configuration) page.

```bash
 ❱ topostats create-config --help
usage: topostats create-config [-h] [-f FILENAME] [-o OUTPUT_DIR] [-c CONFIG] [-s]

Create a configuration file using the defaults.

options:
  -h, --help            show this help message and exit
  -f FILENAME, --filename FILENAME
                        Name of YAML file to save configuration to (default 'config.yaml').
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Path to where the YAML file should be saved (default './' the current directory).
  -c CONFIG, --config CONFIG
                        Configuration to use, currently only one is supported, the 'default'.
  -s, --simple          Create a simple configuration file with only the most common options.
```

**NB** The `process` program has a _lot_ of options as it runs the processing pipeline in full.

### Reducing Output

If you find the output too verbose or of no use you can reduce it by setting the `log_level` to either `error` or
`warning`. This can be done either in the configuration file (see [Configuration](configuration.md) below)
or using the `-l`/`--log-level` flag for example `topostats process --log_level warning`.

## Configuring TopoStats

Configuration of TopoStats is done through a [YAML](https://yaml.org/) file and a full description of the fields used
can be found under the [configuration](configuration.md) section.

Here we will go through generating a configuration file to edit and some of the common changes that you are likely to
want to make to the default configuration and how to make them.

### Generating Configuration File

TopoStats will use some reasonable default parameters by default, but typically you will want to customise the
parameters that are used. This is achieved using a [configuration](configuration) file. This is a
[YAML](https://yaml.org) file that contains parameters for different settings. For convenience you can generate
a sample configuration file in your current working directory using the `topostats create-config` sub-command. It
takes a single argument, the name of the file to save the configuration to (e.g. `config.yaml` or `settings.yaml`), and
it will write the current default configuration to that file.

```bash
topostats create-config --filename my_config.yaml
ls -l
my_config.yaml
sample_image_scan_2022-12-08-1204.spm
```

You can now edit and/or rename the `my_config.yaml`. It can be called anything you want,
e.g. `todays_first_run_configuration.yaml` is a valid name.

### Editing `config.yaml`

**IMPORTANT** This file is an ASCII text file and you should use NotePad (Windows), TextEdit (OSX) or Nano/Emacs/Vim
(GNU/Linux) or any other text editor. Do _not_ use Microsoft Word or any other Word Processor to edit this file.

You can now start customising the configuration you are going to run TopoStats with. All fields have defaults but the
ones you may want to change are....

- `base_dir` (default: `./`) the directory in which to search for scans. By default this is `./` which represents the
  directory from which `topostats process` is called and it is good practice to have one configuration file per batch of
  scans that are being processed.
- `output_dir` (default: `output`) the location where the output is saved, by default this is the directory `output`
  which will be created if it doesn't exist. If you wish for the output to be somewhere else specify it here. If you
  want `Processed` directories to sit within the directories that images are found then simply set the `output_dir` to
  the same value as `base_dir`.
- `log_level` (default: `info`) the verbosity of output to the console and log file, the options in order of verbosity
  are `debug` > `info` > `warning` > `error`. If you want less output set to `warning` or `error`. If you encounter
  errors please set to `debug` and run again and include the log in your [bug
  report](https://github.com/AFM-SPM/TopoStats/issues/new?assignees=&labels=bug&template=bug_report.md&title=).
- `cores` (default: `2`) the number of parallel processes to run processing of all found images. Set this to a maximum
  of one less than the number of cores on your computers CPU. If unsure leave as is, but chances are you can increase
  this to at least `4` quite safely.
- `file_ext` (default: `.spm`) the file extension of scans to search for within the current directory. The default is
  `.spm` but other file format support is in the pipeline.
- `plotting` : `image_set` (default `core`) specifies which steps of the processing to plot images of. The value `all`
  gets images for all stages, `core` saves only a subset of images.

Most of the other configuration options can be left on their default values for now. Once you have made any changes save
the file and return to your terminal.

### Running TopoStats with `my_config.yaml`

To use your new configuration file you need to inform `topostats process` to use that file rather than the defaults,
this is done using the `--config config.yaml` file.

**NB** this assumes that you are in the same directory as your scans where you have saved the `my_config.yaml` file that
you edited. That doesn't _have_ to be the case but it makes life easier for if you are not familiar with absolute
and relative paths.

```bash
topostats --config my_config.yaml process
[Tue, 15 Nov 2022 12:39:48] [INFO    ] [topostats] Configuration is valid.
[Tue, 15 Nov 2022 12:39:48] [INFO    ] [topostats] Plotting configuration is valid.
[Tue, 15 Nov 2022 12:39:48] [INFO    ] [topostats] Configuration file loaded from      : None
[Tue, 15 Nov 2022 12:39:48] [INFO    ] [topostats] Scanning for images in              : /home/neil/work/projects/topostats/TopoStats
[Tue, 15 Nov 2022 12:39:48] [INFO    ] [topostats] Output directory                    : output
[Tue, 15 Nov 2022 12:39:48] [INFO    ] [topostats] Looking for images with extension   : .spm
[Tue, 15 Nov 2022 12:39:48] [INFO    ] [topostats] Images with extension .spm in /home/neil/work/projects/topostats/TopoStats : 1
[Tue, 15 Nov 2022 12:39:48] [INFO    ] [topostats] Thresholding method (Filtering)     : std_dev
[Tue, 15 Nov 2022 12:39:48] [INFO    ] [topostats] Thresholding method (Grains)        : std_dev
...
```

On successful completion you should see the same message noted above.

### Renaming Bruker Files

Older Bruker devices output files with somewhat uninformative numerical file extensions (e.g. `.001`, `.002`, `.003`
etc.). TopoStats will automatically detect and process these if the configuration file is `file_ext: .spm` (or on the
command line the `--file-ext/-f .spm` option is used). However, for convenience a sub-processor is provided which will
append the suffix `.spm` to all files found recursively with numerical extensions in the specified directory.

```bash
topostats --base-dir /path/to/a/directory bruker-rename
[Tue, 24 Jun 2025 12:18:15] [INFO    ] [topostats] The YAML configuration file is valid.
[Tue, 24 Jun 2025 12:18:15] [INFO    ] [topostats] Total Bruker files found : 5
[Tue, 24 Jun 2025 12:18:15] [INFO    ] [topostats] Old style files found    : 5
[Tue, 24 Jun 2025 12:18:15] [INFO    ] [topostats] Renaming files...
[Tue, 24 Jun 2025 12:18:15] [INFO    ] [topostats] test.002 > test.002.spm
[Tue, 24 Jun 2025 12:18:15] [INFO    ] [topostats] test.001 > test.001.spm
[Tue, 24 Jun 2025 12:18:15] [INFO    ] [topostats] nested/directory/test.004 > nested/directory/test.004.spm
[Tue, 24 Jun 2025 12:18:15] [INFO    ] [topostats] nested/directory/test.999999 > nested/directory/test.999999.spm
[Tue, 24 Jun 2025 12:18:15] [INFO    ] [topostats] nested/directory/test.003 > nested/directory/test.003.spm
```

## Output

The output from running TopoStats is saved in the location defined in the configuration file by `output_dir`. The
default is the directory `output` within the directory from which `topostats process`. This may differ if you have
used your own customised configuration file (specifically if you have modified the `output_dir:` option).

At the top level of the output directory are a few files produced:

- `config.yaml` : a copy of the configuration used to process the images.
- `all_statistics.csv` : a Comma Separated Variable ASCII plain-text file of the grain statistics.
- `all_disordered_segment_statistics.csv` : a Comma Separated Variable ASCII plain-text file of the branched skeleton
  statistics.
- `all_mol_statistics.csv` : a Comma Separated Variable ASCII plain-text file of the molecule statistics.

**Note:** - If all grains / branch segments of a column have a `None` or `NaN` value, the column will not be present in
the output `.csv` file.

The remaining directories of results is contingent on the structure of files within the `base_dir` that is specified in
the configuration. If all files are in the top-level directory (i.e. no nesting) then you will have just a `Processed`
directory. If there is a nested structure then there will be a `Processed` directory in each folder that an image with
the specified `file_ext` has been found. This is perhaps best illustrated by way of example.

If you have the following three `.spm` files within your current directory, one at the top level, one under `level1` and
one under `level1/a`...

```bash
[4.0K Nov 15 13:55]  .
|-- [4.0K Nov 15 13:54]  ./level1
|   |-- [4.0K Nov 15 13:54]  ./level1/a
|   |-- [ 32M Nov 15 13:54]  ./level1/a/minicircle.spm
|   |-- [ 32M Nov 15 13:54]  ./level1/minicircle.spm
|-- [ 32M Nov 15 13:54]  ./minicircle.spm
```

...then under `output` (the default for`output_dir`) you will see the following directory structure.

```bash
[4.0K Nov 15 14:06]  output
|-- [ 381 Nov 15 14:06]  output/all_statistics.csv
|-- [ 733 Nov 15 14:06]  output/all_disordered_tracing_statistics.csv
|-- [ 254 Nov 15 14:06]  output/all_mol_statistics.csv
|-- [7.4K Nov 15 14:06]  output/config.yaml
|-- [ 222 Nov 15 14:06]  output/image_stats.csv
|-- [4.0K Nov 15 14:06]  output/level1
|   |-- [4.0K Nov 15 14:06]  output/level1/a
|   |   |-- [4.0K Nov 15 14:06]  output/level1/a/Processed
|   |-- [4.0K Nov 15 14:06]  output/level1/Processed
|-- [4.0K Nov 15 14:06]  output/Processed
```

...where there is one `Processed` directory at the sub-directory level that each image was found.

**NB** If you want `Processed` directories to sit within the directories that images are found then simply set the `output_dir`
to the same value as `base_dir`.

Within each `Processed` directory is a directory for each file found with the specified `file_ext` and within these are
the resulting images from processing scans. If the `plotting` : `image_set` is `core` then there is a single image for
each. If this option is `all` then there is also a sub-directory for each image found within which there are the
directories `filters`, `grains/below` and `grains/above` which contain additional images from the processing stages and
an accompanying histogram for each image showing the distribution of pixel heights for that image.

### `.topostats` files

TopoStats has implemented its own file format for saving data with the extension `.topostats`. This is an [HDF5][hdf5]
file that uses a hierarchical structure for storing data. Initially an independent versioning system was implemented to
denote the version of files, however after `0.2` of the file format indicator and release `v2.4.0` of TopoStats we
switched to storing the version of TopoStats that the file was processed with (this aids with reproducibility and aligns
with the [FAIR for Research Software Principles][fair4rs]).

By saving data in this format it makes it easy to carry out "experiments" on the by re-running processing steps without
having to re-run the whole pipeline (currently a work in progress).

If you want to look at the structure of these files you can use [h5glance][h5glance] at the command line or load them
into Notebooks. The structure is fairly self-explanatory as its based on a dictionary structure with descriptive keys
denoting what the contents are (typically Numpy arrays).

Originally (`< 2.4.0` and upto `0.2`) the structure was orientated around the stage of processing. Subsequent to this
(`>= 2.4.0`) the unit of organisation became the individual detected grains and their sub-molecules/grains.

## Summary Plots

By default TopoStats will take the data that has been summarised across all files and generate a series of plots,
[histograms](https://en.wikipedia.org/wiki/Histogram) with [Kernel Density Estimates
(KDE)](https://en.wikipedia.org/wiki/Kernel_density_estimation) overlaid and [Violin
plots](https://en.wikipedia.org/wiki/Violin_plot). The default location of these if no custom configuration file is used
is `output/summary_distributions`. If you have used a custom configuration file it will be the sub-directory
`summary_distributions` nested under the directory specified for the `output`, e.g. if you used the current directory as
output you will have a `summary_distributions` directory present.

Sometimes you may have a `all_statistics.csv` from a run and wish to plot distributions of additional statistics that
were not already plotted. This can be achieved using the command line program `toposum` which is included.

**NB** Because of the inherent complexity of plots this script is, by design, limited in the scope to which plots can be
configured. It uses the plotting library [Seaborn](https://seaborn.pydata.org/) (which is built on top of
[Matplotlib](https://matplotlib.org/)) to produce basic plots, which are not intended for publication. If you want to
tweak or customise plots it is recommended to load `all_statistics.csv` into a [Jupyter Notebook](https://jupyter.org)
and generate the plots you want there. A sample notebook is included to show how to do this.

### Configuring Summary Plots

Configuration of summary plots is also via a YAML configuration file a description of the fields can be found under
[configuration](configuration.md#summary-configuration) page. You can generate a sample configuration by invoking the
`--create-config-file` option to `toposum`

```bash
toposum --create-config-file custom_summary_config.yaml
```

The file `custom_summary_config.yaml` can then be edited to change what plots are generated, where they are saved to and
so forth. Typically you will only want to adjust a few settings such as toggling the types of plots (`hist`, `kde` and
`violin`), the number of `bins` in a histogram or the statistic to plot in histograms (`count`, `frequency` etc.). You
can change the `palette` that is used by Seaborn and crucially toggle which statistics are summarised by commenting
and uncommenting the statistic names under `stats_to_sum`.

### Labels

Labels for the plots are generated from the file `topostats/var_to_label.yaml` which provides a dictionary that maps the
variable name as the dictionary `key` to its description stored in the dictionary `value`. If you wish to customise
these you can do so and pass it to `toposum` using the `--plotting_dictionary` which takes as an argument the path to
the file you have created.

### Pickles

The option `pickle_plots: True` will save to the specified `output_dir` the file `distribution_plots.pkl` which is a
binary format that saves the plots that have been generated and saved in nested dictionaries so that they can be loaded
again. The Notebook `notebooks/02-Summary-statistics-and-plots.ipynb` shows how to load these and make simple
modifications to the the plots.

[fair4rs]: https://www.nature.com/articles/s41597-022-01710-x
[h5glance]: https://pypi.org/project/h5glance/
[hdf5]: https://www.hdfgroup.org/solutions/hdf5/
