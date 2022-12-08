# Usage

After having [installed](installation) TopoStats you are ready to run it. For convenience TopoStats provides a command
line interface `run_topostats` that will load a default configuration file and process all images with reasonable
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


## Running TopoStats - Test Run

The default location that TopoStats looks for scans is the directory from which it is invoked. Once you start your
shell/terminal you will therefore need to do two things.

1. Navigate to the location of the scans you wish to process using `cd /path/to/where/scans/are/located`.
2. Activate the virtual environment under which you installed TopoStats (refer to [installed](installation) if unsure).

You can now run topostats by invoking `run_topostats` and you should start to see some output similar to that below.

``` bash
cd /path/to/where/scans/are/located
run_topostats
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

## Configuring TopoStats - For Your Data

Configuration of TopoStats is done through a [YAML](https://yuaml.org/) file and a full description of the fields used
can be found under the [configuration](configuration) section.

Here we will go through common changes that you are likely to want to make to the default configuration and how to make
them.

### Copying `default_config.yaml`

If you have used Git to clone the TopoStats repository from GitHub the default configuration can be found in the
sub-directory `topostats/default_config.yaml`. 

If you have installed TopoStats from PyPI then a sample configuration
file can be downloaded from
[here](https://github.com/AFM-SPM/TopoStats/blob/a180198b369e68dd892038dca1893aa396b04d33/topostats/default_config.yaml), or copy and paste the contents into a text file ending with `.yaml`.

Save or copy this file to the folder containing all of your scan files and call it `my_config.yaml`. The command for this is below but you can also use the drag-and-drop feature on your desktop.

``` bash
cp /<path>/<to>/<where>/<topostats>/<is>/<cloned>/TopoStats/topostats/default_config.yaml /<where>/<scans>/<are>/my_config.yaml
```


### Editing `my_config.yaml`

**IMPORTANT** This file is an ASCII text file and  you should use NotePad (Windows), TextEdit (OSX) or Nano/Emacs/Vim
(GNU/Linux) or any other text editor. Do _not_ use Microsoft Word or any other Word Processor to edit this file.


You can now start customising the configuration you are going to run TopoStats with. All fields have defaults but the
ones you may want to change are....

* `base_dir` (default: `./`) the directory in which to search for scans. By default this is `./` which represents the
  folder from which `run_topostats` is called and it is good practice to have one configuration file per batch of
  scans that are being processed.
* `output_dir` (default: `output`) the location where the output is saved, by default this is the folder `output`
  which will be created if it doesn't exist. If you wish for the output to be somewhere else specify it here. If you
  want `Processed` directories to sit within the directories that images are found then simply set the `output_dir` to
  the same value as `base_dir`.
* `cores` (default: `4`) the number of parallel processes to run processing of all found images. Set this to a maximum
  of one less than the number of cores on your computers CPU. If unsure leave as is.
* `file_ext` (default: `.spm`) the file extension of scans to search for within the current directory. The default is
  `.spm` but other file format support is in the pipeline.
* `plotting` : `image_set` (default `core`) specifies which steps of the processing to plot images of. The value `all`
  gets images for all stages, `core** saves only a subset of images.


Most of the other configuration options can be left on their default values for now. Once you have made any changes save
the file and return to your terminal.

### Running TopoStats with `my_config.yaml`

To use your new configuration file you need to inform `run_topostats` to use that file rather than the defaults, this is
done using the `--config my_config.yaml` file.

**NB** this assumes that you are in the same directory as your scans where you have saved the `my_config.yaml` file that
you edited. That doesn't _have_ to be the case but it makes life easier for if you are not familiar with absolute
and relative paths.

``` bash
run_topostats --config my_configy.yaml
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

On a successful completion you should see output similar to this at the bottom.

``` bash
Processing images from tests, results are under output: 100%|XXXXXXXXXXXXXXXX| 1/1 [00:03<00:00,  3.60s/it][Tue, 15 Nov 2022 13:49:14] [INFO    ] [topostats] All statistics combined for 1 images(s) are saved to : output/all_statistics.csv}
[Tue, 15 Nov 2022 13:49:14] [INFO    ] [topostats] Unable to generate folderwise statistics as 'all_statis_df' is empty
[Tue, 15 Nov 2022 13:49:14] [INFO    ] [topostats] Writing configuration to : output/config.yaml
```

## Output

The output from running TopoStats is saved in the location defined in the configuration file by `output_dir`. The
default is the directory `output` within the directory from which `run_topostats` is invoked unless it has been modified
in a copy of the default configuration as described above.

At the top level of the output directory are two files `config.yaml` and `all_statistics.csv`

* `config.yaml` : a copy of the configuration used to process the images.
* `all_statistics.csv` : a Comma Separated Variable ASCII plain-text file of the grain and DNA tracing statistics.

The remaining directories of results is contingent on the structure of files within the `base_dir` that is specified in
the configuration. If all files are in the top-level directory (i.e. no nesting) then you will have just a `Processed`
directory. If there is a nested structure then there will be a `Processed` directory in each folder that an image with
the specified `file_ext` has been found. This is perhaps best illustrated by way of example.

If you have the following three `.spm` files within your current directory, one at the top level, one under `level1` and
one under `level1/a`...

``` bash
[4.0K Nov 15 13:55]  .
|-- [4.0K Nov 15 13:54]  ./level1
|   |-- [4.0K Nov 15 13:54]  ./level1/a
|   |-- [ 32M Nov 15 13:54]  ./level1/a/minicircle.spm
|   |-- [ 32M Nov 15 13:54]  ./level1/minicircle.spm
|-- [ 32M Nov 15 13:54]  ./minicircle.spm
```

...then under `output` (the default for`output_dir`) you will see the following directory structure.

``` bash
[4.0K Nov 15 14:06]  output
|-- [ 381 Nov 15 14:06]  output/all_statistics.csv
|-- [7.4K Nov 15 14:06]  output/config.yaml
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
directories `filters`, `grains/lower` and `grains/upper` which contain additional images from the processing stages and
an accompanying histogram for each image showing the distribution of pixel heights for that image.
