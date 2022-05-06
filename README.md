# TopoStats

[![Documentation Status](https://readthedocs.org/projects/topostats/badge/?version=dev)](https://topostats.readthedocs.io/en/dev/?badge=dev)
[![codecov](https://codecov.io/gh/AFM-SPM/TopoStats/branch/main/graph/badge.svg)](https://codecov.io/gh/AFM-SPM/TopoStats)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-360/)

| [How it works](#how-it-works) | [Run using Docker](#run-using-docker) | [Parameter Configuration](#parameter-configuration) | [Licence](#licence) | [Publications](#publications) | [Contributing](contributing.md) |

An AFM image analysis program to batch process data and obtain statistics from images

## How it works

Image progessing is performed using the 'pygwytracing' script

The algorithm searches recursively for files within a user-defined directory.
This search also excludes any files of the format ‘_cs’ which are cropped files exported by the Nanoscope Analysis software.
AFM images are loaded  using gwyddion functions and topography data is automatically selected using the choosechannels function.
The pixel size and dimensions of each image are determined using the imagedetails function, which allows all inputs to be specified in real, i.e. nanometre values, in place of pixel values.
This is especially important for datasets with changing resolution.

Basic image processing is performed in the function editfile which uses the functions: ‘align rows’ to remove offsets between scan lines; ‘level’ to remove sample tilt as a first order polynomial; ‘flatten base’ which uses a combination of facet and polynomial levelling with automated masking; and ‘zeromean’ which sets the mean value of the image, i.e. the background, to zero.
A gaussian filter (sigma = 1.5) of 3.5 pixels (1-2 nm) is applied to remove pixel errors and high frequency noise.

Single DNA molecules are identified in images using a modified extension of Gwyddion’s automated masking protocols, in which masks are used to define the positions of individual features (grains) on the imaged surface.
The grains within a flattened AFM image are identified using the ‘mask_outliers’ function, which masks data  points with height values that deviate from the mean by more than 1sigma (with 3sigma corresponding to a standard gaussian).
Grains which touch the edge of the image (i.e. are incomplete) are removed using the ‘grains_remove_touching_border’ function and grains which are smaller than 200 nm2 are removed using the ‘grains_remove_by_size’ function.
Erroneous grains are removed using the removelargeobjects and removesmallobjects functions, which themselves use the function find_median_pixel_area to determine the size range of objects to remove. The ‘grains_remove_by_size’ function is then called again to remove grains which fall outside 50 % - 150 % of the median grain area determined in the previous step.

Grain statistics are then calculated for each image using the grainanalysis function which utilises the ‘grains_get_values’ function to obtain a number of statistical properties which are saved using the saveindividualstats function as ‘.json’ and ‘.txt’ files for later use in a subdirectory ‘GrainStatistics’ in the specified path.
In addition, each grain’s values are appended to an array [appended_data], to statistically analyse the morphologies of DNA molecules from all images for a given experiment (presumed to be within a single  directory).
This array is converted to a pandas dataframe using the getdataforallfiles function and saved out using the savestats function as ‘.json’ and ‘.txt’ files with the name of the directory in the original path.

Individual grains (i.e. isolated molecules) are cropped out using the function bbox, which uses the grain centre x and y positions obtained in the grainanalysis function to duplicate the original image and crop it to a predefined size (here 80 nm) around the centre of the grain. These images are then labelled with the grain ID and saved out as tiff files in a subdirectory ‘Cropped’ in the specified path.
To allow for further processing in python, there is an option to obtain the image or mask as a numpy array41, using the function exportasnparray. The processed image, and a copy with the mask overlaid are saved out using the savefiles function to a subdirectory ‘Processed’ in the specified path.

Statistical analysis and plotting is performed using the 'statsplotting' script.
This script uses the importfromjson function to import the JSON format file exported by pygwytracing and calculates various statistical parameters for all grain quantities, e.g. length, width and saves these out as a new JSON file using the savestats function.
Both KDE plots and histograms are generated for any of the grain quantities using the matplotlib42 and seaborn43 libraries within the functions plotkde, plotcolumns and plothist.

## Run using Docker

Firstly, you will need to clone the TopoStats repository if you have not done so. It can be cloned by opening a terminal window and navigating to where you want to save TopoStats, then running the command:

```
git clone https://github.com/AFM-SPM/TopoStats.git
```

If this does not work, then you may have to install git. Alternatively, you can just download and extract the zip file from GitHub directly.

Docker conatiners provide a way to run software in a controlled environment, without having to install lots of packages on your computer. To experiment with TopoStats using Docker you will first need to install [Docker](https://docs.docker.com/get-docker/). Then, using the command line, pull the container image to your computer:

```
docker pull afmspm/topostats:x
```

This allows docker to know how and what to run as a virtual machine. Docker then needs to be able to access TopoStats on your operating system. This can be done by mapping the TopoStats folder on your operating system to a folder in the virtual machine. **The files that you wish to analyse will need to be in your TopoStats folder.** The following command both maps the folders and starts the virtual machine:

```
docker run -it -v <path/to/your/topostats>:/home/TopoStats afmspm/topostats:x
```

This should have changed the location stated in the terminal from 'C:\ 'to something like 'root@', meaning that any commands will be running in the Docker container.

Navigate to the mapped folder in the virtual machine using:

```
cd home/TopoStats
```

Install the package:

```
pip install -e .
```

Then finally run TopoStats, using the command:

```
python -m topostats
```

This should run the pygwytracing.py script for TopoStats, targetting the repository folder that was mapped on your operating system, looking for spm files.

The terminal will output information about the status of the TopoStats script. Any results will be placed in a folder in the TopoStats folder, called 'processed'.

### Testing

To run tests:

```
python -m pytest
```

## Parameter Configuration

The parameters that the software uses for analysis of the data can be configured in `Config.ini` by simply opening the file in a text editor and changing the variables. You do not need to edit the code to change the parameters.

When updating TopoStats, the `Config.ini` file is ignored, so your parameters are maintained. Different sets of parameters can be saved for different sample types in the config file, and the sections for the different sample types are labelled in square brackets.

If no config file is found while running TopoStats, it will make a copy of the default config file.

## Development

Currently Topostats is undergoing heavy revision. This involves removing the dependency on Gwyddion in favour of using
[scikit-image](https://scikit-image.org/) to perform a number of steps in identifying and isolating features and
summarising them. In turn this allows the package to move away from relying on Python 2.7 and instead run under Python
>= 3.8.

To test the development version you should clone the repository and checkout the `dev` branch and install it in editable
mode.

``` bash
git clone https://github.com/AFM-SPM/TopoStats.git
cd TopoStats
git checkout dev
pip install -e .
```

If you want to run the tests then install their dependencies with...

``` bash
pip install ".[tests]"
```

In this development branch configuration is done through [YAML](https://yaml.org/) files. An example configuration file
is included in the directory `config/example.yml` under the `dev` branch and you can run this to process the included
`minicircle.spm` (found under `tests/resources/minicircle.spm`) with the following...

``` bash
python topostats/topotracing.py --config config/example.yaml
```

This version takes command line arguments, and you _have_ to include `--config path/to/valid/config.yaml` option. You
can see what other options are available with...

``` bash
python topostats/topotracing.py --help
```

Any options specified on the command line will over-ride those in the configuration file, for example to suppress log
messages and just have a progress bar you can over-ride the `quiet: false` option on the command line with.

``` bash
python topostats/topotracing.py --config config/example.yaml --quiet True
```

## Licence

**This software is licensed as specified by the [GPL License](COPYING) and [LGPL License](COPYING.LESSER).**

## Publications

- [TopoStats – A program for automated tracing of biomolecules from AFM images](https://www.sciencedirect.com/science/article/pii/S1046202321000207)
