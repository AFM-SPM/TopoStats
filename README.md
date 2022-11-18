# TopoStats

[![Documentation Status](https://readthedocs.org/projects/topostats/badge/?version=dev)](https://topostats.readthedocs.io/en/dev/?badge=dev)
[![codecov](https://codecov.io/gh/AFM-SPM/TopoStats/branch/dev/graph/badge.svg)](https://codecov.io/gh/AFM-SPM/TopoStats)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-360/)

| [How to use](#How-to-use) | [Installation](#installation) | [Tutorials and Examples](#tutorials-and-Examples) | [Contributing](contributing.md) | [Licence](#licence) | [Citation](#citation) |  |

An AFM image analysis program to batch process data and obtain statistics from images.

## How to Use

There is more complete documentation on the projects documentation website. This is hosted in two locations.

* [GitHub Pages : TopoStats](https://afm-spm.github.io/TopoStats/)
* [Readthedocs : TopoStats](https://topostats.readthedocs.io/en/dev/)

### Installation

TopoStats is available via PyPI and can be installed in your Virtual Environment with...

``` bash
pip install topostats
```

For more on installation please see the [installation instructions](https://afm-spm.github.io/TopoStats/installation.html)

### Tutorials and Examples

For a full description of usage please refer to the [usage](https://afm-spm.github.io/TopoStats/usage.html) documentation.

A default configuration is loaded automatically and so the simplest method of processing images is to run
`run_topostats` in the same directory as your scans _after_ having activated the virtual environment in which you have
installed TopoStats

``` bash
run_topostats
```

If you have your own YAML configuration file (see [Usage : Configuring
TopoStats](https://afm-spm.github.io/TopoStats/usage.html#configuring_topostats)) then invoke `run_topostats` and use
the argument for `--config <config_file>.yaml` that points to your file.

``` bash
# Edit and save my_config.yaml then run TopoStats with this configuration file
run_topostats --config my_config.yaml
```

The configuration file is validated before analysis begins and if there are problems you will see errors messages that
are hopefully useful in resolving the error(s) in your modified configuration.

## Contributing

See [contributing guidelines](https://afm-spm.github.io/TopoStats/contributing.html).

## Licence

**This software is licensed as specified by the [GPL License](COPYING) and [LGPL License](COPYING.LESSER).**

## Citation

- [TopoStats – A program for automated tracing of biomolecules from AFM images](https://www.sciencedirect.com/science/article/pii/S1046202321000207)
