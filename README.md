# TopoStats

[![Documentation Status](https://readthedocs.org/projects/topostats/badge/?version=dev)](https://topostats.readthedocs.io/en/dev/?badge=dev)
[![codecov](https://codecov.io/gh/AFM-SPM/TopoStats/branch/dev/graph/badge.svg)](https://codecov.io/gh/AFM-SPM/TopoStats)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-360/)

| [How to use](#How-to-use) | [Installation](#installation) | [Tutorials and Examples](#tutorials-and-Examples) | [Contributing](contributing.md) | [Licence](#licence) | [Citation](#citation) |  |

An AFM image analysis program to batch process data and obtain statistics from images.

## How to Use

### Installation

Please see the [installation instructions](docs/installation.md).

### Tutorials and Examples

*todo: This will initially contain a simple run through, probably using `mincircle.spm`.*

> Currently Topostats is undergoing heavy revision. This involves removing the dependency on Gwyddion in favour of using [scikit-image](https://scikit-image.org/) to perform a number of steps in identifying and isolating features and summarising them. In turn this allows the package to move away from relying on Python 2.7 and instead run under Python >= 3.8.

In this development branch configuration is done through [YAML](https://yaml.org/) files. An example configuration file
is included in the directory `config/example.yml` under the `dev` branch and you can run this to process the included
`minicircle.spm` (found under `tests/resources/minicircle.spm`) with the following...

``` bash
run_topostats.py --config config/example.yaml
```

This version takes command line arguments, and you _have_ to include `--config path/to/valid/config.yaml` option. You
can see what other options are available with...

``` bash
run_topostats.py --help
```

Any options specified on the command line will over-ride those in the configuration file, for example to suppress log
messages and just have a progress bar you can over-ride the `quiet: false` option on the command line with.

``` bash
python ./run_topostats.py --config config/example.yaml --quiet True
```

## Contributing

See [contributing guidelines](contributing.md).

## Licence

**This software is licensed as specified by the [GPL License](COPYING) and [LGPL License](COPYING.LESSER).**

## Citation

- [TopoStats – A program for automated tracing of biomolecules from AFM images](https://www.sciencedirect.com/science/article/pii/S1046202321000207)
