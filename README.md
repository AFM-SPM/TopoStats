# TopoStats

<div align="center">

[![PyPI version](https://badge.fury.io/py/topostats.svg)](https://badge.fury.io/py/topostats)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/topostats)
[![Documentation Status](https://readthedocs.org/projects/topostats/badge/?version=dev)](https://topostats.readthedocs.io/en/dev/?badge=dev)
[![Code style:
Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Code style: flake8](https://img.shields.io/badge/code%20style-flake8-456789.svg)](https://github.com/psf/flake8)
[![codecov](https://codecov.io/gh/AFM-SPM/TopoStats/branch/dev/graph/badge.svg)](https://codecov.io/gh/AFM-SPM/TopoStats)
[![pre-commit.ci
status](https://results.pre-commit.ci/badge/github/AFM-SPM/TopoStats/main.svg)](https://results.pre-commit.ci/latest/github/AFM-SPM/TopoStats/main)
[![ORDA](https://img.shields.io/badge/ORDA--DOI-10.15131%2Fshef.data.22633528.v.1-lightgrey)](https://figshare.shef.ac.uk/articles/software/TopoStats/22633528/1)
[![fair-software.eu](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B-yellow)](https://fair-software.eu)

</div>
<div align="center">

[![Downloads](https://static.pepy.tech/badge/topostats)](https://pepy.tech/project/topostats)
[![Downloads](https://static.pepy.tech/badge/topostats/month)](https://pepy.tech/project/topostats)
[![Downloads](https://static.pepy.tech/badge/topostats/week)](https://pepy.tech/project/topostats)

</div>
<div align="center">

| [Installation](#installation) | [Tutorials and Examples](#tutorials-and-examples) | [Contributing](contributing.md) |
[Licence](#licence) | [Citation](#citation) |

</div>

---

An AFM image analysis program to batch process data and obtain statistics from images.

There is more complete documentation on the projects [documentation website](https://afm-spm.github.io/TopoStats/).

## Installation

TopoStats is available via PyPI and can be installed in your Virtual Environment with...

```bash
pip install topostats
```

For more on installation and how to upgrade please see the [installation
instructions](https://afm-spm.github.io/TopoStats/main/installation.html).

**NB** The minimum supported version of Python is >=3.10 and because of a constraint in a dependency the maximum
supported version is <= 3.11 (for now, we hope to support newer versions in the near future).

## How to Use

### Tutorials and Examples

For a full description of usage please refer to the [usage](https://afm-spm.github.io/TopoStats/main/usage.html) documentation.

A default configuration is loaded automatically that works with `.spm` files. The simplest method of processing images
is to run `topostats process` in the same directory as your scans _after_ having activated the virtual environment in
which you have installed TopoStats

```bash
topostats process
```

If you have files other than `.spm` please refer `topostats --help` and the documentation on how to process those images
with TopoStats.

**NB** If your configuration specifies `.spm` (the default) files with the old-style Bruker extension (i.e. `.001`,
`.002` etc.) will also be processed.

If you have your own YAML configuration file (see [Usage : Configuring
TopoStats](https://afm-spm.github.io/TopoStats/main/usage.html#configuring_topostats)) then invoke `topostats`
and use the argument for `--config <config_file>.yaml` that points to your file with an associated module of
TopoStats e.g. `process`.

```bash
# Edit and save my_config.yaml then run TopoStats with this configuration file
topostats --config my_config.yaml process
```

The configuration file is validated before analysis begins and if there are problems you will see errors messages that
are hopefully useful in resolving the error(s) in your modified configuration.

You can generate a sample configuration file using the `topostats create-config` argument which writes the default
configuration to the file `./config.yaml` (i.e. in the current directory). This will _not_ run any analyses.

### Notebooks

Example Jupyter Notebooks have been developed that show how to use TopoStats package interactively which is useful when
you are unsure of what parameters are most suited to your scans. Other notebooks exist which show how to produce plots
of the summary grain and tracing statistics or how to generate plots of scans from processed images which saves having
to run the processing again. See the documentation on
[Notebooks](https://afm-spm.github.io/TopoStats/main/notebooks.html) for further details.

## Contributing

Please refer to our [contributing guidelines](https://afm-spm.github.io/TopoStats/main/contributing.html) documentation.

## Licence

**This software is licensed as specified by the [GPL License](COPYING) and [LGPL License](COPYING.LESSER).**

## Citation

If you use TopoStats in your work or research please cite us. There is a [Citation File
Format](https://citation-file-format.github.io/) in this repository to aid citation.

### Publications

- [TopoStats - Atomic Force Microscopy image processing and
  analysis](https://orda.shef.ac.uk/articles/software/TopoStats_-_Atomic_Force_Microscopy_image_processing_and_analysis/22633528)
  [doi:10.15131/shef.data.22633528.v2](https://doi.org/10.15131/shef.data.22633528.v2)
- **Pre-Print** [Under or Over? Tracing Complex DNA Structures with High Resolution Atomic Force Microscopy |
  bioRxiv](https://www.biorxiv.org/content/10.1101/2024.06.28.601212v2) [doi:10.1101/2024.06.28.601212](https://doi.org/10.1101/2024.06.28.601212)
