# TopoStats

<div align="center">

[![PyPI version](https://badge.fury.io/py/topostats.svg)](https://badge.fury.io/py/topostats)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/topostats)
[![Documentation Status](https://readthedocs.org/projects/topostats/badge/?version=dev)](https://topostats.readthedocs.io/en/dev/?badge=dev)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json))](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Code style: flake8](https://img.shields.io/badge/code%20style-flake8-456789.svg)](https://github.com/psf/flake8)
[![codecov](https://codecov.io/gh/AFM-SPM/TopoStats/branch/dev/graph/badge.svg)](https://codecov.io/gh/AFM-SPM/TopoStats)
[![pre-commit.ci
status](https://results.pre-commit.ci/badge/github/AFM-SPM/TopoStats/main.svg)](https://results.pre-commit.ci/latest/github/AFM-SPM/TopoStats/main)
[![ORDA](https://img.shields.io/badge/ORDA--DOI-10.15131%2Fshef.data.22633528.v.1-lightgrey)](https://figshare.shef.ac.uk/articles/software/TopoStats/22633528/1)

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

## How to Use

### Tutorials and Examples

For a full description of usage please refer to the [usage](https://afm-spm.github.io/TopoStats/main/usage.html) documentation.

A default configuration is loaded automatically and so the simplest method of processing images is to run
`topostats process` in the same directory as your scans _after_ having activated the virtual environment in which you have
installed TopoStats

```bash
topostats process
```

If you have your own YAML configuration file (see [Usage : Configuring
TopoStats](https://afm-spm.github.io/TopoStats/main/usage.html#configuring_topostats)) then invoke `topostats process`
and use the argument for `--config <config_file>.yaml` that points to your file.

```bash
# Edit and save my_config.yaml then run TopoStats with this configuration file
topostats process --config my_config.yaml
```

The configuration file is validated before analysis begins and if there are problems you will see errors messages that
are hopefully useful in resolving the error(s) in your modified configuration.

You can generate a sample configuration file using the `topostats create-config` argument which writes the default
configuration to the file `./config.yaml` (i.e. in the current directory). This will _not_ run any analyses.

**NB** - This feature is only available in versions > v2.0.0 as it was introduced after v2.0.0 was released. In older
version > 2.0.0 and <= 2.1.2 you can use the older `run_topostats --create-config` option.

```bash
run_topostats --create-config-file config.yaml
```

### Notebooks

Example Jupyter Notebooks have been developed that show how to use TopoStats package interactively which is useful when
you are unsure of what parameters are most suited to your scans. Other notebooks exist which show how to produce plots
of the summary grain and tracing statistics or how to generate plots of scans from processed images which saves having
to run the processing again. See the documentation on
[Notebooks](https://afm-spm.github.io/TopoStats/main/notebooks.html) for further details.

## Contributing

See [contributing guidelines](https://afm-spm.github.io/TopoStats/main/contributing.html).

## Licence

**This software is licensed as specified by the [GPL License](COPYING) and [LGPL License](COPYING.LESSER).**

## Citation

Please use the [Citation File Format](https://citation-file-format.github.io/) which is available in this repository.

### Publications

- [TopoStats – A program for automated tracing of biomolecules from AFM images](https://www.sciencedirect.com/science/article/pii/S1046202321000207)
