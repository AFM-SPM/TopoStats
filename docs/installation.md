# Installation

**NB** - If you have trouble installing TopoStats please do checkout the
[discussion](https://github.com/AFM-SPM/TopoStats/discussions) for possible solutions. If your problem isn't covered
then please do not hesitate to ask a question.


TopoStats is a [Python](https://www.python.org) package designed to run at the command line. If you are using Microsoft
Windows you should use
[Powershell](https://learn.microsoft.com/en-us/powershell/scripting/learn/ps101/01-getting-started?view=powershell-7.3). You
may have Python installed on your system but should use a [Python Virtual
Environment](https://realpython.com/python-virtual-environments-a-primer/) such as
[Miniconda](https://docs.conda.io/en/latest/miniconda.html) and install it under the Virtual Environment. The versions
of Python supported are Python >=3.8 and so when creating your virtual environment you should specify this `3.8` as the
minimum.

## Setting up Conda

Once you have downloaded and installed [Miniconda](https://docs.conda.io/en/latest/miniconda.html) you can create a
virtual environment for installing TopoStats for installing and running TopoStats. We will call this environment `topostats`
(specified with the `--name topostats` option) and use Python 3.10 (the option `python=3.10`). After creating it we can,
as per the instructions printed out, activate the environment.

``` bash
conda create --name topostats python=3.10
conda activate topostats
```

You are now ready to install TopoStats.

**NB** If you are using an Apple M1 Macbook then you need to install Anaconda >=
[2022.05](https://www.anaconda.com/blog/new-release-anaconda-distribution-now-supporting-m1).

## Installing TopoStats

There are two options for installing TopoStats depending on your usage

1. [**Python Package Index**](https://pypi.org/) - appropriate if you are just using TopoStats and don't need to dig into
   the code.
2. Cloning the GitHub Repository - if you want to look at the code, contribute to it, debug errors or perhaps test a new
   feature before a release.

### PyPI Installation

After activating your `topostats` Conda environment you can install TopoStats from PyPI using the following command.

``` bash
pip install topostats
```

This will install TopoStats under your virtual environment and the command `run_topostats` will be available at the
command line. You can upgrade `topostats` by using the `--upgrade` flag...

``` bash
pip install --upgrade topostats
```

You can always install a specific version from PyPI

``` bash
pip install topostats==2.0.0
```

For more information on using `pip` to install and manage packages please refer to the [pip
documentation](https://pip.pypa.io/en/stable/user_guide/).

### Cloning from GitHub

**NB** Cloning and installing from GitHub is only required if you wish to contribute to or debug problems with
TopoStats, if you only intend on using it then please install from PyPI.

If you do not have Git already installed please see [Git](#git). If you intend to contribute to the development of
TopoStats please read through the [contributing](contributing) section.

If you are familiar with the command line then you can clone and install TopoStats with the following _after_ activating
your virtual environment.

``` bash
git clone https://github.com/AFM-SPM/TopoStats.git
# If you have SSH access configured to GitHub then you can use
git clone git@github.com:AFM-SPM/TopoStats.git
```


#### Cloning Using GitKraken

If you are using GitKraken you can clone the repository by selecting "Clone" and then "GitHub.com" and typing
`TopoStats` into the box next to "Repository to Clone" and you should be presented with the option of selecting
"TopoStats" from the AFM-SPM organisation.

Alternatively you can "Clone with URL" and enter `https://github.com/AFM-SPM/TopoStats.git` as the URL to clone from,
selecting a destination to clone to.


#### Installing TopoStats from the Cloned Repository

Once cloned you will have to open a Terminal and navigate to the directory you cloned and _after_ activating your
virtual environment install TopoStats with the following.

``` bash
cd /path/to/where/topostats/was/cloned/TopoStats
pip install .
```

If you wish to make changes to the code and test then make a `git branch`, make your changes and install in editable mode,
i.e. `pip install -e .`.

If you wish to develop features or address an existing [issue](https://github.com/AFM-SPM/TopoStats/issues) please refer
to the [contributing](contributing) section.

If you wish to run the [Jupyter Noteooks](https://jupyter.org/) that reside under `notebooks/` then you can install all
requirements using

``` bash
pip install .[notebooks]
```

## Tests

One of the major changes in the refactoring is the introduction of unit tests. These require certain packages to be
installed which are not installed to your virtual environment by
[setuptools](https://setuptools.pypa.io/en/latest/setuptools.html) in the above steps. If you are intending to modify or
contribute to the development of TopoStats and making changes to the code base you will likely want to be able to run
the tests. Install the necessary dependencies to do so with...


``` bash
cd TopoStats
git checkout dev
pip install ".[tests]"
pytest
```


## Git

[Git](https://git.vc) is a version control system for managing software development and is required to be installed on
your computer in order to clone the TopoStats repository. Instructions on installing Git can be found at [Git Guides -
install git](https://github.com/git-guides/install-git).

A nice Graphical User Interface for working with Git is [GitKraken](https://www.gitkraken.com/) which includes
everything you need.
