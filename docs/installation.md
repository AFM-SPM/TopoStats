# Installing TopoStats

**NB** - The package is currently undergoing heavy revision and these installation instructions apply to installing and
workin on the the `dev` branch.

TopoStats is a [Python](https://www.python.org) package designed to run at the command line. You may have Python
installed on your system but should use a [Python Virtual
Environment](https://realpython.com/python-virtual-environments-a-primer/) such as
[Miniconda](https://docs.conda.io/en/latest/miniconda.html) and install it under the Virtual Environment. The versions
of Python supported are Python >=3.8 and so when creating your virtual environment you should specify this `3.8` as the
minimum.

## Setting up Conda

Once you have downloaded and installed [Miniconda](https://docs.conda.io/en/latest/miniconda.html) you can create a
virtual environment for installing TopoStats with a version of Python that meets the requirements with. We will call
this environment `topostats` (specified with the `--name topostats` option) and to use Python 3.10 (the option
`python=3.10`). After creating it we can, as per the instructions printed out, activate the environment.

``` bash
conda create --name topostats python=3.10
conda activate topostats
```



You are now ready to clone and install TopoStats.

**NB** If you are using an Apple M1 Macbook then you need to install Anaconda >=
[2022.05](https://www.anaconda.com/blog/new-release-anaconda-distribution-now-supporting-m1).

## Cloning and Installing

Currently TopoStats has not been released to the [PyPi](https://pypi.org) repository (although a release is planned). In
the mean time you have to clone the repository from GitHub and install it. If you do not have Git already installed
please see [Git](#git). Work is on-going on the `dev` branch and so
to test that you need to "checkout" this branch prior to installing it.

``` bash
git clone -b dev https://github.com/AFM-SPM/TopoStats.git
```
``` bash
cd TopoStats
```
``` bash
git checkout dev
```
``` bash
pip install .
```

If you wish to make changes to the code then install in editable mode, i.e. `pip install -e .`.

If you wish to develop features or address an existing [issue](https://github.com/AFM-SPM/TopoStats/issues) then you
should create a branch from `dev` and work on that before committing your changes and creating a pull request. The
suggested nomenclature for branches is your GitHub username followed by a slash, then the issue number and a short
description, e.g. `ns-rse/90-refactor-topostats`.

## Tests

One of the major changes in the refactoring is the introduction of unit tests. These require certain packages to be
installed which are not installed to your virtual environment by
[setuptools](https://setuptools.pypa.io/en/latest/setuptools.html) in the above steps. If you are develop and making
changes to the code base you will likely want to be able to run the tests. Install the necessary dependencies to do so
with...


``` bash
cd TopoStats
```
``` bash
git checkout dev
```
``` bash
pip install ".[tests]"
```
``` bash
pytest
```


# Git

[Git](https://git.vc) is a version control system for managing software development and is required to be installed on
your computer in order to clone the TopoStats repository. Instructions on installing Git can be found at [Git Guides -
install git](https://github.com/git-guides/install-git).
