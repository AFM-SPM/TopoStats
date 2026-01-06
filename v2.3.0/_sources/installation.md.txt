# Installation

**NB** - If you have trouble installing TopoStats please do checkout the
[discussion](https://github.com/AFM-SPM/TopoStats/discussions) for possible solutions. If your problem isn't covered
then please do not hesitate to ask a question.

TopoStats is a [Python](https://www.python.org) package designed to run at the command line. If you are using Microsoft
Windows you should use
[Powershell](https://learn.microsoft.com/en-us/powershell/scripting/learn/ps101/01-getting-started?view=powershell-7.3).
You may have Python installed on your system but should use a [Python Virtual
Environment](https://realpython.com/python-virtual-environments-a-primer/) such as
[Miniconda](https://docs.conda.io/en/latest/miniconda.html) and install and use TopoStats under the Virtual
Environment. The versions of Python supported are Python >=3.8 and so when creating your virtual environment you should
specify this `3.8` as the minimum.

## Setting up Conda

Once you have downloaded and installed [Miniconda](https://docs.conda.io/en/latest/miniconda.html) you can create a
virtual environment for installing TopoStats for installing and running TopoStats. We will call this environment `topostats`
(specified with the `--name topostats` option) and use Python 3.10 (the option `python=3.10`). After creating it we can,
as per the instructions printed out, activate the environment.

```bash
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

```bash
pip install topostats
```

This will install TopoStats under your virtual environment and the command `topostats` will be available at the
command line. It has a number of sub-commands which can be displayed by invoking it without any options. You can upgrade
`topostats` by using the `--upgrade` flag...

```bash
pip install --upgrade topostats
```

You can always install a specific version from PyPI

```bash
pip install topostats==2.0.0
```

For more information on using `pip` to install and manage packages please refer to the [pip
documentation](https://pip.pypa.io/en/stable/user_guide/).

### Installing from GitHub

You may wish to consider cloning and installing TopoStats from GitHub if...

- You wish to try out new features that have been developed since the last release (if you find problems please create
  an [issue](https://github.com/AFM-SPM/TopoStats/issues)).
- If you have found an issue in a released version and want to see if it has been fixed in the unreleased version.
- If you wish to develop and extend TopoStats with new features yourself.

There are two options to install from GitHub, which you use will depend on what you want to do.

1. Using PyPI to install directly.
2. Clone the repository and install from there.

If all you want to do is use the development version of TopoStats then you can use option 1. If you wish to change the
underlying code you should use option 2.

#### Installing from GitHub using PyPI

[`pip`][pip] supports [installing packages from GitHub][pip_github]. To install the `main` branch of TopoStats use the
following in your Virtual Environment.

```bash
pip install git+https://github.com/AFM-SPM/TopoStats.git@main
```

You can install any branch on GitHub by modifying the last argument (`@main`) to the branch you wish to install,
e.g. `@another_branch` would install the `another_branch` (if it existed).

#### Cloning the Repository and installing

If you do not have Git already installed please see [Git Installation](https://github.com/git-guides/install-git). If
you intend to contribute to the development of TopoStats please read through the [contributing](contributing) section.

If you are familiar with the command line then you can clone and install TopoStats with the following _after_ activating
your virtual environment. By installing in editable mode (with the `-e` flag) switching branches will make the branch
available.

```bash
cd ~/where/to/clone
git clone git@github.com:AFM-SPM/TopoStats.git
cd TopoStats
pip install -e .
```

If you plan to contribute to development by adding features or address an existing
[issue](https://github.com/AFM-SPM/TopoStats/issues) please refer to the [contributing](contributing) section and pay
particular attention to the section about installing additional dependencies.

We include [notebooks](notebooks) which show how to use different aspects of TopoStats. If you wish to try these out the
[Jupyter Noteooks](https://jupyter.org/) then you can install the dependencies that are required from the cloned
TopoStats repository using...

```bash
pip install ".[notebooks]"
```

#### Cloning Using GitKraken

If you are using GitKraken you can clone the repository by selecting "Clone" and then "GitHub.com" and typing
`TopoStats` into the box next to "Repository to Clone" and you should be presented with the option of selecting
"TopoStats" from the AFM-SPM organisation. Once cloned follow the above instructions to install with `pip` under your
virtual environment.

## Tests

One of the major changes in the refactoring is the introduction of unit tests. These require certain packages to be
installed which are not installed to your virtual environment by
[setuptools](https://setuptools.pypa.io/en/latest/setuptools.html) in the above steps. If you are intending to modify or
contribute to the development of TopoStats or make changes to the code base you will likely want to be able to run
the tests. Install the necessary dependencies to do so with...

```bash
cd TopoStats
git checkout dev
pip install ".[tests]"
pytest
```

## Git

[Git][git] is a version control system for managing software development and is required to be installed on
your computer in order to clone the TopoStats repository. Instructions on installing Git can be found at [Git Guides -
install git](https://github.com/git-guides/install-git).

A nice Graphical User Interface for working with Git is [GitKraken](https://www.gitkraken.com/) which includes
everything you need.

[git]: https://git.vc
[pip]: https://pypi.org/project/pip/
[pip_github]: https://pip.pypa.io/en/stable/getting-started/#install-a-package-from-github
