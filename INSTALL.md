# Installing TopoStats

**NB** - The package is currently undergoing heavy revision and these installation instructions apply to installing and
workin on the the `dev` branch.

Currently TopoStats should be installed from the git repository, ideally under a [Python Virtual
Environment](https://realpython.com/python-virtual-environments-a-primer/)
(e.g. [Miniconda](https://docs.conda.io/en/latest/miniconda.html)).


``` bash
git clone git@github.com:AFM-SPM/TopoStats.git
cd TopoStats
git checkout dev
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
[setuptools](https://setuptools.pypa.io/en/latest/setuptools.html). To install these and run the tests you can...

``` bash
pip install ".[tests]"
pytest
```
