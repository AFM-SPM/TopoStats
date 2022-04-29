# Installing TopoStats

Currently TopoStats should be installed from the git repository, ideally under a [Python Virtual
Environment](https://realpython.com/python-virtual-environments-a-primer/)
(e.g. [Miniconda](https://docs.conda.io/en/latest/miniconda.html)).

**NB** - The package is currently undergoing heavy revision and this applies to the `dev` branch, the following will
install this branch.


``` bash
git clone git@github.com:AFM-SPM/TopoStats.git
cd TopoStats
git checkout dev
pip install .
```

If you wish to make changes to the code then install in editable mode, i.e. `pip install -e .`.

## Tests

One of the major changes in the refactoring is the introduction of unit tests. These require certain packages to be
installed which are not installed to your virtual environment by
[setuptools](https://setuptools.pypa.io/en/latest/setuptools.html). To install these and run the tests you can...

``` bash
pip install ".[tests]"
pytest
```
