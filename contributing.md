# Contributing

This document explains how, technically, to contribute to this project. A code of conduct may be available in the root of this repository, or included in the readme.

## Contribution Workflow

* Check the issues and draft pull requests to see if anyone is working on something similar.
* Make a branch for your contribution.
* Implement your feature, bug fix, documentation, etc. using commits.
* Push your changes.
* Make a pull request against the `master` branch of this repository for bug fixes, or the `dev` branch for new features.

You are advised to make a draft pull request as soon as you start work so nobody else ends up working on the same thing.

## Software Architecture

`topostats` is currently arranged as a Python module in the `topostats/` folder. Other scripts in this repository are not currently considered core elements of `topostats` apart from `Plotting.py`.

Currently the `topostats` module consists of:

* `default_config.ini` The default config file.
* `dnatracing.py` Applies tracing functions to each molecule.
* `pygwytracing.py` The "main" routine.
* `tracingfuncs.py` Skeletonises and generates backbone traces from masks.

The current working plan is to move to a more modular architecture with new (and existing) functionality being grouped by theme within files. We expect to add such files as:

* `filters.py` Raster image filters (e.g. Gaussian blur).
* `morphology.py` Morphological operations (e.g. identify connected components).
* `curves.py` Operations on vectorised "1D" shapes (e.g. determine curvature).
* `io.py` Input and output (e.g. load proprietory AFM data formats).

These can then be called by a "main" routine that performs batch analysis, and functions within them tested in isolation using `pytest` and reused in arbitrary contexts.

Object oriented approaches will be used where appropriate, but not seen as inherently superior to "functional" approaches which can be easier to maintain. Existing object orientation will be reviewed with this in mind.

## Coding Style

* [ ] There is no coding style.
* [x] [PEP8](https://www.python.org/dev/peps/pep-0008/) is the preferred coding style, but is not mandated.
* [ ] Code must be PEP8 compliant.

## Documentation

* [ ] There is no preferred style for documentation, but documentation is required.
* [ ] [PEP257 docstrings](https://www.python.org/dev/peps/pep-0257/) are required.
* [x] [Sphinx docstrings](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html) are required.

## [Typehints](https://docs.python.org/3/library/typing.html)

* [ ] Do not use typehints.
* [x] Typehints are optional, but encouraged.
* [ ] Typehints are required.

## Tests

* [x] The [`pytest`](https://docs.pytest.org/en/stable/) framework is used and tests are encouraged.
* [ ] Code must be covered by unit tests. The `pytest` framework is used.

## Static Analysis

* [x] Use [`pylint`](https://pypi.org/project/pylint/) to analyse your code before submission.

## Debug using docker

1. Install vscode.
2. Click "Open a Remote Window".
3. Click "Reopen in container".
4. Run the debugger as normal.


## Using pycharm and GitHub

Check the issues and draft pull requests.

![](assets/issues-prs.png)

Make a branch for your contribution.

![](assets/pycharm-branch.png)

Add any new files to `git` (i.e. stage them).

![](assets/stage.png)

Commit the changes (with a sensible message).

![](assets/commit.png)

> Add / commit multiple times to break your contribution into easy to follow chunks.

Push changes.

![](assets/push.png)

Make a pull request.

![](assets/pull-request.png)
