# Contributing

This document explains how, technically, to contribute to this project. A code of conduct may be available in the root of this repository, or included in the readme.

## Contribution Workflow

* Check the issues and draft pull requests to see if anyone is working on something similar.
* Make a fork of this repository.
* [Optionally] make a branch for your contribution.
* Implement your feature, bug fix, documentation, etc.
* Make a pull request against the `master` branch of this repository.

You are advised to make a draft pull request as soon as you start work so nobody else ends up working on the same thing.

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