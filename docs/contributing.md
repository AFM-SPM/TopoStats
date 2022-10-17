# Contributing

This document describes how to contribute to the development of this software.

## Contribution Workflow

### Create an Issue

Before starting please search for and review the existing [issues](https://github.com/AFM-SPM/TopoStats/issues) (both
`open` and `closed`) and [pull requests](https://github.com/AFM-SPM/TopoStats/pulls) to see if anyone has reported the
bug or requested the feature already or work is in progress. If nothing exists then you should create a [new
issue](https://github.com/AFM-SPM/TopoStats/issues/new/choose) using one of the templates provided.


### Cloning the repository

If you wish to make changes yourself you will have to
[fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) the repository to your own account and then [clone
that](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) if you are not
a member of AFM-SPM Organisation. If you are a member then you can [clone the
repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) and make
contributions directly.

``` bash
# Member of AFM-SPM Organisation
git clone git@github.com:AFM-SPM/TopoStats.git
# Non-member of AFM-SPM cloning fork
git clone git@github.com:<YOUR_GITHUB_USERNAME>/TopoStats.git
```


### Creating a branch

Typically you will now create a [branch](https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging) to
work on the issue you wish to address. It is not compulsory but we try to use a consistent nomenclature for branches
that shows who has worked on the branch, the issue it pertains to and a short description of the work. To which end you
will see branches with the form `<GITHUB_USERNAME>/<GITHUB_ISSUE>-<DESCRIPTION>`. Some examples are shown below...


| Branch                                | User                                                | Issue                                                  | Description                                                                              |
|:--------------------------------------|:----------------------------------------------------|:-------------------------------------------------------|:-----------------------------------------------------------------------------------------|
| `ns-rse/259-contributing`             | [`ns-rse`](https://github.com/ns-rse)               | [259](https://github.com/AFM-SPM/TopoStats/issues/259) | `contributing` short for the issue subject _Add contributing section to documentation_.  |
| `SylviaWhittle/204-nanometre-scaling` | [`SylviaWhittle`](https://github.com/SylviaWhittle) | [204](https://github.com/AFM-SPM/TopoStats/issues/259) | `nanometre-scaling` short for the issue subject _Colour scale in nanometers not pixels_. |

How you create a branch depends on how you use Git, some use the integration provided by their IDE, others dedicated
clients such as [GitKraken](https://www.gitkraken.com/) and some may use the command line interface. These instructions
use the later but you are of course free to use your chosen method of managing Git and GitHub.

In this example we branch from `dev` and create a new branch called `ns-rse/000-fix-an-issue`.

``` bash
# Ensure you are up-to-date on the dev branch
git checkout dev
git pull
# Create and checkout a branch in one step
git checkout -b ns-rse/000-fix-an-issue
# Create and checkout a branch in two steps
git branch dev ns-rse/000-fix-an-issue
git checkout ns-rse/000-fix-an-issue
```

You can now start working on your issue and making regular commits, but please bear in mind the following section on
Coding Standards.


## Coding Standards

To make the codebase easier to maintain we ask that you follow the guidelines below on coding style, linting, typing,
documentation and testing.

### Coding Style/Linting

Using a consistent coding style has many benefits (see [Linting : What is all the fluff
about?](https://rse.shef.ac.uk/blog/2022-04-19-linting/)). For this project we aim to adhere to [PEP8 - the style Guide
for Python Code](https://pep8.org/) and do so using the formatting linters [black](https://github.com/psf/black) and
[flake8](https://github.com/PyCQA/flake8).  Many popular IDEs such as VSCode, PyCharm, Spyder and Emacs all have support
for integrating these linters into your workflow such that when you save a file the linting/formatting is automatically
applied.

We also like to ensure the code passes [pylint](https://github.com/PyCQA/pylint) which helps identify code duplication
and reduces some of the [code smells](https://en.wikipedia.org/wiki/Code_smell) that we are all prone to
making. A `.pylintrc` is included in the repository. Currently this isn't strictly applied but it is planned for part of
the CI/CD pipeline and so we would be grateful if you could lint your code before making Pull Requests.

### Pre-commit

[pre-commit](https://pre-commit.com) is a powerful and useful tool that runs hooks on your code prior to making
commits. For a more detailed exposition see [pre-commit : Protecting your future
self](https://rse.shef.ac.uk/blog/pre-commit/).

The repository includes `pre-commit` as a development dependency as well as a `.pre-commit-config.yaml`. To use these
locally install `pre-commit` in your virtual environment and then install the configuration and all the configured hooks
(**NB** this will download specific virtual environments that `pre-commit` uses when running hooks so the first time
this is run may take a little while).


``` bash
pip install .[dev]
pre-commit install --install-hooks
```

Currently there are hooks to remove trailing whitespace, check YAML configuration files and a few other common checks as
well as hooks for `black` and `flake8`. If these fail then you will not be able to make a commit until they are
fixed. The `black` hook will automatically format failed files so you can simply `git add` those and try committing
straight away. `flake8` does not correct files automatically so the errors will need manually correcting.

### Typing

Whilst Python is a dynamically typed language (that is the type of an object is determined dynamically) the use of Type
Hints is strongly encouraged as it makes reading and understanding the code considerably easier for contributors. For
more on Type Hints see [PEP483](https://peps.python.org/pep-0483/) and [PEP484](https://peps.python.org/pep-0484/)

### Documentation

All classes, methods and functions should have [Numpy Docstrings](https://numpydoc.readthedocs.io/en/latest/format.html)
defining their functionality, parameters and return values and pylint will note and report the absence of docstrings
by way of the `missing-function-docstring` condition.

Further, when new methods are incorporated into the package that introduce changes to the configuration they should be
documented under [Parameter Configuration](configuration)


### Testing

New features should have unit-tests written and included under the `tests/` directory to ensure the functions work as
expected. The [pytest](https://docs.pytest.org/en/latest/) framework is used for running tests along with a number of
plugins ([pytest-regtest]() for regression testing; [pytest-mpl]())


## Configuration

As described in [Parameter Configuration](configuration) options are primarily passed to TopoStats via a
[YAML](https://yaml.org) configuration file. When introducing new features that require configuration options you will
have to ensure that both the default configuration file (`topostats/default.yaml`) and the example configuration
(`config/example.yaml`) are updated to include your options.

Further the `topostats.validation.validate.config()` function which checks a valid configuration file with all necessary
fields has been passed when invoking TopoStats will also need updating to include new options in the Schema against
which validation of configuration files is made configuration.


### IDE Configuration

Linters such as `black`, `flake8` and `pylint` can be configured to work with your IDE so that say Black and/or
formatting is applied on saving a file or the code is analysed with `pylint` on saving and errors reported. Setting up
and configuring IDEs to work in this manner is beyond the scope of this document but some links to articles on how to do
so are provided.

* [Linting Python in Visual Studio Code](https://code.visualstudio.com/docs/python/linting)
* [Code Analysis — Spyder](http://docs.spyder-ide.org/current/panes/pylint.html) for `pylint` for Black see [How to use
  code formatter Black with Spyder](https://stackoverflow.com/a/66458706).
* [Code Quality Assistance Tips and Tricks, or How to Make Your Code Look Pretty? |
  PyCharm](https://www.jetbrains.com/help/pycharm/tutorial-code-quality-assistance-tips-and-tricks.html#525ee883)
* [Reformat and rearrange code | PyCharm](https://www.jetbrains.com/help/pycharm/reformat-and-rearrange-code.html)
