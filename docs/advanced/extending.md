# Extending

<hr>

**NB** The way in which TopoStats can be extended is very much a work in-progress. This page is an attempt to document
and keep up-to-date the process of doing so but is subject to change and may be outdated at any given point in time.

<hr>

TopoStats design is modular with the intention that users can write extensions to carry out different workflows
contingent on the types of samples being imaged. Here we describe how to add your own module.

## Core Features

All AFM images need to be filtered, flattened and optionally have scars removed. This is core functionality carried out
by the `topostats.filters` module and will be required by all packages that are developed to extend TopoStats.

## Entry Points

Primarily TopoStats was designed with batch processing of multiple images in mind and as such it uses the command line
for invocation and processing of multiple images. To this end you should include entry points in your module. For
consistency it is recommended that you create a `<pkg_path>/entry_point.py` which has an argument parser and sub-parsers
for each of the steps you wish to create. These are added under the `project.scripts` configuration section of _your_
projects `pyproject.toml`.

Your parser should include the common options to TopoStats (see the arguments added to `parser` in
`topostats/entry_point.py`) and the `filter` sub-parser (see the arguments added to `parser.filter` in `topostats/entry_point.py`).

### Custom Configuration

TopoStats uses [YAML][yaml] configuration files to specify the options. In the module you develop you should include a
`default_config.yaml` and copy the base configuration parameters and `filters:` section from
`topostats/default_config.yaml`.

You should, for user convenience add a sub-parser to `create-config` using the example in
`topostats/entry_point.py`. This will allow users to run `<your_package> create-config --config <your_package>` and it
will, by default, write a copy of `default_config.yaml` to the `config.yaml` file of the current directory (users can
modify the output filename with the `<your_package> create-config --filename my_cutsom_config.yaml` should they wish
to). You shouldn't have to do anything else, TopoStats will look for the `default_config.yaml` in your package and write
that to disk.

## Tests

Functions and methods should follow the [single responsibility principle][srp] and perform very small focused
calculations. This means that [unit tests][unit_tests] can be written to test the function does what it purports
to. Ideally these should be [parameterised][param-tests] to cover all the possible combination of options. For example
if a function takes a boolean argument (i.e. `True` or `False`) and this influences whether a particular element of
calculation is undertaken and in turn the output differs, then tests should cover both scenarios.

### Regression Tests

In a number of places we perform [regression tests][regress-test] (also known as integration tests) which tests how
several components perform. Typically this is for whole classes/modules which perform a bunch of different steps in the
processing and the results are compared to what is expected. To do this we have moved to using the [Syrupy][syrupy]
package which writes "snapshots" of tests to the `__snapshot__/test_<test-file-name>.ambr` against which a re-run of
tests are compared. One of the challenges here is the precision of values written to these files and the fact that
precision very occasionally varies between operating systems. To deal with this some of the values that do vary are
separated from the rest of the values being tested and the [`matcher`][matcher] argument is used. For an example of how
this is used see this [thread][matcher_example] of the `tests/test_processing.py::test_grainstats()` test.

#### Paths in output

The `TopoStats.img_path` holds the path to the file that is being tested. This will vary between the machine the tests
are being run on, whether that is a personal laptop or one of the Continuous Integration runners on GitHub. As a
consequence any regression test which includes this will fail if it includes the `img_path` or as it is called in
dataframes/`.csv` files the `basename`.

The solution, when developing tests that might include these fields, is to set `TopoStats.img_path = None` for
`TopoStats` objects and to drop them from data frames (e.g. `df.drop(["basename"], axis=1, inplace=True)`).

#### Paths - Windows

Another consideration in writing regression tests is that TopoStats makes heavy usage of the [pathlib][pathlib]
module. On POSIX systems such as GNU/Linux and OSX this results in path objects of type `PosixPath`, but because
Microsoft Windows is not POSIX compliant tests that compare such values will fail. The solution to this implemented in
`tests/conftest.py` is to mask `pathlib.PosixPath` to be `pathlib.WindowsPath`

```python
import pathlib
import platform

if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath
```

[matcher]: https://syrupy-project.github.io/syrupy/#matcher
[matcher_example]: https://github.com/syrupy-project/syrupy/issues/913
[param-tests]: https://blog.nshephard.dev/posts/pytest-param/
[pathlib]: https://docs.python.org/3/library/pathlib.html
[regress-test]: https://en.wikipedia.org/wiki/Regression_testing
[srp]: https://en.wikipedia.org/wiki/Single-responsibility_principle
[syrupy]: https://syrupy-project.github.io/syrupy/
[unit_tests]: https://en.wikipedia.org/wiki/Unit_testing
[yaml]: https://yaml.org
