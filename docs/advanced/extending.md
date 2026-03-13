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
`topostats/default_config.yaml` as a starting point as all images require flattening. You may wish to remove the
configuration for thresholding though as this is not always required.

You should, for user convenience add a sub-parser to `create-config` using the example in
`topostats/entry_point.py`. This will allow users to run `<your_package> create-config --config <your_package>` and it
will, by default, write a copy of `default_config.yaml` to the `config.yaml` file of the current directory (users can
modify the output filename with the `<your_package> create-config --filename my_cutsom_config.yaml` should they wish
to). You shouldn't have to do anything else, TopoStats will look for the `default_config.yaml` in your package and write
that to disk.

#### Writing Configuration Files

TopoStats includes the `io.write_yaml()` function to write the YAML configuration alongside all output in the output
directory so that users know what parameters were used in processing their images. Packages using and extending
TopoStats can leverage this function to write _their_ configuration files too. In order to customise the header of this
file users should define `CONFIG_DOCUMENTATION_REFERENCE` in the `__init__.py`, import it where required and define a
custom `HEADER_MESSAGE`.

**Example `__init__.py`**

```python
from importlib.metadata import version

import snoop
from packaging.version import Version

# Disable TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

__version__ = version("afmslicer")
__release__ = ".".join(__version__.split(".")[:-2])

AFMSLICER_VERSION = Version(__version__)
if AFMSLICER_VERSION.is_prerelease and AFMSLICER_VERSION.is_devrelease:
    AFMSLICER_BASE_VERSION = str(AFMSLICER_VERSION.base_version)
    AFMSLICER_COMMIT = str(AFMSLICER_VERSION).split("+g")[1]
else:
    AFMSLICER_BASE_VERSION = str(AFMSLICER_VERSION)
    AFMSLICER_COMMIT = ""
CONFIG_DOCUMENTATION_REFERENCE = """# For more information on configuration and how to use it:
# https://afm-spm.github.io/AFMslicer/main/configuration.html\n"""
CONFIG_DOCUMENTATION_REFERENCE += f"# AFMSlicer version : {AFMSLICER_BASE_VERSION}\n"
CONFIG_DOCUMENTATION_REFERENCE += f"# Commit: {AFMSLICER_COMMIT}\n"
```

**Example custom `HEADER_MESSAGE`**

```python
from topostats.io import get_date_time, write_yaml
from afmslicer import CONFIG_DOCUMENTATION_REFERENCE

HEADER_MESSAGE == f"# Configuration from AFMSlicer run complete : {get_date_time()}\n{CONFIG_DOCUMENTATION_REFERENCE}"


def some_function() -> None:
    config_data = get_data(
        package=afmslicer.__package__, resource="default_config.yaml"
    )
    config = yaml.full_load(config_data)
    write.yaml(config, output_dir=config["output_dir"], header_message=HEADER_MESSAGE)
```

This example results in the following header when running `some_function()`

```yaml
# Configuration from AFMSlicer run complete : 2026-03-06 12:40:32
# For more information on configuration and how to use it:
# https://afm-spm.github.io/AFMslicer/main/configuration.html
# AFMSlicer version : 0.1
# Commit: cd24434e2.d20251114
# TopoStats version: 2.4.1
# Commit: 53b1eb591.d20260306
```

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
this is used see this [thread][matcher_example] or the `tests/test_processing.py::test_grainstats()` test.

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
