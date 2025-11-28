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

[yaml]: https://yaml.org
