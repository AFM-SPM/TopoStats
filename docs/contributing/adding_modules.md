# How to add modules

## Basic Functionality

This should reside in `topostats/` or optionally `topostats/<sub_dir>`

Unit tests should be written for each function, classes and method and integration tests to ensure the overall
functionality is as expected now and in the future should refactoring be undertaken

## Adding configuration

Configuration options live in `topostats/default_config.yaml` you should create a nested object for the options
corresponding to your module. The configuration nomenclature should match _exactly_ the options of your modules
class/function as this allows the use of [`**kwargs`][python_kwargs] to be used to pass the options from the loaded
dictionary to the function without having to explicitly map them to the class/function arguments. This might seem fickle
or excessive but it saves you and others work in the long run.

## Processing

Each module needs a processing function. This is defined the `topostats/processing.py` module and should be called from
the `process_all()` function to include it in the end to end pipeline.

## Modularity

TopoStats is a command line program (for now at least!) and the main command `topsotats` has sub-commands which allow
individual steps of the processing to be undertaken, making it faster to test out the impact of different options. For
example once loaded and saved as `.topostats` files the `filter` stage could be re-run to improve flattening. Once
flattening is complete changes in configuration could be made to detect different grains or features using the already
flattened images.

Once you have a module that works it should ideally be included in this pipeline. Here we described how to include it.

### Adding Arguments

You need to add a sub-parser for your module to run it in isolation. To do this entries need to be made to
`topostats/entry_point.py` which sets up [`argparse`][python_argparse]. At the top level the `create_parser()` function
is defined where the `parser` has the main options shown by `topostats --help` are defined. Each sub-command has its own
parser, the most commonly used is `process_parser` which defines the arguments to and the help shown by `topostats
process --help`. The [`create_config_parser`][create_config_parser] which defines the arguments and help shown by
`topostats create-config` is shown below as an example to help guide you through creating your own `subparser`.

```python
create_config_parser = subparsers.add_parser(
    "create-config",
    description="Create a configuration file using the defaults.",
    help="Create a configuration file using the defaults.",
)
create_config_parser.add_argument(
    "-f",
    "--filename",
    dest="filename",
    type=Path,
    required=False,
    default="config.yaml",
    help="Name of YAML file to save configuration to (default 'config.yaml').",
)
create_config_parser.add_argument(
    "-o",
    "--output-dir",
    dest="output_dir",
    type=Path,
    required=False,
    default="./",
    help="Path to where the YAML file should be saved (default './' the current directory).",
)
create_config_parser.add_argument(
    "-c",
    "--config",
    dest="config",
    type=str,
    default=None,
    help="Configuration to use, currently only one is supported, the 'default'.",
)
create_config_parser.add_argument(
    "-s",
    "--simple",
    dest="simple",
    action="store_true",
    help="Create a simple configuration file with only the most common options.",
)
create_config_parser.set_defaults(func=write_config_with_comments)
```

1. Create a subparser with `<module_name>_parser = subparsers.add_parser(...)`.
2. Add arguments to subparser with `<module_name>_parser.add_argument(...)`, include...

- Single letter flag (optional).
- Longer flag (required).
- `dest` the destination to which the variable will be saved. This should match the corresponding value in the
  `default_config.yaml` which makes updating the configuration options straight-forward.
- `type` should ideally be included as this helps define the type of the stored variable, particularly useful if for
  example paths are provided which should be stored as [`Path` pathlib][python_pathlib] objects.
- `default` a sensible default value, typically `None` though so that the value in `default_config.yaml` is used.
- `help` a useful description of what the configuration option changes along with possible options.

<!-- markdownlint-disable MD029 -->

3. Set the default function that will be called with `<module_name_parser.set_defaults(func=run_modules.<function>)`
which will call the function you have defined in `topostats/run_modules.py` which runs your module.
<!-- markdownlint-enable MD029 -->

**NB** In the above substitute `<module_name>` for the meaningful name you have created for your module.

### Running in Parallel

One of the main advantages of TopoStats is the ability to batch process images. As modern computers typically have
multiple cores it means the processing can be done in parallel making the processing faster. To achieve this we use
wrapper functions in `topostats/run_modules.py` that leverage the [`Pool`][python_pool] multiprocessing class.

Your function should first load the `.topostats` files you wish to process that are found from the user specified path
(default being the current directory `./`).

You then need to define a `partial()` function with the `topostats/processing.py<your_module_processing_function>` as
the first argument and the remaining configuration options. These will typically be a subset/list from the
`default_config.yaml` where the configuration options use the _exact_ same names as the arguments of the function you
defined your `topostats/processing.py`.

[create_config_parser]: https://github.com/AFM-SPM/TopoStats/blob/a480afe9d7e9c16501f5af5621ed86555db44708/topostats/entry_point.py#L1104
[python_argparse]: https://docs.python.org/3/library/argparse.html
[python_kwargs]: https://realpython.com/python-kwargs-and-args/
[python_pathlib]: https://docs.python.org/3/library/pathlib.html
[python_pool]: https://docs.python.org/3/library/multiprocessing.html
