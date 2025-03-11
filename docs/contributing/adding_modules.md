# Adding Modules

## Basic Functionality

This should reside in `topostats/` or optionally `topostats/<sub_dir>`

Unit tests should be written for each function, classes and method and integration tests to ensure the overall
functionality is as expected now and in the future should refactoring be undertaken.

## Adding configuration

Configuration options live in `topostats/default_config.yaml` you should create a nested object for the options
corresponding to your module. The configuration nomenclature should match _exactly_ the options of your modules
class/function as this allows the [`**kwargs`][python_kwargs] to be used to pass the options from the loaded dictionary
to the function without having to explicitly map them to the class/function arguments. This might seem fickle or
excessive but it saves you and others work in the long run.

## Modularity

TopoStats is a command line program (for now at least!) and the main command `topsotats` has sub-commands which allow
individual steps of the processing to be undertaken, making it faster to test out the impact of different options. For
example once loaded and saved as `.topostats` files the `filter` stage could be re-run to improve flattening. Once
flattening is complete changes in configuration could be made to detect different grains or features using the already
flattened images.

Once you have a module that works it should ideally be included in this pipeline. Here we described how to include it.

### Processing

Each module needs a processing function. This is defined the `topostats/processing.py` module and should be called from
the `process_all()` function to include it in the end to end pipeline.

#### `process_all`

All of these modules are run in order to process individual images in parallel and this is achieved via the
`process_all()` function which has a parameter for a single image and the configuration dictionary and calls each module
in turn.

This is run in parallel using the [`functools.partial()`][python_partial] and [`Pool`][python_pool]. The former takes
the `process_all()` function as an argument along with all other configuration options. This is combined with a list of
images to be processed and run through the `Pool` class (see examples below).

To add your functionality to this pipeline you should add the calls to your class and its method to the `process_all()`
function.

### Modular Analyses

It is however useful to be able to call the function in isolation, this means we can add in sub-commands to
`topostats` so that we can experiment with different configurations without having to run the whole pipeline for all
images in one big batch job.

As an example we will look at the `Filter` class and how it is added as a sub-command.

The [`topostats.processing.run_filters()`][topostats_processing_run_filters] command is the processing command that
takes in the various options, instantiates an instance of the `Filters()` class and runs the various methods required to
process the image. There are other classes such as `Grains()`, `Grainstats()`, `Tracing()` which form part of the
pipeline.

#### Adding Sub-Command Arguments

You need to add a sub-parser for your module to run it in isolation. To do this entries need to be made to
`topostats/entry_point.py` which sets up [`argparse`][python_argparse]. At the top level the `create_parser()` function
is defined where the `parser` has the main options shown by `topostats --help` are defined. Each sub-command has its own
parser, the most commonly used is `process_parser` which defines the arguments to and the help shown by `topostats
process --help`. A [`filter_parser` subparser][topostats_entry_point_filter] is added which defines the arguments and
help shown by `topostats filter` and this has various arguments added to it with `.add_argument()`. Note it is important
to ensure that the `dest` for each added argument matches the name used in the `default_config.yaml` as these values are
used to update the loaded dictionary. If the names don't match the values do not get updated (and/or an error will
occur).

```python
# Filter
filter_parser = subparsers.add_parser(
    "filter",
    description="Load and filter images, saving as .topostats files for subsequent processing.",
    help="Load and filter images, saving as .topostats files for subsequent processing.",
)
filter_parser.add_argument(
    "--row-alignment-quantile",
    dest="row_alignment_quantile",
    type=float,
    required=False,
    help="Lower values may improve flattening of larger features.",
)
filter_parser.add_argument(
    "--threshold-method",
    dest="threshold_method",
    type=str,
    required=False,
    help="Method for thresholding Filtering. Options are otsu, std_dev, absolute.",
)
filter_parser.add_argument(
    "--otsu-threshold-multiplier",
    dest="otsu_threshold_multiplier",
    type=float,
    required=False,
    help="Factor for scaling the Otsu threshold during Filtering.",
)
filter_parser.add_argument(
    "--threshold-std-dev-below",
    dest="threshold_std_dev_below",
    type=float,
    required=False,
    help="Threshold for data below the image background for std dev method during Filtering.",
)
...
filter_parser.add_argument(
    "--scars-max-scar-length",
    dest="scars_max_scar_length",
    type=int,
    required=False,
    help="Maximum length of scars in pixels",
)
# Run the relevant function with the arguments
filter_parser.set_defaults(func=run_modules.filters)
```

1. Create a subparser with `<module_name>_parser = subparsers.add_parser(...)`.
2. Add arguments to subparser with `<module_name>_parser.add_argument(...)`, include...

- Single letter flag (optional).
- Longer flag (required).
- `dest` the destination to which the variable will be saved. This should match the corresponding value in the
  `default_config.yaml` which makes updating the configuration options straight-forward and will occur automatically
  (the code is in place you don't need to add anything, just ensure the names match).
- `type` should ideally be included as this helps define the type of the stored variable, particularly useful if for
  example paths are provided which should be stored as [`Path` pathlib][python_pathlib] objects.
- `default` a sensible default value, typically `None` though so that the value in `default_config.yaml` is used.
- `help` a useful description of what the configuration option changes along with possible options.

<!-- markdownlint-disable MD029 -->

3. Set the default function that will be called with `<module_name>_parser.set_defaults(func=run_modules.<function>)`
which will call the function you have defined in `topostats/run_modules.py` which runs your module.
<!-- markdownlint-enable MD029 -->

**NB** In the above substitute `<module_name>` for the meaningful name you have created for your module.

#### Running in Parallel

One of the main advantages of TopoStats is the ability to batch process images. As modern computers typically have
multiple cores it means the processing can be done in parallel making the processing faster. To achieve this we use
wrapper functions in `topostats/run_modules.py` that leverage the [`Pool`][python_pool] multiprocessing class.

Your function should first load the `.topostats` files you wish to process that are found from the user specified path
(default being the current directory `./`).

You then need to define a `partial()` function with the `topostats/processing.py<your_module_processing_function>` as
the first argument and the remaining configuration options. These will typically be a subset/list from the
`default_config.yaml` where the configuration options use the _exact_ same names as the arguments of the function you
defined your `topostats/processing.py`. As mentioned above keeping configuration names consistent between configuration
files and functions means [`**kwargs`][python_kwargs] can be used when passing options to functions.

Continuing with our example let's look at the [`topostats.processing.run_filters()`][topostats_entry_point_filters]
function.

```python
def run_filters(
    unprocessed_image: npt.NDArray,
    pixel_to_nm_scaling: float,
    filename: str,
    filter_out_path: Path,
    core_out_path: Path,
    filter_config: dict,
    plotting_config: dict,
) -> npt.NDArray | None:
    """
    Filter and flatten an image. Optionally plots the results, returning the flattened image.

    Parameters
    ----------
    unprocessed_image : npt.NDArray
        Image to be flattened.
    pixel_to_nm_scaling : float
        Scaling factor for converting pixel length scales to nanometres.
        ie the number of pixels per nanometre.
    filename : str
        File name for the image.
    filter_out_path : Path
        Output directory for step-by-step flattening plots.
    core_out_path : Path
        General output directory for outputs such as the flattened image.
    filter_config : dict
        Dictionary of configuration for the Filters class to use when initialised.
    plotting_config : dict
        Dictionary of configuration for plotting output images.

    Returns
    -------
    npt.NDArray | None
        Either a numpy array of the flattened image, or None if an error occurs or
        flattening is disabled in the configuration.
    """
    if filter_config["run"]:
        filter_config.pop("run")
        LOGGER.debug(f"[{filename}] Image dimensions: {unprocessed_image.shape}")
        LOGGER.info(f"[{filename}] : *** Filtering ***")
        filters = Filters(
            image=unprocessed_image,
            filename=filename,
            pixel_to_nm_scaling=pixel_to_nm_scaling,
            **filter_config,
        )
        filters.filter_image()
        # Optionally plot filter stage
        if plotting_config["run"]:
            plotting_config.pop("run")
            LOGGER.info(f"[{filename}] : Plotting Filtering Images")
            if plotting_config["image_set"] == "all":
                filter_out_path.mkdir(parents=True, exist_ok=True)
                LOGGER.debug(
                    f"[{filename}] : Target filter directory created : {filter_out_path}"
                )
            # Generate plots
            for plot_name, array in filters.images.items():
                if plot_name not in ["scan_raw"]:
                    if plot_name == "extracted_channel":
                        array = np.flipud(array.pixels)
                    plotting_config["plot_dict"][plot_name]["output_dir"] = (
                        core_out_path
                        if plotting_config["plot_dict"][plot_name]["core_set"]
                        else filter_out_path
                    )
                    try:
                        Images(
                            array, **plotting_config["plot_dict"][plot_name]
                        ).plot_and_save()
                        Images(
                            array, **plotting_config["plot_dict"][plot_name]
                        ).plot_histogram_and_save()
                    except AttributeError:
                        LOGGER.info(
                            f"[{filename}] Unable to generate plot : {plot_name}"
                        )
            plotting_config["run"] = True
        # Always want the 'z_threshed' plot (aka "Height Thresholded") but in the core_out_path
        plot_name = "z_threshed"
        plotting_config["plot_dict"][plot_name]["output_dir"] = core_out_path
        Images(
            filters.images["gaussian_filtered"],
            filename=filename,
            **plotting_config["plot_dict"][plot_name],
        ).plot_and_save()
        LOGGER.info(f"[{filename}] : Filters stage completed successfully.")
        return filters.images["gaussian_filtered"]
    # Otherwise, return None and warn that initial processing is disabled.
    LOGGER.error(
        "You have not included running the initial filter stage. This is required for all subsequent "
        "stages of processing. Please check your configuration file."
    )
    return None
```

This instantiates (creates) the object `filters` of the class `Filters` with the supplied options and then runs the
`filter_image()` method to perform the filtering. The rest of the code determines what images to plot based on the
configuration. At the end the `gaussian_filtered` image is returned.

This function only processes a single image. As mentioned we use the [`functools.partial()`][python_partial] function to
define a function with the command we want to run, in this case `topostats.processing.run_filters()` (imported as just
`filters()`) as the first argument and then all of the parameters we would normally pass in to `filters()` as the
remainder arguments. This is then used with the [`Pool`][python_pool] class and given a list of the images that are to
be processed and the `partial`ly defined function is run with each image.

In our example the [`run_modules.filters()`][topostats_run_modules_filters] this looks like the following.

```python
def filters(args: argparse.Namespace | None = None) -> None:
    """
    Load files from disk and run filtering.

    Parameters
    ----------
    args : None
        Arguments.
    """
    config, img_files = _parse_configuration(args)
    # If loading existing .topostats files the images need filtering again so we need to extract the raw image
    if config["file_ext"] == ".topostats":
        config["loading"]["extract"] = "raw"
    all_scan_data = LoadScans(img_files, **config["loading"])
    all_scan_data.get_data()

    processing_function = partial(
        process_filters,
        base_dir=config["base_dir"],
        filter_config=config["filter"],
        plotting_config=config["plotting"],
        output_dir=config["output_dir"],
    )

    with Pool(processes=config["cores"]) as pool:
        results = defaultdict()
        with tqdm(
            total=len(img_files),
            desc=f"Processing images from {config['base_dir']}, results are under {config['output_dir']}",
        ) as pbar:
            for img, result in pool.imap_unordered(
                processing_function,
                all_scan_data.img_dict.values(),
            ):
                results[str(img)] = result
                pbar.update()

                # Display completion message for the image
                LOGGER.info(f"[{img}] Filtering completed.")

    # Write config to file
    config["plotting"].pop("plot_dict")
    write_yaml(config, output_dir=config["output_dir"])
    LOGGER.debug(f"Images processed : {len(results)}")
    # Update config with plotting defaults for printing
    completion_message(
        config, img_files, summary_config=None, images_processed=sum(results.values())
    )
```

The files that are to be processed are first loaded to give the list of images that need processing and the `partial()`
function is defined as `processing_function` with all of the arguments before running in parallel using `Pool`. The
[`tqdm`][tqdm] package is leverage to give a progress bar and after completion the configuration file is written to file
before a completion message is run.

##### Results

Before we start the parallel processing we create a dictionary `results = defaultdict()` which will have the results of
processing added to. Some steps in processing return more than one object, for example `grainstats` returns statistics
for the image along with height profiles. In such cases we need to instantiate a dictionary to hold each set of results
across images and included a holder in the `for ... in pool.imap_unordered(...):` to store the results before adding the
results to the dictionary. For `run_modules.grainstats()` this looks like the below code. We create two dictionaries
`results` and `height_profile_all` and in our call for `for` we have `img` (a string representing the image name),
`result` (the returned Pandas DataFrame of grain statistics for the given image) and `height_profiles` (the height
profile dictionary for the grains in that image).

```python
with Pool(processes=config["cores"]) as pool:
    results = defaultdict()
    height_profile_all = defaultdict()
    with tqdm(
        total=len(img_files),
        desc=f"Processing images from {config['base_dir']}, results are under {config['output_dir']}",
    ) as pbar:
        for img, result, height_profiles in pool.imap_unordered(
            processing_function,
            all_scan_data.img_dict.values(),
        ):
            results[str(img)] = results
            height_profile_all[str(img)] = height_profiles
            pbar.update()

            # Display completion message for the image
            LOGGER.info(
                f"[{img}] Grainstats completed (NB - Filtering was *not* re-run)."
            )
```

Note that your `processing.process_<stage>` function which is used in the call to `processing_function` should return a
tuple, the first item of which is the `topostats_object["filename"]` (which will be stored in `img` and used as
dictionary keys), the remaining items are the results that you expect to be returned.

## Conclusion

Adding functionality is useful but it has to integrate into the workflow and ideally be accessible as a stand alone step
in the process. Hopefully the above helps demystify the steps required to achieve this.

[python_argparse]: https://docs.python.org/3/library/argparse.html
[python_kwargs]: https://realpython.com/python-kwargs-and-args/
[python_partial]: https://docs.python.org/3/library/functools.html#functools.partial
[python_pathlib]: https://docs.python.org/3/library/pathlib.html
[python_pool]: https://docs.python.org/3/library/multiprocessing.html
[topostats_entry_point_filter]: https://github.com/AFM-SPM/TopoStats/blob/c769ca3cfd32bf37ab3a158cf032b80fa5bcf51a/topostats/entry_point.py#L597
[topostats_processing_run_filters]: https://github.com/AFM-SPM/TopoStats/blob/c769ca3cfd32bf37ab3a158cf032b80fa5bcf51a/topostats/processing.py#L48
[topostats_run_modules_filters]: https://github.com/AFM-SPM/TopoStats/blob/c769ca3cfd32bf37ab3a158cf032b80fa5bcf51a/topostats/run_modules.py#L402
[tqdm]: https://tqdm.github.io/
