import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #Walkthrough of DNA tracing in TopoStats
    Once grain masks have been obtained through TopoStats (as exemplified in the `grainstats.py` notebook), these can be used to obtain DNA traces - smooth contours that sit along molecular backbones. These traces enable extraction of additional features such as DNA topology, height profiles, contour length, number of crossings etc. enabling greater characterisation of DNA than can be obtained through masks alone.

    This notebook builds on the TopoStats foundations (`Filters`, `Grains` and `Grainstats`) described in the `grainstats.py` notebook. We recommend reviewing that notebook first for essential context before proceeding here.

    The full DNA tracing pipeline is described in detail on our [documentation website](https://github.com/AFM-SPM/TopoStats/tree/main/docs/advanced) where the relevant files are [`disordered_tracing.md`](https://github.com/AFM-SPM/TopoStats/blob/main/docs/advanced/disordered_tracing.md), [`ordered_tracing.md`](https://github.com/AFM-SPM/TopoStats/blob/main/docs/advanced/ordered_tracing.md), and [`splining.md`](https://github.com/AFM-SPM/TopoStats/blob/main/docs/advanced/splining.md).

    You can also see our tracing algorithm in action in [Holmes et al. (2025)](https://www.nature.com/articles/s41467-025-60559-x), where we classify and quantify complex topological DNA structures!
    """
    )
    return


@app.cell
def _():
    import pprint
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from topostats.filters import Filters
    from topostats.grains import Grains
    from topostats.grainstats import GrainStats
    from topostats.io import LoadScans, find_files, read_yaml
    from topostats.measure.curvature import calculate_curvature_stats_image
    from topostats.plottingfuncs import Images
    from topostats.tracing.disordered_tracing import trace_image_disordered
    from topostats.tracing.nodestats import nodestats_image
    from topostats.tracing.ordered_tracing import ordered_tracing_image
    from topostats.tracing.splining import splining_image

    pp = pprint.PrettyPrinter(indent=2, width=100, compact=True)
    return (
        Filters,
        GrainStats,
        Grains,
        Images,
        LoadScans,
        Path,
        calculate_curvature_stats_image,
        find_files,
        mo,
        nodestats_image,
        np,
        ordered_tracing_image,
        plt,
        pp,
        pprint,
        read_yaml,
        splining_image,
        trace_image_disordered,
    )


@app.cell
def _():
    """
    Trace statistics notebook.

    This Marimo notebook demonstrates the TopoStats tracing workflow.
    """
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ##Pre-requisites: filtered image and grain masks
    The following code block is a summarised version of the code within `grainstats.py` to obtain a filtered image and grain mask ready for the DNA tracing pipeline. Here we use an image of DNA plasmids, rather than minicircles, to showcase our method's ability to trace complex DNA structures.

    Here we have additional configs to parameterise the DNA tracing pipeline, including: `disordered_tracing`, `nodestats`, `ordered_tracing` and `splining`. Furthermore, the `curvature` config allows for parameterisation to quantify curvature of DNA traces.
    """
    )
    return


@app.cell
def _(
    Filters,
    GrainStats,
    Grains,
    LoadScans,
    Path,
    find_files,
    np,
    plt,
    read_yaml,
):
    # Set the base directory to be current working directory of the Notebook
    BASE_DIR = Path().cwd()
    # Alternatively if you know where your files are, comment the above line and uncomment the below adjust it for your use.
    # BASE_DIR = Path("/path/to/where/my/files/are")
    # Adjust the file extension appropriately.
    FILE_EXT = ".spm"
    # Search for *.spm files one directory level up from the current notebooks
    image_files = find_files(base_dir=BASE_DIR.parent / "tests", file_ext=FILE_EXT)

    def show_image(arr, cmap="afmhot", size=(8, 8), colorbar=True, title=None):
        arr = np.asarray(arr)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {arr.shape}")

        fig, ax = plt.subplots(figsize=size)
        im = ax.imshow(arr, cmap=cmap)
        if title:
            ax.set_title(title)
        if colorbar:
            fig.colorbar(im, ax=ax, shrink=0.7)
        fig.tight_layout()
        return fig

    config = read_yaml(BASE_DIR.parent / "topostats" / "default_config.yaml")
    loading_config = config["loading"]
    filter_config = config["filter"]
    filter_config["remove_scars"]["run"] = True
    grain_config = config["grains"]
    grain_config.pop("run")
    grainstats_config = config["grainstats"]
    grainstats_config.pop("run")
    disordered_tracing_config = config["disordered_tracing"]
    nodestats_config = config["nodestats"]
    nodestats_config.pop("run")
    ordered_tracing_config = config["ordered_tracing"]
    ordered_tracing_config.pop("run")
    splining_config = config["splining"]
    splining_config.pop("run")
    curvature_config = config["curvature"]
    curvature_config.pop("run")
    plotting_config = config["plotting"]
    plotting_config.pop("run")
    plotting_config["image_set"] = "all"

    all_scan_data = LoadScans(image_files, **config["loading"])
    all_scan_data.get_data()
    img = all_scan_data.img_dict["plasmids"]["image_original"]

    filter_config["remove_scars"]["run"] = True

    filtered_image = Filters(
        image=all_scan_data.img_dict["plasmids"]["image_original"],
        filename=all_scan_data.img_dict["plasmids"]["img_path"],
        pixel_to_nm_scaling=all_scan_data.img_dict["plasmids"]["pixel_to_nm_scaling"],
        row_alignment_quantile=filter_config["row_alignment_quantile"],
        threshold_method=filter_config["threshold_method"],
        otsu_threshold_multiplier=filter_config["otsu_threshold_multiplier"],
        threshold_std_dev=filter_config["threshold_std_dev"],
        threshold_absolute=filter_config["threshold_absolute"],
        gaussian_size=filter_config["gaussian_size"],
        gaussian_mode=filter_config["gaussian_mode"],
        remove_scars=filter_config["remove_scars"],
    )

    filtered_image.filter_image()

    grain_config["area_thresholds"]["above"] = [300, 30000]

    grains = Grains(
        image=filtered_image.images["gaussian_filtered"],
        filename=filtered_image.filename,
        pixel_to_nm_scaling=filtered_image.pixel_to_nm_scaling,
        **grain_config,
    )
    grains.find_grains()

    mask = grains.mask_images["above"]["merged_classes"][:, :, 1].astype(bool)
    image = grains.image

    labelled_regions = Grains.label_regions(
        grains.mask_images["above"]["merged_classes"][:, :, 1].astype(bool).astype(int)
    )

    cfg = grainstats_config.copy()
    cfg.pop("class_names", None)

    grainstats = GrainStats(
        grain_crops=grains.image_grain_crops.above.crops,
        direction="above",
        base_output_dir="grains",
        **cfg,
    )

    _temp = grainstats.calculate_stats()
    grain_stats = {"statistics": _temp[0], "plots": _temp[1]}
    return (
        curvature_config,
        disordered_tracing_config,
        filtered_image,
        grains,
        image,
        nodestats_config,
        ordered_tracing_config,
        show_image,
        splining_config,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    We can view the resulting processed image and grains mask using the code below.
    """
    )
    return


@app.cell
def _(filtered_image, grains, show_image):
    filtered_image_plot = show_image(filtered_image.images["gaussian_filtered"], title="Processed image")

    grains_mask_plot = show_image(
        grains.mask_images["above"]["merged_classes"][:, :, 1].astype(bool),
        cmap="gray",
        title="Grains mask",
        colorbar=False,
    )

    # Return both so they render one after the other
    filtered_image_plot, grains_mask_plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ##Disordered tracing

    The `disordered_tracing.py` module handles all the functions associated with obtaining single-pixel wide, line representations of masked objects.

    The quality and likeness of the resultant pruned skeleton thus depends on the quality of the mask, the effectiveness of smoothing parameters, the method of skeletonisation, and the quality of automating the pruning of incorrect skeletal branches.

    ![disordered_tracing](https://raw.githubusercontent.com/AFM-SPM/TopoStats/refs/heads/main/docs/_static/images/disordered_tracing/overview.png)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We can run the function `trace_image_disordered` to run the `disordered_tracing` workflow. The function outputs a set of diagnostic images for each grain (stored in `disordered_traces_cropped_data`) as well as each image (stored in `disordered_tracing_images`).
    """
    )
    return


@app.cell
def _(disordered_tracing_config, grains, image, trace_image_disordered):
    disordered_traces_cropped_data, disordered_trace_grainstats, disordered_tracing_images, disordered_tracing_stats = (
        trace_image_disordered(
            full_image=image,
            grain_crops=grains.image_grain_crops.above.crops,
            class_index=disordered_tracing_config["class_index"],
            filename="plasmids",
            pixel_to_nm_scaling=grains.pixel_to_nm_scaling,
            min_skeleton_size=disordered_tracing_config["min_skeleton_size"],
            mask_smoothing_params=disordered_tracing_config["mask_smoothing_params"],
            skeletonisation_params=disordered_tracing_config["skeletonisation_params"],
            pruning_params=disordered_tracing_config["pruning_params"],
        )
    )
    return (disordered_traces_cropped_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We can inspect the contents of `disordered_traces_cropped_data` using the code below. "Available grain keys" tells us the number of grains that are described within our dictionary, in this case there is `grain_0` and `grain_1` - the two grains that we saw in our masked image.

    By selecting `grain_0` as an example, we can view the data that is stored for each grain. Here `skeleton` refers to a binary mask of the original grain skeleton, and `pruned_skeleton` is the binary mask following additional processing, known as pruning, to remove spurious branches.
    """
    )
    return


@app.cell
def _(disordered_traces_cropped_data, pp):
    print("Available grain keys:")
    pp.pprint(list(disordered_traces_cropped_data.keys()))

    # Pick the first grain key
    first_key = next(iter(disordered_traces_cropped_data))
    print("\nFirst key:", first_key)

    # Inspect the value
    value = disordered_traces_cropped_data[first_key]

    if isinstance(value, dict):
        print("\nContents of this grain:")
        pp.pprint({k: f"{type(v).__name__}, shape={getattr(v, 'shape', None)}" for k, v in value.items()})
    else:
        print("\nValue preview:")
        pp.pprint(value)
    return (first_key,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We can view the pruned skeletons for both grains using the code block below, notice that these are 1 pixel wide traces that sit along the molecular backbone of the DNA molecules from our AFM image.
    """
    )
    return


@app.cell
def _(disordered_traces_cropped_data, show_image):
    grain0_traceimages = disordered_traces_cropped_data["grain_0"]
    grain1_traceimages = disordered_traces_cropped_data["grain_1"]

    grain0_prunedskel = show_image(
        grain0_traceimages["skeleton"], cmap="gray", colorbar=False, title="Grain 0 - pruned skeleton"
    )
    grain1_prunedskel = show_image(
        grain1_traceimages["skeleton"], cmap="gray", colorbar=False, title="Grain 1 - pruned skeleton"
    )

    grain0_prunedskel, grain1_prunedskel
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ##Nodestats

    The `nodestats.py` module handles all the functions associated with identifying and analysing the crossing regions (nodes) and crossing branches in pruned skeletons.

    The quality of the resultant metrics and over/underlying branch classifications depend on the quality of the pruned skeleton, the effectiveness of automating the joining of skeleton junction points through the parameters.

    ![nodestats](https://github.com/AFM-SPM/TopoStats/raw/main/docs/_static/images/nodestats/overview.png))
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    We can run the `nodestats` workflow using the `nodestats_image` function as in the code block below.
    """
    )
    return


@app.cell
def _(
    disordered_traces_cropped_data,
    grains,
    image,
    nodestats_config,
    nodestats_image,
):
    node_stats = nodestats_image(
        image=image,
        disordered_tracing_direction_data=disordered_traces_cropped_data,
        filename="plasmids",
        pixel_to_nm_scaling=grains.pixel_to_nm_scaling,
        **nodestats_config,
    )

    node_info, grain_stats_df, image_dict, single_grain_images = node_stats

    # Now explore each:

    # 1. Node info (dict)
    print("Node info keys:", node_info.keys())

    # 2. Grain statistics (DataFrame)
    print("Grain stats DataFrame:")
    print(grain_stats_df.head())  # first rows

    # 3. Image dictionary (dict)
    print("Image dict keys:", image_dict.keys())

    # 4. Single grain images (dict)
    print("Single grain images keys:", single_grain_images.keys())
    return image_dict, node_stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Skeleton segments which represent a crossing may not join up perfectly at a single pixel (junction) and as a result of the skeletonisation procedure may be offset from one another and need to be combined to represent the crossing region or "node". Therefore, junctions closer than the `node_joining_length` (set in the config file) of each other define a crossing region, and the pixels which span between the junctions along the skeleton are also labelled as part of the crossing.

    This is exemplified below where the first image shows all identified junctions. In the leftmost grain we can see two crossings, but 3 junctions are identified.

    In the bottom image, 2 of the 3 junctions are combined to correctly show two crossings within the grain.
    """
    )
    return


@app.cell
def _(image_dict, show_image):
    original_nodes = show_image(image_dict["convolved_skeletons"], cmap="gray", colorbar=False, title="Original nodes")
    connected_nodes = show_image(image_dict["connected_nodes"], cmap="gray", colorbar=False, title="Connected nodes")

    original_nodes, connected_nodes
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ##Ordered tracing
    This module orders the disordered trace pixel-by-pixel (topostats method) or segment-by-segment (nodestats method), giving direction to the trace and creating a path to follow. It identifies how many intertwined molecules are within the grain (found by restarting the trace when using the nodestats method), and outputs additional statistics such as DNA topology.

    ![ordered tracing](https://github.com/AFM-SPM/TopoStats/raw/main/docs/_static/images/ordered_tracing/overview.png)

    We can run the full ordered tracing workflow using the `ordered_tracing_image` function as below.
    """
    )
    return


@app.cell
def _(
    disordered_traces_cropped_data,
    image,
    node_stats,
    ordered_tracing_config,
    ordered_tracing_image,
):
    # nodestats output as input for ordered_tracing_image
    nodestats_direction_data = {
        "stats": node_stats[0],
        "images": node_stats[3],
    }

    all_traces_data, grainstats_additions_df, molstats_df, ordered_trace_full_images = ordered_tracing_image(
        image=image,
        disordered_tracing_direction_data=disordered_traces_cropped_data,
        nodestats_direction_data=nodestats_direction_data,
        filename="plasmids",
        **ordered_tracing_config,
    )
    return (
        all_traces_data,
        grainstats_additions_df,
        molstats_df,
        ordered_trace_full_images,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    We can inspect all of the outputs of `ordered_tracing_image` using the code below, we see a combination of images to view the tracing outputs as well as data frames containing statistics such as topology and writhe that were extracted from ordered traces.
    """
    )
    return


@app.cell
def _(
    all_traces_data,
    grainstats_additions_df,
    molstats_df,
    ordered_trace_full_images,
):
    # 1. Dict of all traces
    print("All traces keys:", all_traces_data.keys())
    # Inspect the first item
    first_key_traces = next(iter(all_traces_data))
    print("Example trace entry:", all_traces_data[first_key_traces])

    # 2. Grain stats DataFrame
    print("\nGrain stats additions DataFrame:")
    print(grainstats_additions_df.head())  # first rows

    # 3. Mol stats DataFrame
    print("\nMolecule stats DataFrame:")
    print(molstats_df.head())

    # 4. Dict of full images
    print("\nOrdered trace full images keys:", ordered_trace_full_images.keys())
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Below we plot `ordered_trace_full_images['ordered_traces']` which shows the ordered trace coordinates, coloured using a sequential colour map with lighter colours indicating the start of the trace, and darker colours indicating the end.
    """
    )
    return


@app.cell
def _(ordered_trace_full_images, show_image):
    show_image(ordered_trace_full_images["ordered_traces"], cmap="Reds")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ##Splining

    The `splining.py` module handles all the functions associated with smoothing the ordered pixel-wise trace from the "ordered_tracing" step, in order to produce curves which more closely follow the samples structure.

    ![splining](https://github.com/AFM-SPM/TopoStats/raw/main/docs/_static/images/splining/overview.png)

    We can use the `splining_image` function as in the code block below to run the full splining workflow. We can also inspect the outputs produced, notice that we now have two data frames where one consists of statistics extracted from whole grains and the other contains statistics extracted at molecule level (e.g. statistics to describe each molecule if two molecules are interconnected to form a grain.)
    """
    )
    return


@app.cell
def _(
    all_traces_data,
    first_key,
    grains,
    image,
    pprint,
    splining_config,
    splining_image,
):
    splined_traces, spline_grainstats_df, spline_molstats_df = splining_image(
        image=image,
        ordered_tracing_direction_data=all_traces_data,
        filename="plasmids",
        pixel_to_nm_scaling=grains.pixel_to_nm_scaling,
        **splining_config,
    )

    # 1. Splined traces (dict)
    print("Splined Traces")
    print("Keys:", list(splined_traces.keys()))
    # show one example entry
    first_key_splining = next(iter(splined_traces))
    print(f"\nExample entry for key '{first_key_splining}':")
    pprint.pprint(splined_traces[first_key])

    # 2. Grain stats DataFrame
    print("Spline Grain Stats DataFrame")
    print("Shape:", spline_grainstats_df.shape)
    print(spline_grainstats_df.head())

    # 3. Molecule stats DataFrame
    print("Spline Molecule Stats DataFrame")
    print("Shape:", spline_molstats_df.shape)
    print(spline_molstats_df.head())
    return (splined_traces,)


@app.cell
def _(plt, splined_traces):
    plt.plot(
        splined_traces["grain_0"]["mol_0"]["spline_coords"][:, 0],
        splined_traces["grain_0"]["mol_0"]["spline_coords"][:, 1],
    )
    plt.title("Splined trace for grain 0")
    plt.xlabel("X coordinates")
    plt.ylabel("Y coordinates")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ##Curvature

    Once we have the smooth traces for each molecule, we can calculate the curvature at each point along the molecular backbone. We can then look at how curvature changes at different points along the molecule, determine whether certain features (such as protein binding or damage) affect DNA curvature, and calculate an average curvature value for the entire molecule.

    We make use of the `calculate_curvature_stats_image` function to calculate curvature for all grains identified within our image.
    """
    )
    return


@app.cell
def _(calculate_curvature_stats_image, grains, splined_traces):
    curvature_stats = calculate_curvature_stats_image(
        all_grain_smoothed_data=splined_traces, pixel_to_nm_scaling=grains.pixel_to_nm_scaling
    )
    return (curvature_stats,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We can print the resulting `curvature_stats` which provides a dictionary of curvature values for each grain, with one value per coordinate along the grain's trace.
    """
    )
    return


@app.cell
def _(curvature_stats):
    print(curvature_stats)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The following code block allows us to visualise curvature values at different points along the grain traces, with light reds indicating regions with lower curvature, and darker reds indicating high curvature regions.
    """
    )
    return


@app.cell
def _(
    Images,
    curvature_config,
    curvature_stats,
    grains,
    image,
    np,
    splined_traces,
):
    direction = "above"
    crops = grains.image_grain_crops.above.crops

    # Build the dict that plot_curvatures expects: { "grain_i": {"image","bbox","pad_width"} }
    cropped_images = {}
    for i, gc in crops.items():
        b = getattr(gc, "bbox", None)
        afm_img = getattr(gc, "image", None)
        if afm_img is None:
            afm_img = getattr(gc, "crop", None)
        if afm_img is None and b is not None:
            x0, y0, x1, y1 = b
            afm_img = image[x0:x1, y0:y1]
        if afm_img is None:
            afm_img = np.zeros((1, 1), dtype=image.dtype)
        cropped_images[f"grain_{i}"] = {
            "image": afm_img,
            "bbox": b,
            "pad_width": getattr(gc, "pad_width", getattr(gc, "pad", 0)),
        }

    img1 = Images(
        np.zeros((2, 2)), output_dir=".", filename=f"plasmids_{direction}_curvature", save=False, savefig_dpi=300
    )

    fig1, ax1 = img1.plot_curvatures(
        image=image,
        cropped_images=cropped_images,
        grains_curvature_stats_dict=curvature_stats,
        all_grain_smoothed_data=splined_traces,
        colourmap_normalisation_bounds=curvature_config["colourmap_normalisation_bounds"],
    )

    fig1
    return


if __name__ == "__main__":
    app.run()
