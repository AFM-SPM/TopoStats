import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    #Making use of TopoStats outputs
    Running TopoStats from the command line produces a .topostats file for each processed image. This file contains all of the data used at different stages within the TopoStats pipeline including filtered images, grain masks, and traces. These files are a really useful resource for using TopoStats outputs to develop new functions for your own analyses. Here we show one example of using the .topostats file to create a height profile of a DNA minicircle, enabling us to measure the distance between helices.
    """
    )
    return


@app.cell
def _():
    from pprint import pprint

    import h5py
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.signal import find_peaks

    from topostats.io import hdf5_to_dict

    return find_peaks, h5py, hdf5_to_dict, mo, np, plt, pprint


@app.cell
def _(mo):
    mo.md(r"""First we read in the .topostats file and convert it to a usable dictionary format.""")
    return


@app.cell
def _(h5py, hdf5_to_dict):
    filepath = "/Users/laura/TopoStats/tests/resources/notebook3_image.topostats"  # edit this path to locate the notebook3_image.topostats file

    with h5py.File(filepath, "r") as f:
        data_dict = hdf5_to_dict(f, "/")
    return (data_dict,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Printing the `data_dict` allows us to see what is contained within our .topostats file, including all of the file's metadata and other data generated through TopoStats."""
    )
    return


@app.cell
def _(data_dict, pprint):
    pprint(data_dict, depth=3, width=200)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Below is a helper function used to enable visualisation of plots within marimo. We can use it to plot our TopoStats processed image which is stored in `data_dict["image"]`. Note you could also view the raw image using `data_dict[image_original]`."""
    )
    return


@app.cell
def _(data_dict, np, plt):
    def show_image(arr, cmap="afmhot", size=(8, 8), colorbar=True, title=None):
        """Display a 2D NumPy array as an image."""
        arr = np.asarray(arr)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {arr.shape}")

        fig, ax = plt.subplots(figsize=size)
        im = ax.imshow(arr, cmap=cmap)
        if title:
            ax.set_title(title)
        if colorbar:
            fig.colorbar(im, ax=ax, shrink=0.8)
        plt.show()

    show_image(data_dict["image"])
    return (show_image,)


@app.cell
def _(mo):
    mo.md(
        r"""
    In the code below, we use the bounding box information for grain 0, stored in `data_dict["ordered_traces"]["above"]["grain_0"]["mol_0"]["bbox"]` to crop the image and centre the grain of interest.

    We can then extract the height values, provided in `data_dict["ordered_traces"]["above"]["grain_0"]["mol_0"]["heights"]` as well as distances along the trace, provided in `data_dict["ordered_traces"]["above"]["grain_0"]["mol_0"]["distances"]`.
    """
    )
    return


@app.cell
def _(data_dict, np, show_image):
    bbox = data_dict["ordered_traces"]["above"]["grain_0"]["mol_0"]["bbox"]
    cropped_image = data_dict["image"][bbox[0] : bbox[2], bbox[1] : bbox[3]]

    ordered_coords = data_dict["ordered_traces"]["above"]["grain_0"]["mol_0"]["ordered_coords"]
    ordered_image = np.zeros_like(cropped_image)
    ordered_image[ordered_coords[:, 0], ordered_coords[:, 1]] = np.arange(1, len(ordered_coords) + 1)

    height_trace = data_dict["ordered_traces"]["above"]["grain_0"]["mol_0"]["heights"]

    px2nm_ratio = data_dict["pixel_to_nm_scaling"]
    distance_trace = data_dict["ordered_traces"]["above"]["grain_0"]["mol_0"]["distances"] * px2nm_ratio

    show_image(cropped_image)
    return cropped_image, distance_trace, height_trace, ordered_image


@app.cell
def _(mo):
    mo.md(
        r"""We can use the height trace profiles to count the number of helices within our grain, and to measure the distance between helices. By using a threshold of 4nm, we see that DNA helices are shown as equidistant peaks within the height profile plot."""
    )
    return


@app.cell
def _(
    cropped_image,
    distance_trace,
    find_peaks,
    height_trace,
    np,
    ordered_image,
    plt,
):
    fig, ax = plt.subplots(1, 3, gridspec_kw={"width_ratios": [1, 3, 3]}, figsize=(8, 3), dpi=300)

    masked_arr = np.ma.masked_where(ordered_image == 0, ordered_image)

    ax[0].imshow(cropped_image, cmap="nanoscope")
    ax[0].imshow(masked_arr, cmap="viridis")
    ax[0].set_title("Cropped\nimage\nwith trace")
    ax[0].set_axis_off()

    viridis = plt.get_cmap("viridis")
    norm = plt.Normalize(distance_trace.min(), distance_trace.max())
    line_colours = viridis(norm(distance_trace))

    ax[1].plot(distance_trace, height_trace, lw=0.5, zorder=0)
    ax[1].scatter(distance_trace, height_trace, color=line_colours, s=3, zorder=1)
    ax[1].set_title("Trace height profile")

    # identify peaks using scipy
    peaks, _ = find_peaks(height_trace, height=0, distance=4)

    # Calculate distance between DNA helix peaks
    peak_dist_difference = distance_trace[peaks[1:]] - distance_trace[peaks[:-1]]
    print(f"Number of peaks: {len(peaks)}")
    print(f"Mean inter-peak distance: {peak_dist_difference.mean():.1f} ± {peak_dist_difference.std():.1f}")

    line_colours = viridis(norm(distance_trace[peaks]))
    ax[2].plot(distance_trace, height_trace, lw=0.5, zorder=0)
    ax[2].scatter(distance_trace[peaks], height_trace[peaks], color=line_colours, s=3, zorder=1)
    ax[2].set_title("Peaks separated by >4nm")
    return


if __name__ == "__main__":
    app.run()
