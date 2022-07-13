"""Plotting data."""
from pathlib import Path
from typing import Union
import logging
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


from topostats.logs.logs import LOGGER_NAME

LOGGER = logging.getLogger(LOGGER_NAME)


def plot_and_save(
    data: np.array,
    output_dir: Union[str, Path],
    filename: str,
    pixel_to_nm_scaling_factor: float,
    title: str = None,
    interpolation: str = "nearest",
    cmap: str = "afmhot",
    region_properties: dict = None,
    colorbar: bool = False,
):
    """Plot and save an image.

    Parameters
    ----------
    data : np.array
        Numpy array to plot.
    output_dir: Union[str, Path]
        Output directory to save the file to,
    filename : Union[str, Path]
        Filename to save image as.
    title : str
        Title for plot.
    interpolation: str
        Interpolation to use (default 'nearest').
    cmap : str
        Colour map to use (default 'afmhot')
    region_properties: dict
        Dictionary of region properties, adds bounding boxes if specified.
    colorbar: bool
        Optionally add a colorbar to plots, default is False.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    shape = data.shape
    if isinstance(data, np.ndarray):
        im = ax.imshow(data, extent=(0, shape[0] * pixel_to_nm_scaling_factor, 0, shape[1] * pixel_to_nm_scaling_factor), interpolation=interpolation, cmap=cmap)
        plt.title(title)
        plt.xlabel("Nanometres")
        plt.ylabel("Nanometres")
        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
        if region_properties:
            fig, ax = add_bounding_boxes_to_plot(fig, ax, region_properties)
        plt.savefig(output_dir / filename)
    else:
        plt.xlabel("Nanometres")
        plt.ylabel("Nanometres")
        data.show(ax=ax, extent=(0, shape[0] * pixel_to_nm_scaling_factor, 0, shape[1] * pixel_to_nm_scaling_factor), interpolation=interpolation, cmap=cmap)
    plt.close()
    LOGGER.info(f"[{Path(output_dir).parts[1]}] : Image saved to : {str(output_dir / filename)}")
    return fig, ax


def add_bounding_boxes_to_plot(fig, ax, region_properties) -> None:
    """Add the bounding boxes to a plot.

    Parameters
    ----------
    fig :

    ax :
    region_properties:
        Region properties to add bounding boxes from.
    """
    for region in region_properties:
        min_row, min_col, max_row, max_col = region.bbox
        rectangle = Rectangle(
            (min_col, min_row), max_col - min_col, max_row - min_row, fill=False, edgecolor="white", linewidth=2
        )
        ax.add_patch(rectangle)
    return fig, ax
