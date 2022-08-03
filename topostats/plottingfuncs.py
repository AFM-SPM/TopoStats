"""Plotting data."""
from pathlib import Path
from typing import Union
import logging

from matplotlib.patches import Rectangle, Patch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from topostats.logs.logs import LOGGER_NAME
from topostats.theme import Colormap

LOGGER = logging.getLogger(LOGGER_NAME)


def plot_and_save(
    data: np.array,
    output_dir: Union[str, Path],
    filename: str,
    pixel_to_nm_scaling_factor: float,
    data2: np.array = None,
    title: str = None,
    type: str = "non-binary",
    image_set: str = "core",
    core_set: bool = False,
    interpolation: str = "nearest",
    cmap: str = "nanoscope",
    region_properties: dict = None,
    zrange: list = [None, None],
    colorbar: bool = True,
    save: bool = True,
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
        Colour map to use (default 'nanoscope', 'afmhot' also available)
    region_properties: dict
        Dictionary of region properties, adds bounding boxes if specified.
    colorbar: bool
        Optionally add a colorbar to plots, default is False.
    save: bool
        Whether to save the image.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    shape = data.shape
    if isinstance(data, np.ndarray):
        if not core_set:
            zrange=[None,None]
        im = ax.imshow(
            data,
            extent=(0, shape[0] * pixel_to_nm_scaling_factor, 0, shape[1] * pixel_to_nm_scaling_factor),
            interpolation=interpolation,
            cmap=Colormap(cmap).get_cmap(),
            vmin=zrange[0], 
            vmax=zrange[1],
        )
        if isinstance(data2, np.ndarray):
            mask = np.ma.masked_where(data2==0, data2)
            ax.imshow(mask,
             'jet_r',
             extent=(0, shape[0] * pixel_to_nm_scaling_factor, 0, shape[1] * pixel_to_nm_scaling_factor),
             interpolation=interpolation,
             alpha=0.7)
            patch = [Patch(color=plt.get_cmap('jet_r')(1, 0.7), label='Mask')]
            plt.legend(handles=patch, loc='upper right', bbox_to_anchor=(1,1.06))

        plt.title(title)
        plt.xlabel("Nanometres")
        plt.ylabel("Nanometres")
        if colorbar and type == "non-binary":
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax, label="Height (Nanometres)")
        if region_properties:
            fig, ax = add_bounding_boxes_to_plot(fig, ax, region_properties, pixel_to_nm_scaling_factor)

        if save:
            if image_set=="all" or core_set:
                plt.savefig(output_dir / filename)
                if '_processed' in filename:
                    LOGGER.info(f"[{filename.split('_processed')[0]}] : Image saved to : {str(output_dir / filename)}")
    else:
        plt.xlabel("Nanometres")
        plt.ylabel("Nanometres")
        data.show(
            ax=ax,
            extent=(0, shape[0] * pixel_to_nm_scaling_factor, 0, shape[1] * pixel_to_nm_scaling_factor),
            interpolation=interpolation,
            cmap=Colormap(cmap).get_cmap(),
        )
    plt.close()
    return fig, ax


def add_bounding_boxes_to_plot(fig, ax, region_properties: list, pixel_to_nm_scaling_factor: float) -> None:
    """Add the bounding boxes to a plot.

    Parameters
    ----------
    fig :

    ax :
    region_properties:
        Region properties to add bounding boxes from.
    pixel_to_nm_scaling_factor: float
    """
    for region in region_properties:
        min_y, min_x, max_y, max_x = [x * pixel_to_nm_scaling_factor for x in region.bbox]
        # Correct y-axis
        min_y = (1024 * pixel_to_nm_scaling_factor) - min_y
        max_y = (1024 * pixel_to_nm_scaling_factor) - max_y
        rectangle = Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, fill=False, edgecolor="white", linewidth=2)
        ax.add_patch(rectangle)
    return fig, ax
