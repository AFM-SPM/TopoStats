from pathlib import Path
from typing import Union
import logging
from configparser import Interpolation
import matplotlib.pyplot as plt
import numpy as np


def plot_and_save(data: np.array, filename: Union[str, Path], title: str = None, interpolation: str='nearest', cmap: str
                  = 'afmhot', region_properties: dict = None):
    """Plot and save an image.

    Parameters
    ----------
    data : np.array
        Numpy array to plot.
    filename : Union[str, Path]
        Filename to save image as.
    title : str
        Title for plot.
    interpolation: str
        Interpolation to use (default 'nearest').
    cmap : str
        Colour map to use (default 'afmhot')

    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    if isinstance(data, np.ndarray):
        ax.imshow(data, interpolation=interpolation, cmap=cmap)
        if region_properties:
            fig, ax = add_bounding_boxes_to_plot(fig, ax, region_properties)
        plt.title(title)
        plt.savefig(filename)
    else:
        data.show(ax=ax, interpolation=interpolation, cmap=cmap)
    plt.close()
    logging.info(f'Image saved to : {filename}')
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
        rectangle = mpl.patches.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row,
                                            fill=False, edgecolor='white', linewidth=2)
        ax.add_patch(rectangle)
    return fig, ax
