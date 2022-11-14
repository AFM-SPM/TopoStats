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


class Images:
    """Plots image arrays"""

    def __init__(
        self,
        data: np.array,
        output_dir: Union[str, Path],
        filename: str,
        pixel_to_nm_scaling: float = 1.0,
        data2: np.array = None,
        title: str = None,
        image_type: str = "non-binary",
        image_set: str = "core",
        core_set: bool = False,
        interpolation: str = "nearest",
        cmap: str = "nanoscope",
        region_properties: dict = None,
        zrange: list = [None, None],
        colorbar: bool = True,
        axes: bool = True,
        save: bool = True,
        save_format: str = "png",
        histogram_log_axis: bool = True,
    ) -> None:
        """
        Initialise the class.

        Parameters
        ----------
        data : np.array
            Numpy array to plot.
        output_dir : Union[str, Path]
            Output directory to save the file to.
        filename : Union[str, Path]
            Filename to save image as.
        pixel_to_nm_scaling : float
            The scaling factor showing the real length of 1 pixel, in nm.
        data2 : np.ndarray
            Optional mask array to overlay onto an image.
        title : str
            Title for plot.
        image_type : str
            The image data type - binary or non-binary.
        image_set : str
            The set of images to process - core or all.
        core_set : bool
            Flag to identify image as part of the core image set or not.
        interpolation: str
            Interpolation to use (default 'nearest').
        cmap : str
            Colour map to use (default 'nanoscope', 'afmhot' also available)
        region_properties: dict
            Dictionary of region properties, adds bounding boxes if specified.
        zrange : list
            Lower and upper bound to clip core images to.
        colorbar: bool
            Optionally add a colorbar to plots, default is False.
        axes: bool
            Optionally add/remove axes from the image.
        save: bool
            Whether to save the image.
        save_format: str
            Format to save the image as.
        histogram_log_axis: bool
            Optionally use a logarithmic y axis for the histogram plots
        """

        self.data = data
        self.output_dir = Path(output_dir)
        self.filename = filename
        self.pixel_to_nm_scaling = pixel_to_nm_scaling
        self.data2 = data2
        self.title = title
        self.image_type = image_type
        self.image_set = image_set
        self.core_set = core_set
        self.interpolation = interpolation
        self.cmap = cmap
        self.region_properties = region_properties
        self.zrange = zrange
        self.colorbar = colorbar
        self.axes = axes
        self.save = save
        self.save_format = save_format
        self.histogram_log_axis = histogram_log_axis

    def plot_histogram_and_save(self):
        """
        Plot and save a histogram of the height map

        Returns
        -------
        fig: plt.figure.Figure
            Matplotlib.pyplot figure object
        ax: plt.axes._subplots.AxesSubplot
            Matplotlib.pyplot axes object
        """
        if self.image_set == "all":
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))

            ax.hist(self.data.flatten().astype(float), bins="auto", log=self.histogram_log_axis)
            ax.set_xlabel("pixel height")
            if self.histogram_log_axis:
                ax.set_ylabel("frequency in image (log)")
            else:
                ax.set_ylabel("frequency in image")
            plt.title(self.title)
            plt.savefig(
                (self.output_dir / f"{self.filename}_histogram.{self.save_format}"),
                format=self.save_format,
                bbox_inches="tight",
                pad_inches=0.5,
            )
            plt.close()

            return fig, ax
        else:
            return None

    def plot_and_save(self):
        """
        Plot and save the images with savefig or imsave depending on config file parameters.

        Returns
        -------
        fig: plt.figure.Figure
            Matplotlib.pyplot figure object
        ax: plt.axes._subplots.AxesSubplot
            Matplotlib.pyplot axes object
        """
        fig, ax = None, None
        if self.save:
            if self.image_set == "all" or self.core_set:
                if self.axes or self.colorbar:
                    fig, ax = self.save_figure()
                else:
                    if isinstance(self.data2, np.ndarray) or self.region_properties:
                        fig, ax = self.save_figure()
                    else:
                        self.save_array_figure()
        if "_processed" in self.filename:
            LOGGER.info(
                f"[{self.filename.split('_processed')[0]}] : Image saved to : {str(self.output_dir / self.filename)}"
            )
        return fig, ax

    def save_figure(self):
        """
        This function saves figures as plt.savefig objects.

        Returns
        -------
        fig: plt.figure.Figure
            Matplotlib.pyplot figure object
        ax: plt.axes._subplots.AxesSubplot
            Matplotlib.pyplot axes object
        """
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        shape = self.data.shape
        if isinstance(self.data, np.ndarray):
            im = ax.imshow(
                self.data,
                extent=(0, shape[1] * self.pixel_to_nm_scaling, 0, shape[0] * self.pixel_to_nm_scaling),
                interpolation=self.interpolation,
                cmap=Colormap(self.cmap).get_cmap(),
                vmin=self.zrange[0],
                vmax=self.zrange[1],
            )
            if isinstance(self.data2, np.ndarray):
                self.data2[self.data2 != 0] = 1
                mask = np.ma.masked_where(self.data2 == 0, self.data2)
                ax.imshow(
                    mask,
                    "jet_r",
                    extent=(
                        0,
                        shape[1] * self.pixel_to_nm_scaling,
                        0,
                        shape[0] * self.pixel_to_nm_scaling,
                    ),
                    interpolation=self.interpolation,
                    alpha=0.7,
                )
                patch = [Patch(color=plt.get_cmap("jet_r")(1, 0.7), label="Mask")]
                plt.legend(handles=patch, loc="upper right", bbox_to_anchor=(1, 1.06))

            plt.title(self.title)
            plt.xlabel("Nanometres")
            plt.ylabel("Nanometres")
            plt.axis(self.axes)
            if self.colorbar and self.image_type == "non-binary":
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax, label="Height (Nanometres)")
            if self.region_properties:
                fig, ax = add_bounding_boxes_to_plot(fig, ax, shape, self.region_properties, self.pixel_to_nm_scaling)
            if not self.axes and not self.colorbar:
                plt.title("")
                fig.frameon = False
                plt.savefig(
                    (self.output_dir / f"{self.filename}.{self.save_format}"),
                    format=self.save_format,
                    bbox_inches="tight",
                    pad_inches=0,
                )
            else:
                plt.savefig((self.output_dir / f"{self.filename}.{self.save_format}"), format=self.save_format)
        else:
            plt.xlabel("Nanometres")
            plt.ylabel("Nanometres")
            self.data.show(
                ax=ax,
                extent=(0, shape[1] * self.pixel_to_nm_scaling, 0, shape[0] * self.pixel_to_nm_scaling),
                interpolation=self.interpolation,
                cmap=Colormap(self.cmap).get_cmap(),
            )
        plt.close()
        return fig, ax

    def save_array_figure(self) -> None:
        """This function saves only the image array as an image using plt.imsave"""
        plt.imsave(
            (self.output_dir / f"{self.filename}.{self.save_format}"),
            self.data,
            cmap=Colormap(self.cmap).get_cmap(),
            vmin=self.zrange[0],
            vmax=self.zrange[1],
            format=self.save_format,
        )
        plt.close()


def add_bounding_boxes_to_plot(fig, ax, shape, region_properties: list, pixel_to_nm_scaling: float) -> None:
    """Add the bounding boxes to a plot.

    Parameters
    ----------
    fig: plt.figure.Figure
        Matplotlib.pyplot figure object
    ax: plt.axes._subplots.AxesSubplot.
        Matplotlib.pyplot axes object
    shape: tuple
        Tuple of the image-to-be-plot's shape.
    region_properties:
        Region properties to add bounding boxes from.
    pixel_to_nm_scaling: float
        The scaling factor from px to nm.

    Returns
    -------
    fig: plt.figure.Figure
        Matplotlib.pyplot figure object.
    ax: plt.axes._subplots.AxesSubplot
        Matplotlib.pyplot axes object.
    """
    for region in region_properties:
        min_y, min_x, max_y, max_x = [x * pixel_to_nm_scaling for x in region.bbox]
        # Correct y-axis
        min_y = (shape[0] * pixel_to_nm_scaling) - min_y
        max_y = (shape[0] * pixel_to_nm_scaling) - max_y
        rectangle = Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, fill=False, edgecolor="white", linewidth=2)
        ax.add_patch(rectangle)
    return fig, ax
