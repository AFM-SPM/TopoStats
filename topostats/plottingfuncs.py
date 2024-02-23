"""Plotting data."""

from __future__ import annotations

import importlib.resources as pkg_resources
import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.morphology import binary_dilation

import topostats
from topostats.logs.logs import LOGGER_NAME
from topostats.theme import Colormap

# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
# pylint: disable=dangerous-default-value

LOGGER = logging.getLogger(LOGGER_NAME)

# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=dangerous-default-value


def add_pixel_to_nm_to_plotting_config(plotting_config: dict, pixel_to_nm_scaling: float) -> dict:
    """Add the pixel to nanometre scaling factor to plotting configs.

    Ensures plots are in nanometres and not pixels.

    Parameters
    ----------
    plotting_config: dict
        TopoStats plotting configuration dictionary
    pixel_to_nm_scaling: float
        Pixel to nanometre scaling factor for the image.

    Returns
    -------
    plotting_config: dict
        Updated plotting config with the pixel to nanometre scaling factor
        applied to all the image configurations.
    """
    # Update PLOT_DICT with pixel_to_nm_scaling (can't add _output_dir since it changes)
    plot_opts = {"pixel_to_nm_scaling": pixel_to_nm_scaling}
    for image, options in plotting_config["plot_dict"].items():
        plotting_config["plot_dict"][image] = {**options, **plot_opts}
    return plotting_config


def dilate_binary_image(binary_image: np.ndarray, dilation_iterations: int) -> np.ndarray:
    """Dilate a supplied binary image a given number of times.

    Parameters
    ----------
    binary_image: np.ndarray
        Binary image to be dilated
    dilation_iterations: int
        Number of dilation iterations to be performed

    Returns
    -------
    binary_image: np.ndarray
        Dilated binary image
    """
    binary_image = binary_image.copy()
    for _ in range(dilation_iterations):
        binary_image = binary_dilation(binary_image)

    return binary_image


def load_mplstyle(style: str | Path) -> None:
    """Load the Matplotlibrc parameter file.

    Parameters
    ----------
    style: str | Path
        Path to a Matplotlib Style file.

    Returns
    -------
    None
        Only loads the style file.
    """
    if style == "topostats.mplstyle":
        plt.style.use(pkg_resources.files(topostats) / style)
    else:
        plt.style.use(style)


class Images:
    """Plots image arrays."""

    def __init__(
        self,
        data: np.array,
        output_dir: str | Path,
        filename: str,
        style: str | Path = None,
        pixel_to_nm_scaling: float = 1.0,
        masked_array: np.array = None,
        plot_coords: np.array = None,
        title: str = None,
        image_type: str = "non-binary",
        image_set: str = "core",
        core_set: bool = False,
        pixel_interpolation: str | None = None,
        cmap: str | None = None,
        mask_cmap: str = "jet_r",
        region_properties: dict = None,
        zrange: list = None,
        colorbar: bool = True,
        axes: bool = True,
        num_ticks: list[int | None, int | None] = (None, None),
        save: bool = True,
        savefig_format: str | None = None,
        histogram_log_axis: bool = True,
        histogram_bins: int | None = None,
        savefig_dpi: str | float | None = None,
    ) -> None:
        """
        Initialise the class.

        There are two key parameters that ensure whether and image is plotted that are passed in from the update
        plotting dictionary. These are the `image_set` which defines whether to plot 'all' images or just the `core`
        set. There is then the 'core_set' which defines whether an individual images belongs to the 'core_set' or not.
        If it doesn't then it is not plotted when `image_set == "core"`.

        Parameters
        ----------
        data : np.array
            Numpy array to plot.
        output_dir : Union[str, Path]
            Output directory to save the file to.
        filename : Union[str, Path]
            Filename to save image as.
        style: dict
            Filename of matploglibrc Params.
        pixel_to_nm_scaling : float
            The scaling factor showing the real length of 1 pixel, in nm.
        masked_array : np.ndarray
            Optional mask array to overlay onto an image.
        title : str
            Title for plot.
        image_type : str
            The image data type - binary or non-binary.
        image_set : str
            The set of images to process - core or all.
        core_set : bool
            Flag to identify image as part of the core image set or not.
        pixel_interpolation: Union[str, None]
            Interpolation to use (default: None).
        cmap : str
            Colour map to use (default 'nanoscope', 'afmhot' also available).
        mask_cmap : str
            Colour map to use for the secondary (masked) data (default 'jet_r', 'blu' proivides more contrast).
        region_properties: dict
            Dictionary of region properties, adds bounding boxes if specified.
        zrange : list
            Lower and upper bound to clip core images to.
        colorbar: bool
            Optionally add a colorbar to plots, default is False.
        axes: bool
            Optionally add/remove axes from the image.
        num_ticks: list[int, int]
            The number of x and y ticks to display on the image.
        save: bool
            Whether to save the image.
        save_format: str
            Format to save the image as.
        histogram_log_axis: bool
            Optionally use a logarithmic y axis for the histogram plots.
        histogram_bin: int
            Number of bins for histograms to use.
        savefig_dpi: Union[str, float]
            The resolution of the saved plot (default 'figure').
        """
        if style is None:
            style = "topostats.mplstyle"
        load_mplstyle(style)
        if zrange is None:
            zrange = [None, None]
        self.data = data
        self.output_dir = Path(output_dir)
        self.filename = filename
        self.pixel_to_nm_scaling = pixel_to_nm_scaling
        self.masked_array = masked_array
        self.plot_coords = plot_coords
        self.title = title
        self.image_type = image_type
        self.image_set = image_set
        self.core_set = core_set
        self.interpolation = mpl.rcParams["image.interpolation"] if pixel_interpolation is None else pixel_interpolation
        cmap = mpl.rcParams["image.cmap"] if cmap is None else cmap
        self.cmap = Colormap(cmap).get_cmap()
        self.mask_cmap = Colormap(mask_cmap).get_cmap()
        self.region_properties = region_properties
        self.zrange = zrange
        self.colorbar = colorbar
        self.axes = axes
        self.num_ticks = num_ticks
        self.save = save
        self.savefig_format = mpl.rcParams["savefig.format"] if savefig_format is None else savefig_format
        self.histogram_log_axis = histogram_log_axis
        self.histogram_bins = mpl.rcParams["hist.bins"] if histogram_bins is None else histogram_bins
        self.savefig_dpi = mpl.rcParams["savefig.dpi"] if savefig_dpi is None else savefig_dpi

    def plot_histogram_and_save(self):
        """
        Plot and save a histogram of the height map.

        Returns
        -------
        fig: plt.figure.Figure
            Matplotlib.pyplot figure object
        ax: plt.axes._subplots.AxesSubplot
            Matplotlib.pyplot axes object
        """
        if self.image_set == "all":
            fig, ax = plt.subplots(1, 1)

            ax.hist(self.data.flatten().astype(float), bins=self.histogram_bins, log=self.histogram_log_axis)
            ax.set_xlabel("pixel height")
            if self.histogram_log_axis:
                ax.set_ylabel("frequency in image (log)")
            else:
                ax.set_ylabel("frequency in image")
            plt.title(self.title)
            plt.savefig(
                (self.output_dir / f"{self.filename}_histogram.{self.savefig_format}"),
                bbox_inches="tight",
                pad_inches=0.5,
                dpi=self.savefig_dpi,
            )
            plt.close()

            return fig, ax
        return None

    def plot_and_save(self):
        """
        Plot and save the image.

        Returns
        -------
        fig: plt.figure.Figure
            Matplotlib.pyplot figure object
        ax: plt.axes._subplots.AxesSubplot
            Matplotlib.pyplot axes object
        """
        fig, ax = None, None
        if self.save:
            # Only plot if image_set is "all" (i.e. user wants all images) or an image is in the core_set
            if self.image_set == "all" or self.core_set:
                fig, ax = self.save_figure()
                LOGGER.info(
                    f"[{self.filename}] : Image saved to : {str(self.output_dir / self.filename)}.{self.savefig_format}"
                    f" | DPI: {self.savefig_dpi}"
                )
                plt.close()
                return fig, ax
        return fig, ax

    def save_figure(self):
        """Save figures as plt.savefig objects.

        Returns
        -------
        fig: plt.figure.Figure
            Matplotlib.pyplot figure object
        ax: plt.axes._subplots.AxesSubplot
            Matplotlib.pyplot axes object
        """
        fig, ax = plt.subplots(1, 1)
        shape = self.data.shape
        if isinstance(self.data, np.ndarray):
            im = ax.imshow(
                self.data,
                extent=(0, shape[1] * self.pixel_to_nm_scaling, 0, shape[0] * self.pixel_to_nm_scaling),
                interpolation=self.interpolation,
                cmap=self.cmap,
                vmin=self.zrange[0],
                vmax=self.zrange[1],
            )
            if isinstance(self.masked_array, np.ndarray):
                #self.masked_array[self.masked_array != 0] = 1
                # If the image is too large for singles to be resolved in the mask, then dilate the mask proportionally
                # to image size to enable clear viewing.
                if np.max(self.masked_array.shape) > 500:
                    dilation_strength = int(np.max(self.masked_array.shape) / 256)
                    """
                    self.masked_array = dilate_binary_image(
                        binary_image=self.masked_array, dilation_iterations=dilation_strength
                    )
                    """
                mask = np.ma.masked_where(self.masked_array == 0, self.masked_array)
                ax.imshow(
                    mask,
                    cmap=self.mask_cmap,
                    extent=(
                        0,
                        shape[1] * self.pixel_to_nm_scaling,
                        0,
                        shape[0] * self.pixel_to_nm_scaling,
                    ),
                    interpolation=self.interpolation,
                    alpha=0.7,
                )
                patch = [Patch(color=self.mask_cmap(1, 0.7), label="Mask")]
                plt.legend(handles=patch, loc="upper right", bbox_to_anchor=(1.02, 1.09))
            # if coordinates are provided (such as in splines, plot those)
            elif self.plot_coords is not None:
                for grain_coords in self.plot_coords:
                    plt.plot(
                        grain_coords[:, 1] * self.pixel_to_nm_scaling,
                        (shape[1] - grain_coords[:, 0]) * self.pixel_to_nm_scaling,
                        c="c",
                        linewidth=2.5,
                    )

            plt.title(self.title)
            plt.xlabel("Nanometres")
            plt.ylabel("Nanometres")
            set_n_ticks(ax, self.num_ticks)
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
                plt.box(False)
                plt.tight_layout()
                plt.savefig(
                    (self.output_dir / f"{self.filename}.{self.savefig_format}"),
                    bbox_inches="tight",
                    pad_inches=0,
                    dpi=self.savefig_dpi,
                )
            else:
                plt.savefig((self.output_dir / f"{self.filename}.{self.savefig_format}"), dpi=self.savefig_dpi)
        else:
            plt.xlabel("Nanometres")
            plt.ylabel("Nanometres")
            self.data.show(
                ax=ax,
                extent=(0, shape[1] * self.pixel_to_nm_scaling, 0, shape[0] * self.pixel_to_nm_scaling),
                interpolation=self.interpolation,
                cmap=self.cmap,
            )
        plt.close()
        return fig, ax


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
        min_y, min_x, max_y, max_x = (x * pixel_to_nm_scaling for x in region.bbox)
        # Correct y-axis
        min_y = (shape[0] * pixel_to_nm_scaling) - min_y
        max_y = (shape[0] * pixel_to_nm_scaling) - max_y
        rectangle = Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, fill=False, edgecolor="white", linewidth=2)
        ax.add_patch(rectangle)
    return fig, ax


def set_n_ticks(ax: plt.Axes.axes, n_xy: list[int | None, int | None]) -> None:
    """Set the number of ticks along the y and x axes and lets matplotlib assign the values.

    Parameters
    ----------
    ax : plt.Axes.axes
        The axes to add ticks to.
    n_xy : list[int, int]
        The number of ticks.

    Returns
    -------
    plt.Axes.axes
        The axes with the new ticks.
    """
    if n_xy[0] is not None:
        xlim = ax.get_xlim()
        xstep = (max(xlim) - min(xlim)) / (n_xy[0] - 1)
        xticks = np.arange(min(xlim), max(xlim) + xstep, xstep)
        ax.set_xticks(np.round(xticks))
    if n_xy[1] is not None:
        ylim = ax.get_ylim()
        ystep = (max(ylim) - min(ylim)) / (n_xy[1] - 1)
        yticks = np.arange(min(ylim), max(ylim) + ystep, ystep)
        ax.set_yticks(np.round(yticks))
