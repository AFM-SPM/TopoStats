"""Plotting data."""

import logging
from importlib import resources
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.patches import Patch, Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.morphology import binary_dilation

import topostats
from topostats.logs.logs import LOGGER_NAME
from topostats.theme import Colormap

# pylint: disable=dangerous-default-value
# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-locals
# pylint: disable=too-many-positional-arguments
# pylint: disable=unused-argument

LOGGER = logging.getLogger(LOGGER_NAME)


def add_pixel_to_nm_to_plotting_config(plotting_config: dict, pixel_to_nm_scaling: float) -> dict:
    """
    Add the pixel to nanometre scaling factor to plotting configs.

    Ensures plots are in nanometres and not pixels.

    Parameters
    ----------
    plotting_config : dict
        TopoStats plotting configuration dictionary.
    pixel_to_nm_scaling : float
        Pixel to nanometre scaling factor for the image.

    Returns
    -------
    dict
        Updated plotting config with the pixel to nanometre scaling factor applied to all the image configurations.
    """
    # Update PLOT_DICT with pixel_to_nm_scaling (can't add _output_dir since it changes)
    plot_opts = {"pixel_to_nm_scaling": pixel_to_nm_scaling}
    for image, options in plotting_config["plot_dict"].items():
        plotting_config["plot_dict"][image] = {**options, **plot_opts}
    return plotting_config


def dilate_binary_image(binary_image: npt.NDArray, dilation_iterations: int) -> npt.NDArray:
    """
    Dilate a supplied binary image a given number of times.

    Parameters
    ----------
    binary_image : npt.NDArray
        Binary image to be dilated.
    dilation_iterations : int
        Number of dilation iterations to be performed.

    Returns
    -------
    npt.NDArray
        Dilated binary image.
    """
    binary_image = binary_image.copy()
    for _ in range(dilation_iterations):
        binary_image = binary_dilation(binary_image)

    return binary_image


def load_mplstyle(style: str | Path) -> None:
    """
    Load the Matplotlibrc parameter file.

    Parameters
    ----------
    style : str | Path
        Path to a Matplotlib Style file.
    """
    if style == "topostats.mplstyle":
        plt.style.use(resources.files(topostats) / style)
    else:
        plt.style.use(style)


class Images:
    """
    Plots image arrays.

    Parameters
    ----------
    data : npt.NDArray
        Numpy array to plot.
    output_dir : str | Path
        Output directory to save the file to.
    filename : str
        Filename to save image as.
    style : str | Path
        Filename of matplotlibrc parameters.
    pixel_to_nm_scaling : float
        The scaling factor showing the real length of 1 pixel in nanometers (nm).
    masked_array : npt.NDArray
        Optional mask array to overlay onto an image.
    plot_coords : npt.NDArray
        ??? Needs defining.
    title : str
        Title for plot.
    image_type : str
        The image data type, options are 'binary' or 'non-binary'.
    module : str
            The name of the module plotting the images.
    image_set : str
        The set of images to process, options are 'core' or 'all'.
    core_set : bool
        Flag to identify image as part of the core image set or not.
    pixel_interpolation : str, optional
        Interpolation to use (default is 'None').
    grain_crop_plot_size_nm : float, optional
        Size in nm of the square cropped grain images if using the grains image set. If -1,
        will use the grain's default bounding box size.
    cmap : str, optional
        Colour map to use (default 'nanoscope', 'afmhot' also available).
    mask_cmap : str
        Colour map to use for the secondary (masked) data (default 'jet_r', 'blu' provides more contrast).
    region_properties : dict
        Dictionary of region properties, adds bounding boxes if specified.
    zrange : list
        Lower and upper bound to clip core images to.
    colorbar : bool
        Optionally add a colorbar to plots, default is False.
    axes : bool
        Optionally add/remove axes from the image.
    num_ticks : tuple[int | None]
        The number of x and y ticks to display on the iage.
    save : bool
        Whether to save the image.
    savefig_format : str, optional
        Format to save the image as.
    histogram_log_axis : bool
        Optionally use a loagrithmic y-axis for the histogram plots.
    histogram_bins : int, optional
        Number of bins for histograms to use.
    savefig_dpi : str | float, optional
        The resolution of the saved plot (default 'figure').
    number_grains : bool
            Optionally number each grain in a plot.
    """

    def __init__(
        self,
        data: npt.NDArray,
        output_dir: str | Path,
        filename: str,
        style: str | Path = None,
        pixel_to_nm_scaling: float = 1.0,
        masked_array: npt.NDArray = None,
        plot_coords: npt.NDArray = None,
        title: str = None,
        image_type: str = "non-binary",
        module: str = "",
        image_set: str = "core",
        core_set: bool = False,
        pixel_interpolation: str | None = None,
        grain_crop_plot_size_nm: float = -1,
        cmap: str | None = None,
        mask_cmap: str = "jet_r",
        region_properties: dict = None,
        zrange: list = None,
        colorbar: bool = True,
        axes: bool = True,
        num_ticks: tuple[int | None] = (None, None),
        save: bool = True,
        savefig_format: str | None = None,
        histogram_log_axis: bool = True,
        histogram_bins: int | None = None,
        savefig_dpi: str | float | None = None,
        number_grains: bool = False,
    ) -> None:
        """
        Initialise the class.

        There are two key parameters that ensure whether an image is plotted that are passed in from the updated
        plotting dictionary. These are the ``image_set`` which defines which images to plot. ``all`` images plots
        everything, or ``core`` only plots the core set.
        There is then the 'core_set' which defines whether an individual images belongs to the 'core_set' or
        not. If it doesn't then it is not plotted when `image_set` is `["core"]`.

        Parameters
        ----------
        data : npt.NDArray
            Numpy array to plot.
        output_dir : str | Path
            Output directory to save the file to.
        filename : str
            Filename to save image as.
        style : str | Path
            Filename of matplotlibrc parameters.
        pixel_to_nm_scaling : float
            The scaling factor showing the real length of 1 pixel in nanometers (nm).
        masked_array : npt.NDArray
            Optional mask array to overlay onto an image.
        plot_coords : npt.NDArray
            ??? Needs defining.
        title : str
            Title for plot.
        image_type : str
            The image data type, options are 'binary' or 'non-binary'.
        module : str
            The name of the module plotting the images.
        image_set : str
            The set of images to process, options are 'core' or 'all'.
        core_set : bool
            Flag to identify image as part of the core image set or not.
        pixel_interpolation : str, optional
            Interpolation to use (default is 'None').
        grain_crop_plot_size_nm : float, optional
            Size in nm of the square cropped grain images if using the grains image set. If -1,
            will use the grain's default bounding box size.
        cmap : str, optional
            Colour map to use (default 'nanoscope', 'afmhot' also available).
        mask_cmap : str
            Colour map to use for the secondary (masked) data (default 'jet_r', 'blu' provides more contrast).
        region_properties : dict
            Dictionary of region properties, adds bounding boxes if specified.
        zrange : list
            Lower and upper bound to clip core images to.
        colorbar : bool
            Optionally add a colorbar to plots, default is False.
        axes : bool
            Optionally add/remove axes from the image.
        num_ticks : tuple[int | None]
            The number of x and y ticks to display on the iage.
        save : bool
            Whether to save the image.
        savefig_format : str, optional
            Format to save the image as.
        histogram_log_axis : bool
            Optionally use a loagrithmic y-axis for the histogram plots.
        histogram_bins : int, optional
            Number of bins for histograms to use.
        savefig_dpi : str | float, optional
            The resolution of the saved plot (default 'figure').
        number_grains : bool
            Optionally number each grain in a plot.
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
        self.module = module
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
        self.number_grains = number_grains

    def plot_histogram_and_save(self) -> tuple | None:
        """
        Plot and save a histogram of the height map.

        Returns
        -------
        tuple | None
            Matplotlib.pyplot figure object and Matplotlib.pyplot axes object.
        """
        if "all" in self.image_set:
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

    def plot_curvatures(
        self,
        image: npt.NDArray,
        cropped_images: dict,
        grains_curvature_stats_dict: dict,
        all_grain_smoothed_data: dict,
        colourmap_normalisation_bounds: tuple[float, float],
    ) -> tuple[plt.Figure | None, plt.Axes | None]:
        """
        Plot curvature intensity and defects of grains in an image.

        Parameters
        ----------
        image : npt.NDArray
            Image to plot.
        cropped_images : dict
            Dictionary containing cropped images of grains and the bounding boxes and padding.
        grains_curvature_stats_dict : dict
            Dictionary of grain curvature statistics.
        all_grain_smoothed_data : dict
            Dictionary containing smoothed grain traces.
        colourmap_normalisation_bounds : tuple[float, float]
            Tuple of the colour map normalisation bounds.

        Returns
        -------
        tuple[plt.Figure | None, plt.Axes | None]
            Matplotlib.pyplot figure object and Matplotlib.pyplot axes object.
        """
        fig, ax = None, None

        # Only plot if image_set is "all" (i.e. user wants all images) or an image is in the core_set
        if "all" in self.image_set or self.module in self.image_set or self.core_set:
            # Get the shape of the image

            shape = image.shape
            fig, ax = plt.subplots(1, 1)
            ax.imshow(
                image,
                extent=(0, shape[1] * self.pixel_to_nm_scaling, 0, shape[0] * self.pixel_to_nm_scaling),
                interpolation=self.interpolation,
                cmap=self.cmap,
                vmin=self.zrange[0],
                vmax=self.zrange[1],
            )

            # For each grain, plot the points with the colour determined by the curvature value
            # Iterate over the grains
            for (_, grain_data_curvature), (_, grain_data_smoothed_trace), (_, grain_image_container) in zip(
                grains_curvature_stats_dict.items(), all_grain_smoothed_data.items(), cropped_images.items()
            ):
                # Get the coordinate for the grain to accurately position the points
                min_row = grain_image_container["bbox"][0]
                min_col = grain_image_container["bbox"][1]

                pad_width = grain_image_container["pad_width"]

                # Iterate over molecules
                for (_, molecule_data_curvature), (
                    _,
                    molecule_data_smoothed_trace,
                ) in zip(grain_data_curvature.items(), grain_data_smoothed_trace.items()):
                    # Normalise the curvature values to the colourmap bounds
                    normalised_curvature = np.array(molecule_data_curvature)
                    normalised_curvature = normalised_curvature - colourmap_normalisation_bounds[0]
                    normalised_curvature = normalised_curvature / (
                        colourmap_normalisation_bounds[1] - colourmap_normalisation_bounds[0]
                    )

                    molecule_trace_coords = molecule_data_smoothed_trace["spline_coords"]
                    # pylint cannot see that mpl.cm.viridis is a valid attribute
                    # pylint: disable=no-member
                    cmap = mpl.cm.coolwarm
                    for index, point in enumerate(molecule_trace_coords):
                        color = cmap(normalised_curvature[index])
                        if index > 0:
                            previous_point = molecule_trace_coords[index - 1]
                            ax.plot(
                                [
                                    (min_col - pad_width + previous_point[1]) * self.pixel_to_nm_scaling,
                                    (min_col - pad_width + point[1]) * self.pixel_to_nm_scaling,
                                ],
                                [
                                    (image.shape[0] - (min_row - pad_width + previous_point[0]))
                                    * self.pixel_to_nm_scaling,
                                    (image.shape[0] - (min_row - pad_width + point[0])) * self.pixel_to_nm_scaling,
                                ],
                                color=color,
                                linewidth=1,
                            )

            # save the figure
            plt.title(self.title)
            plt.xlabel("Nanometres")
            plt.ylabel("Nanometres")
            set_n_ticks(ax, self.num_ticks)
            plt.axis(self.axes)
            fig.tight_layout()
            plt.savefig(
                (self.output_dir / f"{self.filename}.{self.savefig_format}"),
                bbox_inches="tight",
                pad_inches=0,
                dpi=self.savefig_dpi,
            )
            plt.close()

        return fig, ax

    def plot_curvatures_individual_grains(
        self,
        cropped_images: dict,
        grains_curvature_stats_dict: dict,
        all_grains_smoothed_data: dict,
        colourmap_normalisation_bounds: tuple[float, float],
    ) -> None:
        """
        Plot curvature intensity and defects of individual grains.

        Parameters
        ----------
        cropped_images : dict
            Dictionary of cropped images.
        grains_curvature_stats_dict : dict
            Dictionary of grain curvature statistics.
        all_grains_smoothed_data : dict
            Dictionary containing smoothed grain traces.
        colourmap_normalisation_bounds : tuple
            Tuple of the colour map normalisation bounds.
        """
        fig, ax = None, None
        # Only plot if image_set is "all" (i.e. user wants all images) or an image is in the core_set
        if "all" in self.image_set or self.module in self.image_set or self.core_set:
            # Iterate over grains
            for (
                (grain_index, grain_data_curvature),
                (_, grain_data_smoothed_trace),
                (_, grain_image_container),
            ) in zip(grains_curvature_stats_dict.items(), all_grains_smoothed_data.items(), cropped_images.items()):
                grain_image = grain_image_container["original_image"]
                shape = grain_image.shape
                fig, ax = plt.subplots(1, 1)
                ax.imshow(
                    grain_image,
                    extent=(0, shape[1] * self.pixel_to_nm_scaling, 0, shape[0] * self.pixel_to_nm_scaling),
                    interpolation=self.interpolation,
                    cmap=self.cmap,
                    vmin=self.zrange[0],
                    vmax=self.zrange[1],
                )

                # Iterate over molecules
                for (_, molecule_data_curvature), (_, molecule_data_smoothed_trace) in zip(
                    grain_data_curvature.items(), grain_data_smoothed_trace.items()
                ):
                    molecule_trace_coords = molecule_data_smoothed_trace["spline_coords"]

                    # Normalise the curvature values to the colourmap bounds
                    normalised_curvature = np.array(molecule_data_curvature)
                    normalised_curvature = normalised_curvature - colourmap_normalisation_bounds[0]
                    normalised_curvature = normalised_curvature / (
                        colourmap_normalisation_bounds[1] - colourmap_normalisation_bounds[0]
                    )

                    # pylint cannot see that mpl.cm.viridis is a valid attribute
                    # pylint: disable=no-member
                    cmap = mpl.cm.coolwarm

                    for index, point in enumerate(molecule_trace_coords):
                        colour = cmap(normalised_curvature[index])
                        if index > 0:
                            previous_point = molecule_trace_coords[index - 1]
                            ax.plot(
                                [
                                    previous_point[1] * self.pixel_to_nm_scaling,
                                    point[1] * self.pixel_to_nm_scaling,
                                ],
                                [
                                    (shape[0] - previous_point[0]) * self.pixel_to_nm_scaling,
                                    (shape[0] - point[0]) * self.pixel_to_nm_scaling,
                                ],
                                color=colour,
                                linewidth=3,
                            )

                plt.title(self.title)
                plt.xlabel("Nanometres")
                plt.ylabel("Nanometres")
                set_n_ticks(ax, self.num_ticks)
                plt.axis(self.axes)
                fig.tight_layout()
                # plt.savefig(f"./grain_{grain_index}_curvature.png")
                fig.savefig(
                    (self.output_dir / f"{grain_index}_curvature.{self.savefig_format}"),
                    bbox_inches="tight",
                    pad_inches=0,
                    dpi=self.savefig_dpi,
                )
                plt.close()

            LOGGER.debug(
                f"[{self.filename}] : Image saved to : {str(self.output_dir / self.filename)}.{self.savefig_format}"
                f" | DPI: {self.savefig_dpi}"
            )

    def plot_and_save(self):
        """
        Plot and save the image.

        Returns
        -------
        tuple
            Matplotlib.pyplot figure object and Matplotlib.pyplot axes object.
        """
        fig, ax = None, None
        if self.save:
            # Only plot if image_set is "all" (i.e. user wants all images) or an image is in the core_set
            if "all" in self.image_set or self.module in self.image_set or self.core_set:
                fig, ax = self.save_figure()
                LOGGER.debug(
                    f"[{self.filename}] : Image saved to : {str(self.output_dir / self.filename)}.{self.savefig_format}"
                    f" | DPI: {self.savefig_dpi}"
                )
                plt.close()
                return fig, ax
        return fig, ax

    def save_figure(self):
        """
        Save figures as plt.savefig objects.

        Returns
        -------
        tuple
            Matplotlib.pyplot figure object and Matplotlib.pyplot axes object.
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
                    ax.plot(
                        grain_coords[:, 1] * self.pixel_to_nm_scaling,
                        (shape[0] - grain_coords[:, 0]) * self.pixel_to_nm_scaling,
                        c="c",
                        linewidth=1,
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
                if self.number_grains:
                    fig, ax = number_grain_plots(
                        fig,
                        ax,
                        shape,
                        self.region_properties,
                        self.pixel_to_nm_scaling,
                        (2, -2),
                    )
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


def add_bounding_boxes_to_plot(fig, ax, shape: tuple, region_properties: list, pixel_to_nm_scaling: float) -> tuple:
    """
    Add the bounding boxes to a plot.

    Parameters
    ----------
    fig : plt.figure.Figure
        Matplotlib.pyplot figure object.
    ax : plt.axes._subplots.AxesSubplot
        Matplotlib.pyplot axes object.
    shape : tuple
        Tuple of the image-to-be-plot's shape.
    region_properties : list
        Region properties to add bounding boxes from.
    pixel_to_nm_scaling : float
        The scaling factor from px to nm.

    Returns
    -------
    tuple
        Matplotlib.pyplot figure object and Matplotlib.pyplot axes object.
    """
    for region in region_properties:
        min_y, min_x, max_y, max_x = (x * pixel_to_nm_scaling for x in region.bbox)
        # Correct y-axis
        min_y = (shape[0] * pixel_to_nm_scaling) - min_y
        max_y = (shape[0] * pixel_to_nm_scaling) - max_y
        rectangle = Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, fill=False, edgecolor="white", linewidth=2)
        ax.add_patch(rectangle)
    return fig, ax


def number_grain_plots(
    fig, ax, shape: tuple, region_properties: list, pixel_to_nm_scaling: float, offset: tuple
) -> tuple:
    """
    Add the grain numbers to the plot.

    Parameters
    ----------
    fig : plt.figure.Figure
        Matplotlib.pyplot figure object.
    ax : plt.axes._subplots.AxesSubplot
        Matplotlib.pyplot axes object.
    shape : tuple
        Tuple of the image-to-be-plot's shape.
    region_properties : list
        Region properties to add bounding boxes from.
    pixel_to_nm_scaling : float
        The scaling factor from px to nm.
    offset : tuple
        The amount to shift the number to avoid overlap with bounding boxes (x, y).

    Returns
    -------
    tuple
        Matplotlib.pyplot figure object and Matplotlib.pyplot axes object.
    """
    for i, region in enumerate(region_properties):
        min_y, min_x, _max_y, _max_x = (x * pixel_to_nm_scaling for x in region.bbox)
        # Correct y-axis
        min_y = (shape[0] * pixel_to_nm_scaling) - min_y

        # Stop overlap with bbox
        x_loc = min_x + offset[0]
        y_loc = min_y + offset[1]
        numbering = ax.text(x_loc, y_loc, i, fontsize=10, color="white", ha="left", va="top")
        numbering.set_path_effects([path_effects.Stroke(linewidth=2, foreground="black"), path_effects.Normal()])
    return fig, ax


def set_n_ticks(ax: plt.Axes.axes, n_xy: list[int | None, int | None]) -> None:
    """
    Set the number of ticks along the y and x axes and lets matplotlib assign the values.

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
