"""Module for filtering 2D Numpy arrays."""

import logging

import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit

# pylint: disable=no-name-in-module
from skimage.filters import gaussian

from topostats import scars
from topostats.logs.logs import LOGGER_NAME
from topostats.utils import get_mask, get_thresholds

LOGGER = logging.getLogger(LOGGER_NAME)

# noqa: PLR0913
# pylint: disable=fixme
# pylint: disable=broad-except
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-branches
# pylint: disable=dangerous-default-value


class Filters:
    """
    Class for filtering scans.

    Parameters
    ----------
    image : npt.NDArray
        The raw image from the Atomic Force Microscopy machine.
    filename : str
        The filename (used in logging only).
    pixel_to_nm_scaling : float
        Value for converting pixels to nanometers.
    row_alignment_quantile : float
        Quantile (0.0 to 1.0) to be used to determine the average background for the image below values may improve
        flattening of large features.
    threshold_method : str
        Method for thresholding, default 'otsu', valid options 'otsu', 'std_dev' and 'absolute'.
    otsu_threshold_multiplier : float
        Value for scaling the derived Otsu threshold.
    threshold_std_dev : dict
        If using the 'std_dev' threshold method. Dictionary that contains above and below threshold values for the
        number of standard deviations from the mean to threshold.
    threshold_absolute : dict
        If using the 'absolute' threshold method. Dictionary that contains above and below absolute threshold values
        for flattening.
    gaussian_size : float
        If using the 'absolute' threshold method. Dictionary that contains above and below absolute threshold values
        for flattening.
    gaussian_mode : str
        Method passed to 'skimage.filters.gaussian(mode = gaussian_mode)'.
    remove_scars : dict
        Dictionary containing configuration parameters for the scar removal function.
    """  # numpydoc: ignore=PR01

    def __init__(
        self,
        image: npt.NDArray,
        filename: str,
        pixel_to_nm_scaling: float,
        row_alignment_quantile: float = 0.5,
        threshold_method: str = "otsu",
        otsu_threshold_multiplier: float = 1.7,
        threshold_std_dev: dict | None = None,
        threshold_absolute: dict | None = None,
        gaussian_size: float = None,
        gaussian_mode: str = "nearest",
        remove_scars: dict = None,
    ):
        """
        Initialise the class.

        Parameters
        ----------
        image : npt.NDArray
            The raw image from the Atomic Force Microscopy machine.
        filename : str
            The filename (used in logging only).
        pixel_to_nm_scaling : float
            Value for converting pixels to nanometers.
        row_alignment_quantile : float
            Quantile (0.0 to 1.0) to be used to determine the average background for the image below values may improve
            flattening of large features.
        threshold_method : str
            Method for thresholding, default 'otsu', valid options 'otsu', 'std_dev' and 'absolute'.
        otsu_threshold_multiplier : float
            Value for scaling the derived Otsu threshold.
        threshold_std_dev : dict
            If using the 'std_dev' threshold method. Dictionary that contains above and below threshold values for the
            number of standard deviations from the mean to threshold.
        threshold_absolute : dict
            If using the 'absolute' threshold method. Dictionary that contains above and below absolute threshold values
            for flattening.
        gaussian_size : float
            If using the 'absolute' threshold method. Dictionary that contains above and below absolute threshold values
            for flattening.
        gaussian_mode : str
            Method passed to 'skimage.filters.gaussian(mode = gaussian_mode)'.
        remove_scars : dict
            Dictionary containing configuration parameters for the scar removal function.
        """
        self.filename = filename
        self.pixel_to_nm_scaling = pixel_to_nm_scaling
        self.gaussian_size = gaussian_size
        self.gaussian_mode = gaussian_mode
        self.row_alignment_quantile = row_alignment_quantile
        self.threshold_method = threshold_method
        self.otsu_threshold_multiplier = otsu_threshold_multiplier
        # Convert to lists since the thresholding function expects lists of thresholds but
        # we don't want to use more than one value for the filters step.
        if threshold_std_dev is None:
            threshold_std_dev = {"above": 1.0, "below": 1.0}
        else:
            self.threshold_std_dev = {
                "above": [threshold_std_dev["above"]],
                "below": [threshold_std_dev["below"]],
            }
        if threshold_absolute is None:
            threshold_absolute = {"above": 1.0, "below": 10.0}
        else:
            self.threshold_absolute = {
                "above": [threshold_absolute["above"]],
                "below": [threshold_absolute["below"]],
            }
        self.remove_scars_config = remove_scars
        self.images = {
            "pixels": image,
            "initial_median_flatten": None,
            "initial_tilt_removal": None,
            "initial_quadratic_removal": None,
            "initial_scar_removal": None,
            "initial_zero_average_background": None,
            "masked_median_flatten": None,
            "masked_tilt_removal": None,
            "masked_quadratic_removal": None,
            "secondary_scar_removal": None,
            "scar_mask": None,
            "mask": None,
            "final_zero_average_background": None,
            "gaussian_filtered": None,
        }
        self.thresholds = None
        self.medians = {"rows": None, "cols": None}
        self.results = {
            "diff": None,
            "median_row_height": None,
            "x_gradient": None,
            "y_gradient": None,
            "threshold": None,
        }

    def median_flatten(
        self, image: npt.NDArray, mask: npt.NDArray = None, row_alignment_quantile: float = 0.5
    ) -> npt.NDArray:
        """
        Flatten images using median differences.

        Flatten the rows of an image, aligning the rows and centering the median around zero. When used with a mask,
        this has the effect of centering the background data on zero.

        Note this function does not handle scars.

        Parameters
        ----------
        image : npt.NDArray
            2-D image of the data to align the rows of.
        mask : npt.NDArray
            Boolean array of points to mask (ignore).
        row_alignment_quantile : float
            Quantile (in the range 0.0 to 1.0) used for defining the average background.

        Returns
        -------
        npt.NDArray
            Copy of the input image with rows aligned.
        """
        image = image.copy()
        if mask is not None:
            read_matrix = np.ma.masked_array(image, mask=mask, fill_value=np.nan).filled()
            LOGGER.debug(f"[{self.filename}] : Median flattening with mask")
        else:
            read_matrix = image
            LOGGER.debug(f"[{self.filename}] : Median flattening without mask")

        for row in range(image.shape[0]):
            # Get the median of the row
            m = np.nanquantile(read_matrix[row, :], row_alignment_quantile)
            if not np.isnan(m):
                image[row, :] -= m
            else:
                LOGGER.warning(
                    """f[{self.filename}] Large grain detected image can not be
processed, please refer to https://github.com/AFM-SPM/TopoStats/discussions for more information."""
                )

        return image

    def remove_tilt(self, image: npt.NDArray, mask: npt.NDArray = None) -> npt.NDArray:
        """
        Remove the planar tilt from an image (linear in 2D spaces).

        Uses a linear fit of the medians of the rows and columns to determine the linear slants in x and y directions
        and then subtracts the fit from the columns.

        Parameters
        ----------
        image : npt.NDArray
            2-D image of the data to remove the planar tilt from.
        mask : npt.NDArray
            Boolean array of points to mask (ignore).

        Returns
        -------
        npt.NDArray
            Numpy array of image with tilt removed.
        """
        image = image.copy()
        if mask is not None:
            read_matrix = np.ma.masked_array(image, mask=mask, fill_value=np.nan).filled()
            LOGGER.debug(f"[{self.filename}] : Plane tilt removal with mask")
        else:
            read_matrix = image
            LOGGER.debug(f"[{self.filename}] : Plane tilt removal without mask")

        # Line of best fit
        # Calculate medians
        medians_x = [np.nanmedian(read_matrix[:, i]) for i in range(read_matrix.shape[1])]
        medians_y = [np.nanmedian(read_matrix[j, :]) for j in range(read_matrix.shape[0])]
        LOGGER.debug(f"[{self.filename}] [remove_tilt] medians_x   : {medians_x}")
        LOGGER.debug(f"[{self.filename}] [remove_tilt] medians_y   : {medians_y}")

        # Fit linear x
        px = np.polyfit(range(0, len(medians_x)), medians_x, 1)
        LOGGER.debug(f"[{self.filename}] : x-polyfit 1st order: {px}")
        py = np.polyfit(range(0, len(medians_y)), medians_y, 1)
        LOGGER.debug(f"[{self.filename}] : y-polyfit 1st order: {py}")

        if px[0] != 0:
            if not np.isnan(px[0]):
                LOGGER.debug(f"[{self.filename}] : Removing x plane tilt")
                for row in range(0, image.shape[0]):
                    for col in range(0, image.shape[1]):
                        image[row, col] -= px[0] * (col)
            else:
                LOGGER.debug(f"[{self.filename}] : x gradient is nan, skipping plane tilt x removal")
        else:
            LOGGER.debug("[{self.filename}] : x gradient is zero, skipping plane tilt x removal")

        if py[0] != 0:
            if not np.isnan(py[0]):
                LOGGER.debug(f"[{self.filename}] : removing y plane tilt")
                for row in range(0, image.shape[0]):
                    for col in range(0, image.shape[1]):
                        image[row, col] -= py[0] * (row)
            else:
                LOGGER.debug("[{self.filename}] : y gradient is nan, skipping plane tilt y removal")
        else:
            LOGGER.debug("[{self.filename}] : y gradient is zero, skipping plane tilt y removal")

        return image

    def remove_nonlinear_polynomial(self, image: npt.NDArray, mask: npt.NDArray | None = None) -> npt.NDArray:
        """
        Fit and remove a "saddle" shaped nonlinear polynomial from the image.

        "Saddles" with the form a + b * x * y - c * x - d * y from the supplied image. AFM images sometimes contain a
        "saddle" shape trend to their background, and so to remove them we fit a nonlinear polynomial of x and y and
        then subtract the fit from the image.

        If these trends are not removed, then the image will not flatten properly and will leave opposite diagonal
        corners raised or lowered.

        Parameters
        ----------
        image : npt.NDArray
            2-D numpy height-map array of floats with a polynomial trend to remove.
        mask : npt.NDArray, optional
            2-D Numpy boolean array used to mask any points in the image that are deemed not to be part of the
            height-map's background data.

        Returns
        -------
        npt.NDArray
            Image with the polynomial trend subtracted.
        """
        # Script has a lot of locals but I feel this is necessary for readability?
        # pylint: disable=too-many-locals

        # Define the polynomial function to fit to the image
        def model_func(x: float, y: float, a: float, b: float, c: float, d: float) -> float:
            """
            Polynomial function to fit to the image.

            Parameters
            ----------
            x : float
                X.
            y : float
                Y.
            a : float
                A.
            b : float
                B.
            c : float
                C.
            d : float
                D.

            Returns
            -------
            float
                Result of applying the polynomial a + (b * x * y) - (c * x) - (d * y).
            """
            return a + b * x * y - c * x - d * y

        image = image.copy()
        if mask is not None:
            read_matrix = np.ma.masked_array(image, mask=mask, fill_value=np.nan).filled()
        else:
            read_matrix = image

        # Construct a meshgrid of x and y points for fitting to the z heights
        xdata, ydata = np.meshgrid(np.arange(read_matrix.shape[1]), np.arange(read_matrix.shape[0]))
        zdata = read_matrix

        # Only use data that is not nan. Nans may be in the image from the
        # masked array. Curve fitting cannot handle nans.
        nan_mask = ~np.isnan(zdata)
        xdata_nans_removed = xdata[nan_mask]
        ydata_nans_removed = ydata[nan_mask]
        zdata_nans_removed = zdata[nan_mask]

        # Convert the z data to a 1D array
        zdata = zdata.ravel()
        zdata_nans_removed = zdata_nans_removed.ravel()

        # Stack the x, y meshgrid data after converting them to 1D
        xy_data_stacked = np.vstack((xdata_nans_removed.ravel(), ydata_nans_removed.ravel()))

        # Fit the model to the data
        # Note: pylint is flagging the tuple unpacking regarding an internal line of scipy.optimize._minpack_py : 910.
        # This isn't actually an issue though as the extended tuple output is only provided if the 'full_output' flag is
        # provided as a kwarg in curve_fit.
        popt, _pcov = curve_fit(  # pylint: disable=unbalanced-tuple-unpacking
            lambda x, a, b, c, d: model_func(x[0], x[1], a, b, c, d),
            xy_data_stacked,
            zdata_nans_removed,
        )

        # Unpack the optimised parameters
        a, b, c, d = popt
        LOGGER.debug(
            f"[{self.filename}] : Nonlinear polynomial removal optimal params: const: {a} xy: {b} x: {c} y: {d}"
        )

        # Use the optimised parameters to construct a prediction of the underlying surface
        z_pred = model_func(xdata, ydata, a, b, c, d)
        # Subtract the fitted nonlinear polynomial from the image
        image -= z_pred

        return image

    def remove_quadratic(self, image: npt.NDArray, mask: npt.NDArray = None) -> npt.NDArray:
        """
        Remove the quadratic bowing that can be seen in some large-scale AFM images.

        Use a simple quadratic fit on the medians of the columns of the image and then subtracts the calculated
        quadratic from the columns.

        Parameters
        ----------
        image : npt.NDArray
            2-D image of the data to remove the quadratic from.
        mask : npt.NDArray
            Boolean array of points to mask (ignore).

        Returns
        -------
        npt.NDArray
            Image with the quadratic bowing removed.
        """
        image = image.copy()
        if mask is not None:
            read_matrix = np.ma.masked_array(image, mask=mask, fill_value=np.nan).filled()
            LOGGER.debug(f"[{self.filename}] : Remove quadratic bow with mask")
        else:
            read_matrix = image
            LOGGER.debug(f"[{self.filename}] : Remove quadratic bow without mask")

        # Calculate medians
        medians_x = [np.nanmedian(read_matrix[:, i]) for i in range(read_matrix.shape[1])]

        # Fit quadratic x
        px = np.polyfit(range(0, len(medians_x)), medians_x, 2)
        LOGGER.debug(f"[{self.filename}] : x polyfit 2nd order: {px}")

        # Handle divide by zero
        if px[0] != 0:
            if not np.isnan(px[0]):
                # Remove quadratic in x
                cx = -px[1] / (2 * px[0])
                for row in range(0, image.shape[0]):
                    for col in range(0, image.shape[1]):
                        image[row, col] -= px[0] * (col - cx) ** 2
            else:
                LOGGER.debug(f"[{self.filename}] : Quadratic polyfit returns nan, skipping quadratic removal")
        else:
            LOGGER.debug(f"[{self.filename}] : Quadratic polyfit returns zero, skipping quadratic removal")

        return image

    @staticmethod
    def calc_diff(array: npt.NDArray) -> npt.NDArray:
        """
        Calculate the difference between the last and first rows of a 2-D array.

        Parameters
        ----------
        array : npt.NDArray
            A Numpy array.

        Returns
        -------
        npt.NDArray
            An array of the difference between the last and first rows of an array.
        """
        return array[-1] - array[0]

    def calc_gradient(self, array: npt.NDArray, shape: int) -> npt.NDArray:
        """
        Calculate the gradient of an array.

        Parameters
        ----------
        array : npt.NDArray
            Array for gradient to be calculated.
        shape : int
            Shape of the array.

        Returns
        -------
        npt.NDArray
            Gradient across the array.
        """
        return self.calc_diff(array) / shape

    def average_background(self, image: npt.NDArray, mask: npt.NDArray = None) -> npt.NDArray:
        """
        Zero the background by subtracting the non-masked mean from all pixels.

        Parameters
        ----------
        image : npt.NDArray
            Numpy array representing the image.
        mask : npt.NDArray
            Mask of the array, should have the same dimensions as image.

        Returns
        -------
        npt.NDArray
            Numpy array of image zero averaged.
        """
        if mask is None:
            mask = np.zeros_like(image)
        mean = np.mean(image[mask == 0])
        LOGGER.debug(f"[{self.filename}] : Zero averaging background : {mean} nm")
        return image - mean

    def gaussian_filter(self, image: npt.NDArray, **kwargs) -> npt.NDArray:
        """
        Apply Gaussian filter to an image.

        Parameters
        ----------
        image : npt.NDArray
            Numpy array representing the image.
        **kwargs
            Keyword arguments passed on to the skimage.filters.gaussian() function.

        Returns
        -------
        npt.NDArray
            Numpy array that represent the image after Gaussian filtering.
        """
        LOGGER.debug(
            f"[{self.filename}] : Applying Gaussian filter (mode : {self.gaussian_mode};"
            f" Gaussian blur (px) : {self.gaussian_size})."
        )
        return gaussian(
            image,
            sigma=(self.gaussian_size),
            mode=self.gaussian_mode,
            **kwargs,
        )

    def filter_image(self) -> None:  # numpydoc: ignore=GL07
        """
        Process a single image, filtering, finding grains and calculating their statistics.

        Returns
        -------
        None
            Does not return anything.

        Examples
        --------
        from topostats.io import LoadScan
        from topostats.filters import Filter
        from topostats.processing import process_scan

        filter = Filter(image=load_scan.image,
        ...             pixel_to_nm_scaling=load_scan.pixel_to_nm_scaling,
        ...             filename=load_scan.filename,
        ...             threshold_method='otsu')
        filter.filter_image()
        """
        self.images["initial_median_flatten"] = self.median_flatten(
            self.images["pixels"], mask=None, row_alignment_quantile=self.row_alignment_quantile
        )
        self.images["initial_tilt_removal"] = self.remove_tilt(self.images["initial_median_flatten"], mask=None)
        self.images["initial_quadratic_removal"] = self.remove_quadratic(self.images["initial_tilt_removal"], mask=None)
        self.images["initial_nonlinear_polynomial_removal"] = self.remove_nonlinear_polynomial(
            self.images["initial_quadratic_removal"], mask=None
        )

        # Remove scars
        run_scar_removal = self.remove_scars_config.pop("run")
        if run_scar_removal:
            LOGGER.debug(f"[{self.filename}] : Initial scar removal")
            self.images["initial_scar_removal"], _ = scars.remove_scars(
                self.images["initial_nonlinear_polynomial_removal"],
                filename=self.filename,
                **self.remove_scars_config,
            )
        else:
            LOGGER.debug(f"[{self.filename}] : Skipping scar removal as requested from config")
            self.images["initial_scar_removal"] = self.images["initial_nonlinear_polynomial_removal"]

        # Zero the data before thresholding, helps with absolute thresholding
        self.images["initial_zero_average_background"] = self.average_background(
            self.images["initial_scar_removal"], mask=None
        )

        # Get the thresholds
        try:
            self.thresholds = get_thresholds(
                image=self.images["initial_zero_average_background"],
                threshold_method=self.threshold_method,
                otsu_threshold_multiplier=self.otsu_threshold_multiplier,
                threshold_std_dev=self.threshold_std_dev,
                absolute=self.threshold_absolute,
            )
        except TypeError as type_error:
            raise type_error
        self.images["mask"] = get_mask(
            image=self.images["initial_zero_average_background"],
            thresholds=self.thresholds,
            img_name=self.filename,
        )
        self.images["masked_median_flatten"] = self.median_flatten(
            self.images["initial_tilt_removal"],
            self.images["mask"],
            row_alignment_quantile=self.row_alignment_quantile,
        )
        self.images["masked_tilt_removal"] = self.remove_tilt(self.images["masked_median_flatten"], self.images["mask"])
        self.images["masked_quadratic_removal"] = self.remove_quadratic(
            self.images["masked_tilt_removal"], self.images["mask"]
        )
        self.images["masked_nonlinear_polynomial_removal"] = self.remove_nonlinear_polynomial(
            self.images["masked_quadratic_removal"], self.images["mask"]
        )
        # Remove scars
        if run_scar_removal:
            LOGGER.debug(f"[{self.filename}] : Secondary scar removal")
            self.images["secondary_scar_removal"], scar_mask = scars.remove_scars(
                self.images["masked_nonlinear_polynomial_removal"],
                filename=self.filename,
                **self.remove_scars_config,
            )
            self.images["scar_mask"] = scar_mask
        else:
            LOGGER.debug(f"[{self.filename}] : Skipping scar removal as requested from config")
            self.images["secondary_scar_removal"] = self.images["masked_nonlinear_polynomial_removal"]
        self.images["final_zero_average_background"] = self.average_background(
            self.images["secondary_scar_removal"], self.images["mask"]
        )
        self.images["gaussian_filtered"] = self.gaussian_filter(self.images["final_zero_average_background"])
