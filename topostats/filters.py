"""Contains filter functions that take a 2D array representing an image as an input, as well as necessary parameters,
and return a 2D array of the same size representing the filtered image."""
import logging

from skimage.filters import gaussian
import numpy as np

from topostats.logs.logs import LOGGER_NAME
from topostats.utils import get_thresholds, get_mask
from topostats.scars import Scars

LOGGER = logging.getLogger(LOGGER_NAME)

# pylint: disable=fixme
# pylint: disable=broad-except


class Filters:
    """Class for filtering scans."""

    def __init__(
        self,
        image: np.ndarray,
        filename: str,
        pixel_to_nm_scaling: float,
        threshold_method: str = "otsu",
        otsu_threshold_multiplier: float = 1.7,
        threshold_std_dev: dict = None,
        threshold_absolute: dict = None,
        gaussian_size: float = None,
        gaussian_mode: str = "nearest",
        quiet: bool = False,
    ):
        """Initialise the class.

        Parameters
        ----------
        image: np.ndarray
            The raw image from the AFM.
        filename: str
            The filename (used for logging outputs only).
        pixel_to_nm_scaling: float
            Value for converting pixels to nanometers.
        threshold_method: str
            Method for thresholding, default 'otsu', valid options 'otsu', 'std_dev' and 'absolute'.
        otsu_threshold_multiplier: float
            Value for scaling the derived Otsu threshold (optional).
        threshold_std_dev: dict
            If using the 'std_dev' threshold method. Dictionary that contains upper and lower
            threshold values for the number of standard deviations from the mean to threshold.
        threshold_absolute: dict
            If using the 'absolute' threshold method. Dictionary that contains upper and lower
            absolute threshold values for flattening.
        quiet: bool
            Whether to silence output.
        """
        self.filename = filename
        self.pixel_to_nm_scaling = pixel_to_nm_scaling
        self.gaussian_size = gaussian_size
        self.gaussian_mode = gaussian_mode
        self.threshold_method = threshold_method
        self.otsu_threshold_multiplier = otsu_threshold_multiplier
        self.threshold_std_dev = threshold_std_dev
        self.threshold_absolute = threshold_absolute
        self.images = {
            "pixels": image,
            "initial_median_flatten": None,
            "initial_tilt_removal": None,
            "initial_quadratic_removal": None,
            "masked_median_flatten": None,
            "masked_tilt_removal": None,
            "masked_quadratic_removal": None,
            "mask": None,
            "zero_average_background": None,
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

        if quiet:
            LOGGER.setLevel("ERROR")

    def median_flatten(self, image: np.ndarray, mask: np.ndarray = None, img_name: str = None) -> np.ndarray:
        """
        Uses the method of median differences to flatten the rows of an image, aligning the rows and centering the
        median around zero. When used with a mask, this has the effect of centering the background data on zero.
        Note this function does not handle scars.

        Parameters
        ----------
        image: np.ndarray
            2-D image of the data to align the rows of.
        mask: np.ndarray
            Boolean array of points to mask out (ignore).
        img_name: str
            Name of the image (to be able to print information in the console).
        Returns
        -------
        np.ndarray
            Returns a copy of the input image with rows aligned
        """
        image = image.copy()
        if mask is not None:
            read_matrix = np.ma.masked_array(image, mask=mask, fill_value=np.nan).filled()
            LOGGER.info(f"[{img_name}] : Median flattening with mask")
        else:
            read_matrix = image
            LOGGER.info(f"[{img_name}] : Median flattening without mask")

        for row in range(image.shape[0]):
            # Get the median of the row
            m = np.nanmedian(read_matrix[row, :])
            if not np.isnan(m):
                image[row, :] -= m
            else:
                LOGGER.warning(
                    """f[{self.filename}] Large grain detected image can not be
processed, please refer to <url to page where we document common problems> for more information."""
                )

        return image

    def remove_tilt(self, image: np.ndarray, mask: np.ndarray = None, img_name: str = None):
        """
        Removes planar tilt from an image (linear in 2D space). It uses a linear fit of the medians
        of the rows and columns to determine the linear slants in x and y directions and then subtracts the fit from the columns.

        Parameters
        ----------
        image: np.ndarray
            2-D image of the data to remove the planar tilt from.
        mask: np.ndarray
            Boolean array of points to mask out (ignore).
        img_name: str
            Name of the image (to be able to print information in the console).
        Returns
        -------
        np.ndarray
            Returns a copy of the input image with the planar tilt removed
        """
        image = image.copy()
        if mask is not None:
            read_matrix = np.ma.masked_array(image, mask=mask, fill_value=np.nan).filled()
            LOGGER.info(f"[{img_name}] : Plane tilt removal with mask")
        else:
            read_matrix = image
            LOGGER.info(f"[{img_name}] : Plane tilt removal without mask")

        # Line of best fit
        # Calculate medians
        medians_x = [np.nanmedian(read_matrix[:, i]) for i in range(read_matrix.shape[1])]
        medians_y = [np.nanmedian(read_matrix[j, :]) for j in range(read_matrix.shape[0])]

        # Fit linear x
        px = np.polyfit(range(0, len(medians_x)), medians_x, 1)
        LOGGER.info(f"x-polyfit 1st order: {px}")
        py = np.polyfit(range(0, len(medians_y)), medians_y, 1)
        LOGGER.info(f"y-polyfit 1st order: {py}")

        if px[0] != 0:
            if not np.isnan(px[0]):
                LOGGER.info("removing x plane tilt")
                for row in range(0, image.shape[0]):
                    for col in range(0, image.shape[1]):
                        image[row, col] -= px[0] * (col)
            else:
                LOGGER.info("x gradient is nan, skipping plane tilt x removal")
        else:
            LOGGER.info("x gradient is zero, skipping plane tilt x removal")

        if py[0] != 0:
            if not np.isnan(py[0]):
                LOGGER.info("removing y plane tilt")
                for row in range(0, image.shape[0]):
                    for col in range(0, image.shape[1]):
                        image[row, col] -= py[0] * (row)
            else:
                LOGGER.info("y gradient is nan, skipping plane tilt y removal")
        else:
            LOGGER.info("y gradient is zero, skipping plane tilt y removal")

        return image

    def remove_quadratic(self, image: np.ndarray, mask: np.ndarray = None, img_name: str = None):
        """
        Removes the quadratic bowing that can be seen in some large-scale AFM images. It uses a simple quadratic fit
        on the medians of the columns of the image and then subtracts the calculated quadratic from the columns.

        Parameters
        ----------
        image: np.ndarray
            2-D image of the data to remove the quadratic from.
        mask: np.ndarray
            Boolean array of points to mask out (ignore).
        img_name: str
            Name of the image (to be able to print information in the console).
        Returns
        -------
        np.ndarray
            Returns a copy of the input image with the quadratic bowing removed
        """
        image = image.copy()
        if mask is not None:
            read_matrix = np.ma.masked_array(image, mask=mask, fill_value=np.nan).filled()
            LOGGER.info(f"[{img_name}] : Remove quadratic bow with mask")
        else:
            read_matrix = image
            LOGGER.info(f"[{img_name}] : Remove quadratic bow without mask")

        # Calculate medians
        medians_x = [np.nanmedian(read_matrix[:, i]) for i in range(read_matrix.shape[1])]

        # Fit quadratic x
        px = np.polyfit(range(0, len(medians_x)), medians_x, 2)
        LOGGER.info(f"x polyfit 2nd order: {px}")

        # Handle divide by zero
        if px[0] != 0:
            if not np.isnan(px[0]):
                # Remove quadratic in x
                cx = -px[1] / (2 * px[0])
                for row in range(0, image.shape[0]):
                    for col in range(0, image.shape[1]):
                        image[row, col] -= px[0] * (col - cx) ** 2
            else:
                LOGGER.info("quadratic polyfit returns nan, skipping quadratic removal")
        else:
            LOGGER.info("quadratic polyfit returns zero, skipping quadratic removal")

        return image

    @staticmethod
    def calc_diff(array: np.ndarray) -> np.ndarray:
        """Calculate the difference of an array."""
        return array[-1] - array[0]

    def calc_gradient(self, array: np.ndarray, shape: int) -> np.ndarray:
        """Calculate the gradient of an array."""
        return self.calc_diff(array) / shape

    def average_background(self, image: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """Zero the background by subtracting the non-masked mean from all pixels.

        Parameters
        ----------
        image: np.array
            Numpy array representing image.
        mask: np.array
            Mask of the array, should have the same dimensions as image.

        Returns
        -------
        np.ndarray
            Numpy array of image zero averaged.
        """
        if mask is None:
            mask = np.zeros_like(image)
        mean = np.mean(image[mask == 0])
        LOGGER.info(f"[{self.filename}] : Zero averaging background : {mean} nm")
        return image - mean

    def gaussian_filter(self, image: np.ndarray, **kwargs) -> np.array:
        """Apply Gaussian filter to an image.

        Parameters
        ----------
        image: np.array
            Numpy array representing image.

        Returns
        -------
        np.array
            Numpy array of gaussian blurred image.

        """
        LOGGER.info(
            f"[{self.filename}] : Applying Gaussian filter (mode : {self.gaussian_mode};"
            f" Gaussian blur (px) : {self.gaussian_size})."
        )
        return gaussian(
            image,
            sigma=(self.gaussian_size),
            mode=self.gaussian_mode,
            **kwargs,
        )

    def filter_image(self) -> None:
        """Process a single image, filtering, finding grains and calculating their statistics.

        Example
        -------
        from topostats.io import LoadScan
        from topostats.topotracing import Filter, process_scan

        filter = Filter(image=load_scan.image,
        ...             pixel_to_nm_scaling=load_scan.pixel_to_nm_scaling,
        ...             filename=load_scan.filename,
        ...             threshold_method='otsu')
        filter.filter_image()

        """
        self.images["initial_median_flatten"] = self.median_flatten(
            self.images["pixels"], mask=None, img_name=self.filename
        )
        self.images["initial_tilt_removal"] = self.remove_tilt(
            self.images["initial_median_flatten"], mask=None, img_name=self.filename
        )
        self.images["initial_quadratic_removal"] = self.remove_quadratic(
            self.images["initial_tilt_removal"], mask=None, img_name=self.filename
        )

        # FIXME: Link this up with the pipeline
        # Remove scars
        scars_removed = Scars(
            # Config here
        )
        self.images["scar_removal"] = scars_removed.remove_scars()

        # Get the thresholds
        try:
            self.thresholds = get_thresholds(
                image=self.images["initial_quadratic_removal"],
                threshold_method=self.threshold_method,
                otsu_threshold_multiplier=self.otsu_threshold_multiplier,
                threshold_std_dev=self.threshold_std_dev,
                absolute=self.threshold_absolute,
            )
        except TypeError as type_error:
            raise type_error
        self.images["mask"] = get_mask(
            image=self.images["initial_quadratic_removal"], thresholds=self.thresholds, img_name=self.filename
        )
        self.images["masked_median_flatten"] = self.median_flatten(
            self.images["initial_tilt_removal"], self.images["mask"], img_name=self.filename
        )
        self.images["masked_tilt_removal"] = self.remove_tilt(
            self.images["masked_median_flatten"], self.images["mask"], img_name=self.filename
        )
        self.images["masked_quadratic_removal"] = self.remove_quadratic(
            self.images["masked_tilt_removal"], self.images["mask"], img_name=self.filename
        )
        self.images["zero_average_background"] = self.average_background(
            self.images["masked_quadratic_removal"], self.images["mask"]
        )
        self.images["gaussian_filtered"] = self.gaussian_filter(self.images["zero_average_background"])
