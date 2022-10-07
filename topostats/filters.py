"""Contains filter functions that take a 2D array representing an image as an input, as well as necessary parameters,
and return a 2D array of the same size representing the filtered image."""
import logging
from pathlib import Path
from typing import Union

from skimage.filters import gaussian
import numpy as np

from topostats.logs.logs import LOGGER_NAME
from topostats.utils import get_thresholds, get_mask

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
        threshold_std_dev: float = None,
        threshold_absolute_lower: float = None,
        threshold_absolute_upper: float = None,
        amplify_level: float = None,
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
        amplify_level : float
            Factor by which to amplify the image.
        threshold_method: str
            Method for thresholding, default 'otsu'.
        quiet: bool
            Whether to silence output.
        """
        self.filename = filename
        self.pixel_to_nm_scaling = pixel_to_nm_scaling
        self.amplify_level = amplify_level
        self.gaussian_size = gaussian_size
        self.gaussian_mode = gaussian_mode
        self.threshold_method = threshold_method
        self.otsu_threshold_multiplier = otsu_threshold_multiplier
        self.threshold_std_dev = threshold_std_dev
        self.threshold_absolute_lower = threshold_absolute_lower
        self.threshold_absolute_upper = threshold_absolute_upper
        self.images = {
            "pixels": image,
            "initial_align": None,
            "initial_tilt_removal": None,
            "masked_align": None,
            "masked_tilt_removal": None,
            "zero_averaged_background": None,
            "mask": None,
            "gaussian_filtered": None,
        }
        self.thresholds = None
        self.medians = {"rows": None, "cols": None}
        self.results = {
            "diff": None,
            "amplify": self.amplify_level,
            "median_row_height": None,
            "x_gradient": None,
            "y_gradient": None,
            "threshold": None,
        }

        if quiet:
            LOGGER.setLevel("ERROR")

    def amplify(self) -> None:
        """The amplify filter mulitplies the value of all extracted pixels by the `level` argument."""
        self.images["pixels"] = self.images["pixels"] * self.amplify_level
        LOGGER.info(f"[{self.filename}] : Image amplified (x {self.amplify_level})")

    def flatten_image(self, image: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """Flatten an image.

        Flattening an image involves first aligining rows and then removing tilt, with a mask optionally applied to both
        stages. These methods could be called independently but this method is provided as a convenience.

        Parameters
        ----------
        image: np.ndarray
            2-D image to be flattened.
        mask: np.ndarray
            2-D mask to apply to image, should be the same dimensions as iamge.
        stage: str
            Indicator of the stage of flattneing.

        Returns
        -------
        np.ndarray
            2-D flattened image.all
        """
        image = self.align_rows(image, mask)
        image = self.remove_tilt(image, mask)
        return image

    def row_col_medians(self, image: np.ndarray, mask: np.ndarray = None) -> dict:
        """Returns the height value medians for the rows and columns.

        Parameters
        ----------
        image: np.ndarray
            2-D image to calculate row and column medians.
        mask: np.ndarray
            Boolean array of points to mask.
        Returns
        -------
        dict
            Dict of two Numpy arrays corresponding to row height value medians and column height value medians.
        """
        if mask is not None:
            image = np.ma.masked_array(image, mask=mask, fill_value=np.nan).filled()
            LOGGER.info(f"[{self.filename}] : Masking enabled")
        else:
            LOGGER.info(f"[{self.filename}] : Masking disabled")
        medians = {}
        medians["rows"] = np.nanmedian(image, axis=1)
        medians["cols"] = np.nanmedian(image, axis=0)
        LOGGER.info(f"[{self.filename}] : Row and column medians calculated.")
        return medians

    def align_rows(self, image: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """Returns a copy of the input image with rows aligned by median height.

        Parameters
        ----------
        image: np.ndarray
            2-D image to align rows.
        mask: np.ndarray
            Boolean array of points to mask.
        Returns
        -------
        np.ndarray
            Returns a copy of the input image with rows aligned by median height.
        """
        image_cp = image.copy()
        if mask is not None:
            if mask.all():
                LOGGER.error(f"[{self.filename}] : Mask covers entire image. Adjust filtering thresholds/method.")

        medians = self.row_col_medians(image_cp, mask)
        row_medians = medians["rows"]
        median_row_height = self._median_row_height(row_medians)
        LOGGER.info(f"[{self.filename}] : Median Row Height: {median_row_height}")

        # Calculate the differences between the row medians and the median row height
        row_median_diffs = self._row_median_diffs(row_medians, median_row_height)

        # Adjust the row medians accordingly
        # FIXME : I think this can be done using arrays directly, no need to loop.
        for i in range(image_cp.shape[0]):
            # if np.isnan(row_median_diffs[i]):
            #     LOGGER.info(f"{i} Row_median is nan! : {row_median_diffs[i]}")
            image_cp[i] -= row_median_diffs[i]
        LOGGER.info(f"[{self.filename}] : Rows aligned")
        return image_cp

    @staticmethod
    def _median_row_height(array: np.ndarray) -> float:
        """Calculate median of row medians"""
        return np.nanmedian(array)

    @staticmethod
    def _row_median_diffs(row_medians: np.array, median_row_height: float) -> np.array:
        """Calculate difference of row medians from the median row height"""
        return row_medians - median_row_height

    def remove_tilt(self, image: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """Returns a copy of the input image after removing any linear plane slant.

        Parameters
        ----------
        image: np.ndarray
            2-D image to align rows.
        mask: np.ndarray
            Boolean array of points to mask.
        Returns
        -------
        np.ndarray
            Returns a copy of the input image after removing any linear plane slant.
        """
        image_cp = image.copy()
        medians = self.row_col_medians(image_cp, mask)
        gradient = {}
        gradient["x"] = self.calc_gradient(array=medians["rows"], shape=medians["rows"].shape[0])
        gradient["y"] = self.calc_gradient(medians["cols"], medians["cols"].shape[0])
        LOGGER.info(f'[{self.filename}] : X-gradient: {gradient["x"]}')
        LOGGER.info(f'[{self.filename}] : Y-gradient: {gradient["y"]}')

        for i in range(image_cp.shape[0]):
            for j in range(image_cp.shape[1]):
                image_cp[i, j] -= gradient["x"] * i
                image_cp[i, j] -= gradient["y"] * j
        LOGGER.info(f"[{self.filename}] : X/Y tilt removed")
        return image_cp

    @staticmethod
    def calc_diff(array: np.ndarray) -> np.ndarray:
        """Calculate the difference of an array."""
        return array[-1] - array[0]

    def calc_gradient(self, array: np.ndarray, shape: int) -> np.ndarray:
        """Calculate the gradient of an array."""
        return self.calc_diff(array) / shape

    def average_background(self, image: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """Zero the background

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
        medians = self.row_col_medians(image, mask)
        LOGGER.info(f"[{self.filename}] : Zero averaging background")
        return (image.T - np.array(medians["rows"], ndmin=1)).T

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
            f"[{self.filename}] : Applying Gaussian filter (mode : {self.gaussian_mode}; Gaussian blur (nm) : {self.gaussian_size})."
        )
        return gaussian(
            image,
            sigma=(self.gaussian_size / self.pixel_to_nm_scaling),
            mode=self.gaussian_mode,
            **kwargs,
        )

    def filter_image(self) -> None:
        """Process a single image, filtering, finding grains and calculating their statistics.

        Example
        -------
        from topostats.topotracing import Filter, process_scan
        filter = Filter(image_path='minicircle.spm',
        ...             channel='Height',
        ...             amplify_level=1.0,
        ...             threshold_method='otsu')
        filter.filter_image()

        """

        if self.amplify_level != 1.0:
            self.amplify()
        self.images["initial_align"] = self.align_rows(self.images["pixels"], mask=None)
        self.images["initial_tilt_removal"] = self.remove_tilt(self.images["initial_align"], mask=None)
        # Get the thresholds
        try:
            self.thresholds = get_thresholds(
                image=self.images["initial_tilt_removal"],
                threshold_method=self.threshold_method,
                otsu_threshold_multiplier=self.otsu_threshold_multiplier,
                threshold_std_dev=self.threshold_std_dev,
                absolute=(self.threshold_absolute_lower, self.threshold_absolute_upper),
            )
        except TypeError as type_error:
            raise type_error
        self.images["mask"] = get_mask(
            image=self.images["initial_tilt_removal"], thresholds=self.thresholds, img_name=self.filename
        )
        self.images["masked_align"] = self.align_rows(self.images["initial_tilt_removal"], self.images["mask"])
        self.images["masked_tilt_removal"] = self.remove_tilt(self.images["masked_align"], self.images["mask"])
        self.images["zero_averaged_background"] = self.average_background(
            self.images["masked_tilt_removal"], self.images["mask"]
        )
        self.images["gaussian_filtered"] = self.gaussian_filter(self.images["zero_averaged_background"])
