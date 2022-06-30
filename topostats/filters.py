"""Contains filter functions that take a 2D array representing an image as an input, as well as necessary parameters,
and return a 2D array of the same size representing the filtered image."""
import logging
from pathlib import Path
from typing import Union
import sys

import numpy as np

from topostats.io import load_scan
from topostats.thresholds import threshold
from topostats.logs.logs import LOGGER_NAME
from topostats.utils import get_thresholds, get_mask

from topostats.plottingfuncs import plot_and_save

LOGGER = logging.getLogger(LOGGER_NAME)

# pylint: disable=fixme
# pylint: disable=broad-except


class Filters:
    """Class for filtering scans."""

    def __init__(
        self,
        img_path: Union[str, Path],
        threshold_method: str = "otsu",
        otsu_threshold_multiplier: float = 1.7,
        threshold_std_dev: float = None,
        threshold_absolute_lower: float = None,
        threshold_absolute_upper: float = None,
        channel: str = "Height",
        amplify_level: float = None,
        output_dir: Union[str, Path] = None,
        quiet: bool = False,
    ):
        """Initialise the class.

        Parameters
        ----------
        img_path: Union[str, Path]
            Path to a valid image to load.
        channel: str
            Channel to extract from the image.
        amplify_level : float
            Factor by which to amplify the image.
        threshold_method: str
            Method for thresholding, default 'otsu'.
        quiet: bool
            Whether to silence output.
        output_dir: Union[str, Path]
            Directory to save output to, if it does not exist it will be created.

        Notes
        -----

        A directory under the 'outdir' will be created using the filename.
        """
        self.img_path = Path(img_path)
        self.channel = channel
        self.amplify_level = amplify_level
        self.threshold_method = threshold_method
        self.otsu_threshold_multiplier = otsu_threshold_multiplier
        self.threshold_std_dev = threshold_std_dev
        self.threshold_absolute_lower = threshold_absolute_lower
        self.threshold_absolute_upper = threshold_absolute_upper
        self.filename = self.extract_filename()
        self.output_dir = Path(output_dir) if output_dir else Path("./output")
        self.images = {
            "scan_raw": None,
            "extracted_channel": None,
            "pixels": None,
            "initial_align": None,
            "initial_tilt_removal": None,
            "masked_align": None,
            "masked_tilt_removal": None,
            "zero_averaged_background": None,
            "mask": None,
        }
        self.threshold = None
        self.pixel_to_nm_scaling = None
        self.medians = {"rows": None, "cols": None}
        LOGGER.info(f"Filename : {self.filename}")
        self.results = {
            "diff": None,
            "amplify": self.amplify_level,
            "median_row_height": None,
            "x_gradient": None,
            "y_gradient": None,
            "threshold": None,
        }
        self.load_scan()

        if quiet:
            LOGGER.setLevel("ERROR")

    def extract_filename(self) -> str:
        """Extract the filename from the img_path"""
        LOGGER.info(f"Extracting filename from : {self.img_path}")
        return self.img_path.stem

    def load_scan(self) -> None:
        """Load the scan."""
        self.images["scan_raw"] = load_scan(self.img_path)
        LOGGER.info(f"[{self.filename}] : Loaded image from : {self.img_path}")
        # LOGGER.info(f'Loaded file : {self.img_path}')
        # FIXME : Get this working, needs to include additional except for image in incorrect format.
        # try:
        #     self.images['scan_raw'] = load_scan(self.img_path)
        # except FileNotFoundError as error:
        #     LOGGER.error(f'File not found : {self.img_path}')
        #     pass
        # sys.exit(1)

    def make_output_directory(self) -> None:
        """Create the output directory for saving files to."""
        self.output_dir = self.output_dir / self.filename
        self.output_dir.mkdir(exist_ok=True, parents=True)
        LOGGER.info(f"[{self.filename}] : Created directory : {self.output_dir}")

    def extract_channel(self):
        """Extract the channel"""
        try:
            print("channel: ", self.channel)
            self.images["extracted_channel"] = self.images["scan_raw"].get_channel(self.channel)
            LOGGER.info(f"[{self.filename}] : Extracted channel {self.channel}")
        except Exception as exception:
            LOGGER.error(f"[{self.filename}] : {exception}")

    def extract_pixel_to_nm_scaling(self) -> float:
        """Extract the pixel to nanometer scaling from the image metadata."""
        self.pixel_to_nm_scaling = self.images["extracted_channel"].get_extent()[1] / len(
            self.images["extracted_channel"].pixels
        )
        LOGGER.info(f"[{self.filename}] : Pixels to nm scaling : {self.pixel_to_nm_scaling}")

    def extract_pixels(self) -> None:
        """Flatten the scan to a Numpy Array."""
        self.images["pixels"] = np.flipud(np.array(self.images["extracted_channel"].pixels))
        LOGGER.info(f"[{self.filename}] : Pixels extracted")

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
            2-D mage to be flattened.
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

    def row_col_medians(self, image: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """Returns the height value medians for the rows and columns.

        Returns
        -------

        Returns two Numpy arrays, the first is the row height value medians, the second the column height value
        medians.
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

    def align_rows(self, image: np.ndarray, mask=None) -> np.ndarray:
        """Returns the input image with rows aligned by median height"""
        if mask is not None:
            if mask.all():
                sys.exit(
                    "Filtering mask takes up entire image - there will be no image left to process. Try adjusting the flattening thresholds."
                )
        medians = self.row_col_medians(image, mask)
        row_medians = medians["rows"]
        median_row_height = self._median_row_height(row_medians)
        LOGGER.info(f"[{self.filename}] : Median Row Height: {median_row_height}")

        # Calculate the differences between the row medians and the median row height
        row_median_diffs = self._row_median_diffs(row_medians, median_row_height)

        # Adjust the row medians accordingly
        # FIXME : I think this can be done using arrays directly, no need to loop.
        for i in range(image.shape[0]):
            # if np.isnan(row_median_diffs[i]):
            #     LOGGER.info(f"{i} Row_median is nan! : {row_median_diffs[i]}")
            image[i] -= row_median_diffs[i]
        LOGGER.info(f"[{self.filename}] : Rows aligned")
        return image

    @staticmethod
    def _median_row_height(array: np.ndarray) -> float:
        """Calculate median of row medians"""
        return np.nanmedian(array)

    @staticmethod
    def _row_median_diffs(row_medians: np.array, median_row_height: float) -> np.array:
        """Calculate difference of row medians from the median row height"""
        return row_medians - median_row_height

    def remove_tilt(self, image: np.ndarray, mask=None) -> np.ndarray:
        """Returns the input image after removing any linear plane slant"""
        medians = self.row_col_medians(image, mask)
        gradient = {}
        gradient["x"] = self.calc_gradient(array=medians["rows"], shape=medians["rows"].shape[0])
        gradient["y"] = self.calc_gradient(medians["cols"], medians["cols"].shape[0])
        LOGGER.info(f'[{self.filename}] X-gradient: {gradient["x"]}')
        LOGGER.info(f'[{self.filename}] Y-gradient: {gradient["y"]}')

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                image[i, j] -= gradient["x"] * i
                image[i, j] -= gradient["y"] * j
        LOGGER.info(f"[{self.filename}] : X/Y tilt removed")
        return image

    @staticmethod
    def calc_diff(array: np.ndarray) -> np.ndarray:
        """Calculate the difference of an array."""
        return array[-1] - array[0]

    @classmethod
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
        np.array
            Numpy array of image zero averaged.
        """
        medians = self.row_col_medians(image, mask)
        LOGGER.info(f"[{self.filename}] : Zero averaging background")
        return image - np.array(medians["rows"], ndmin=1).T

    def filter_image(self) -> None:
        """Process a single image, filtering, finding grains and calculating their statistics.

        Example
        -------

        from topostats.topotracing import Filter, process_scan

        filter = Filter(image_path='minicircle.spm',
                        channel='Height',
                        amplify_level=1.0,
                        threshold_method='otsu')
        filter.filter_image()
        """
        self.extract_filename()
        self.load_scan()
        self.make_output_directory()
        self.extract_channel()
        self.extract_pixels()
        plot_and_save(self.images["pixels"], self.output_dir, "raw_heightmap.png")
        self.extract_pixel_to_nm_scaling()
        if self.amplify_level != 1.0:
            self.amplify()
        self.images["initial_align"] = self.align_rows(self.images["pixels"], mask=None)
        plot_and_save(self.images["initial_align"], self.output_dir, "initial_align.png")
        self.images["initial_tilt_removal"] = self.remove_tilt(self.images["initial_align"], mask=None)
        plot_and_save(self.images["initial_tilt_removal"], self.output_dir, "initial_remove_tilt.png")
        # Get the thresholds
        self.thresholds = get_thresholds(
            image=self.images["initial_tilt_removal"],
            threshold_method=self.threshold_method,
            otsu_threshold_multiplier=self.otsu_threshold_multiplier,
            deviation_from_mean=self.threshold_std_dev,
            absolute=(self.threshold_absolute_lower, self.threshold_absolute_upper),
        )
        print(f"THRESHOLDS: {self.thresholds}")
        self.images["mask"] = get_mask(image=self.images["initial_tilt_removal"], thresholds=self.thresholds)
        plot_and_save(self.images["mask"], self.output_dir, "filtering_mask.png")
        self.images["mask"] = get_mask(image=self.images["initial_tilt_removal"], thresholds=thresholds)
        self.images["masked_align"] = self.align_rows(self.images["initial_tilt_removal"], self.images["mask"])
        plot_and_save(self.images["masked_align"], self.output_dir, "masked_align.png")
        self.images["masked_tilt_removal"] = self.remove_tilt(self.images["masked_align"], self.images["mask"])
        plot_and_save(self.images["masked_tilt_removal"], self.output_dir, "masked_tilt_removal.png")
        print(f' masked tilt removal: {self.images["masked_tilt_removal"]}')
        self.images["zero_averaged_background"] = self.average_background(
            self.images["masked_tilt_removal"], self.images["mask"]
        )
        plot_and_save(self.images["zero_averaged_background"], self.output_dir, "zero_averaged_background.png")
