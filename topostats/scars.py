"""Image artefact correction functions that interpolates values filling the space of any detected scars."""

import logging

import numpy as np
import numpy.typing as npt

from topostats.logs.logs import LOGGER_NAME

LOGGER = logging.getLogger(LOGGER_NAME)

# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes


def _mark_if_positive_scar(
    row_col: tuple,
    stddev: float,
    img: npt.NDArray,
    marked: npt.NDArray,
    threshold_low: float,
    max_scar_width: int,
) -> None:
    """
    Mark scars as positive (i.e. a ridge rather than a dip).

    Determine if the points below and including the pixel at the specified row and column are a positive scar (a
    ridge rather than a dip). If they are, mark them in the marked 2d npt.NDArray. Note that this only detects positive
    scars.

    Parameters
    ----------
    row_col : tuple
        A tuple containing the row and column indices of the pixel for the top of the potential scar. Note that
        the first value is the row index, and the second is the column index.
    stddev : float
        The standard deviation, or the root-mean-square value for the image.
    img : npt.NDArray
        A 2-D image of the data to remove scars from.
    marked : np.ndarry
        A 2-D image of pixels that stores the positions of scars marked for removal. The value of the pixel is a
        floating point value that represents how strongly the algorithm considers it to be a scar.
        This may or may not already contain non-zero values given that previous iterations of scar removal
        may have been performed.
    threshold_low : float
        A value that when multiplied with the standard deviation, acts as a threshold to determine if an increase
        or decrease in height might constitute the top or bottom of a scar.
    max_scar_width : int
        A value that dictates the maximum width that a scar can be. Note that this does not mean horizontal width,
        rather vertical, this is because we consider scars to be laying flat, horizontally, so their width is
        vertical and their length is horizontal.
    """
    # Unpack row, col
    row = row_col[0]
    col = row_col[1]

    # If sharp enough rise
    min_scar_value = img[row + 1, col]
    max_border_value = img[row, col]
    if min_scar_value - max_border_value > threshold_low * stddev:
        # Possibly in a scar
        for k in range(1, max_scar_width + 1):
            if row + k + 1 >= img.shape[0]:
                # Bottom of image, break
                LOGGER.debug("Bottom of image.")
                break
            min_scar_value = min(min_scar_value, img[row + k, col])
            max_border_value = max(img[row, col], img[row + k + 1, col])
            # Check if scar ended
            if min_scar_value - max_border_value > threshold_low * stddev:
                while k:
                    val = (img[row + k, col] - max_border_value) / stddev
                    marked[row + k, col] = val
                    k -= 1


def _mark_if_negative_scar(
    row_col: tuple,
    stddev: float,
    img: npt.NDArray,
    marked: npt.NDArray,
    threshold_low: float,
    max_scar_width: int,
) -> None:
    """
    Mark scars as negative (i.e. a dip rather than a ridge).

    Determine if the points below and including the pixel at the specified row and column are a negative scar (a
    dip rather than a ridge). If they are, mark them in the marked 2d npt.NDArray. Note that this only detects negative
    scars.

    Parameters
    ----------
    row_col : tuple
        A tuple containing the row and column indices of the pixel for the top of the potential scar. Note that
        the first value is the row index, and the second is the column index.
    stddev : float
        The standard deviation, or the root-mean-square value for the image.
    img : npt.NDArray
        A 2-D image of the data to remove scars from.
    marked : np.ndarry
        A 2-D image of pixels that stores the positions of scars marked for removal. The value of the pixel is a
        floating point value that represents how strongly the algorithm considers it to be a scar.
        This may or may not already contain non-zero values given that previous iterations of scar removal
        may have been performed.
    threshold_low : float
        A value that when multiplied with the standard deviation, acts as a threshold to determine if an increase
        or decrease in height might constitute the top or bottom of a scar.
    max_scar_width : int
        A value that dictates the maximum width that a scar can be. Note that this does not mean horizontal width,
        rather vertical, this is because we consider scars to be laying flat, horizontally, so their width is
        vertical and their length is horizontal.
    """
    # Unpack row, col
    row = row_col[0]
    col = row_col[1]

    # If sharp enough dip
    min_scar_value = img[row + 1, col]
    max_border_value = img[row, col]
    if min_scar_value - max_border_value < -threshold_low * stddev:
        # Possibly in a scar
        for k in range(1, max_scar_width + 1):
            if row + k + 1 >= img.shape[0]:
                # Bottom of image, break
                LOGGER.debug("Bottom of image.")
                break
            min_scar_value = max(min_scar_value, img[row + k, col])
            max_border_value = min(img[row, col], img[row + k + 1, col])
            # Check if scar ended
            if min_scar_value - max_border_value < -threshold_low * stddev:
                while k:
                    val = (max_border_value - img[row + k, col]) / stddev
                    marked[row + k, col] = val
                    k -= 1


def _spread_scars(
    marked: npt.NDArray,
    threshold_low: float,
    threshold_high: float,
) -> None:
    """
    Spread high-marked pixels into adjacent low-marked pixels.

    This is a smudging function that attempts to catch any pixels that are parts of scars that might not have been
    extreme enough to get marked above the high_threshold. Any remaining marked pixels below high_threshold are
    considered not to be scars and are removed from the mask.

    Parameters
    ----------
    marked : npt.NDArray
        A 2-D image of pixels that stores the positions of scars marked for removal. The value of the pixel is a
        floating point value that represents how strongly the algorithm considers it to be a scar.
        This may or may not already contain non-zero values given that previous iterations of scar removal
        may have been performed.
    threshold_low : float
        A floating point value, that when multiplied by the standard deviation of the image, acts as a
        threshold for sharp inclines or descents in pixel values and thus marks potential scars.
        A lower value will make the algorithm
        more sensitive, and a higher value will make it less sensitive.
    threshold_high : float
        A floating point value that is used similarly to threshold_low, however sharp inclines or descents
        that result in values in the mask higher than this threshold are automatically considered scars.
    """
    # Spread scars that have close to threshold edge-points
    for row in range(marked.shape[0]):
        # Spread right
        for col in range(1, marked.shape[1]):
            if marked[row, col] >= threshold_low and marked[row, col - 1] >= threshold_high:
                marked[row, col] = threshold_high

        # Spread left
        for col in range(marked.shape[1] - 1, 0, -1):
            if marked[row, col - 1] >= threshold_low and marked[row, col] >= threshold_high:
                marked[row, col - 1] = threshold_high


def _remove_short_scars(marked: npt.NDArray, threshold_high: float, min_scar_length: int) -> None:
    """
    Remove scars that are too short (horizontally), based on the minimum length.

    Parameters
    ----------
    marked : npt.NDArray
        A 2-D image of pixels that stores the positions of scars marked for removal. The value of the pixel is a
        floating point value that represents how strongly the algorithm considers it to be a scar.
        This may or may not already contain non-zero values given that previous iterations of scar removal
        may have been performed.

    threshold_high : float
        A floating point value that is used similarly to threshold_low, however sharp inclines or descents
        that result in values in the mask higher than this threshold are automatically considered scars.

    min_scar_length : int
        A value that dictates the maximum width that a scar can be. Note that this does not mean horizontal width,
        rather vertical, this is because we consider scars to be laying flat, horizontally, so their width is
        vertical and their length is horizontal.
    """
    # Remove too-short scars
    for row in range(marked.shape[0]):
        k = 0
        for col in range(marked.shape[1]):
            # If greater than threshold, set value to true
            if marked[row, col] >= threshold_high:
                marked[row, col] = 1.0
                k += 1
                continue
            # If not greater than threshold,
            # either we reached the end of the scar,
            # or haven't found one.
            # Check if too short. If so, remove.
            if k and k < min_scar_length:
                while k:
                    marked[row, col - k] = 0.0
                    k -= 1
            # Long enough, just not bright anymore.
            # Reached the end of the scar that is long enough. Stop and reset.
            marked[row, col] = 0.0
            k = 0
        # Reached the end of the line, so check current scar's
        # length to see if too short and should be deleted.
        if k and k < min_scar_length:
            while k:
                marked[row, col - k] = 0.0
                k -= 1


def _mark_scars(
    img: npt.NDArray,
    direction: str,
    threshold_low: float,
    threshold_high: float,
    max_scar_width: int,
    min_scar_length: int,
) -> npt.NDArray:
    """
    Mark scars within an image, returning a boolean 2D npt.NDArray of pixels that have been detected as scars.

    Parameters
    ----------
    img : npt.NDArray
        A 2-D image of the data to detect scars in.
    direction : str
        Options : 'positive', 'negative'. The direction of scars to detect. For example, to detect scars
        that lie above the data, select 'positive'.
    threshold_low : float
        A floating point value, that when multiplied by the standard deviation of the image, acts as a
        threshold for sharp inclines or descents in pixel values and thus marks potential scars.
        A lower value will make the algorithm
        more sensitive, and a higher value will make it less sensitive.
    threshold_high : float
        A floating point value that is used similarly to threshold_low, however sharp inclines or descents
        that result in values in the mask higher than this threshold are automatically considered scars.
    max_scar_width : int
        An integer that restricts the algorithm to only mark scars that are as thin or thinner than this width.
        This is important for ensuring that legitimate grain data is not detected as scarred data.
        Note that width here is vertical, as scars are thin, horizontal features.
    min_scar_length : int
        An integer that restricts the algorithm to only mark scars that are as long or longer than this length.
        This is important for ensuring that noise or legitimate but sharp datapoints do not get detected as scars.
        Note that length here is horizontal, as scars are thin, horizontal features.

    Returns
    -------
    marked: npt.NDArray
        Returns a 2-D image of the same shape as the data image, where each pixel's value represents
        a metric for how strongly that pixel is considered to be a scar. Higher values mean more likely
        to be a scar.
    """
    image = np.copy(img)

    stddev = np.std(image)
    marked = np.zeros(image.shape)

    for row in range(image.shape[0] - 1):
        for col in range(image.shape[1]):
            if direction == "positive":
                _mark_if_positive_scar(
                    row_col=(row, col),
                    stddev=stddev,
                    img=image,
                    marked=marked,
                    threshold_low=threshold_low,
                    max_scar_width=max_scar_width,
                )

            elif direction == "negative":
                _mark_if_negative_scar(
                    row_col=(row, col),
                    stddev=stddev,
                    img=image,
                    marked=marked,
                    threshold_low=threshold_low,
                    max_scar_width=max_scar_width,
                )

            else:
                raise ValueError(f"direction {direction} invalid.")

    _spread_scars(marked=marked, threshold_low=threshold_low, threshold_high=threshold_high)

    _remove_short_scars(marked=marked, threshold_high=threshold_high, min_scar_length=min_scar_length)

    return marked


def _remove_marked_scars(img: npt.NDArray, scar_mask: npt.NDArray) -> None:
    """
    Interpolate values covered by marked scars.

    Takes an image, and a marked scar boolean mask for that image. Returns the image where the marked scars are replaced
    by interpolated values.

    Parameters
    ----------
    img : npt.NDArray
        A 2-D image of the data to remove scars from.
    scar_mask : npt.NDArray
        A boolean image of pixels that determine which values are flagged as scars and therefore should
        be interpolated over in the original data image.
    """
    for row, col in np.ndindex(img.shape):
        if scar_mask[row, col] == 1.0:
            # Determine how wide the scar is by incrementing scar_width until either the bottom
            # of the image is reached, or a non-marked pixel is encountered.
            scar_width = 1
            while row + scar_width < img.shape[0] and scar_mask[row + scar_width, col] == 1.0:
                scar_width += 1

            above = img[row - 1, col]
            below = img[row + scar_width, col]
            k = scar_width
            while k:
                # Linearly interpolate
                interp_val = (k / (scar_width + 1)) * below + (1 - (k / (scar_width + 1))) * above
                img[row + k - 1, col] = interp_val
                scar_mask[row + k - 1, col] = 0.0
                k -= 1
                LOGGER.debug("Scar removed")


def remove_scars(
    img: npt.NDArray,
    filename: str,
    removal_iterations: int = 2,
    threshold_low: float = 0.250,
    threshold_high: float = 0.666,
    max_scar_width: int = 4,
    min_scar_length: int = 16,
):
    """
    Remove scars from an image.

    Scars are long, typically 1-4 pixels wide streaks of high or low data in AFM images. They are a problem
    resulting from random errors in the AFM data collection process and are hard to avoid. This function
    detects and removes these artefacts by interpolating over them between the pixels above and below them.
    This method takes no parameters as it uses parameters already established as instance variables when the
    class was instantiated.

    Parameters
    ----------
    img : npt.NDArray
        A 2-D image to remove scars from.
    filename : str
        The filename (used for logging outputs only).
    removal_iterations : int
        The number of times the scar removal should run on the image.
        Running just once sometimes isn't enough to remove some of the
        more difficult to remove scars.
    threshold_low : float
        A value that when multiplied with the standard deviation, acts as a threshold to determine if an increase
        or decrease in height might constitute the top or bottom of a scar.
    threshold_high : float
        A floating point value that is used similarly to threshold_low, however sharp inclines or descents
        that result in values in the mask higher than this threshold are automatically considered scars.
    max_scar_width : int
        A value that dictates the maximum width that a scar can be. Note that this does not mean horizontal width,
        rather vertical, this is because we consider scars to be laying flat, horizontally, so their width is
        vertical and their length is horizontal.
    min_scar_length : int
        An integer that restricts the algorithm to only mark scars that are as long or longer than this length.
        This is important for ensuring that noise or legitimate but sharp datapoints do not get detected as scars.
        Note that length here is horizontal, as scars are thin, horizontal features.

    Returns
    -------
    self.img
        The original 2-D image with scars removed, unless the config has run set to False, in which case it
        will not remove the scars.
    """
    LOGGER.info(f"[{filename}] : Removing scars")

    first_marked_mask = None
    for i in range(removal_iterations):
        marked_positive = _mark_scars(
            img=img,
            direction="positive",
            threshold_low=threshold_low,
            threshold_high=threshold_high,
            max_scar_width=max_scar_width,
            min_scar_length=min_scar_length,
        )
        marked_negative = _mark_scars(
            img=img,
            direction="negative",
            threshold_low=threshold_low,
            threshold_high=threshold_high,
            max_scar_width=max_scar_width,
            min_scar_length=min_scar_length,
        )
        # Combine the upper and lower scar masks
        marked_both = np.bitwise_or(marked_positive.astype(bool), marked_negative.astype(bool))

        if i == 0:
            first_marked_mask = marked_both

        _remove_marked_scars(img, np.copy(marked_both))

        LOGGER.debug("Scars removed")

    return img, first_marked_mask
