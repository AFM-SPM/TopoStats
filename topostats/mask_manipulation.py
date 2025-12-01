"""Code for manipulating binary masks."""

import logging

import numpy as np
import numpy.typing as npt
from scipy import ndimage
from skimage import filters
from skimage.morphology import label

from topostats.logs.logs import LOGGER_NAME

LOGGER = logging.getLogger(LOGGER_NAME)


def re_add_holes(
    pixel_to_nm_scaling: float,
    orig_mask: npt.NDArray,
    smoothed_mask: npt.NDArray,
    holearea_min_max: tuple[float | int | None, float | int | None] = (2, None),
) -> npt.NDArray:
    """
    Restore holes in masks that were occluded by dilation.

    As Gaussian dilation smoothing methods can close holes in the original mask, this function obtains those holes
    (based on the general background being the first due to padding) and adds them back into the smoothed mask. When
    paired with ``smooth_mask``, this essentially just smooths the outer edge of the mask.

    Parameters
    ----------
    pixel_to_nm_scaling : float
        Pixel to nanometer scaling of the image.
    orig_mask : npt.NDArray
        Original mask.
    smoothed_mask : npt.NDArray
        Original mask but with inner and outer edged smoothed. The smoothing operation may have closed up important
        holes in the mask.
    holearea_min_max : tuple[float | int | None, float | int | None]
        Tuple of minimum and maximum hole area (in nanometers) to replace from the original mask into the smoothed
        mask.

    Returns
    -------
    npt.NDArray
        Smoothed mask with holes restored.
    """
    # handle Nones
    # If both none, do nothing
    if holearea_min_max[0] is None and holearea_min_max[1] is None:
        return smoothed_mask
    # If min is none, set to 0.0 (use float to avoid inferring int type)
    if holearea_min_max[0] is None:
        hole_area_min: float = 0.0
    else:
        hole_area_min = float(holearea_min_max[0])
    # If max is none, set to inf
    if holearea_min_max[1] is None:
        hole_area_max: float = np.inf
    else:
        hole_area_max = float(holearea_min_max[1])

    # obtain px holesizes
    holesize_min_px = hole_area_min / ((pixel_to_nm_scaling) ** 2)
    holesize_max_px = hole_area_max / ((pixel_to_nm_scaling) ** 2)

    # obtain a hole mask
    holes = 1 - orig_mask
    holes = label(holes)
    hole_sizes = [holes[holes == i].size for i in range(1, holes.max() + 1)]
    holes[holes == 1] = 0  # set background to 0 assuming it is the first hole seen (from top left)

    # remove too small or too big holes from mask
    for i, hole_size in enumerate(hole_sizes):
        if hole_size < holesize_min_px or hole_size > holesize_max_px:  # small holes may be fake are left out
            holes[holes == i + 1] = 0
    holes[holes != 0] = 1  # set correct sixe holes to 1

    # replace correct sized holes
    return np.where(holes == 1, 0, smoothed_mask)


def smooth_mask(
    filename: str,
    pixel_to_nm_scaling: float,
    grain: npt.NDArray,
    dilation_iterations: int = 2,
    gaussian_sigma: float | int = 2,
    holearea_min_max: tuple[int | float | None, int | float | None] = (0, None),
) -> npt.NDArray:
    """
    Smooth a grain mask based on the lower number of binary pixels added from dilation or gaussian.

    This method ensures gaussian smoothing isn't too aggressive and covers / creates gaps in the mask.

    Parameters
    ----------
    filename : str
        Filename of the image being processed (for logging purposes).
    pixel_to_nm_scaling : float
        Pixel to nanometer scaling of the image.
    grain : npt.NDArray
        Numpy array of the grain mask.
    dilation_iterations : int
        Number of times to dilate the grain to smooth it. Default is 2.
    gaussian_sigma : float | None
        Gaussian sigma value to smooth the grains after an Otsu threshold. If None, defaults to 2.
    holearea_min_max : tuple[float | int | None]
        Tuple of minimum and maximum hole area (in nanometers) to replace from the original mask into the smoothed
        mask.

    Returns
    -------
    npt.NDArray
        Numpy array of smoothed image.
    """
    # Option to disable the smoothing (i.e. U-Net masks are already smooth)
    if dilation_iterations is None and gaussian_sigma is None:
        LOGGER.debug(f"[{filename}] : no grain smoothing done")
        return grain

    # Option to only do gaussian or dilation
    if dilation_iterations is not None:
        dilation = ndimage.binary_dilation(grain, iterations=dilation_iterations).astype(np.int32)
    else:
        gauss = filters.gaussian(grain, sigma=gaussian_sigma)
        gauss = np.where(gauss > filters.threshold_otsu(gauss) * 1.3, 1, 0)
        gauss = gauss.astype(np.int32)
        LOGGER.debug(f"[{filename}] : smoothing done by gaussian {gaussian_sigma}")
        return re_add_holes(grain, gauss, holearea_min_max)
    if gaussian_sigma is not None:
        gauss = filters.gaussian(grain, sigma=gaussian_sigma)
        gauss = np.where(gauss > filters.threshold_otsu(gauss) * 1.3, 1, 0)
        gauss = gauss.astype(np.int32)
    else:
        LOGGER.debug(f"[{filename}] : smoothing done by dilation {dilation_iterations}")
        return re_add_holes(grain, dilation, holearea_min_max)

    # Competition option between dilation and gaussian mask differences wrt original grains
    if abs(dilation.sum() - grain.sum()) > abs(gauss.sum() - grain.sum()):
        LOGGER.debug(f"[{filename}] : smoothing done by gaussian {gaussian_sigma}")
        return re_add_holes(
            pixel_to_nm_scaling=pixel_to_nm_scaling,
            orig_mask=grain,
            smoothed_mask=gauss,
            holearea_min_max=holearea_min_max,
        )
    LOGGER.debug(f"[{filename}] : smoothing done by dilation {dilation_iterations}")
    return re_add_holes(
        pixel_to_nm_scaling=pixel_to_nm_scaling,
        orig_mask=grain,
        smoothed_mask=dilation,
        holearea_min_max=holearea_min_max,
    )
