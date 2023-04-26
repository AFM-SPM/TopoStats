"""Module for flattening perovskite images"""
import logging
import perovskite_plotting as perov_plot
import numpy as np

LOGGER = logging.getLogger()


def plane_tilt_removal(image: np.ndarray):
    """Fit a plane to the data and subtract it."""

    read_matrix = image.copy()
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
            LOGGER.info("Removing x plane tilt")
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


def remove_polynomial(image: np.ndarray, mask: np.ndarray = None, order: int = 2):
    """Fit a polynomial to the data in the x-direction and remove it. Repeat this process
    on the transpose of the array to remove y-oriented trends."""

    image = image.copy()
    if mask is not None:
        read_matrix = np.ma.masked_array(image, mask=mask, fill_value=np.nan).filled()
    else:
        read_matrix = image

    # Calculate the median over the horizontal direction.
    # (axis=0 since that's the dimension in which the median is taken for each
    # element of the median array)
    col_medians = np.nanmedian(read_matrix, axis=0)

    # Fit the row median data to a polynomial
    coeffs = np.polyfit(range(0, read_matrix.shape[1]), col_medians, order)
    LOGGER.info(f"x polyfit nth order: {coeffs}")
    row_fit = np.polyval(coeffs, range(0, read_matrix.shape[1]))

    for row in range(0, read_matrix.shape[0]):
        image[row, :] -= row_fit

    return image


def zero_average(heightmap: np.ndarray) -> np.ndarray:
    """Set the median value of the image to zero by subtracting the median
    from the image."""
    median = np.nanmedian(heightmap)
    return np.subtract(heightmap, median)


def flatten_image(image: np.ndarray, order=2, plot_steps: bool = False):
    """Flatten an image by removing polynomial trends in x and y, then zero the median
    image value. Optionally plot the intermediary images with residuals."""

    # Flatten image
    if plot_steps:
        perov_plot.plot_with_means(image, title="raw")
    image = remove_polynomial(image, order=order)
    if plot_steps:
        perov_plot.plot_with_means(image, title="x quadratic removed")
    image = remove_polynomial(image.T, order=order).T
    if plot_steps:
        perov_plot.plot_with_means(image, title="y quadratic removed")
    image = zero_average(image)
    if plot_steps:
        perov_plot.plot_with_means(image, title="flattened")

    return image
