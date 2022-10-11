"""Tests of the filters module."""
from pathlib import Path
import numpy as np
import pytest
from pySPM.SPM import SPM_image
from pySPM.Bruker import Bruker
from skimage.filters import gaussian

from topostats.filters import Filters
from topostats.utils import get_thresholds, get_mask

# pylint: disable=protected-access

# Specify the absolute and relattive tolerance for floating point comparison
TOLERANCE = {"atol": 1e-07, "rtol": 1e-07}

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"


def test_load_image(test_filters: Filters) -> None:
    """Test loading of image."""
    filters = Filters(RESOURCES / "minicircle.spm")
    filters.load_scan()
    assert isinstance(test_filters.images["scan_raw"], Bruker)


def test_extract_filename(test_filters: Filters) -> None:
    """Test extraction of filename."""
    test_filters.extract_filename()
    assert test_filters.filename == "minicircle"


def test_make_output_directory(test_filters: Filters, tmp_path: Path) -> None:
    """Test creation of output directory"""
    test_filters.make_output_directory()
    assert tmp_path.exists()


def test_load_image_that_does_not_exist(caplog) -> None:
    """Test logging and exceptions when file does not exist."""
    Filters("nothing")

    assert "File not found : nothing" in caplog.text


def test_extract_channel(test_filters: Filters) -> None:
    """Test extraction of channel."""
    test_filters.extract_channel()
    assert isinstance(test_filters.images["extracted_channel"], SPM_image)


def test_extract_channel_exception(test_filters_random: Filters) -> None:
    """Test extraction of channel exceptions when not an SPM image."""
    test_filters_random.extracted_channel = test_filters_random.pixels
    with pytest.raises(Exception):
        assert isinstance(test_filters_random.images["pixels"], SPM_image)


def test_extract_pixels(test_filters: Filters) -> None:
    """Test extraction of channel."""
    test_filters.extract_channel()
    test_filters.extract_pixels()
    assert isinstance(test_filters.images["pixels"], np.ndarray)
    assert test_filters.images["pixels"].shape == (1024, 1024)


@pytest.mark.parametrize(
    "unit, x, y, expected",
    [
        ("um", 100, 100, 97.65625),
        ("nm", 50, 50, 0.048828125),
    ],
)
def test_extract_pixel_to_nm_scaling(test_filters_random: Filters, unit, x, y, expected) -> None:
    """Test extraction of pixels to nanometer scaling."""
    test_filters_random.images["extracted_channel"].size["real"] = {"unit": unit, "x": x, "y": y}
    test_filters_random.extract_pixel_to_nm_scaling()
    assert test_filters_random.pixel_to_nm_scaling == expected


def test_row_col_medians_no_mask(
    test_filters_random: Filters, image_random_row_medians: np.array, image_random_col_medians: np.array
) -> None:
    """Test calculation of row and column medians without masking."""
    medians = test_filters_random.row_col_medians(test_filters_random.pixels, mask=None)

    assert isinstance(medians, dict)
    assert list(medians.keys()) == ["rows", "cols"]
    assert isinstance(medians["rows"], np.ndarray)
    assert isinstance(medians["cols"], np.ndarray)

    np.testing.assert_array_equal(medians["rows"], image_random_row_medians)
    np.testing.assert_array_equal(medians["cols"], image_random_col_medians)


def test_align_rows_no_mask(test_filters_random: Filters, image_random_aligned_rows: np.array) -> None:
    """Test aligning of rows by median height."""
    aligned_rows = test_filters_random.align_rows(test_filters_random.pixels, mask=None)

    assert isinstance(aligned_rows, np.ndarray)
    assert aligned_rows.shape == (1024, 1024)
    np.testing.assert_allclose(aligned_rows, image_random_aligned_rows, **TOLERANCE)


def test_remove_tilt_no_mask(test_filters_random: Filters, image_random_remove_x_y_tilt: np.array) -> None:
    """Test removal of x/y tilt."""
    tilt_removed = test_filters_random.remove_tilt(test_filters_random.pixels, mask=None)

    assert isinstance(tilt_removed, np.ndarray)
    assert tilt_removed.shape == (1024, 1024)
    np.testing.assert_allclose(tilt_removed, image_random_remove_x_y_tilt, **TOLERANCE)


def test_median_row_height(test_filters_random: Filters):
    """Test calculation of median row height."""
    medians = test_filters_random.row_col_medians(test_filters_random.pixels, mask=None)
    row_medians = medians["rows"]
    median_row_height = test_filters_random._median_row_height(row_medians)
    target = np.nanmedian(medians["rows"])

    assert median_row_height == target


def test_row_median_diffs(test_filters_random: Filters):
    """Test calculation of median row differences."""
    medians = test_filters_random.row_col_medians(test_filters_random.pixels, mask=None)
    row_medians = medians["rows"]
    median_row_height = test_filters_random._median_row_height(row_medians)
    row_median_diffs = test_filters_random._row_median_diffs(row_medians, median_row_height)
    target = medians["rows"] - np.nanmedian(medians["rows"])

    np.testing.assert_equal(row_median_diffs, target)


def test_calc_diff(test_filters_random: Filters, image_random: np.ndarray) -> None:
    """Test calculation of difference in array."""
    target = image_random[-1] - image_random[0]
    calculated = test_filters_random.calc_diff(test_filters_random.pixels)

    np.testing.assert_array_equal(target, calculated)


def test_calc_gradient(test_filters_random: Filters, image_random: np.ndarray) -> None:
    """Test calculation of gradient."""
    target = (image_random[-1] - image_random[0]) / image_random.shape[0]
    calculated = test_filters_random.calc_gradient(test_filters_random.pixels, test_filters_random.pixels.shape[0])

    np.testing.assert_array_equal(target, calculated)


def test_row_col_medians_with_mask(
    test_filters_random_with_mask: Filters,
    image_random_row_medians_masked: np.array,
    image_random_col_medians_masked: np.array,
) -> None:
    """Test calculation of row and column medians without masking."""
    medians = test_filters_random_with_mask.row_col_medians(
        test_filters_random_with_mask.images["pixels"], mask=test_filters_random_with_mask.images["mask"]
    )

    assert isinstance(medians, dict)
    assert list(medians.keys()) == ["rows", "cols"]
    assert isinstance(medians["rows"], np.ndarray)
    assert isinstance(medians["cols"], np.ndarray)
    np.testing.assert_array_equal(medians["rows"], image_random_row_medians_masked)
    np.testing.assert_array_equal(medians["cols"], image_random_col_medians_masked)


def test_non_square_img(test_filters_random: Filters):
    test_filters_random.images["pixels"] = test_filters_random.images["pixels"][:, 0:512]
    test_filters_random.images["zero_averaged_background"] = test_filters_random.average_background(
        image=test_filters_random.images["pixels"], mask=None
    )
    assert isinstance(test_filters_random.images["zero_averaged_background"], np.ndarray)
    assert test_filters_random.images["zero_averaged_background"].shape == (1024, 512)
    assert test_filters_random.images["zero_averaged_background"].sum() == 44426.48188033322


def test_gaussian_filter(small_array_filters: Filters, filter_config: dict) -> None:
    """Test Gaussian filter."""
    small_array_filters.images["gaussian_filtered"] = small_array_filters.gaussian_filter(
        image=small_array_filters.images["zero_averaged_background"]
    )
    target = gaussian(
        small_array_filters.images["zero_averaged_background"],
        sigma=(filter_config["gaussian_size"] / 0.5),
        mode=filter_config["gaussian_mode"],
    )
    assert isinstance(small_array_filters.images["gaussian_filtered"], np.ndarray)
    np.testing.assert_array_equal(small_array_filters.images["gaussian_filtered"], target)
