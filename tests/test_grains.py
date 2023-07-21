"""Test finding of grains."""
import logging
import numpy as np
import pytest

# Pylint returns this error for from skimage.filters import gaussian
# pylint: disable=no-name-in-module
from topostats.grains import Grains

LOGGER = logging.getLogger(__name__)
LOGGER.propagate = True

# Specify the absolute and relattive tolerance for floating point comparison
TOLERANCE = {"atol": 1e-07, "rtol": 1e-07}


grain_array = np.array(
    [
        [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 1, 0, 0, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
        [0, 3, 3, 0, 0, 0, 0, 0, 2, 2],
        [3, 3, 3, 3, 3, 0, 0, 2, 2, 2],
    ]
)

grain_array2 = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 2, 2, 0, 0, 0, 0, 0, 1, 1],
        [2, 2, 2, 2, 2, 0, 0, 1, 1, 1],
    ]
)

grain_array3 = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
)

grain_array4 = np.array(
    [
        [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
)


@pytest.mark.parametrize(
    "area_thresh_nm, expected",
    [([None, None], grain_array), ([None, 32], grain_array2), ([12, 24], grain_array3), ([32, 44], grain_array4)],
)
def test_known_array_threshold(area_thresh_nm, expected) -> None:
    "Tests that arrays are thresholded on size as expected."
    grains = Grains(image=np.zeros((10, 6)), filename="xyz", pixel_to_nm_scaling=2)
    assert (grains.area_thresholding(grain_array, area_thresh_nm) == expected).all()


# def test_random_grains(random_grains: Grains, caplog) -> None:
#     """Test errors raised when processing images without grains."""
#     # FIXME : I can see for myself that the error message is logged but the assert fails as caplog.text is empty?
#     # assert "No gains found." in caplog.text
#     assert True


def test_remove_small_objects():
    """Test the remove_small_objects method of the Grains class."""

    grains_object = Grains(
        image=None,
        filename="",
        pixel_to_nm_scaling=1.0,
    )

    test_img = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 3, 3, 0],
            [0, 0, 1, 0, 3, 3, 0],
            [0, 0, 0, 0, 0, 3, 0],
            [0, 2, 0, 2, 0, 3, 0],
            [0, 2, 2, 2, 0, 3, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )

    expected = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )

    grains_object.minimum_grain_size = 5
    result = grains_object.remove_small_objects(test_img)

    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "test_labelled_image, area_thresholds, expected",
    [
        (
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 3, 3, 0],
                    [0, 0, 1, 0, 3, 3, 0],
                    [0, 0, 0, 0, 0, 3, 0],
                    [0, 2, 0, 2, 0, 3, 0],
                    [0, 2, 2, 2, 0, 3, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ]
            ),
            [4.0, 6.0],
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 1, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ]
            ),
        ),
        (
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 3, 3, 0],
                    [0, 0, 1, 0, 3, 3, 0],
                    [0, 0, 0, 0, 0, 3, 0],
                    [0, 2, 0, 2, 0, 3, 0],
                    [0, 2, 2, 2, 0, 3, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ]
            ),
            [None, None],
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 3, 3, 0],
                    [0, 0, 1, 0, 3, 3, 0],
                    [0, 0, 0, 0, 0, 3, 0],
                    [0, 2, 0, 2, 0, 3, 0],
                    [0, 2, 2, 2, 0, 3, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ]
            ),
        ),
    ],
)
def test_area_thresholding(test_labelled_image, area_thresholds, expected):
    """Test the area_thresholding() method of the Grains class."""

    grains_object = Grains(
        image=None,
        filename="",
        pixel_to_nm_scaling=1.0,
    )

    result = grains_object.area_thresholding(test_labelled_image, area_thresholds=area_thresholds)

    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "remove_edge_intersecting_grains, expected_number_of_grains",
    [
        (True, 6),
        (False, 9),
    ],
)
def test_remove_edge_intersecting_grains(
    grains_config: dict, remove_edge_intersecting_grains: bool, expected_number_of_grains: int
) -> None:
    """Test that Grains successfully does and doesn't remove edge intersecting grains"""

    # Ensure that a sensible number of grains are found
    grains_config["remove_edge_intersecting_grains"] = remove_edge_intersecting_grains
    grains_config["threshold_absolute"]["above"] = 1.0
    grains_config["threshold_method"] = "absolute"
    grains_config["smallest_grain_size_nm2"] = 20
    grains_config["absolute_area_threshold"]["above"] = [20, 10000000]

    grains = Grains(
        image=np.load("./tests/resources/minicircle_cropped_flattened.npy"),
        filename="minicircle_cropped_flattened",
        pixel_to_nm_scaling=0.4940029296875,
        **grains_config,
    )
    grains.find_grains()
    number_of_grains = len(grains.region_properties["above"])

    assert number_of_grains == expected_number_of_grains
