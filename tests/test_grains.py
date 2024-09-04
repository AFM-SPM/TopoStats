"""Test finding of grains."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.typing as npt
import pytest

from topostats.grains import Grains

# Pylint returns this error for from skimage.filters import gaussian
# pylint: disable=no-name-in-module
# pylint: disable=too-many-arguments

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
    ("area_thresh_nm", "expected"),
    [([None, None], grain_array), ([None, 32], grain_array2), ([12, 24], grain_array3), ([32, 44], grain_array4)],
)
def test_known_array_threshold(area_thresh_nm, expected) -> None:
    """Tests that arrays are thresholded on size as expected."""
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
    ("test_labelled_image", "area_thresholds", "expected"),
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
    ("remove_edge_intersecting_grains", "expected_number_of_grains"),
    [
        (True, 6),
        (False, 9),
    ],
)
def test_remove_edge_intersecting_grains(
    grains_config: dict, remove_edge_intersecting_grains: bool, expected_number_of_grains: int
) -> None:
    """Test that Grains successfully does and doesn't remove edge intersecting grains."""
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


# Find grains without unet
@pytest.mark.parametrize(
    (
        "image",
        "pixel_to_nm_scaling",
        "threshold_method",
        "otsu_threshold_multiplier",
        "threshold_std_dev",
        "threshold_absolute",
        "absolute_area_threshold",
        "direction",
        "smallest_grain_size_nm2",
        "remove_edge_intersecting_grains",
        "expected_grain_mask",
    ),
    [
        pytest.param(
            np.array(
                [
                    [0.1, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2],
                    [0.2, 1.1, 1.0, 1.2, 0.2, 0.1, 1.5, 1.6, 1.7, 0.1],
                    [0.1, 1.1, 0.2, 1.0, 0.1, 0.2, 1.6, 0.2, 1.6, 0.2],
                    [0.2, 1.0, 1.1, 1.1, 0.2, 0.1, 1.6, 1.5, 1.5, 0.1],
                    [0.1, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2],
                    [1.5, 1.5, 0.2, 1.5, 1.5, 0.1, 2.0, 1.9, 1.8, 0.1],
                    [0.1, 0.1, 0.2, 0.0, 0.0, 0.2, 0.1, 0.2, 1.7, 0.2],
                    [0.2, 1.5, 1.5, 0.1, 0.2, 0.1, 0.2, 0.1, 1.6, 0.1],
                    [0.1, 0.1, 1.5, 0.1, 1.5, 0.2, 1.3, 1.4, 1.5, 0.2],
                    [0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1],
                ]
            ),
            1.0,
            "absolute",
            None,
            None,
            {"above": 0.9, "below": 0.0},
            {"above": [1, 10000000], "below": [1, 10000000]},
            "above",
            1,
            True,
            # Move axis required to force a (10, 10, 2) shape
            np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
                            [1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
                            [1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 0, 0, 1, 0, 0, 0, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                            [1, 0, 0, 1, 1, 1, 1, 1, 0, 1],
                            [1, 1, 0, 1, 0, 1, 0, 0, 0, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 0, 0, 2, 2, 2, 0],
                            [0, 1, 0, 1, 0, 0, 2, 0, 2, 0],
                            [0, 1, 1, 1, 0, 0, 2, 2, 2, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 3, 3, 0, 4, 4, 4, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 4, 0],
                            [0, 5, 5, 0, 0, 0, 0, 0, 4, 0],
                            [0, 0, 5, 0, 6, 0, 4, 4, 4, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ],
                    ),
                ],
                axis=-1,
            ).astype(np.int32),
            id="absolute, above 0.9, remove edge, smallest grain 1",
        ),
        pytest.param(
            np.array(
                [
                    [0.1, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2],
                    [0.2, 1.1, 1.0, 1.2, 0.2, 0.1, 1.5, 1.6, 1.7, 0.1],
                    [0.1, 1.1, 0.2, 1.0, 0.1, 0.2, 1.6, 0.2, 1.6, 0.2],
                    [0.2, 1.0, 1.1, 1.1, 0.2, 0.1, 1.6, 1.5, 1.5, 0.1],
                    [0.1, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2],
                    [1.5, 1.5, 0.2, 1.5, 1.5, 0.1, 2.0, 1.9, 1.8, 0.1],
                    [0.1, 0.1, 0.2, 0.0, 0.0, 0.2, 0.1, 0.2, 1.7, 0.2],
                    [0.2, 1.5, 1.5, 0.1, 0.2, 0.1, 0.2, 0.1, 1.6, 0.1],
                    [0.1, 0.1, 1.5, 0.1, 1.5, 0.2, 1.3, 1.4, 1.5, 0.2],
                    [0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1],
                ]
            ),
            1.0,
            "absolute",
            None,
            None,
            {"above": 0.9, "below": 0.0},
            {"above": [1, 10000000], "below": [1, 10000000]},
            "above",
            2,
            False,
            np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
                            [1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
                            [1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                            [1, 0, 0, 1, 1, 1, 1, 1, 0, 1],
                            [1, 1, 0, 1, 1, 1, 0, 0, 0, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 0, 0, 2, 2, 2, 0],
                            [0, 1, 0, 1, 0, 0, 2, 0, 2, 0],
                            [0, 1, 1, 1, 0, 0, 2, 2, 2, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [3, 3, 0, 4, 4, 0, 5, 5, 5, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 5, 0],
                            [0, 6, 6, 0, 0, 0, 0, 0, 5, 0],
                            [0, 0, 6, 0, 0, 0, 5, 5, 5, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ],
                    ),
                ],
                axis=-1,
            ).astype(np.int32),
            id="absolute, above 0.9, no remove edge, smallest grain 2",
        ),
    ],
)
def test_find_grains(
    image: npt.NDArray[np.float32],
    pixel_to_nm_scaling: float,
    threshold_method: str,
    otsu_threshold_multiplier: float,
    threshold_std_dev: dict,
    threshold_absolute: dict,
    absolute_area_threshold: dict,
    direction: str,
    smallest_grain_size_nm2: int,
    remove_edge_intersecting_grains: bool,
    expected_grain_mask: npt.NDArray[np.int32],
) -> None:
    """Test the find_grains method of the Grains class."""
    # Initialise the grains object
    grains_object = Grains(
        image=image,
        filename="test_image",
        pixel_to_nm_scaling=pixel_to_nm_scaling,
        unet_config=None,
        threshold_method=threshold_method,
        otsu_threshold_multiplier=otsu_threshold_multiplier,
        threshold_std_dev=threshold_std_dev,
        threshold_absolute=threshold_absolute,
        absolute_area_threshold=absolute_area_threshold,
        direction=direction,
        smallest_grain_size_nm2=smallest_grain_size_nm2,
        remove_edge_intersecting_grains=remove_edge_intersecting_grains,
    )

    grains_object.find_grains()

    result_removed_small_objects = grains_object.directions[direction]["removed_small_objects"]

    assert result_removed_small_objects.shape == expected_grain_mask.shape
    assert result_removed_small_objects.dtype == expected_grain_mask.dtype
    np.testing.assert_array_equal(result_removed_small_objects, expected_grain_mask)


# Find grains with unet - needs mocking
@pytest.mark.parametrize(
    ("image", "expected_removed_small_objects_tensor", "expected_labelled_regions_tensor"),
    [
        pytest.param(
            # Image
            np.array(
                [
                    [0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.2, 0.2, 0.1],
                    [0.1, 1.1, 1.2, 1.0, 0.1, 1.1, 0.2, 1.1, 0.2],
                    [0.2, 1.2, 1.1, 1.3, 0.2, 1.2, 0.1, 0.2, 0.2],
                    [0.1, 1.0, 1.2, 1.2, 0.1, 1.1, 1.2, 1.1, 0.1],
                    [0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1, 0.2, 0.1],
                    [0.1, 0.2, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.2],
                    [0.1, 0.2, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.2],
                    [0.1, 0.2, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1],
                    [0.1, 0.2, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1],
                ]
            ),
            # Expected removed small objects tensor
            np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 0, 0, 0, 1, 0, 1, 0, 1],
                            [1, 0, 0, 0, 1, 0, 1, 1, 1],
                            [1, 0, 0, 0, 1, 0, 0, 0, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 0, 1, 0, 1, 0],
                            [0, 1, 0, 1, 0, 1, 0, 0, 0],
                            [0, 1, 1, 1, 0, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ),
            # Expected labelled regions tensor
            np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 0, 0, 0, 1, 0, 1, 0, 1],
                            [1, 0, 0, 0, 1, 0, 1, 1, 1],
                            [1, 0, 0, 0, 1, 0, 0, 0, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 0, 2, 0, 3, 0],
                            [0, 1, 0, 1, 0, 2, 0, 0, 0],
                            [0, 1, 1, 1, 0, 2, 2, 2, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ),
            id="unet, 5x5, multi class, 3 grains",
        ),
        pytest.param(
            # Image
            np.array(
                [
                    [0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.2, 0.2, 0.1],
                    [0.1, 0.1, 0.2, 0.0, 0.1, 0.1, 0.2, 0.1, 0.2],
                    [0.2, 0.2, 0.1, 0.3, 0.2, 0.2, 0.1, 0.2, 0.2],
                    [0.1, 0.0, 0.2, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1],
                    [0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1, 0.2, 0.1],
                    [0.1, 0.2, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.2],
                    [0.1, 0.2, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.2],
                    [0.1, 0.2, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1],
                    [0.1, 0.2, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1],
                ]
            ),
            # Expected removed small objects tensor
            np.stack(
                [
                    np.ones((9, 9)),
                    np.zeros((9, 9)),
                ],
                axis=-1,
            ),
            # Expected labelled regions tensor
            np.stack(
                [
                    np.zeros((9, 9)),
                    np.zeros((9, 9)),
                ],
                axis=-1,
            ),
            id="unet, 5x5, no grains",
        ),
    ],
)
def test_find_grains_unet(
    mock_model_5_by_5_single_class: MagicMock,
    image: npt.NDArray[np.float32],
    expected_removed_small_objects_tensor: npt.NDArray[np.bool_],
    expected_labelled_regions_tensor: npt.NDArray[np.int32],
) -> None:
    """Test the find_grains method of the Grains class with a unet model."""
    with patch("keras.models.load_model") as mock_load_model:
        mock_load_model.return_value = mock_model_5_by_5_single_class

        # Initialise the grains object
        grains_object = Grains(
            image=image,
            filename="test_image",
            pixel_to_nm_scaling=1.0,
            unet_config={
                "model_path": "dummy_model_path",
                "confidence": 0.5,
                "model_input_shape": (None, 5, 5, 1),
                "upper_norm_bound": 1.0,
                "lower_norm_bound": 0.0,
                "grain_crop_padding": 1,
            },
            threshold_method="absolute",
            threshold_absolute={"above": 0.9, "below": 0.0},
            absolute_area_threshold={"above": [1, 10000000], "below": [1, 10000000]},
            direction="above",
            smallest_grain_size_nm2=1,
            remove_edge_intersecting_grains=True,
        )

        grains_object.find_grains()

        result_removed_small_objects = grains_object.directions["above"]["removed_small_objects"]
        result_labelled_regions = grains_object.directions["above"]["labelled_regions_02"]

        assert expected_removed_small_objects_tensor.shape == (9, 9, 2)
        assert expected_labelled_regions_tensor.shape == (9, 9, 2)

        assert result_removed_small_objects.shape == expected_removed_small_objects_tensor.shape
        assert result_labelled_regions.shape == expected_labelled_regions_tensor.shape

        np.testing.assert_array_equal(result_removed_small_objects, expected_removed_small_objects_tensor)
        np.testing.assert_array_equal(result_labelled_regions, expected_labelled_regions_tensor)


@pytest.mark.parametrize(
    (
        "image",
        "unet_config",
        "traditional_threshold_labelled_regions",
        "expected_boolean_mask_tensor",
        "expected_labelled_regions_tensor",
    ),
    [
        pytest.param(
            np.array(
                [
                    [0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.2, 0.2, 0.1],
                    [0.1, 1.1, 1.2, 1.0, 0.1, 1.1, 0.2, 1.1, 0.2],
                    [0.2, 1.2, 1.1, 1.3, 0.2, 1.2, 0.1, 0.2, 0.2],
                    [0.1, 1.0, 1.2, 1.2, 0.1, 1.1, 1.2, 1.1, 0.1],
                    [0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1, 0.2, 0.1],
                    [0.1, 0.2, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.2],
                    [0.1, 0.2, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.2],
                    [0.1, 0.2, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1],
                    [0.1, 0.2, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1],
                ]
            ),
            {
                "model_path": "dummy_model_path",
                "confidence": 0.5,
                "model_input_shape": (None, 5, 5, 1),
                "upper_norm_bound": 1.0,
                "lower_norm_bound": 0.0,
                "grain_crop_padding": 1,
            },
            # This has the centre pixel filled in, representing a feature that is impossible to segment
            # with just thresholding. The U-Net is simulated to be able to recognise that there should be a
            # hole in the grain and thus improves the mask.
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0, 2, 0, 3, 0],
                    [0, 1, 1, 1, 0, 2, 0, 0, 0],
                    [0, 1, 1, 1, 0, 2, 2, 2, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
            np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 0, 0, 0, 1, 0, 1, 0, 1],
                            [1, 0, 0, 0, 1, 0, 1, 1, 1],
                            [1, 0, 0, 0, 1, 0, 0, 0, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 0, 1, 0, 1, 0],
                            [0, 1, 0, 1, 0, 1, 0, 0, 0],
                            [0, 1, 1, 1, 0, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ).astype(np.bool_),
            np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 0, 0, 0, 1, 0, 1, 0, 1],
                            [1, 0, 0, 0, 1, 0, 1, 1, 1],
                            [1, 0, 0, 0, 1, 0, 0, 0, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 0, 2, 0, 3, 0],
                            [0, 1, 0, 1, 0, 2, 0, 0, 0],
                            [0, 1, 1, 1, 0, 2, 2, 2, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ).astype(np.int32),
        )
    ],
)
def test_improve_grain_segmentation_unet(
    mock_model_5_by_5_single_class: MagicMock,
    image: npt.NDArray[np.float32],
    unet_config: dict[str, str | int | float | tuple[int | None, int, int, int]],
    traditional_threshold_labelled_regions: npt.NDArray[np.int32],
    expected_boolean_mask_tensor: npt.NDArray[np.bool_],
    expected_labelled_regions_tensor: npt.NDArray[np.int32],
) -> None:
    """Test the improve_grain_segmentation method of the Grains class with a unet model."""
    with patch("keras.models.load_model") as mock_load_model:
        mock_load_model.return_value = mock_model_5_by_5_single_class

        result_boolean_masks_tensor, result_labelled_regions_tensor = Grains.improve_grain_segmentation_unet(
            filename="test_image",
            direction="above",
            unet_config=unet_config,
            image=image,
            labelled_grain_regions=traditional_threshold_labelled_regions,
        )

        assert result_boolean_masks_tensor.shape == expected_boolean_mask_tensor.shape
        assert result_labelled_regions_tensor.shape == expected_labelled_regions_tensor.shape
        np.testing.assert_array_equal(result_boolean_masks_tensor, expected_boolean_mask_tensor)
        np.testing.assert_array_equal(result_labelled_regions_tensor, expected_labelled_regions_tensor)


@pytest.mark.parametrize(
    ("labelled_image", "expected_labelled_image"),
    [
        pytest.param(
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
            id="empty array",
        ),
        pytest.param(
            np.array(
                [
                    [0, 1, 1, 1, 0, 2, 2, 0],
                    [0, 1, 1, 1, 0, 2, 2, 0],
                    [0, 0, 0, 0, 0, 2, 2, 0],
                    [0, 3, 3, 3, 0, 2, 2, 0],
                    [0, 3, 3, 3, 0, 0, 0, 0],
                    [0, 3, 3, 0, 0, 4, 4, 0],
                    [0, 0, 3, 0, 0, 4, 4, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ).astype(np.bool_),
            id="simple",
        ),
    ],
)
def test_keep_largest_labelled_region(
    labelled_image: npt.NDArray[np.int32], expected_labelled_image: npt.NDArray[np.int32]
) -> None:
    """Test the keep_largest_labelled_region method of the Grains class."""
    result = Grains.keep_largest_labelled_region(labelled_image)

    np.testing.assert_array_equal(result, expected_labelled_image)
