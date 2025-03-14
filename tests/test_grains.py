"""Test finding of grains."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.typing as npt
import pytest

from topostats.grains import GrainCrop, GrainCropsDirection, Grains, ImageGrainCrops, validate_full_mask_tensor_shape
from topostats.io import dict_almost_equal

# Pylint returns this error for from skimage.filters import gaussian
# pylint: disable=no-name-in-module
# pylint: disable=too-many-arguments
# pylint: disable=too-many-lines
# pylint: disable=too-many-locals
# pylint: disable=too-many-positional-arguments

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


def test_grain_crop_to_dict(dummy_graincrop: GrainCrop):
    """Test the GrainCrop.grain_crop_to_dict() method."""
    expected = {
        "image": dummy_graincrop.image,
        "mask": dummy_graincrop.mask,
        "padding": dummy_graincrop.padding,
        "bbox": dummy_graincrop.bbox,
        "pixel_to_nm_scaling": dummy_graincrop.pixel_to_nm_scaling,
        "filename": dummy_graincrop.filename,
    }
    np.testing.assert_array_equal(dummy_graincrop.grain_crop_to_dict(), expected)


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
    ("binary_image", "minimum_size_px", "minimum_bbox_size_px", "expected_image"),
    [
        pytest.param(
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
            8,
            4,
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
        )
    ],
)
def test_remove_objects_too_small_to_process(
    binary_image: npt.NDArray, minimum_size_px: int, minimum_bbox_size_px: int, expected_image: npt.NDArray
) -> None:
    """Test the remove_objects_too_small_to_process method of the Grains class."""
    grains_object = Grains(
        image=np.array([[0, 0], [0, 0]]),
        filename="",
        pixel_to_nm_scaling=1.0,
    )

    result = grains_object.remove_objects_too_small_to_process(
        image=binary_image, minimum_size_px=minimum_size_px, minimum_bbox_size_px=minimum_bbox_size_px
    )

    np.testing.assert_array_equal(result, expected_image)


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
        "expected_labelled_regions",
        "expected_imagegraincrops",
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
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0, 1, 1, 1, 0],
                    [0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
                    [0, 1, 1, 1, 0, 0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 1, 1, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 6, 0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
            ).astype(bool),
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
            ).astype(np.int32),
            ImageGrainCrops(
                above=GrainCropsDirection(
                    crops={
                        0: GrainCrop(
                            bbox=(0, 0, 5, 5),
                            filename="test_image",
                            padding=1,
                            pixel_to_nm_scaling=1.0,
                            image=np.array(
                                [
                                    [0.1, 0.1, 0.2, 0.1, 0.1],
                                    [0.2, 1.1, 1.0, 1.2, 0.2],
                                    [0.1, 1.1, 0.2, 1.0, 0.1],
                                    [0.2, 1.0, 1.1, 1.1, 0.2],
                                    [0.1, 0.1, 0.2, 0.1, 0.1],
                                ]
                            ),
                            mask=np.stack(
                                [
                                    np.array(
                                        [
                                            [1, 1, 1, 1, 1],
                                            [1, 0, 0, 0, 1],
                                            [1, 0, 1, 0, 1],
                                            [1, 0, 0, 0, 1],
                                            [1, 1, 1, 1, 1],
                                        ]
                                    ),
                                    np.array(
                                        [
                                            [0, 0, 0, 0, 0],
                                            [0, 1, 1, 1, 0],
                                            [0, 1, 0, 1, 0],
                                            [0, 1, 1, 1, 0],
                                            [0, 0, 0, 0, 0],
                                        ]
                                    ),
                                ],
                                axis=-1,
                            ),
                        ),
                        1: GrainCrop(
                            bbox=(0, 5, 5, 10),
                            filename="test_image",
                            padding=1,
                            pixel_to_nm_scaling=1.0,
                            image=np.array(
                                [
                                    [0.2, 0.1, 0.2, 0.1, 0.2],
                                    [0.1, 1.5, 1.6, 1.7, 0.1],
                                    [0.2, 1.6, 0.2, 1.6, 0.2],
                                    [0.1, 1.6, 1.5, 1.5, 0.1],
                                    [0.2, 0.1, 0.2, 0.1, 0.2],
                                ]
                            ),
                            mask=np.stack(
                                [
                                    np.array(
                                        [
                                            [1, 1, 1, 1, 1],
                                            [1, 0, 0, 0, 1],
                                            [1, 0, 1, 0, 1],
                                            [1, 0, 0, 0, 1],
                                            [1, 1, 1, 1, 1],
                                        ]
                                    ),
                                    np.array(
                                        [
                                            [0, 0, 0, 0, 0],
                                            [0, 1, 1, 1, 0],
                                            [0, 1, 0, 1, 0],
                                            [0, 1, 1, 1, 0],
                                            [0, 0, 0, 0, 0],
                                        ]
                                    ),
                                ],
                                axis=-1,
                            ),
                        ),
                        2: GrainCrop(
                            bbox=(4, 2, 8, 6),
                            filename="test_image",
                            padding=1,
                            pixel_to_nm_scaling=1.0,
                            image=np.array(
                                [
                                    [0.2, 0.1, 0.1, 0.2],
                                    [0.2, 1.5, 1.5, 0.1],
                                    [0.2, 0.0, 0.0, 0.2],
                                    [1.5, 0.1, 0.2, 0.1],
                                ]
                            ),
                            mask=np.stack(
                                [
                                    np.array(
                                        [
                                            [1, 1, 1, 1],
                                            [1, 0, 0, 1],
                                            [1, 1, 1, 1],
                                            [1, 1, 1, 1],
                                        ]
                                    ),
                                    np.array(
                                        [
                                            [0, 0, 0, 0],
                                            [0, 1, 1, 0],
                                            [0, 0, 0, 0],
                                            [0, 0, 0, 0],
                                        ]
                                    ),
                                ],
                                axis=-1,
                            ),
                        ),
                        3: GrainCrop(
                            bbox=(4, 4, 10, 10),
                            filename="test_image",
                            padding=1,
                            pixel_to_nm_scaling=1.0,
                            image=np.array(
                                [
                                    [0.1, 0.2, 0.1, 0.2, 0.1, 0.2],
                                    [1.5, 0.1, 2.0, 1.9, 1.8, 0.1],
                                    [0.0, 0.2, 0.1, 0.2, 1.7, 0.2],
                                    [0.2, 0.1, 0.2, 0.1, 1.6, 0.1],
                                    [1.5, 0.2, 1.3, 1.4, 1.5, 0.2],
                                    [0.2, 0.1, 0.2, 0.1, 0.2, 0.1],
                                ]
                            ),
                            mask=np.stack(
                                [
                                    np.array(
                                        [
                                            [1, 1, 1, 1, 1, 1],
                                            [1, 1, 0, 0, 0, 1],
                                            [1, 1, 1, 1, 0, 1],
                                            [1, 1, 1, 1, 0, 1],
                                            [1, 1, 0, 0, 0, 1],
                                            [1, 1, 1, 1, 1, 1],
                                        ]
                                    ),
                                    np.array(
                                        [
                                            [0, 0, 0, 0, 0, 0],
                                            [0, 0, 1, 1, 1, 0],
                                            [0, 0, 0, 0, 1, 0],
                                            [0, 0, 0, 0, 1, 0],
                                            [0, 0, 1, 1, 1, 0],
                                            [0, 0, 0, 0, 0, 0],
                                        ]
                                    ),
                                ],
                                axis=-1,
                            ),
                        ),
                        4: GrainCrop(
                            bbox=(6, 0, 10, 4),
                            filename="test_image",
                            padding=1,
                            pixel_to_nm_scaling=1.0,
                            image=np.array(
                                [
                                    [0.1, 0.1, 0.2, 0.0],
                                    [0.2, 1.5, 1.5, 0.1],
                                    [0.1, 0.1, 1.5, 0.1],
                                    [0.2, 0.1, 0.2, 0.1],
                                ]
                            ),
                            mask=np.stack(
                                [
                                    np.array(
                                        [
                                            [1, 1, 1, 1],
                                            [1, 0, 0, 1],
                                            [1, 1, 0, 1],
                                            [1, 1, 1, 1],
                                        ]
                                    ),
                                    np.array(
                                        [
                                            [0, 0, 0, 0],
                                            [0, 1, 1, 0],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 0],
                                        ]
                                    ),
                                ],
                                axis=-1,
                            ),
                        ),
                        5: GrainCrop(
                            bbox=(7, 3, 10, 6),
                            filename="test_image",
                            padding=1,
                            pixel_to_nm_scaling=1.0,
                            image=np.array([[0.1, 0.2, 0.1], [0.1, 1.5, 0.2], [0.1, 0.2, 0.1]]),
                            mask=np.stack(
                                [
                                    np.array(
                                        [
                                            [1, 1, 1],
                                            [1, 0, 1],
                                            [1, 1, 1],
                                        ]
                                    ),
                                    np.array(
                                        [
                                            [0, 0, 0],
                                            [0, 1, 0],
                                            [0, 0, 0],
                                        ]
                                    ),
                                ],
                                axis=-1,
                            ),
                        ),
                    },
                    full_mask_tensor=np.stack(
                        [
                            np.array(
                                [
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
                                    [1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
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
                                    [0, 1, 1, 1, 0, 0, 1, 1, 1, 0],
                                    [0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
                                    [0, 1, 1, 1, 0, 0, 1, 1, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 1, 1, 0, 1, 1, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                                    [0, 1, 1, 0, 0, 0, 0, 0, 1, 0],
                                    [0, 0, 1, 0, 1, 0, 1, 1, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                ],
                            ),
                        ],
                        axis=-1,
                    ).astype(np.bool_),
                ),
                below=None,
            ),
            id="absolute, above 0.9, remove edge, smallest grain 1",
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
    expected_labelled_regions: npt.NDArray[np.int32],
    expected_imagegraincrops: ImageGrainCrops,
) -> None:
    """Test the find_grains method of the Grains class without unet."""
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

    # Override grains' minimum grain size just for this test to allow for small grains in the test image
    grains_object.minimum_grain_size_px = 1
    grains_object.minimum_bbox_size_px = 1

    grains_object.find_grains()

    result_removed_small_objects = grains_object.directions[direction]["removed_objects_too_small_to_process"]
    result_labelled_regions = grains_object.directions[direction]["labelled_regions_02"]

    assert result_removed_small_objects.shape == expected_grain_mask.shape
    assert result_removed_small_objects.dtype == expected_grain_mask.dtype
    np.testing.assert_array_equal(result_removed_small_objects, expected_grain_mask)

    assert result_labelled_regions.shape == expected_labelled_regions.shape
    assert result_labelled_regions.dtype == expected_labelled_regions.dtype
    np.testing.assert_array_equal(result_labelled_regions, expected_labelled_regions)

    # Check the image_grain_crops
    result_image_grain_crops = grains_object.image_grain_crops
    result_above = result_image_grain_crops.above
    expected_above = expected_imagegraincrops.above
    for _index, (expected_graincrop, result_graincrop) in enumerate(
        zip(expected_above.crops.values(), result_above.crops.values())
    ):
        result_graincrop_mask = result_graincrop.mask
        expected_graincrop_mask = expected_graincrop.mask
        assert np.array_equal(result_graincrop_mask, expected_graincrop_mask)
        assert result_graincrop == expected_graincrop
    result_full_mask_tensor = result_above.full_mask_tensor
    expected_full_mask_tensor = expected_above.full_mask_tensor
    assert np.array_equal(result_full_mask_tensor, expected_full_mask_tensor)
    assert result_above == expected_above
    assert result_image_grain_crops == expected_imagegraincrops


# Find grains with unet - needs mocking
@pytest.mark.parametrize(
    ("image", "expected_grain_mask", "expected_labelled_regions", "expected_imagegraincrops"),
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
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0, 1, 0, 1, 0],
                    [0, 1, 1, 1, 0, 1, 0, 0, 0],
                    [0, 1, 1, 1, 0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ).astype(bool),
            # Expected labelled regions tensor
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
            # Expected image grain crops
            ImageGrainCrops(
                above=GrainCropsDirection(
                    full_mask_tensor=np.stack(
                        [
                            np.array(
                                [
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 0, 0, 0, 1, 0, 1, 0, 1],
                                    [1, 0, 1, 0, 1, 0, 1, 1, 1],
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
                    ).astype(bool),
                    crops={
                        0: GrainCrop(
                            bbox=(0, 0, 5, 5),
                            filename="test_image",
                            padding=1,
                            pixel_to_nm_scaling=1.0,
                            image=np.array(
                                [
                                    [0.1, 0.2, 0.1, 0.2, 0.1],
                                    [0.1, 1.1, 1.2, 1.0, 0.1],
                                    [0.2, 1.2, 1.1, 1.3, 0.2],
                                    [0.1, 1.0, 1.2, 1.2, 0.1],
                                    [0.1, 0.1, 0.2, 0.2, 0.1],
                                ]
                            ),
                            mask=np.stack(
                                [
                                    np.array(
                                        [
                                            [1, 1, 1, 1, 1],
                                            [1, 0, 0, 0, 1],
                                            [1, 0, 1, 0, 1],
                                            [1, 0, 0, 0, 1],
                                            [1, 1, 1, 1, 1],
                                        ]
                                    ),
                                    np.array(
                                        [
                                            [0, 0, 0, 0, 0],
                                            [0, 1, 1, 1, 0],
                                            [0, 1, 0, 1, 0],
                                            [0, 1, 1, 1, 0],
                                            [0, 0, 0, 0, 0],
                                        ]
                                    ),
                                ],
                                axis=-1,
                            ),
                        ),
                        1: GrainCrop(
                            bbox=(0, 4, 5, 9),
                            filename="test_image",
                            padding=1,
                            pixel_to_nm_scaling=1.0,
                            image=np.array(
                                [
                                    [0.1, 0.2, 0.2, 0.2, 0.1],
                                    [0.1, 1.1, 0.2, 1.1, 0.2],
                                    [0.2, 1.2, 0.1, 0.2, 0.2],
                                    [0.1, 1.1, 1.2, 1.1, 0.1],
                                    [0.1, 0.1, 0.1, 0.2, 0.1],
                                ]
                            ),
                            mask=np.stack(
                                [
                                    np.array(
                                        [
                                            [1, 1, 1, 1, 1],
                                            [1, 0, 1, 0, 1],
                                            [1, 0, 1, 1, 1],
                                            [1, 0, 0, 0, 1],
                                            [1, 1, 1, 1, 1],
                                        ]
                                    ),
                                    np.array(
                                        [
                                            [0, 0, 0, 0, 0],
                                            [0, 1, 0, 1, 0],
                                            [0, 1, 0, 0, 0],
                                            [0, 1, 1, 1, 0],
                                            [0, 0, 0, 0, 0],
                                        ]
                                    ),
                                ],
                                axis=-1,
                            ),
                        ),
                        2: GrainCrop(
                            bbox=(0, 6, 3, 9),
                            filename="test_image",
                            padding=1,
                            pixel_to_nm_scaling=1.0,
                            image=np.array(
                                [
                                    [0.2, 0.2, 0.1],
                                    [0.2, 1.1, 0.2],
                                    [0.1, 0.2, 0.2],
                                ]
                            ),
                            mask=np.stack(
                                [
                                    np.array(
                                        [
                                            [1, 1, 1],
                                            [1, 0, 1],
                                            [1, 1, 1],
                                        ]
                                    ),
                                    np.array(
                                        [
                                            [0, 0, 0],
                                            [0, 1, 0],
                                            [0, 0, 0],
                                        ]
                                    ),
                                ],
                                axis=-1,
                            ),
                        ),
                    },
                ),
                below=None,
            ),
            id="unet, 5x5, multi class, 3 grains",
        ),
    ],
)
def test_find_grains_unet(
    mock_model_5_by_5_single_class: MagicMock,
    image: npt.NDArray[np.float32],
    expected_grain_mask: npt.NDArray[np.bool_],
    expected_labelled_regions: npt.NDArray[np.int32],
    expected_imagegraincrops: ImageGrainCrops,
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
                "remove_disconnected_grains": False,
            },
            threshold_method="absolute",
            threshold_absolute={"above": 0.9, "below": 0.0},
            absolute_area_threshold={"above": [1, 10000000], "below": [1, 10000000]},
            direction="above",
            smallest_grain_size_nm2=1,
            remove_edge_intersecting_grains=True,
        )

        # Override grains' minimum grain size just for this test to allow for small grains in the test image
        grains_object.minimum_grain_size_px = 1
        grains_object.minimum_bbox_size_px = 1

        grains_object.find_grains()

        result_grain_mask = grains_object.directions["above"]["removed_objects_too_small_to_process"]
        result_labelled_regions = grains_object.directions["above"]["labelled_regions_02"]
        result_image_grain_crops = grains_object.image_grain_crops

        assert result_grain_mask.shape == expected_grain_mask.shape
        assert result_labelled_regions.shape == expected_labelled_regions.shape

        np.testing.assert_array_equal(result_grain_mask, expected_grain_mask)
        np.testing.assert_array_equal(result_labelled_regions, expected_labelled_regions)

        result_image_grain_crops.debug_locate_difference(expected_imagegraincrops)

        assert result_image_grain_crops == expected_imagegraincrops


def test_find_grains_no_grains_found():
    """Test the find_grains method of the Grains class when no grains are found."""
    # Image
    image = np.array(
        [
            [0.1, 0.1, 0.2, 0.1, 0.1],
            [0.2, 0.1, 0.1, 0.1, 0.2],
            [0.1, 0.1, 0.1, 0.1, 0.1],
            [0.2, 0.1, 0.1, 0.1, 0.2],
            [0.1, 0.1, 0.2, 0.1, 0.1],
        ]
    )

    # Expected removed small objects tensor
    expected_grain_mask = np.zeros_like(image, dtype=bool)

    # Expected labelled regions tensor
    expected_labelled_regions = np.zeros_like(image, dtype=int)

    # Expected image grain crops
    expected_imagegraincrops = ImageGrainCrops(
        above=None,
        below=None,
    )

    # Initialise the grains object
    grains_object = Grains(
        image=image,
        filename="test_image",
        pixel_to_nm_scaling=1.0,
        unet_config=None,
        threshold_method="absolute",
        threshold_absolute={"above": 0.9, "below": 0.0},
        absolute_area_threshold={"above": [1, 10000000], "below": [1, 10000000]},
        direction="above",
        smallest_grain_size_nm2=1,
        remove_edge_intersecting_grains=True,
    )

    # Override grains' minimum grain size just for this test to allow for small grains in the test image
    grains_object.minimum_grain_size_px = 1
    grains_object.minimum_bbox_size_px = 1

    grains_object.find_grains()

    result_grain_mask = grains_object.directions["above"]["removed_objects_too_small_to_process"]
    result_labelled_regions = grains_object.directions["above"]["labelled_regions_02"]
    result_image_grain_crops = grains_object.image_grain_crops

    assert result_grain_mask.shape == expected_grain_mask.shape
    assert result_labelled_regions.shape == expected_labelled_regions.shape

    np.testing.assert_array_equal(result_grain_mask, expected_grain_mask)
    np.testing.assert_array_equal(result_labelled_regions, expected_labelled_regions)

    assert result_image_grain_crops == expected_imagegraincrops


@pytest.mark.parametrize(
    (
        "unet_config",
        "graincrops",
        "expected_graincrops",
    ),
    [
        pytest.param(
            #     # Unet config
            {
                "model_path": "dummy_model_path",
                "confidence": 0.5,
                "model_input_shape": (None, 5, 5, 1),
                "upper_norm_bound": 1.0,
                "lower_norm_bound": 0.0,
                "remove_disconnected_grains": False,
            },
            # Traditionally generated graincrops
            {
                0: GrainCrop(
                    image=np.array(
                        [
                            [0.1, 0.2, 0.1, 0.2, 0.1],
                            [0.1, 1.1, 1.2, 1.0, 0.1],
                            [0.2, 1.2, 1.1, 1.3, 0.2],
                            [0.1, 1.0, 1.2, 1.2, 0.1],
                            [0.1, 0.1, 0.2, 0.2, 0.1],
                        ],
                    ),
                    mask=np.stack(
                        [
                            np.array(
                                [
                                    [1, 1, 1, 1, 1],
                                    [1, 0, 0, 0, 1],
                                    [1, 0, 0, 0, 1],
                                    [1, 0, 0, 0, 1],
                                    [1, 1, 1, 1, 1],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0, 0, 0],
                                    [0, 1, 1, 1, 0],
                                    [0, 1, 1, 1, 0],
                                    [0, 1, 1, 1, 0],
                                    [0, 0, 0, 0, 0],
                                ]
                            ),
                        ],
                        axis=-1,
                    ),
                    bbox=(0, 0, 5, 5),
                    padding=1,
                    pixel_to_nm_scaling=1.0,
                    filename="test_image",
                ),
                1: GrainCrop(
                    image=np.array(
                        [
                            [0.1, 0.2, 0.2, 0.2, 0.1],
                            [0.1, 1.1, 0.2, 1.1, 0.2],
                            [0.2, 1.2, 0.1, 0.2, 0.2],
                            [0.1, 1.1, 1.2, 1.1, 0.1],
                            [0.1, 0.1, 0.1, 0.2, 0.1],
                        ],
                    ),
                    mask=np.stack(
                        [
                            np.array(
                                [
                                    [1, 1, 1, 1, 1],
                                    [1, 0, 1, 0, 1],
                                    [1, 0, 1, 1, 1],
                                    [1, 0, 0, 0, 1],
                                    [1, 1, 1, 1, 1],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0, 0, 0],
                                    [0, 1, 0, 1, 0],
                                    [0, 1, 0, 0, 0],
                                    [0, 1, 1, 1, 0],
                                    [0, 0, 0, 0, 0],
                                ]
                            ),
                        ],
                        axis=-1,
                    ),
                    bbox=(0, 4, 5, 9),
                    padding=1,
                    pixel_to_nm_scaling=1.0,
                    filename="test_image",
                ),
                2: GrainCrop(
                    image=np.array(
                        [
                            [0.2, 0.2, 0.1],
                            [0.2, 1.1, 0.2],
                            [0.1, 0.2, 0.2],
                        ]
                    ),
                    mask=np.stack(
                        [
                            np.array(
                                [
                                    [1, 1, 1],
                                    [1, 0, 1],
                                    [1, 1, 1],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 0],
                                ]
                            ),
                        ],
                        axis=-1,
                    ),
                    bbox=(0, 6, 3, 9),
                    padding=1,
                    pixel_to_nm_scaling=1.0,
                    filename="test_image",
                ),
            },
            {
                0: GrainCrop(
                    image=np.array(
                        [
                            [0.1, 0.2, 0.1, 0.2, 0.1],
                            [0.1, 1.1, 1.2, 1.0, 0.1],
                            [0.2, 1.2, 1.1, 1.3, 0.2],
                            [0.1, 1.0, 1.2, 1.2, 0.1],
                            [0.1, 0.1, 0.2, 0.2, 0.1],
                        ],
                    ),
                    mask=np.stack(
                        [
                            np.array(
                                [
                                    [1, 1, 1, 1, 1],
                                    [1, 0, 0, 0, 1],
                                    [1, 0, 1, 0, 1],
                                    [1, 0, 0, 0, 1],
                                    [1, 1, 1, 1, 1],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0, 0, 0],
                                    [0, 1, 1, 1, 0],
                                    [0, 1, 0, 1, 0],
                                    [0, 1, 1, 1, 0],
                                    [0, 0, 0, 0, 0],
                                ]
                            ),
                        ],
                        axis=-1,
                    ),
                    bbox=(0, 0, 5, 5),
                    padding=1,
                    pixel_to_nm_scaling=1.0,
                    filename="test_image",
                ),
                1: GrainCrop(
                    image=np.array(
                        [
                            [0.1, 0.2, 0.2, 0.2, 0.1],
                            [0.1, 1.1, 0.2, 1.1, 0.2],
                            [0.2, 1.2, 0.1, 0.2, 0.2],
                            [0.1, 1.1, 1.2, 1.1, 0.1],
                            [0.1, 0.1, 0.1, 0.2, 0.1],
                        ],
                    ),
                    mask=np.stack(
                        [
                            np.array(
                                [
                                    [1, 1, 1, 1, 1],
                                    [1, 0, 1, 0, 1],
                                    [1, 0, 1, 1, 1],
                                    [1, 0, 0, 0, 1],
                                    [1, 1, 1, 1, 1],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0, 0, 0],
                                    [0, 1, 0, 1, 0],
                                    [0, 1, 0, 0, 0],
                                    [0, 1, 1, 1, 0],
                                    [0, 0, 0, 0, 0],
                                ]
                            ),
                        ],
                        axis=-1,
                    ),
                    bbox=(0, 4, 5, 9),
                    padding=1,
                    pixel_to_nm_scaling=1.0,
                    filename="test_image",
                ),
                2: GrainCrop(
                    image=np.array(
                        [
                            [0.2, 0.2, 0.1],
                            [0.2, 1.1, 0.2],
                            [0.1, 0.2, 0.2],
                        ]
                    ),
                    mask=np.stack(
                        [
                            np.array(
                                [
                                    [1, 1, 1],
                                    [1, 0, 1],
                                    [1, 1, 1],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 0],
                                ]
                            ),
                        ],
                        axis=-1,
                    ),
                    bbox=(0, 6, 3, 9),
                    padding=1,
                    pixel_to_nm_scaling=1.0,
                    filename="test_image",
                ),
            },
            id="unet, 5x5, multi class, 3 grains",
        ),
        # Unet produces empty predictions for traditional mask
        pytest.param(
            # U-Net config
            {
                "model_path": "dummy_model_path",
                "confidence": 0.5,
                "model_input_shape": (None, 5, 5, 1),
                "upper_norm_bound": 1.0,
                "lower_norm_bound": 0.0,
                "grain_crop_padding": 1,
                "remove_disconnected_grains": False,
            },
            # Traditional graincrop
            {
                0: GrainCrop(
                    image=np.array(
                        [
                            [0.1, 0.2, 0.1, 0.2, 0.1],
                            [0.2, 0.1, 1.1, 0.1, 0.2],
                            [0.1, 1.1, 1.1, 1.1, 0.1],
                            [0.2, 0.1, 1.1, 0.1, 0.2],
                            [0.1, 0.2, 0.1, 0.2, 0.1],
                        ]
                    ),
                    mask=np.stack(
                        [
                            np.array(
                                [
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 0, 1, 1],
                                    [1, 0, 0, 0, 1],
                                    [1, 1, 0, 1, 1],
                                    [1, 1, 1, 1, 1],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0],
                                    [0, 1, 1, 1, 0],
                                    [0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 0],
                                ]
                            ),
                        ],
                        axis=-1,
                    ),
                    bbox=(0, 0, 5, 5),
                    padding=1,
                    pixel_to_nm_scaling=1.0,
                    filename="test_image",
                )
            },
            # Expected empty graincrops dictionary
            {},
            id="unet, 5x5, single class, no grains",
        ),
    ],
)
def test_improve_grain_segmentation_unet(
    mock_model_5_by_5_single_class: MagicMock,
    unet_config: dict[str, str | int | float | tuple[int | None, int, int, int]],
    graincrops: dict[int, GrainCrop],
    expected_graincrops: dict[int, GrainCrop],
) -> None:
    """Test the improve_grain_segmentation method of the Grains class with a unet model."""
    with patch("keras.models.load_model") as mock_load_model:
        mock_load_model.return_value = mock_model_5_by_5_single_class

        result_graincrops: dict[int, GrainCrop] = Grains.improve_grain_segmentation_unet(
            filename="test_image",
            direction="above",
            graincrops=graincrops,
            unet_config=unet_config,
        )

    assert result_graincrops == expected_graincrops


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


@pytest.mark.parametrize(
    ("multi_class_image", "expected_flattened_mask"),
    [
        pytest.param(
            np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1],
                            [1, 1, 0, 1, 1],
                            [1, 0, 0, 0, 1],
                            [1, 1, 0, 1, 1],
                            [1, 1, 1, 1, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0],
                            [0, 1, 0, 1, 0],
                            [0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ),
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            ),
            id="two class plus background, no overlap in classes",
        ),
        pytest.param(
            np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1],
                            [1, 1, 0, 1, 1],
                            [1, 0, 0, 0, 1],
                            [1, 1, 0, 1, 1],
                            [1, 1, 1, 1, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0],
                            [0, 1, 1, 1, 0],
                            [0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ),
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            ),
            id="two class plus background, overlap in class 1 and 2",
        ),
    ],
)
def test_flatten_multi_class_tensor(
    multi_class_image: npt.NDArray[np.int32], expected_flattened_mask: npt.NDArray[np.int32]
) -> None:
    """Test the flatten_multi_class_image method of the Grains class."""
    result = Grains.flatten_multi_class_tensor(multi_class_image)
    np.testing.assert_array_equal(result, expected_flattened_mask)


@pytest.mark.parametrize(
    ("grain_mask_tensor", "expected_bounding_boxes"),
    [
        pytest.param(
            np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 0, 1, 0, 0, 1, 1, 0, 1, 1],
                            [1, 1, 1, 0, 0, 1, 0, 0, 0, 1],
                            [1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ),
            {0: (0, 0, 3, 3), 1: (0, 2, 4, 6), 2: (0, 5, 5, 10)},
        )
    ],
)
def test_get_multi_class_grain_bounding_boxes(grain_mask_tensor: npt.NDArray, expected_bounding_boxes: dict) -> None:
    """Test the get_multi_class_grain_bounding_boxes method of the Grains class."""
    result = Grains.get_multi_class_grain_bounding_boxes(grain_mask_tensor)
    assert dict_almost_equal(result, expected_bounding_boxes, abs_tol=1e-12)


@pytest.mark.parametrize(
    ("grain_mask_tensor", "expected_updated_background_class_image_tensor"),
    [
        pytest.param(
            np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 0, 1, 0, 1, 1, 1, 1],
                            [1, 1, 1, 0, 1, 0, 1, 1, 1, 1],
                            [1, 1, 1, 0, 1, 0, 1, 1, 1, 1],
                            [1, 1, 1, 0, 1, 0, 0, 1, 1, 1],
                            [1, 1, 1, 0, 1, 0, 0, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 0, 0, 1, 0, 0, 0, 1, 1, 1],
                            [1, 0, 0, 1, 0, 0, 0, 0, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
                            [0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
                            [0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ),
            np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 0, 1, 0, 1, 0, 1, 1, 1, 1],
                            [1, 0, 1, 0, 1, 0, 1, 1, 1, 1],
                            [1, 0, 1, 0, 1, 0, 1, 1, 1, 1],
                            [1, 0, 1, 0, 1, 0, 0, 1, 1, 1],
                            [1, 0, 1, 0, 1, 0, 0, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                            [1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
                            [0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
                            [0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ),
        )
    ],
)
def test_update_background_class(
    grain_mask_tensor: npt.NDArray[np.int32], expected_updated_background_class_image_tensor: npt.NDArray[np.int32]
) -> None:
    """Test the update_background_class method of the Grains class."""
    result = Grains.update_background_class(grain_mask_tensor)
    np.testing.assert_array_equal(result, expected_updated_background_class_image_tensor)


@pytest.mark.parametrize(
    ("single_grain_mask_tensor", "keep_largest_labelled_regions_classes", "expected_result_grain_tensor"),
    [
        pytest.param(
            np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1, 1],
                            [1, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 1],
                            [1, 1, 0, 1, 0, 1],
                            [1, 0, 0, 0, 0, 1],
                            [1, 1, 1, 1, 1, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 1, 1, 0],
                            [0, 1, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 1, 0, 1, 0],
                            [0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ),
            [1],
            np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1, 1],
                            [1, 1, 0, 0, 0, 1],
                            [1, 1, 0, 0, 0, 1],
                            [1, 1, 0, 1, 0, 1],
                            [1, 1, 1, 1, 0, 1],
                            [1, 1, 1, 1, 1, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 0],
                            [0, 0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 1, 0, 1, 0],
                            [0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ),
        ),
        pytest.param(
            np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1, 1],
                            [1, 1, 0, 1, 1, 1],
                            [1, 1, 0, 1, 1, 1],
                            [1, 1, 0, 1, 0, 1],
                            [1, 1, 1, 1, 0, 1],
                            [1, 1, 1, 1, 1, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 1, 0, 1, 0],
                            [0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ),
            [1],
            np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1, 1],
                            [1, 1, 0, 1, 1, 1],
                            [1, 1, 0, 1, 1, 1],
                            [1, 1, 0, 1, 0, 1],
                            [1, 1, 1, 1, 0, 1],
                            [1, 1, 1, 1, 1, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 1, 0, 1, 0],
                            [0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ),
            id="no regions to keep",
        ),
    ],
)
def test_keep_largest_labelled_region_classes(
    single_grain_mask_tensor: npt.NDArray[np.int32],
    keep_largest_labelled_regions_classes: list,
    expected_result_grain_tensor: npt.NDArray[np.int32],
) -> None:
    """Test the keep_largest_labelled_regions method of the Grains class."""
    result = Grains.keep_largest_labelled_region_classes(
        single_grain_mask_tensor=single_grain_mask_tensor,
        keep_largest_labelled_regions_classes=keep_largest_labelled_regions_classes,
    )

    np.testing.assert_array_equal(result, expected_result_grain_tensor)


@pytest.mark.parametrize(
    ("grain_mask_tensor", "padding", "expected_result_grain_crops", "expected_bounding_boxes", "expected_padding"),
    [
        pytest.param(
            np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 0, 1, 0, 0, 0, 0, 1, 1],
                            [1, 1, 0, 1, 1, 1, 1, 0, 1, 1],
                            [1, 1, 0, 0, 0, 0, 1, 0, 0, 1],
                            [1, 1, 1, 1, 1, 1, 1, 0, 0, 1],
                            [1, 1, 1, 1, 1, 1, 1, 0, 0, 1],
                            [1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ).astype(bool),
            1,
            [
                np.stack(
                    [
                        np.array(
                            [
                                [1, 1, 1, 1, 1, 1, 1],
                                [1, 0, 0, 1, 1, 1, 1],
                                [1, 0, 0, 1, 1, 1, 1],
                                [1, 0, 0, 1, 1, 1, 1],
                                [1, 1, 0, 0, 0, 0, 1],
                                [1, 1, 1, 1, 1, 1, 1],
                            ]
                        ),
                        np.array(
                            [
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 1, 1, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0, 0],
                                [0, 0, 1, 1, 1, 1, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                            ]
                        ),
                        np.array(
                            [
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 1, 0, 0, 0, 0, 0],
                                [0, 1, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                            ],
                        ),
                    ],
                    axis=-1,
                ).astype(bool),
                np.stack(
                    [
                        np.array(
                            [
                                [1, 1, 1, 1, 1, 1, 1],
                                [1, 0, 0, 0, 0, 1, 1],
                                [1, 1, 1, 1, 0, 1, 1],
                                [1, 1, 1, 1, 0, 0, 1],
                                [1, 1, 1, 1, 0, 0, 1],
                                [1, 1, 1, 1, 0, 0, 1],
                                [1, 1, 1, 1, 0, 1, 1],
                                [1, 1, 1, 1, 0, 1, 1],
                                [1, 1, 1, 1, 1, 1, 1],
                            ]
                        ),
                        np.array(
                            [
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 1, 1, 1, 1, 0, 0],
                                [0, 0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                            ]
                        ),
                        np.array(
                            [
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                            ]
                        ),
                    ],
                    axis=-1,
                ).astype(bool),
            ],
            [np.array([0, 0, 6, 7]), np.array([1, 3, 10, 10])],
            1,
        ),
    ],
)
def test_get_individual_grain_crops(
    grain_mask_tensor: npt.NDArray[np.int32],
    padding: int,
    expected_result_grain_crops: list[npt.NDArray[np.int32]],
    expected_bounding_boxes: list[npt.NDArray[np.int32]],
    expected_padding: int,
) -> None:
    """Test the get_individual_grain_crops method of the Grains class."""
    result_grain_crops, result_bounding_boxes, result_padding = Grains.get_individual_grain_crops(
        grain_mask_tensor, padding
    )
    np.testing.assert_equal(result_grain_crops, expected_result_grain_crops)
    np.testing.assert_equal(result_bounding_boxes, expected_bounding_boxes)
    np.testing.assert_equal(result_padding, expected_padding)


@pytest.mark.parametrize(
    ("single_grain_mask_tensor", "class_region_number_thresholds", "expected_grain_mask_tensor", "expected_passed"),
    [
        pytest.param(
            np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1],
                            [1, 0, 0, 1, 1],
                            [1, 0, 0, 1, 1],
                            [1, 0, 0, 1, 1],
                            [1, 1, 1, 1, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ),
            [[1, 2, None]],
            np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ),
            False,
            id="too few regions in class 1",
        ),
        pytest.param(
            np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1],
                            [1, 0, 0, 1, 1],
                            [1, 0, 0, 1, 1],
                            [1, 0, 0, 1, 1],
                            [1, 1, 1, 1, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ),
            [[1, None, 1], [2, 1, 1]],
            np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ),
            False,
            id="too many regions in class 1",
        ),
        pytest.param(
            np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1],
                            [1, 0, 0, 1, 1],
                            [1, 0, 0, 1, 1],
                            [1, 0, 0, 1, 1],
                            [1, 1, 1, 1, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ),
            [[1, 2, 2], [2, 1, 1]],
            np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1],
                            [1, 0, 0, 1, 1],
                            [1, 0, 0, 1, 1],
                            [1, 0, 0, 1, 1],
                            [1, 1, 1, 1, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ),
            True,
            id="correct number of regions in all classes",
        ),
        pytest.param(
            np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1],
                            [1, 0, 0, 1, 1],
                            [1, 0, 0, 1, 1],
                            [1, 0, 0, 1, 1],
                            [1, 1, 1, 1, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ),
            [[1, None, None], [2, None, None]],
            np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1],
                            [1, 0, 0, 1, 1],
                            [1, 0, 0, 1, 1],
                            [1, 0, 0, 1, 1],
                            [1, 1, 1, 1, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ),
            True,
            id="none bounds",
        ),
        pytest.param(
            np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1],
                            [1, 0, 0, 1, 1],
                            [1, 0, 0, 1, 1],
                            [1, 0, 0, 1, 1],
                            [1, 1, 1, 1, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ),
            [],
            np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1],
                            [1, 0, 0, 1, 1],
                            [1, 0, 0, 1, 1],
                            [1, 0, 0, 1, 1],
                            [1, 1, 1, 1, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ),
            True,
            id="no thresholds provided",
        ),
    ],
)
def test_vet_numbers_of_regions_single_grain(
    single_grain_mask_tensor: npt.NDArray[np.int32],
    class_region_number_thresholds: dict,
    expected_grain_mask_tensor: npt.NDArray[np.int32],
    expected_passed: bool,
) -> None:
    """Test the vet_numbers_of_regions method of the Grains class."""
    result_crop, result_passed = Grains.vet_numbers_of_regions_single_grain(
        single_grain_mask_tensor, class_region_number_thresholds
    )
    np.testing.assert_array_equal(result_crop, expected_grain_mask_tensor)
    assert result_passed == expected_passed


@pytest.mark.parametrize(
    ("grain_mask_tensor", "classes_to_convert", "class_touching_threshold", "expected_result_grain_mask_tensor"),
    [
        pytest.param(
            np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 1, 1, 0],
                            [0, 0, 0, 1, 0, 0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0, 0, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ).astype(bool),
            [(2, 1)],
            1,
            np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                            [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ).astype(bool),
        ),
        pytest.param(
            np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ).astype(bool),
            [(2, 1)],
            1,
            np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ).astype(bool),
            id="empty class to convert",
        ),
    ],
)
def test_convert_classes_to_nearby_classes(
    grain_mask_tensor: npt.NDArray[np.int32],
    classes_to_convert: list[tuple[int, int]],
    class_touching_threshold: int,
    expected_result_grain_mask_tensor: npt.NDArray[np.int32],
) -> None:
    """Test the convert_classes_to_nearby_classes method of the Grains class."""
    result_grain_mask_tensor = Grains.convert_classes_to_nearby_classes(
        grain_mask_tensor, classes_to_convert, class_touching_threshold
    )

    np.testing.assert_array_equal(result_grain_mask_tensor, expected_result_grain_mask_tensor)


@pytest.mark.parametrize(
    (
        "grain_mask_tensor",
        "classes",
        "expected_num_connection_regions",
        "expected_intersection_labels",
        "expected_intersection_points",
    ),
    [
        pytest.param(
            np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ),
            (1, 2),
            3,
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
            {
                1: np.array([[1, 3], [2, 3]]),
                2: np.array([[4, 3], [5, 3]]),
                3: np.array([[7, 3], [8, 3]]),
            },
        )
    ],
)
def test_calculate_region_connection_regions(
    grain_mask_tensor: npt.NDArray[np.int32],
    classes: tuple[int, int],
    expected_num_connection_regions: int,
    expected_intersection_labels: npt.NDArray[np.int32],
    expected_intersection_points: list[tuple[int, int]],
) -> None:
    """Test the calculate_region_connection_regions method of the Grains class."""
    (result_num_connection_regions, result_intersection_labels, result_intersection_points) = (
        Grains.calculate_region_connection_regions(grain_mask_tensor, classes)
    )

    assert result_num_connection_regions == expected_num_connection_regions
    np.testing.assert_array_equal(result_intersection_labels, expected_intersection_labels)
    np.testing.assert_equal(result_intersection_points, expected_intersection_points)


@pytest.mark.parametrize(
    ("grain_mask_tensor", "pixel_to_nm_scaling", "class_conversion_size_thresholds", "expected_grain_mask_tensor"),
    [
        pytest.param(
            np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
                            [1, 1, 1, 0, 1, 0, 1, 0, 1, 1],
                            [1, 1, 1, 1, 1, 0, 1, 0, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
                            [1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 0, 1, 0, 1, 1, 1, 1, 1, 1],
                            [1, 0, 1, 0, 1, 0, 1, 1, 1, 1],
                            [1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
                            [0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
                            [0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ).astype(bool),
            1.0,
            [[(1, None, 2), (2, 3)], [(2, None, 1), (2, 3)]],
            np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 0, 1, 0, 1, 0, 1, 1],
                            [1, 1, 1, 0, 1, 0, 1, 0, 1, 1],
                            [1, 1, 1, 1, 1, 0, 1, 0, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
                            [1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 0, 1, 0, 1, 1, 1, 1, 1, 1],
                            [1, 0, 1, 0, 1, 0, 1, 1, 1, 1],
                            [1, 0, 1, 0, 1, 0, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ).astype(bool),
            id="switch classes 1 and 2",
        ),
    ],
)
def test_convert_classes_when_too_big_or_small(
    grain_mask_tensor: npt.NDArray[np.int32],
    pixel_to_nm_scaling: float,
    class_conversion_size_thresholds: list[tuple[tuple[int, int], tuple[int, int]]],
    expected_grain_mask_tensor: npt.NDArray[np.int32],
) -> None:
    """Test the convert_classes_when_too_big_or_small method of the Grains class."""
    result_grain_mask_tensor = Grains.convert_classes_when_too_big_or_small(
        grain_mask_tensor, pixel_to_nm_scaling, class_conversion_size_thresholds
    )

    np.testing.assert_array_equal(result_grain_mask_tensor, expected_grain_mask_tensor)


@pytest.mark.parametrize(
    ("grain_mask_tensor", "class_connection_point_thresholds", "expected_pass"),
    [
        pytest.param(
            np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ),
            [((1, 2), (4, 5))],
            False,
            id="not enough connection regions",
        ),
        pytest.param(
            np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ),
            [((1, 2), (1, 2))],
            False,
            id="too many connection regions",
        ),
        pytest.param(
            np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ),
            [((1, 2), (2, 4))],
            True,
            id="correct number of connection regions",
        ),
    ],
)
def test_vet_class_connection_points(
    grain_mask_tensor: npt.NDArray[np.int32],
    class_connection_point_thresholds: dict[tuple[int, int], tuple[int, int]],
    expected_pass: bool,
) -> None:
    """Test the vet_class_connection_points method of the Grains class."""
    result_pass = Grains.vet_class_connection_points(grain_mask_tensor, class_connection_point_thresholds)

    assert result_pass == expected_pass


@pytest.mark.parametrize(
    ("grain_mask_tensor_shape", "grain_crops_dicts", "expected_grain_mask_tensor"),
    [
        pytest.param(
            np.array([12, 12, 3]),
            [
                {
                    "grain_tensor": np.stack(
                        [
                            np.array(
                                [
                                    [1, 1, 1, 1, 1, 1],
                                    [1, 0, 0, 0, 0, 1],
                                    [1, 0, 0, 0, 0, 1],
                                    [1, 0, 0, 0, 0, 1],
                                    [1, 0, 0, 0, 0, 1],
                                    [1, 1, 1, 1, 1, 1],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0, 0, 0, 0],
                                    [0, 1, 0, 0, 1, 0],
                                    [0, 1, 0, 0, 1, 0],
                                    [0, 1, 1, 1, 1, 0],
                                    [0, 1, 1, 1, 1, 0],
                                    [0, 0, 0, 0, 0, 0],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 1, 0, 0],
                                    [0, 0, 1, 1, 0, 0],
                                    [0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0],
                                ]
                            ),
                        ],
                        axis=-1,
                    ).astype(bool),
                    "bounding_box": (0, 0, 6, 6),
                    "padding": 1,
                },
                {
                    "grain_tensor": np.stack(
                        [
                            np.array(
                                [
                                    [1, 1, 1, 1],
                                    [1, 0, 0, 1],
                                    [1, 0, 0, 1],
                                    [1, 1, 1, 1],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 0],
                                    [0, 1, 1, 0],
                                    [0, 0, 0, 0],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0, 0],
                                    [0, 1, 1, 0],
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 0],
                                ]
                            ),
                        ],
                        axis=-1,
                    ).astype(bool),
                    "bounding_box": (6, 6, 10, 10),
                    "padding": 1,
                },
            ],
            np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                            [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                            [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                            [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ).astype(bool),
        )
    ],
)
def test_assemble_grain_mask_tensor_from_crops(
    grain_mask_tensor_shape: npt.NDArray[np.int32],
    grain_crops_dicts: list[dict[str, np.ndarray]],
    expected_grain_mask_tensor: npt.NDArray[np.int32],
) -> None:
    """Test the assemble_grain_mask_tensor_from_crops method of the Grains class."""
    result_grain_mask_tensor = Grains.assemble_grain_mask_tensor_from_crops(grain_mask_tensor_shape, grain_crops_dicts)

    np.testing.assert_array_equal(result_grain_mask_tensor, expected_grain_mask_tensor)


@pytest.mark.parametrize(
    ("grain_mask_tensor", "classes_to_merge", "expected_result_grain_mask_tensor"),
    [
        pytest.param(
            np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1, 1],
                            [1, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 1],
                            [1, 1, 1, 1, 1, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 1, 0],
                            [0, 1, 1, 0, 1, 0],
                            [0, 1, 0, 0, 1, 0],
                            [0, 1, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0, 0],
                            [0, 0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ),
            [(1, 2)],
            np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1, 1],
                            [1, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 1],
                            [1, 1, 1, 1, 1, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 1, 0],
                            [0, 1, 1, 0, 1, 0],
                            [0, 1, 0, 0, 1, 0],
                            [0, 1, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0, 0],
                            [0, 0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 1, 0],
                            [0, 1, 1, 1, 1, 0],
                            [0, 1, 1, 1, 1, 0],
                            [0, 1, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ),
        )
    ],
)
def test_merge_classes(
    grain_mask_tensor: npt.NDArray[np.int32],
    classes_to_merge: list[tuple[int, int]],
    expected_result_grain_mask_tensor: npt.NDArray[np.int32],
) -> None:
    """Test the merge_classes method of the Grains class."""
    result_grain_mask_tensor = Grains.merge_classes(grain_mask_tensor, classes_to_merge)

    np.testing.assert_array_equal(result_grain_mask_tensor, expected_result_grain_mask_tensor)


@pytest.mark.parametrize(
    (
        "graincrops",
        "vet_grains_conf",
        "expected_graincrops",
    ),
    [
        pytest.param(
            # Graincrops
            {
                # Class conversion when too big or small
                0: GrainCrop(
                    image=np.array(
                        [
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                        ]
                    ).astype(np.float32),
                    mask=np.stack(
                        [
                            np.array(
                                [
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 0, 1, 0, 1, 0, 1, 1, 1],
                                    [1, 1, 1, 0, 1, 0, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 0, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 0, 0, 1, 0, 0, 1, 1, 1],
                                    [1, 0, 0, 1, 0, 0, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 1, 0, 1, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 1, 0, 0, 1, 1, 0, 0, 0],
                                    [0, 1, 0, 0, 1, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                ]
                            ),
                        ],
                        axis=-1,
                    ),
                    padding=1,
                    bbox=(0, 0, 10, 10),
                    pixel_to_nm_scaling=1.0,
                    filename="test",
                )
            },
            {
                # Convert class 1 to class 2 if too small, class 3 if too big
                "class_conversion_size_thresholds": [[(1, 2, 3), (2, 2)]],
                "class_size_thresholds": None,
                "class_region_number_thresholds": None,
                # Convert class 2 to 3 if touching and not largest of class 2
                "nearby_conversion_classes_to_convert": [(2, 3)],
                "class_touching_threshold": 1,
                "keep_largest_labelled_regions_classes": None,
                "class_connection_point_thresholds": None,
            },
            # Expected graincrops
            {
                # Class conversion when too big or small
                0: GrainCrop(
                    image=np.array(
                        [
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                        ]
                    ).astype(np.float32),
                    mask=np.stack(
                        [
                            np.array(
                                [
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 0, 1, 0, 1, 0, 1, 1, 1],
                                    [1, 1, 1, 0, 1, 0, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 0, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 0, 0, 1, 0, 0, 1, 1, 1],
                                    [1, 0, 0, 1, 0, 0, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 1, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 1, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 1, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 1, 1, 0, 0, 0, 0, 0, 0],
                                    [0, 1, 1, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                ]
                            ),
                        ],
                        axis=-1,
                    ),
                    padding=1,
                    bbox=(0, 0, 10, 10),
                    pixel_to_nm_scaling=1.0,
                    filename="test",
                )
            },
            id="class conversion based on size & class conversion when touching",
        ),
        pytest.param(
            # Graincrops
            {
                0: GrainCrop(
                    image=np.array(
                        [
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                        ]
                    ).astype(np.float32),
                    mask=np.stack(
                        [
                            np.array(
                                [
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 0, 0, 0, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 0, 1, 1, 0, 0, 1, 1, 1],
                                    [1, 0, 1, 0, 1, 1, 0, 1, 1],
                                    [1, 0, 1, 0, 1, 0, 0, 0, 1],
                                    [1, 1, 1, 1, 0, 0, 0, 0, 1],
                                    [1, 1, 1, 1, 1, 0, 0, 0, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 1, 1, 1, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 1, 1, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 1, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 1, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 1, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 1, 0, 0, 0, 1, 1, 1, 0],
                                    [0, 0, 0, 0, 0, 1, 1, 1, 0],
                                    [0, 0, 0, 0, 0, 1, 1, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                ]
                            ),
                        ],
                        axis=-1,
                    ),
                    padding=1,
                    bbox=(0, 0, 10, 10),
                    pixel_to_nm_scaling=1.0,
                    filename="test",
                ),
            },
            {
                "class_conversion_size_thresholds": None,
                "class_size_thresholds": None,
                "class_region_number_thresholds": None,
                "nearby_conversion_classes_to_convert": None,
                "class_touching_threshold": 1,
                "keep_largest_labelled_regions_classes": [1, 2],
                "class_connection_point_thresholds": None,
            },
            # Expected graincrops
            {
                0: GrainCrop(
                    image=np.array(
                        [
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                        ]
                    ).astype(np.float32),
                    mask=np.stack(
                        [
                            np.array(
                                [
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 0, 0, 1, 1, 1],
                                    [1, 1, 1, 0, 1, 1, 0, 1, 1],
                                    [1, 1, 1, 0, 1, 0, 0, 0, 1],
                                    [1, 1, 1, 1, 0, 0, 0, 0, 1],
                                    [1, 1, 1, 1, 1, 0, 0, 0, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 1, 1, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 1, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 1, 1, 1, 0],
                                    [0, 0, 0, 0, 0, 1, 1, 1, 0],
                                    [0, 0, 0, 0, 0, 1, 1, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                ]
                            ),
                        ],
                        axis=-1,
                    ),
                    padding=1,
                    bbox=(0, 0, 10, 10),
                    pixel_to_nm_scaling=1.0,
                    filename="test",
                ),
            },
            id="keep largest labelled regions classes",
        ),
        pytest.param(
            # Graincrops
            {
                # Class 1 too small
                0: GrainCrop(
                    image=np.array(
                        [
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                        ]
                    ),
                    mask=np.stack(
                        [
                            np.array(
                                [
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 0, 1, 1, 1],
                                    [1, 1, 1, 1, 1],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 1, 0, 1, 0],
                                    [0, 0, 0, 0, 0],
                                ]
                            ),
                        ],
                        axis=-1,
                    ),
                    padding=1,
                    bbox=(0, 0, 6, 6),
                    pixel_to_nm_scaling=1.0,
                    filename="test",
                ),
                # Class 1 too big
                1: GrainCrop(
                    image=np.array(
                        [
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                        ]
                    ),
                    mask=np.stack(
                        [
                            np.array(
                                [
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 0, 0, 0, 1],
                                    [1, 1, 1, 1, 1],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 1, 1, 1, 0],
                                    [0, 0, 0, 0, 0],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 1, 0, 1, 0],
                                    [0, 0, 0, 0, 0],
                                ]
                            ),
                        ],
                        axis=-1,
                    ),
                    padding=1,
                    bbox=(0, 0, 6, 6),
                    pixel_to_nm_scaling=1.0,
                    filename="test",
                ),
                # Class 2 too few regions
                2: GrainCrop(
                    image=np.array(
                        [
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                        ]
                    ),
                    mask=np.stack(
                        [
                            np.array(
                                [
                                    [1, 1, 1, 1, 1],
                                    [1, 0, 0, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 0, 1],
                                    [1, 1, 1, 1, 1],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0, 0, 0],
                                    [0, 1, 1, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0],
                                ]
                            ),
                        ],
                        axis=-1,
                    ),
                    padding=1,
                    bbox=(0, 0, 6, 6),
                    pixel_to_nm_scaling=1.0,
                    filename="test",
                ),
                # Class 2 too many regions
                3: GrainCrop(
                    image=np.array(
                        [
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                        ]
                    ),
                    mask=np.stack(
                        [
                            np.array(
                                [
                                    [1, 1, 1, 1, 1],
                                    [1, 0, 0, 0, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 0, 1, 0, 1],
                                    [1, 1, 1, 1, 1],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0, 0, 0],
                                    [0, 1, 1, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 1, 0, 1, 0],
                                    [0, 0, 0, 0, 0],
                                ]
                            ),
                        ],
                        axis=-1,
                    ),
                    padding=1,
                    bbox=(0, 0, 6, 6),
                    pixel_to_nm_scaling=1.0,
                    filename="test",
                ),
                # Correct size class 1 and correct number of regions class 2
                4: GrainCrop(
                    image=np.array(
                        [
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                        ]
                    ),
                    mask=np.stack(
                        [
                            np.array(
                                [
                                    [1, 1, 1, 1, 1],
                                    [1, 0, 0, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 0, 1, 0, 1],
                                    [1, 1, 1, 1, 1],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0, 0, 0],
                                    [0, 1, 1, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 1, 0, 1, 0],
                                    [0, 0, 0, 0, 0],
                                ]
                            ),
                        ],
                        axis=-1,
                    ),
                    padding=1,
                    bbox=(0, 0, 6, 6),
                    pixel_to_nm_scaling=1.0,
                    filename="test",
                ),
            },
            {
                "class_conversion_size_thresholds": None,
                "class_size_thresholds": [(1, 2, 2)],
                "class_region_number_thresholds": [(2, 2, 2)],
                "nearby_conversion_classes_to_convert": None,
                "class_touching_threshold": 1,
                "keep_largest_labelled_regions_classes": None,
                "class_connection_point_thresholds": None,
            },
            # Expected graincrops
            {
                4: GrainCrop(
                    image=np.array(
                        [
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                        ]
                    ),
                    mask=np.stack(
                        [
                            np.array(
                                [
                                    [1, 1, 1, 1, 1],
                                    [1, 0, 0, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 0, 1, 0, 1],
                                    [1, 1, 1, 1, 1],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0, 0, 0],
                                    [0, 1, 1, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 1, 0, 1, 0],
                                    [0, 0, 0, 0, 0],
                                ]
                            ),
                        ],
                        axis=-1,
                    ),
                    padding=1,
                    bbox=(0, 0, 6, 6),
                    pixel_to_nm_scaling=1.0,
                    filename="test",
                ),
            },
            id="class size & region number thresholds",
        ),
        pytest.param(
            # Graincrops
            {
                # Class 1 too small
                0: GrainCrop(
                    image=np.array(
                        [
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                        ]
                    ),
                    mask=np.stack(
                        [
                            np.array(
                                [
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 0, 1, 1, 1],
                                    [1, 1, 1, 1, 1],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 1, 0, 1, 0],
                                    [0, 0, 0, 0, 0],
                                ]
                            ),
                        ],
                        axis=-1,
                    ),
                    padding=1,
                    bbox=(0, 0, 6, 6),
                    pixel_to_nm_scaling=1.0,
                    filename="test",
                ),
                # Class 1 too big
                1: GrainCrop(
                    image=np.array(
                        [
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                        ]
                    ),
                    mask=np.stack(
                        [
                            np.array(
                                [
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 0, 0, 0, 1],
                                    [1, 1, 1, 1, 1],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 1, 1, 1, 0],
                                    [0, 0, 0, 0, 0],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 1, 0, 1, 0],
                                    [0, 0, 0, 0, 0],
                                ]
                            ),
                        ],
                        axis=-1,
                    ),
                    padding=1,
                    bbox=(0, 0, 6, 6),
                    pixel_to_nm_scaling=1.0,
                    filename="test",
                ),
                # Class 2 too few regions
                2: GrainCrop(
                    image=np.array(
                        [
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                        ]
                    ),
                    mask=np.stack(
                        [
                            np.array(
                                [
                                    [1, 1, 1, 1, 1],
                                    [1, 0, 0, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 0, 1],
                                    [1, 1, 1, 1, 1],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0, 0, 0],
                                    [0, 1, 1, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0],
                                ]
                            ),
                        ],
                        axis=-1,
                    ),
                    padding=1,
                    bbox=(0, 0, 6, 6),
                    pixel_to_nm_scaling=1.0,
                    filename="test",
                ),
                # Class 2 too many regions
                3: GrainCrop(
                    image=np.array(
                        [
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                        ]
                    ),
                    mask=np.stack(
                        [
                            np.array(
                                [
                                    [1, 1, 1, 1, 1],
                                    [1, 0, 0, 0, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 0, 1, 0, 1],
                                    [1, 1, 1, 1, 1],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0, 0, 0],
                                    [0, 1, 1, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 1, 0, 1, 0],
                                    [0, 0, 0, 0, 0],
                                ]
                            ),
                        ],
                        axis=-1,
                    ),
                    padding=1,
                    bbox=(0, 0, 6, 6),
                    pixel_to_nm_scaling=1.0,
                    filename="test",
                ),
                # Correct size class 1 and correct number of regions class 2
                4: GrainCrop(
                    image=np.array(
                        [
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                        ]
                    ),
                    mask=np.stack(
                        [
                            np.array(
                                [
                                    [1, 1, 1, 1, 1],
                                    [1, 0, 0, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 0, 1, 0, 1],
                                    [1, 1, 1, 1, 1],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0, 0, 0],
                                    [0, 1, 1, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 1, 0, 1, 0],
                                    [0, 0, 0, 0, 0],
                                ]
                            ),
                        ],
                        axis=-1,
                    ),
                    padding=1,
                    bbox=(0, 0, 6, 6),
                    pixel_to_nm_scaling=1.0,
                    filename="test",
                ),
            },
            {
                "class_conversion_size_thresholds": None,
                "class_size_thresholds": None,
                "class_region_number_thresholds": None,
                "nearby_conversion_classes_to_convert": None,
                "class_touching_threshold": 1,
                "keep_largest_labelled_regions_classes": None,
                "class_connection_point_thresholds": None,
            },
            # Expected graincrops
            {
                # Class 1 too small
                0: GrainCrop(
                    image=np.array(
                        [
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                        ]
                    ),
                    mask=np.stack(
                        [
                            np.array(
                                [
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 0, 1, 1, 1],
                                    [1, 1, 1, 1, 1],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 1, 0, 1, 0],
                                    [0, 0, 0, 0, 0],
                                ]
                            ),
                        ],
                        axis=-1,
                    ),
                    padding=1,
                    bbox=(0, 0, 6, 6),
                    pixel_to_nm_scaling=1.0,
                    filename="test",
                ),
                # Class 1 too big
                1: GrainCrop(
                    image=np.array(
                        [
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                        ]
                    ),
                    mask=np.stack(
                        [
                            np.array(
                                [
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 0, 0, 0, 1],
                                    [1, 1, 1, 1, 1],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 1, 1, 1, 0],
                                    [0, 0, 0, 0, 0],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 1, 0, 1, 0],
                                    [0, 0, 0, 0, 0],
                                ]
                            ),
                        ],
                        axis=-1,
                    ),
                    padding=1,
                    bbox=(0, 0, 6, 6),
                    pixel_to_nm_scaling=1.0,
                    filename="test",
                ),
                # Class 2 too few regions
                2: GrainCrop(
                    image=np.array(
                        [
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                        ]
                    ),
                    mask=np.stack(
                        [
                            np.array(
                                [
                                    [1, 1, 1, 1, 1],
                                    [1, 0, 0, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 0, 1],
                                    [1, 1, 1, 1, 1],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0, 0, 0],
                                    [0, 1, 1, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0],
                                ]
                            ),
                        ],
                        axis=-1,
                    ),
                    padding=1,
                    bbox=(0, 0, 6, 6),
                    pixel_to_nm_scaling=1.0,
                    filename="test",
                ),
                # Class 2 too many regions
                3: GrainCrop(
                    image=np.array(
                        [
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                        ]
                    ),
                    mask=np.stack(
                        [
                            np.array(
                                [
                                    [1, 1, 1, 1, 1],
                                    [1, 0, 0, 0, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 0, 1, 0, 1],
                                    [1, 1, 1, 1, 1],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0, 0, 0],
                                    [0, 1, 1, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 1, 0, 1, 0],
                                    [0, 0, 0, 0, 0],
                                ]
                            ),
                        ],
                        axis=-1,
                    ),
                    padding=1,
                    bbox=(0, 0, 6, 6),
                    pixel_to_nm_scaling=1.0,
                    filename="test",
                ),
                # Correct size class 1 and correct number of regions class 2
                4: GrainCrop(
                    image=np.array(
                        [
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                        ]
                    ),
                    mask=np.stack(
                        [
                            np.array(
                                [
                                    [1, 1, 1, 1, 1],
                                    [1, 0, 0, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 0, 1, 0, 1],
                                    [1, 1, 1, 1, 1],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0, 0, 0],
                                    [0, 1, 1, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                ]
                            ),
                            np.array(
                                [
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 1, 0, 1, 0],
                                    [0, 0, 0, 0, 0],
                                ]
                            ),
                        ],
                        axis=-1,
                    ),
                    padding=1,
                    bbox=(0, 0, 6, 6),
                    pixel_to_nm_scaling=1.0,
                    filename="test",
                ),
            },
            id="no parameters supplied, no edits",
        ),
    ],
)
def test_vet_grains(
    graincrops: dict[int, GrainCrop],
    vet_grains_conf: dict[str, Any],
    expected_graincrops: npt.NDArray[np.int32],
) -> None:
    """Test the vet_grains function."""
    result_graincrops = Grains.vet_grains(
        graincrops=graincrops,
        **vet_grains_conf,
    )

    assert result_graincrops == expected_graincrops


def test_graincrops_merge_classes() -> None:
    """Test the merge_classes function."""
    graincrops = {
        0: GrainCrop(
            image=np.array(
                [
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1],
                ]
            ),
            mask=np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1, 1],
                            [1, 0, 0, 0, 0, 1],
                            [1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ),
            padding=1,
            bbox=(0, 0, 7, 7),
            pixel_to_nm_scaling=1.0,
            filename="test",
        )
    }

    expected_graincrops = {
        0: GrainCrop(
            image=np.array(
                [
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1],
                ]
            ),
            mask=np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1, 1],
                            [1, 0, 0, 0, 0, 1],
                            [1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ),
            padding=1,
            bbox=(0, 0, 7, 7),
            pixel_to_nm_scaling=1.0,
            filename="test",
        )
    }

    result_graincrops = Grains.graincrops_merge_classes(graincrops=graincrops, classes_to_merge=[[1, 2], [3, 4]])

    assert result_graincrops == expected_graincrops


def test_graincrops_update_background_class() -> None:
    """Test the update_background_class function."""
    graincrops = {
        0: GrainCrop(
            image=np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ]
            ).astype(np.float32),
            mask=np.stack(
                [
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0, 0],
                            [0, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ),
            padding=1,
            bbox=(0, 0, 7, 7),
            pixel_to_nm_scaling=1.0,
            filename="test",
        ),
        1: GrainCrop(
            image=np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ]
            ).astype(np.float32),
            mask=np.stack(
                [
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0, 0],
                            [0, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ),
            padding=1,
            bbox=(0, 0, 7, 7),
            pixel_to_nm_scaling=1.0,
            filename="test",
        ),
    }

    expected_graincrops = {
        0: GrainCrop(
            image=np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ]
            ).astype(np.float32),
            mask=np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1, 1],
                            [1, 0, 0, 1, 1, 1],
                            [1, 0, 0, 1, 1, 1],
                            [1, 1, 1, 1, 0, 1],
                            [1, 1, 1, 0, 0, 1],
                            [1, 1, 1, 1, 1, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0, 0],
                            [0, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ),
            padding=1,
            bbox=(0, 0, 7, 7),
            pixel_to_nm_scaling=1.0,
            filename="test",
        ),
        1: GrainCrop(
            image=np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ]
            ).astype(np.float32),
            mask=np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 0, 1],
                            [1, 1, 1, 1, 1, 1],
                            [1, 0, 0, 1, 1, 1],
                            [1, 0, 0, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0, 0],
                            [0, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ),
            padding=1,
            bbox=(0, 0, 7, 7),
            pixel_to_nm_scaling=1.0,
            filename="test",
        ),
    }

    result_graincrops = Grains.graincrops_update_background_class(graincrops=graincrops)

    assert result_graincrops == expected_graincrops


def test_graincrops_remove_objects_too_small_to_process() -> None:
    """Test the remove_objects_too_small_to_process function."""
    graincrops = {
        0: GrainCrop(
            image=np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ]
            ).astype(np.float32),
            mask=np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1, 1],
                            [1, 0, 0, 1, 0, 1],
                            [1, 0, 0, 1, 0, 1],
                            [1, 1, 1, 0, 0, 1],
                            [1, 1, 1, 0, 0, 1],
                            [1, 1, 1, 1, 1, 1],
                        ]
                    ),
                    # One correct, one bbox not large enough
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 1, 0],
                            [0, 1, 1, 0, 1, 0],
                            [0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ),
                    # One correct, one not enough pixels
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 0],
                            [0, 0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ),
            padding=1,
            bbox=(0, 0, 7, 7),
            pixel_to_nm_scaling=1.0,
            filename="test",
        ),
    }

    expected_graincrops = {
        0: GrainCrop(
            image=np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ]
            ).astype(np.float32),
            mask=np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1, 1],
                            [1, 0, 0, 1, 1, 1],
                            [1, 0, 0, 1, 1, 1],
                            [1, 1, 1, 0, 0, 1],
                            [1, 1, 1, 0, 0, 1],
                            [1, 1, 1, 1, 1, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0, 0],
                            [0, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 0],
                            [0, 0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ),
            padding=1,
            bbox=(0, 0, 7, 7),
            pixel_to_nm_scaling=1.0,
            filename="test",
        ),
    }

    result_graincrops = Grains.graincrops_remove_objects_too_small_to_process(
        graincrops=graincrops,
        min_object_size=4,
        min_object_bbox_size=2,
    )

    assert result_graincrops == expected_graincrops


def test_graincrop_init() -> None:
    """Test the GrainCrop class initialisation."""
    graincrop = GrainCrop(
        image=np.array(
            [
                [1.1, 1.2, 1.3, 1.4],
                [1.5, 1.6, 1.7, 1.8],
                [1.9, 2.0, 2.1, 2.2],
                [2.3, 2.4, 2.5, 2.6],
            ]
        ),
        mask=np.stack(
            [
                np.array(
                    [
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                    ]
                ),
                np.array(
                    [
                        [0, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ]
                ),
                np.array(
                    [
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 0],
                    ]
                ),
            ],
            axis=-1,
        ),
        padding=1,
        bbox=(0, 0, 5, 5),
        pixel_to_nm_scaling=1.0,
        filename="test",
    )

    assert graincrop.image.sum() == 29.6
    assert graincrop.mask.shape == (4, 4, 3)


def test_graincrop_mask_setter_mask_dimensions_dont_match(dummy_graincrop: GrainCrop) -> None:
    """Test the GrainCrop class mask setter."""
    with pytest.raises(ValueError, match="Mask dimensions do not match image"):
        dummy_graincrop.mask = np.zeros((3, 3, 3)).astype(bool)


@pytest.mark.parametrize(
    ("mask_size", "padding", "graincrop_mask", "expected_graincrop_mask"),
    [
        pytest.param(
            6,
            1,
            np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1, 1],
                            [0, 0, 0, 1, 1, 1],
                            [1, 0, 0, 1, 1, 1],
                            [1, 1, 1, 0, 0, 1],
                            [1, 1, 1, 0, 0, 1],
                            [1, 1, 1, 0, 0, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 0],
                            [0, 0, 0, 1, 1, 0],
                            [0, 0, 0, 1, 1, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [1, 1, 1, 0, 0, 0],
                            [0, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ),
            np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1, 1],
                            [1, 0, 0, 1, 1, 1],
                            [1, 0, 0, 1, 1, 1],
                            [1, 1, 1, 0, 0, 1],
                            [1, 1, 1, 0, 0, 1],
                            [1, 1, 1, 1, 1, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 0],
                            [0, 0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0, 0],
                            [0, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ),
            id="padding 1",
        )
    ],
)
def test_graincrop_mask_setter(
    mask_size: int,
    padding: int,
    graincrop_mask: npt.NDArray,
    expected_graincrop_mask: npt.NDArray,
) -> None:
    """Test the GrainCrop class mask setter."""
    graincrop = GrainCrop(
        image=np.ones((mask_size, mask_size)).astype(np.float32),
        mask=graincrop_mask,
        padding=padding,
        bbox=(0, 0, mask_size, mask_size),
        pixel_to_nm_scaling=1.0,
        filename="test",
    )

    result_graincrop_mask = graincrop.mask

    np.testing.assert_array_equal(result_graincrop_mask, expected_graincrop_mask)


def test_graincrop_padding_setter(dummy_graincrop: GrainCrop) -> None:
    """Test the GrainCrop class padding setter."""
    with pytest.raises(ValueError, match="Padding must be an integer"):
        # Sylvia: need to find a way to get mypy to be able to ignore type errors
        # using type: ignore doesn't work, let me know if solution is found.
        dummy_graincrop.padding = "a"
    with pytest.raises(ValueError, match="Padding must be >= 1"):
        dummy_graincrop.padding = 0


@pytest.mark.parametrize(
    ("graincrop_1", "graincrop_2", "expected_result"),
    [
        pytest.param(
            GrainCrop(
                image=np.array(
                    [
                        [1.1, 1.2, 1.3, 1.4],
                        [1.5, 1.6, 1.7, 1.8],
                        [1.9, 2.0, 2.1, 2.2],
                        [2.3, 2.4, 2.5, 2.6],
                    ]
                ),
                mask=np.stack(
                    [
                        np.array(
                            [
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                            ]
                        ),
                        np.array(
                            [
                                [0, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                            ]
                        ),
                        np.array(
                            [
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 0],
                            ]
                        ),
                    ],
                    axis=-1,
                ),
                padding=1,
                bbox=(0, 0, 5, 5),
                pixel_to_nm_scaling=1.0,
                filename="test",
            ),
            GrainCrop(
                image=np.array(
                    [
                        [1.1, 1.2, 1.3, 1.4],
                        [1.5, 1.6, 1.7, 1.8],
                        [1.9, 2.0, 2.1, 2.2],
                        [2.3, 2.4, 2.5, 2.6],
                    ]
                ),
                mask=np.stack(
                    [
                        np.array(
                            [
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                            ]
                        ),
                        np.array(
                            [
                                [0, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                            ]
                        ),
                        np.array(
                            [
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 0],
                            ]
                        ),
                    ],
                    axis=-1,
                ),
                padding=1,
                bbox=(0, 0, 5, 5),
                pixel_to_nm_scaling=1.0,
                filename="test",
            ),
            True,
            id="equal",
        ),
        pytest.param(
            GrainCrop(
                image=np.array(
                    [
                        [1.1, 1.2, 1.3, 1.4],
                        [1.5, 1.6, 1.7, 1.8],
                        [1.9, 2.0, 2.1, 2.2],
                        [2.3, 2.4, 2.5, 2.6],
                    ]
                ),
                mask=np.stack(
                    [
                        np.array(
                            [
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                            ]
                        ),
                        np.array(
                            [
                                [0, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                            ]
                        ),
                        np.array(
                            [
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 0],
                            ]
                        ),
                    ],
                    axis=-1,
                ),
                padding=1,
                bbox=(0, 0, 5, 5),
                pixel_to_nm_scaling=1.0,
                filename="test",
            ),
            GrainCrop(
                image=np.array(
                    [
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ]
                ),
                mask=np.stack(
                    [
                        np.array(
                            [
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                            ]
                        ),
                        np.array(
                            [
                                [0, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                            ]
                        ),
                        np.array(
                            [
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 0],
                            ]
                        ),
                    ],
                    axis=-1,
                ),
                padding=1,
                bbox=(0, 0, 5, 5),
                pixel_to_nm_scaling=1.0,
                filename="test",
            ),
            False,
            id="image not equal",
        ),
        pytest.param(
            GrainCrop(
                image=np.array(
                    [
                        [1.1, 1.2, 1.3, 1.4],
                        [1.5, 1.6, 1.7, 1.8],
                        [1.9, 2.0, 2.1, 2.2],
                        [2.3, 2.4, 2.5, 2.6],
                    ]
                ),
                mask=np.stack(
                    [
                        np.array(
                            [
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                            ]
                        ),
                        np.array(
                            [
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                            ]
                        ),
                        np.array(
                            [
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                            ]
                        ),
                    ],
                    axis=-1,
                ),
                padding=1,
                bbox=(0, 0, 5, 5),
                pixel_to_nm_scaling=1.0,
                filename="test",
            ),
            GrainCrop(
                image=np.array(
                    [
                        [1.1, 1.2, 1.3, 1.4],
                        [1.5, 1.6, 1.7, 1.8],
                        [1.9, 2.0, 2.1, 2.2],
                        [2.3, 2.4, 2.5, 2.6],
                    ]
                ),
                mask=np.stack(
                    [
                        np.array(
                            [
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                            ]
                        ),
                        np.array(
                            [
                                [0, 0, 0, 0],
                                [0, 1, 1, 0],
                                [0, 1, 1, 0],
                                [0, 0, 0, 0],
                            ]
                        ),
                        np.array(
                            [
                                [0, 0, 0, 0],
                                [0, 1, 1, 0],
                                [0, 1, 1, 0],
                                [0, 0, 0, 0],
                            ]
                        ),
                    ],
                    axis=-1,
                ),
                padding=1,
                bbox=(0, 0, 5, 5),
                pixel_to_nm_scaling=1.0,
                filename="test",
            ),
            False,
            id="mask not equal",
        ),
        pytest.param(
            GrainCrop(
                image=np.array(
                    [
                        [1.1, 1.2, 1.3, 1.4],
                        [1.5, 1.6, 1.7, 1.8],
                        [1.9, 2.0, 2.1, 2.2],
                        [2.3, 2.4, 2.5, 2.6],
                    ]
                ),
                mask=np.stack(
                    [
                        np.array(
                            [
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                            ]
                        ),
                        np.array(
                            [
                                [0, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                            ]
                        ),
                        np.array(
                            [
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 0],
                            ]
                        ),
                    ],
                    axis=-1,
                ),
                padding=1,
                bbox=(0, 0, 5, 5),
                pixel_to_nm_scaling=1.0,
                filename="test",
            ),
            GrainCrop(
                image=np.array(
                    [
                        [1.1, 1.2, 1.3, 1.4],
                        [1.5, 1.6, 1.7, 1.8],
                        [1.9, 2.0, 2.1, 2.2],
                        [2.3, 2.4, 2.5, 2.6],
                    ]
                ),
                mask=np.stack(
                    [
                        np.array(
                            [
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                            ]
                        ),
                        np.array(
                            [
                                [0, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                            ]
                        ),
                        np.array(
                            [
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 0],
                            ]
                        ),
                    ],
                    axis=-1,
                ),
                padding=2,
                bbox=(0, 0, 5, 5),
                pixel_to_nm_scaling=1.0,
                filename="test",
            ),
            False,
            id="padding not equal",
        ),
        pytest.param(
            GrainCrop(
                image=np.array(
                    [
                        [1.1, 1.2, 1.3, 1.4],
                        [1.5, 1.6, 1.7, 1.8],
                        [1.9, 2.0, 2.1, 2.2],
                        [2.3, 2.4, 2.5, 2.6],
                    ]
                ),
                mask=np.stack(
                    [
                        np.array(
                            [
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                            ]
                        ),
                        np.array(
                            [
                                [0, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                            ]
                        ),
                        np.array(
                            [
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 0],
                            ]
                        ),
                    ],
                    axis=-1,
                ),
                padding=1,
                bbox=(0, 0, 5, 5),
                pixel_to_nm_scaling=1.0,
                filename="test",
            ),
            GrainCrop(
                image=np.array(
                    [
                        [1.1, 1.2, 1.3, 1.4],
                        [1.5, 1.6, 1.7, 1.8],
                        [1.9, 2.0, 2.1, 2.2],
                        [2.3, 2.4, 2.5, 2.6],
                    ]
                ),
                mask=np.stack(
                    [
                        np.array(
                            [
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                            ]
                        ),
                        np.array(
                            [
                                [0, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                            ]
                        ),
                        np.array(
                            [
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 0],
                            ]
                        ),
                    ],
                    axis=-1,
                ),
                padding=1,
                bbox=(0, 0, 4, 4),
                pixel_to_nm_scaling=1.0,
                filename="test",
            ),
            False,
            id="padding not equal",
        ),
        pytest.param(
            GrainCrop(
                image=np.array(
                    [
                        [1.1, 1.2, 1.3, 1.4],
                        [1.5, 1.6, 1.7, 1.8],
                        [1.9, 2.0, 2.1, 2.2],
                        [2.3, 2.4, 2.5, 2.6],
                    ]
                ),
                mask=np.stack(
                    [
                        np.array(
                            [
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                            ]
                        ),
                        np.array(
                            [
                                [0, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                            ]
                        ),
                        np.array(
                            [
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 0],
                            ]
                        ),
                    ],
                    axis=-1,
                ),
                padding=1,
                bbox=(0, 0, 5, 5),
                pixel_to_nm_scaling=1.0,
                filename="test",
            ),
            GrainCrop(
                image=np.array(
                    [
                        [1.1, 1.2, 1.3, 1.4],
                        [1.5, 1.6, 1.7, 1.8],
                        [1.9, 2.0, 2.1, 2.2],
                        [2.3, 2.4, 2.5, 2.6],
                    ]
                ),
                mask=np.stack(
                    [
                        np.array(
                            [
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                            ]
                        ),
                        np.array(
                            [
                                [0, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                            ]
                        ),
                        np.array(
                            [
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 0],
                            ]
                        ),
                    ],
                    axis=-1,
                ),
                padding=1,
                bbox=(0, 0, 5, 5),
                pixel_to_nm_scaling=2.0,
                filename="test",
            ),
            False,
            id="p2nm not equal",
        ),
        pytest.param(
            GrainCrop(
                image=np.array(
                    [
                        [1.1, 1.2, 1.3, 1.4],
                        [1.5, 1.6, 1.7, 1.8],
                        [1.9, 2.0, 2.1, 2.2],
                        [2.3, 2.4, 2.5, 2.6],
                    ]
                ),
                mask=np.stack(
                    [
                        np.array(
                            [
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                            ]
                        ),
                        np.array(
                            [
                                [0, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                            ]
                        ),
                        np.array(
                            [
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 0],
                            ]
                        ),
                    ],
                    axis=-1,
                ),
                padding=1,
                bbox=(0, 0, 5, 5),
                pixel_to_nm_scaling=1.0,
                filename="test",
            ),
            GrainCrop(
                image=np.array(
                    [
                        [1.1, 1.2, 1.3, 1.4],
                        [1.5, 1.6, 1.7, 1.8],
                        [1.9, 2.0, 2.1, 2.2],
                        [2.3, 2.4, 2.5, 2.6],
                    ]
                ),
                mask=np.stack(
                    [
                        np.array(
                            [
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                            ]
                        ),
                        np.array(
                            [
                                [0, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                            ]
                        ),
                        np.array(
                            [
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 0],
                            ]
                        ),
                    ],
                    axis=-1,
                ),
                padding=1,
                bbox=(0, 0, 5, 5),
                pixel_to_nm_scaling=1.0,
                filename="wrong filename",
            ),
            False,
            id="filename not equal",
        ),
    ],
)
def test_graincrop_eq(graincrop_1: GrainCrop, graincrop_2: GrainCrop, expected_result: bool) -> None:
    """Test the GrainCrop class __eq__ method."""
    assert (graincrop_1 == graincrop_2) == expected_result


def test_validate_full_mask_tensor_shape() -> None:
    """Test the validate_full_mask_tensor_shape function."""
    with pytest.raises(ValueError, match="Full mask tensor must be WxHxC with C >= 2 but has shape"):
        validate_full_mask_tensor_shape(np.zeros((3, 3, 1)))

    assert validate_full_mask_tensor_shape(np.zeros((3, 3, 2))) is not None


def test_graincropsdirection_init(dummy_graincrop: GrainCrop) -> None:
    """Test the GrainCropsDirection class initialisation."""
    graincropsdirection = GrainCropsDirection(
        crops={
            0: dummy_graincrop,
            1: dummy_graincrop,
        },
        full_mask_tensor=np.zeros((10, 10, 2)).astype(bool),
    )

    assert len(graincropsdirection.crops) == 2
    assert graincropsdirection.full_mask_tensor.shape == (10, 10, 2)


def test_graincropsdirection_update_full_mask_tensor() -> None:
    """Test the update_full_mask_tensor method of the GrainCropsDirection class."""
    # Create a graincropsdirection instance
    graincropsdirection = GrainCropsDirection(
        crops={
            0: GrainCrop(
                image=np.array(
                    [
                        [1.1, 1.2, 1.3, 1.4],
                        [1.5, 1.6, 1.7, 1.8],
                        [1.9, 2.0, 2.1, 2.2],
                        [2.3, 2.4, 2.5, 2.6],
                    ]
                ),
                mask=np.stack(
                    [
                        np.array(
                            [
                                [1, 1, 1, 1],
                                [1, 1, 0, 1],
                                [1, 0, 1, 1],
                                [1, 1, 1, 1],
                            ]
                        ),
                        np.array(
                            [
                                [0, 0, 0, 0],
                                [0, 0, 1, 0],
                                [0, 1, 0, 0],
                                [0, 0, 0, 0],
                            ]
                        ),
                    ],
                    axis=-1,
                ).astype(bool),
                padding=1,
                bbox=(0, 0, 4, 4),
                pixel_to_nm_scaling=1.0,
                filename="test",
            )
        },
        full_mask_tensor=np.stack(
            [
                np.array(
                    [
                        [1, 1, 1, 1, 1, 1],
                        [1, 1, 0, 1, 1, 1],
                        [1, 0, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1],
                    ]
                ),
                np.array(
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ]
                ),
            ],
            axis=-1,
        ).astype(bool),
    )

    # Edit the grain so the full mask tensor needs to be updated
    graincropsdirection.crops[0].mask = np.stack(
        [
            np.array(
                [
                    [1, 1, 1, 1],
                    [1, 0, 0, 1],
                    [1, 0, 0, 1],
                    [1, 1, 1, 1],
                ]
            ),
            np.array(
                [
                    [0, 0, 0, 0],
                    [0, 1, 1, 0],
                    [0, 1, 1, 0],
                    [0, 0, 0, 0],
                ]
            ),
        ],
        axis=-1,
    ).astype(bool)

    # Update the full mask tensor
    graincropsdirection.update_full_mask_tensor()

    expected_full_mask_tensor = np.stack(
        [
            np.array(
                [
                    [1, 1, 1, 1, 1, 1],
                    [1, 0, 0, 1, 1, 1],
                    [1, 0, 0, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1],
                ]
            ),
            np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ]
            ),
        ],
        axis=-1,
    ).astype(bool)

    result_full_mask_tensor = graincropsdirection.full_mask_tensor

    np.testing.assert_array_equal(result_full_mask_tensor, expected_full_mask_tensor)


@pytest.mark.parametrize(
    ("original_grain_tensor", "predicted_grain_tensor", "expected_result_grain_tensor"),
    [
        pytest.param(
            np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ),
            np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ),
            np.stack(
                [
                    np.array(
                        [
                            [1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                        ]
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                        ]
                    ),
                ],
                axis=-1,
            ),
        )
    ],
)
def test_remove_disconnected_grains(
    original_grain_tensor: npt.NDArray[np.int32],
    predicted_grain_tensor: npt.NDArray[np.int32],
    expected_result_grain_tensor: npt.NDArray[np.int32],
) -> None:
    """Test the remove_disconnected_grains method of the Grains class."""
    result_grain_tensor = Grains.remove_disconnected_grains(
        original_grain_tensor=original_grain_tensor, predicted_grain_tensor=predicted_grain_tensor
    )

    np.testing.assert_array_equal(result_grain_tensor, expected_result_grain_tensor)
