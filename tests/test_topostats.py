"""Tests for the main module components defined in ''__init__.py''."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

import topostats
from topostats.grains import GrainCrop

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"
GRAINCROP_DIR = RESOURCES / "graincrop"

# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments

SEED = 4092024
rng = np.random.default_rng(SEED)


@pytest.mark.parametrize(
    (
        "topostats_object",
        "crops",
        "full_mask_tensor",
        "filename",
        "pixel_to_nm_scaling",
        "topostats_version",
        "img_path",
        "image",
        "image_original",
    ),
    [
        pytest.param(
            "topostats_catenanes_2_4_0",
            "crops_catenanes",
            "full_mask_tensor_catenanes",
            "example_catenanes.spm",
            0.488,
            "2.4.0",
            str(GRAINCROP_DIR),
            None,
            None,
            id="catenane v2.4.0",
        ),
        pytest.param(
            "topostats_rep_int_2_4_0",
            "crops_rep_int",
            "full_mask_tensor_rep_int",
            "example_rep_int.spm",
            0.488,
            "2.4.0",
            str(GRAINCROP_DIR),
            None,
            None,
            id="rep_int v2.4.0",
        ),
    ],
)
def test_topostats_to_dict(
    topostats_object: topostats.TopoStats,
    crops: dict[int, GrainCrop],
    full_mask_tensor: npt.NDArray[np.bool_],
    filename: str,
    pixel_to_nm_scaling: float,
    topostats_version: str,
    img_path: str,
    image: npt.NDArray | None,
    image_original: npt.NDArray | None,
    request,
) -> None:
    """Test conversion of TopoStats object to dictionary."""
    topostats_object = request.getfixturevalue(topostats_object)
    crops = request.getfixturevalue(crops)
    full_mask_tensor = request.getfixturevalue(full_mask_tensor)

    expected = {
        "crops": crops,
        "full_mask_tensor": full_mask_tensor,
        "filename": filename,
        "pixel_to_nm_scaling": pixel_to_nm_scaling,
        "topostats_version": topostats_version,
        "img_path": Path(img_path),
        "image": image,
        "image_original": image_original,
    }
    np.testing.assert_array_equal(topostats_object.topostats_to_dict(), expected)


@pytest.mark.parametrize(
    (
        "topostats_object",
        "crops",
        "full_mask_tensor",
        "filename",
        "pixel_to_nm_scaling",
        "topostats_version",
        "img_path",
        "image",
        "image_original",
    ),
    [
        pytest.param(
            "topostats_catenanes_2_4_0",
            "crops_catenanes",
            "full_mask_tensor_catenanes",
            "example_catenanes.spm",
            0.488,
            "2.4.0",
            str(GRAINCROP_DIR),
            rng.random((10, 10)),
            rng.random((10, 10)),
            id="catenane v2.4.0",
        ),
        pytest.param(
            "topostats_rep_int_2_4_0",
            "crops_rep_int",
            "full_mask_tensor_rep_int",
            "example_rep_int.spm",
            0.488,
            "2.4.0",
            str(GRAINCROP_DIR),
            rng.random((10, 10)),
            rng.random((10, 10)),
            id="tep_int v2.4.0",
        ),
    ],
)
def test_topostats_eq(
    topostats_object: topostats.TopoStats,
    crops: dict[int, GrainCrop],
    full_mask_tensor: npt.NDArray[np.bool_],
    filename: str,
    pixel_to_nm_scaling: float,
    topostats_version: str,
    img_path: str,
    image: npt.NDArray | None,
    image_original: npt.NDArray | None,
    request,
) -> None:
    """Test the TopoStats.__eq__ method."""
    topostats_object = request.getfixturevalue(topostats_object)
    topostats_object.image = image
    topostats_object.image_original = image_original
    crops = request.getfixturevalue(crops)
    full_mask_tensor = request.getfixturevalue(full_mask_tensor)
    expected = topostats.TopoStats(
        crops, full_mask_tensor, filename, pixel_to_nm_scaling, img_path, image, image_original, topostats_version
    )
    assert topostats_object == expected


def test_update_full_mask_tensor() -> None:
    """Test the update_full_mask_tensor method of the Topostats class."""
    # Create  TopoStats instance and add custom crops and full mask tensor
    topostats_object = topostats.TopoStats(img_path="dummy/path")
    topostats_object.crops = {
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
            threshold_idx=0,
        )
    }
    topostats_object.full_mask_tensor = np.stack(
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
    ).astype(bool)

    # Edit the grain so the full mask tensor needs to be updated
    topostats_object.crops[0].mask = np.stack(
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
    topostats_object.update_full_mask_tensor()

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

    result_full_mask_tensor = topostats_object.full_mask_tensor

    np.testing.assert_array_equal(result_full_mask_tensor, expected_full_mask_tensor)
