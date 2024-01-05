"""Find grains in an image."""

from pathlib import Path

# pylint: disable=no-name-in-module
from collections import defaultdict
import logging
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt

from skimage.segmentation import clear_border
from skimage import morphology
from skimage.measure import regionprops
from skimage.color import label2rgb

from topostats.grain_finding_haribo_unet import (
    predict_unet,
    load_model,
    predict_unet_multiclass_and_get_angle,
)
from topostats.logs.logs import LOGGER_NAME
from topostats.thresholds import threshold
from topostats.utils import _get_mask, get_thresholds

LOGGER = logging.getLogger(LOGGER_NAME)

# pylint: disable=fixme
# pylint: disable=line-too-long
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-arguments
# pylint: disable=bare-except
# pylint: disable=dangerous-default-value


class Grains:
    """Find grains in an image."""

    def __init__(
        self,
        image: np.ndarray,
        filename: str,
        pixel_to_nm_scaling: float,
        data_save_dir: Path,
        threshold_method: str = None,
        otsu_threshold_multiplier: float = None,
        threshold_std_dev: dict = None,
        threshold_absolute: dict = None,
        absolute_area_threshold: dict = {
            "above": [None, None],
            "below": [None, None],
        },
        direction: str = None,
        smallest_grain_size_nm2: float = None,
        remove_edge_intersecting_grains: bool = True,
    ):
        """Initialise the class.

        Parameters
        ----------
        image: np.ndarray
            2D Numpy array of image
        filename: str
            File being processed
        pixel_to_nm_scaling: float
            Sacling of pixels to nanometre.
        threshold_multiplier : Union[int, float]
            Factor by which below threshold is to be scaled prior to masking.
        threshold_method: str
            Method for determining threshold to mask values, default is 'otsu'.
        threshold_std_dev: dict
            Dictionary of 'below' and 'above' factors by which standard deviation is multiplied to derive the threshold if threshold_method is 'std_dev'.
        threshold_absolute: dict
            Dictionary of absolute 'below' and 'above' thresholds for grain finding.
        absolute_area_threshold: dict
            Dictionary of above and below grain's area thresholds
        direction: str
            Direction for which grains are to be detected, valid values are above, below and both.
        remove_edge_intersecting_grains: bool
            Whether or not to remove grains that intersect the edge of the image.
        """
        self.data_save_dir = data_save_dir
        self.image = image
        self.filename = filename
        self.pixel_to_nm_scaling = pixel_to_nm_scaling
        self.threshold_method = threshold_method
        self.otsu_threshold_multiplier = otsu_threshold_multiplier
        self.threshold_std_dev = threshold_std_dev
        self.threshold_absolute = threshold_absolute
        self.absolute_area_threshold = absolute_area_threshold
        # Only detect grains for the desired direction
        self.direction = [direction] if direction != "both" else ["above", "below"]
        self.smallest_grain_size_nm2 = smallest_grain_size_nm2
        self.remove_edge_intersecting_grains = remove_edge_intersecting_grains
        self.thresholds = None
        self.images = {
            "mask_grains": None,
            "tidied_border": None,
            "tiny_objects_removed": None,
            "objects_removed": None,
            # "labelled_regions": None,
            # "coloured_regions": None,
        }
        self.directions = defaultdict()
        self.minimum_grain_size = None
        self.region_properties = defaultdict()
        self.bounding_boxes = defaultdict()
        self.grainstats = None

    def tidy_border(self, image: np.array, **kwargs) -> np.array:
        """Remove grains touching the border.

        Parameters
        ----------
        image: np.array
            Numpy array representing image.

        Returns
        -------
        np.array
            Numpy array of image with borders tidied.
        """
        LOGGER.info(f"[{self.filename}] : Tidying borders")
        return clear_border(image, **kwargs)

    def label_regions(self, image: np.array) -> np.array:
        """Label regions.

        This method is used twice, once prior to removal of small regions, and again afterwards, hence requiring an
        argument of what image to label.

        Parameters
        ----------
        image: np.array
            Numpy array representing image.

        Returns
        -------
        np.array
            Numpy array of image with objects coloured.
        """
        LOGGER.info(f"[{self.filename}] : Labelling Regions")
        return morphology.label(image, background=0)

    def calc_minimum_grain_size(self, image: np.ndarray) -> float:
        """Calculate the minimum grain size in pixels squared.

        Very small objects are first removed via thresholding before calculating the below extreme.
        """
        region_properties = self.get_region_properties(image)
        grain_areas = np.array([grain.area for grain in region_properties])
        if len(grain_areas > 0):
            # Exclude small objects less than a given threshold first
            grain_areas = grain_areas[
                grain_areas >= threshold(grain_areas, method="otsu", otsu_threshold_multiplier=1.0)
            ]
            self.minimum_grain_size = np.median(grain_areas) - (
                1.5 * (np.quantile(grain_areas, 0.75) - np.quantile(grain_areas, 0.25))
            )
        else:
            self.minimum_grain_size = -1

    def remove_noise(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Removes noise which are objects smaller than the 'smallest_grain_size_nm2'.

        This ensures that the smallest objects ~1px are removed regardless of the size distribution of the grains.

        Parameters
        ----------
        image: np.ndarray
            2D Numpy image to be cleaned.

        Returns
        -------
        np.ndarray
            2D Numpy array of image with objects < smallest_grain_size_nm2 removed.
        """
        LOGGER.info(
            f"[{self.filename}] : Removing noise (< {self.smallest_grain_size_nm2} nm^2"
            "{self.smallest_grain_size_nm2 / (self.pixel_to_nm_scaling**2):.2f} px^2)"
        )
        return morphology.remove_small_objects(
            image, min_size=self.smallest_grain_size_nm2 / (self.pixel_to_nm_scaling**2), **kwargs
        )

    def remove_small_objects(self, image: np.array, **kwargs):
        """Remove small objects from the input image. Threshold determined by the minimum_grain_size variable of the
        Grains class which is in pixels squared.

        Parameters
        ----------
        image: np.ndarray
            2D Numpy image to remove small objects from.
        Returns
        -------
        np.ndarray
            2D Numpy array of image with objects < minimum_grain_size removed.
        """
        # If self.minimum_grain_size is -1, then this means that
        # there were no grains to calculate the minimum grian size from.
        if self.minimum_grain_size != -1:
            small_objects_removed = morphology.remove_small_objects(
                image,
                min_size=self.minimum_grain_size,  # minimum_grain_size is in pixels squared
                **kwargs,
            )
            LOGGER.info(
                f"[{self.filename}] : Removed small objects (< \
{self.minimum_grain_size} px^2 / {self.minimum_grain_size / (self.pixel_to_nm_scaling)**2} nm^2)"
            )
            return small_objects_removed > 0.0
        return image

    def area_thresholding(self, image: np.ndarray, area_thresholds: list):
        """Removes objects larger and smaller than the specified thresholds.

        Parameters
        ----------
        image: np.ndarray
            Image array where the background == 0 and grains are labelled as integers > 0.
        area_thresholds: list
            List of area thresholds (in nanometres squared, not pixels squared), first should be
            the lower limit for size, and the second should be the upper limit for size.

        Returns
        -------
        np.ndarray
            Image where grains outside the thresholds have been removed, as a re-numbered labeled image.

        """
        image_cp = image.copy()
        lower_size_limit, upper_size_limit = area_thresholds
        # if one value is None adjust for comparison
        if upper_size_limit is None:
            upper_size_limit = image.size * self.pixel_to_nm_scaling**2
        if lower_size_limit is None:
            lower_size_limit = 0
        # Get array of grain numbers (discounting zero)
        uniq = np.delete(np.unique(image), 0)
        grain_count = 0
        LOGGER.info(
            f"[{self.filename}] : Area thresholding grains | Thresholds: L: {(lower_size_limit / self.pixel_to_nm_scaling**2):.2f},"
            f"U: {(upper_size_limit / self.pixel_to_nm_scaling**2):.2f} px^2, L: {lower_size_limit:.2f}, U: {upper_size_limit:.2f} nm^2."
        )
        for grain_no in uniq:  # Calculate grian area in nm^2
            grain_area = np.sum(image_cp == grain_no) * (self.pixel_to_nm_scaling**2)
            # Compare area in nm^2 to area thresholds
            if grain_area > upper_size_limit or grain_area < lower_size_limit:
                image_cp[image_cp == grain_no] = 0
            else:
                grain_count += 1
                image_cp[image_cp == grain_no] = grain_count
        return image_cp

    def colour_regions(self, image: np.array, **kwargs) -> np.array:
        """Colour the regions.

        Parameters
        ----------
        image: np.array
            Numpy array representing image.

        Returns
        -------
        np.array
            Numpy array of image with objects coloured.
        """

        coloured_regions = label2rgb(image, **kwargs)
        LOGGER.info(f"[{self.filename}] : Coloured regions")
        return coloured_regions

    @staticmethod
    def get_region_properties(image: np.array, **kwargs) -> List:
        """Extract the properties of each region.

        Parameters
        ----------
        image: np.array
            Numpy array representing image

        Returns
        -------
        List
            List of region property objects.
        """
        return regionprops(image, **kwargs)

    def get_bounding_boxes(self, direction) -> Dict:
        """Derive a list of bounding boxes for each region from the derived region_properties

        Parameters
        ----------
        direction: str
            Direction of threshold for which bounding boxes are being calculated.

        Returns
        -------
        dict
            Dictionary of bounding boxes indexed by region area.
        """
        return {region.area: region.area_bbox for region in self.region_properties[direction]}

    def find_grains(self):
        """Find grains."""
        LOGGER.info(f"[{self.filename}] : Thresholding method (grains) : {self.threshold_method}")
        self.thresholds = get_thresholds(
            image=self.image,
            threshold_method=self.threshold_method,
            otsu_threshold_multiplier=self.otsu_threshold_multiplier,
            threshold_std_dev=self.threshold_std_dev,
            absolute=self.threshold_absolute,
        )
        for direction in self.direction:
            LOGGER.info(
                f"[{self.filename}] : Finding {direction} grains, threshold: ({self.thresholds[direction]})"
            )
            self.directions[direction] = {}
            self.directions[direction]["mask_grains"] = _get_mask(
                self.image,
                thresh=self.thresholds[direction],
                threshold_direction=direction,
                img_name=self.filename,
            )
            self.directions[direction]["labelled_regions_01"] = self.label_regions(
                self.directions[direction]["mask_grains"]
            )

            if self.remove_edge_intersecting_grains:
                self.directions[direction]["tidied_border"] = self.tidy_border(
                    self.directions[direction]["labelled_regions_01"]
                )
            else:
                self.directions[direction]["tidied_border"] = self.directions[direction][
                    "labelled_regions_01"
                ]

            LOGGER.info(f"[{self.filename}] : Removing noise ({direction})")
            self.directions[direction]["removed_noise"] = self.area_thresholding(
                self.directions[direction]["tidied_border"],
                [self.smallest_grain_size_nm2, None],
            )

            LOGGER.info(f"[{self.filename}] : Removing small / large grains ({direction})")
            # if no area thresholds specified, use otsu
            if self.absolute_area_threshold[direction].count(None) == 2:
                self.calc_minimum_grain_size(self.directions[direction]["removed_noise"])
                self.directions[direction]["removed_small_objects"] = self.remove_small_objects(
                    self.directions[direction]["removed_noise"]
                )
            else:
                self.directions[direction]["removed_small_objects"] = self.area_thresholding(
                    self.directions[direction]["removed_noise"],
                    self.absolute_area_threshold[direction],
                )
            self.directions[direction]["labelled_regions_02"] = self.label_regions(
                self.directions[direction]["removed_small_objects"]
            )

            self.region_properties[direction] = self.get_region_properties(
                self.directions[direction]["labelled_regions_02"]
            )
            LOGGER.info(f"[{self.filename}] : Region properties calculated ({direction})")
            self.directions[direction]["coloured_regions"] = self.colour_regions(
                self.directions[direction]["labelled_regions_02"]
            )
            self.bounding_boxes[direction] = self.get_bounding_boxes(direction=direction)
            LOGGER.info(f"[{self.filename}] : Extracted bounding boxes ({direction})")

            # For each detected molecule, create an image of just that molecule and run the UNet
            # on that image to segment it
            unet_mask = np.zeros_like(self.image)

            # Create image save dir
            IMAGE_SAVE_DIR = Path(self.data_save_dir / "angle_data/" / self.filename)
            IMAGE_SAVE_DIR.mkdir(parents=True, exist_ok=True)

            sample_type = "dna_only"
            LOGGER.info(f"SAMPLE TYPE: {sample_type}")

            if sample_type == "dna_only":
                # Get the path to a file in the topostats package
                model_path = (
                    Path(__file__).parent
                    / "haribonet_single_class_2023-12-20_10-44-01_image-size-256x256_epochs-30_batch-size-32_learning-rate-0.001.h5"
                )

                LOGGER.info(f"Loading Unet model: {model_path.stem}")
                model = load_model(model_path=model_path)
                LOGGER.info(f"Loaded Unet model: {model_path.stem}")

                for grain_number, region in enumerate(self.region_properties[direction]):
                    LOGGER.info(
                        f"Unet predicting mask for grain {grain_number} of {len(self.region_properties[direction])}"
                    )

                    # Get the bounding box for the region
                    bounding_box = np.array(region.bbox)

                    # Make the bounding box square within the confines of the image
                    # Calculate the width and height of the bounding box
                    LOGGER.info(
                        f"bounding_box: [0]: {bounding_box[0]} [1]: {bounding_box[1]} [2]: {bounding_box[2]} [3]:"
                        f"{bounding_box[3]}"
                    )
                    width = bounding_box[3] - bounding_box[1]
                    height = bounding_box[2] - bounding_box[0]

                    # Pad the bounding box by 20% if it fits within the image
                    if bounding_box[0] - (height * 0.2) >= 0:
                        # Expand up
                        bounding_box[0] -= height * 0.2
                    if bounding_box[1] - (width * 0.2) >= 0:
                        # Expand left
                        bounding_box[1] -= width * 0.2
                    if bounding_box[2] + (height * 0.2) <= self.image.shape[0]:
                        # Expand down
                        bounding_box[2] += height * 0.2
                    if bounding_box[3] + (width * 0.2) <= self.image.shape[1]:
                        # Expand right
                        bounding_box[3] += width * 0.2

                    width = bounding_box[3] - bounding_box[1]
                    height = bounding_box[2] - bounding_box[0]

                    # # Plot the cropped region for testing
                    # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                    # ax.imshow(
                    #     self.image[bounding_box[0] : bounding_box[2], bounding_box[1] : bounding_box[3]]
                    # )
                    # fig.tight_layout()
                    # plt.savefig(f"{self.filename}_grain_{grain_number}_cropped.png")

                    # Make the width and height the same
                    if width > height:
                        # Make the height the same as the width
                        difference = width - height
                        # Check which direction to expand the bounding box
                        # Check if can expand up
                        if bounding_box[0] - difference >= 0:
                            # Expand up
                            bounding_box[0] -= difference
                        else:
                            # Expand down
                            bounding_box[2] += difference

                    elif height > width:
                        # Make the width the same as the height
                        difference = height - width
                        # Check which direction to expand the bounding box
                        # Check if can expand left
                        if bounding_box[1] - difference >= 0:
                            # Expand left
                            bounding_box[1] -= difference
                        else:
                            # Expand right
                            bounding_box[3] += difference

                    LOGGER.info(
                        f"Bounding box shape: width: {bounding_box[3] - bounding_box[1]} height: {bounding_box[2] - bounding_box[0]}"
                    )

                    # Get the image of just the region
                    region_image = self.image[
                        bounding_box[0] : bounding_box[2], bounding_box[1] : bounding_box[3]
                    ]

                    LOGGER.info(f"Region image shape: {region_image.shape}")

                    # Run the UNet on the region
                    predicted_mask = predict_unet(
                        image=region_image,
                        model=model,
                        confidence=0.5,
                        model_image_size=256,
                        image_output_dir=Path("./"),
                        filename=self.filename + f"_grain_{grain_number}",
                    )

                    LOGGER.info(f"Predicted mask shape: {predicted_mask.shape}")

                    # Plot region image and predicted mask
                    # fig, ax = plt.subplots(1, 2, figsize=(20, 7))
                    # ax[0].imshow(region_image)
                    # ax[0].set_title("region image")
                    # ax[1].imshow(predicted_mask)
                    # ax[1].set_title("predicted mask")
                    # fig.tight_layout()
                    # plt.savefig(f"{self.filename}_grain_{grain_number}_predicted_mask.png")

                    LOGGER.info(f"bbox 2 - 0: {bounding_box[2] - bounding_box[0]}")
                    LOGGER.info(f"bbox 3 - 1: {bounding_box[3] - bounding_box[1]}")

                    # Add the predicted mask to the overall mask
                    unet_mask[
                        bounding_box[0] : bounding_box[2], bounding_box[1] : bounding_box[3]
                    ] = np.logical_or(
                        unet_mask[
                            bounding_box[0] : bounding_box[2], bounding_box[1] : bounding_box[3]
                        ],
                        predicted_mask,
                    )

                    self.directions[direction]["removed_small_objects"] = unet_mask
                    unet_labelled_regions = self.label_regions(unet_mask)
                    self.directions[direction]["labelled_regions_02"] = unet_labelled_regions

            elif sample_type == "dna_protein":
                # Get the path to a file in the topostats package
                model_path = (
                    Path(__file__).parent
                    / "haribonet_multiclass_2023-10-20_14-01-15_intial_results_multiclass_cropped.h5"
                )

                LOGGER.info(f"Loading Unet model: {model_path.stem}")
                model = load_model(model_path=model_path)
                LOGGER.info(f"Loaded Unet model: {model_path.stem}")

                angles = []

                for grain_number, region in enumerate(self.region_properties[direction]):
                    LOGGER.info(
                        f"Unet predicting mask for grain {grain_number} of {len(self.region_properties[direction])}"
                    )

                    # Get the bounding box for the region
                    bounding_box = np.array(region.bbox)

                    # Make the bounding box square within the confines of the image
                    # Calculate the width and height of the bounding box
                    LOGGER.info(
                        f"bounding_box: [0]: {bounding_box[0]} [1]: {bounding_box[1]} [2]: {bounding_box[2]} [3]:"
                        f"{bounding_box[3]}"
                    )
                    width = bounding_box[3] - bounding_box[1]
                    height = bounding_box[2] - bounding_box[0]

                    # Pad the bounding box by 20% if it fits within the image
                    if bounding_box[0] - (height * 0.2) >= 0:
                        # Expand up
                        bounding_box[0] -= height * 0.2
                    if bounding_box[1] - (width * 0.2) >= 0:
                        # Expand left
                        bounding_box[1] -= width * 0.2
                    if bounding_box[2] + (height * 0.2) <= self.image.shape[0]:
                        # Expand down
                        bounding_box[2] += height * 0.2
                    if bounding_box[3] + (width * 0.2) <= self.image.shape[1]:
                        # Expand right
                        bounding_box[3] += width * 0.2

                    width = bounding_box[3] - bounding_box[1]
                    height = bounding_box[2] - bounding_box[0]

                    # # Plot the cropped region for testing
                    # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                    # ax.imshow(
                    #     self.image[bounding_box[0] : bounding_box[2], bounding_box[1] : bounding_box[3]]
                    # )
                    # fig.tight_layout()
                    # plt.savefig(f"{self.filename}_grain_{grain_number}_cropped.png")

                    # Make the width and height the same
                    if width > height:
                        # Make the height the same as the width
                        difference = width - height
                        # Check which direction to expand the bounding box
                        # Check if can expand up
                        if bounding_box[0] - difference >= 0:
                            # Expand up
                            bounding_box[0] -= difference
                        else:
                            # Expand down
                            bounding_box[2] += difference

                    elif height > width:
                        # Make the width the same as the height
                        difference = height - width
                        # Check which direction to expand the bounding box
                        # Check if can expand left
                        if bounding_box[1] - difference >= 0:
                            # Expand left
                            bounding_box[1] -= difference
                        else:
                            # Expand right
                            bounding_box[3] += difference

                    LOGGER.info(
                        f"Bounding box shape: width: {bounding_box[3] - bounding_box[1]} height: {bounding_box[2] - bounding_box[0]}"
                    )

                    # Get the image of just the region
                    region_image = self.image[
                        bounding_box[0] : bounding_box[2], bounding_box[1] : bounding_box[3]
                    ]

                    LOGGER.info(f"Region image shape: {region_image.shape}")

                    # Run the UNet on the region
                    try:
                        predicted_mask, angle = predict_unet_multiclass_and_get_angle(
                            image=region_image,
                            model=model,
                            confidence=0.5,
                            model_image_size=512,
                            image_output_dir=Path("./"),
                            filename=self.filename + f"_grain_{grain_number}",
                            IMAGE_SAVE_DIR=IMAGE_SAVE_DIR,
                            image_index=grain_number,
                        )
                    except ValueError as e:
                        # Check if "found array witn 0 sample(s)" in error message
                        if "Found array with 0 sample(s)" in str(e):
                            # If so, skip this grain
                            LOGGER.info(
                                f"Angle calculation failed: k means 0 samples. Skipping grain {grain_number}"
                            )
                            continue
                        else:
                            raise e

                    angles.append(angle)

                    LOGGER.info(f"Predicted mask shape: {predicted_mask.shape}")

                    # Plot region image and predicted mask
                    # fig, ax = plt.subplots(1, 2, figsize=(20, 7))
                    # ax[0].imshow(region_image)
                    # ax[0].set_title("region image")
                    # ax[1].imshow(predicted_mask)
                    # ax[1].set_title("predicted mask")
                    # fig.tight_layout()
                    # plt.savefig(f"{self.filename}_grain_{grain_number}_predicted_mask.png")

                    LOGGER.info(f"bbox 2 - 0: {bounding_box[2] - bounding_box[0]}")
                    LOGGER.info(f"bbox 3 - 1: {bounding_box[3] - bounding_box[1]}")

                    # Add the predicted mask to the overall mask
                    unet_mask[
                        bounding_box[0] : bounding_box[2], bounding_box[1] : bounding_box[3]
                    ] = np.logical_or(
                        unet_mask[
                            bounding_box[0] : bounding_box[2], bounding_box[1] : bounding_box[3]
                        ],
                        predicted_mask,
                    )

                    self.directions[direction]["removed_small_objects"] = unet_mask
                    unet_labelled_regions = self.label_regions(unet_mask)
                    self.directions[direction]["labelled_regions_02"] = unet_labelled_regions

                # Save the angles
                np.save(IMAGE_SAVE_DIR / "angles.npy", np.array(angles))
