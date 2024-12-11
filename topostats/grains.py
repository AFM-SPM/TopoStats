"""Find grains in an image."""

# pylint: disable=no-name-in-module
from __future__ import annotations

import logging
import sys
from collections import defaultdict

import keras
import numpy as np
import numpy.typing as npt
from skimage import morphology
from skimage.color import label2rgb
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation
from skimage.segmentation import clear_border

from topostats.logs.logs import LOGGER_NAME
from topostats.thresholds import threshold
from topostats.unet_masking import (
    iou_loss,
    make_bounding_box_square,
    mean_iou,
    pad_bounding_box,
    predict_unet,
)
from topostats.utils import _get_mask, get_thresholds

LOGGER = logging.getLogger(LOGGER_NAME)

# pylint: disable=fixme
# pylint: disable=line-too-long
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-arguments
# pylint: disable=bare-except
# pylint: disable=dangerous-default-value
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-lines
# pylint: disable=too-many-public-methods


class Grains:
    """
    Find grains in an image.

    Parameters
    ----------
    image : npt.NDArray
        2-D Numpy array of image.
    filename : str
        File being processed (used in logging).
    pixel_to_nm_scaling : float
        Scaling of pixels to nanometres.
    unet_config : dict[str, str | int | float | tuple[int | None, int, int, int] | None]
        Configuration for the UNet model.
        model_path: str
            Path to the UNet model.
        grain_crop_padding: int
            Padding to add to the bounding box of the grain before cropping.
        upper_norm_bound: float
            Upper bound for normalising the image.
        lower_norm_bound: float
            Lower bound for normalising the image.
    threshold_method : str
        Method for determining thershold to mask values, default is 'otsu'.
    otsu_threshold_multiplier : float
        Factor by which the below threshold is to be scaled prior to masking.
    threshold_std_dev : dict
        Dictionary of 'below' and 'above' factors by which standard deviation is multiplied to derive the threshold
        if threshold_method is 'std_dev'.
    threshold_absolute : dict
        Dictionary of absolute 'below' and 'above' thresholds for grain finding.
    absolute_area_threshold : dict
        Dictionary of above and below grain's area thresholds.
    direction : str
        Direction for which grains are to be detected, valid values are 'above', 'below' and 'both'.
    smallest_grain_size_nm2 : float
        Whether or not to remove grains that intersect the edge of the image.
    remove_edge_intersecting_grains : bool
        Direction for which grains are to be detected, valid values are 'above', 'below' and 'both'.
    classes_to_merge : list[tuple[int, int]] | None
        List of tuples of classes to merge.
    vetting : dict | None
        Dictionary of vetting parameters.
    """

    def __init__(
        self,
        image: npt.NDArray,
        filename: str,
        pixel_to_nm_scaling: float,
        unet_config: dict[str, str | int | float | tuple[int | None, int, int, int] | None] | None = None,
        threshold_method: str | None = None,
        otsu_threshold_multiplier: float | None = None,
        threshold_std_dev: dict | None = None,
        threshold_absolute: dict | None = None,
        absolute_area_threshold: dict | None = None,
        direction: str | None = None,
        smallest_grain_size_nm2: float | None = None,
        remove_edge_intersecting_grains: bool = True,
        classes_to_merge: list[tuple[int, int]] | None = None,
        vetting: dict | None = None,
    ):
        """
        Initialise the class.

        Parameters
        ----------
        image : npt.NDArray
            2-D Numpy array of image.
        filename : str
            File being processed (used in logging).
        pixel_to_nm_scaling : float
            Scaling of pixels to nanometres.
        unet_config : dict[str, str | int | float | tuple[int | None, int, int, int] | None]
            Configuration for the UNet model.
            model_path: str
                Path to the UNet model.
            grain_crop_padding: int
                Padding to add to the bounding box of the grain before cropping.
            upper_norm_bound: float
                Upper bound for normalising the image.
            lower_norm_bound: float
                Lower bound for normalising the image.
        threshold_method : str
            Method for determining thershold to mask values, default is 'otsu'.
        otsu_threshold_multiplier : float
            Factor by which the below threshold is to be scaled prior to masking.
        threshold_std_dev : dict
            Dictionary of 'below' and 'above' factors by which standard deviation is multiplied to derive the threshold
            if threshold_method is 'std_dev'.
        threshold_absolute : dict
            Dictionary of absolute 'below' and 'above' thresholds for grain finding.
        absolute_area_threshold : dict
            Dictionary of above and below grain's area thresholds.
        direction : str
            Direction for which grains are to be detected, valid values are 'above', 'below' and 'both'.
        smallest_grain_size_nm2 : float
            Whether or not to remove grains that intersect the edge of the image.
        remove_edge_intersecting_grains : bool
            Direction for which grains are to be detected, valid values are 'above', 'below' and 'both'.
        classes_to_merge : list[tuple[int, int]] | None
            List of tuples of classes to merge.
        vetting : dict | None
            Dictionary of vetting parameters.
        """
        if unet_config is None:
            unet_config = {
                "model_path": None,
                "grain_crop_padding": 0,
                "upper_norm_bound": 1.0,
                "lower_norm_bound": 0.0,
            }
        if absolute_area_threshold is None:
            absolute_area_threshold = {"above": [None, None], "below": [None, None]}
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
        self.unet_config = unet_config
        self.vetting = vetting
        self.classes_to_merge = classes_to_merge

        # Hardcoded minimum pixel size for grains. This should not be able to be changed by the user as this is
        # determined by what is processable by the rest of the pipeline.
        self.minimum_grain_size_px = 10
        self.minimum_bbox_size_px = 5

    def tidy_border(self, image: npt.NDArray, **kwargs) -> npt.NDArray:
        """
        Remove grains touching the border.

        Parameters
        ----------
        image : npt.NDarray
            2-D Numpy array representing the image.
        **kwargs
            Arguments passed to 'skimage.segmentation.clear_border(**kwargs)'.

        Returns
        -------
        npt.NDarray
            2-D Numpy array of image without objects touching the border.
        """
        LOGGER.debug(f"[{self.filename}] : Tidying borders")
        return clear_border(image, **kwargs)

    @staticmethod
    def label_regions(image: npt.NDArray, background: int = 0) -> npt.NDArray:
        """
        Label regions.

        This method is used twice, once prior to removal of small regions and again afterwards which is why an image
        must be supplied rather than using 'self'.

        Parameters
        ----------
        image : npt.NDArray
            2-D Numpy array of image.
        background : int
            Value used to indicate background of image. Default = 0.

        Returns
        -------
        npt.NDArray
            2-D Numpy array of image with regions numbered.
        """
        return morphology.label(image, background)

    def calc_minimum_grain_size(self, image: npt.NDArray) -> float:
        """
        Calculate the minimum grain size in pixels squared.

        Very small objects are first removed via thresholding before calculating the below extreme.

        Parameters
        ----------
        image : npt.NDArray
            2-D Numpy image from which to calculate the minimum grain size.

        Returns
        -------
        float
            Minimum grains size in pixels squared. If there are areas a value of -1 is returned.
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

    def remove_noise(self, image: npt.NDArray, **kwargs) -> npt.NDArray:
        """
        Remove noise which are objects smaller than the 'smallest_grain_size_nm2'.

        This ensures that the smallest objects ~1px are removed regardless of the size distribution of the grains.

        Parameters
        ----------
        image : npt.NDArray
            2-D Numpy array to be cleaned.
        **kwargs
            Arguments passed to 'skimage.morphology.remove_small_objects(**kwargs)'.

        Returns
        -------
        npt.NDArray
            2-D Numpy array of image with objects < smallest_grain_size_nm2 removed.
        """
        LOGGER.debug(
            f"[{self.filename}] : Removing noise (< {self.smallest_grain_size_nm2} nm^2"
            "{self.smallest_grain_size_nm2 / (self.pixel_to_nm_scaling**2):.2f} px^2)"
        )
        return morphology.remove_small_objects(
            image, min_size=self.smallest_grain_size_nm2 / (self.pixel_to_nm_scaling**2), **kwargs
        )

    def remove_small_objects(self, image: np.array, **kwargs) -> npt.NDArray:
        """
        Remove small objects from the input image.

        Threshold determined by the minimum grain size, in pixels squared, of the classes initialisation.

        Parameters
        ----------
        image : np.array
            2-D Numpy array to remove small objects from.
        **kwargs
            Arguments passed to 'skimage.morphology.remove_small_objects(**kwargs)'.

        Returns
        -------
        npt.NDArray
            2-D Numpy array of image with objects < minimumm_grain_size removed.
        """
        # If self.minimum_grain_size is -1, then this means that
        # there were no grains to calculate the minimum grian size from.
        if self.minimum_grain_size != -1:
            small_objects_removed = morphology.remove_small_objects(
                image.astype(bool),
                min_size=self.minimum_grain_size,  # minimum_grain_size is in pixels squared
                **kwargs,
            )
            LOGGER.debug(
                f"[{self.filename}] : Removed small objects (< \
{self.minimum_grain_size} px^2 / {self.minimum_grain_size / (self.pixel_to_nm_scaling)**2} nm^2)"
            )
            return small_objects_removed > 0.0
        return image

    def remove_objects_too_small_to_process(
        self, image: npt.NDArray, minimum_size_px: int, minimum_bbox_size_px: int
    ) -> npt.NDArray[np.bool_]:
        """
        Remove objects whose dimensions in pixels are too small to process.

        Parameters
        ----------
        image : npt.NDArray
            2-D Numpy array of image.
        minimum_size_px : int
            Minimum number of pixels for an object.
        minimum_bbox_size_px : int
            Limit for the minimum dimension of an object in pixels. Eg: 5 means the object's bounding box must be at
            least 5x5.

        Returns
        -------
        npt.NDArray
            2-D Numpy array of image with objects removed that are too small to process.
        """
        labelled_image = label(image)
        region_properties = self.get_region_properties(labelled_image)
        for region in region_properties:
            # If the number of true pixels in the region is less than the minimum number of pixels, remove the region
            if region.area < minimum_size_px:
                labelled_image[labelled_image == region.label] = 0
            bbox_width = region.bbox[2] - region.bbox[0]
            bbox_height = region.bbox[3] - region.bbox[1]
            # If the minimum dimension of the bounding box is less than the minimum dimension, remove the region
            if min(bbox_width, bbox_height) < minimum_bbox_size_px:
                labelled_image[labelled_image == region.label] = 0

        return labelled_image.astype(bool)

    def area_thresholding(self, image: npt.NDArray, area_thresholds: tuple) -> npt.NDArray:
        """
        Remove objects larger and smaller than the specified thresholds.

        Parameters
        ----------
        image : npt.NDArray
            Image array where the background == 0 and grains are labelled as integers >0.
        area_thresholds : tuple
            List of area thresholds (in nanometres squared, not pixels squared), first is the lower limit for size,
            second is the upper.

        Returns
        -------
        npt.NDArray
            Array with small and large objects removed.
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
        LOGGER.debug(
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

    def colour_regions(self, image: npt.NDArray, **kwargs) -> npt.NDArray:
        """
        Colour the regions.

        Parameters
        ----------
        image : npt.NDArray
            2-D array of labelled regions to be coloured.
        **kwargs
            Arguments passed to 'skimage.color.label2rgb(**kwargs)'.

        Returns
        -------
        np.array
            Numpy array of image with objects coloured.
        """
        coloured_regions = label2rgb(image, **kwargs)
        LOGGER.debug(f"[{self.filename}] : Coloured regions")
        return coloured_regions

    @staticmethod
    def get_region_properties(image: np.array, **kwargs) -> list:
        """
        Extract the properties of each region.

        Parameters
        ----------
        image : np.array
            Numpy array representing image.
        **kwargs :
            Arguments passed to 'skimage.measure.regionprops(**kwargs)'.

        Returns
        -------
        list
            List of region property objects.
        """
        return regionprops(image, **kwargs)

    def get_bounding_boxes(self, direction: str) -> dict:
        """
        Derive a list of bounding boxes for each region from the derived region_properties.

        Parameters
        ----------
        direction : str
            Direction of threshold for which bounding boxes are being calculated.

        Returns
        -------
        dict
            Dictionary of bounding boxes indexed by region area.
        """
        return {region.area: region.area_bbox for region in self.region_properties[direction]}

    def find_grains(self):
        """Find grains."""
        LOGGER.debug(f"[{self.filename}] : Thresholding method (grains) : {self.threshold_method}")
        self.thresholds = get_thresholds(
            image=self.image,
            threshold_method=self.threshold_method,
            otsu_threshold_multiplier=self.otsu_threshold_multiplier,
            threshold_std_dev=self.threshold_std_dev,
            absolute=self.threshold_absolute,
        )

        for direction in self.direction:
            LOGGER.debug(f"[{self.filename}] : Finding {direction} grains, threshold: ({self.thresholds[direction]})")
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
                self.directions[direction]["tidied_border"] = self.directions[direction]["labelled_regions_01"]

            LOGGER.debug(f"[{self.filename}] : Removing noise ({direction})")
            self.directions[direction]["removed_noise"] = self.area_thresholding(
                self.directions[direction]["tidied_border"],
                [self.smallest_grain_size_nm2, None],
            )

            LOGGER.debug(f"[{self.filename}] : Removing small / large grains ({direction})")
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
            self.directions[direction]["removed_objects_too_small_to_process"] = (
                self.remove_objects_too_small_to_process(
                    image=self.directions[direction]["removed_small_objects"],
                    minimum_size_px=self.minimum_grain_size_px,
                    minimum_bbox_size_px=self.minimum_bbox_size_px,
                )
            )
            self.directions[direction]["labelled_regions_02"] = self.label_regions(
                self.directions[direction]["removed_objects_too_small_to_process"]
            )

            self.region_properties[direction] = self.get_region_properties(
                self.directions[direction]["labelled_regions_02"]
            )
            LOGGER.debug(f"[{self.filename}] : Region properties calculated ({direction})")
            self.directions[direction]["coloured_regions"] = self.colour_regions(
                self.directions[direction]["labelled_regions_02"]
            )
            self.bounding_boxes[direction] = self.get_bounding_boxes(direction=direction)
            LOGGER.debug(f"[{self.filename}] : Extracted bounding boxes ({direction})")
            thresholding_grain_count = self.directions[direction]["labelled_regions_02"].max()

            # Force labelled_regions_02 to be of shape NxNx2, where the two classes are a binary background mask and the second is a binary grain mask.
            # This is because we want to support multiple classes, and so we standardise so that the first layer is background mask, then feature mask 1, then feature mask 2 etc.

            # Get a binary mask where 1s are background and 0s are grains
            labelled_regions_background_mask = np.where(self.directions[direction]["labelled_regions_02"] == 0, 1, 0)
            # keep only the largest region
            labelled_regions_background_mask = label(labelled_regions_background_mask)
            areas = [region.area for region in regionprops(labelled_regions_background_mask)]
            labelled_regions_background_mask = np.where(
                labelled_regions_background_mask == np.argmax(areas) + 1, labelled_regions_background_mask, 0
            )

            self.directions[direction]["labelled_regions_02"] = np.stack(
                [
                    labelled_regions_background_mask,
                    self.directions[direction]["labelled_regions_02"],
                ],
                axis=-1,
            ).astype(
                np.int32
            )  # Will produce an NxNx2 array

            # Do the same for removed_small_objects, using the same labelled_regions_backgroudn_mask as the background since they will be the same
            self.directions[direction]["removed_small_objects"] = np.stack(
                [
                    labelled_regions_background_mask,
                    self.directions[direction]["removed_small_objects"],
                ],
                axis=-1,
            ).astype(
                np.int32
            )  # Will produce an NxNx2 array

            # Check whether to run the UNet model
            if self.unet_config["model_path"] is not None:

                # Run unet segmentation on only the class 1 layer of the labelled_regions_02. Need to make this configurable
                # later on along with all the other hardcoded class 1s.
                unet_mask, unet_labelled_regions = Grains.improve_grain_segmentation_unet(
                    filename=self.filename,
                    direction=direction,
                    unet_config=self.unet_config,
                    image=self.image,
                    labelled_grain_regions=self.directions[direction]["labelled_regions_02"][:, :, 1],
                )

                # Update the image masks to be the unet masks instead
                self.directions[direction]["removed_small_objects"] = unet_mask
                self.directions[direction]["labelled_regions_02"] = unet_labelled_regions

                class_counts = [
                    unet_labelled_regions[class_idx].max() for class_idx in range(unet_labelled_regions.shape[2])
                ]
                LOGGER.info(
                    f"[{self.filename}] : Overridden {thresholding_grain_count} grains with {class_counts} UNet predictions ({direction})"
                )

            # Vet the grains
            if self.vetting is not None:
                vetted_grains = Grains.vet_grains(
                    grain_mask_tensor=self.directions[direction]["labelled_regions_02"].astype(bool),
                    pixel_to_nm_scaling=self.pixel_to_nm_scaling,
                    **self.vetting,
                )
            else:
                vetted_grains = self.directions[direction]["labelled_regions_02"].astype(bool)

            # Merge classes if necessary
            merged_classes = Grains.merge_classes(
                vetted_grains,
                self.classes_to_merge,
            )

            # Update the background class
            final_grains = Grains.update_background_class(grain_mask_tensor=merged_classes)

            # Label each class in the tensor
            labelled_final_grains = np.zeros_like(final_grains).astype(int)
            # The background class will be the same as the binary mask
            labelled_final_grains[:, :, 0] = final_grains[:, :, 0]
            # Iterate over each class and label the regions
            for class_index in range(final_grains.shape[2]):
                labelled_final_grains[:, :, class_index] = Grains.label_regions(final_grains[:, :, class_index])

            self.directions[direction]["removed_small_objects"] = labelled_final_grains.astype(bool)
            self.directions[direction]["labelled_regions_02"] = labelled_final_grains.astype(np.int32)

    # pylint: disable=too-many-locals
    @staticmethod
    def improve_grain_segmentation_unet(
        filename: str,
        direction: str,
        unet_config: dict[str, str | int | float | tuple[int | None, int, int, int] | None],
        image: npt.NDArray,
        labelled_grain_regions: npt.NDArray,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Use a UNet model to re-segment existing grains to improve their accuracy.

        Parameters
        ----------
        filename : str
            File being processed (used in logging).
        direction : str
            Direction of threshold for which bounding boxes are being calculated.
        unet_config : dict[str, str | int | float | tuple[int | None, int, int, int] | None]
            Configuration for the UNet model.
            model_path: str
                Path to the UNet model.
            grain_crop_padding: int
                Padding to add to the bounding box of the grain before cropping.
            upper_norm_bound: float
                Upper bound for normalising the image.
            lower_norm_bound: float
                Lower bound for normalising the image.
        image : npt.NDArray
            2-D Numpy array of image.
        labelled_grain_regions : npt.NDArray
            2-D Numpy array of labelled grain regions.

        Returns
        -------
        npt.NDArray
            NxNxC Numpy array of the UNet mask.
        npt.NDArray
            NxNxC Numpy array of the labelled regions from the UNet mask.
        """
        LOGGER.debug(f"[{filename}] : Running UNet model on {direction} grains")

        # When debugging, you might find that the custom_objects are incorrect. This is entirely based on what the model used
        # for its loss during training and so this will need to be changed a lot.
        # Once the group has gotten used to training models, this can be made configurable, but currently it's too changeable.
        # unet_model = keras.models.load_model(
        #     self.unet_config["model_path"], custom_objects={"dice_loss": dice_loss, "iou_loss": iou_loss}
        # )
        # You may also get an error referencing a "group_1" parameter, this is discussed in this issue:
        # https://github.com/keras-team/keras/issues/19441 which also has an experimental fix that we can try but
        # I haven't tested it yet.

        try:
            unet_model = keras.models.load_model(
                unet_config["model_path"], custom_objects={"mean_iou": mean_iou, "iou_loss": iou_loss}, compile=False
            )
        except Exception as e:
            LOGGER.debug(f"Python executable: {sys.executable}")
            LOGGER.debug(f"Keras version: {keras.__version__}")
            LOGGER.debug(f"Model path: {unet_config['model_path']}")
            raise e

        # unet_model = keras.models.load_model(unet_config["model_path"], custom_objects={"mean_iou": mean_iou})
        LOGGER.debug(f"Output shape of UNet model: {unet_model.output_shape}")

        # Initialise an empty mask to iteratively add to for each grain, with the correct number of class channels based on
        # the loaded model's output shape
        # Note that the minimum number of classes is 2, as even for binary outputs, we will force categorical type
        # data, so we have a class for background.
        unet_mask = np.zeros((image.shape[0], image.shape[1], np.max([2, unet_model.output_shape[-1]]))).astype(
            np.bool_
        )
        # Set the background class to be all 1s by default since not all of the image will be covered by the
        # u-net predictions.
        unet_mask[:, :, 0] = 1
        # Labelled regions will be the same by default, but will be overwritten if there are any grains present.
        unet_labelled_regions = np.zeros_like(unet_mask).astype(np.int32)
        # For each detected molecule, create an image of just that molecule and run the UNet
        # on that image to segment it
        grain_region_properties = regionprops(labelled_grain_regions)
        for grain_number, region in enumerate(grain_region_properties):
            LOGGER.debug(f"Unet predicting mask for grain {grain_number} of {len(grain_region_properties)}")

            # Get the bounding box for the region
            bounding_box: tuple[int, int, int, int] = tuple(region.bbox)  # min_row, min_col, max_row, max_col

            # Pad the bounding box
            bounding_box = pad_bounding_box(
                crop_min_row=bounding_box[0],
                crop_min_col=bounding_box[1],
                crop_max_row=bounding_box[2],
                crop_max_col=bounding_box[3],
                image_shape=(image.shape[0], image.shape[1]),
                padding=unet_config["grain_crop_padding"],
            )

            # Make the bounding box square within the confines of the image
            if (bounding_box[2] - bounding_box[0]) != (bounding_box[3] - bounding_box[1]):
                bounding_box = make_bounding_box_square(
                    crop_min_row=bounding_box[0],
                    crop_min_col=bounding_box[1],
                    crop_max_row=bounding_box[2],
                    crop_max_col=bounding_box[3],
                    image_shape=(image.shape[0], image.shape[1]),
                )

            # Grab the cropped image. Using slice since the bounding box from skimage is
            # half-open, so the max_row and max_col are not included in the region.
            region_image = image[
                bounding_box[0] : bounding_box[2],
                bounding_box[1] : bounding_box[3],
            ]

            # Run the UNet on the region. This is allowed to be a single channel
            # as we can add a background channel afterwards if needed.
            # Remember that this region is cropped from the original image, so it's not
            # the same size as the original image.
            predicted_mask = predict_unet(
                image=region_image,
                model=unet_model,
                confidence=0.1,
                model_input_shape=unet_model.input_shape,
                upper_norm_bound=unet_config["upper_norm_bound"],
                lower_norm_bound=unet_config["lower_norm_bound"],
            )

            assert len(predicted_mask.shape) == 3
            LOGGER.debug(f"Predicted mask shape: {predicted_mask.shape}")

            # Add each class of the predicted mask to the overall full image mask
            for class_index in range(unet_mask.shape[2]):

                # Grab the unet mask for the class
                unet_predicted_mask_labelled = morphology.label(predicted_mask[:, :, class_index])

                # Directly set the background to be equal instead of logical or since they are by default
                # 1, and should be turned off if any other class is on
                if class_index == 0:
                    unet_mask[
                        bounding_box[0] : bounding_box[2],
                        bounding_box[1] : bounding_box[3],
                        class_index,
                    ] = unet_predicted_mask_labelled
                else:
                    unet_mask[
                        bounding_box[0] : bounding_box[2],
                        bounding_box[1] : bounding_box[3],
                        class_index,
                    ] = np.logical_or(
                        unet_mask[
                            bounding_box[0] : bounding_box[2],
                            bounding_box[1] : bounding_box[3],
                            class_index,
                        ],
                        unet_predicted_mask_labelled,
                    )

            assert len(unet_mask.shape) == 3, f"Unet mask shape: {unet_mask.shape}"
            assert unet_mask.shape[-1] >= 2, f"Unet mask shape: {unet_mask.shape}"

            # For each class in the unet mask tensor, label the mask and add to unet_labelled_regions
            # The labelled background class will be identical to the binary one from the unet mask.
            unet_labelled_regions[:, :, 0] = unet_mask[:, :, 0]
            # Iterate over each class and label the regions
            for class_index in range(unet_mask.shape[2]):
                unet_labelled_regions[:, :, class_index] = Grains.label_regions(unet_mask[:, :, class_index])

        return unet_mask, unet_labelled_regions

    @staticmethod
    def keep_largest_labelled_region(
        labelled_image: npt.NDArray[np.int32],
    ) -> npt.NDArray[np.bool_]:
        """
        Keep only the largest region in a labelled image.

        Parameters
        ----------
        labelled_image : npt.NDArray
            2-D Numpy array of labelled regions.

        Returns
        -------
        npt.NDArray
            2-D Numpy boolean array of labelled regions with only the largest region.
        """
        # Check if there are any labelled regions
        if labelled_image.max() == 0:
            return np.zeros_like(labelled_image).astype(np.bool_)
        # Get the sizes of the regions
        sizes = np.array([(labelled_image == label).sum() for label in range(1, labelled_image.max() + 1)])
        # Keep only the largest region
        return np.where(labelled_image == sizes.argmax() + 1, labelled_image, 0).astype(bool)

    @staticmethod
    def flatten_multi_class_tensor(grain_mask_tensor: npt.NDArray) -> npt.NDArray:
        """
        Flatten a multi-class image tensor to a single binary mask.

        The returned tensor is of boolean type in case there are multiple hits in the same pixel. We dont want to have
        2s, 3s etc because this would cause issues in labelling and cause erroneous grains within grains.

        Parameters
        ----------
        grain_mask_tensor : npt.NDArray
            Multi class grain mask tensor tensor of shape (N, N, C).

        Returns
        -------
        npt.NDArray
            Combined binary mask of all but the background class (:, :, 0).
        """
        assert len(grain_mask_tensor.shape) == 3, f"Tensor not 3D: {grain_mask_tensor.shape}"
        return np.sum(grain_mask_tensor[:, :, 1:], axis=-1).astype(bool)

    @staticmethod
    def get_multi_class_grain_bounding_boxes(grain_mask_tensor: npt.NDArray) -> dict:
        """
        Get the bounding boxes for each grain in a multi-class image tensor.

        Finds the bounding boxes for each grain in a multi-class image tensor. Grains can span multiple classes, so the
        bounding boxes are found for the combined binary mask of contiguous grains across all classes.

        Parameters
        ----------
        grain_mask_tensor : npt.NDArray
            3-D Numpy array of grain mask tensor.

        Returns
        -------
        dict
            Dictionary of bounding boxes indexed by grain number.
        """
        flattened_mask = Grains.flatten_multi_class_tensor(grain_mask_tensor)
        labelled_regions = Grains.label_regions(flattened_mask)
        region_properties = Grains.get_region_properties(labelled_regions)
        bounding_boxes = {index: region.bbox for index, region in enumerate(region_properties)}
        return {
            index: pad_bounding_box(
                crop_min_row=bbox[0],
                crop_min_col=bbox[1],
                crop_max_row=bbox[2],
                crop_max_col=bbox[3],
                image_shape=(grain_mask_tensor.shape[0], grain_mask_tensor.shape[1]),
                padding=1,
            )
            for index, bbox in bounding_boxes.items()
        }

    @staticmethod
    def update_background_class(
        grain_mask_tensor: npt.NDArray,
    ) -> npt.NDArray:
        """
        Update the background class to reflect the other classes.

        Parameters
        ----------
        grain_mask_tensor : npt.NDArray
            3-D Numpy array of the grain mask tensor.

        Returns
        -------
        npt.NDArray
            3-D Numpy array of image tensor with updated background class.
        """
        flattened_mask = Grains.flatten_multi_class_tensor(grain_mask_tensor)
        new_background = np.where(flattened_mask == 0, 1, 0)
        grain_mask_tensor[:, :, 0] = new_background
        return grain_mask_tensor.astype(bool)

    @staticmethod
    def vet_class_sizes_single_grain(
        single_grain_mask_tensor: npt.NDArray,
        pixel_to_nm_scaling: float,
        class_size_thresholds: list[tuple[int, int, int]] | None,
    ) -> tuple[npt.NDArray, bool]:
        """
        Remove regions of particular classes based on size thresholds.

        Regions of classes that are too large or small may need to be removed for many reasons (eg removing noise
        erroneously detected by the model or larger-than-expected molecules that are obviously erroneous), this method
        allows for the removal of these regions based on size thresholds.

        Parameters
        ----------
        single_grain_mask_tensor : npt.NDArray
            3-D Numpy array of the mask tensor.
        pixel_to_nm_scaling : float
            Scaling of pixels to nanometres.
        class_size_thresholds : list[list[int, int, int]] | None
            List of class size thresholds. Structure is [(class_index, lower, upper)].

        Returns
        -------
        npt.NDArray
            3-D Numpy array of the mask tensor with grains removed based on size thresholds.
        bool
            True if the grain passes the vetting, False if it fails.
        """
        if class_size_thresholds is None:
            return single_grain_mask_tensor, True

        # Iterate over the classes and check the sizes
        for class_index in range(1, single_grain_mask_tensor.shape[2]):
            class_size = np.sum(single_grain_mask_tensor[:, :, class_index]) * pixel_to_nm_scaling**2
            # Check the size against the thresholds

            classes_to_vet = [vetting_criteria[0] for vetting_criteria in class_size_thresholds]

            if class_index not in classes_to_vet:
                continue

            lower_threshold, upper_threshold = [
                vetting_criteria[1:] for vetting_criteria in class_size_thresholds if vetting_criteria[0] == class_index
            ][0]

            if lower_threshold is not None:
                if class_size < lower_threshold:
                    # Return empty tensor
                    empty_crop_tensor = np.zeros_like(single_grain_mask_tensor)
                    # Fill the background class with 1s
                    empty_crop_tensor[:, :, 0] = 1
                    return empty_crop_tensor, False
            if upper_threshold is not None:
                if class_size > upper_threshold:
                    # Return empty tensor
                    empty_crop_tensor = np.zeros_like(single_grain_mask_tensor)
                    # Fill the background class with 1s
                    empty_crop_tensor[:, :, 0] = 1
                    return empty_crop_tensor, False

        return single_grain_mask_tensor, True

    @staticmethod
    def get_individual_grain_crops(
        grain_mask_tensor: npt.NDArray,
        padding: int = 1,
    ) -> tuple[list[npt.NDArray], list[npt.NDArray], int]:
        """
        Get individual grain crops from an image tensor.

        Fetches individual grain crops from an image tensor, but zeros any non-connected grains
        in the crop region. This is to ensure that other grains do not affect further processing
        steps.

        Parameters
        ----------
        grain_mask_tensor : npt.NDArray
            3-D Numpy array of image tensor.
        padding : int
            Padding to add to the bounding box of the grain before cropping. Default is 1.

        Returns
        -------
        list[npt.NDArray]
            List of individual grain crops.
        list[npt.NDArray]
            List of bounding boxes for each grain.
        int
            Padding used for the bounding boxes.
        """
        grain_crops = []
        bounding_boxes = []

        # Label the regions
        flattened_multi_class_mask = Grains.flatten_multi_class_tensor(grain_mask_tensor)
        labelled_regions = Grains.label_regions(flattened_multi_class_mask)

        # Iterate over the regions and return the crop, but zero any non-connected grains
        for region in Grains.get_region_properties(labelled_regions):

            binary_labelled_regions = labelled_regions == region.label

            # Zero any non-connected grains
            # For each class, set all pixels to zero that are not in the current region
            this_region_only_grain_tensor = np.copy(grain_mask_tensor)
            # Iterate over the non-background classes
            for class_index in range(1, grain_mask_tensor.shape[2]):
                # Set all pixels to zero that are not in the current region
                this_region_only_grain_tensor[:, :, class_index] = (
                    binary_labelled_regions * grain_mask_tensor[:, :, class_index]
                )

            # Update background class to reflect the removal of any non-connected grains
            this_region_only_grain_tensor = Grains.update_background_class(
                grain_mask_tensor=this_region_only_grain_tensor
            )

            # Get the bounding box
            bounding_box = region.bbox

            # Pad the bounding box
            bounding_box = pad_bounding_box(
                crop_min_row=bounding_box[0],
                crop_min_col=bounding_box[1],
                crop_max_row=bounding_box[2],
                crop_max_col=bounding_box[3],
                image_shape=(grain_mask_tensor.shape[0], grain_mask_tensor.shape[1]),
                padding=padding,
            )

            # Crop the grain
            grain_crop = this_region_only_grain_tensor[
                bounding_box[0] : bounding_box[2],
                bounding_box[1] : bounding_box[3],
                :,
            ]

            # Add the crop to the list
            grain_crops.append(grain_crop.astype(bool))
            bounding_boxes.append(bounding_box)

        return grain_crops, bounding_boxes, padding

    @staticmethod
    def vet_numbers_of_regions_single_grain(
        grain_mask_tensor: npt.NDArray,
        class_region_number_thresholds: list[tuple[int, int, int]] | None,
    ) -> tuple[npt.NDArray, bool]:
        """
        Check if the number of regions of different classes for a single grain is within thresholds.

        Parameters
        ----------
        grain_mask_tensor : npt.NDArray
            3-D Numpy array of the grain mask tensor, should be of only one grain.
        class_region_number_thresholds : list[list[int, int, int]]
            List of class region number thresholds. Structure is [(class_index, lower, upper)].

        Returns
        -------
        npt.NDArray
            3-D Numpy array of the grain mask tensor with grains removed based on region number thresholds.
        bool
            True if the grain passes the vetting, False if it fails.
        """
        if class_region_number_thresholds is None:
            return grain_mask_tensor, True

        # Iterate over the classes and check the number of regions
        for class_index in range(1, grain_mask_tensor.shape[2]):
            # Get the number of regions
            class_labelled_regions = Grains.label_regions(grain_mask_tensor[:, :, class_index])
            number_of_regions = np.unique(class_labelled_regions).shape[0] - 1
            # Check the number of regions against the thresholds, skip if no thresholds provided
            # Get the classes we are trying to vet (the first element of each tuple)
            classes_to_vet = [vetting_criteria[0] for vetting_criteria in class_region_number_thresholds]

            if class_index not in classes_to_vet:
                continue

            lower_threshold, upper_threshold = [
                vetting_criteria[1:]
                for vetting_criteria in class_region_number_thresholds
                if vetting_criteria[0] == class_index
            ][0]

            # Check the number of regions against the thresholds
            if lower_threshold is not None:
                if number_of_regions < lower_threshold:
                    # Return empty tensor
                    empty_crop_tensor = np.zeros_like(grain_mask_tensor)
                    # Fill the background class with 1s
                    empty_crop_tensor[:, :, 0] = 1
                    return empty_crop_tensor, False
            if upper_threshold is not None:
                if number_of_regions > upper_threshold:
                    # Return empty tensor
                    empty_crop_tensor = np.zeros_like(grain_mask_tensor)
                    # Fill the background class with 1s
                    empty_crop_tensor[:, :, 0] = 1
                    return empty_crop_tensor, False

        return grain_mask_tensor, True

    @staticmethod
    def convert_classes_to_nearby_classes(
        grain_mask_tensor: npt.NDArray,
        classes_to_convert: list[tuple[int, int]] | None,
        class_touching_threshold: int = 1,
    ) -> npt.NDArray:
        """
        Convert all but the largest regions of one class into another class provided the former touches the latter.

        Specifically, it takes a list of tuples of two integers (dubbed class A and class B). For each class A, class B
        pair, it will find the largest region of class A and flag it to be ignored. Then for each non-largest region of
        class A, it will check if it touches any class B region (within the ``class_touching_threshold`` distance). If it
        does, it will convert the region to class B.

        This is useful for situations where you want just one region of class A and the model has a habit of producing
        small regions of class A interspersed in the class B regions, which should be class B instead.

        Parameters
        ----------
        grain_mask_tensor : npt.NDArray
            3-D Numpy array of the grain mask tensor.
        classes_to_convert : list
            List of tuples of classes to convert. Structure is [(class_a, class_b)].
        class_touching_threshold : int
            Number of dilation passes to do to determine class A connectivity with class B.

        Returns
        -------
        npt.NDArray
            3-D Numpy array of the grain mask tensor with classes converted.
        """
        # If no classes to convert, return the original tensor
        if not classes_to_convert:
            return grain_mask_tensor

        # Iterate over class pairs
        for class_a, class_b in classes_to_convert:
            # Get the binary mask for class A and class B
            class_a_mask = grain_mask_tensor[:, :, class_a]
            class_b_mask = grain_mask_tensor[:, :, class_b]

            # Skip if no regions of class A
            if np.max(class_a_mask) == 0:
                continue

            # Find the largest region of class A
            class_a_labelled_regions = Grains.label_regions(class_a_mask)
            class_a_region_properties = Grains.get_region_properties(class_a_labelled_regions)
            class_a_areas = [region.area for region in class_a_region_properties]
            largest_class_a_region = class_a_region_properties[np.argmax(class_a_areas)]

            # For all other regions, check if they touch the class B region
            for region in class_a_region_properties:
                if region.label == largest_class_a_region.label:
                    continue
                # Get only the pixels in the region
                region_mask = class_a_labelled_regions == region.label
                # Dilate the region
                dilated_region_mask = region_mask
                for _ in range(class_touching_threshold):
                    dilated_region_mask = binary_dilation(dilated_region_mask)
                # Get the intersection with the class B mask
                intersection = dilated_region_mask & class_b_mask
                # If there is any intersection, turn the region into class B
                if np.any(intersection):
                    # Add to the class B mask
                    class_b_mask = np.where(region_mask, class_b, class_b_mask)
                    # Remove from the class A mask
                    class_a_mask = np.where(region_mask, 0, class_a_mask)

            # Update the tensor
            grain_mask_tensor[:, :, class_a] = class_a_mask
            grain_mask_tensor[:, :, class_b] = class_b_mask

        return grain_mask_tensor.astype(bool)

    @staticmethod
    def keep_largest_labelled_region_classes(
        single_grain_mask_tensor: npt.NDArray,
        keep_largest_labelled_regions_classes: list[int] | None,
    ) -> npt.NDArray:
        """
        Keep only the largest region in specific classes.

        Parameters
        ----------
        single_grain_mask_tensor : npt.NDArray
            3-D Numpy array of the grain mask tensor.
        keep_largest_labelled_regions_classes : list[int]
            List of classes to keep only the largest region.

        Returns
        -------
        npt.NDArray
            3-D Numpy array of the grain mask tensor with only the largest regions in specific classes.
        """
        if keep_largest_labelled_regions_classes is None:
            return single_grain_mask_tensor

        # Iterate over the classes
        for class_index in keep_largest_labelled_regions_classes:
            # Get the binary mask for the class
            class_mask = single_grain_mask_tensor[:, :, class_index]

            # Skip if no regions
            if np.max(class_mask) == 0:
                continue

            # Label the regions
            labelled_regions = Grains.label_regions(class_mask)
            # Get the region properties
            region_properties = Grains.get_region_properties(labelled_regions)
            # Get the region areas
            region_areas = [region.area for region in region_properties]
            # Keep only the largest region
            largest_region = region_properties[np.argmax(region_areas)]
            class_mask_largest_only = np.where(labelled_regions == largest_region.label, labelled_regions, 0)
            # Update the tensor
            single_grain_mask_tensor[:, :, class_index] = class_mask_largest_only.astype(bool)

        # Update the background class
        return Grains.update_background_class(single_grain_mask_tensor)

    @staticmethod
    def calculate_region_connection_regions(
        grain_mask_tensor: npt.NDArray,
        classes: tuple[int, int],
    ) -> tuple[int, npt.NDArray, dict[int, npt.NDArray[int]]]:
        """
        Get a list of connection regions between two classes.

        Parameters
        ----------
        grain_mask_tensor : npt.NDArray
            3-D Numpy array of the grain mask tensor.
        classes : tuple[int, int]
            Tuple pair of classes to calculate the connection regions.

        Returns
        -------
        int
            Number of connection regions.
        npt.NDArray
            2-D Numpy array of the intersection labels.
        dict
            Dictionary of connection points indexed by region label.
        """
        # Get the binary masks for the classes
        class_a_mask = grain_mask_tensor[:, :, classes[0]]
        class_b_mask = grain_mask_tensor[:, :, classes[1]]

        # Dilate class A mask
        dilated_class_a_mask = binary_dilation(class_a_mask)
        # Get the intersection with the class B mask
        intersection = dilated_class_a_mask & class_b_mask

        # Get number of separate intersection regions
        intersection_labels = label(intersection)
        intersection_regions = regionprops(intersection_labels)
        num_connection_regions = len(intersection_regions)
        # Create a dictionary of the connection points
        intersection_points = {region.label: region.coords for region in intersection_regions}

        return num_connection_regions, intersection_labels, intersection_points

    @staticmethod
    def vet_class_connection_points(
        grain_mask_tensor: npt.NDArray,
        class_connection_point_thresholds: list[tuple[tuple[int, int], tuple[int, int]]] | None,
    ) -> bool:
        """
        Vet the number of connection points between regions in specific classes.

        Parameters
        ----------
        grain_mask_tensor : npt.NDArray
            3-D Numpy array of the grain mask tensor.
        class_connection_point_thresholds : list[tuple[tuple[int, int], tuple[int, int]]] | None
            List of tuples of classes and connection point thresholds. Structure is [(class_pair, (lower, upper))].

        Returns
        -------
        bool
            True if the grain passes the vetting, False if it fails.
        """
        if class_connection_point_thresholds is None:
            return True

        # Iterate over the class pairs
        for class_pair, connection_point_thresholds in class_connection_point_thresholds:
            # Get the connection regions
            num_connection_regions, _, _ = Grains.calculate_region_connection_regions(
                grain_mask_tensor=grain_mask_tensor,
                classes=class_pair,
            )
            # Check the number of connection regions against the thresholds
            lower_threshold, upper_threshold = connection_point_thresholds
            if lower_threshold is not None:
                if num_connection_regions < lower_threshold:
                    return False
            if upper_threshold is not None:
                if num_connection_regions > upper_threshold:
                    return False

        return True

    @staticmethod
    def assemble_grain_mask_tensor_from_crops(
        grain_mask_tensor_shape: tuple[int, int, int],
        grain_crops_and_bounding_boxes: list[dict[str, npt.NDArray]],
    ) -> npt.NDArray:
        """
        Combine individual grain crops into a single grain mask tensor.

        Parameters
        ----------
        grain_mask_tensor_shape : tuple
            Shape of the grain mask tensor.
        grain_crops_and_bounding_boxes : list
            List of dictionaries containing the grain crops and bounding boxes.
            Structure: [{"grain_tensor": npt.NDArray, "bounding_box": tuple, "padding": int}].

        Returns
        -------
        npt.NDArray
            3-D Numpy array of the grain mask tensor.
        """
        # Initialise the grain mask tensor
        grain_mask_tensor = np.zeros(grain_mask_tensor_shape).astype(np.int32)

        # Iterate over the grain crops
        for grain_crop_and_bounding_box in grain_crops_and_bounding_boxes:
            # Get the grain crop and bounding box
            grain_crop = grain_crop_and_bounding_box["grain_tensor"]
            bounding_box = grain_crop_and_bounding_box["bounding_box"]
            padding = grain_crop_and_bounding_box["padding"]

            # Get the bounding box coordinates
            min_row, min_col, max_row, max_col = bounding_box

            # Crop the grain
            cropped_grain = grain_crop[
                padding:-padding,
                padding:-padding,
                :,
            ]

            # Update the grain mask tensor
            grain_mask_tensor[
                min_row + padding : max_row - padding,
                min_col + padding : max_col - padding,
                :,
            ] = cropped_grain

        # Update the background class
        grain_mask_tensor = Grains.update_background_class(grain_mask_tensor)

        return grain_mask_tensor.astype(bool)

    # Ignore too complex, to break the function down into smaller functions would make it more complex.
    # ruff: noqa: C901
    @staticmethod
    def convert_classes_when_too_big_or_small(
        grain_mask_tensor: npt.NDArray,
        pixel_to_nm_scaling: float,
        class_conversion_size_thresholds: list[tuple[tuple[int, int, int], tuple[int, int]]] | None,
    ) -> npt.NDArray:
        """
        Convert classes when they are too big or too small based on size thresholds.

        Parameters
        ----------
        grain_mask_tensor : npt.NDArray
            3-D Numpy array of the grain mask tensor.
        pixel_to_nm_scaling : float
            Scaling of pixels to nanometres.
        class_conversion_size_thresholds : list
            List of class conversion size thresholds.
            Structure is [(class_index, class_to_convert_to_if_to_small, class_to_convert_to_if_too_big),
            (lower_threshold, upper_threshold)].

        Returns
        -------
        npt.NDArray
            3-D Numpy array of the grain mask tensor with classes converted based on size thresholds.
        """
        if class_conversion_size_thresholds is None:
            return grain_mask_tensor

        new_grain_mask_tensor = np.copy(grain_mask_tensor)
        classes_to_vet = [vetting_criteria[0][0] for vetting_criteria in class_conversion_size_thresholds]
        for class_index in range(1, grain_mask_tensor.shape[2]):
            if class_index not in classes_to_vet:
                continue

            lower_threshold, upper_threshold = [
                vetting_criteria[1]
                for vetting_criteria in class_conversion_size_thresholds
                if vetting_criteria[0][0] == class_index
            ][0]

            class_to_convert_to_if_too_small, class_to_convert_to_if_too_big = [
                vetting_criteria[0][1:]
                for vetting_criteria in class_conversion_size_thresholds
                if vetting_criteria[0][0] == class_index
            ][0]

            # For each region in the class, check its size and convert if needed
            labelled_regions = Grains.label_regions(grain_mask_tensor[:, :, class_index])
            region_properties = Grains.get_region_properties(labelled_regions)
            for region in region_properties:
                region_mask = labelled_regions == region.label
                region_size = np.sum(region_mask) * pixel_to_nm_scaling**2
                if lower_threshold is not None:
                    if region_size < lower_threshold:
                        if class_to_convert_to_if_too_small is not None:
                            # Add the region to the class to convert to in the new tensor
                            new_grain_mask_tensor[:, :, class_to_convert_to_if_too_small] = np.where(
                                region_mask,
                                class_to_convert_to_if_too_small,
                                new_grain_mask_tensor[:, :, class_to_convert_to_if_too_small],
                            )
                        # Remove the region from the original class
                        new_grain_mask_tensor[:, :, class_index] = np.where(
                            region_mask,
                            0,
                            new_grain_mask_tensor[:, :, class_index],
                        )
                if upper_threshold is not None:
                    if region_size > upper_threshold:
                        if class_to_convert_to_if_too_big is not None:
                            # Add the region to the class to convert to in the new tensor
                            new_grain_mask_tensor[:, :, class_to_convert_to_if_too_big] = np.where(
                                region_mask,
                                class_to_convert_to_if_too_big,
                                new_grain_mask_tensor[:, :, class_to_convert_to_if_too_big],
                            )
                        # Remove the region from the original class
                        new_grain_mask_tensor[:, :, class_index] = np.where(
                            region_mask,
                            0,
                            new_grain_mask_tensor[:, :, class_index],
                        )

        # Update the background class
        new_grain_mask_tensor = Grains.update_background_class(new_grain_mask_tensor)

        return new_grain_mask_tensor.astype(bool)

    @staticmethod
    def vet_grains(
        grain_mask_tensor: npt.NDArray,
        pixel_to_nm_scaling: float,
        class_conversion_size_thresholds: list[tuple[tuple[int, int, int], tuple[int, int]]] | None,
        class_size_thresholds: list[tuple[int, int, int]] | None,
        class_region_number_thresholds: list[tuple[int, int, int]] | None,
        nearby_conversion_classes_to_convert: list[tuple[int, int]] | None,
        class_touching_threshold: int,
        keep_largest_labelled_regions_classes: list[int] | None,
        class_connection_point_thresholds: list[tuple[tuple[int, int], tuple[int, int]]] | None,
    ) -> npt.NDArray:
        """
        Vet grains in a grain mask tensor based on a variety of criteria.

        Parameters
        ----------
        grain_mask_tensor : npt.NDArray
            3-D Numpy array of the grain mask tensor.
        pixel_to_nm_scaling : float
            Scaling of pixels to nanometres.
        class_conversion_size_thresholds : list
            List of class conversion size thresholds. Structure is [(class_index, class_to_convert_to_if_too_small,
            class_to_convert_to_if_too_big), (lower_threshold, upper_threshold)].
        class_size_thresholds : list
            List of class size thresholds. Structure is [(class_index, lower, upper)].
        class_region_number_thresholds : list
            List of class region number thresholds. Structure is [(class_index, lower, upper)].
        nearby_conversion_classes_to_convert : list
            List of tuples of classes to convert. Structure is [(class_a, class_b)].
        class_touching_threshold : int
            Number of dilation passes to do to determine class A connectivity with class B.
        keep_largest_labelled_regions_classes : list
            List of classes to keep only the largest region.
        class_connection_point_thresholds : list
            List of tuples of classes and connection point thresholds. Structure is [(class_pair, (lower, upper))].

        Returns
        -------
        npt.NDArray
            3-D Numpy array of the vetted grain mask tensor.
        """
        # Get individual grain crops
        grain_tensor_crops, bounding_boxes, padding = Grains.get_individual_grain_crops(grain_mask_tensor)

        passed_grain_crops_and_bounding_boxes = []

        # Iterate over the grain crops
        for _, (single_grain_mask_tensor, bounding_box) in enumerate(zip(grain_tensor_crops, bounding_boxes)):

            # Convert small / big areas to other classes
            single_grain_mask_tensor = Grains.convert_classes_when_too_big_or_small(
                grain_mask_tensor=single_grain_mask_tensor,
                pixel_to_nm_scaling=pixel_to_nm_scaling,
                class_conversion_size_thresholds=class_conversion_size_thresholds,
            )

            # Vet number of regions (foreground and background)
            _, passed = Grains.vet_numbers_of_regions_single_grain(
                grain_mask_tensor=single_grain_mask_tensor,
                class_region_number_thresholds=class_region_number_thresholds,
            )
            if not passed:
                continue

            # Vet size of regions (foreground and background)
            _, passed = Grains.vet_class_sizes_single_grain(
                single_grain_mask_tensor=single_grain_mask_tensor,
                pixel_to_nm_scaling=pixel_to_nm_scaling,
                class_size_thresholds=class_size_thresholds,
            )
            if not passed:
                continue

            # Turn all but largest region of class A into class B provided that the class A region touched a class B
            # region
            converted_single_grain_mask_tensor = Grains.convert_classes_to_nearby_classes(
                grain_mask_tensor=single_grain_mask_tensor,
                classes_to_convert=nearby_conversion_classes_to_convert,
                class_touching_threshold=class_touching_threshold,
            )

            # Remove all but largest region in specific classes
            largest_only_single_grain_mask_tensor = Grains.keep_largest_labelled_region_classes(
                single_grain_mask_tensor=converted_single_grain_mask_tensor,
                keep_largest_labelled_regions_classes=keep_largest_labelled_regions_classes,
            )

            # Vet number of connection points between regions in specific classes
            if not Grains.vet_class_connection_points(
                grain_mask_tensor=largest_only_single_grain_mask_tensor,
                class_connection_point_thresholds=class_connection_point_thresholds,
            ):
                continue

            # If passed all vetting steps, add to the list of passed grain crops
            passed_grain_crops_and_bounding_boxes.append(
                {
                    "grain_tensor": largest_only_single_grain_mask_tensor,
                    "bounding_box": bounding_box,
                    "padding": padding,
                }
            )

        # Construct a new grain mask tensor from the passed grains
        return Grains.assemble_grain_mask_tensor_from_crops(
            grain_mask_tensor_shape=(
                grain_mask_tensor.shape[0],
                grain_mask_tensor.shape[1],
                grain_mask_tensor.shape[2],
            ),
            grain_crops_and_bounding_boxes=passed_grain_crops_and_bounding_boxes,
        )

    @staticmethod
    def merge_classes(
        grain_mask_tensor: npt.NDArray,
        classes_to_merge: list[tuple[int]] | None,
    ) -> npt.NDArray:
        """
        Merge classes in a grain mask tensor and add them to the grain tensor.

        Parameters
        ----------
        grain_mask_tensor : npt.NDArray
            3-D Numpy array of the grain mask tensor.
        classes_to_merge : list | None
            List of tuples for classes to merge, can be any number of classes.

        Returns
        -------
        npt.NDArray
            3-D Numpy array of the grain mask tensor with classes merged.
        """
        if classes_to_merge is None:
            return grain_mask_tensor
        # For each set of classes to merge:
        for classes in classes_to_merge:
            # Get the binary masks for all the classes
            class_masks = [grain_mask_tensor[:, :, class_index] for class_index in classes]
            # Combine the masks
            combined_mask = np.logical_or.reduce(class_masks)

            # Add new class to the grain tensor with the combined mask
            grain_mask_tensor = np.dstack([grain_mask_tensor, combined_mask])

        return grain_mask_tensor.astype(bool)
