"""Find grains in an image."""

# pylint: disable=no-name-in-module
from __future__ import annotations

import logging
from collections import defaultdict

import h5py
import keras
import numpy as np
import numpy.typing as npt
from skimage import morphology
from skimage.color import label2rgb
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border

from topostats.logs.logs import LOGGER_NAME
from topostats.thresholds import threshold
from topostats.unet_masking import (
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
        LOGGER.info(f"[{self.filename}] : Tidying borders")
        return clear_border(image, **kwargs)

    def label_regions(self, image: npt.NDArray, background: int = 0) -> npt.NDArray:
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
        LOGGER.info(f"[{self.filename}] : Labelling Regions")
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
        LOGGER.info(
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
        LOGGER.info(f"[{self.filename}] : Coloured regions")
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
        LOGGER.info(f"[{self.filename}] : Thresholding method (grains) : {self.threshold_method}")
        self.thresholds = get_thresholds(
            image=self.image,
            threshold_method=self.threshold_method,
            otsu_threshold_multiplier=self.otsu_threshold_multiplier,
            threshold_std_dev=self.threshold_std_dev,
            absolute=self.threshold_absolute,
        )

        for direction in self.direction:
            LOGGER.info(f"[{self.filename}] : Finding {direction} grains, threshold: ({self.thresholds[direction]})")
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

                LOGGER.info(f"[{self.filename}] : Running UNet model on {direction} grains")

                # When debugging, you might find that the custom_objects are incorrect. This is entirely based on what the model used
                # for its loss during training and so this will need to be changed a lot.
                # Once the group has gotten used to training models, this can be made configurable, but currently it's too changeable.
                # unet_model = keras.models.load_model(
                #     self.unet_config["model_path"], custom_objects={"dice_loss": dice_loss, "iou_loss": iou_loss}
                # )

                try:
                    unet_model = keras.models.load_model(
                        self.unet_config["model_path"], custom_objects={"mean_iou": mean_iou}
                    )
                except ValueError as e:
                    if "group" in str(e):
                        LOGGER.info(
                            "Issues loading model due to the groups error, trying to convert model to correct format"
                        )
                        f = h5py.File(self.unet_config["model_path"], mode="r+")
                        model_config_string = f.attrs.get("model_config")
                        if model_config_string.find('"groups": 1,') != -1:
                            model_config_string = model_config_string.replace('"groups": 1,', "")
                            f.attrs.modify("model_config", model_config_string)
                            f.flush()
                            model_config_string = f.attrs.get("model_config")
                            assert model_config_string.find('"groups": 1,') == -1
                        f.close()
                        unet_model = keras.models.load_model(
                            self.unet_config["model_path"], custom_objects={"mean_iou": mean_iou}
                        )

                # Get output shape
                output_shape = unet_model.output_shape
                LOGGER.info(f"Output shape of UNet model: {output_shape}")

                # For each detected molecule, create an image of just that molecule and run the UNet
                # on that image to segment it

                # Initialise an empty mask to iteratively add to for each grain, with the correct number of class channels based on
                # the loaded model's output shape
                # Note that the minimum number of classes is 2, as even for binary outputs, we will force categorical type
                # data, so we have a class for background.
                num_model_classes = output_shape[-1]
                if num_model_classes < 2:
                    LOGGER.info(f"Model has {num_model_classes} classes, forcing to have at least 2 classes")
                    num_model_classes = 2
                unet_mask = np.zeros((self.image.shape[0], self.image.shape[1], num_model_classes)).astype(np.bool_)
                # Set the background class to be all 1s by default since not all of the image will be covered by the
                # u-net predictions.
                unet_mask[:, :, 0] = 1
                # Sylvia: Sadly mypy cannot infer the type of regionprops since skimage don't provide stubs AFAIK
                for grain_number, region in enumerate(self.region_properties[direction]):
                    LOGGER.info(
                        f"Unet predicting mask for grain {grain_number} of {len(self.region_properties[direction])}"
                    )

                    # Get the bounding box for the region
                    bounding_box = np.array(region.bbox)  # min_row, min_col, max_row, max_col

                    # Pad the bounding box
                    padded_bounding_box = pad_bounding_box(
                        crop_min_row=bounding_box[0],
                        crop_min_col=bounding_box[1],
                        crop_max_row=bounding_box[2],
                        crop_max_col=bounding_box[3],
                        image_shape=self.image.shape,
                        padding=self.unet_config["grain_crop_padding"],
                    )

                    # Make the bounding box square within the confines of the image
                    square_bounding_box = make_bounding_box_square(
                        crop_min_row=padded_bounding_box[0],
                        crop_min_col=padded_bounding_box[1],
                        crop_max_row=padded_bounding_box[2],
                        crop_max_col=padded_bounding_box[3],
                        image_shape=self.image.shape,
                    )

                    # Grab the cropped image. Using slice since the bounding box from skimage is
                    # half-open, so the max_row and max_col are not included in the region.
                    region_image = self.image[
                        square_bounding_box[0] : square_bounding_box[2],
                        square_bounding_box[1] : square_bounding_box[3],
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
                        upper_norm_bound=self.unet_config["upper_norm_bound"],
                        lower_norm_bound=self.unet_config["lower_norm_bound"],
                    )

                    assert len(predicted_mask.shape) == 3
                    LOGGER.info(f"Predicted mask shape: {predicted_mask.shape}")

                    # Use only the largest segmentation object
                    # For each class in the unet mask tensor, label the mask
                    unet_predicted_mask_labelled = np.zeros_like(predicted_mask).astype(np.bool_)
                    for class_index in range(predicted_mask.shape[2]):
                        # Get the mask for the class
                        predicted_mask_class = predicted_mask[:, :, class_index]
                        # Label the mask
                        predicted_mask_class_labelled = morphology.label(predicted_mask_class)
                        # Get the sizes of the regions
                        sizes = np.array(
                            [
                                (predicted_mask_class_labelled == label).sum()
                                for label in range(1, predicted_mask_class_labelled.max() + 1)
                            ]
                        )
                        # Keep only the largest region
                        predicted_mask_class = np.where(
                            predicted_mask_class_labelled == sizes.argmax() + 1, predicted_mask_class, 0
                        )

                        # Add the class mask to the overall mask
                        unet_predicted_mask_labelled[:, :, class_index] = predicted_mask_class

                    # NEED TO MAKE THIS GENERAL TO MULTIPLE CLASSES
                    # Add the predicted mask to the overall mask
                    # unet_mask[
                    #     square_bounding_box[0] : square_bounding_box[2],
                    #     square_bounding_box[1] : square_bounding_box[3],
                    # ] = np.logical_or(
                    #     unet_mask[
                    #         square_bounding_box[0] : square_bounding_box[2],
                    #         square_bounding_box[1] : square_bounding_box[3],
                    #     ],
                    #     predicted_mask,
                    # )

                    # Add each class of the predicted mask to the overall full image mask
                    for class_index in range(unet_mask.shape[2]):
                        # Directly set the background to be equal since they are by default
                        # 1, and should be turned off if any other class is on
                        if class_index == 0:
                            unet_mask[
                                square_bounding_box[0] : square_bounding_box[2],
                                square_bounding_box[1] : square_bounding_box[3],
                                class_index,
                            ] = unet_predicted_mask_labelled[:, :, class_index]
                        else:
                            unet_mask[
                                square_bounding_box[0] : square_bounding_box[2],
                                square_bounding_box[1] : square_bounding_box[3],
                                class_index,
                            ] = np.logical_or(
                                unet_mask[
                                    square_bounding_box[0] : square_bounding_box[2],
                                    square_bounding_box[1] : square_bounding_box[3],
                                    class_index,
                                ],
                                unet_predicted_mask_labelled[:, :, class_index],
                            )

                    # If the unet mask is a single channel, in an NxNx1 array, then we need to add a background channel to force
                    # it to have the standard NxNxC shape where C >= 2 (background and DNA at least)
                    # if len(unet_mask.shape) == 2 or unet_mask.shape[-1] == 1:
                    #     unet_mask = np.stack([np.where(unet_mask == 0, 1, 0), unet_mask], axis=-1)

                    assert len(unet_mask.shape) == 3, f"Unet mask shape: {unet_mask.shape}"
                    assert unet_mask.shape[-1] >= 2, f"Unet mask shape: {unet_mask.shape}"

                    # Update the image masks to be the unet masks instead
                    self.directions[direction]["removed_small_objects"] = unet_mask

                    # For each class in the unet mask tensor, label the mask
                    unet_labelled_regions = np.zeros_like(unet_mask).astype(np.int32)
                    for class_index in range(unet_mask.shape[2]):
                        unet_labelled_regions[:, :, class_index] = self.label_regions(unet_mask[:, :, class_index])

                    self.directions[direction]["labelled_regions_02"] = unet_labelled_regions

                    LOGGER.info(f"[{self.filename}] : Overridden grains with UNet predictions ({direction})")
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
        # Get the sizes of the regions
        sizes = np.array([(labelled_image == label).sum() for label in range(1, labelled_image.max() + 1)])
        # Keep only the largest region
        return np.where(labelled_image == sizes.argmax() + 1, labelled_image, 0).astype(bool)
