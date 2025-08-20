# Disable ruff 301 - pickle loading is unsafe, but we don't care for tests.
# ruff: noqa: S301
"""Fixtures for testing."""

import importlib.resources as pkg_resources
import pickle
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import yaml
from skimage import draw, filters
from skimage.measure._regionprops import RegionProperties
from skimage.morphology import skeletonize

import topostats
from topostats.filters import Filters, combine_mask_directions
from topostats.grains import GrainCrop, GrainCropsDirection, Grains, ImageGrainCrops
from topostats.grainstats import GrainStats
from topostats.io import LoadScans, read_yaml
from topostats.plotting import TopoSum
from topostats.utils import get_thresholds

# This is required because of the inheritance used throughout
# pylint: disable=redefined-outer-name
# pylint: disable=too-many-lines
BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"
GRAINCROP_DIR = RESOURCES / "graincrop"

RNG = np.random.default_rng(seed=1000)
SMALL_ARRAY_SIZE = (10, 10)
THRESHOLD = 0.5
CHANNEL = "Height"


# ruff: noqa: D401


@pytest.fixture()
def default_config() -> dict:
    """Sample configuration."""
    config = read_yaml(BASE_DIR / "topostats" / "default_config.yaml")
    plotting_dictionary = pkg_resources.open_text(topostats, "plotting_dictionary.yaml")
    config["plotting"]["plot_dict"] = yaml.safe_load(plotting_dictionary.read())
    config["filter"]["threshold_method"] = "std_dev"
    config["filter"]["remove_scars"]["run"] = True
    config["grains"]["threshold_method"] = "absolute"
    config["grains"]["threshold_absolute"] = [1.0]
    config["grains"]["area_thresholds"]["above"] = [10, 60000000]
    return config


@pytest.fixture()
def process_scan_config() -> dict:
    """Sample configuration."""
    config = read_yaml(BASE_DIR / "topostats" / "default_config.yaml")
    config["filter"]["remove_scars"]["run"] = True
    config["grains"]["threshold_std_dev"] = [-1.0]
    config["grains"]["area_thresholds"]["above"] = [500, 800]
    config["plotting"]["zrange"] = [0, 3]
    plotting_dictionary = pkg_resources.open_text(topostats, "plotting_dictionary.yaml")
    config["plotting"]["plot_dict"] = yaml.safe_load(plotting_dictionary.read())
    return config


@pytest.fixture()
def plot_dict() -> dict:
    """Load the plot_dict dictionary.

    This is required because the above configs have the 'plot_dict' key/value popped.
    """
    plotting_dictionary = pkg_resources.open_text(topostats, "plotting_dictionary.yaml")
    return yaml.safe_load(plotting_dictionary.read())


@pytest.fixture()
def summary_config() -> dict:
    """Sample summary configuration."""
    summary_yaml = pkg_resources.open_text(topostats, "summary_config.yaml")
    summary_config = yaml.safe_load(summary_yaml.read())
    # Tweak configuration (for now) to match the tests
    summary_config.pop("violin")
    summary_config.pop("csv_file")
    summary_config.pop("stats_to_sum")
    summary_config["figsize"] = (15, 12)
    summary_config["kde"] = True
    summary_config["hist"] = True
    summary_config["stat_to_sum"] = "area"
    plotting_yaml = pkg_resources.open_text(topostats, "var_to_label.yaml")
    summary_config["var_to_label"] = yaml.safe_load(plotting_yaml.read())
    return summary_config


@pytest.fixture()
def toposum_object_single_directory(summary_config: dict) -> TopoSum:
    """Set up a TopoSum object fixture for testing plotting.

    Uses a dataframe containing data from a single directory.
    """
    return TopoSum(csv_file=RESOURCES / "toposum_all_statistics_single_directory.csv", **summary_config)


@pytest.fixture()
def toposum_object_multiple_directories(summary_config: dict) -> TopoSum:
    """Set up a TopoSum object fixture for testing plotting.

    Uses a dataframe containing data from several directories.
    """
    return TopoSum(csv_file=RESOURCES / "toposum_all_statistics_multiple_directories.csv", **summary_config)


@pytest.fixture()
def loading_config(default_config: dict) -> dict:
    """Configuration for loading scans."""
    return default_config["loading"]


@pytest.fixture()
def filter_config(default_config: dict) -> dict:
    """Configurations for filtering."""
    config = default_config["filter"]
    config.pop("run")
    return config


@pytest.fixture()
def grains_config(default_config: dict) -> dict:
    """Configurations for grain finding."""
    config = default_config["grains"]
    config.pop("run")
    return config


@pytest.fixture()
def grainstats_config(default_config: dict) -> dict:
    """Configurations for grainstats."""
    config = default_config["grainstats"]
    config["direction"] = "above"
    config.pop("run")
    config.pop("class_names")
    return config


@pytest.fixture()
def plotting_config(default_config: dict) -> dict:
    """Configurations for plotting."""
    config = default_config["plotting"]
    config["image_set"] = "all"
    config.pop("run")
    config.pop("plot_dict")
    return config


@pytest.fixture()
def plotting_config_with_plot_dict(default_config: dict) -> dict:
    """Plotting configuration with plot dict."""
    return default_config["plotting"]


@pytest.fixture()
def image_random() -> npt.NDArray:
    """Random image as NumPy array."""
    rng = np.random.default_rng(seed=1000)
    return rng.random((1024, 1024))


@pytest.fixture()
def small_array() -> npt.NDArray:
    """Small (10x10) image array for testing."""
    return RNG.random(SMALL_ARRAY_SIZE)


@pytest.fixture()
def small_mask() -> npt.NDArray:
    """Small (10x10) mask array for testing."""
    return RNG.uniform(low=0, high=1, size=SMALL_ARRAY_SIZE) > 0.5


@pytest.fixture()
def synthetic_scars_image() -> npt.NDArray:
    """Small synthetic image for testing scar removal."""
    return np.load(RESOURCES / "test_scars_synthetic_scar_image.npy")


@pytest.fixture()
def synthetic_marked_scars() -> npt.NDArray:
    """Small synthetic boolean array of marked scar coordinates corresponding to synthetic_scars_image."""
    return np.load(RESOURCES / "test_scars_synthetic_mark_scars.npy")


@pytest.fixture()
def image_random_row_medians() -> np.array:
    """Expected row medians (unmasked)."""
    return np.loadtxt(RESOURCES / "image_random_row_medians.csv", delimiter=",")


@pytest.fixture()
def image_random_col_medians() -> np.array:
    """Expected column medians (unmasked)."""
    return np.loadtxt(RESOURCES / "image_random_col_medians.csv", delimiter=",")


@pytest.fixture()
def image_random_median_flattened() -> np.array:
    """Expected aligned rows (unmasked)."""
    df = pd.read_csv(RESOURCES / "image_random_median_flattened.csv.bz2", header=None)
    return df.to_numpy()


@pytest.fixture()
def image_random_remove_x_y_tilt() -> np.array:
    """Expected removed tilt (unmasked)."""
    df = pd.read_csv(RESOURCES / "image_random_remove_x_y_tilt.csv.bz2", header=None)
    return df.to_numpy()


@pytest.fixture()
def image_random_remove_quadratic() -> np.array:
    """Expected removed quadratic (unmasked)."""
    df = pd.read_csv(RESOURCES / "image_random_remove_quadratic.csv.bz2", header=None)
    return df.to_numpy()


@pytest.fixture()
def image_random_mask() -> np.array:
    """Expected mask."""
    df = pd.read_csv(RESOURCES / "image_random_mask.csv.bz2", header=None)
    return df.to_numpy()


@pytest.fixture()
def image_random_row_medians_masked() -> np.array:
    """Expected row medians (masked)."""
    return np.loadtxt(RESOURCES / "image_random_row_medians_masked.csv", delimiter=",")


@pytest.fixture()
def image_random_col_medians_masked() -> np.array:
    """Expected column medians (masked)."""
    return np.loadtxt(RESOURCES / "image_random_col_medians_masked.csv", delimiter=",")


@pytest.fixture()
def test_filters(load_scan: LoadScans, filter_config: dict) -> Filters:
    """Filters class for testing."""
    load_scan.get_data()
    return Filters(
        image=load_scan.image,
        filename=load_scan.filename,
        pixel_to_nm_scaling=load_scan.pixel_to_nm_scaling,
        **filter_config,
    )


@pytest.fixture()
def test_filters_random(test_filters: Filters, image_random: np.ndarray) -> Filters:
    """Filters class for testing with pixels replaced by random image."""
    test_filters.images["pixels"] = image_random
    return test_filters


@pytest.fixture()
def test_filters_random_with_mask(filter_config: dict, test_filters: Filters, image_random: np.ndarray) -> Filters:
    """Filters class for testing with pixels replaced by random image."""
    test_filters.images["pixels"] = image_random
    thresholds_list = get_thresholds(
        image=test_filters.images["pixels"],
        threshold_method="otsu",
        otsu_threshold_multiplier=filter_config["otsu_threshold_multiplier"],
    )
    if len(thresholds_list) > 1:
        thresholds = {"above": [thresholds_list[0]], "below": [thresholds_list[1]]}
    else:
        thresholds = {"above": [thresholds_list[0]], "below": [-thresholds_list[0]]}
    test_filters.images["mask"] = combine_mask_directions(image=test_filters.images["pixels"], thresholds=thresholds)
    return test_filters


@pytest.fixture()
def random_filters(test_filters_random_with_mask: Filters) -> Filters:
    """Process random with filters, for use in grains fixture."""
    test_filters_random_with_mask.images["initial_median_flatten"] = test_filters_random_with_mask.median_flatten(
        test_filters_random_with_mask.images["pixels"], mask=None
    )
    test_filters_random_with_mask.images["initial_tilt_removal"] = test_filters_random_with_mask.remove_tilt(
        test_filters_random_with_mask.images["initial_median_flatten"], mask=None
    )
    test_filters_random_with_mask.images["masked_median_flatten"] = test_filters_random_with_mask.median_flatten(
        test_filters_random_with_mask.images["initial_tilt_removal"],
        mask=test_filters_random_with_mask.images["mask"],
    )
    test_filters_random_with_mask.images["masked_tilt_removal"] = test_filters_random_with_mask.remove_tilt(
        test_filters_random_with_mask.images["masked_median_flatten"],
        mask=test_filters_random_with_mask.images["mask"],
    )

    return test_filters_random_with_mask


@pytest.fixture()
def remove_scars_config(synthetic_scars_image: np.ndarray, default_config: dict) -> dict:
    """Configuration for testing scar removal."""
    config = default_config["filter"]["remove_scars"]
    config["img"] = synthetic_scars_image
    config["filename"] = " "
    config["removal_iterations"] = 2
    config["threshold_low"] = 1.5
    config["threshold_high"] = 1.8
    config["max_scar_width"] = 2
    config["min_scar_length"] = 1
    config.pop("run")
    return config


@pytest.fixture()
def random_grains(grains_config: dict, random_filters: Filters) -> Grains:
    """Grains object based on random image which has no grains."""
    grains = Grains(
        image=random_filters.images["zero_averaged_background"],
        filename="random",
        pixel_to_nm_scaling=0.5,
        **grains_config,
    )
    grains.find_grains()
    return grains


@pytest.fixture()
def dummy_graincrop() -> GrainCrop:
    """Dummy GrainCrop object for testing."""
    image = RNG.random(size=(10, 10)).astype(np.float32)
    mask = np.stack(
        arrays=[
            np.array(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
                    [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
                    [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
                    [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
                    [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
                    [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ]
            ),
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
        ],
        axis=-1,
    )
    return GrainCrop(
        image=image,
        mask=mask,
        padding=2,
        bbox=(1, 1, 11, 11),
        pixel_to_nm_scaling=1.0,
        filename="dummy",
        stats={1: {0: {"centre_x": 5, "centre_y": 5}}},
        height_profiles={1: {0: np.asarray([1, 2, 3, 4, 5])}},
    )


@pytest.fixture()
def dummy_graincrops_dict(dummy_graincrop: GrainCrop) -> dict[int, GrainCrop]:
    """Dummy dictionary of GrainCrop objects for testing."""
    return {0: dummy_graincrop}


@pytest.fixture()
def dummy_graincropsdirection(dummy_graincrops_dict: dict[int, GrainCrop]) -> GrainCropsDirection:
    """Dummy GrainCropsDirection object for testing."""
    full_mask_tensor = np.stack(
        [
            np.array(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
                    [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
                    [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
                    [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
                    [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
                    [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ]
            ),
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
            ),
        ],
        axis=-1,
    ).astype(np.bool_)
    return GrainCropsDirection(full_mask_tensor=full_mask_tensor, crops=dummy_graincrops_dict)


@pytest.fixture()
def graincrop_catenanes_0() -> GrainCrop:
    """Catenanes GrainCrop object."""
    image: npt.NDArray[float] = np.load(GRAINCROP_DIR / "example_catenanes_image_0.npy")
    mask: npt.NDArray[bool] = np.load(GRAINCROP_DIR / "example_catenanes_mask_0.npy")
    return GrainCrop(
        image=image,
        mask=mask,
        padding=1,
        bbox=(0, 2, 323, 325),
        pixel_to_nm_scaling=0.488,
        filename="example_catenanes",
    )


@pytest.fixture()
def graincrop_catenanes_1() -> GrainCrop:
    """Catenane GrainCrop object."""
    image: npt.NDArray[float] = np.load(GRAINCROP_DIR / "example_catenanes_image_1.npy")
    mask: npt.NDArray[bool] = np.load(GRAINCROP_DIR / "example_catenanes_mask_1.npy")
    return GrainCrop(
        image=image,
        mask=mask,
        padding=1,
        bbox=(77, 75, 400, 398),
        pixel_to_nm_scaling=0.488,
        filename="example_catenanes",
    )


@pytest.fixture()
def graincrops_above_catenanes(
    graincrop_catenanes_0: GrainCrop, graincrop_catenanes_1: GrainCrop
) -> GrainCropsDirection:
    """GrainCropsDirection object of example catenanes."""
    full_mask_tensor: npt.NDArray[bool] = np.load(GRAINCROP_DIR / "example_catenanes_full_mask_tensor.npy")
    return GrainCropsDirection(
        crops={0: graincrop_catenanes_0, 1: graincrop_catenanes_1}, full_mask_tensor=full_mask_tensor
    )


@pytest.fixture()
def imagegraincrops_catenanes(graincrops_above_catenanes: GrainCropsDirection) -> ImageGrainCrops:
    """ImageGrainCrops object of example catenanes."""
    return ImageGrainCrops(above=graincrops_above_catenanes, below=None)


@pytest.fixture()
def topostats_catenanes_2_4_0(imagegraincrops_catenanes) -> topostats.TopoStats:
    """TopoStats object of example catenanes."""
    return topostats.TopoStats(
        image_grain_crops=imagegraincrops_catenanes,
        filename="example_catenanes.spm",
        pixel_to_nm_scaling=0.488,
        topostats_version="2.4.0",
        img_path=str(GRAINCROP_DIR),
        image=None,
        image_original=None,
    )


@pytest.fixture()
def graincrop_rep_int_0() -> GrainCrop:
    """Rep_Int GrainCrop object."""
    image: npt.NDArray[float] = np.load(GRAINCROP_DIR / "example_rep_int_image_0.npy")
    mask: npt.NDArray[bool] = np.load(GRAINCROP_DIR / "example_rep_int_mask_0.npy")
    return GrainCrop(
        image=image,
        mask=mask,
        padding=1,
        bbox=(19, 4, 341, 326),
        pixel_to_nm_scaling=0.488,
        filename="example_rep",
    )


@pytest.fixture()
def graincrops_above_rep_int(graincrop_rep_int_0: GrainCrop) -> GrainCropsDirection:
    """GrainCropsDirection object of example rep_int."""
    full_mask_tensor: npt.NDArray[bool] = np.load(GRAINCROP_DIR / "example_rep_int_full_mask_tensor.npy")
    return GrainCropsDirection(crops={0: graincrop_rep_int_0}, full_mask_tensor=full_mask_tensor)


@pytest.fixture()
def imagegraincrops_rep_int(graincrops_above_rep_int: GrainCropsDirection) -> ImageGrainCrops:
    """ImageGrainCrops object of example rep_int."""
    return ImageGrainCrops(above=graincrops_above_rep_int, below=None)


@pytest.fixture()
def topostats_rep_int_2_4_0(imagegraincrops_rep_int) -> topostats.TopoStats:
    """TopoStats object of example rep_int."""
    return topostats.TopoStats(
        image_grain_crops=imagegraincrops_rep_int,
        filename="example_rep_int.spm",
        pixel_to_nm_scaling=0.488,
        topostats_version="2.4.0",
        img_path=str(GRAINCROP_DIR),
        image=None,
        image_original=None,
    )


@pytest.fixture()
def small_array_filters(small_array: np.ndarray, load_scan: LoadScans, filter_config: dict) -> Grains:
    """Filters object based on small_array."""
    filter_obj = Filters(
        image=load_scan.image,
        filename=load_scan.filename,
        pixel_to_nm_scaling=load_scan.pixel_to_nm_scaling,
        **filter_config,
    )
    filter_obj.pixel_to_nm_scaling = 0.5
    filter_obj.images["zero_averaged_background"] = filter_obj.gaussian_filter(image=small_array)
    return filter_obj


# IO fixtures
@pytest.fixture()
def load_scan_dummy() -> LoadScans:
    """Instantiate a dummy LoadScans object for use in testing `.gwy` IO methods."""
    return LoadScans(img_paths="dummy", channel="dummy")


@pytest.fixture()
def load_scan_topostats_test_file(tmp_path: Path, loading_config: dict) -> LoadScans:
    """Instantiate a LoadScans object for a temporarily saved test .topostats file."""
    loading_config["extract"] = "all"
    return LoadScans([tmp_path / "topostats_file_test.topostats"], **loading_config)


@pytest.fixture()
def load_scan(loading_config: dict) -> LoadScans:
    """Instantiate a LoadScans object from a small .topostats image file."""
    return LoadScans([RESOURCES / "test_image" / "minicircle_small.topostats"], **loading_config)


@pytest.fixture()
def load_scan_data(loading_config: dict) -> LoadScans:
    """Instance of a LoadScans object after applying the get_data func."""
    scan_data = LoadScans([RESOURCES / "test_image" / "minicircle_small.topostats"], **loading_config)
    scan_data.get_data()
    return scan_data


@pytest.fixture()
def load_scan_spm() -> LoadScans:
    """Instantiate a LoadScans object from a .spm file."""
    return LoadScans([RESOURCES / "minicircle.spm"], channel="Height")


@pytest.fixture()
def load_scan_ibw() -> LoadScans:
    """Instantiate a LoadScans object from a .ibw file."""
    return LoadScans([RESOURCES / "minicircle2.ibw"], channel="HeightTracee")


@pytest.fixture()
def load_scan_jpk() -> LoadScans:
    """Instantiate a LoadScans object from a .jpk file."""
    return LoadScans([RESOURCES / "file.jpk"], channel="height_trace")


@pytest.fixture()
def load_scan_jpk_qi() -> LoadScans:
    """Instantiate a LoadScans object from a .jpk-qi-image file."""
    return LoadScans([RESOURCES / "file.jpk-qi-image"], channel="height_trace")


@pytest.fixture()
def load_scan_gwy() -> LoadScans:
    """Instantiate a LoadScans object from a .gwy file."""
    return LoadScans([RESOURCES / "file.gwy"], channel="ZSensor")


@pytest.fixture()
def load_scan_stp() -> LoadScans:
    """Instantiate a LoadScans object from a .stp file."""
    return LoadScans([RESOURCES / "file.stp"], channel=None)


@pytest.fixture()
def load_scan_top() -> LoadScans:
    """Instantiate a LoadScans object from a .top file."""
    return LoadScans([RESOURCES / "file.top"], channel=None)


@pytest.fixture()
def load_scan_topostats() -> LoadScans:
    """Instantiate a LoadScans object from a .topostats file."""
    return LoadScans([RESOURCES / "file.topostats"], channel="dummy_channel")


@pytest.fixture()
def load_scan_asd() -> LoadScans:
    """Instantiate a LoadScans object from a .asd file."""
    return LoadScans([RESOURCES / "file.asd"], channel="TP")


# Minicircle fixtures
@pytest.fixture()
def minicircle(load_scan: LoadScans, filter_config: dict) -> Filters:
    """Instantiate a Filters object, creates the output directory and loads the image."""
    load_scan.get_data()
    return Filters(
        image=load_scan.image,
        filename=load_scan.filename,
        pixel_to_nm_scaling=load_scan.pixel_to_nm_scaling,
        **filter_config,
    )


@pytest.fixture()
def minicircle_initial_median_flatten(minicircle: Filters) -> Filters:
    """Initial align on unmasked data."""
    minicircle.images["initial_median_flatten"] = minicircle.median_flatten(minicircle.images["pixels"], mask=None)
    return minicircle


@pytest.fixture()
def minicircle_initial_tilt_removal(minicircle_initial_median_flatten: Filters) -> Filters:
    """Initial x/y tilt removal on unmasked data."""
    minicircle_initial_median_flatten.images["initial_tilt_removal"] = minicircle_initial_median_flatten.remove_tilt(
        minicircle_initial_median_flatten.images["initial_median_flatten"], mask=None
    )
    return minicircle_initial_median_flatten


@pytest.fixture()
def minicircle_initial_quadratic_removal(minicircle_initial_tilt_removal: Filters) -> Filters:
    """Initial quadratic removal on unmasked data."""
    minicircle_initial_tilt_removal.images["initial_quadratic_removal"] = (
        minicircle_initial_tilt_removal.remove_quadratic(
            minicircle_initial_tilt_removal.images["initial_tilt_removal"], mask=None
        )
    )
    return minicircle_initial_tilt_removal


@pytest.fixture()
def minicircle_threshold_otsu(minicircle_initial_tilt_removal: Filters) -> Filters:
    """Calculate threshold."""
    thresholds_list = get_thresholds(
        image=minicircle_initial_tilt_removal.images["initial_tilt_removal"],
        threshold_method="otsu",
        otsu_threshold_multiplier=1.0,
    )
    if len(thresholds_list) > 1:
        minicircle_initial_tilt_removal.thresholds = {"above": thresholds_list[0], "below": thresholds_list[1]}
    elif thresholds_list[0] > 0:
        minicircle_initial_tilt_removal.thresholds = {"above": thresholds_list[0]}
    else:
        minicircle_initial_tilt_removal.thresholds = {"below": thresholds_list[0]}
    return minicircle_initial_tilt_removal


@pytest.fixture()
def minicircle_threshold_stddev(minicircle_initial_tilt_removal: Filters) -> Filters:
    """Calculate threshold."""
    minicircle_initial_tilt_removal.thresholds = get_thresholds(
        image=minicircle_initial_tilt_removal.images["initial_tilt_removal"],
        threshold_method="std_dev",
        otsu_threshold_multiplier=None,
        threshold_std_dev={"below": [10.0], "above": [1.0]},
    )
    return minicircle_initial_tilt_removal


@pytest.fixture()
def minicircle_threshold_abs(minicircle_initial_tilt_removal: Filters) -> Filters:
    """Calculate threshold."""
    minicircle_initial_tilt_removal.thresholds = get_thresholds(
        image=minicircle_initial_tilt_removal.images["initial_tilt_removal"],
        threshold_method="absolute",
        otsu_threshold_multiplier=None,
        absolute={"below": [-1.5], "above": [1.5]},
    )
    return minicircle_initial_tilt_removal


@pytest.fixture()
def minicircle_mask(minicircle_threshold_otsu: Filters) -> Filters:
    """Derive mask based on threshold."""
    minicircle_threshold_otsu.images["mask"] = combine_mask_directions(
        image=minicircle_threshold_otsu.images["initial_tilt_removal"],
        thresholds=minicircle_threshold_otsu.thresholds,
    )
    return minicircle_threshold_otsu


@pytest.fixture()
def minicircle_masked_median_flatten(minicircle_mask: Filters) -> Filters:
    """Secondary alignment using mask."""
    minicircle_mask.images["masked_median_flatten"] = minicircle_mask.median_flatten(
        minicircle_mask.images["initial_tilt_removal"], mask=minicircle_mask.images["mask"]
    )
    return minicircle_mask


@pytest.fixture()
def minicircle_masked_tilt_removal(minicircle_masked_median_flatten: Filters) -> Filters:
    """Secondary x/y tilt removal using mask."""
    minicircle_masked_median_flatten.images["masked_tilt_removal"] = minicircle_masked_median_flatten.remove_tilt(
        minicircle_masked_median_flatten.images["masked_median_flatten"],
        mask=minicircle_masked_median_flatten.images["mask"],
    )
    return minicircle_masked_median_flatten


@pytest.fixture()
def minicircle_masked_quadratic_removal(minicircle_masked_tilt_removal: Filters) -> Filters:
    """Secondary quadratic removal using mask."""
    minicircle_masked_tilt_removal.images["masked_quadratic_removal"] = minicircle_masked_tilt_removal.remove_quadratic(
        minicircle_masked_tilt_removal.images["masked_tilt_removal"],
        mask=minicircle_masked_tilt_removal.images["mask"],
    )
    return minicircle_masked_tilt_removal


@pytest.fixture()
def minicircle_grain_gaussian_filter(minicircle_masked_quadratic_removal: Filters) -> Filters:
    """Apply Gaussian filter."""
    minicircle_masked_quadratic_removal.images["gaussian_filtered"] = (
        minicircle_masked_quadratic_removal.gaussian_filter(
            image=minicircle_masked_quadratic_removal.images["masked_quadratic_removal"]
        )
    )
    return minicircle_masked_quadratic_removal


# Derive fixtures for grain finding
@pytest.fixture()
def minicircle_grains(minicircle_grain_gaussian_filter: Filters, grains_config: dict) -> Grains:
    """Grains object based on filtered minicircle."""
    return Grains(
        image=minicircle_grain_gaussian_filter.images["gaussian_filtered"],
        filename=minicircle_grain_gaussian_filter.filename,
        pixel_to_nm_scaling=minicircle_grain_gaussian_filter.pixel_to_nm_scaling,
        **grains_config,
    )


@pytest.fixture()
def minicircle_grain_threshold_otsu(minicircle_grains: Grains, grains_config: dict) -> Grains:
    """Calculate threshold."""
    grains_config.pop("threshold_method")
    grains_config["threshold_method"] = "otsu"
    minicircle_grains.thresholds = get_thresholds(
        image=minicircle_grains.image,
        threshold_method="otsu",
        otsu_threshold_multiplier=1.0,
    )
    return minicircle_grains


@pytest.fixture()
def minicircle_grain_threshold_stddev(minicircle_grains: Grains, grains_config: dict) -> Grains:
    """Calculate threshold."""
    grains_config["threshold_method"] = "std_dev"
    minicircle_grains.thresholds = get_thresholds(
        image=minicircle_grains.image,
        threshold_method="std_dev",
        otsu_threshold_multiplier=None,
        threshold_std_dev={"below": [10.0], "above": [1.0]},
        absolute=None,
    )
    return minicircle_grains


@pytest.fixture()
def minicircle_grain_threshold_abs(minicircle_grains: Grains) -> Grains:
    """Calculate threshold."""
    minicircle_grains.thresholds = get_thresholds(
        image=minicircle_grains.image,
        threshold_method="absolute",
        otsu_threshold_multiplier=None,
        absolute={"below": [-1.0], "above": [1.0]},
    )
    return minicircle_grains


@pytest.fixture()
def minicircle_grain_traditional_thresholding(minicircle_grain_threshold_abs: Grains) -> Grains:
    """Boolean mask."""
    minicircle_grain_threshold_abs.mask_images["above"] = {}
    # Typing conformity
    assert minicircle_grain_threshold_abs.thresholds is not None
    minicircle_grain_threshold_abs.mask_images["above"]["thresholded_grains"] = Grains.multi_class_thresholding(
        image=minicircle_grain_threshold_abs.image,
        thresholds=minicircle_grain_threshold_abs.thresholds["above"],
        image_name="minicircle_grain_threshold_abs",
    )
    return minicircle_grain_threshold_abs


@pytest.fixture()
def minicircle_small_graincrops() -> dict[int, GrainCrop]:
    """Dictionary of graincrops for the minicircle_small image."""
    with Path.open(RESOURCES / "minicircle_small_graincrops.pkl", "rb") as f:  # pylint: disable=unspecified-encoding
        return pickle.load(f)


@pytest.fixture()
def minicircle_grain_clear_border(minicircle_grain_traditional_thresholding: Grains) -> Grains:
    """Cleared borders."""
    minicircle_grain_traditional_thresholding.mask_images["above"]["tidied_border"] = Grains.tidy_border_tensor(
        minicircle_grain_traditional_thresholding.mask_images["above"]["thresholded_grains"]
    )
    return minicircle_grain_traditional_thresholding


@pytest.fixture()
def minicircle_grain_remove_objects_too_small_to_process(minicircle_grain_clear_border: Grains) -> Grains:
    """Fixture to test removing noise."""
    area_thresholded = Grains.area_thresholding_tensor(
        minicircle_grain_clear_border.mask_images["above"]["tidied_border"],
        area_thresholds=[10 * minicircle_grain_clear_border.pixel_to_nm_scaling**2, None],
        pixel_to_nm_scaling=minicircle_grain_clear_border.pixel_to_nm_scaling,
    )
    minicircle_grain_clear_border.mask_images["above"]["removed_objects_too_small_to_process"] = (
        Grains.bbox_size_thresholding_tensor(grain_mask_tensor=area_thresholded, bbox_size_thresholds=(5, None))
    )
    return minicircle_grain_clear_border


@pytest.fixture()
def minicircle_grain_area_thresholding(minicircle_grain_remove_objects_too_small_to_process: Grains) -> Grains:
    """Small objects removed."""
    area_thresholds = [30, 2000]
    minicircle_grain_remove_objects_too_small_to_process.mask_images["above"]["area_thresholded"] = (
        Grains.area_thresholding_tensor(
            grain_mask_tensor=minicircle_grain_remove_objects_too_small_to_process.mask_images["above"][
                "removed_objects_too_small_to_process"
            ],
            area_thresholds=area_thresholds,
            pixel_to_nm_scaling=minicircle_grain_remove_objects_too_small_to_process.pixel_to_nm_scaling,
        )
    )

    return minicircle_grain_remove_objects_too_small_to_process


@pytest.fixture()
def minicircle_grain_area_thresholding_regionprops(
    minicircle_grain_area_thresholding: Grains,
) -> list[RegionProperties]:
    """Region properties of the area thresholded image."""
    labelled_image = Grains.label_regions(
        image=minicircle_grain_area_thresholding.mask_images["above"]["area_thresholded"][:, :, 1]
    )
    return Grains.get_region_properties(image=labelled_image)


# Derive fixture for grainstats
@pytest.fixture()
def dummy_grainstats(
    dummy_graincrops_dict: dict[int, GrainCrop], grainstats_config: dict, tmp_path: Path
) -> GrainStats:
    """Grainstats class for testing functions."""
    return GrainStats(
        grain_crops=dummy_graincrops_dict,
        base_output_dir=tmp_path,
        **grainstats_config,
    )


@pytest.fixture()
def minicircle_grainstats(
    minicircle_small_graincrops: dict[int, GrainCrop],
    grainstats_config: dict,
    tmp_path: Path,
) -> GrainStats:
    """GrainStats object."""
    return GrainStats(
        grain_crops=minicircle_small_graincrops,
        base_output_dir=tmp_path,
        plot_opts={
            "grain_image": {"core_set": True},
            "grain_mask": {"core_set": False},
            "grain_mask_image": {"core_set": False},
        },
        **grainstats_config,
    )


# Random shapes
# Generate a random skeletons, first is a skeleton with a closed loop with side branches
kwargs = {
    "image_shape": (60, 32),
    "max_shapes": 10,
    "channel_axis": None,
    "shape": None,
    "allow_overlap": True,
    "min_size": 20,
}


@pytest.fixture()
def utils_skeleton_linear1() -> npt.NDArray:
    """Linear skeleton."""
    random_images, _ = draw.random_shapes(rng=1, **kwargs)
    return skeletonize(random_images != 255)


@pytest.fixture()
def utils_skeleton_linear2() -> npt.NDArray:
    """Linear skeleton T-junction and side-branch."""
    random_images, _ = draw.random_shapes(rng=165103, **kwargs)
    return skeletonize(random_images != 255)


@pytest.fixture()
def utils_skeleton_linear3() -> npt.NDArray:
    """Linear skeleton with several branches."""
    random_images, _ = draw.random_shapes(rng=7334281, **kwargs)
    return skeletonize(random_images != 255)


# DNA Tracing Fixtures
@pytest.fixture()
def minicircle_all_statistics() -> pd.DataFrame:
    """Expected statistics for minicricle."""
    return pd.read_csv(RESOURCES / "minicircle_default_all_statistics.csv", header=0)


# Skeletonizing Fixtures
@pytest.fixture()
def skeletonize_circular() -> npt.NDArray:
    """A circular molecule for testing skeletonizing."""
    return np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0],
            [0, 0, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1, 0, 0],
            [0, 0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 2, 1, 0, 0],
            [0, 0, 1, 2, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 2, 1, 0, 0],
            [0, 0, 1, 2, 3, 4, 3, 2, 2, 2, 2, 2, 2, 2, 3, 4, 3, 2, 1, 0, 0],
            [0, 0, 1, 2, 3, 4, 3, 2, 1, 1, 1, 1, 1, 2, 3, 4, 3, 2, 1, 0, 0],
            [0, 0, 1, 2, 3, 4, 3, 2, 1, 0, 0, 0, 1, 2, 3, 4, 3, 2, 1, 0, 0],
            [0, 0, 1, 2, 3, 4, 3, 2, 1, 0, 0, 0, 1, 2, 3, 4, 3, 2, 1, 0, 0],
            [0, 0, 1, 2, 3, 4, 3, 2, 1, 0, 0, 0, 1, 2, 3, 4, 3, 2, 1, 0, 0],
            [0, 0, 1, 2, 3, 4, 3, 2, 1, 1, 1, 1, 1, 2, 3, 4, 3, 2, 1, 0, 0],
            [0, 0, 1, 2, 3, 4, 3, 2, 2, 2, 2, 2, 2, 2, 3, 4, 3, 2, 1, 0, 0],
            [0, 0, 1, 2, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 2, 1, 0, 0],
            [0, 0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 2, 1, 0, 0],
            [0, 0, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1, 0, 0],
            [0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )


@pytest.fixture()
def skeletonize_circular_bool_int(skeletonize_circular: np.ndarray) -> npt.NDArray:
    """A circular molecule for testing skeletonizing as a boolean integer array."""
    return np.array(skeletonize_circular, dtype="bool").astype(int)


@pytest.fixture()
def skeletonize_linear() -> npt.NDArray:
    """A linear molecule for testing skeletonizing."""
    return np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 2, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 3, 3, 4, 3, 2, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 3, 3, 4, 4, 3, 2, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 4, 4, 3, 3, 2, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 3, 3, 2, 2, 1, 0, 0],
            [0, 0, 0, 0, 1, 2, 2, 2, 3, 3, 3, 4, 4, 3, 2, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 2, 3, 3, 3, 4, 4, 4, 3, 3, 2, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 3, 3, 3, 2, 2, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 2, 3, 4, 4, 3, 3, 2, 2, 2, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 2, 3, 4, 3, 3, 2, 2, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 2, 3, 4, 3, 3, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 2, 3, 4, 3, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 2, 2, 2, 3, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 2, 3, 3, 3, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 2, 3, 4, 4, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 2, 3, 3, 3, 3, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )


@pytest.fixture()
def skeletonize_linear_bool_int(skeletonize_linear) -> npt.NDArray:
    """A linear molecule for testing skeletonizing as a boolean integer array."""
    return np.array(skeletonize_linear, dtype="bool").astype(int)


# Pruning and Height profile fixtures
#
# Skeletons are generated by...
#
# 1. Generate random boolean images using scikit-image.
# 2. Skeletonize these shapes (gives boolean skeletons), these are our targets
# 3. Scale the skeletons by a factor (100)
# 4. Apply Gaussian filter to blur the heights and give an example original image with heights


def _generate_heights(skeleton: npt.NDArray, scale: float = 100, sigma: float = 5.0, cval: float = 20.0) -> npt.NDArray:
    """Generate heights from skeletons by scaling image and applying Gaussian blurring.

    Uses scikit-image 'skimage.filters.gaussian()' to generate heights from skeletons.

    Parameters
    ----------
    skeleton : npt.NDArray
        Binary array of skeleton.
    scale : float
        Factor to scale heights by. Boolean arrays are 0/1 and so the factor will be the height of the skeleton ridge.
    sigma : float
        Standard deviation for Gaussian kernel passed to `skimage.filters.gaussian()'.
    cval : float
        Value to fill past edges of input, passed to `skimage.filters.gaussian()'.

    Returns
    -------
    npt.NDArray
        Array with heights of image based on skeleton which will be the backbone and target.
    """
    return filters.gaussian(skeleton * scale, sigma=sigma, cval=cval)


def _generate_random_skeleton(**extra_kwargs):
    """Generate random skeletons and heights using skimage.draw's random_shapes()."""
    kwargs = {
        "image_shape": (128, 128),
        "max_shapes": 20,
        "channel_axis": None,
        "shape": None,
        "allow_overlap": True,
    }
    # kwargs.update
    heights = {"scale": 1e2, "sigma": 5.0, "cval": 20.0}
    kwargs = {**kwargs, **extra_kwargs}
    random_image, _ = draw.random_shapes(**kwargs)
    mask = random_image != 255
    skeleton = skeletonize(mask)
    return {"original": mask, "img": _generate_heights(skeleton, **heights), "skeleton": skeleton}


@pytest.fixture()
def skeleton_loop1() -> dict:
    """Skeleton with loop to be retained and side-branches."""
    return _generate_random_skeleton(rng=1, min_size=20)


@pytest.fixture()
def skeleton_loop2() -> dict:
    """Skeleton with loop to be retained and side-branches."""
    return _generate_random_skeleton(rng=165103, min_size=60)


@pytest.fixture()
def skeleton_linear1() -> dict:
    """Linear skeleton with lots of large side-branches, some forked."""
    return _generate_random_skeleton(rng=13588686514, min_size=20)


@pytest.fixture()
def skeleton_linear2() -> dict:
    """Linear Skeleton with simple fork at one end."""
    return _generate_random_skeleton(rng=21, min_size=20)


@pytest.fixture()
def skeleton_linear3() -> dict:
    """Linear Skeletons (i.e. multiple) with branches."""
    return _generate_random_skeleton(rng=894632511, min_size=20)


@pytest.fixture()
def pruning_skeleton() -> dict:
    """Smaller skeleton for testing parameters of prune_all_skeletons(). Has a T-junction."""
    return _generate_random_skeleton(rng=69432138, min_size=15, image_shape=(30, 30))


## Helper function visualising for generating skeletons and heights


# import matplotlib.pyplot as plt
# def pruned_plot(gen_shape: dict) -> None:
#     """Plot the original skeleton, its derived height and the pruned skeleton."""
#     img_skeleton = gen_shape
#     pruned = topostatsPrune(
#         img_skeleton["img"],
#         img_skeleton["skeleton"],
#         max_length=-1,
#         height_threshold=90,
#         method_values="min",
#         method_outlier="abs",
#     )
#     pruned_skeleton = pruned._prune_by_length(pruned.skeleton, pruned.max_length)
#     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
#     ax1.imshow(img_skeleton["original"])
#     ax1.set_title("Original mask")
#     ax2.imshow(img_skeleton["skeleton"])
#     ax2.set_title("Skeleton")
#     ax3.imshow(img_skeleton["img"])
#     ax3.set_title("Gaussian Blurring")
#     ax4.imshow(pruned_skeleton)
#     ax4.set_title("Pruned Skeleton")
#     plt.show()


# pruned_plot(pruning_skeleton_loop1())
# pruned_plot(pruning_skeleton_loop2())
# pruned_plot(pruning_skeleton_linear1())
# pruned_plot(pruning_skeleton_linear2())
# pruned_plot(pruning_skeleton_linear3())
# pruned_plot(pruning_skeleton())


# U-Net fixtures
@pytest.fixture()
def mock_model_5_by_5_single_class() -> MagicMock:
    """Create a mock model."""
    model_mocker = MagicMock()

    # Define a custom side effect function for the predict method
    def side_effect_predict(input_array: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        assert input_array.shape == (1, 5, 5, 1), "Input shape is not as expected"
        assert input_array.dtype == np.float32, "Input data type is not as expected"

        input_array_without_batch_and_channel = input_array[0, :, :, 0]
        print(input_array_without_batch_and_channel)

        # Different output for different input
        if np.array_equal(
            input_array_without_batch_and_channel,
            np.array(
                [
                    [0.1, 0.2, 0.2, 0.2, 0.1],
                    [0.1, 1.0, 0.2, 1.0, 0.2],
                    [0.2, 1.0, 0.1, 0.2, 0.2],
                    [0.1, 1.0, 1.0, 1.0, 0.1],
                    [0.1, 0.1, 0.1, 0.2, 0.1],
                ]
            ).astype(np.float32),
        ):
            return (
                np.array(
                    [
                        [0, 0, 0, 0, 0],
                        [0, 1, 0, 1, 0],
                        [0, 1, 0, 0, 0],
                        [0, 1, 1, 1, 0],
                        [0, 0, 0, 0, 0],
                    ]
                )
                .reshape((1, 5, 5, 1))
                .astype(np.float32)
            )
        if np.array_equal(
            input_array_without_batch_and_channel,
            np.array(
                [
                    [0.1, 0.2, 0.1, 0.2, 0.1],
                    [0.1, 1.0, 1.0, 1.0, 0.1],
                    [0.2, 1.0, 1.0, 1.0, 0.2],
                    [0.1, 1.0, 1.0, 1.0, 0.1],
                    [0.1, 0.1, 0.2, 0.2, 0.1],
                ]
            ).astype(np.float32),
        ):
            return (
                np.array(
                    [
                        [0, 0, 0, 0, 0],
                        [0, 1, 1, 1, 0],
                        [0, 1, 0, 1, 0],
                        [0, 1, 1, 1, 0],
                        [0, 0, 0, 0, 0],
                    ]
                )
                .reshape((1, 5, 5, 1))
                .astype(np.float32)
            )
        if np.allclose(
            input_array_without_batch_and_channel,
            np.array(
                [
                    [0.20455678, 0.18093494, 0.13962264, 0.10629401, 0.08889285],
                    [0.18093495, 0.3309646, 0.5164179, 0.28279683, 0.10629401],
                    [0.13962264, 0.5164179, 1.0, 0.5164179, 0.13962264],
                    [0.10629401, 0.28279683, 0.5164179, 0.3309646, 0.18093495],
                    [0.08889285, 0.10629401, 0.13962264, 0.18093494, 0.20455678],
                ]
            ).astype(np.float32),
            atol=1e-6,
        ):
            return (
                np.array(
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ]
                )
                .reshape((1, 5, 5, 1))
                .astype(np.float32)
            )
        if np.array_equal(
            input_array_without_batch_and_channel,
            np.array(
                [
                    [0.1, 0.2, 0.1, 0.2, 0.1],
                    [0.2, 0.1, 1.0, 0.1, 0.2],
                    [0.1, 1.0, 1.0, 1.0, 0.1],
                    [0.2, 0.1, 1.0, 0.1, 0.2],
                    [0.1, 0.2, 0.1, 0.2, 0.1],
                ]
            ).astype(np.float32),
        ):
            return (
                np.array(
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ]
                )
                .reshape((1, 5, 5, 1))
                .astype(np.float32)
            )
        raise ValueError(
            "Input is not as expected. Check the image crop sent to the model and check the"
            "mocked unet predict function has a case for that exact input."
        )

    # Assign the side effect to the mock's predict method
    model_mocker.predict.side_effect = side_effect_predict
    # Override the output of the input_shape property
    model_mocker.input_shape = (1, 5, 5, 1)
    model_mocker.output_shape = (1, 5, 5, 1)

    return model_mocker
