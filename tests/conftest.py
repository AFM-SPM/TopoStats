"""Fixtures for testing."""

from __future__ import annotations

import importlib.resources as pkg_resources
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pandas as pd
import pySPM
import pytest
import yaml
from skimage import draw, filters
from skimage.morphology import skeletonize

import topostats
from topostats.filters import Filters
from topostats.grains import Grains
from topostats.grainstats import GrainStats
from topostats.io import LoadScans, read_yaml
from topostats.plotting import TopoSum
from topostats.tracing.dnatracing import dnaTrace
from topostats.utils import _get_mask, get_mask, get_thresholds

# This is required because of the inheritance used throughout
# pylint: disable=redefined-outer-name
# pylint: disable=too-many-lines
BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"

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
    config["grains"]["threshold_absolute"]["above"] = 1.0
    config["grains"]["threshold_absolute"]["below"] = -1.0
    config["grains"]["smallest_grain_size_nm2"] = 10
    config["grains"]["absolute_area_threshold"]["above"] = [10, 60000000]
    return config


@pytest.fixture()
def process_scan_config() -> dict:
    """Sample configuration."""
    config = read_yaml(BASE_DIR / "topostats" / "default_config.yaml")
    config["filter"]["remove_scars"]["run"] = True
    config["grains"]["threshold_std_dev"]["below"] = 1.0
    config["grains"]["absolute_area_threshold"]["above"] = [500, 800]
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
    return config


@pytest.fixture()
def dnatracing_config(default_config: dict) -> dict:
    """Configurations for dnatracing."""
    config = default_config["dnatracing"]
    config.pop("run")
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
    thresholds = get_thresholds(
        image=test_filters.images["pixels"],
        threshold_method="otsu",
        otsu_threshold_multiplier=filter_config["otsu_threshold_multiplier"],
    )
    test_filters.images["mask"] = get_mask(image=test_filters.images["pixels"], thresholds=thresholds)
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
    return LoadScans([tmp_path / "topostats_file_test.topostats"], **loading_config)


@pytest.fixture()
def load_scan(loading_config: dict) -> LoadScans:
    """Instantiate a LoadScans object from a small .topostats image file."""
    return LoadScans([RESOURCES / "test_image" / "minicircle_small.topostats"], **loading_config)


@pytest.fixture()
def load_scan_data() -> LoadScans:
    """Instance of a LoadScans object after applying the get_data func."""
    scan_data = LoadScans([RESOURCES / "test_image" / "minicircle_small.topostats"], channel="Height")
    scan_data.get_data()
    return scan_data


@pytest.fixture()
def load_scan_spm() -> LoadScans:
    """Instantiate a LoadScans object from a .spm file."""
    return LoadScans([RESOURCES / "minicircle.spm"], channel="Height")


@pytest.fixture()
def spm_channel_data() -> pySPM.SPM.SPM_image:
    """Instantiate channel data from a LoadScans object."""
    scan = pySPM.Bruker(RESOURCES / "minicircle.spm")
    return scan.get_channel("Height")


@pytest.fixture()
def load_scan_ibw() -> LoadScans:
    """Instantiate a LoadScans object from a .ibw file."""
    return LoadScans([RESOURCES / "minicircle2.ibw"], channel="HeightTracee")


@pytest.fixture()
def load_scan_jpk() -> LoadScans:
    """Instantiate a LoadScans object from a .jpk file."""
    return LoadScans([RESOURCES / "file.jpk"], channel="height_trace")


@pytest.fixture()
def load_scan_gwy() -> LoadScans:
    """Instantiate a LoadScans object from a .gwy file."""
    return LoadScans([RESOURCES / "file.gwy"], channel="ZSensor")


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
def minicircle_threshold_otsu(minicircle_initial_tilt_removal: Filters, filter_config: dict) -> Filters:
    """Calculate threshold."""
    minicircle_initial_tilt_removal.thresholds = get_thresholds(
        minicircle_initial_tilt_removal.images["initial_tilt_removal"], **filter_config
    )
    return minicircle_initial_tilt_removal


@pytest.fixture()
def minicircle_threshold_stddev(minicircle_initial_tilt_removal: Filters) -> Filters:
    """Calculate threshold."""
    minicircle_initial_tilt_removal.thresholds = get_thresholds(
        minicircle_initial_tilt_removal.images["initial_tilt_removal"],
        threshold_method="std_dev",
        otsu_threshold_multiplier=None,
        threshold_std_dev={"below": 10.0, "above": 1.0},
    )
    return minicircle_initial_tilt_removal


@pytest.fixture()
def minicircle_threshold_abs(minicircle_initial_tilt_removal: Filters) -> Filters:
    """Calculate threshold."""
    minicircle_initial_tilt_removal.thresholds = get_thresholds(
        minicircle_initial_tilt_removal.images["initial_tilt_removal"],
        threshold_method="absolute",
        otsu_threshold_multiplier=None,
        absolute={"below": -1.5, "above": 1.5},
    )
    return minicircle_initial_tilt_removal


@pytest.fixture()
def minicircle_mask(minicircle_threshold_otsu: Filters) -> Filters:
    """Derive mask based on threshold."""
    minicircle_threshold_otsu.images["mask"] = get_mask(
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
def minicircle_grains(minicircle_grain_gaussian_filter: Grains, grains_config: dict) -> Grains:
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
        **grains_config,
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
        threshold_std_dev={"below": 10.0, "above": 1.0},
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
        absolute={"below": -1.0, "above": 1.0},
    )
    return minicircle_grains


@pytest.fixture()
def minicircle_grain_mask(minicircle_grain_threshold_abs: Grains) -> Grains:
    """Boolean mask."""
    minicircle_grain_threshold_abs.directions["above"] = {}
    minicircle_grain_threshold_abs.directions["above"]["mask_grains"] = _get_mask(
        image=minicircle_grain_threshold_abs.image,
        thresh=minicircle_grain_threshold_abs.thresholds["above"],
        threshold_direction="above",
        img_name=minicircle_grain_threshold_abs.filename,
    )
    return minicircle_grain_threshold_abs


@pytest.fixture()
def minicircle_grain_clear_border(minicircle_grain_mask: np.array) -> Grains:
    """Cleared borders."""
    minicircle_grain_mask.directions["above"]["tidied_border"] = minicircle_grain_mask.tidy_border(
        minicircle_grain_mask.directions["above"]["mask_grains"]
    )
    return minicircle_grain_mask


@pytest.fixture()
def minicircle_grain_remove_noise(minicircle_grain_clear_border: Grains) -> Grains:
    """Fixture to test removing noise."""
    minicircle_grain_clear_border.directions["above"]["removed_noise"] = minicircle_grain_clear_border.remove_noise(
        minicircle_grain_clear_border.directions["above"]["tidied_border"]
    )
    return minicircle_grain_clear_border


@pytest.fixture()
def minicircle_grain_labelled_all(minicircle_grain_remove_noise: Grains) -> Grains:
    """Labelled regions."""
    minicircle_grain_remove_noise.directions["above"]["labelled_regions_01"] = (
        minicircle_grain_remove_noise.label_regions(minicircle_grain_remove_noise.directions["above"]["removed_noise"])
    )
    return minicircle_grain_remove_noise


@pytest.fixture()
def minicircle_minimum_grain_size(minicircle_grain_labelled_all: Grains) -> Grains:
    """Minimum grain size."""
    minicircle_grain_labelled_all.calc_minimum_grain_size(
        minicircle_grain_labelled_all.directions["above"]["labelled_regions_01"]
    )
    return minicircle_grain_labelled_all


@pytest.fixture()
def minicircle_small_objects_removed(minicircle_minimum_grain_size: Grains) -> Grains:
    """Small objects removed."""
    minicircle_minimum_grain_size.directions["above"]["removed_small_objects"] = (
        minicircle_minimum_grain_size.remove_small_objects(
            minicircle_minimum_grain_size.directions["above"]["labelled_regions_01"]
        )
    )
    return minicircle_minimum_grain_size


@pytest.fixture()
def minicircle_area_thresholding(minicircle_grain_labelled_all: Grains) -> Grains:
    """Small objects removed."""
    absolute_area_thresholds = [30, 2000]
    minicircle_grain_labelled_all.directions["above"]["removed_small_objects"] = (
        minicircle_grain_labelled_all.area_thresholding(
            image=minicircle_grain_labelled_all.directions["above"]["labelled_regions_01"],
            area_thresholds=absolute_area_thresholds,
        )
    )
    return minicircle_grain_labelled_all


@pytest.fixture()
def minicircle_grain_labelled_post_removal(minicircle_small_objects_removed: np.array) -> Grains:
    """Labelled regions."""
    minicircle_small_objects_removed.directions["above"]["labelled_regions_02"] = (
        minicircle_small_objects_removed.label_regions(
            minicircle_small_objects_removed.directions["above"]["removed_small_objects"]
        )
    )
    return minicircle_small_objects_removed


@pytest.fixture()
def minicircle_grain_region_properties_post_removal(
    minicircle_grain_labelled_post_removal: np.array,
) -> np.array:
    """Region properties."""
    return minicircle_grain_labelled_post_removal.get_region_properties(
        minicircle_grain_labelled_post_removal.directions["above"]["labelled_regions_02"]
    )


@pytest.fixture()
def minicircle_grain_coloured(minicircle_grain_labelled_post_removal: np.array) -> Grains:
    """Coloured regions."""
    minicircle_grain_labelled_post_removal.directions["above"]["coloured_regions"] = (
        minicircle_grain_labelled_post_removal.colour_regions(
            minicircle_grain_labelled_post_removal.directions["above"]["labelled_regions_02"]
        )
    )
    return minicircle_grain_labelled_post_removal


# Derive fixture for grainstats
@pytest.fixture()
def grainstats(image_random: np.array, grainstats_config: dict, tmp_path) -> GrainStats:
    """Grainstats class for testing functions."""
    return GrainStats(
        image_random,
        image_random,
        pixel_to_nanometre_scaling=0.5,
        base_output_dir=tmp_path,
        **grainstats_config,
    )


# Minicircle
@pytest.fixture()
def minicircle_grainstats(
    minicircle_grain_gaussian_filter: Filters,
    minicircle_grain_labelled_post_removal: Grains,
    load_scan: LoadScans,
    grainstats_config: dict,
    tmp_path: Path,
) -> GrainStats:
    """GrainStats object."""
    return GrainStats(
        data=minicircle_grain_gaussian_filter.images["gaussian_filtered"],
        labelled_data=minicircle_grain_labelled_post_removal.directions["above"]["labelled_regions_02"],
        pixel_to_nanometre_scaling=load_scan.pixel_to_nm_scaling,
        base_output_dir=tmp_path,
        plot_opts={
            "grain_image": {"core_set": True},
            "grain_mask": {"core_set": False},
            "grain_mask_image": {"core_set": False},
        },
        **grainstats_config,
    )


# Derive fixtures for DNA Tracing
GRAINS = np.array(
    [
        [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 2],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 2],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 2],
        [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
        [0, 0, 3, 3, 3, 3, 3, 0, 0, 0, 2],
        [0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 2],
        [0, 0, 3, 3, 3, 3, 3, 0, 0, 0, 2],
        [0, 0, 4, 4, 4, 4, 4, 0, 0, 0, 2],
    ]
)
FULL_IMAGE = RNG.random((GRAINS.shape[0], GRAINS.shape[1]))


@pytest.fixture()
def test_dnatracing() -> dnaTrace:
    """Instantiate a dnaTrace object."""
    return dnaTrace(image=FULL_IMAGE, grain=GRAINS, filename="Test", pixel_to_nm_scaling=1.0)


@pytest.fixture()
def minicircle_dnatracing(
    minicircle_grain_gaussian_filter: Filters,
    minicircle_grain_coloured: Grains,
    dnatracing_config: dict,
) -> dnaTrace:
    """DnaTrace object instantiated with minicircle data."""  # noqa: D403
    dnatracing_config.pop("pad_width")
    dna_traces = dnaTrace(
        image=minicircle_grain_coloured.image.T,
        grain=minicircle_grain_coloured.directions["above"]["labelled_regions_02"],
        filename=minicircle_grain_gaussian_filter.filename,
        pixel_to_nm_scaling=minicircle_grain_gaussian_filter.pixel_to_nm_scaling,
        **dnatracing_config,
    )
    dna_traces.trace_dna()
    return dna_traces


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
    heights = {"scale": 100, "sigma": 5.0, "cval": 20.0}
    random_image, _ = draw.random_shapes(**kwargs, **extra_kwargs)
    mask = random_image != 255
    skeleton = skeletonize(mask)
    return {"img": _generate_heights(skeleton, **heights), "skeleton": skeleton}


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


# Helper functions for visualising skeletons and heights
#
# def pruned_plot(gen_shape: dict) -> None:
#     """Plot the original skeleton, its derived height and the pruned skeleton."""
#     img_skeleton = gen_shape()
#     pruned = topostatsPrune(
#         img_skeleton["heights"],
#         img_skeleton["skeleton"],
#         max_length=-1,
#         height_threshold=90,
#         method_values="min",
#         method_outlier="abs",
#     )
#     pruned_skeleton = pruned._prune_by_length(pruned.skeleton, pruned.max_length)
#     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
#     ax1.imshow(img_skeleton["skeleton"])
#     ax2.imshow(img_skeleton["heights"])
#     ax3.imshow(pruned_skeleton)
#     plt.show()


# pruned_plot(skeleton_loop1)
# pruned_plot(skeleton_loop2)
# pruned_plot(skeleton_linear1)
# pruned_plot(skeleton_linear2)
# pruned_plot(skeleton_linear3)


# U-Net fixtures
@pytest.fixture()
def mock_model_5_by_5() -> MagicMock:
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
        raise ValueError(
            "Input is not as expected. Check the image crop sent to the model and check the"
            "mocked unet predict function has a case for that exact input."
        )

    # Assign the side effect to the mock's predict method
    model_mocker.predict.side_effect = side_effect_predict
    # Override the output of the input_shape property
    model_mocker.input_shape = (1, 5, 5, 1)

    return model_mocker
