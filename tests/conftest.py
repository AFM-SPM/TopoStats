"""Fixtures for testing"""
import importlib.resources as pkg_resources
from pathlib import Path
from typing import Dict
import yaml

import numpy as np
import pandas as pd
import pytest

import topostats
from topostats.filters import Filters
from topostats.grains import Grains
from topostats.grainstats import GrainStats
from topostats.io import read_yaml, LoadScans
from topostats.tracing.dnatracing import dnaTrace, traceStats
from topostats.utils import get_thresholds, get_mask, _get_mask


# This is required because of the inheritance used throughout
# pylint: disable=redefined-outer-name
BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"

RNG = np.random.default_rng(seed=1000)
SMALL_ARRAY_SIZE = (10, 10)
THRESHOLD = 0.5
CHANNEL = "Height"


@pytest.fixture
def default_config() -> Dict:
    """Sample configuration"""
    config = read_yaml(BASE_DIR / "topostats" / "default_config.yaml")
    plotting_dictionary = pkg_resources.open_text(topostats, "plotting_dictionary.yaml")
    config["plotting"]["plot_dict"] = yaml.safe_load(plotting_dictionary.read())
    config["filter"]["threshold_method"] = "otsu"
    config["grains"]["threshold_method"] = "otsu"
    config["grains"]["otsu_threshold_multiplier"] = 1.7
    config["grains"]["absolute_area_threshold"]["upper"] = [400, 600]
    return config


@pytest.fixture
def process_scan_config() -> Dict:
    """Sample configuration"""
    config = read_yaml(BASE_DIR / "topostats" / "default_config.yaml")
    config["grains"]["threshold_std_dev"]["lower"] = 1.0
    config["grains"]["absolute_area_threshold"]["upper"] = [500, 800]
    config["plotting"]["zrange"] = [0, 3]
    plotting_dictionary = pkg_resources.open_text(topostats, "plotting_dictionary.yaml")
    config["plotting"]["plot_dict"] = yaml.safe_load(plotting_dictionary.read())
    return config


@pytest.fixture
def plot_dict() -> Dict:
    """Load the plot_dict dictionary. This is required because the above configs have the 'plot_dict' key/value
    popped."""
    plotting_dictionary = pkg_resources.open_text(topostats, "plotting_dictionary.yaml")
    return yaml.safe_load(plotting_dictionary.read())


@pytest.fixture
def loading_config(default_config: Dict) -> Dict:
    """Configuration for loading scans"""
    config = default_config["loading"]
    return config


@pytest.fixture
def filter_config(default_config: Dict) -> Dict:
    """Configurations for filtering"""
    config = default_config["filter"]
    config.pop("run")
    return config


@pytest.fixture
def grains_config(default_config: Dict) -> Dict:
    """Configurations for grain finding."""
    config = default_config["grains"]
    config.pop("run")
    return config


@pytest.fixture
def grainstats_config(default_config: Dict) -> Dict:
    """Configurations for grainstats"""
    config = default_config["grainstats"]
    config["direction"] = "upper"
    config.pop("run")
    return config


@pytest.fixture
def dnatracing_config(default_config: Dict) -> Dict:
    """Configurations for dnatracing"""
    config = default_config["dnatracing"]
    config.pop("run")
    return config


@pytest.fixture
def plotting_config(default_config: Dict) -> Dict:
    """Configurations for filtering"""
    config = default_config["plotting"]
    config.pop("run")
    config.pop("plot_dict")
    return config


@pytest.fixture
def image_random() -> np.ndarray:
    """Random image as NumPy array."""
    rng = np.random.default_rng(seed=1000)
    return rng.random((1024, 1024))


@pytest.fixture
def small_array() -> np.ndarray:
    """Small (10x10) image array for testing"""
    return RNG.random(SMALL_ARRAY_SIZE)


@pytest.fixture
def small_mask() -> np.ndarray:
    """Small (10x10) mask array for testing."""
    return RNG.uniform(low=0, high=1, size=SMALL_ARRAY_SIZE) > 0.5


@pytest.fixture
def synthetic_scars_image() -> np.array:
    """Small synthetic image for testing scar removal."""
    return np.load(RESOURCES / "test_scars_synthetic_scar_image.npy")


@pytest.fixture
def synthetic_marked_scars() -> np.array:
    """Small synthetic boolean array of marked scar coordinates corresponding to synthetic_scars_image."""
    return np.load(RESOURCES / "test_scars_synthetic_mark_scars.npy")


@pytest.fixture
def image_random_row_medians() -> np.array:
    """Expected row medians (unmasked)."""
    return np.loadtxt(RESOURCES / "image_random_row_medians.csv", delimiter=",")


@pytest.fixture
def image_random_col_medians() -> np.array:
    """Expected column medians (unmasked)."""
    return np.loadtxt(RESOURCES / "image_random_col_medians.csv", delimiter=",")


@pytest.fixture
def image_random_median_flattened() -> np.array:
    """Expected aligned rows (unmasked)."""
    df = pd.read_csv(RESOURCES / "image_random_median_flattened.csv.bz2", header=None)
    return df.to_numpy()


@pytest.fixture
def image_random_remove_x_y_tilt() -> np.array:
    """Expected removed tilt (unmasked)."""
    df = pd.read_csv(RESOURCES / "image_random_remove_x_y_tilt.csv.bz2", header=None)
    return df.to_numpy()


@pytest.fixture
def image_random_remove_quadratic() -> np.array:
    """Expected removed quadratic (unmasked)"""
    df = pd.read_csv(RESOURCES / "image_random_remove_quadratic.csv.bz2", header=None)
    return df.to_numpy()


@pytest.fixture
def image_random_mask() -> np.array:
    """Expected mask."""
    df = pd.read_csv(RESOURCES / "image_random_mask.csv.bz2", header=None)
    return df.to_numpy()


@pytest.fixture
def image_random_row_medians_masked() -> np.array:
    """Expected row medians (masked)."""
    return np.loadtxt(RESOURCES / "image_random_row_medians_masked.csv", delimiter=",")


@pytest.fixture
def image_random_col_medians_masked() -> np.array:
    """Expected column medians (masked)."""
    return np.loadtxt(RESOURCES / "image_random_col_medians_masked.csv", delimiter=",")


@pytest.fixture()
def test_load_scan_minicircle() -> LoadScans:
    """Load the minicricle.spm and return image (np.ndarray), pixel_to_nm_scaling (float) and filename (str) for use in
    subsequent fixtures."""
    scan_loader = LoadScans(RESOURCES / "minicircle.spm", channel="Height")
    scan_loader.get_data()
    return scan_loader


@pytest.fixture
def test_filters(load_scan: LoadScans, filter_config: dict) -> Filters:
    """Filters class for testing."""
    load_scan.get_data()
    filters = Filters(
        image=load_scan.image,
        filename=load_scan.filename,
        pixel_to_nm_scaling=load_scan.pixel_to_nm_scaling,
        **filter_config,
    )
    return filters


@pytest.fixture
def test_filters_random(test_filters: Filters, image_random: np.ndarray) -> Filters:
    """Filters class for testing with pixels replaced by random image."""
    test_filters.images["pixels"] = image_random
    return test_filters


@pytest.fixture
def test_filters_random_with_mask(filter_config: dict, test_filters: Filters, image_random: np.ndarray) -> Filters:
    """Filters class for testing with pixels replaced by random image."""
    test_filters.images["pixels"] = image_random
    thresholds = get_thresholds(
        image=test_filters.images["pixels"],
        threshold_method=filter_config["threshold_method"],
        otsu_threshold_multiplier=filter_config["otsu_threshold_multiplier"],
    )
    test_filters.images["mask"] = get_mask(image=test_filters.images["pixels"], thresholds=thresholds)
    return test_filters


@pytest.fixture
def random_filters(test_filters_random_with_mask: Filters) -> Filters:
    """Process random with filters, for use in grains fixture."""
    test_filters_random_with_mask.images["initial_median_flatten"] = test_filters_random_with_mask.median_flatten(
        test_filters_random_with_mask.images["pixels"], mask=None
    )
    test_filters_random_with_mask.images["initial_tilt_removal"] = test_filters_random_with_mask.remove_tilt(
        test_filters_random_with_mask.images["initial_median_flatten"], mask=None
    )
    test_filters_random_with_mask.images["masked_median_flatten"] = test_filters_random_with_mask.median_flatten(
        test_filters_random_with_mask.images["initial_tilt_removal"], mask=test_filters_random_with_mask.images["mask"]
    )
    test_filters_random_with_mask.images["masked_tilt_removal"] = test_filters_random_with_mask.remove_tilt(
        test_filters_random_with_mask.images["masked_median_flatten"], mask=test_filters_random_with_mask.images["mask"]
    )

    return test_filters_random_with_mask


@pytest.fixture
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


@pytest.fixture
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


@pytest.fixture
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
@pytest.fixture
def load_scan(loading_config: dict) -> LoadScans:
    """Instantiate a LoadScans object from a .spm file."""
    scan_loader = LoadScans([RESOURCES / "minicircle.spm"], **loading_config)
    return scan_loader


@pytest.fixture
def load_scan_data() -> LoadScans:
    """Instance of a LoadScans object after applying the get_data func."""
    scan_data = LoadScans([RESOURCES / "minicircle.spm"], channel="Height")
    scan_data.get_data()
    return scan_data


@pytest.fixture
def load_scan_ibw() -> LoadScans:
    """Instantiate a LoadScans object from a .ibw file."""
    scan_loader = LoadScans([RESOURCES / "minicircle2.ibw"], channel="HeightTracee")
    return scan_loader


@pytest.fixture
def load_scan_jpk() -> LoadScans:
    """Instantiate a LoadScans object from a .jpk file."""
    scan_loader = LoadScans([RESOURCES / "file.jpk"], channel="height_trace")
    return scan_loader


# Minicircle fixtures
@pytest.fixture
def minicircle(load_scan: LoadScans, filter_config: dict) -> Filters:
    """Instantiate a Filters object, creates the output directory and loads the image."""
    load_scan.get_data()
    filters = Filters(
        image=load_scan.image,
        filename=load_scan.filename,
        pixel_to_nm_scaling=load_scan.pixel_to_nm_scaling,
        **filter_config,
    )
    return filters


@pytest.fixture
def minicircle_initial_median_flatten(minicircle: Filters) -> Filters:
    """Initial align on unmasked data."""
    minicircle.images["initial_median_flatten"] = minicircle.median_flatten(minicircle.images["pixels"], mask=None)
    return minicircle


@pytest.fixture
def minicircle_initial_tilt_removal(minicircle_initial_median_flatten: Filters) -> Filters:
    """Initial x/y tilt removal on unmasked data."""
    minicircle_initial_median_flatten.images["initial_tilt_removal"] = minicircle_initial_median_flatten.remove_tilt(
        minicircle_initial_median_flatten.images["initial_median_flatten"], mask=None
    )
    return minicircle_initial_median_flatten


@pytest.fixture
def minicircle_initial_quadratic_removal(minicircle_initial_tilt_removal: Filters) -> Filters:
    """Initial quadratic removal on unmasked data."""
    minicircle_initial_tilt_removal.images[
        "initial_quadratic_removal"
    ] = minicircle_initial_tilt_removal.remove_quadratic(
        minicircle_initial_tilt_removal.images["initial_tilt_removal"], mask=None
    )
    return minicircle_initial_tilt_removal


@pytest.fixture
def minicircle_threshold_otsu(minicircle_initial_tilt_removal: Filters, filter_config: dict) -> Filters:
    """Calculate threshold."""
    minicircle_initial_tilt_removal.thresholds = get_thresholds(
        minicircle_initial_tilt_removal.images["initial_tilt_removal"], **filter_config
    )
    return minicircle_initial_tilt_removal


@pytest.fixture
def minicircle_threshold_stddev(minicircle_initial_tilt_removal: Filters) -> Filters:
    """Calculate threshold."""
    minicircle_initial_tilt_removal.thresholds = get_thresholds(
        minicircle_initial_tilt_removal.images["initial_tilt_removal"],
        threshold_method="std_dev",
        otsu_threshold_multiplier=None,
        threshold_std_dev={"lower": 10.0, "upper": 1.0},
    )
    return minicircle_initial_tilt_removal


@pytest.fixture
def minicircle_threshold_abs(minicircle_initial_tilt_removal: Filters) -> Filters:
    """Calculate threshold."""
    minicircle_initial_tilt_removal.thresholds = get_thresholds(
        minicircle_initial_tilt_removal.images["initial_tilt_removal"],
        threshold_method="absolute",
        otsu_threshold_multiplier=None,
        absolute={"lower": -1.5, "upper": 1.5},
    )
    return minicircle_initial_tilt_removal


@pytest.fixture
def minicircle_mask(minicircle_threshold_otsu: Filters) -> Filters:
    """Derive mask based on threshold."""
    minicircle_threshold_otsu.images["mask"] = get_mask(
        image=minicircle_threshold_otsu.images["initial_tilt_removal"], thresholds=minicircle_threshold_otsu.thresholds
    )
    return minicircle_threshold_otsu


@pytest.fixture
def minicircle_masked_median_flatten(minicircle_mask: Filters) -> Filters:
    """Secondary alignment using mask."""
    minicircle_mask.images["masked_median_flatten"] = minicircle_mask.median_flatten(
        minicircle_mask.images["initial_tilt_removal"], mask=minicircle_mask.images["mask"]
    )
    return minicircle_mask


@pytest.fixture
def minicircle_masked_tilt_removal(minicircle_masked_median_flatten: Filters) -> Filters:
    """Secondary x/y tilt removal using mask."""
    minicircle_masked_median_flatten.images["masked_tilt_removal"] = minicircle_masked_median_flatten.remove_tilt(
        minicircle_masked_median_flatten.images["masked_median_flatten"],
        mask=minicircle_masked_median_flatten.images["mask"],
    )
    return minicircle_masked_median_flatten


@pytest.fixture
def minicircle_masked_quadratic_removal(minicircle_masked_tilt_removal: Filters) -> Filters:
    """Secondary quadratic removal using mask."""
    minicircle_masked_tilt_removal.images["masked_quadratic_removal"] = minicircle_masked_tilt_removal.remove_quadratic(
        minicircle_masked_tilt_removal.images["masked_tilt_removal"], mask=minicircle_masked_tilt_removal.images["mask"]
    )
    return minicircle_masked_tilt_removal


@pytest.fixture
def minicircle_grain_gaussian_filter(minicircle_masked_quadratic_removal: Filters) -> Filters:
    """Apply Gaussian filter."""
    minicircle_masked_quadratic_removal.images[
        "gaussian_filtered"
    ] = minicircle_masked_quadratic_removal.gaussian_filter(
        image=minicircle_masked_quadratic_removal.images["masked_quadratic_removal"]
    )
    return minicircle_masked_quadratic_removal


# Derive fixtures for grain finding
@pytest.fixture
def minicircle_grains(minicircle_grain_gaussian_filter: Filters, grains_config: dict) -> Grains:
    """Grains object based on filtered minicircle."""
    grains = Grains(
        image=minicircle_grain_gaussian_filter.images["gaussian_filtered"],
        filename=minicircle_grain_gaussian_filter.filename,
        pixel_to_nm_scaling=minicircle_grain_gaussian_filter.pixel_to_nm_scaling,
        **grains_config,
    )
    return grains


@pytest.fixture
def minicircle_grain_threshold_otsu(minicircle_grains: np.array, grains_config: dict) -> Grains:
    """Calculate threshold."""
    grains_config.pop("threshold_method")
    grains_config["threshold_method"] = "otsu"
    minicircle_grains.thresholds = get_thresholds(
        image=minicircle_grains.image,
        **grains_config,
    )
    return minicircle_grains


@pytest.fixture
def minicircle_grain_threshold_stddev(minicircle_grains: np.array, grains_config: dict) -> Grains:
    """Calculate threshold."""
    grains_config["threshold_method"] = "std_dev"
    minicircle_grains.thresholds = get_thresholds(
        image=minicircle_grains.image,
        threshold_method="std_dev",
        otsu_threshold_multiplier=None,
        threshold_std_dev={"lower": 10.0, "upper": 1.0},
        absolute=None,
    )
    return minicircle_grains


@pytest.fixture
def minicircle_grain_threshold_abs(minicircle_grains: np.array) -> Grains:
    """Calculate threshold."""
    minicircle_grains.thresholds = get_thresholds(
        image=minicircle_grains.image,
        threshold_method="absolute",
        otsu_threshold_multiplier=None,
        absolute={"lower": -1.0, "upper": 1.0},
    )
    return minicircle_grains


@pytest.fixture
def minicircle_grain_mask(minicircle_grain_threshold_otsu: Grains) -> Grains:
    """Boolean mask."""
    minicircle_grain_threshold_otsu.directions["upper"] = {}
    minicircle_grain_threshold_otsu.directions["upper"]["mask_grains"] = _get_mask(
        image=minicircle_grain_threshold_otsu.image,
        threshold=minicircle_grain_threshold_otsu.thresholds["upper"],
        threshold_direction="upper",
        img_name=minicircle_grain_threshold_otsu.filename,
    )
    return minicircle_grain_threshold_otsu


@pytest.fixture
def minicircle_grain_clear_border(minicircle_grain_mask: np.array) -> Grains:
    """Cleared borders."""
    minicircle_grain_mask.directions["upper"]["tidied_border"] = minicircle_grain_mask.tidy_border(
        minicircle_grain_mask.directions["upper"]["mask_grains"]
    )
    return minicircle_grain_mask


@pytest.fixture
def minicircle_grain_remove_noise(minicircle_grain_clear_border: np.array) -> Grains:
    """Fixture to test removing noise."""
    minicircle_grain_clear_border.directions["upper"]["removed_noise"] = minicircle_grain_clear_border.remove_noise(
        minicircle_grain_clear_border.directions["upper"]["tidied_border"]
    )
    return minicircle_grain_clear_border


@pytest.fixture
def minicircle_grain_labelled_all(minicircle_grain_remove_noise: np.array) -> Grains:
    """Labelled regions."""
    minicircle_grain_remove_noise.directions["upper"][
        "labelled_regions_01"
    ] = minicircle_grain_remove_noise.label_regions(minicircle_grain_remove_noise.directions["upper"]["removed_noise"])
    return minicircle_grain_remove_noise


@pytest.fixture
def minicircle_minimum_grain_size(minicircle_grain_labelled_all: np.array) -> float:
    """Minimum grain size."""
    minicircle_grain_labelled_all.calc_minimum_grain_size(
        minicircle_grain_labelled_all.directions["upper"]["labelled_regions_01"]
    )
    return minicircle_grain_labelled_all


@pytest.fixture
def minicircle_small_objects_removed(minicircle_minimum_grain_size: np.array) -> Grains:
    """Small objects removed."""
    minicircle_minimum_grain_size.directions["upper"][
        "removed_small_objects"
    ] = minicircle_minimum_grain_size.remove_small_objects(
        minicircle_minimum_grain_size.directions["upper"]["labelled_regions_01"]
    )
    return minicircle_minimum_grain_size


@pytest.fixture
def minicircle_area_thresholding(minicircle_grain_labelled_all: np.array) -> Grains:
    """Small objects removed."""
    absolute_area_thresholds = [400, 600]
    minicircle_grain_labelled_all.directions["upper"][
        "removed_small_objects"
    ] = minicircle_grain_labelled_all.area_thresholding(
        image=minicircle_grain_labelled_all.directions["upper"]["labelled_regions_01"],
        area_thresholds=absolute_area_thresholds,
    )
    return minicircle_grain_labelled_all


@pytest.fixture
def minicircle_grain_labelled_post_removal(minicircle_small_objects_removed: np.array) -> Grains:
    """Labelled regions."""
    minicircle_small_objects_removed.directions["upper"][
        "labelled_regions_02"
    ] = minicircle_small_objects_removed.label_regions(
        minicircle_small_objects_removed.directions["upper"]["removed_small_objects"]
    )
    return minicircle_small_objects_removed


@pytest.fixture
def minicircle_grain_region_properties_post_removal(minicircle_grain_labelled_post_removal: np.array) -> np.array:
    """Region properties."""
    return minicircle_grain_labelled_post_removal.get_region_properties(
        minicircle_grain_labelled_post_removal.directions["upper"]["labelled_regions_02"]
    )


@pytest.fixture
def minicircle_grain_coloured(minicircle_grain_labelled_post_removal: np.array) -> Grains:
    """Coloured regions."""
    minicircle_grain_labelled_post_removal.directions["upper"][
        "coloured_regions"
    ] = minicircle_grain_labelled_post_removal.colour_regions(
        minicircle_grain_labelled_post_removal.directions["upper"]["labelled_regions_02"]
    )
    return minicircle_grain_labelled_post_removal


# Derive fixture for grainstats
@pytest.fixture
def grainstats(image_random: np.array, grainstats_config: dict, tmp_path) -> GrainStats:
    """Grainstats class for testing functions."""
    gstats = GrainStats(
        image_random,
        image_random,
        pixel_to_nanometre_scaling=0.5,
        base_output_dir=tmp_path,
        **grainstats_config,
    )
    return gstats


# Minicircle
@pytest.fixture
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
        labelled_data=minicircle_grain_labelled_post_removal.directions["upper"]["labelled_regions_02"],
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


@pytest.fixture
def test_dnatracing() -> dnaTrace:
    """Instantiate a dnaTrace object."""
    return dnaTrace(full_image_data=FULL_IMAGE, grains=GRAINS, filename="Test", pixel_size=1.0)


@pytest.fixture
def minicircle_dnatracing(
    minicircle_grain_gaussian_filter: Filters, minicircle_grain_coloured: Grains, dnatracing_config: dict
) -> dnaTrace:
    """dnaTrace object instantiated with minicircle data."""
    dna_traces = dnaTrace(
        full_image_data=minicircle_grain_coloured.image.T,
        grains=minicircle_grain_coloured.directions["upper"]["labelled_regions_02"],
        filename=minicircle_grain_gaussian_filter.filename,
        pixel_size=minicircle_grain_gaussian_filter.pixel_to_nm_scaling,
        **dnatracing_config,
    )
    dna_traces.trace_dna()
    return dna_traces


@pytest.fixture
def minicircle_tracestats(minicircle_dnatracing: dnaTrace) -> pd.DataFrame:
    """DNA Tracing Statistics"""
    tracing_stats = traceStats(trace_object=minicircle_dnatracing, image_path="tmp")
    return tracing_stats.df


# DNA Tracing Fixtures
@pytest.fixture
def minicircle_all_statistics() -> pd.DataFrame:
    """Expected statistics for minicricle."""
    return pd.read_csv(RESOURCES / "minicircle_default_all_statistics.csv", header=0)


# Skeletonizing Fixtures
@pytest.fixture
def skeletonize_circular() -> np.ndarray:
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


@pytest.fixture
def skeletonize_circular_bool_int(skeletonize_circular: np.ndarray) -> np.ndarray:
    """A circular molecule for testing skeletonizing as a boolean integer array."""
    return np.array(skeletonize_circular, dtype="bool").astype(int)


@pytest.fixture
def skeletonize_linear() -> np.ndarray:
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


@pytest.fixture
def skeletonize_linear_bool_int(skeletonize_linear) -> np.ndarray:
    """A linear molecule for testing skeletonizing as a boolean integer array."""
    return np.array(skeletonize_linear, dtype="bool").astype(int)


# Curvature Fixtures
