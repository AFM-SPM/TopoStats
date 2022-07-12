"""Fixtures for testing"""
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from topostats.filters import Filters
from topostats.grains import Grains
from topostats.grainstats import GrainStats
from topostats.io import read_yaml
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
def sample_config() -> Dict:
    """Sample configuration"""
    return read_yaml(RESOURCES / "sample_config.yaml")


@pytest.fixture
def grain_config(sample_config) -> dict:
    """Configurations for grain finding."""
    return sample_config["grains"]


@pytest.fixture
def filter_config(sample_config) -> dict:
    """Configurations for filtering"""
    return sample_config["filter"]


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
def image_random_row_medians() -> np.array:
    """Expected row medians (unmasked)."""
    return np.loadtxt(RESOURCES / "image_random_row_medians.csv", delimiter=",")


@pytest.fixture
def image_random_col_medians() -> np.array:
    """Expected column medians (unmasked)."""
    return np.loadtxt(RESOURCES / "image_random_col_medians.csv", delimiter=",")


@pytest.fixture
def image_random_aligned_rows() -> np.array:
    """Expected aligned rows (unmasked)."""
    df = pd.read_csv(RESOURCES / "image_random_aligned_rows.csv.bz2", header=None)
    return df.to_numpy()


@pytest.fixture
def image_random_remove_x_y_tilt() -> np.array:
    """Expected removed tilt (unmasked)."""
    df = pd.read_csv(RESOURCES / "image_random_remove_x_y_tilt.csv.bz2", header=None)
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


@pytest.fixture
def test_filters(filter_config: dict, sample_config: dict, tmpdir) -> Filters:
    """Filters class for testing."""
    filters = Filters(
        RESOURCES / "minicircle.spm",
        amplify_level=sample_config["amplify_level"],
        threshold_method=filter_config["threshold"]["method"],
        output_dir=tmpdir,
    )
    filters.load_scan()
    return filters


@pytest.fixture
def test_filters_random(sample_config: dict, tmpdir, image_random: np.array) -> Filters:
    """Filters class for testing with pixels replaced by random image."""
    filters = Filters(RESOURCES / "minicircle.spm", amplify_level=sample_config["amplify_level"], output_dir=tmpdir)
    filters.load_scan()
    filters.extract_channel()
    filters.extract_pixels()
    filters.pixels = image_random
    return filters


@pytest.fixture
def test_filters_random_with_mask(sample_config: dict, tmpdir, image_random: np.array) -> Filters:
    """Filters class for testing with pixels replaced by random image."""
    filters = Filters(RESOURCES / "minicircle.spm", amplify_level=sample_config["amplify_level"], output_dir=tmpdir)
    filters.load_scan()
    filters.extract_channel()
    filters.extract_pixels()
    filters.images["pixels"] = image_random
    # filters.get_threshold(filters.images["pixels"])
    thresholds = get_thresholds(
        image=filters.images["pixels"],
        threshold_method=sample_config["filter"]["threshold"]["method"],
        otsu_threshold_multiplier=sample_config["filter"]["threshold"]["otsu_multiplier"],
    )
    # filters.get_mask(filters.images["pixels"])
    filters.images["mask"] = get_mask(image=filters.images["pixels"], thresholds=thresholds)
    return filters


@pytest.fixture
def random_filters(test_filters_random_with_mask: Filters) -> Filters:
    """Process random with filters, for use in grains fixture."""
    test_filters_random_with_mask.images["initial_align"] = test_filters_random_with_mask.align_rows(
        test_filters_random_with_mask.images["pixels"], mask=None
    )
    test_filters_random_with_mask.images["initial_tilt_removal"] = test_filters_random_with_mask.remove_tilt(
        test_filters_random_with_mask.images["initial_align"], mask=None
    )
    test_filters_random_with_mask.images["masked_align"] = test_filters_random_with_mask.align_rows(
        test_filters_random_with_mask.images["initial_tilt_removal"], mask=test_filters_random_with_mask.images["mask"]
    )
    test_filters_random_with_mask.images["masked_tilt_removal"] = test_filters_random_with_mask.remove_tilt(
        test_filters_random_with_mask.images["masked_align"], mask=test_filters_random_with_mask.images["mask"]
    )

    test_filters_random_with_mask.images["zero_averaged_background"] = test_filters_random_with_mask.average_background(
        test_filters_random_with_mask.images["masked_tilt_removal"], mask=test_filters_random_with_mask.images["mask"]
    )
    return test_filters_random_with_mask


@pytest.fixture
def random_grains(grain_config: dict, random_filters: Filters, tmpdir) -> Grains:
    """Grains object based on random image which has no grains."""
    grains = Grains(
        image=random_filters.images["zero_averaged_background"],
        filename="random",
        pixel_to_nm_scaling=0.5,
        gaussian_size=grain_config["gaussian_size"],
        gaussian_mode=grain_config["gaussian_mode"],
        threshold_method=grain_config["threshold"]["method"],
        otsu_threshold_multiplier=grain_config["threshold"]["otsu_multiplier"],
        background=grain_config["background"],
        base_output_dir=Path(tmpdir),
    )
    grains.find_grains()
    return grains


## Minicircle fixtures
@pytest.fixture
def minicircle(sample_config: dict, tmpdir) -> Filters:
    """Instantiate a Filters object, creates the output directory and loads the image."""
    filters = Filters(
        img_path=RESOURCES / "minicircle.spm",
        channel=sample_config["channel"],
        amplify_level=sample_config["amplify_level"],
        output_dir=tmpdir,
    )
    return filters


@pytest.fixture
def minicircle_filename(minicircle) -> Filters:
    """Extract the filename."""
    minicircle.extract_filename()
    return minicircle


@pytest.fixture
def minicircle_load_scan(minicircle) -> Filters:
    """Test loading of scan."""
    minicircle.load_scan()
    return minicircle


@pytest.fixture
def minicircle_make_output_directory(minicircle) -> Filters:
    """Make output directory."""
    minicircle.make_output_directory()
    return minicircle


@pytest.fixture
def minicircle_channel(minicircle) -> Filters:
    """Extract the image channel."""
    minicircle.extract_channel()
    return minicircle


@pytest.fixture
def minicircle_pixels(minicircle_channel) -> Filters:
    """Extract Pixels"""
    minicircle_channel.extract_pixels()
    return minicircle_channel


@pytest.fixture
def minicircle_extract_pixel_to_nm_scaling(minicircle_channel) -> Filters:
    """Extract the pixel to nm scaling"""
    minicircle_channel.extract_pixel_to_nm_scaling()
    return minicircle_channel


@pytest.fixture
def minicircle_initial_align(minicircle_pixels: np.array) -> Filters:
    """Initial align on unmasked data."""
    minicircle_pixels.extract_pixel_to_nm_scaling()
    minicircle_pixels.images["initial_align"] = minicircle_pixels.align_rows(
        minicircle_pixels.images["pixels"], mask=None
    )
    return minicircle_pixels


@pytest.fixture
def minicircle_initial_tilt_removal(minicircle_initial_align: np.array) -> Filters:
    """Initial x/y tilt removal on unmasked data."""
    minicircle_initial_align.images["initial_tilt_removal"] = minicircle_initial_align.remove_tilt(
        minicircle_initial_align.images["initial_align"], mask=None
    )
    return minicircle_initial_align


@pytest.fixture
def minicircle_threshold_otsu(minicircle_initial_tilt_removal: np.array, sample_config: dict) -> Filters:
    """Calculate threshold."""
    minicircle_initial_tilt_removal.thresholds = get_thresholds(
        minicircle_initial_tilt_removal.images["initial_tilt_removal"],
        threshold_method=sample_config["filter"]["threshold"]["method"],
        otsu_threshold_multiplier=sample_config["filter"]["threshold"]["otsu_multiplier"],
    )
    return minicircle_initial_tilt_removal


@pytest.fixture
def minicircle_threshold_stddev(minicircle_initial_tilt_removal: np.array) -> Filters:
    """Calculate threshold."""
    minicircle_initial_tilt_removal.thresholds = get_thresholds(
        minicircle_initial_tilt_removal.images["initial_tilt_removal"],
        threshold_method="std_dev",
        otsu_threshold_multiplier=None,
        deviation_from_mean=1.0,
    )
    return minicircle_initial_tilt_removal


@pytest.fixture
def minicircle_threshold_abs(minicircle_initial_tilt_removal: np.array) -> Filters:
    """Calculate threshold."""
    minicircle_initial_tilt_removal.thresholds = get_thresholds(
        minicircle_initial_tilt_removal.images["initial_tilt_removal"],
        threshold_method="absolute",
        otsu_threshold_multiplier=None,
        absolute=(-1.5, 1.5),
    )
    return minicircle_initial_tilt_removal


@pytest.fixture
def minicircle_mask(minicircle_threshold_otsu: np.array) -> Filters:
    """Derive mask based on threshold."""
    minicircle_threshold_otsu.images["mask"] = get_mask(
        image=minicircle_threshold_otsu.images["initial_tilt_removal"], thresholds=minicircle_threshold_otsu.thresholds
    )
    return minicircle_threshold_otsu


@pytest.fixture
def minicircle_masked_align(minicircle_mask: np.array) -> np.array:
    """Secondary alignment using mask."""
    minicircle_mask.images["masked_align"] = minicircle_mask.align_rows(
        minicircle_mask.images["initial_tilt_removal"], mask=minicircle_mask.images["mask"]
    )
    return minicircle_mask


@pytest.fixture
def minicircle_masked_tilt_removal(minicircle_masked_align: np.array) -> np.array:
    """Secondary x/y tilt removal using mask."""
    minicircle_masked_align.images["masked_tilt_removal"] = minicircle_masked_align.remove_tilt(
        minicircle_masked_align.images["masked_align"], mask=minicircle_masked_align.images["mask"]
    )
    return minicircle_masked_align


@pytest.fixture
def minicircle_zero_average_background(minicircle_masked_tilt_removal: np.array) -> np.array:
    """Zero average background"""
    minicircle_masked_tilt_removal.images[
        "zero_averaged_background"
    ] = minicircle_masked_tilt_removal.average_background(
        minicircle_masked_tilt_removal.images["masked_tilt_removal"], mask=minicircle_masked_tilt_removal.images["mask"]
    )
    return minicircle_masked_tilt_removal


## Derive fixtures for grain finding
@pytest.fixture
def small_array_grains(small_array: np.ndarray, grain_config: dict, tmpdir) -> Grains:
    """Grains object based on small_array."""
    grains = Grains(
        image=small_array,
        filename="small_array",
        pixel_to_nm_scaling=0.5,
        gaussian_size=grain_config["gaussian_size"],
        gaussian_mode=grain_config["gaussian_mode"],
        threshold_method=grain_config["threshold"]["method"],
        otsu_threshold_multiplier=grain_config["threshold"]["otsu_multiplier"],
        background=grain_config["background"],
        base_output_dir=Path(tmpdir),
    )
    return grains


@pytest.fixture
def minicircle_grains(minicircle_zero_average_background: Filters, grain_config: dict, tmpdir) -> Grains:
    """Grains object based on filtered minicircle."""
    grains = Grains(
        image=minicircle_zero_average_background.images["zero_averaged_background"],
        filename=minicircle_zero_average_background.filename,
        pixel_to_nm_scaling=minicircle_zero_average_background.pixel_to_nm_scaling,
        gaussian_size=grain_config["gaussian_size"],
        gaussian_mode=grain_config["gaussian_mode"],
        threshold_method=grain_config["threshold"]["method"],
        otsu_threshold_multiplier=grain_config["threshold"]["otsu_multiplier"],
        background=grain_config["background"],
        base_output_dir=Path(tmpdir),
        absolute_smallest_grain_size=grain_config["absolute_smallest_grain_size"],
    )
    return grains


@pytest.fixture
def minicircle_grain_threshold_otsu(minicircle_grains: np.array) -> Grains:
    """Calculate threshold."""
    minicircle_grains.thresholds = get_thresholds(
        image=minicircle_grains.image,
        threshold_method=minicircle_grains.threshold_method,
        otsu_threshold_multiplier=minicircle_grains.otsu_threshold_multiplier,
    )
    return minicircle_grains


@pytest.fixture
def minicircle_grain_threshold_stddev(minicircle_grains: np.array) -> Grains:
    """Calculate threshold."""
    minicircle_grains.thresholds = get_thresholds(
        image=minicircle_grains.image,
        threshold_method="std_dev",
        otsu_threshold_multiplier=None,
        deviation_from_mean=1.0,
    )
    return minicircle_grains


@pytest.fixture
def minicircle_grain_threshold_abs(minicircle_grains: np.array) -> Grains:
    """Calculate threshold."""
    minicircle_grains.thresholds = get_thresholds(
        image=minicircle_grains.image, threshold_method="absolute", otsu_threshold_multiplier=None, absolute=(-1.0, 1.0)
    )
    return minicircle_grains


@pytest.fixture
def minicircle_grain_gaussian_filter(minicircle_grain_threshold_otsu: np.array) -> Grains:
    """Apply Gaussian filter."""
    minicircle_grain_threshold_otsu.gaussian_filter()
    return minicircle_grain_threshold_otsu


@pytest.fixture
def minicircle_grain_mask(minicircle_grain_gaussian_filter: np.array) -> Grains:
    """Boolean mask."""
    minicircle_grain_gaussian_filter.directions["upper"] = {}
    minicircle_grain_gaussian_filter.directions["upper"]["mask_grains"] = _get_mask(
        image=minicircle_grain_gaussian_filter.images["gaussian_filtered"],
        threshold=minicircle_grain_gaussian_filter.thresholds["upper"],
        threshold_direction="upper",
        img_name=minicircle_grain_gaussian_filter.filename,
    )
    return minicircle_grain_gaussian_filter


@pytest.fixture
def minicircle_grain_clear_border(minicircle_grain_mask: np.array) -> Grains:
    """Cleared borders."""
    minicircle_grain_mask.directions["upper"]["tidied_border"] = minicircle_grain_mask.tidy_border(
        minicircle_grain_mask.directions["upper"]["mask_grains"]
    )
    return minicircle_grain_mask


@pytest.fixture
def minicircle_grain_remove_noise(minicircle_grain_clear_border: np.array) -> Grains:
    """Cleared borders."""
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
def grainstats(image_random: np.array, minicircle_filename: str, tmpdir) -> GrainStats:
    """Grainstats class for testing functions."""
    gstats = GrainStats(
        image_random,
        image_random,
        pixel_to_nanometre_scaling=0.5,
        direction=minicircle_filename,
        base_output_dir=tmpdir,
    )
    return gstats


# Minicircle
@pytest.fixture
def minicircle_grainstats(
    minicircle_grain_gaussian_filter: np.array,
    minicircle_grain_labelled_post_removal: np.array,
    minicircle_extract_pixel_to_nm_scaling: float,
    minicircle_filename,
    tmpdir: Path,
) -> GrainStats:
    """GrainStats object."""
    return GrainStats(
        data=minicircle_grain_gaussian_filter.images["gaussian_filtered"],
        labelled_data=minicircle_grain_labelled_post_removal.directions["upper"]["labelled_regions_02"],
        pixel_to_nanometre_scaling=minicircle_extract_pixel_to_nm_scaling.pixel_to_nm_scaling,
        direction=minicircle_filename.filename,
        base_output_dir=tmpdir,
    )


# Target statistics
#
# These are date specific as we expect statistics to change as the underlying methods used to calculate them
# are tweaked.
@pytest.fixture
def minicircle_grainstats_20220526() -> pd.DataFrame:
    """Statistics for minicircle for comparison."""
    return pd.read_csv(RESOURCES / "minicircle_grainstats_20220526.csv", index_col=0)


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
def minicircle_dnatracing(minicircle_grain_coloured, minicircle_zero_average_background) -> pd.DataFrame:
    """DNA Tracing Statistics"""
    dna_traces = dnaTrace(
        full_image_data=minicircle_grain_coloured.images["gaussian_filtered"].T,
        grains=minicircle_grain_coloured.directions["upper"]["labelled_regions_02"],
        filename=minicircle_zero_average_background.filename,
        pixel_size=minicircle_zero_average_background.pixel_to_nm_scaling,
    )
    dna_traces.trace_dna()
    tracing_stats = traceStats(trace_object=dna_traces, image_path="tmp")
    return tracing_stats.df


@pytest.fixture
def minicircle_dnastats() -> pd.DataFrame:
    """DNA Statistics for minicircle for comparison."""
    return pd.read_csv(RESOURCES / "dna_tracing.csv", index_col=0)
