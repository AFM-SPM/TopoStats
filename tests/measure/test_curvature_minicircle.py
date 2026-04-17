"""Integration tests for curvature.py."""

from typing import Literal

import numpy as np
import pytest

from topostats.classes import GrainCurvatureStats, MoleculeCurvatureStats, TopoStats
from topostats.measure.curvature import (
    calculate_curvature_stats_image,
)


@pytest.mark.parametrize(
    (
        "smoothing_method",
        "smoothing_gaussian_sigma_nm",
        "smoothing_savgol_window_length_nm",
        "smoothing_savgol_polyorder",
        "curvature_turn_minimum_delay_nm",
        "curvature_turn_threshold_iqr_multiplier",
    ),
    [
        pytest.param("gaussian", 1.0, 10, 2, 20.0, 1.5, id="gaussian smoothing, default turn"),
    ],
)
def test_calculate_curvature_stats_image(
    minicircle_small_topostats: TopoStats,
    smoothing_gaussian_sigma_nm: float,
    smoothing_method: Literal["gaussian", "savitzky_golay"],
    smoothing_savgol_window_length_nm: int,
    smoothing_savgol_polyorder: int,
    curvature_turn_minimum_delay_nm: float,
    curvature_turn_threshold_iqr_multiplier: float,
) -> None:
    """Test the calculation of curvature statistics for an image."""
    # Construct test grains with 'ordered_trace' attribute (dictionary of 'Molecule', each with splined_coords
    # attribute)

    expected_grain_curvature_stats_0 = GrainCurvatureStats(
        num_turns=1,
        curvature_mean=0.127569,
        curvature_max=0.455729,
        curvature_min=0.001343,
        curvature_std=0.090176,
        curvature_var=0.008132,
        curvature_total=13.267195,
        curvature_median=0.104371,
        curvature_iqr=0.344076,
        curvature_90th=0.236088,
    )

    expected_molecule_curvature_stats_0_0 = MoleculeCurvatureStats(
        curvatures=np.array(
            [
                -0.45572924,
                -0.376373,
                -0.26413153,
                -0.17011704,
                -0.11752514,
                -0.10247107,
                -0.10819661,
                -0.11733628,
                -0.12272463,
                -0.12953083,
                -0.1440292,
                -0.16416507,
                -0.18349049,
                -0.19824689,
                -0.20800912,
                -0.21476315,
                -0.2215725,
                -0.22815455,
                -0.23067909,
                -0.22906904,
                -0.22920639,
                -0.2343663,
                -0.23932401,
                -0.2368257,
                -0.22484628,
                -0.20460554,
                -0.17780011,
                -0.14785646,
                -0.11962367,
                -0.09761269,
                -0.08486321,
                -0.08042983,
                -0.07883827,
                -0.07575094,
                -0.07213229,
                -0.06934453,
                -0.06388789,
                -0.05152316,
                -0.03381076,
                -0.01682713,
                -0.00541677,
                -0.00134333,
                -0.00573012,
                -0.0199206,
                -0.04239585,
                -0.06569283,
                -0.07991265,
                -0.08146684,
                -0.07611099,
                -0.07121562,
                -0.06866773,
                -0.06773608,
                -0.06972816,
                -0.07536659,
                -0.08155201,
                -0.08522887,
                -0.08818641,
                -0.09428388,
                -0.10263823,
                -0.1061031,
                -0.10235773,
                -0.10814217,
                -0.15083904,
                -0.23430494,
                -0.32224047,
                -0.36960037,
                -0.36471982,
                -0.33033216,
                -0.29018747,
                -0.24826212,
                -0.20094276,
                -0.15515843,
                -0.12466143,
                -0.11512244,
                -0.12069568,
                -0.13065844,
                -0.13544484,
                -0.13046484,
                -0.11746089,
                -0.10156746,
                -0.08816714,
                -0.08152135,
                -0.08189782,
                -0.08527247,
                -0.08936231,
                -0.09694775,
                -0.10971902,
                -0.12288152,
                -0.1293736,
                -0.12524558,
                -0.10898832,
                -0.08149086,
                -0.04872146,
                -0.02124315,
                -0.00986414,
                -0.02024954,
                -0.04620982,
                -0.06843341,
                -0.06781736,
                -0.04288027,
                -0.00979776,
                0.01420067,
                0.0239454,
                0.0253142,
            ]
        ),
        is_circular=True,
        num_turns=1,
        curvature_mean=0.127569,
        curvature_max=0.455729,
        curvature_min=0.001343,
        curvature_std=0.090176,
        curvature_var=0.008132,
        curvature_total=13.267195,
        curvature_median=0.104371,
        curvature_iqr=0.344076,
        curvature_90th=0.236088,
    )

    calculate_curvature_stats_image(
        topostats_object=minicircle_small_topostats,
        smoothing_method=smoothing_method,
        smoothing_gaussian_sigma_nm=smoothing_gaussian_sigma_nm,
        smoothing_savgol_window_length_nm=smoothing_savgol_window_length_nm,
        smoothing_savgol_polyorder=smoothing_savgol_polyorder,
        curvature_turn_minimum_delay_nm=curvature_turn_minimum_delay_nm,
        curvature_turn_threshold_iqr_multiplier=curvature_turn_threshold_iqr_multiplier,
    )

    result_grain_curvature_stats_0 = minicircle_small_topostats.require_grain_crops()[
        0
    ].ordered_trace.grain_curvature_stats
    assert result_grain_curvature_stats_0 == expected_grain_curvature_stats_0
    result_molecule_curvature_stats_0_0 = (
        minicircle_small_topostats.require_grain_crops()[0].ordered_trace.require_molecule_data()[0].curvature_stats
    )
    assert result_molecule_curvature_stats_0_0 == expected_molecule_curvature_stats_0_0
