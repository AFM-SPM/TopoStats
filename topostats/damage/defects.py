"""Scripts for finding defects in traces."""

from typing import Literal

import numpy as np

from topostats.damage.array_manipulation import distances_nm
from topostats.damage.damage import GrainCollection, MoleculeDefectData, get_defects_and_gaps_from_bool_array


def find_curvature_defects(
    grain_collection: GrainCollection,
    curvature_defect_method: Literal["iqr", "absolute"],
    curvature_threshold_iqr_multiplier: float,
    curvature_threshold_absolute_pernm: float,
) -> set[int]:
    """
    Find curvature defects for all molecules in all grains in the grain collection.

    Parameters
    ----------
    grain_collection : GrainCollection
        The GrainCollection.
    curvature_defect_method : Literal["iqr", "absolute"]
        The method to use for finding curvature defects. Options are "iqr" for interquartile range method or "absolute"
        for an absolute threshold in inverse nanometres.
    curvature_threshold_iqr_multiplier : float
        The multiplier for the interquartile range when using the "iqr" method for curvature defect detection.
    curvature_threshold_absolute_pernm : float
        The absolute threshold in inverse nanometres for curvature defect detection when using the "absolute"
        method.
    """
    # find curvature defects
    bad_grains = set()
    if curvature_defect_method == "iqr":
        # iterate over each grain
        for global_grain_id, grain_model in grain_collection.items():
            for molecule_id, molecule_data in grain_model.molecule_data_collection.items():
                molecule_data_curvature_data = molecule_data.curvature_data
                if molecule_data_curvature_data is None:
                    print(
                        f"no curvature data for grain {global_grain_id} molecule {molecule_id}, skipping curvature defect detection for this molecule"
                    )
                    bad_grains.add(global_grain_id)
                    continue
                curvatures = molecule_data_curvature_data["smoothed_curvatures"]
                assert isinstance(
                    curvatures, np.ndarray
                ), f"expected curvatures to be a numpy array, but got {type(curvatures)}"
                # calculate the curvature thresholds
                iqr = np.percentile(curvatures, 75) - np.percentile(curvatures, 25)
                # the threshold is the median + factor * iqr
                curvature_threshold_iqr = curvature_threshold_iqr_multiplier * iqr + np.percentile(curvatures, 50)
                curvature_defects_bool = np.abs(curvatures) > curvature_threshold_iqr

                ordered_defect_gap_list = get_defects_and_gaps_from_bool_array(
                    defects_bool=curvature_defects_bool,
                    trace_points_nm=molecule_data.spline_coords,
                    distance_to_previous_points_nm=distances_nm(
                        molecule_data.spline_coords, circular=molecule_data.circular
                    ),
                    circular=molecule_data.circular,
                )

                grain_model.curvature_defect_data.molecule_defect_data_dict[molecule_id] = MoleculeDefectData(
                    ordered_defects_and_gaps=ordered_defect_gap_list
                )
    elif curvature_defect_method == "absolute":
        for global_grain_id, grain_model in grain_collection.items():
            for molecule_id, molecule_data in grain_model.molecule_data_collection.items():
                molecule_data_curvature_data = molecule_data.curvature_data
                if molecule_data_curvature_data is None:
                    print(
                        f"no curvature data for grain {global_grain_id} molecule {molecule_id}, skipping curvature defect detection for this molecule"
                    )
                    bad_grains.add(global_grain_id)
                    continue
                curvatures = molecule_data_curvature_data["smoothed_curvatures"]
                assert isinstance(
                    curvatures, np.ndarray
                ), f"expected curvatures to be a numpy array, but got {type(curvatures)}"
                curvature_threshold_absolute = curvature_threshold_absolute_pernm / molecule_data.pixel_to_nm_scaling
                curvature_defects_bool = np.abs(curvatures) > curvature_threshold_absolute

                ordered_defect_gap_list = get_defects_and_gaps_from_bool_array(
                    defects_bool=curvature_defects_bool,
                    trace_points_nm=molecule_data.spline_coords,
                    distance_to_previous_points_nm=distances_nm(
                        molecule_data.spline_coords, circular=molecule_data.circular
                    ),
                    circular=molecule_data.circular,
                )

                grain_model.curvature_defect_data.molecule_defect_data_dict[molecule_id] = MoleculeDefectData(
                    ordered_defects_and_gaps=ordered_defect_gap_list
                )
    else:
        raise ValueError(f"Invalid curvature defect method: {curvature_defect_method}")

    return bad_grains


def find_height_defects(
    grain_collection: GrainCollection,
    height_defect_method: Literal["iqr", "absolute"],
    height_threshold_iqr_multiplier: float,
    height_threshold_absolute_nm: float,
) -> set[int]:
    """
    Find height defects for all molecules in all grains in the grain collection.

    Parameters
    ----------
    grain_collection : GrainCollection
        The GrainCollection.
    height_defect_method : Literal["iqr", "absolute"]
        The method to use for finding height defects. Options are "iqr" for interquartile range method or "absolute"
        for an absolute threshold in nanometres.
    height_threshold_iqr_multiplier : float
        The multiplier for the interquartile range when using the "iqr" method for height defect detection.
    height_threshold_absolute_nm : float
        The absolute threshold in nanometres for height defect detection when using the "absolute" method.

    Returns
    -------
    set[int]
        A set of global grain IDs for which height defect detection failed.
    """
    bad_grains = set()
    if height_defect_method == "iqr":
        for global_grain_id, grain_model in grain_collection.items():
            for molecule_id, molecule_data in grain_model.molecule_data_collection.items():
                heights_nm = molecule_data.spline_coords_heights
                spline_coords = molecule_data.spline_coords
                assert len(heights_nm) == len(spline_coords), (
                    f"length of heights does not match length of spline coords "
                    f"for grain {global_grain_id} molecule {molecule_id}, got {len(heights_nm)} heights and "
                    f"{len(spline_coords)} spline coords"
                )
                iqr = np.percentile(heights_nm, 75) - np.percentile(heights_nm, 25)
                height_threshold_iqr = np.percentile(heights_nm, 50) - height_threshold_iqr_multiplier * iqr
                height_defects_bool = heights_nm < height_threshold_iqr

                ordered_defect_gap_list = get_defects_and_gaps_from_bool_array(
                    defects_bool=height_defects_bool,
                    trace_points_nm=molecule_data.spline_coords,
                    distance_to_previous_points_nm=distances_nm(
                        coords_nm=molecule_data.spline_coords, circular=molecule_data.circular
                    ),
                    circular=molecule_data.circular,
                )

                grain_model.height_defect_data.molecule_defect_data_dict[molecule_id] = MoleculeDefectData(
                    ordered_defects_and_gaps=ordered_defect_gap_list
                )
    elif height_defect_method == "absolute":
        for _global_grain_id, grain_model in grain_collection.items():
            for molecule_id, molecule_data in grain_model.molecule_data_collection.items():
                heights_nm = molecule_data.spline_coords_heights
                spline_coords = molecule_data.spline_coords
                height_defects_bool = heights_nm > height_threshold_absolute_nm

                ordered_defect_gap_list = get_defects_and_gaps_from_bool_array(
                    defects_bool=height_defects_bool,
                    trace_points_nm=molecule_data.spline_coords,
                    distance_to_previous_points_nm=distances_nm(
                        coords_nm=molecule_data.spline_coords, circular=molecule_data.circular
                    ),
                    circular=molecule_data.circular,
                )

                grain_model.height_defect_data.molecule_defect_data_dict[molecule_id] = MoleculeDefectData(
                    ordered_defects_and_gaps=ordered_defect_gap_list
                )
    else:
        raise ValueError(f"Invalid height defect method: {height_defect_method}")

    return bad_grains


def find_defects_in_height_and_curvature(
    grain_collection: GrainCollection,
    height_defect_method: Literal["iqr", "absolute"],
    height_threshold_iqr_multiplier: float,
    height_threshold_absolute_nm: float,
    curvature_defect_method: Literal["iqr", "absolute"],
    curvature_threshold_iqr_multiplier: float,
    curvature_threshold_absolute_pernm: float,
) -> set[int]:
    """
    Find defects in height and curvature for all molecules in all grains in the grain collection.

    Parameters
    ----------
    grain_collection : GrainCollection
        The GrainCollection.
    height_defect_method : Literal["iqr", "absolute"]
        The method to use for finding height defects. Options are "iqr" for interquartile range method or "absolute"
        for an absolute threshold in nanometres.
    height_threshold_iqr_multiplier : float
        The multiplier for the interquartile range when using the "iqr" method for height defect detection.
    height_threshold_absolute_nm : float
        The absolute threshold in nanometres for height defect detection when using the "absolute" method
    curvature_defect_method : Literal["iqr", "absolute"]
        The method to use for finding curvature defects. Options are "iqr" for interquartile range method or "absolute"
        for an absolute threshold in inverse nanometres.
    curvature_threshold_iqr_multiplier : float
        The multiplier for the interquartile range when using the "iqr" method for curvature defect detection.
    curvature_threshold_absolute_pernm : float
        The absolute threshold in inverse nanometres for curvature defect detection when using the "absolute" method.

    Returns
    -------
    set[int]
        A set of global grain IDs for which height or curvature defect detection failed.
    """
    bad_grains = set()

    # find curvature defects
    additional_bad_grains = find_curvature_defects(
        grain_collection=grain_collection,
        curvature_defect_method=curvature_defect_method,
        curvature_threshold_iqr_multiplier=curvature_threshold_iqr_multiplier,
        curvature_threshold_absolute_pernm=curvature_threshold_absolute_pernm,
    )
    bad_grains.update(additional_bad_grains)

    additional_bad_grains = find_height_defects(
        grain_collection=grain_collection,
        height_defect_method=height_defect_method,
        height_threshold_iqr_multiplier=height_threshold_iqr_multiplier,
        height_threshold_absolute_nm=height_threshold_absolute_nm,
    )
    bad_grains.update(additional_bad_grains)

    grain_collection.remove_grains(bad_grains)
    print(f"removed grains with ids {bad_grains}, {len(grain_collection)} grains remain")

    return bad_grains
