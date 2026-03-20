"""Scripts for curvature postprocessing."""

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from topostats.damage.array_manipulation import cumulative_distances_nm, distances_nm
from topostats.damage.damage import GrainCollection
from topostats.measure.curvature import smooth_curvature


def smooth_grain_curvatures(
    grain_collection: GrainCollection,
    method: Literal["gaussian", "savitzky_golay"],
    gaussian_sigma_nm: float,
    savgol_window_length_nm: int,
    savgol_polyorder: int,
    plot=False,
) -> set[int]:
    """
    Smooth the curvature values for all molecules in all grains in the grain collection.

    Parameters
    ----------
    grain_collection : GrainCollection
        The GrainCollection.
    method : Literal["gaussian", "savitzky_golay"]
        The method to use for smoothing the curvature values. Options are "gaussian" for a Gaussian filter or
        "savitzky_golay" for a Savitzky-Golay filter.
    gaussian_sigma_nm : float
        The standard deviation of the Gaussian kernel in nanometres.
    savgol_window_length_nm : int
        The length of the filter window in nanometres.
    savgol_polyorder : int
        The order of the polynomial used to fit the samples in the savgol filter.
    plot : bool, optional
        Whether to do diagnostic plotting.

    Returns
    -------
    set[int]
        A set of global grain IDs for which curvature smoothing failed.
    """
    bad_grains = set()

    if plot:
        used_p2nm_values = set()

    for global_grain_id, grain in grain_collection.items():
        pixel_to_nm_scaling = grain.pixel_to_nm_scaling
        if plot:
            if np.round(pixel_to_nm_scaling, 1) in used_p2nm_values:
                continue
            used_p2nm_values.add(np.round(pixel_to_nm_scaling, 1))
        for molecule_id, molecule_data in grain.molecule_data_collection.items():
            if molecule_data.curvature_data is not None:
                curvature_data = molecule_data.curvature_data
                curvatures = curvature_data["curvatures"]
                circular = molecule_data.circular
                spline_coords_nm = molecule_data.spline_coords
                spline_distances = distances_nm(spline_coords_nm, circular=circular)
                mean_spline_distance = np.float64(np.mean(spline_distances))
                spline_cumulative_distances = cumulative_distances_nm(spline_coords_nm, circular=circular)
                assert len(curvatures) == len(spline_coords_nm)
                assert len(spline_cumulative_distances) == len(curvatures)
                try:
                    smoothed_curvatures = smooth_curvature(
                        curvatures=curvatures,
                        point_spacing_nm=mean_spline_distance,
                        method=method,
                        gaussian_sigma_nm=gaussian_sigma_nm,
                        savgol_window_length_nm=savgol_window_length_nm,
                        savgol_polyorder=savgol_polyorder,
                    )
                except AssertionError as e:
                    if "savgol_window_length" in str(e):
                        print(
                            f"smoothing failed for grain {global_grain_id} molecule {molecule_id} due to savgol window length being too long"
                        )
                        bad_grains.add(global_grain_id)
                        continue

                    raise e

                molecule_data.curvature_data["smoothed_curvatures"] = smoothed_curvatures

                if plot:
                    fig, ax = plt.subplots(1, 1, figsize=(20, 4))
                    plt.plot(spline_cumulative_distances, curvatures, label="original", alpha=0.5)
                    plt.plot(spline_cumulative_distances, smoothed_curvatures, label="gaussian smoothed")
                    plt.xlim(0, 380)
                    plt.legend(
                        loc="upper right",
                    )
                    plt.title(
                        f"Grain {global_grain_id} Molecule {molecule_id} Curvature Smoothing\np2nm:"
                        f"{pixel_to_nm_scaling:.2f} mean_spline_distance: {mean_spline_distance:.2f}"
                        f"\ngaussian_sigma_nm: {gaussian_sigma_nm} savgol_window_length_nm: {savgol_window_length_nm} savgol_polyorder: {savgol_polyorder}"
                    )
                    plt.show()
            else:
                bad_grains.add(global_grain_id)

    return bad_grains
