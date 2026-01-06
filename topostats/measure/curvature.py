"""Calculate various curvature metrics for traces."""

import logging

import pandas as pd
import numpy as np
import numpy.typing as npt

from topostats.logs.logs import LOGGER_NAME
from topostats.classes import TopoStatsBaseModel

LOGGER = logging.getLogger(LOGGER_NAME)


def angle_diff_signed(v1: npt.NDArray[np.number], v2: npt.NDArray[np.number]):
    """
    Calculate the signed angle difference between two point vecrtors in 2D space.

    Positive angles are clockwise, negative angles are counterclockwise.

    Parameters
    ----------
    v1 : npt.NDArray[np.number]
        First vector.
    v2 : npt.NDArray[np.number]
        Second vector.

    Returns
    -------
    float
        The signed angle difference in radians.
    """
    if v1.shape != (2,) or v2.shape != (2,):
        raise ValueError("Vectors must be of shape (2,)")

    angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
    if angle > np.pi:
        angle -= 2 * np.pi
    elif angle < -np.pi:
        angle += 2 * np.pi

    return angle


def discrete_angle_difference_per_nm_circular(
    trace_nm: npt.NDArray[np.number],
) -> npt.NDArray[np.number]:
    """
    Calculate the discrete angle difference per nm along a trace.

    Parameters
    ----------
    trace_nm : npt.NDArray[np.number]
        The coordinate trace, in nanometre units.

    Returns
    -------
    npt.NDArray[np.number]
        The discrete angle difference per nm.
    """
    angles_per_nm = np.zeros(trace_nm.shape[0])
    for index, point in enumerate(trace_nm):
        if index == 0:
            v1 = point - trace_nm[-1]
            v2 = trace_nm[index + 1] - point
        elif index == trace_nm.shape[0] - 1:
            v1 = point - trace_nm[index - 1]
            v2 = trace_nm[0] - point
        else:
            v1 = point - trace_nm[index - 1]
            v2 = trace_nm[index + 1] - point

        # Normalise vectors to unit length
        norm_v1 = v1 / np.linalg.norm(v1)
        norm_v2 = v2 / np.linalg.norm(v2)

        # Calculate the signed angle difference between the previous direction and the current direction
        angle = angle_diff_signed(norm_v1, norm_v2)

        # Calculate distance travelled between previous point and the current point
        distance = np.linalg.norm(v1)

        # Calculate the angle difference per nm
        angles_per_nm[index] = angle / distance

    return angles_per_nm


def discrete_angle_difference_per_nm_linear(
    trace_nm: npt.NDArray[np.number],
) -> npt.NDArray[np.number]:
    """
    Calculate the discrete angle difference per nm along a trace.

    Parameters
    ----------
    trace_nm : npt.NDArray[np.number]
        The coordinate trace, in nanometre units.

    Returns
    -------
    npt.NDArray[np.number]
        The discrete angle difference per nm.
    """
    angles_per_nm = np.zeros(trace_nm.shape[0])
    for index, point in enumerate(trace_nm):
        if index == 0:
            # No previous point so cannot calculate angle
            v2 = trace_nm[index + 1] - point
            angle = 0.0
            distance = np.linalg.norm(v2)
        elif index == trace_nm.shape[0] - 1:
            # No next point so cannot calculate angle
            v1 = point - trace_nm[index - 1]
            angle = 0.0
            distance = np.linalg.norm(v1)
        else:
            v1 = point - trace_nm[index - 1]
            v2 = trace_nm[index + 1] - point

            # Normalise vectors to unit length
            norm_v1 = v1 / np.linalg.norm(v1)
            norm_v2 = v2 / np.linalg.norm(v2)

            # Calculate the signed angle difference between the previous direction and the current direction
            angle = angle_diff_signed(norm_v1, norm_v2)

            # Calculate distance travelled between previous point and the current point
            distance = np.linalg.norm(v1)

        angles_per_nm[index] = angle / distance

    return angles_per_nm


class MoleculeCurvatureStats(TopoStatsBaseModel):
    """Data model for storing curvature statistics for a single molecule."""

    curvatures: npt.NDArray[np.number]
    is_circular: bool
    mean_curvature: float
    max_curvature: float
    min_curvature: float
    std_curvature: float
    total_curvature: float
    median_curvature: float
    curvature_iqr: float


class GrainCurvatureStats(TopoStatsBaseModel):
    """Data model for storing curvature statistics for a single grain."""

    molecules: dict[str, MoleculeCurvatureStats]
    mean_curvature: float
    max_curvature: float
    min_curvature: float
    std_curvature: float
    total_curvature: float
    median_curvature: float
    curvature_iqr: float


class AllGrainCurvatureStats(TopoStatsBaseModel):
    """Data model for storing curvature statistics for all grains."""

    grains: dict[str, GrainCurvatureStats]
    filename: str

    def create_grain_curvature_stats_dataframe(self) -> pd.DataFrame:
        """
        Create a dataframe of grain curvature statistics.

        Returns
        -------
        pd.DataFrame
            Dataframe of grain curvature statistics.
        """

        # Format: grain_index | <metric>
        records = []
        for grain_index, grain_curvature_stats in self.grains.items():
            entry = {
                "image": self.filename,
                "grain_number": int(grain_index.split("_")[-1]),
                "mean_curvature": grain_curvature_stats.mean_curvature,
                "max_curvature": grain_curvature_stats.max_curvature,
                "min_curvature": grain_curvature_stats.min_curvature,
                "std_curvature": grain_curvature_stats.std_curvature,
                "total_curvature": grain_curvature_stats.total_curvature,
                "median_curvature": grain_curvature_stats.median_curvature,
                "curvature_iqr": grain_curvature_stats.curvature_iqr,
            }
            records.append(entry)
        return pd.DataFrame.from_records(records)


def _calculate_curvature_metrics(curvatures: npt.NDArray[np.float64]) -> dict[str, float]:
    """
    Calculate curvature metrics from an array of curvatures.

    Parameters
    ----------
    curvatures : npt.NDArray[np.float64]
        Array of curvature values.

    Returns
    -------
    dict[str, float]
        Dictionary of curvature metrics.
    """
    return {
        "mean_curvature": float(np.mean(curvatures)),
        "max_curvature": float(np.max(curvatures)),
        "min_curvature": float(np.min(curvatures)),
        "std_curvature": float(np.std(curvatures)),
        "total_curvature": float(np.sum(curvatures)),
        "median_curvature": float(np.median(curvatures)),
        "curvature_iqr": float(np.percentile(curvatures, 75) - np.percentile(curvatures, 25)),
    }


def calculate_curvature_stats_image(
    filename: str,
    all_grain_smoothed_data: dict,
    pixel_to_nm_scaling: float,
) -> tuple[AllGrainCurvatureStats, pd.DataFrame]:
    """
    Perform curvature analysis for a whole image of grains.

    Parameters
    ----------
    filename : str
        Filename of the image.
    all_grain_smoothed_data : dict
        Dictionary containing grain traces in pixel units.
    pixel_to_nm_scaling : float
        Pixel to nm scaling factor.

    Returns
    -------
    tuple[AllGrainCurvatureStats, pd.DataFrame]
        All grain curvature statistics and dataframe of grain curvature statistics.
    """
    grains: dict[str, GrainCurvatureStats] = {}

    # Iterate over grains
    for grain_key, grain_data in all_grain_smoothed_data.items():
        # Iterate over molecules
        molecules: dict[str, MoleculeCurvatureStats] = {}
        for molecule_key, molecule_data in grain_data.items():
            trace_nm = molecule_data["spline_coords"] * pixel_to_nm_scaling
            is_circular = molecule_data["tracing_stats"]["end_to_end_distance"] == 0.0

            curvatures = (
                discrete_angle_difference_per_nm_circular(trace_nm)
                if is_circular
                else discrete_angle_difference_per_nm_linear(trace_nm)
            )

            curvatures_absolute = np.abs(curvatures)

            metrics = _calculate_curvature_metrics(curvatures_absolute)
            molecules[molecule_key] = MoleculeCurvatureStats(
                curvatures=curvatures_absolute,
                is_circular=is_circular,
                **metrics,
            )

        # Collate stats
        all_curvatures = np.concatenate([molecule.curvatures for molecule in molecules.values()])
        grain_metrics = _calculate_curvature_metrics(all_curvatures)
        grains[grain_key] = GrainCurvatureStats(
            molecules=molecules,
            **grain_metrics,
        )

        all_grain_curvature_stats = AllGrainCurvatureStats(filename=filename, grains=grains)
        all_grain_curvature_stats_df = all_grain_curvature_stats.create_grain_curvature_stats_dataframe()

    return all_grain_curvature_stats, all_grain_curvature_stats_df
