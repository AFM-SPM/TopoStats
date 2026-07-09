"""Classes for damage analysis."""

from collections.abc import Generator
from copy import deepcopy
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.ticker import FuncFormatter
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from topostats.array_manipulation import calculate_distance_of_region, distances_nm
from topostats.classes import MoleculeCurvatureStats
from topostats.plottingfuncs import Colormap

colormap = Colormap()
CMAP: plt.Colormap = colormap.get_cmap()
VMIN = -3
VMAX = 4
IMGPLOTARGS: dict = {"cmap": CMAP, "vmin": VMIN, "vmax": VMAX}


class BaseDamageAnalysis(BaseModel):
    """Base class for damage classes."""

    # Allow numpy arrays to be used as fields
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class Defect(BaseDamageAnalysis):
    """
    A class to represent a defect in a trace.

    Attributes
    ----------
    start_index : int
        The index of the first point of the defect.
    end_index : int
        The index of the last point of the defect.
    length_nm : float
        The length of the defect in nanometres.
    position_along_trace_nm : float
        The position of the defect along the trace in nanometres.
    total_turn_radians : tuple[float, float]
        The total turn of the defect in radians, as a tuple of (total_left_turn, total_right_turn).
    """

    start_index: int
    end_index: int
    length_nm: float
    position_along_trace_nm: float
    total_turn_radians: tuple[float, float]
    depth_nm: float
    volume_nm3: float


class Gap(BaseDamageAnalysis):
    """
    A class to represent a gap between defects in a trace.

    Attributes
    ----------
    start_index : int
        The index of the first point of the gap.
    end_index : int
        The index of the last point of the gap.
    length_nm : float
        The length of the gap in nanometres.
    position_along_trace_nm : float
        The position of the gap along the trace in nanometres.
    total_turn_radians : tuple[float, float]
        The total turn of the gap in radians, as a tuple of (total_left_turn, total_right_turn).
    """

    start_index: int
    end_index: int
    length_nm: float
    position_along_trace_nm: float
    total_turn_radians: tuple[float, float]


class OrderedDefectGapList(BaseDamageAnalysis):
    """A class to represent an ordered list of defects and gaps in a trace."""

    defect_gap_list: list[Defect | Gap] = Field(default_factory=list)

    def sort_defect_gap_list(self) -> None:
        """Sort the defect_gap_list by start_index."""
        self.defect_gap_list.sort(key=lambda x: x.start_index)

    def model_post_init(self, __context: Any) -> None:
        """Ensure that the defect_gap_list is ordered by start_index."""
        self.sort_defect_gap_list()

    @field_validator("defect_gap_list", mode="before")
    def _copy_defect_gap_list(cls, v: list[Defect | Gap]) -> list[Defect | Gap]:
        """Make a deep copy of the defect_gap_list to ensure the original isn't edited."""
        return deepcopy(v)

    def add_item(self, item: Defect | Gap) -> None:
        """Add a defect or gap to the defect_gap_list and sort it."""
        self.defect_gap_list.append(item)
        self.sort_defect_gap_list()

    def __eq__(self, other: object) -> bool:
        """Check if two OrderedDefectGapList instances are equal, with floating point tolerance."""
        if not isinstance(other, OrderedDefectGapList):
            return False

        if len(self.defect_gap_list) != len(other.defect_gap_list):
            return False

        # compare each item with floating point tolerance
        for self_item, other_item in zip(self.defect_gap_list, other.defect_gap_list):
            if not isinstance(self_item, type(other_item)):
                return False

            # Check if start and end indices are equal
            if self_item.start_index != other_item.start_index or self_item.end_index != other_item.end_index:
                return False

            # Lengths approximately equal
            if not np.isclose(self_item.length_nm, other_item.length_nm, atol=1e-6):
                return False

            # position along trace
            if not np.isclose(self_item.position_along_trace_nm, other_item.position_along_trace_nm, atol=1e-6):
                return False

            # total turn radians
            if not (
                np.isclose(self_item.total_turn_radians[0], other_item.total_turn_radians[0], atol=1e-6)
                and np.isclose(self_item.total_turn_radians[1], other_item.total_turn_radians[1], atol=1e-6)
            ):
                return False

        return True


class NewMoleculeCurvatureStats(BaseDamageAnalysis):
    """Data object to hold new curvature stats for a molecule."""

    curvatures: npt.NDArray[np.float64]
    is_circular: bool
    num_turns: int
    curvature_mean: float
    curvature_max: float
    curvature_min: float
    curvature_std: float
    curvature_var: float
    curvature_total: float
    curvature_median: float
    curvature_iqr: float
    curvature_90th: float
    turn_in_distance_window_length_nm: float | None = None
    turn_in_distance_window_end_sampling_points: int | None = None
    turn_in_distances_deg: npt.NDArray[np.float64] | None = None

    def from_molecule_curvature_stats(
        molecule_curvature_stats: MoleculeCurvatureStats,
    ) -> "NewMoleculeCurvatureStats":
        """Create a NewMoleculeCurvatureStats object from a MoleculeCurvatureStats object."""
        return NewMoleculeCurvatureStats(
            curvatures=molecule_curvature_stats.curvatures,
            is_circular=molecule_curvature_stats.is_circular,
            num_turns=molecule_curvature_stats.num_turns,
            curvature_mean=molecule_curvature_stats.curvature_mean,
            curvature_max=molecule_curvature_stats.curvature_max,
            curvature_min=molecule_curvature_stats.curvature_min,
            curvature_std=molecule_curvature_stats.curvature_std,
            curvature_var=molecule_curvature_stats.curvature_var,
            curvature_total=molecule_curvature_stats.curvature_total,
            curvature_median=molecule_curvature_stats.curvature_median,
            curvature_iqr=molecule_curvature_stats.curvature_iqr,
            curvature_90th=molecule_curvature_stats.curvature_90th,
        )


class UnanalysedMoleculeData(BaseDamageAnalysis):
    """Data object to hold unanalysed molecule data."""

    molecule_id: int
    ordered_coords_heights: npt.NDArray[np.float64]
    spline_coords_heights: npt.NDArray[np.float64]
    distances: npt.NDArray[np.float64]
    circular: bool
    spline_coords_px: npt.NDArray[np.float64]
    ordered_coords: npt.NDArray[np.float64]
    curvature_data: MoleculeCurvatureStats
    pixel_to_nm_scaling: float

    @computed_field
    @property
    def spline_coords_nm(self) -> npt.NDArray[np.float64]:
        """Get the spline coordinates in nanometres."""
        return self.spline_coords_px * self.pixel_to_nm_scaling


class MoleculeDefectData(BaseDamageAnalysis):
    """Data object to hold the defect and gap data for a molecule."""

    ordered_defects_and_gaps: OrderedDefectGapList

    @computed_field
    @property
    def num_defects(self) -> int:
        """Calculate the number of defects."""
        return sum(isinstance(item, Defect) for item in self.ordered_defects_and_gaps.defect_gap_list)

    @computed_field
    @property
    def num_gaps(self) -> int:
        """Calculate the number of gaps."""
        return sum(isinstance(item, Gap) for item in self.ordered_defects_and_gaps.defect_gap_list)

    @computed_field
    @property
    def defect_lengths_nm(self) -> list[float]:
        """Get a list of the lengths of the defects in nanometres."""
        return [
            defect_or_gap.length_nm
            for defect_or_gap in self.ordered_defects_and_gaps.defect_gap_list
            if isinstance(defect_or_gap, Defect)
        ]


class MoleculeData(UnanalysedMoleculeData):
    """Data object to hold the analysed molecule data."""

    curvature_data: NewMoleculeCurvatureStats
    curvature_defect_data: MoleculeDefectData | None = None
    height_defect_data: MoleculeDefectData | None = None
    beak_defect_data: MoleculeDefectData | None = None
    coinciding_defect_threshold_nm: float
    smoothed_spline_coords_heights: npt.NDArray[np.float64] | None = None

    def from_unanalysed_molecule_data(
        unanalysed_data: UnanalysedMoleculeData,
        coinciding_defect_threshold_nm: float,
    ) -> "MoleculeData":
        """Create a MoleculeData object from an UnanalysedMoleculeData object."""
        new_molecule_curvature_stats = NewMoleculeCurvatureStats.from_molecule_curvature_stats(
            molecule_curvature_stats=unanalysed_data.curvature_data,
        )

        return MoleculeData(
            molecule_id=unanalysed_data.molecule_id,
            ordered_coords_heights=unanalysed_data.ordered_coords_heights,
            spline_coords_heights=unanalysed_data.spline_coords_heights,
            distances=unanalysed_data.distances,
            circular=unanalysed_data.circular,
            spline_coords_px=unanalysed_data.spline_coords_px,
            ordered_coords=unanalysed_data.ordered_coords,
            curvature_data=new_molecule_curvature_stats,
            pixel_to_nm_scaling=unanalysed_data.pixel_to_nm_scaling,
            coinciding_defect_threshold_nm=coinciding_defect_threshold_nm,
            smoothed_spline_coords_heights=None,
        )

    @computed_field
    @property
    def distances_nm(self) -> npt.NDArray[np.float64]:
        """Get the distances between points in nanometres."""
        distances_nm = []
        for index in range(len(self.spline_coords_nm)):
            if index == 0:
                if not self.circular:
                    distances_nm.append(0.0)
                    continue
            point = self.spline_coords_nm[index]
            previous_point = self.spline_coords_nm[index - 1]
            distance = np.linalg.norm(point - previous_point)
            distances_nm.append(distance)
        return np.array(distances_nm)

    @computed_field
    @property
    def point_spacing_nm(self) -> np.float64:
        """Calculate the average spacing between points in nanometres."""
        # drop the first point since this one is often errant and there's nothing that can be done about it,
        # we are just checking if the rest are regular.
        distances_nm = self.distances_nm[1:]
        if np.max(np.diff(distances_nm, axis=0)) > 1e-3:
            raise ValueError(
                f"molecule with id {self.molecule_id} has inconsistent point spacing, with distances {distances_nm}"
            )
        return np.mean(distances_nm)

    @computed_field
    @property
    def distance_to_previous_points_nm(self) -> npt.NDArray[np.float64]:
        """Return an array of the distances from each point to the preceding point."""
        return distances_nm(self.spline_coords_px, self.circular)

    @computed_field
    @property
    def coinciding_defects(  # noqa: C901
        self,
    ) -> list[tuple[Defect, Defect]]:
        """Get a list of coinciding curvature and height defects."""
        coinciding_defects = []
        curvature_defect_data = self.curvature_defect_data
        height_defect_data = self.height_defect_data
        if curvature_defect_data is None or height_defect_data is None:
            raise ValueError(
                f"molecule with id {self.molecule_id} has missing defect data, cannot calculate coinciding defects"
            )
        for curvature_defect_or_gap in curvature_defect_data.ordered_defects_and_gaps.defect_gap_list:
            if isinstance(curvature_defect_or_gap, Defect):
                curvature_defect = curvature_defect_or_gap
                for height_defect_or_gap in height_defect_data.ordered_defects_and_gaps.defect_gap_list:
                    if isinstance(height_defect_or_gap, Defect):
                        height_defect = height_defect_or_gap
                        # Check if they coincide, by checking if the distance between either start or end points are within
                        # the threshold.

                        # Check if wrapping around the end of the array is needed, and if the trace is not circular,
                        # then disallow this.
                        if curvature_defect.end_index > height_defect.start_index:
                            # wrapping needed
                            if not self.circular:
                                continue
                        curvature_end_to_height_start_distance_nm = calculate_distance_of_region(
                            start_index=curvature_defect.end_index,
                            end_index=height_defect.start_index,
                            distance_to_previous_points_nm=self.distance_to_previous_points_nm,
                            circular=self.circular,
                        )

                        # Check the other way around as well
                        if height_defect.end_index > curvature_defect.start_index:
                            # wrapping needed
                            if not self.circular:
                                continue
                        height_end_to_curvature_start_distance_nm = calculate_distance_of_region(
                            start_index=height_defect.end_index,
                            end_index=curvature_defect.start_index,
                            distance_to_previous_points_nm=self.distance_to_previous_points_nm,
                            circular=self.circular,
                        )

                        if curvature_end_to_height_start_distance_nm <= self.coinciding_defect_threshold_nm:
                            # We have found a coinciding defect
                            coinciding_defects.append((curvature_defect, height_defect))
                        elif height_end_to_curvature_start_distance_nm <= self.coinciding_defect_threshold_nm:
                            # We have found a coinciding defect
                            coinciding_defects.append((curvature_defect, height_defect))
        return coinciding_defects


class UnanalysedMoleculeDataCollection(BaseDamageAnalysis):
    """Data object to hold a collection of unanalysed molecule data."""

    molecules: dict[int, UnanalysedMoleculeData]

    def __getitem__(self, key: int) -> UnanalysedMoleculeData:
        """Get the molecule data for a given molecule id."""
        return self.molecules[key]

    def __len__(self) -> int:
        """Get the number of molecules in the collection."""
        return len(self.molecules)

    def __contains__(self, key: int) -> bool:
        """Check if a molecule id is in the collection."""
        return key in self.molecules

    def items(self) -> Generator[tuple[int, UnanalysedMoleculeData], None, None]:
        """Get the items of the molecule data collection, yielding tuples of molecule id and molecule data."""
        return (item for item in self.molecules.items())

    def keys(self) -> Generator[int, None, None]:
        """Get the keys of the molecule data collection."""
        return (key for key in self.molecules.keys())

    def values(self) -> Generator[UnanalysedMoleculeData, None, None]:
        """Get the values of the molecule data collection."""
        return (value for value in self.molecules.values())

    def get(self, key: int, default: UnanalysedMoleculeData | None = None) -> UnanalysedMoleculeData | None:
        """Get the molecule data for a given molecule id, or return a default value if the molecule id is not in the collection."""
        return self.molecules.get(key, default)

    def add_molecule(self, molecule: UnanalysedMoleculeData) -> None:
        """Add a molecule to the collection."""
        self.molecules[molecule.molecule_id] = molecule

    def remove_molecule(self, molecule_id: int) -> None:
        """Remove a molecule from the collection by its molecule id."""
        if molecule_id not in self.molecules:
            raise KeyError(f"molecule with id {molecule_id} not found in collection, cannot remove")
        del self.molecules[molecule_id]


class UnanalysedGrain(BaseDamageAnalysis):
    """Data object to hold unanalysed grain data."""

    global_grain_id: int | None = None
    file_grain_id: int
    filename: str
    pixel_to_nm_scaling: float
    folder: str
    sample_type: str
    percent_damage: float
    bbox: tuple[int, int, int, int]
    image: npt.NDArray[np.float64]
    aspect_ratio: float
    smallest_bounding_area: float
    total_contour_length: float
    num_crossings: int
    molecule_data_collection: UnanalysedMoleculeDataCollection
    added_left: int
    added_top: int
    padding: int
    mask: npt.NDArray[np.bool_]

    def __str__(self) -> str:
        """Return a simplified string representation of the grain."""
        return (
            f"GrainModel(global_grain_id={self.global_grain_id}), {self.percent_damage}% "
            f"damage, from file {self.filename}."
        )

    def plot(self, mask_alpha: float = 0.3) -> None:
        """Plot the grain image with the mask overlaid."""
        plt.imshow(self.image, **IMGPLOTARGS)
        plt.imshow(self.mask[:, :], alpha=mask_alpha, cmap="gray")
        plt.title(f"grain {self.global_grain_id}, {self.percent_damage}% damage")

        for molecule_data in self.molecule_data_collection.values():
            spline_coords = molecule_data.spline_coords_px
            plt.plot(spline_coords[:, 1], spline_coords[:, 0])

        plt.show()


class UnanalysedGrainCollection(BaseDamageAnalysis):
    """Data object to hold a collection of unanalysed grains."""

    unanalysed_grains: dict[int, UnanalysedGrain] = Field(default_factory=dict)
    current_global_grain_id: int = 0

    @computed_field
    @property
    def folder_counts(self) -> dict[str, int]:
        """Get a count of the number of grains from each folder."""
        counts: dict[str, int] = {}
        for grain in self.unanalysed_grains.values():
            folder = grain.folder
            counts[folder] = counts.get(folder, 0) + 1
        return counts

    # pretty print
    def __str__(self) -> str:
        """Return a simplified string representation of the grain collection."""
        grain_indexes = range(self.current_global_grain_id)
        missing_grain_indexes = [index for index in grain_indexes if index not in self.unanalysed_grains]
        folder_counts_str = "".join([f"- {folder}: {count}\n" for folder, count in self.folder_counts.items()])
        return (
            f"GrainModelCollection with {len(self.unanalysed_grains)} grains, with {len(missing_grain_indexes)} "
            f"omitted grains: {missing_grain_indexes}.\nFolder counts:\n{folder_counts_str}"
        )

    def add_grain(self, unanalysed_grain: UnanalysedGrain) -> None:
        """Add a grain to the collection, assigning it a global grain id."""
        # note: a grain might already have a global grain id if it came from another collection, but we can
        # just overwrite it.
        unanalysed_grain.global_grain_id = self.current_global_grain_id
        self.unanalysed_grains[self.current_global_grain_id] = unanalysed_grain
        self.current_global_grain_id += 1

    def remove_grain(self, global_grain_id: int) -> None:
        """Remove a grain from the collection by its global id."""
        if global_grain_id not in self.unanalysed_grains:
            raise KeyError(f"grain with global id {global_grain_id} not found in collection, cannot remove")
        del self.unanalysed_grains[global_grain_id]

    def remove_grains(self, global_grain_ids: list[int] | set[int]) -> None:
        """Remove multiple grains from the collection by their global ids."""
        global_grain_ids_set = set(global_grain_ids)
        for grain_id in global_grain_ids_set:
            self.remove_grain(grain_id)


def combine_unanalysed_grain_collections(collections: list[UnanalysedGrainCollection]) -> UnanalysedGrainCollection:
    """Combine multiple UnanalysedGrainCollections into a single UnanalysedGrainCollection."""
    combined_collection = UnanalysedGrainCollection(unanalysed_grains={})
    for collection in collections:
        for grain in collection.unanalysed_grains.values():
            combined_collection.add_grain(grain)
    return combined_collection


class MoleculeDataCollection(BaseDamageAnalysis):
    """Data object to hold a collection of analysed molecule data."""

    molecules: dict[int, MoleculeData]

    def from_unanalysed_molecule_data_collection(
        unanalysed_collection: UnanalysedMoleculeDataCollection,
        coinciding_defect_threshold_nm: float,
    ) -> "MoleculeDataCollection":
        """Create a MoleculeDataCollection object from an UnanalysedMoleculeDataCollection object."""
        molecule_data_dict = {}
        for molecule_id, unanalysed_molecule_data in unanalysed_collection.molecules.items():
            molecule_data = MoleculeData.from_unanalysed_molecule_data(
                unanalysed_molecule_data,
                coinciding_defect_threshold_nm=coinciding_defect_threshold_nm,
            )
            molecule_data_dict[molecule_id] = molecule_data
        return MoleculeDataCollection(molecules=molecule_data_dict)

    def __getitem__(self, key: int) -> MoleculeData:
        """Get the molecule data for a given molecule id."""
        return self.molecules[key]

    def __len__(self) -> int:
        """Get the number of molecules in the collection."""
        return len(self.molecules)

    def __contains__(self, key: int) -> bool:
        """Check if a molecule id is in the collection."""
        return key in self.molecules

    def items(self) -> Generator[tuple[int, MoleculeData], None, None]:
        """Get the items of the molecule data collection, yielding tuples of molecule id and molecule data."""
        return (item for item in self.molecules.items())

    def keys(self) -> Generator[int, None, None]:
        """Get the keys of the molecule data collection."""
        return (key for key in self.molecules.keys())

    def values(self) -> Generator[MoleculeData, None, None]:
        """Get the values of the molecule data collection."""
        return (value for value in self.molecules.values())

    def get(self, key: int, default: MoleculeData | None = None) -> MoleculeData | None:
        """Get the molecule data for a given molecule id, or return a default value if the molecule id is not in the collection."""
        return self.molecules.get(key, default)

    def add_molecule(self, molecule: MoleculeData) -> None:
        """Add a molecule to the collection."""
        self.molecules[molecule.molecule_id] = molecule

    def remove_molecule(self, molecule_id: int) -> None:
        """Remove a molecule from the collection by its molecule id."""
        if molecule_id not in self.molecules:
            raise KeyError(f"molecule with id {molecule_id} not found in collection, cannot remove")
        del self.molecules[molecule_id]


class GrainModel(UnanalysedGrain):
    """Data object to hold the analysed grain data."""

    molecule_data_collection: MoleculeDataCollection

    def from_unanalysed_grain(
        unanalysed_grain: UnanalysedGrain,
        coinciding_defect_threshold_nm: float,
    ) -> "GrainModel":
        """Create a GrainModel object from an UnanalysedGrain object."""
        # Create the new molecule data collection
        molecule_data_collection = MoleculeDataCollection.from_unanalysed_molecule_data_collection(
            unanalysed_grain.molecule_data_collection,
            coinciding_defect_threshold_nm=coinciding_defect_threshold_nm,
        )
        return GrainModel(
            global_grain_id=unanalysed_grain.global_grain_id,
            file_grain_id=unanalysed_grain.file_grain_id,
            filename=unanalysed_grain.filename,
            pixel_to_nm_scaling=unanalysed_grain.pixel_to_nm_scaling,
            folder=unanalysed_grain.folder,
            sample_type=unanalysed_grain.sample_type,
            percent_damage=unanalysed_grain.percent_damage,
            bbox=unanalysed_grain.bbox,
            image=unanalysed_grain.image,
            aspect_ratio=unanalysed_grain.aspect_ratio,
            smallest_bounding_area=unanalysed_grain.smallest_bounding_area,
            total_contour_length=unanalysed_grain.total_contour_length,
            num_crossings=unanalysed_grain.num_crossings,
            molecule_data_collection=molecule_data_collection,
            added_left=unanalysed_grain.added_left,
            added_top=unanalysed_grain.added_top,
            padding=unanalysed_grain.padding,
            mask=unanalysed_grain.mask,
        )

    def __str__(self) -> str:
        """Return a simplified string representation of the grain."""
        return (
            f"GrainModel(global_grain_id={self.global_grain_id}), {self.percent_damage}% damage, "
            f"with {len(self.molecule_data_collection)} molecules, {self.num_crossings} crossings "
            f"from file {self.filename}."
        )

    @computed_field
    @property
    def num_height_defects(self) -> int:
        """Calculate the total number of height defects across all molecules in the grain."""
        num_height_defects = 0
        for molecule_data in self.molecule_data_collection.values():
            if molecule_data.height_defect_data is None:
                raise ValueError(f"molecule with id {molecule_data.molecule_id} has no height defect data")
            num_height_defects += molecule_data.height_defect_data.num_defects
        return num_height_defects

    @computed_field
    @property
    def num_curvature_defects(self) -> int:
        """Calculate the total number of curvature defects across all molecules in the grain."""
        num_curvature_defects = 0
        for molecule_data in self.molecule_data_collection.values():
            if molecule_data.curvature_defect_data is None:
                raise ValueError(f"molecule with id {molecule_data.molecule_id} has no curvature defect data")
            num_curvature_defects += molecule_data.curvature_defect_data.num_defects
        return num_curvature_defects

    @computed_field
    @property
    def num_coinciding_defects(self) -> int:
        """Calculate the total number of coinciding defects across all molecules in the grain."""
        num_coinciding_defects = 0
        for molecule_data in self.molecule_data_collection.values():
            num_coinciding_defects += len(molecule_data.coinciding_defects)
        return num_coinciding_defects

    def plot(  # noqa: C901
        self,
        mask_alpha: float = 0.3,
        linemode: str = "",
        linewidth: float = 1.0,
        curvature_defects: bool = False,
        height_defects: bool = False,
        coinciding_defects: bool = False,
        beak_defects: bool = False,
        title_mode: str = "basic",
        curvature_absolute: bool = False,
        curvature_norm_bounds: tuple[float, float] = (-0.1, 0.1),
        turn_in_distance_absolute: bool = False,
        turn_in_distance_deg_norm_bounds: tuple[float, float] = (-180, 180),
        turn_in_distance_display_value_interval: int = -1,
        figsize: tuple[float, float] = (5, 5),
        save_path: str | None = None,
        show: bool = True,
    ) -> None:
        """Plot the grain image with the mask and molecule data overlaid."""

        def nm_formatter(px, pos):
            return f"{px * self.pixel_to_nm_scaling:.0f}"

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(self.image, **IMGPLOTARGS)
        ax.imshow(self.mask[:, :], alpha=mask_alpha, cmap="gray")
        if linemode == "spline":
            for _molecule_id, molecule_data in self.molecule_data_collection.items():
                spline_coords = molecule_data.spline_coords_px
                ax.plot(spline_coords[:, 1], spline_coords[:, 0])
        elif linemode == "curvature":
            for _molecule_id, molecule_data in self.molecule_data_collection.items():
                spline_coords = molecule_data.spline_coords_px
                curvature_data = molecule_data.curvature_data
                if curvature_data is not None:
                    curvature_values = curvature_data.curvatures
                    # plot the curvature values as a colormap along the spline coords
                    assert len(curvature_values) == len(spline_coords), (
                        f"length of curvature values {len(curvature_values)} does not match"
                        f"length of spline coords {len(spline_coords)}"
                    )
                    if curvature_absolute:
                        curvature_values = np.abs(curvature_values)
                        curvature_norm_lower_bound = 0
                        curvature_cmap = plt.get_cmap("Blues")
                    else:
                        curvature_norm_lower_bound = curvature_norm_bounds[0]
                        curvature_cmap = plt.get_cmap("coolwarm")
                    curvature_norm_upper_bound = curvature_norm_bounds[1]
                    curvature_values_clipped = np.clip(
                        curvature_values, curvature_norm_lower_bound, curvature_norm_upper_bound
                    )
                    curvature_values_normalised = (curvature_values_clipped - curvature_norm_lower_bound) / (
                        curvature_norm_upper_bound - curvature_norm_lower_bound
                    )

                    for index, point in enumerate(spline_coords):
                        color = curvature_cmap(curvature_values_normalised[index])
                        if index > 0:
                            previous_point = spline_coords[index - 1]
                            ax.plot(
                                [previous_point[1], point[1]],
                                [previous_point[0], point[0]],
                                color=color,
                                linewidth=linewidth,
                            )
        elif linemode == "turn_in_distance":
            for _molecule_id, molecule_data in self.molecule_data_collection.items():
                spline_coords = molecule_data.spline_coords_px
                curvature_data = molecule_data.curvature_data
                if curvature_data is not None:
                    if curvature_data.turn_in_distances_deg is not None:
                        turn_in_distances_deg = np.copy(curvature_data.turn_in_distances_deg)
                        turn_in_distance_window_length_nm = curvature_data.turn_in_distance_window_length_nm
                        # display the window length in the bottom left of the plot
                        if turn_in_distance_window_length_nm is not None:
                            turn_in_distance_window_length_px = (
                                turn_in_distance_window_length_nm / self.pixel_to_nm_scaling
                            )
                            window_visual_thickness_px = self.image.shape[1] * 0.01
                            border_offset_px = self.image.shape[1] * 0.01
                            window_visual_top_left_x = border_offset_px
                            window_visual_top_left_y = (
                                self.image.shape[0] - border_offset_px - window_visual_thickness_px
                            )
                            ax.add_patch(
                                plt.Rectangle(
                                    (window_visual_top_left_x, window_visual_top_left_y),
                                    turn_in_distance_window_length_px,
                                    window_visual_thickness_px,
                                    edgecolor="white",
                                    facecolor="white",
                                )
                            )
                            ax.text(
                                window_visual_top_left_x,
                                window_visual_top_left_y - 5,
                                f"turn in distance window length: {turn_in_distance_window_length_nm:.0f} nm",
                                fontsize=10,
                                color="white",
                            )

                        assert len(turn_in_distances_deg) == len(spline_coords), (
                            f"length of turn in distances {len(turn_in_distances_deg)} does not match"
                            f"length of spline coords {len(spline_coords)}"
                        )
                        if turn_in_distance_absolute:
                            turn_in_distance_deg_norm_bounds = (0, turn_in_distance_deg_norm_bounds[1])
                            turn_in_distances_deg = np.abs(turn_in_distances_deg)
                            turn_in_distance_cmap = plt.get_cmap("viridis")
                        else:
                            turn_in_distance_cmap = plt.get_cmap("coolwarm")
                        # normalise the values
                        turn_in_distances_deg_clipped = np.clip(
                            turn_in_distances_deg,
                            turn_in_distance_deg_norm_bounds[0],
                            turn_in_distance_deg_norm_bounds[1],
                        )
                        turn_in_distances_deg_normalised = (
                            turn_in_distances_deg_clipped - turn_in_distance_deg_norm_bounds[0]
                        ) / (turn_in_distance_deg_norm_bounds[1] - turn_in_distance_deg_norm_bounds[0])
                        nan_colour = (0.5, 0.5, 0.5, 1)
                        for index, point in enumerate(spline_coords):
                            turn_in_distance_deg = turn_in_distances_deg[index]
                            if np.isnan(turn_in_distance_deg):
                                colour = nan_colour
                            else:
                                colour = turn_in_distance_cmap(turn_in_distances_deg_normalised[index])
                            if index > 0:
                                previous_point = spline_coords[index - 1]
                                ax.plot(
                                    [previous_point[1], point[1]],
                                    [previous_point[0], point[0]],
                                    color=colour,
                                    linewidth=linewidth,
                                )
                                if turn_in_distance_display_value_interval > 0:
                                    if index % turn_in_distance_display_value_interval == 0:
                                        ax.text(
                                            point[1],
                                            point[0],
                                            f"{turn_in_distance_deg:.1f}°",
                                            fontsize=6,
                                            color="w",
                                        )
        if curvature_defects:
            # plot all the curvature defects as pink dots
            for molecule_id, molecule_data in self.molecule_data_collection.items():
                molecule_curvature_defect_data = molecule_data.curvature_defect_data
                if molecule_curvature_defect_data is None:
                    raise ValueError(f"molecule with id {molecule_data.molecule_id} has no curvature defect data")
                for item in molecule_curvature_defect_data.ordered_defects_and_gaps.defect_gap_list:
                    if isinstance(item, Defect):
                        defect_start_index = item.start_index
                        defect_end_index = item.end_index
                        spline_coords = self.molecule_data_collection[molecule_id].spline_coords_px
                        defect_coords = spline_coords[defect_start_index:defect_end_index]
                        ax.scatter(defect_coords[:, 1], defect_coords[:, 0], color="magenta", s=10)
        if height_defects:
            # plot all the height defects as cyan dots
            for molecule_id, molecule_data in self.molecule_data_collection.items():
                molecule_height_defect_data = molecule_data.height_defect_data
                if molecule_height_defect_data is None:
                    raise ValueError(f"molecule with id {molecule_data.molecule_id} has no height defect data")
                for item in molecule_height_defect_data.ordered_defects_and_gaps.defect_gap_list:
                    if isinstance(item, Defect):
                        defect_start_index = item.start_index
                        defect_end_index = item.end_index
                        spline_coords = self.molecule_data_collection[molecule_id].spline_coords_px
                        defect_coords = spline_coords[defect_start_index:defect_end_index]
                        ax.scatter(defect_coords[:, 1], defect_coords[:, 0], color="cyan", s=10)
        if coinciding_defects:
            # plot all correlated defects as yellow dots
            for molecule_id, molecule_data in self.molecule_data_collection.items():
                for curvature_defect, height_defect in molecule_data.coinciding_defects:
                    curvature_defect_start_index = curvature_defect.start_index
                    curvature_defect_end_index = curvature_defect.end_index
                    height_defect_start_index = height_defect.start_index
                    height_defect_end_index = height_defect.end_index
                    spline_coords = self.molecule_data_collection[molecule_id].spline_coords_px
                    if curvature_defect_start_index > curvature_defect_end_index:
                        curvature_defect_indexes = list(range(0, curvature_defect_end_index + 1)).extend(
                            range(curvature_defect_start_index, len(spline_coords) + 1)
                        )
                    else:
                        curvature_defect_indexes = range(curvature_defect_start_index, curvature_defect_end_index + 1)
                    curvature_defect_coords = spline_coords[curvature_defect_indexes]
                    if height_defect_start_index > height_defect_end_index:
                        height_defect_indexes = list(range(0, height_defect_end_index + 1)).extend(
                            range(height_defect_start_index, len(spline_coords) + 1)
                        )
                    else:
                        height_defect_indexes = range(height_defect_start_index, height_defect_end_index + 1)
                    height_defect_coords = spline_coords[height_defect_indexes]
                    # calculate the mean of the coords of the two defects to get a single point to plot
                    mean_curvature_defect_coords = np.mean(curvature_defect_coords, axis=0)
                    mean_height_defect_coords = np.mean(height_defect_coords, axis=0)
                    mean_defect_coords = (mean_curvature_defect_coords + mean_height_defect_coords) / 2
                    ax.scatter(
                        mean_defect_coords[1],
                        mean_defect_coords[0],
                        color="yellow",
                        s=300,
                        facecolors="none",
                        edgecolors="yellow",
                        linewidths=1,
                    )
        if beak_defects:
            # plot all the beak defects as red dots
            for molecule_id, molecule_data in self.molecule_data_collection.items():
                molecule_beak_defect_data = molecule_data.beak_defect_data
                if molecule_beak_defect_data is None:
                    raise ValueError(f"molecule with id {molecule_data.molecule_id} has no beak defect data")
                for item in molecule_beak_defect_data.ordered_defects_and_gaps.defect_gap_list:
                    if isinstance(item, Defect):
                        defect_start_index = item.start_index
                        defect_end_index = item.end_index
                        spline_coords = self.molecule_data_collection[molecule_id].spline_coords_px
                        defect_coords = spline_coords[defect_start_index : defect_end_index + 1]
                        ax.scatter(defect_coords[:, 1], defect_coords[:, 0], color="red", s=10)

        # set x ticks to be in nm
        # x_ticks = ax.get_xticks()
        # x_ticks_nm = x_ticks * self.pixel_to_nm_scaling
        # ax.set_xticklabels([f"{x_tick_nm:.0f}" for x_tick_nm in x_ticks_nm])
        ax.xaxis.set_major_formatter(FuncFormatter(nm_formatter))
        ax.set_xlabel("nm")
        # set y ticks to be in nm
        # y_ticks = ax.get_yticks()
        # y_ticks_nm = y_ticks * self.pixel_to_nm_scaling
        # ax.set_yticklabels([f"{y_tick_nm:.0f}" for y_tick_nm in y_ticks_nm])
        ax.yaxis.set_major_formatter(FuncFormatter(nm_formatter))
        ax.set_ylabel("nm")
        if title_mode == "basic":
            num_curvature_defects = self.num_curvature_defects
            num_height_defects = self.num_height_defects
            ax.set_title(
                f"grain {self.global_grain_id} | {self.sample_type} {self.percent_damage}% dam | "
                f"defects: {num_curvature_defects} C, {num_height_defects} H"
                f"\n{self.filename}"
            )
        if save_path is not None:
            plt.savefig(save_path)
        if show:
            plt.show()
        else:
            plt.close()


class GrainCollection(BaseDamageAnalysis):
    """Data object to hold a collection of analysed grains."""

    grains: dict[int, GrainModel]
    current_global_grain_id: int = 0

    @computed_field
    @property
    def folder_counts(self) -> dict[str, int]:
        """Get a count of the number of grains from each folder."""
        counts: dict[str, int] = {}
        for grain in self.grains.values():
            folder = grain.folder
            counts[folder] = counts.get(folder, 0) + 1
        return counts

    def __str__(self) -> str:
        """Return a simplified string representation of the grain collection."""
        grain_indexes = range(self.current_global_grain_id)
        missing_grain_indexes = [index for index in grain_indexes if index not in self.grains]
        folder_counts_str = "".join([f"- {folder}: {count}\n" for folder, count in self.folder_counts.items()])
        return (
            f"GrainModelCollection with {len(self.grains)} grains, with {len(missing_grain_indexes)} "
            f"omitted grains: {missing_grain_indexes}.\nFolder counts:\n{folder_counts_str}"
        )

    def __getitem__(self, key: int) -> GrainModel:
        """Get a grain from the collection by its global id."""
        return self.grains[key]

    def __len__(self) -> int:
        """Get the number of grains in the collection."""
        return len(self.grains)

    def __contains__(self, key: int) -> bool:
        """Check if a grain with a given global id is in the collection."""
        return key in self.grains

    def items(self) -> Generator[tuple[int, GrainModel], None, None]:
        """Get the items of the grain collection, yielding tuples of global grain id and grain."""
        return (item for item in self.grains.items())

    def keys(self) -> Generator[int, None, None]:
        """Get the keys of the grain collection."""
        return (key for key in self.grains.keys())

    def values(self) -> Generator[GrainModel, None, None]:
        """Get the values of the grain collection."""
        return (value for value in self.grains.values())

    def get(self, key: int, default: GrainModel | None = None) -> GrainModel | None:
        """Get a grain from the collection by its global id, returning a default value if not present."""
        return self.grains.get(key, default)

    def add_grain(self, grain_model: GrainModel) -> None:
        """Add a grain to the collection, assigning it a global grain id."""
        # note: a grain might already have a global grain id if it came from another collection, but we can
        # just overwrite it.
        grain_model.global_grain_id = self.current_global_grain_id
        self.grains[self.current_global_grain_id] = grain_model
        self.current_global_grain_id += 1

    def remove_grain(self, global_grain_id: int) -> None:
        """Remove a grain from the collection by its global id."""
        if global_grain_id not in self.grains:
            raise KeyError(f"grain with global id {global_grain_id} not found in collection, cannot remove")
        del self.grains[global_grain_id]

    def remove_grains(self, global_grain_ids: list[int] | set[int]) -> None:
        """Remove multiple grains from the collection by their global ids."""
        global_grain_ids_set = set(global_grain_ids)
        for grain_id in global_grain_ids_set:
            self.remove_grain(grain_id)

    def combine_with_other_collection(self, other_collection: "GrainCollection") -> None:
        """Combine another GrainCollection into this GrainCollection."""
        for grain_model in other_collection.values():
            self.add_grain(grain_model)

    def from_unanalysed_grain_collection(
        unanalysed_collection: UnanalysedGrainCollection,
        coinciding_defect_threshold_nm: float,
    ) -> "GrainCollection":
        """Create a GrainCollection object from an UnanalysedGrainCollection object."""
        grain_dict = {}
        for global_grain_id, unanalysed_grain in unanalysed_collection.unanalysed_grains.items():
            try:
                grain_model = GrainModel.from_unanalysed_grain(
                    unanalysed_grain,
                    coinciding_defect_threshold_nm=coinciding_defect_threshold_nm,
                )
            except ValueError as e:
                if "window length exceeds total length of the trace, cannot construct window" in str(e):
                    # if we cannot construct the window to calculate the turn in distance, then we cannot calculate
                    # the curvature defects, so we will skip this grain.
                    continue
                # else raise the error as it is unexpected.
                raise e
            grain_dict[global_grain_id] = grain_model
        return GrainCollection(grains=grain_dict, current_global_grain_id=unanalysed_collection.current_global_grain_id)

    def sample(self, n: int, seed: int = 0) -> "GrainCollection":
        """Return a sample of n grains from each of the sample type combinations in the collection."""
        rng = np.random.default_rng(seed)
        sample_dict: dict[int, GrainModel] = {}
        # group grains by sample type combination - sample type and damage
        sample_type_groups: dict[tuple[str, float], list[GrainModel]] = {}
        for grain in self.grains.values():
            sample_type_combination = (grain.sample_type, grain.percent_damage)
            if sample_type_combination not in sample_type_groups:
                sample_type_groups[sample_type_combination] = []
            sample_type_groups[sample_type_combination].append(grain)
        # sample n grains from each sample type combination group
        for _, grains in sample_type_groups.items():
            if len(grains) <= n:
                sampled_grains = grains
            else:
                sampled_grain_indexes = rng.choice(len(grains), size=n, replace=False)
                sampled_grains: list[GrainModel] = [grains[i] for i in sampled_grain_indexes]
            for grain in sampled_grains:
                global_grain_id = grain.global_grain_id
                assert global_grain_id is not None
                sample_dict[global_grain_id] = grain
        return GrainCollection(grains=sample_dict)

    def get_random_grain(self, seed: int = 0) -> GrainModel:
        """Return a random grain from the collection."""
        rng = np.random.default_rng(seed)
        grain_ids = list(self.grains.keys())
        random_grain_id = rng.choice(grain_ids)
        return self.grains[random_grain_id]
