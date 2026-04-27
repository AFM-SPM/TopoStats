"""Classes for damage analysis."""

from collections.abc import Generator
from copy import deepcopy
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

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


class UnanalysedMoleculeData(BaseDamageAnalysis):
    """Data object to hold unanalysed molecule data."""

    molecule_id: int
    ordered_coords_heights: npt.NDArray[np.float64]
    spline_coords_heights: npt.NDArray[np.float64]
    distances: npt.NDArray[np.float64]
    circular: bool
    spline_coords: npt.NDArray[np.float64]
    ordered_coords: npt.NDArray[np.float64]
    curvature_data: MoleculeCurvatureStats | None


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


class GrainDefectData(BaseDamageAnalysis):
    """Data object to hold the defect and gap data for a grain."""

    molecule_defect_data_dict: dict[int, MoleculeDefectData] = Field(default_factory=dict)

    @computed_field
    @property
    def num_defects(self) -> int:
        """Calculate the total number of defects across all molecules."""
        return sum(molecule_defect_data.num_defects for molecule_defect_data in self.molecule_defect_data_dict.values())

    @computed_field
    @property
    def num_gaps(self) -> int:
        """Calculate the total number of gaps across all molecules."""
        return sum(molecule_defect_data.num_gaps for molecule_defect_data in self.molecule_defect_data_dict.values())

    @computed_field
    @property
    def defect_lengths_nm(self) -> list[float]:
        """Get a list of the lengths of all defects across all molecules in nanometres."""
        defect_lengths = []
        for molecule_defect_data in self.molecule_defect_data_dict.values():
            defect_lengths.extend(molecule_defect_data.defect_lengths_nm)
        return defect_lengths


class MoleculeData(UnanalysedMoleculeData):
    """Data object to hold the analysed molecule data."""

    def from_unanalysed_molecule_data(unanalysed_data: UnanalysedMoleculeData) -> "MoleculeData":
        """Create a MoleculeData object from an UnanalysedMoleculeData object."""
        return MoleculeData(
            molecule_id=unanalysed_data.molecule_id,
            ordered_coords_heights=unanalysed_data.ordered_coords_heights,
            spline_coords_heights=unanalysed_data.spline_coords_heights,
            distances=unanalysed_data.distances,
            circular=unanalysed_data.circular,
            spline_coords=unanalysed_data.spline_coords,
            ordered_coords=unanalysed_data.ordered_coords,
            curvature_data=unanalysed_data.curvature_data,
        )


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
    node_coords: npt.NDArray[np.float64]
    num_nodes: int

    def __str__(self) -> str:
        """Return a simplified string representation of the grain."""
        return (
            f"GrainModel(global_grain_id={self.global_grain_id}), {self.percent_damage}% "
            f"damage, from file {self.filename}."
        )

    def plot(self, mask_alpha: float = 0.3) -> None:
        """Plot the grain image with the mask overlaid."""
        plt.imshow(self.image, **IMGPLOTARGS)
        plt.imshow(self.mask[:, :, 1], alpha=mask_alpha, cmap="gray")
        plt.title(f"grain {self.global_grain_id}, {self.percent_damage}% damage")
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
    ) -> "MoleculeDataCollection":
        """Create a MoleculeDataCollection object from an UnanalysedMoleculeDataCollection object."""
        molecule_data_dict = {}
        for molecule_id, unanalysed_molecule_data in unanalysed_collection.molecules.items():
            molecule_data = MoleculeData.from_unanalysed_molecule_data(unanalysed_molecule_data)
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

    curvature_defect_data: GrainDefectData = Field(default_factory=GrainDefectData)
    height_defect_data: GrainDefectData = Field(default_factory=GrainDefectData)
    molecule_data_collection: MoleculeDataCollection

    def from_unanalysed_grain(unanalysed_grain: UnanalysedGrain) -> "GrainModel":
        """Create a GrainModel object from an UnanalysedGrain object."""
        # Create the new molecule data collection
        molecule_data_collection = MoleculeDataCollection.from_unanalysed_molecule_data_collection(
            unanalysed_grain.molecule_data_collection
        )
        return GrainModel(
            global_grain_id=unanalysed_grain.global_grain_id,
            file_grain_id=unanalysed_grain.file_grain_id,
            filename=unanalysed_grain.filename,
            pixel_to_nm_scaling=unanalysed_grain.pixel_to_nm_scaling,
            folder=unanalysed_grain.folder,
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
            node_coords=unanalysed_grain.node_coords,
            num_nodes=unanalysed_grain.num_nodes,
        )

    def __str__(self) -> str:
        """Return a simplified string representation of the grain."""
        return (
            f"GrainModel(global_grain_id={self.global_grain_id}), {self.percent_damage}% damage, "
            f"with {len(self.molecule_data_collection)} molecules, {self.num_crossings} crossings "
            f"from file {self.filename}."
        )

    def plot(  # noqa: C901
        self, mask_alpha: float = 0.3, linemode: str = "", curvature_defects: bool = False, height_defects: bool = False
    ) -> None:
        """Plot the grain image with the mask and molecule data overlaid."""
        plt.imshow(self.image, **IMGPLOTARGS)
        plt.imshow(self.mask[:, :], alpha=mask_alpha, cmap="gray")
        if linemode == "spline":
            for _molecule_id, molecule_data in self.molecule_data_collection.items():
                spline_coords = molecule_data.spline_coords
                plt.plot(spline_coords[:, 1], spline_coords[:, 0])
        elif linemode == "curvature":
            for _molecule_id, molecule_data in self.molecule_data_collection.items():
                spline_coords = molecule_data.spline_coords
                curvature_data = molecule_data.curvature_data
                if curvature_data is not None:
                    curvature_values = curvature_data.curvatures
                    # plot the curvature values as a colormap along the spline coords
                    assert len(curvature_values) == len(spline_coords), (
                        f"length of curvature values {len(curvature_values)} does not match"
                        f"length of spline coords {len(spline_coords)}"
                    )
                    curvature_norm_bounds_lower = -0.1
                    curvature_norm_bounds_upper = 0.1
                    curvature_values_clipped = np.clip(
                        curvature_values, curvature_norm_bounds_lower, curvature_norm_bounds_upper
                    )
                    curvature_values_normalised = (curvature_values_clipped - curvature_norm_bounds_lower) / (
                        curvature_norm_bounds_upper - curvature_norm_bounds_lower
                    )
                    curvature_cmap = plt.get_cmap("coolwarm")
                    for index, point in enumerate(spline_coords):
                        color = curvature_cmap(curvature_values_normalised[index])
                        if index > 0:
                            previous_point = spline_coords[index - 1]
                            plt.plot(
                                [previous_point[1], point[1]],
                                [previous_point[0], point[0]],
                                color=color,
                                linewidth=1,
                            )
        if curvature_defects:
            # plot all the curvature defects as pink dots
            for molecule_id, molecule_defect_data in self.curvature_defect_data.molecule_defect_data_dict.items():
                for item in molecule_defect_data.ordered_defects_and_gaps.defect_gap_list:
                    if isinstance(item, Defect):
                        defect_start_index = item.start_index
                        defect_end_index = item.end_index
                        spline_coords = self.molecule_data_collection[molecule_id].spline_coords
                        defect_coords = spline_coords[defect_start_index:defect_end_index]
                        plt.scatter(defect_coords[:, 1], defect_coords[:, 0], color="magenta", s=10)
        if height_defects:
            # plot all the height defects as cyan dots
            for molecule_id, molecule_defect_data in self.height_defect_data.molecule_defect_data_dict.items():
                for item in molecule_defect_data.ordered_defects_and_gaps.defect_gap_list:
                    if isinstance(item, Defect):
                        defect_start_index = item.start_index
                        defect_end_index = item.end_index
                        spline_coords = self.molecule_data_collection[molecule_id].spline_coords
                        defect_coords = spline_coords[defect_start_index:defect_end_index]
                        plt.scatter(defect_coords[:, 1], defect_coords[:, 0], color="cyan", s=10)
        plt.show()


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
    ) -> "GrainCollection":
        """Create a GrainCollection object from an UnanalysedGrainCollection object."""
        grain_dict = {}
        for global_grain_id, unanalysed_grain in unanalysed_collection.unanalysed_grains.items():
            grain_model = GrainModel.from_unanalysed_grain(unanalysed_grain)
            grain_dict[global_grain_id] = grain_model
        return GrainCollection(grains=grain_dict, current_global_grain_id=unanalysed_collection.current_global_grain_id)
