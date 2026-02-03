# Classes

TopoStats uses its own [Pydantic Data Classes][pydantic_data_classes] to store the original data, configuration settings
and derived datasets. These are described in detail in the API. Using Pydantic means that data validation, checking that
an attribute which is meant to be an integer is an integer, or that an attribute that is meant to be a Numpy array
actually is, is done automatically. This makes code development much easier as the automated validation avoids errors
that can arise when passing data of the wrong "type" into dataclasses that is inherent in dynamically typed languages
such as Python.

This page aims to...

- Give an overview of the classes and their attributes.
- Details how to go about adding classes and all of the additional changes that are required.
- How to load `.topostats` files and reconstruct the classes.

## Class Overview

| Class             | Description                                                                              |
| ----------------- | ---------------------------------------------------------------------------------------- |
| `TopoStats`       | Top-level object, all other classes are attributes of this class.                        |
| `GrainCrop`       | A region of an image that contains a "grain" (typically a molecule of some description). |
| `DisorderedTrace` | Outline of a grain, but without ordering of coordinates.                                 |
| `Node`            | A junction in a grain where one or more molecules overlap.                               |
| `OrderedTrace`    | Outline of a grain, with coordinates ordered.                                            |
| `MatchedBranch`   |                                                                                          |
| `UnmatchedBranch` |                                                                                          |
| `Molecule`        | A sub-unit of a grain after de-tangling overlapping molecules.                           |

The attributes of each class are described below. The type of an attribute is noted which informs you what the value
should be. In many cases the attributes can also be `None` which is for convenience so that intermediary objects can be
constructed, for example when first creating a `TopoStats` object after reading it from disk there will be no
`grain_crops` to include because the flattening and grain detection has not been performed.

## `TopoStats`

| Attribute             | Type                   | Description                                                          |
| --------------------- | ---------------------- | -------------------------------------------------------------------- |
| `grain_crops`         | `dict[int, GrainCrop]` | A dictionary of `GrainCrop` objects detected within the image.       |
| `filename`            | `str`                  | The filename the image was loaded from.                              |
| `pixel_to_nm_scaling` | `str`                  | Pixel to nanometre scaling, typically derived from the image itself. |
| `img_path`            | `str`                  | Original path to image.                                              |
| `image`               | `npt.NDArray`          | Flattened image (post `Filter()`).                                   |
| `image_original`      | `npt.NDArray`          | Original image.                                                      |
| `full_mask_tensor`    | `npt.NDArray`          | Tensor mask for the full image.                                      |
| `topostats_version`   | `str`                  | TopoStats version the image was last processed with.                 |
| `config`              | `dict[str, Any]`       | Configuration used when processing the grain.                        |

## `GrainCrop`

| Attribute             | Type                                        | Description                                                     |
| --------------------- | ------------------------------------------- | --------------------------------------------------------------- |
| `image`               | `npt.NDArray[np.float32]`                   | 2-D Numpy array of the cropped image.                           |
| `mask`                | `npt.NDArray[np.bool_]`                     | 3-D Numpy tensor of the cropped mask.                           |
| `padding`             | `int`                                       | Padding added to the bounding box of the grain during cropping. |
| `bbox`                | `tuple[int, int, int, int]`                 | Bounding box of the crop including padding.                     |
| `pixel_to_nm_scaling` | `float`                                     | Pixel to nanometre scaling factor for the crop.                 |
| `thresholds`          | `float`                                     | Thresholds used to find the grain.                              |
| `filename`            | `str`                                       | Filename of the image from which the crop was taken.            |
| `skeleton`            | `npt.NDArray[np.bool_]`                     | 3-D Numpy tensor of the skeletonised mask.                      |
| `height_profiles`     | `dict[int, [int, npt.NDArray[np.float32]]]` | 3-D Numpy tensor of the height profiles.                        |
| `stats`               | `dict[int, dict[int, Any]]`                 | Dictionary of grain statistics.                                 |
| `disordered_trace`    | `DisorderedTrace`                           | A disordered trace for the grain.                               |
| `nodes`               | `dict[str, Nodes]`                          | Dictionary of grain nodes.                                      |
| `ordered_trace`       | `OrderedTrace`                              | An ordered trace for the grain.                                 |
| `threshold_method`    | `str`                                       | Threshold method used to find grains.`                          |

## `DisorderedTrace`

| Attribute             | Type                     | Description                                                                                   |
| --------------------- | ------------------------ | --------------------------------------------------------------------------------------------- |
| `images`              | `dict[str: npt.NDArray]` | Dictionary of images generated during disordered tracing, should include ''pruned_skeleton''. |
| `grain_endpoints`     | `int`                    | Number of Grain endpoints.                                                                    |
| `grain_junctions`     | `int`                    | Number of Grain junctions.                                                                    |
| `total_branch_length` | `float`                  | Total branch length in nanometres.                                                            |
| `grain_width_mean`    | `float`                  | Mean grain width in nanometres.                                                               |

## `Node`

| Attribute                | Type                         | Description                                                     |
| ------------------------ | ---------------------------- | --------------------------------------------------------------- |
| `error`                  | `bool`                       | Whether an error occurred calculating statistics for this node. |
| `pixel_to_nm_scaling`    | `np.float64`                 | Pixel to nanometre scaling.                                     |
| `branch_stats`           | `dict[int, MatchedBranch]`   | Dictionary of branch statistics.                                |
| `unmatched_branch_stats` | `dict[int, UnMatchedBranch]` | Dictionary of unmatched branch statistics.                      |
| `node_coords`            | `npt.NDArray[np.int32]`      | Numpy array of node coordinates.                                |
| `confidence`             | `np.float64`                 | Confidence in ???.                                              |
| `reduced_node_area`      | `???`                        | Reduced node area.                                              |
| `node_area_skeleton`     | `npt.NDArray[np.int32]`      | Numpy array of skeleton.                                        |
| `node_branch_mask`       | `npt.NDArray[np.int32]`      | Numpy array of branch mask.                                     |
| `node_avg_mask`          | `npt.NDArray[np.int32]`      | Numpy array of averaged mask.`                                  |

## `OrderedTrace`

| Attribute             | Type                     | Description                                                                                            |
| --------------------- | ------------------------ | ------------------------------------------------------------------------------------------------------ |
| `molecule_data`       | `dict[int, Molecule]`    | Dictionary of ordered trace data for individual molecules within the grain indexed by molecule number. |
| `tracing_stats`       | `dict`                   | Tracing statistics.                                                                                    |
| `grain_molstats`      | `Any`                    | Grain molecule statistics.                                                                             |
| `molecules`           | `int`                    | Number of molecules within the grain.                                                                  |
| `writhe`              | `str`                    | The writhe sign, can be either `+`, `-` or `0` for positive, negative or no writhe.                    |
| `pixel_to_nm_scaling` | `np.float64`             | Pixel to nm scaling.                                                                                   |
| `images`              | `dict[str, npt.NDArray]` | Dictionary of diagnostic images for debugging.                                                         |
| `error`               | `bool`                   | Errors encountered?                                                                                    |

## `MatchedBranch`

| Attribute        | Type                     | Description                          |
| ---------------- | ------------------------ | ------------------------------------ |
| `ordered_coords` | `npt.NDArray[np.int32]`  | Numpy array of ordered coordinates.  |
| `heights`        | `npt.NDArray[np.number]` | Numpy array of heights.              |
| `distances`      | `npt.NDArray[np.number]` | Numpy array of distances.            |
| `fwhm`           | `float`                  | Full-width half maximum.             |
| `fwhm_half_maxs` | `list[float]`            | Half-maximums from a matched branch. |
| `fwhm_peaks`     | `list[float]`            | Peaks from a matched branch.         |
| `angles`         | `float`                  | Angle between branches.`             |

## `UnmatchedBranch`

| Attribute | Type    | Description             |
| --------- | ------- | ----------------------- |
| `angles`  | `float` | Angle between branches. |

## `Molecule`

Class for Molecules identified during ordered tracing.

| Attribute             | Type                        | Description                                                                                               |
| --------------------- | --------------------------- | --------------------------------------------------------------------------------------------------------- |
| `circular`            | `str, bool`                 | Whether the molecule is circular or linear.                                                               |
| `topology`            | `str`                       | Topological classification of the molecule.                                                               |
| `topology_flip`       | `Any`                       | Unknown?                                                                                                  |
| `ordered_coords`      | `npt.NDArray`               | Ordered coordinates for the molecule.                                                                     |
| `splined_coords`      | `npt.NDArray`               | Smoothed (aka splined) coordinates for the molecule.                                                      |
| `contour_length`      | `float`                     | Length of the molecule.                                                                                   |
| `end_to_end_distance` | `float`                     | Distance between ends of molecule. Will be `0.0` for circular molecules which don't have ends.            |
| `heights`             | `npt.NDArray`               | Height along molecule.                                                                                    |
| `distances`           | `npt.NDArray`               | Distance between points on the molecule.                                                                  |
| `curvature_stats`     | `npt.NDArray, optional`     | Angle changes along molecule. NB - These are always positive due to use of `np.abs()` during calculation. |
| `bbox`                | `tuple[int, int, int, int]` | Bounding box.                                                                                             |

## Hierarchical Structure

The top-level object is always the `TopoStats` class, the remaining objects are nested within. This nesting structure is
retained when writing to `.topostats` a custom [HDF5][hdf5] format. The Python tool [h5glance][h5glance] can be used to
show the nested structure.

The top level of nesting reflects the `TopoStats` object itself.

```shell
h5glance output/processed/minicircle_small.topostats --depth 1
```

With a `--depth 2` we can see the second level of nesting.

Individual items can be viewed with `h5glance` by specifying the path through the nesting structure as an argument at
the command line.

```shell
h5glance output/processed/minicircle_small.topostats filename
h5glance output/processed/minicircle_small.topostats grain_crop/0/
```

## Extending

Typically the unit of analysis is a `GrainCrop` so let's assume we are adding a new attribute `GrainCrop.new_feature` to
the `GrainCrop` class and it is an object of class type `NewFeature`. We define `NewFeature` in the `classes.py` module.

```python
import numpy.typing as npt
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass


@dataclass(
    repr=True,
    eq=True,
    config=ConfigDict(arbitrary_types_allowed=True, validate_assignment=True),
    validate_on_init=True,
)
class NewFeature:
    """
    A class that adds a new feature.
    """

    left: bool | None
    right: bool | None
    data: dict[int, npt.NDArray | None]
```

This is to be an attribute of `GrainCrop` so we add a new attribute to the `__init___` definition of `GrainCrop` in
`classes.py` (note that `GrainCrop` is an exception as it is _not_ a Pydantic dataclass). Don't forget to include a
description of the attribute to both the class docstring and the `__init__` docstring otherwise the Numpydoc validation
will fail on commits and/or pull requests.

```python
class GrainCrop:
    def __init__(self, new_feature: NewFeature | None = None):
        """
        Parameters
        ----------
        new_feature : NewFeature
            A ``NewFeature``.
        """

    self.new_feature = new_feature
```

### `__str__` method

The `__str__` method provides a convenient method to represent the class when called. If you would like some of the
attributes you have defined shown then you should add them here. An example is shown below.

```python
def __str__(self) -> str:
    """
    Readable attributes.

    Returns
    -------
    str
        Set of formatted statistics on matched branches.
    """
    return (
        f"left  : {self.left}\n" f"right : {self.right}\n" f"data  :\n\n {self.data}\n"
    )
```

### `__eq__` method

Adding the equality dunder method means the object can easily be compared to another of the same type and tested for
equality.

```python
def __eq__(self, other: object) -> bool:
    """
    Check equality of ``NewFeature`` object against another.

    Parameters
    ----------
    other : object
        Other object to be compared.

    Returns
    -------
    bool
        ``True`` if the objects are equal, ``False`` otherwise.
    """
    if not isinstance(other, NewFeature):
        return False
    return (
        self.left == other.left
        and self.right == other.right
        and self.data == other.data
    )
```

## IO

### Output

`TopoStats` classes are saved to HDF5 `.topostats` files and in order to include any newly defined class in these files
for subsequent processing the `io.dict_to_hdf5()` function needs to have the class adding to its list of supported
objects and a clause that recurrsively calls `io.dict_to_hdf5()` with the class converted to a dictionary as the `item`
argument.

```python
if isinstance(
    item,
    (
        list
        | str
        | int
        | float
        | np.ndarray
        | Path
        | dict
        | GrainCrop
        | GrainCropsDirection
        | ImageGrainCrops
        | Node
        | OrderedTrace
        | DisorderedTrace
        | MatchedBranch
        | Molecule
        | NewFeature
    ),
):  # noqa: UP038
    # Lists need to be converted to numpy arrays
    if isinstance(item, list):
        LOGGER.debug(f"[dict_to_hdf5] {key} is of type : {type(item)}")
        item = np.array(item)
        open_hdf5_file[group_path + key] = item
    elif isinstance(item, NewFeature):
        logger.debug(f"[dict_to_hdf5] {key} is of type : {type(item)}")
        dict_to_hdf5(
            open_hdf5_file, group_path + key + "/", item.fancy_new_class_to_dict()
        )
```

### Input

In order to work with the modular design of TopoStats we need to be able to import `.topostats` files which are in HDF5
and convert them to `TopoStats` classes with all of the nested features. HDF5 files are read by [AFMReader][afmreader]
which returns plain dictionaries. These are converted to `TopoStats` using the `io.dict_to_topostats()` function.

Where you add the new class depends on what existing Class it is an attribute of. In this example it is an attribute of
`GrainCrop` and so it should be added within the loop that iterates over the `crops` dictionary that has been read. Not
all attributes will be present so we use an `if ... else None` single line construct to unpack the components of the
nested dictionary to the attribute only if the key is present.

```python
for grain, crop in dictionary["image_grain_crops"][direction]["crops"].items():
    image = crop["image"] if "image" in crop.keys() else None
    ...
    new_feature = (
        NewFeature(**crop["new_feature"]) if "new_feature" in crop.keys() else None
    )
```

[afmreader]: https://afm-spm.github.io/AFMReader
[h5glance]: https://github.com/European-XFEL/h5glance
[hdf5]: https://www.hdfgroup.org/solutions/hdf5/
[pydantic_data_classes]: https://docs.pydantic.dev/latest/concepts/dataclasses/
