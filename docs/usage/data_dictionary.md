# Data Dictionary

Output from TopoStats includes two sets of statistics in ASCII text `.csv` files. The tables below detail the columns of
these files, the data types, a description and their units where appropriate.

## `all_statistics.csv`

The `all_statistics.csv` file contains details on each grain that has been detected and traced and has the following fields.

| Column / field / feature      | Description                                                                                                                                                                                                          | Type      | Units  |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- | ------ |
| `image`                       | Filename (minus extension) of scan.                                                                                                                                                                                  | `str`     | N/A    |
| `threshold`                   | The index of the threshold used from the list.                                                                                                                                                                       | `str`     | N/A    |
| `grain_number`                | Number of found grain (starts at `0`)                                                                                                                                                                                | `int`     | N/A    |
| `centre_x`                    | x coordinate of grain centre.                                                                                                                                                                                        | `float`   | m      |
| `centre_y`                    | y coordinate of grain centre.                                                                                                                                                                                        | `float`   | m      |
| `radius_min`                  | minimum distance from the centroid to edge of the grain.                                                                                                                                                             | `float`   | m      |
| `radius_max`                  | maximum distance from the centroid to edge of the grain.                                                                                                                                                             | `float`   | m      |
| `radius_mean`                 | mean distance from the centroid to the edge of the grain.                                                                                                                                                            | `float`   | m      |
| `radius_median`               | median distance from the centroid to the edge of the grain.                                                                                                                                                          | `float`   | m      |
| `height_min`                  | Minimum height of grain.                                                                                                                                                                                             | `float`   | m      |
| `height_max`                  | Maximum height of grain.                                                                                                                                                                                             | `float`   | m      |
| `height_median`               | Median height of grain.                                                                                                                                                                                              | `float`   | m      |
| `height_mean`                 | Mean height of grain.                                                                                                                                                                                                | `float`   | m      |
| `volume`                      | Volume of the grain calculated as the number of pixels multiplied by each height and scaled to metres.                                                                                                               | `float`   | m^3    |
| `area`                        | Area of the grain itself calculated as the number of pixels scaled to metres.                                                                                                                                        | `float`   | m^2    |
| `area_cartesian_bbox`         | Area of the bounding box for the grain along the cartesian axes. (Not the smallest bounding box).                                                                                                                    | `float`   | m^2    |
| `smallest_bounding_width`     | Width of the smallest bounding box for the grain (not along cartesian axes).                                                                                                                                         | `float`   | m      |
| `smallest_bounding_length`    | Length of the smallest bounding box for the grain (not along cartesian axes).                                                                                                                                        | `float`   | m      |
| `smallest_bounding_area`      | Area of the smallest bounding box for the grain (not along cartesian axes).                                                                                                                                          | `float`   | m^2    |
| `aspect_ratio`                | Aspect ratio of the grain (length / width), always >= 1.                                                                                                                                                             | `float`   | N/A    |
| `max_feret`                   | Longest length of the grain (see [Feret diameter](https://en.wikipedia.org/wiki/Feret_diameter)).                                                                                                                    | `float`   | m      |
| `min_feret`                   | Shortest width of the grain (see [Feret diameter](https://en.wikipedia.org/wiki/Feret_diameter)).                                                                                                                    | `float`   | m      |
| `basename`                    | Directory in which images was found.                                                                                                                                                                                 | `str`     | N/A    |
| `grain_endpoints`             | The number of pixels designated as endpoints (only 1 neighbour) in the pruned skeleton. **NB** molecules with zero end-points are circular/closed loops.                                                             | `integer` | N/A    |
| `grain_junctions`             | The number of pixels designated as junctions (>2 neighbours) in the pruned skeleton.                                                                                                                                 | `integer` | N/A    |
| `total_branch_length`         | The sum of all branch lengths in the pruned skeleton.                                                                                                                                                                | `float`   | m      |
| `grain_width_mean`            | The mean width of the grain.                                                                                                                                                                                         | `float`   | m      |
| `num_crossings`               | The number of crossing regions found in the grain. Note: this will be equal to or lower than the number of junctions explained in the previous section.                                                              | `integer` | N/A    |
| `avg_crossing_confidence`     | The average of all pseudo crossing confidences. Used to estimate quality of predictions.                                                                                                                             | `float`   | N/A    |
| `min_crossing_confidence`     | The minimum of all pseudo crossing confidences. Used to estimate quality of predictions.                                                                                                                             | `float`   | N/A    |
| `num_molecules`               | The number of molecules found by following the tracing paths. Note: This will always be 1 for the TopoStats method.                                                                                                  | `integer` | N/A    |
| `writhe_string`               | The writhe sign (+/-) which describes the crossing directionality. If a crossing contains > 2 crossing branches, the single crossing region is split into pairs and the writhe calculated in brackets i.e. "+(-++)". | `str`     | N/A    |
| `curvature_grain_num_turns`   | The number of turns in the grain calculated from the curvature.                                                                                                                                                      | `int`     | N/A    |
| `curvature_mean`              | The mean curvature for the grain including all molecules.                                                                                                                                                            | `float`   | 1/nm   |
| `curvature_max`               | The maximum curvature for the grain including all molecules.                                                                                                                                                         | `float`   | 1/nm   |
| `curvature_min`               | The minimum curvature for the grain including all molecules.                                                                                                                                                         | `float`   | 1/nm   |
| `curvature_std`               | The standard deviation of curvatures for the grain including all molecules.                                                                                                                                          | `float`   | 1/nm   |
| `curvature_var`               | The variance of curvatures for the grain including all molecules.                                                                                                                                                    | `float`   | 1/nm^2 |
| `curvature_total`             | The total curvature for the grain including all molecules.                                                                                                                                                           | `float`   | 1/nm   |
| `curvature_median`            | The median curvature for the grain including all molecules.                                                                                                                                                          | `float`   | 1/nm   |
| `curvature_iqr`               | Interquartile range of curvatures for the grain including all molecules.                                                                                                                                             | `float`   | 1/nm   |
| `curvature_90th`              | The 90th percentile curvature for the grain including all molecules.                                                                                                                                                 | `float`   | 1/nm   |
| `total_contour_length`        | The total length along the splined trace of all identified molecules.                                                                                                                                                | `float`   | m      |
| `average_end_to_end_distance` | The average distance from two endpoints of the spline of all identified linear molecules.                                                                                                                            | `float`   | m      |

## `image_stats.csv`

The `image_stats.csv` summarises the metrics for a processed image as a whole. The fields are as follows:

| Column / field / feature | Description                                                                                                                                                                                                                                                                  | Type    | Units |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | ----- |
| `image`                  | Filename of image statistics pertain to.                                                                                                                                                                                                                                     | `str`   | N/A   |
| `image_size_x_m`         | Width of image.                                                                                                                                                                                                                                                              | `float` | m     |
| `image_size_y_m`         | Height of image.                                                                                                                                                                                                                                                             | `float` | m     |
| `image_area_m2`          | Area of image (width x height).                                                                                                                                                                                                                                              | `float` | m^2   |
| `image_size_x_px`        | Width of image in pixels.                                                                                                                                                                                                                                                    | `int`   | N/A   |
| `image_size_y_px`        | Height of image in pixels.                                                                                                                                                                                                                                                   | `int`   | N/A   |
| `image_area_px2`         | Area of image in pixels squared.                                                                                                                                                                                                                                             | `int`   | N/A   |
| `grains_number`          | Number of grains found above/ below threshold.                                                                                                                                                                                                                               | `int`   | N/A   |
| `grains_per_m2`          | Density of grains above/ below the threshold.                                                                                                                                                                                                                                | `int`   | N/A   |
| `rms_roughness`          | Root Mean Square Roughness, the square root of the mean squared heights across the surface ([Surface Roughness](https://www.sciencedirect.com/topics/materials-science/surface-roughness); [Surface roughness (Wikipedia)](https://en.wikipedia.org/wiki/Surface_roughness)) | `float` | N/A   |

## `branch_statistics.csv`

The `branch_statistics.csv` file contains details on each branch that has been detected and traced and has the
following fields:

| Column / field / feature | Description                                                                                                                | Type    | Units |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------- | ------- | ----- |
| `image`                  | Filename (minus extension) of scan.                                                                                        | `str`   | N/A   |
| `grain_number`           | Id of the grain the branch is a part of.                                                                                   | `int`   | N/A   |
| `index`                  | Index of the branch within the grain.                                                                                      | `int`   | N/A   |
| `branch_distance`        | Distance of the branch from end to end.                                                                                    | `float` | nm    |
| `branch_type`            | Branch classification of endpoint-to-endpoint (0), endpoint-to-junction (1), junction-to-junction (2), isolated cycle (3). | `int`   | N/A   |
| `connected_segments`     | The index of the branch segments that this current branch is connected to via a junction point.                            | `list`  | N/A   |
| `mean_pixel_value`       | Average pixel value across the branch.                                                                                     | `float` | nm    |
| `std_pixel_value`        | Std pixel value across the branch.                                                                                         | `float` | nm    |
| `min_value`              | Minimum value of all pixels across the branch.                                                                             | `float` | nm    |
| `median_value`           | Median value of all pixels across the branch.                                                                              | `float` | nm    |
| `basename`               | Directory in which images was found.                                                                                       | `str`   | N/A   |

The `matched_branch_statistics.csv` file contains data about matched branches that have been detected and has the
following fields:

| Column / field / feature | Description                              | Type          | Units |
| ------------------------ | ---------------------------------------- | ------------- | ----- |
| `image`                  | Filename (minus extension) of scan.      | `str`         | N/A   |
| `grain_number`           | Id of the grain the branch is a part of. | `int`         | N/A   |
| `node`                   | Id of the node.                          | `int`         | N/A   |
| `branch`                 | Id of the branch.                        | `int`         | N/A   |
| `fwhm`                   | Full-width half maximum.                 | `float`       | nm    |
| `fwhm_half_maxs`         | Half-maximums from a matched branch.     | `list[float]` | nm    |
| `fwhm_peaks`             | Pea s from a matched branch.             | `list[float]` | nm    |
| `basename`               | D rectory in which images was found.     | `str`         | N/A   |

## `molecule_statistics.csv`

The `molecule_statistics.csv` file contains data about molecules that have been detected and has the following fields:

| Column / field / feature | Description                                                                       | Type    | Units  |
| ------------------------ | --------------------------------------------------------------------------------- | ------- | ------ |
| `image`                  | Filename (minus extension) of scan.                                               | `str`   | N/A    |
| `grain_number`           | Id of the grain the molecule is a part of.                                        | `int`   | N/A    |
| `molecule_number`        | Id of the molecule within its grain.                                              | `int`   | N/A    |
| `circular`               | If the molecule is circular, meaning it has no endpoints.                         | `bool`  | N/A    |
| `contour_length`         | Length of the contour                                                             | `float` | nm     |
| `curvature_num_turns`    | The number of turns in the molecule calculated from the curvature.                | `int`   | N/A    |
| `curvature_mean`         | The mean curvature for the molecule.                                              | `float` | 1/nm   |
| `curvature_max`          | The maximum curvature for the molecule.                                           | `float` | 1/nm   |
| `curvature_min`          | The minimum curvature for the molecule.                                           | `float` | 1/nm   |
| `curvature_std`          | The standard deviation of curvatures for the molecule.                            | `float` | 1/nm   |
| `curvature_var`          | The variance of curvatures for the molecule.                                      | `float` | 1/nm^2 |
| `curvature_total`        | The total curvature for the molecule.                                             | `float` | 1/nm   |
| `curvature_median`       | The median curvature for the molecule.                                            | `float` | 1/nm   |
| `curvature_iqr`          | Interquartile range of curvatures for the molecule.                               | `float` | 1/nm   |
| `curvature_90th`         | The 90th percentile curvature for the molecule.                                   | `float` | 1/nm   |
| `end_to_end_distance`    | Distance between start and end points of a molecule, if circular this value is 0. | `float` | nm     |
| `topology`               | Topological classification of the molecule.                                       | `str`   | N/A    |
| `topology`               | Reverse of the topological classification of the molecule.                        | `str`   | N/A    |
