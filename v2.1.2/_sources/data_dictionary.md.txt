# Data Dictionary

Output from TopoStats includes two sets of statistics in ASCII text `.csv` files. The tables below detail the columns of
these files, the data types, a description and their units where appropriate.

## `all_statistics.csv`

The `all_statistics.csv` file contains details on each grain that has been detected and traced and has the following fields.

| Column / field / feature   | Description                                                                                          | Type    | Units            |
|----------------------------|------------------------------------------------------------------------------------------------------|---------|------------------|
| `image`                    | Filename (minus extension) of scan.                                                                  | `str`   | N/A              |
| `threshold`                | Whether grain is `above` or `below` a threshold.                                                     | `str`   | N/A              |
| `molecule_number`          | Number of found grain (starts at `0`)                                                                | `int`   | N/A              |
| `centre_x`                 | x co-ordinate of grain centre.                                                                       | `float` | m                |
| `centre_y`                 | y co-ordinate of grain centre.                                                                       | `float` | m                |
| `radius_min`               | minimum distance from the centroid to edge of the grain.                                             | `float` | m                |
| `radius_max`               | maximum distance from the centroid to edge of the grain.                                             | `float` | m                |
| `radius_mean`              | mean distance from the centroid to the edge of the grain.                                            | `float` | m                |
| `radius_median`            | median distance from the centroid to the edge of the grain.                                          | `float` | m                |
| `height_min`               | Minimum height of grain.                                                                             | `float` | m                |
| `height_max`               | Maximum height of grain.                                                                             | `float` | m                |
| `height_median`            | Median height of grain.                                                                              | `float` | m                |
| `height_mean`              | Mean height of grain.                                                                                | `float` | m                |
| `volume`                   | Volume of the grain calculated as the number of pixels multiplied by each height and scaled to metres.         | `float` | m^3              |
| `area`                     | Area of the grain itself calculated as the number of pixels scaled to metres.      | `float` | m^2              |
| `area_cartesian_bbox`      | Area of the bounding box for the grain along the cartesian axes. (Not the smallest bounding box).    | `float` | m^2              |
| `smallest_bounding_width`  | Width of the smallest bounding box for the grain (not along cartesian axes).                         | `float` | m                |
| `smallest_bounding_length` | Length of the smallest bounding box for the grain (not along cartesian axes).                        | `float` | m                |
| `smallest_bounding_area`   | Area of the smallest bounding box for the grain (not along cartesian axes).                          | `float` | m^2              |
| `aspect_ratio`             | Aspect ratio of the grain (length / width), always >= 1.                                             | `float` | N/A              |
| `max_feret`                | Longest length of the grain (see [Feret diameter](https://en.wikipedia.org/wiki/Feret_diameter)).    | `float` | m                |
| `min_feret`                | Shortest width of the grain (see [Feret diameter](https://en.wikipedia.org/wiki/Feret_diameter)).    | `float` | m                |
| `contour_length`          | UNKNOWN                                                                                              | `float` | m                |
| `circular`                 | Whether the grain is a circular loop or not.                                                         | `float` | `True` / `False` |
| `end_to_end_distance`      | UNKNOWN                                                                                              | `float` | m                |
| `basename`                 | Directory in which images was found.                                                                 | `str`   | N/A              |

## `image_stats.csv`

The `image_stats.csv` summarises the metrics

| Column / field / feature | Description                                                                                     | Type    | Units |
|--------------------------|-------------------------------------------------------------------------------------------------|---------|-------|
| `image`                  | Filename of image statistics pertain to.                                                        | `str`   | N/A   |
| `image_size_x_m`         | Width of image.                                                                                 | `float` | m     |
| `image_size_y_m`         | Height of image.                                                                                | `float` | m     |
| `image_area_m2`          | Area of image (width x height).                                                                 | `float` | m^2   |
| `image_size_x_px`        | Width of image in pixels.                                                                       | `int`   | N/A   |
| `image_size_y_px`        | Height of image in pixels.                                                                      | `int`   | N/A   |
| `image_area_px2`         | Area of image in pixels squared.                                                                | `int`   | N/A   |
| `grains_number_above`    | Number of grains found above threshold.                                                         | `int`   | N/A   |
| `grains_per_m2_above`    | Density of grains above upper threshold.                                                        | `int`   | N/A   |
| `grains_number_below`    | Number of grains found below threshold.                                                         | `int`   | N/A   |
| `grains_per_m2_below`    | Density of grains below lower threshold.                                                        | `int`   | N/A   |
| `rms_roughness`          | Route Mean Square Roughness, the square root of the mean squared heights across the surface[^1] | `float` | N/A   |

[^1] [Surface Roughness](https://www.sciencedirect.com/topics/materials-science/surface-roughness); [Surface roughness -
Wikipedia](https://en.wikipedia.org/wiki/Surface_roughness)
