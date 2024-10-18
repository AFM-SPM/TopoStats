# Data Dictionary

The resulting statistics file has the following fields.

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
