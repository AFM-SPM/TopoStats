# Grainstats

## At a Glance - Measures Objects

TopoStats automatically tries to measure the grains (objects of interest) found in the grain finding section, in your
AFM images, and outputs them into the `all_statistics.csv` file.

The metrics are briefly summarised in the table below:

| Column Name                      | Description                                                                                                                                                | Data Type |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- |
| `center_x/y`                     | The center of the grain.                                                                                                                                   | `float`   |
| `radius_min/max/mean/median`     | The distance from the center to each pixel on the perimeter.                                                                                               | `float`   |
| `height_min/max/mean/median`     | The pixel values underlying the grain mask.                                                                                                                | `float`   |
| `area`                           | The area of the pixel-wise grain mask.                                                                                                                     | `float`   |
| `volume`                         | Volume of the pixel-wise grain mask.                                                                                                                       | `float`   |
| `area_cartesian_bbox`            | The area of a box bounding the grain along cardinal directions.                                                                                            | `float`   |
| `smallest_bounding_width/length` | The shortest bounding box length and perpendicular width of the grain in non-cardinal directions.                                                          | `float`   |
| `smallest_bounding_area`         | The area of the smallest possible box bounding the grain.                                                                                                  | `float`   |
| `aspect_ratio`                   | Ratio of the smallest bounding width to smallest bounding length.                                                                                          | `float`   |
| `max/min_feret`                  | The largest and shortest distance of the calipers rotating the grain between calipers. See [feret diameter](https://en.wikipedia.org/wiki/Feret_diameter). | `float`   |

&nbsp;

![Grain Stats image table pt1](../_static/images/grainstats/ts2_gs_metrics.png)
