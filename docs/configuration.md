## Parameter Configuration

The parameters that the software uses for analysis of the data can be configured in a [YAML](https://yaml.org/) file. An example configuration file is included in the directory `config/example.yml`. You do not need to edit the code to change the parameters.

If no configuration YAML file is found while running TopoStats, it will generate a default file.

### Example

```{yaml}
base_dir: ./tests/
output_dir: ./output
save_plots: true
warnings: ignore
cores: 4
quiet: false
file_ext: .spm
channel: Height
amplify_level: 1.0
filter:
  threshold:
    method: std_dev # Options : otsu, std_dev, absolute
    otsu_multiplier: 1.0
    std_dev: 1.0
    absolute:
      - 1
      - -1
grains:
  gaussian_size: 2.0
  gaussian_mode: nearest
  absolute_smallest_grain_size: 100
  background: 0.0
  zrange: [0, 3] # low and high height range
  mask_direction: upper # Options: upper, lower (currently only applied to thresh+mask image)
  threshold:
    method: std_dev # Options : otsu, std_dev, absolute
    otsu_multiplier: 1.0
    std_dev: 1.0
    absolute:
      - 1
      - -1
plotting:
  save: True
  colorbar: True
  cmap: nanoscope # Options : nanoscope, afmhot
```

### Core parameters

| Parameter | Description | Example | Allowable |
| --- | --- | --- | --- |
| `base_dir` | The directory (folder) in which the software should search for images to analyse. | `./tests/` | Path. |
| `output_dir` | The directory (folder) in which output should be saved. | `./output` | Path. |
| `save_plots` | Should plots be saved? | `true` | `true` or `false` |
| `warnings` | *Not currently in use.* | `ignore` | |
| `cores` | The number of processor cores to use in parallel. | `4` | `integer` <= available cores |
| `quiet` | Log errors only? | `false` | `true` or `false` |
| `file_ext` | The extension of the image files to be analysed. | `.spm` | `string` staring with a `.` |
| `channel` | The name of the channel to be analysed. This depends on the instrument used to collect the data. | `Height Sensor` | Any channel name supported by the input file. |
| `amplify_level` | Increase the value of all pixels by this multiplier. | `1.0` | `number` |

### Filter

The filter block contains one [threshold block](#threshold).

### Grains

| Parameter | Description | Example | Allowable |
| --- | --- | --- | --- |
| `gaussian_size` | The standard deviation of the Gaussian kernel used to filter grain sizes | 2.0 | Any positive real number. |
| `gaussian_mode` | *TBD* | nearest | *TBD* |
| `absolute_smallest_grain_size` | The smallest grain size for analysis (pixels?). | 100 | Positive integer. |
| `background` | *TBD* | 0.0 | *TBD* |
| `threshold` | One [threshold block](#threshold) | | |

### Threshold

| Parameter | Description | Example | Allowable |
| --- | --- | --- | --- |
| `method` | Thresholding method. | `otsu` | `otsu`, `std_dev`, `absolute` |
| `otsu_multiplier` | | `1.7` | |
| `std_dev` | | `1.0` | |
| `absolute` | List of... | `1` <br/> `-1` | |
