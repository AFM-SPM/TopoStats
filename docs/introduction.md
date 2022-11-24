# Introduction

TopoStats is a [Python](https://www.python.org/) package that aims to simplify batch processing Atomic
Force Microscopy (AFM) images.

Input directories are recursively searched for files of a given type. Each image is then loaded and processed. Multiple
images can be processed in parallel.

Once an image has been loaded the specified channel of data extracted along with the pixel to nanometre scaling. This
data is then aligned and the tilt is removed. Configurable thresholds are then used to generate masks and a second round of tilt
removal and row alignment is performed.

Molecules/regions of interest known as Grains are then detected based on user specified thresholds and the detected
regions are labelled and have preliminary statistics calculated. The labelled regions of each grain then have individual
statistics calculated capturing the height, volume, radius and the location of the centroid.

Optionally DNA Tracing is then performed, which traces the backbone of the DNA molecules to calculate further statistics
such as whether grains are linear or circular, their contour length and end-to-end distances etc.

The resulting statistics are written to a [CSV file](data_dictionary) and optionally plots are then generated from
various stages of the processing as well as cropped images of each grain. The amount of images produced is also
configurable.


An schematic overview of the classes and methods that are run in processing files can be found in the
[workflows](workflows) page along with more detailed information on [installation](installation), [usage](usage),
[configuration](configuration) and [contributing](contributing).

If you have questions, please post them on the [discussion](https://github.com/AFM-SPM/TopoStats/discussions), if you
think you've encountered a bug whilst running the code or suggestions for improvements please create an
[issue](https://github.com/AFM-SPM/TopoStats/issues) in the GitHub project page.
