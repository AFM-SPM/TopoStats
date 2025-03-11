# Matplotlib Style

TopoStats includes its own [Matplotlib][mpl] style file . This resides at `topostats/topostats.mplstyle` (see also
[GitHub repository][topostats-mpl]).

If you wish to customise the style of plots you can create a copy of this using `topostats create-matplotlibrc`. For
more information see

```shell
topostats create-matplotlibrc --help
```

Once you have modified and saved the file you can run your analyses with it using...

```shell
topostats --matplotlibrc <filename>
```

Alternatively you can change the parameters in a custom configuration file to point to the newly created style file.

## Color Maps

Several custom [colormaps][mpl-colormaps] for plotting data are also included. These are defined within the
`topostats.themes.Colormap` class. For full details refer to the [API](../api/theme.md).

- `nanoscope` colormap is provided and used by default.
- `gwyddion` colormap is provided that matches the colormap used by default in the [Gwyddion][gwyddion]
- `blue` colormap is used for masks by default.
- `blue_purple_green` colormap is available and is used when plotting traces of overlapping molecules.

[gwyddion]: https://gwyddion.net/
[mpl]: https://matplotlib.org/
[mpl-colormaps]: https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.Colormap.html#matplotlib.colors.Colormap
[topostats-mpl]: https://github.com/AFM-SPM/TopoStats/blob/main/topostats/topostats.mplstyle
