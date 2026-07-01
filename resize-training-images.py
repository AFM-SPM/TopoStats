"""Script for resizing the images before labelling."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize

from topostats.plottingfuncs import Colormap

CMAP = Colormap().get_cmap()
VMIN = -3
VMAX = 4


def resize_images_in_directory(
    directory: Path,
    target_size: tuple[int, int],
) -> None:
    """
    Recursively resize all images in the specified directory to the target size.

    Parameters
    ----------
    directory : Path
        The directory containing the images to resize.
    target_size : tuple[int, int]
        The target size for the images, specified as (width, height).
    """
    resized_dir = directory.parent / f"{directory.name}_resized_{target_size[0]}x{target_size[1]}"
    resized_dir.mkdir(parents=True, exist_ok=True)

    for image_path in directory.rglob("*.npy"):
        image = np.load(image_path)
        resized_image = resize(image, target_size, anti_aliasing=False, preserve_range=True)
        resized_image_path = resized_dir / image_path.name
        np.save(resized_image_path, resized_image)
        plt.imsave(resized_image_path.with_suffix(".png"), resized_image, cmap=CMAP, vmin=VMIN, vmax=VMAX)


if __name__ == "__main__":
    directory = Path("/Users/sylvi/topo_data/crossings_net/crops_sorted")
    target_size = (64, 64)
    resize_images_in_directory(directory, target_size)
