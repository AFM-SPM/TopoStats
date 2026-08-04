"""Script to re-align the images and masks for dl training."""

import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def sort_filenames(filenames: list[Path]) -> list[Path]:
    """
    Sort a list of filenames, but with underscores being treated as being less than numbers.

    Parameters
    ----------
    filenames : list[Path]
        The list of filenames to sort.

    Returns
    -------
    list[Path]
        The sorted list of filenames.
    """

    # try to treat underscores as being less than numbers, so "TE_5_" comes before "TE_50"
    def sort_key(path: Path):
        # split the filename into parts where each is either a number of a non-number string
        parts = re.split(r"(\d+|)", path.name)
        # create a key that is a list of tuples, where each tuple is like (0, ""), (1, 5), (2, "TE") etc
        key = []
        for part in parts:
            # if empty, skip
            if not part:
                continue
            # blank has lowest priority
            if part == "":
                key.append((0, ""))
            # digits have next priority, converted to int for sorting
            elif part.isdigit():
                key.append((1, int(part)))
            # everything else has higher priority, so that will be underscores, or letters etc. hopefully this
            # doesn't mess up the sorting if letters are prioritised wrongly over numbers?
            else:
                # casefold makes it case insensitive, so that "TE" and "te" are treated the same
                key.append((2, part.casefold()))
        return key

    # sorted can take a key function, where each element of form (0, ""), (1, 5), (2, "TE") etc is compared in order
    # so that the first element of the tuple is compared first, then the second, etc. This means that underscores
    # will be treated as being less than numbers, and numbers will be treated as being less than letters, and
    # letters will be treated as being less than other characters
    return sorted(filenames, key=sort_key)


if __name__ == "__main__":
    input_images_dir = Path("/Users/sylvi/topo_data/dna-damage-unet/data/cesium/group_0")
    input_labels_dir = Path("/Users/sylvi/topo_data/dna-damage-unet/data/cesium/group_0_tasks")
    output_dir = Path("/Users/sylvi/topo_data/dna-damage-unet/data/cesium/group_0_ready")

    label_files_npy = sort_filenames(list(input_labels_dir.glob("*.npy")))
    image_files_npy = sort_filenames(list(input_images_dir.glob("*.npy")))

    for image_file, label_file in zip(image_files_npy, label_files_npy):
        image = np.load(image_file)
        label = np.load(label_file)

        filename = image_file.stem
        filename = filename.replace("_image", "")

        # plot for visual inspection
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(image, cmap="gray")
        ax[1].imshow(label, cmap="gray")
        plt.suptitle(f"Image {image_file}")
        plt.savefig(output_dir / f"{filename}.png")
        plt.close()

        np.save(output_dir / f"{filename}_image.npy", image)
        np.save(output_dir / f"{filename}_label.npy", label)
        # move the metadata too
        metadata_file = input_images_dir / f"{filename}_metadata.yaml"
        assert metadata_file.exists()
        shutil.copy(metadata_file, output_dir)
