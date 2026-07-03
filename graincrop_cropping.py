import pickle
from pathlib import Path
from typing import cast

import cmapy
import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from topostats.damage.classes import GrainCollection, GrainModel

if __name__ == "__main__":

    grain_collection_path = Path("/Users/sylvi/topo_data/crossings_net/grain_collection.pkl")
    assert grain_collection_path.exists(), f"File not found: {grain_collection_path}"
    output_crop_path = Path("/Users/sylvi/topo_data/crossings_net/crops")
    output_crop_path.mkdir(parents=True, exist_ok=True)

    print("\n--- Loading the grain collection ---\n")
    with Path.open(grain_collection_path, "rb") as f:
        grain_collection: GrainCollection = cast(GrainCollection, pickle.load(f))  # noQA: S301
        assert isinstance(grain_collection, GrainCollection), "Loaded object is not a GrainCollection"

    # make a display window to show the grain images

    grain_collection_keys = list(grain_collection.grains.keys())

    key_index = 207

    x, y = 0, 0
    h, w = 100, 100

    while True:
        grain_id = grain_collection_keys[key_index]
        grain: GrainModel = grain_collection.grains[grain_id]
        assert grain is not None
        grain_image = grain.image
        grain_filename = grain.filename
        print(f"grain filename: {grain_filename}")
        folder = grain.folder
        print(f"grain folder: {folder}")
        file_name = f"{grain_filename}_grain_{grain_id}"
        print(f"file_name: {file_name}")
        # find the maximum index of saved crops for this grain by checking the existing files in the current directory)
        existing_crops = list(output_crop_path.glob(f"{file_name}_crop_*.png"))
        # remove any files that don't match the pattern
        # remove the "full" images from the list
        existing_crops = [crop_path for crop_path in existing_crops if not crop_path.stem.endswith("_full")]
        # find the maximum index from this list
        if existing_crops:
            print("already done image")
            crop_index = max(int(crop_path.stem.split("_")[-1]) for crop_path in existing_crops) + 1
        else:
            crop_index = 0
        file_name_with_index = f"{file_name}_crop_{crop_index}"

        # display the image with a rectangle around the crop area
        display_image = grain_image.copy()
        display_image = cv2.normalize(display_image, None, 0, 255, cv2.NORM_MINMAX)
        display_image = cv2.applyColorMap(display_image.astype(np.uint8), cmapy.cmap("afmhot"))
        cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("image", display_image)

        # get the actual crop
        cropped_image = grain_image[y : y + h, x : x + w]
        display_cropped_image = cv2.normalize(cropped_image, None, 0, 255, cv2.NORM_MINMAX)
        display_cropped_image = cv2.applyColorMap(display_cropped_image.astype(np.uint8), cmapy.cmap("afmhot"))
        cv2.imshow("crop", display_cropped_image)

        # wait for a key press
        key = cv2.waitKey(1)

        if key == ord("q"):
            print("Exiting...")
            cv2.destroyAllWindows()
            exit(0)
        elif key == ord("a"):
            x = max(0, x - 10)
        elif key == ord("d"):
            x = min(grain_image.shape[1] - w, x + 10)
        elif key == ord("w"):
            y = max(0, y - 10)
        elif key == ord("s"):
            y = min(grain_image.shape[0] - h, y + 10)
        elif key == ord("r"):
            if x + w + 10 <= grain_image.shape[1] and y + h + 10 <= grain_image.shape[0]:
                w += 10
                h += 10
        elif key == ord("e"):
            if w - 10 > 0 and h - 10 > 0:
                w -= 10
                h -= 10
        elif key == ord("f"):
            key_index = (key_index + 1) % len(grain_collection_keys)
            x = 0
            y = 0
        elif key == ord("g"):
            key_index = (key_index - 1) % len(grain_collection_keys)
            x = 0
            y = 0
        elif key == ord(" "):
            # save the cropped image
            print(f"output crop path: {output_crop_path}")
            print(f"file name with index: {file_name_with_index}")
            np.save(output_crop_path / f"{file_name_with_index}.npy", cropped_image)
            plt.imsave(
                output_crop_path / f"{file_name_with_index}.png",
                cropped_image,
                cmap="afmhot",
                vmin=grain_image.min(),
                vmax=grain_image.max(),
            )
            # save the full image too
            plt.imsave(
                output_crop_path / f"{file_name}_full.png",
                grain_image,
                cmap="afmhot",
                vmin=grain_image.min(),
                vmax=grain_image.max(),
            )
        crop_index += 1
