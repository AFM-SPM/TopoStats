"""Script to collate the labels of labelled data into single npy files from disparate files."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def get_class_name_from_filename(filename: str, classes: list[str]) -> str | None:
    """
    Determine which class the file belongs to based on its filename.

    Parameters
    ----------
    filename : str
        The filename to check.
    classes : list[str]
        The list of class names to check against.

    Returns
    -------
    str | None
        The class name if found, otherwise None.
    """
    for class_name in classes:
        if class_name in filename:
            return class_name
    return None


if __name__ == "__main__":
    input_labels_dir = Path("/Users/sylvi/topo_data/crossings_net/dataset_false_true_slot/labels_true_slot")
    output_labels_dir = input_labels_dir.parent / f"{input_labels_dir.name}_combined"
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    classes = ["strand1", "strand2"]
    image_size = (64, 64)
    always_all_classes = True  # ensure each image has all classes

    # grab all label files
    label_files_npy = sorted(input_labels_dir.glob("*.npy"))

    # for each set of files that are the same except containing different label classes, merge the labels into one
    # numpy file
    processed_files = set()
    for label_file_npy in label_files_npy:
        print(f"processing file {label_file_npy}")

        # get the class name from the filename
        file_class_name = get_class_name_from_filename(label_file_npy.name, classes)
        assert file_class_name is not None, f"Could not determine class name from filename: {label_file_npy.name}"
        label_filename_without_class = label_file_npy.name.replace(f"{file_class_name}", "")

        print(f" | label_filename_without_class: {label_filename_without_class}")

        partner_files = []
        if label_file_npy in processed_files:
            continue

        for other_label_file_npy in label_files_npy:
            if other_label_file_npy in processed_files:
                continue

            other_file_class_name = get_class_name_from_filename(other_label_file_npy.name, classes)
            other_filename_without_class = other_label_file_npy.name.replace(f"{other_file_class_name}", "")

            if label_filename_without_class == other_filename_without_class:
                partner_files.append(other_label_file_npy)

        if always_all_classes:
            assert len(partner_files) == len(
                classes
            ), f"Expected {len(classes)} partner files, but found {len(partner_files)} for {label_filename_without_class}"

        # merge the labels into one numpy file
        label_tensor = np.zeros((len(classes), *image_size), dtype=np.uint8)
        for file in partner_files:
            label = np.load(file)
            assert (
                label.shape == image_size
            ), f"Label shape {label.shape} does not match expected image size {image_size}"
            file_class_name = get_class_name_from_filename(file.name, classes)
            assert file_class_name is not None, f"Could not determine class name from filename: {file.name}"
            class_index = classes.index(file_class_name)
            label_tensor[class_index, :, :] = label

        # save the merged label tensor
        merged_label_file_npy = output_labels_dir / label_filename_without_class
        np.save(merged_label_file_npy, label_tensor)

        # save the merged label tensor as a png image
        merged_label_file_png = output_labels_dir / label_filename_without_class.replace(".npy", ".png")
        fig, ax = plt.subplots(1, len(classes), figsize=(len(classes) * 3, 3))
        for i, class_name in enumerate(classes):
            ax[i].imshow(label_tensor[i, :, :], cmap="viridis")
            ax[i].set_title(class_name)
            ax[i].axis("off")
        plt.tight_layout()
        plt.savefig(merged_label_file_png)
        plt.close()

        # mark the partner files as processed
        processed_files.update(partner_files)

        print(f"Processed {len(partner_files)} files into {merged_label_file_npy} and {merged_label_file_png}")
