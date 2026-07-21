"""Script for turning directories of images into a gallery html."""

import base64
from pathlib import Path
import matplotlib.pyplot as plt


def images_to_gallery(
    dir_directories: Path,
) -> None:
    """
    Make a gallery html doc of all images in the directories of a given directory.

    It embeds the images, rather than just linking to them.

    Parameters
    ----------
    dir_directories : Path
        The directory containing directories of images.

    Returns
    -------
    None
    """
    html_path = dir_directories / "gallery.html"

    with Path.open(html_path, "w") as f:
        f.write("<html><body>\n")
        for dir_images in dir_directories.iterdir():
            if dir_images.is_dir():
                f.write(f"<h2>{dir_images.name}</h2>\n")
                for image_path in dir_images.iterdir():
                    if image_path.suffix in [".png", ".jpg", ".jpeg"]:
                        with Path.open(image_path, "rb") as img_file:
                            img_data = img_file.read()
                            img_base64 = base64.b64encode(img_data).decode("utf-8")
                            f.write(
                                f'<img src="data:image/png;base64,{img_base64}" '
                                f'alt="{image_path.name}" style="max-width: 300px; margin: 10px;">\n'
                            )
        f.write("</body></html>\n")


if __name__ == "__main__":
    # use current directory
    images_to_gallery(Path.cwd())
