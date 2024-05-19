
from pathlib import Path

import numpy as np
import cv2
import cmapy

image_index = 0
file = files[image_index]
# Find the pixel to floating point nm conversion factor from the filename that occurs after the "_p2nm_" string. It contains a decimal point.
# px_to_nm = float(re.search(r"(?<=_p2nm_)\d+\.\d+", file.stem).group(0))
# print(f"px to nm: {px_to_nm}")
image = np.load(file)

bounding_box_size = 120
# Define the bounding box
x, y, w, h = 100, 100, bounding_box_size, bounding_box_size

window_name = "image_display"
cropped_window_name = "cropped_image_display"

while True:
    # Get cropped image
    cropped_image = image[y : y + h, x : x + w]
    cropped_image_rgb = cropped_image.copy()

    crop_size_nm = px_to_nm * w

    # Make a copy of the image
    display_image = image.copy()

    # Turn the heightmap into a color image
    display_image_norm = cv2.normalize(display_image, None, 0, 255, cv2.NORM_MINMAX)
    display_image = cv2.applyColorMap(display_image_norm.astype(np.uint8), cmapy.cmap("afmhot"))

    # Draw the bounding box
    cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Show the image with the bounding box and also show the cropped image. They cannot be hstacked though because they are different sizes
    file_name = file.stem
    cv2.imshow(window_name, display_image)

    # Apply a colormap to the cropped image where the minimum and maximum are set to the minimum and maximum of the original image
    cropped_image_rgb = cv2.normalize(cropped_image_rgb, None, 0, 255, cv2.NORM_MINMAX)
    cropped_image_rgb = cv2.applyColorMap(cropped_image_rgb.astype(np.uint8), cmapy.cmap("afmhot"))
    cv2.imshow(cropped_window_name, cropped_image_rgb)

    # Wait for a key press
    key = cv2.waitKey(1)

    # Move the bounding box
    if key == ord("a"):
        x -= 10
    elif key == ord("d"):
        x += 10
    elif key == ord("w"):
        y -= 10
    elif key == ord("s"):
        y += 10
    elif key == ord("e"):
        w -= 10
        h -= 10
        crop_size_nm = px_to_nm * w
    elif key == ord("r"):
        w += 10
        h += 10
        crop_size_nm = px_to_nm * w
    elif key == ord("f"):
        image_index += 1
        file = files[image_index]
        # px_to_nm = float(re.search(r"(?<=_p2nm_)\d+\.\d+", file.stem).group(0))
        image = np.load(file)
    elif key == ord("g"):
        image_index -= 1
        file = files[image_index]
        # px_to_nm = float(re.search(r"(?<=_p2nm_)\d+\.\d+", file.stem).group(0))
        # print(f"loading image: {file.stem} px to nm: {px_to_nm} index: {image_index} / {len(files)}")

        image = np.load(file)
    # Save the region in the bounding box when space is pressed
    elif key == ord(" "):
        # Get the index of the output file
        output_index = len(list(CROP_OUTPUT_DIR.glob("*.npy")))

        filename = f"image_{output_index}"
        if filename + ".png" in [f.stem for f in CROP_OUTPUT_DIR.glob("*.png")]:
            print("File already exists")
            exit()

        # Save the cropped image
        # np.save(CROP_OUTPUT_DIR / f"image_{output_index}_{px_to_nm}.npy", cropped_image)
        np.save(CROP_OUTPUT_DIR / f"image_{output_index}.npy", cropped_image)
        # Save as png
        plt.imsave(
            # CROP_OUTPUT_DIR / f"image_{output_index}_{px_to_nm}.png",
            CROP_OUTPUT_DIR / f"image_{}_{output_index}.png",
            cropped_image,
            vmin=image.min(),
            vmax=image.max(),
        )
        print(f"saving image_{output_index}.png")

    if x < 0:
        x = 10
        w = 100
        h = 100
    if y < 0:
        y = 10
        w = 100
        h = 100
    if x + w > image.shape[1]:
        x = 10
        w = 100
        h = 100
    if y + h > image.shape[0]:
        y = 10
        w = 100
        h = 100

    # Quit the program when 'q' is pressed
    elif key == ord("q"):
        # Clean up
        cv2.destroyAllWindows()
        cv2.destroyWindow(file_name)
        break