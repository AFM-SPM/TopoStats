import numpy as np
import cv2
import cmapy
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description='Image Cropper with Bounding Box')
parser.add_argument('directory', type=str, help='Directory containing PNG images')
parser.add_argument('--output', type=str, default='cropped_new', help='Output directory for cropped images')
args = parser.parse_args()

# Define the directory containing the images and the output directory for cropped images
directory = Path(args.directory)
CROP_OUTPUT_DIR = Path(directory / args.output)

# Create the output directory if it does not exist
CROP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Define files list
files = list(directory.glob('*.npy'))  # Adjusted to look for .png files

if len(files) == 0:
    print("No .npy files found in the specified directory.")
    exit()

# Initialize variables
image_index = 0
file = files[image_index]
image = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)

# Define pixel to nm conversion factor (use a placeholder if not available)
px_to_nm = 1.0  # Placeholder value

# Define the bounding box size
bounding_box_size = 256
x, y = 100, 100  # Initial coordinates
w, h = bounding_box_size, bounding_box_size

window_name = "image_display"
cropped_window_name = "cropped_image_display"

while True:
    # Ensure the bounding box stays within the image boundaries
    x = max(0, min(x, image.shape[1] - w))
    y = max(0, min(y, image.shape[0] - h))
    
    # Get cropped image
    cropped_image = image[y:y + h, x:x + w]
    cropped_image_rgb = cropped_image.copy()

    # Normalize and apply colormap to the main image
    display_image_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    display_image = cv2.applyColorMap(display_image_norm.astype(np.uint8), cmapy.cmap("afmhot"))

    # Draw the bounding box
    cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the images
    cv2.imshow(window_name, display_image)

    # Normalize and apply colormap to the cropped image
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
        w = max(10, w - 10)  # Ensure width does not go below 10
        h = max(10, h - 10)  # Ensure height does not go below 10
    elif key == ord("r"):
        w += 10
        h += 10
    elif key == ord("f"):
        image_index = (image_index + 1) % len(files)
        file = files[image_index]
        image = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
        print(f"Loaded image from: {file.stem}")
    elif key == ord("g"):
        image_index = (image_index - 1) % len(files)
        file = files[image_index]
        image = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
        print(f"Loaded image from: {file.stem}")
    elif key == ord(" "):
        # Save the cropped image
        output_index = len(list(CROP_OUTPUT_DIR.glob("*.npy")))

        filename = f"image_{output_index}"
        if filename + ".png" in [f.stem for f in CROP_OUTPUT_DIR.glob("*.png")]:
            print("File already exists")
            exit()

        np.save(CROP_OUTPUT_DIR / f"image_{output_index}.npy", cropped_image)
        plt.imsave(
            CROP_OUTPUT_DIR / f"image_{output_index}.png",
            cropped_image,
            vmin=image.min(),
            vmax=image.max(),
            cmap="afmhot"
        )
        print(f"saving image_{output_index}.png")

    # Ensure the bounding box stays within the image boundaries
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x + w > image.shape[1]:
        x = image.shape[1] - w
    if y + h > image.shape[0]:
        y = image.shape[0] - h

    # Quit the program when 'q' is pressed
    if key == ord("q"):
        cv2.destroyAllWindows()
        break