from PIL import Image
import configparser
import numpy as np

# Read .tiff files of masks
config = configparser.ConfigParser()
config.read("config/config.ini")
imagepath1 = config.get("MainSection", "imagepath1")
imagepath2 = config.get("MainSection", "imagepath2")
image_manual = Image.open(imagepath1)
image_automatic = Image.open(imagepath2)

# Convert images to numpy arrays
array_manual = np.array(image_manual)[:, :, 0]
array_automatic = np.array(image_automatic)[:, :, 0]
# Convert arrays to binary
array_manual[array_manual != 0] = 1
array_automatic[array_automatic != 0] = 1

# Calculate the Jaccard index
array_sum = array_manual + array_automatic
list_sum = list(array_sum.flatten())
overlap = list_sum.count(2)  # Number of pixels masked in both images
total = list_sum.count(1) + list_sum.count(2)  # Number of pixels masked in at least one image
jaccard_index = overlap / total
print(jaccard_index)