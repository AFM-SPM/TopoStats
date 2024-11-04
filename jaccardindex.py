from PIL import Image
import configparser
import numpy as np
import os

# Read .tiff or .npy files of masks
config = configparser.ConfigParser()
config.read("config/config.ini")
imagepath1 = config.get("MainSection", "imagepath1")
imagepath2 = config.get("MainSection", "imagepath2")

# Convert data into numpy array
arraylist = []
for maskpath in [imagepath1, imagepath2]:
    if os.path.splitext(maskpath)[1] == '.tiff':
        maskimage = Image.open(maskpath)
        maskarray = np.array(maskimage)[:, :, 0]
        maskarray[maskarray != 0] = 1  # Convert array into binary
        arraylist.append(maskarray)
    elif os.path.splitext(maskpath)[1] == '.npy':
        maskarray = np.load(maskpath).astype(int)
        maskarray = np.transpose(maskarray)
        maskarray[maskarray != 0] = 1  # Convert array into binary
        arraylist.append(maskarray)

# Calculate the Jaccard index
array_sum = arraylist[0] + arraylist[1]
list_sum = list(array_sum.flatten())
overlap = list_sum.count(2)  # Number of pixels masked in both images
total = list_sum.count(1) + list_sum.count(2)  # Number of pixels masked in at least one image
jaccard_index = overlap / total
print(jaccard_index)