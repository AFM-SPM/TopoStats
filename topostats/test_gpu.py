"""Test GPU functionality in isolation from TopoStats"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from topostats.grain_finding_cats_unet import predict_unet

# Grab a test image
file_path = Path("/Users/sylvi/topo_data/cats/training_data/cropped/cropped_images/image_0.npy")
image = np.load(file_path)

# Predict the mask
predicted_mask = predict_unet(image, confidence=0.5, filename="test", image_output_dir=Path("./"), model_image_size=512)

# Plot the results
fig, ax = plt.subplots(1, 2)
ax[0].imshow(image, cmap="binary")
ax[1].imshow(predicted_mask, cmap="binary")
plt.show()
