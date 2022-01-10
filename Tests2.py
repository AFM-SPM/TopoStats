import pySPM
print(pySPM.__version__)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm

from skimage.filters import median, gaussian, sobel, threshold_otsu
from skimage.morphology import erosion, dilation, opening, closing, skeletonize, skeletonize_3d
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops, regionprops_table
from skimage.color import label2rgb

from scipy.stats import linregress

scan = pySPM.Bruker("minicircle.spm")


scan.list_channels()

topoB = scan.get_channel()
zsense = scan.get_channel("Height Sensor")
force_err = scan.get_channel("Peak Force Error")
adh = scan.get_channel("Adhesion")
deform = scan.get_channel("Deformation")
mod = scan.get_channel("DMTModulus")
log_mod = scan.get_channel("LogDMTModulus")
height = scan.get_channel("Height")

'''
# Channels that PySPM finds in the Bruker datafile
topoB = np.array(topoB.pixels)
zsense = np.array(zsense.pixels)
force_err =  np.array(force_err.pixels) 
adh =  np.array(adh.pixels) 
deform = np.array(deform.pixels) 
mod =  np.array(mod.pixels) 
log_mod = np.array(log_mod.pixels) 
height =  np.array(height.pixels) 
'''

"""
# What Ben thinks is going on with the AFM data analysis from the TopoStats paper
1) Choose a channel
2) Plot channel
3) Choose colour sale range
4) Correct background for tilt in X and Y
5) Correct for z-axis offset
6) Segment the molecules from the background
7) Skeletonise the molecules
8) Generate stats from the image
​
​
Topostats paper algorithm description
1) Raw z-scanner height image
2) Correct for tilt using a 'plane' -> 1D polynomial in both X and Y
3) ensure height variations between adjacent scanlines are zero
4) Flatten base (gwyddion function) -> facet and polynomial levelling and automated masking
5) Offset the height to ensure that the mean height value is zero.
6) Remove high frequency noise to smooth the image (gaussian filter => sigma=1 pixel)
7) 
​
"""

'''
​
### This bit is just for Ben's notes
#1) Raw z-scanner height image
img = height.copy()
​
#2) Correct for tilt using a 'plane' -> 1D polynomial in both X and Y
tilt_corr = correctTilt(img)
​
# Normalise the img (0-1024)
​
​
a = tilt_corr
# Normalised [0,1]
b = (a - np.min(a))/np.ptp(a)
​
# Normalised [0,255] as integer: don't forget the parenthesis before astype(int)
norm = (1024*(a - np.min(a))/np.ptp(a)).astype(int)        
​
# Sum along the X axis 
col_mean = norm.mean(axis=0)
​
# Normalise vectorso that mean=0
mean = np.mean(col_mean)
col_mean = col_mean - mean
col_mean = np.negative(col_mean)
​
print(col_mean)
​
# Extend the vector into a matrix where each column is the same vector
bg = np.repeat(col_mean,norm.shape[1],axis=0)
bg = bg.reshape(norm.shape)
​
# Then remove this from the image
z_offset = norm - bg
​
#3) ensure height variations between adjacent scanlines are zero
​
​
​
​
print(np.min(tilt_corr), np.max(tilt_corr))
'''

'''
bg_corr_img = bg_corr_img.astype("uint8")
​
thresh = bg_corr_img
​
thresh[thresh < 20] = 0
thresh[thresh >= 20] = 1
​
​
​
open = opening(bg_corr_img.copy())
​
​
​
skeleton = skeletonize(open)
​
'''

fig = plt.figure(figsize=(12,7))
ax1 = fig.add_subplot(241)
ax2 = fig.add_subplot(242)
ax3 = fig.add_subplot(243)
ax4 = fig.add_subplot(244)
ax5 = fig.add_subplot(245)
ax6 = fig.add_subplot(246)
ax7 = fig.add_subplot(247)
ax8 = fig.add_subplot(248)

# Plot the raw height data
ax1.imshow(np.array(height.pixels), cmap="afmhot")
ax1.set_title('1) Raw SPM Height')
ax1.set_axis_off()

# Correct for sample tilt
height = height.correct_plane()
#
ax2.imshow(np.array(height.pixels), cmap="afmhot")
ax2.set_title('2) Tilt corrected')
ax2.set_axis_off()

# Correct the lines across the AFM data
height = height.correct_lines()
#
ax3.imshow(np.array(height.pixels), cmap="afmhot")
ax3.set_title('3) Lines corrected')
ax3.set_axis_off()

#topo4 = copy.deepcopy(height) # make deepcopy of object otherwise you will just change the original
#height.correct_median_diff()
#ax4.imshow(np.array(height.pixels), cmap="afmhot")
ax4.set_title(r"Median Correction" "\n" "not good result at the moment", fontsize=10)
ax4.set_axis_off()

# Remove 'scars' from the sample
scar_rem = height.filter_scars_removal(.7,inline=False)
#
ax5.imshow(np.array(scar_rem.pixels), cmap="afmhot")
ax5.set_title('Remove scars')
ax5.set_axis_off()

# Threshold the data to remove background - I think this might be in nm units, not pixel units
thresh_val = 1
thresh = np.array(scar_rem.pixels)
thresh[thresh < thresh_val] = 0

# Apply Gaussian filter to smooth the image a bit
filtered = gaussian(thresh, sigma=1)

# Morphological filtering to refine the 'bright' objects borders slightly
filtered = opening(filtered)

ax6.imshow(filtered, cmap="afmhot")
ax6.set_title(r'Gaussian Filtered ($\sigma$=1) + Threshold')
ax6.set_axis_off()

# threshold - cant remember why this was here but it made sense a few weeks ago
map = np.zeros(filtered.shape, dtype="uint8")
class_map = np.zeros(filtered.shape)
class_map[ filtered >= thresh_val ] = 128
class_map[ filtered > 3.0  ] = 255

map[filtered >= thresh_val ] = 1

# remove artifacts connected to image border
# Seemed like a good function to try from scikit-image
cleared = clear_border(map)

# label explicitly separated image regions
label_image = label(cleared)

# Produce an image overlay - copied from https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_label.html
# to make the background transparent, pass the value of `bg_label`,
# and leave `bg_color` as `None` and `kind` as `overlay`
#image_label_overlay = label2rgb(label_image, image=image, bg_label=0)
 
# Measure the shape properties of each of the objects
regions = regionprops_table(label_image)

#print(regions)
#print(dir(regions))

# Plot the labelled regions
ax7.imshow(label_image, cmap='viridis')
ax7.set_title('Labelled Objects')
ax7.set_axis_off()

# Very basic skeletonisation using scikit-image builtin function.
skeleton = skeletonize_3d(map)
#
ax8.imshow(skeleton, cmap='gray')
ax8.set_title('Skeletonise - not complete')
ax8.set_axis_off()

plt.tight_layout()
# Save the figure to the script folder
#plt.savefig('test_analysis.png', dpi=400)
plt.show()