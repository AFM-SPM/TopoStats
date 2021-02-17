"""
File for tracking hard-coded variables (for now)
"""

import argparse

### List of hard-coded variables in pywgwytracing.py

# global variables
gwy_colobar_ln48 = "/module/pixmap/ztype"
gwy_linematchmethod_ln51= '/module/linematch/method'
gwy_linematchmaxdegree_ln52 = "/module/linematch/max_degree"
gwy_polylevel_coldegree_ln53 = "/module/polylevel/col_degree"

# heightediting()
datafield_grains_mark_height_ln128 = int()
gwy_polylevel_masking_ln131 = int()
gwy_linematch_masking_ln135 = int()
filterwidth_ln144 = float()

# editfile()
datafield_markheight_ln170 = int()
gwy_polylevel_masking_ln173 = int()
gwy_linematch_masking_ln177 = gwy_linematch_masking_ln135
gwy_polylevel_masking_ln181 = int()
datafield_gaussfilter_ln195 = int()

# grainfinding()
grain_gauss_ln211 = float()
datafield_mask_outliers_ln220 = float()

# remove_large_objects()
datafield_mask_outliers_ln257 = float()

# remove_small_objects()
datafield_mask_outliers_ln283 = float()

# grain_thinning()
gaussian_size_ln488 = float()

# savefiles()
savefile_type_ln535 = ["/module/pixmap/xytype"]

# saveunknownfiles()
savefile_type_ln573 = ["/module/pixmap/xytype"]

# savecroppedfiles()
savefile_type_ln645 = ["/module/pixmap/xytype"]

# __main__()
path_ln709 = str()
fileend_ln712 = []
filetype_ln713 = str()
extension_ln715 = str()
minheightscale_ln717 = float()
maxheightscale_ln718 = float()
minarea_ln722 = float()
maxdeviation_ln726 = float()
mindeviation_ln727 = float()
cropwidth_ln733 = float()
splitwidth_ln734 = float()
bins_ln736 = int()
channels_ln755 = list()


### dnatracing.py

# getNumpyArraysfromGwyddion()
sigma_ln94 = float()

# getDisorderedTrace()
sigma_ln109 = float()

# purgeObviousCrap()
maxlen_ln126 = int()

# getFittedTraces()
index_width_ln212 = int()

# getSplinedTraces()
step_size_ln376 = float()
interp_size_ln377 = float()

### tracingfuncs.py

# I didn't find any hard-coded variables so far

### statsplotting.py
### TFOplot.py
### traceplotting.py

# not checking these for now due to ongoing merging project
