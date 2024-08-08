"""Order single pixel skeletons with or without NodeStats Statistics."""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt
import pandas as pd
from skimage.morphology import label

from topostats.logs.logs import LOGGER_NAME
from topostats.tracing.tracingfuncs import coord_dist, genTracingFuncs, order_branch, reorderTrace
from topostats.utils import convolve_skeleton, coords_2_img

LOGGER = logging.getLogger(LOGGER_NAME)


class splineTrace:

    def __init__(
        self,
        image: npt.NDArray,
        ordered_tracing_data: dict,
        pixel_to_nm_scaling: float,
        filename: str,
        splining_config: dict,
        results_df: pd.DataFrame = None,
    ) -> None:
        return
    

def splining_image(
    image: npt.NDArray,
    ordered_tracing_data: dict,
    pixel_to_nm_scaling: float,
    filename: str,
    spline_step_size: float,
  spline_linear_smoothing: float,
  spline_circular_smoothing: float,
  pad_width: int,
):
    return