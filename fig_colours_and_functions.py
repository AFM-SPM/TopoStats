"""Colour definitions and functions for figures for the topostats 2 paper."""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import numpy.typing as npt
from scipy.ndimage import binary_dilation

plt.rcParams["font.family"] = "Arial"
good_pixels_colormap = "cool"
good_pixels_alpha = 0.8
good_pixels_colour = "#3792AF"
good_pixels_colour_alpha = 1.0
bad_pixels_colormap = "cool_r"
bad_pixels_alpha = 0.8
regular_pixels_colormap = "gray"
regular_pixels_alpha = 0.3
branch_underlying_colour = "#F972BF"
branch_overlying_colour = "#4EDB7A"
molecule_1_colour = "#82DEFF"
molecule_2_colour = "#0059FF"
good_mask_color = "#3792AF"
good_mask_alpha = 0.8


def overlay_mask(mask: npt.NDArray, colour: str) -> None:
    """Overlay a binary mask on the current figure using a specified colour."""
    # Convert colour to RGBA with specified alpha
    rgba = mcolors.to_rgba(colour, good_mask_alpha)
    # Create an RGBA image where the mask is applied
    colored_mask = np.zeros(mask.shape + (4,), dtype=np.float32)
    # Apply colour to the mask by multiplying each channel by the mask
    for i in range(3):  # RGB channels
        colored_mask[..., i] = mask * rgba[i]
    # Apply alpha channel
    colored_mask[..., 3] = mask * rgba[3]
    plt.imshow(colored_mask)


def dilate_branches_in_mask(
    branch_index_mask: npt.NDArray,
):
    """Dilate a branch mask where the branches are dilated for visibility."""
    branch_mask_dilated = np.zeros_like(branch_index_mask)
    for branch_index in np.unique(branch_index_mask):
        branch_mask = branch_index_mask == branch_index
        dilated_branch_mask = binary_dilation(branch_mask, iterations=1)
        branch_mask_dilated[dilated_branch_mask] = branch_index
    return branch_mask_dilated


def plot_overlay_dilated_branches_in_mask(branch_index_mask: npt.NDArray, colourmap: str | None = None, alpha=1):
    """Plot a branch mask where the branches are dilated for visibility."""
    branch_mask_dilated = dilate_branches_in_mask(branch_index_mask)
    plt.imshow(np.ma.masked_where(branch_mask_dilated == 0, branch_mask_dilated), cmap=colourmap, alpha=alpha)


def overlay_mask_with_specific_colour(mask: npt.NDArray, colour: str, alpha: float) -> None:
    """Overlay a binary mask on the current figure using a specified colour."""
    # Convert colour to RGBA with specified alpha
    rgba = mcolors.to_rgba(colour, alpha)
    # Create an RGBA image where the mask is applied
    colored_mask = np.zeros(mask.shape + (4,), dtype=np.float32)
    # Apply colour to the mask by multiplying each channel by the mask
    for i in range(3):  # RGB channels
        colored_mask[..., i] = mask * rgba[i]
    # Apply alpha channel
    colored_mask[..., 3] = mask * rgba[3]
    plt.imshow(colored_mask)


def plot_overlay_dilated_branches_in_mask_specific_colours(
    branch_index_mask: npt.NDArray, colours: dict[int, str], alpha: float = 1
) -> None:
    """Plot a branch mask where the branches are dilated for visibility, using specific colours for each branch."""
    # check that the number of colours matches the number of unique branches
    unique_branches = np.unique(branch_index_mask)
    # drop the 0 from the unique branches
    unique_branches = unique_branches[unique_branches != 0]
    print(f"unique branches: {unique_branches}")
    assert len(colours) == len(unique_branches), "Number of colours must match number of unique branches"
    branch_mask_dilated = dilate_branches_in_mask(branch_index_mask)
    for branch_index in unique_branches:
        branch_mask = branch_mask_dilated == branch_index
        overlay_mask_with_specific_colour(mask=branch_mask, colour=colours[branch_index], alpha=alpha)
