"""Plotting scripts for plotting perovskite images"""
import numpy as np
import matplotlib.pyplot as plt


def plot(img, title="", savepath=None, figsize=(5, 5)):
    """Plot an image and optionally save it. Figsize can be set with 'figsize'."""

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    ax.set_title(title)
    if savepath is not None:
        fig.savefig(savepath)
    plt.show()


def plot_with_means(img: np.ndarray, title: str = ""):
    """Plot images with residuals for flattening quality control."""

    fig, ax = plt.subplots(1, 3, figsize=(10, 4))
    ax[0].imshow(img)
    ax[0].set_title("image")
    ax[1].plot(np.nanmedian(img, axis=0), ".")
    ax[1].set_title("axis: 0")
    ax[2].plot(np.nanmedian(img, axis=1), ".")
    ax[2].set_title("axis: 1")
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
