from pathlib import Path
from typing import Union
import logging
from configparser import Interpolation
import matplotlib.pyplot as plt
import numpy as np


def plot_and_save(data: np.array,
                  filename: Union[str, Path],
                  title: str,
                  cmap: str = 'afmhot'):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    if isinstance(data, np.ndarray):
        ax.imshow(data, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.savefig(filename)
    else:
        data.show(ax=ax, interpolation='nearest', cmap=cmap)
    plt.close()
    logging.info(f'Image saved to : {filename}')
    return fig, ax
