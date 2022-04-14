from configparser import Interpolation
import matplotlib.pyplot as plt
import numpy as np

def plot_and_save(data, filename):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    if isinstance(data, np.ndarray):
        ax.imshow(data, interpolation='nearest', cmap='afmhot')
        plt.savefig(filename)
    else:
        data.show(ax=ax, interpolation='nearest', cmap='afmhot')
    plt.close()
