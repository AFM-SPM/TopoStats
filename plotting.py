from __future__ import unicode_literals
import os
import fnmatch
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import seaborn as sns
import numpy as np
import scipy
import glob
from scipy import stats
from cycler import cycler

# Set seaborn to override matplotlib for plot output
sns.set()
sns.set_style("white", {'font.family': ['sans-serif']})
# The four preset contexts, in order of relative size, are paper, notebook, talk, and poster.
# The notebook style is the default
sns.set_context("poster", font_scale=1.4)
sns.set_palette(sns.color_palette('BuPu'))
# plt.style.use("dark_background")

def plothist(json_path):
    pass
def plotLinearVsCircular(contour_lengths_df):
    pass
def plotkde(df, directory, name, plotextension, grouparg, plotarg):
    pass
def plotviolin(df, directory, name, plotextension, grouparg, plotarg):
    print 'Plotting violin of %s' % plotarg


if __name__ == '__main__':
    # Set the file path, i.e. the directory where the files are here'
    path = 'insert/path/here'

    # Set the name of the json file to import here
    name = '*.json'
    name = 'ExampleName'
    file_name = os.path.join(path, name)
    plotextension = '.pdf'
    bins = 30

    # Import data form the json file specified as a dataframe
    df = pd.read_json(file_name)

    # Create a saving name format/directory
    savedir = os.path.join(os.path.dirname(file_name), 'Plots')
    if not os.path.exists(savedir):
        os.makedirs(savedir)
