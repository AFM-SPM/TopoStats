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
# sns.set_context("notebook", font_scale=1.5)
sns.set_context("poster", font_scale=1.4)
# plt.style.use("dark_background")
sns.set_palette(sns.color_palette('bright'))
# sns.set_palette(sns.color_palette('BuPu'))
defextension = '.png'

colname2label = {
    'grain_bound_len': 'Circumference / %s',
    'aspectratio': 'Aspect Ratio',
    'grain_curvature1': 'Smaller Curvature',
    'grain_curvature2': 'Larger Curvature',
    'grain_ellipse_major': 'Ellipse Major Axis Length / %s',
    'grain_ellipse_minor': 'Ellipse Minor Axis Length / %s',
    'grain_half_height_area': 'Area Above Half Height / %s^2',
    'grain_maximum': 'Grain maximum / %s',
    'grain_mean': 'Grain mean / %s',
    'grain_median': 'Grain median / %s',
    'grain_min_bound_size': 'Width / %s',
    'grain_max_bound_size': 'Length / %s',
    'grain_mean_radius': 'Mean Radius / %s',
    'grain_pixel_area': 'Area / Pixels',
    'grain_proj_area': 'Area / %s^2'
}


def importfromjson(path):
    """Importing the data needed from the json file specified by the user"""

    print (path)
    importeddata = pd.read_json(path)

    return importeddata


def savestats(path, dataframetosave):
    print 'Saving stats for: ' + str(os.path.basename(path)[:-5]) + '_evaluated'

    dataframetosave.to_json(path[:-5] + '_evaluated.json')
    dataframetosave.to_csv(path[:-5] + '_evaluated.txt')


def pathman(path):
    """Splitting the path into directory and file name; creating or specifying a directory to save the plots in"""

    directory = os.path.dirname(path)
    name = os.path.basename(path)[:-5]
    savedir = os.path.join(directory, 'Plots')
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    plotname = os.path.join(savedir, name)
    return plotname


def labelunitconversion(plotarg, nm):
    """Adding units to the axis labels"""
    label = colname2label[plotarg]
    if '%s' in label:
        if nm is True:
            label = label % 'nm'
        else:
            label = label % 'm'
    return label


def dataunitconversion(data, plotarg, nm):
    """Converting the data based on the unit specified by the user. Only nm and m are supported at the moment."""
    label = colname2label[plotarg]
    if nm is True:
        if '%s' in label:
            if '^2' in label:
                data = data*1e18
            else:
                data = data*1e9
    return data


def plotkde(df, plotarg, grouparg=None, xmin=None, xmax=None, nm=False, specpath=None, plotextension=defextension):
    """Creating a KDE plot for the chose variable. Can be grouped. The x axis range can be defined by the user."""

    print 'Plotting kde of %s' % plotarg

    # Set the name of the file
    if specpath is None:
        specpath = path
    savename = os.path.join(pathman(specpath) + '_' + plotarg + '_KDE' + plotextension)

    # Convert the unit of the data to nm if specified by the user
    df[plotarg] = dataunitconversion(df[plotarg], plotarg, nm)

    # Plot figure
    fig, ax = plt.subplots(figsize=(15, 10))
    # Simple KDE plot
    if grouparg is None:
        df = df[plotarg]
        df.plot.kde(ax=ax, alpha=1, linewidth=7.0)
    # Grouped KDE plots
    else:
        df = df[[grouparg, plotarg]]
        df.groupby(grouparg)[plotarg].plot.kde(ax=ax, legend=True, alpha=1, linewidth=7.0)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(reversed(handles), reversed(labels), title=grouparg, loc='upper right')

    # Label plot and save figure
    plt.xlim(xmin, xmax)
    plt.xlabel(labelunitconversion(plotarg, nm), alpha=1)
    plt.ylabel('Probability Density', alpha=1)
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
    plt.savefig(savename)


def plothist(df, plotarg, grouparg=None, xmin=None, xmax=None, bins=20, nm=False, specpath=None, plotextension=defextension):
    print 'Plotting histogram of %s' % plotarg

    # Set the name of the file
    if specpath is None:
        specpath = path
    savename = os.path.join(pathman(specpath) + '_' + plotarg + '_histogram' + plotextension)

    # Convert the unit of the data to nm if specified by the user
    df[plotarg] = dataunitconversion(df[plotarg], plotarg, nm)

    # Plot figure
    fig, ax = plt.subplots(figsize=(15, 10))
    # Simple histogram
    if grouparg is None:
        df = df[plotarg]
        df.plot.hist(ax=ax, alpha=1, linewidth=7.0, bins=bins)
    # Grouped histogram
    else:
        df = df[[grouparg, plotarg]]
        df.groupby(grouparg)[plotarg].plot.hist(ax=ax, legend=True, alpha=1, linewidth=7.0, bins=bins)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(reversed(handles), reversed(labels), title=grouparg, loc='upper right')

    # Label plot and save figure
    plt.xlim(xmin, xmax)
    plt.xlabel(labelunitconversion(plotarg, nm), alpha=1)
    plt.ylabel('Count', alpha=1)
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
    plt.savefig(savename)


def plotviolin(df, plotarg, grouparg=None, ymin=None, ymax=None, nm=False, specpath=None, plotextension=defextension):
    print 'Plotting violin of %s' % plotarg

    # Set the name of the file
    if specpath is None:
        specpath = path
    savename = os.path.join(pathman(specpath) + '_' + plotarg + '_violin' + plotextension)

    # Plot and save figures
    df[plotarg] = dataunitconversion(df[plotarg], plotarg, nm)
    fig, ax = plt.subplots(figsize=(15, 10))
    # Single violin plot
    if grouparg is None:
        df = df[plotarg]
        ax = sns.violinplot(data=df)
    # Grouped violin plot
    else:
        df = df[[grouparg, plotarg]]
        ax = sns.violinplot(x=grouparg, y=plotarg, data=df)
        ax.invert_xaxis()  # Useful for topoisomers with negative writhe

    # Label plot and save figure
    plt.ylim(ymin, ymax)
    plt.ylabel(labelunitconversion(plotarg, nm), alpha=1)
    plt.xlabel(' ')
    plt.savefig(savename)


def plotjoint(df, arg1, arg2, xmin=None, xmax=None, ymin=None, ymax=None, nm=False, specpath=None, plotextension=defextension):
    print 'Plotting joint plot for %s and %s' % (arg1, arg2)

    # Set the name of the file
    if specpath is None:
        specpath = path
    savename = os.path.join(pathman(specpath) + '_' + arg1 + '_and_' + arg2 + plotextension)

    df[arg1] = dataunitconversion(df[arg1], arg1, nm)
    df[arg2] = dataunitconversion(df[arg2], arg2, nm)

    # Plot data using seaborn
    sns.jointplot(arg1, arg2, data=df, kind='reg', height=15)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel(labelunitconversion(arg1, nm), alpha=1)
    plt.ylabel(labelunitconversion(arg2, nm), alpha=1)
    plt.savefig(savename)


def plotLinearVsCircular(contour_lengths_df):
    pass


if __name__ == '__main__':
    # Path to the json file, e.g. C:\\Users\\username\\Documents\\Data\\Data.json
    path = 'C:\\Users\\dumin\\Documents\\PhD\\Data\\Kavit-Top1\\Non-incubation\\Non-incubation.json'

    # Set the name of the json file to import here
    # name = 'Non-incubation'
    bins = 50

    # import data form the json file specified as a dataframe
    df = importfromjson(path)
    # Rename directory column as appropriate
    df = df.rename(columns={"directory": "Experimental Conditions"})
    # Calculate the aspect ratio for each grain
    df['aspectratio'] = df['grain_min_bound_size'] / df['grain_max_bound_size']
    # Get list of unique directory names i.e. topoisomers
    # # topos = df['Proteins'].unique()
    # # topos = sorted(topos, reverse=False)

    # Convert original (rounded) delta Lk to correct delta Lk
    # # dfnew = df
    # # dfnew['Proteins'] = df['Proteins'].astype(str).replace({'-2': '-1.8', '-3': '-2.8', '-6': '-4.9'})
    # Get list of unique directory names i.e. topoisomers
    # # newtopos = dfnew['Proteins']
    # # newtopos = pd.to_numeric(newtopos, errors='ignore')
    # # dfnew['Proteins'] = newtopos

    # Obtain list of unique topoisomers
    # # topos = df['Proteins'].unique()
    # # topos = sorted(topos, reverse=False)

    # Get statistics for different topoisoimers
    # # allstats = df.groupby('Proteins').describe()
    # transpose allstats dataframe to get better saving output
    # # allstats1 = allstats.transpose()
    # Save out statistics file
    # # savestats(path, allstats1)
    # Set palette for all plots with length number of topoisomers and reverse
    # # palette = sns.color_palette('PuBu', n_colors=len(topos))

# Setting group argument
grouparg = 'Experimental Conditions'


# Plot one column of the dataframe e.g. 'grain_mean_radius'; grouparg can be specified for plotkde, plothist and
# plotviolin by entering e.g. 'xmin = 0'; xmin and xmax can be specified for plotkde, plothist, and plotjoint;
# ymin and ymax can be specified for plotviolin and plotjoint; bins can be speficied for plothist. The default unit is
# m; add "nm=True" to change from m to nm.

plotkde(df, 'grain_bound_len',  xmin=0, xmax=1e-7)
plotkde(df, 'grain_mean_radius')
# plotkde(df, 'grain_proj_area', nm=True)
# plotkde (df, 'aspectratio')
# plotkde(df, 'grain_min_bound_size', nm=True)
# plotkde(df, 'grain_max_bound_size', xmax=3.5e-8)
# plotkde(df, 'grain_half_height_area', grouparg=grouparg)
plothist(df, 'grain_min_bound_size', xmax=2.5e-8, bins=bins)
# plothist(df, 'grain_proj_area', xmax=3e-16)
# plothist(df, 'aspectratio')
# plotviolin(df, "grain_proj_area")
plotjoint(df, 'grain_bound_len', 'grain_mean_radius', xmax=200, ymax=20, nm=True)
