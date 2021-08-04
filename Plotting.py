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


def importfromjson(path, name):
    """Importing the data needed from a json file"""

    filename = os.path.join(path, name + '.json')
    print (filename)
    importeddata = pd.read_json(filename)

    return importeddata


def savestats(dataframetosave):
    print 'Saving stats for: ' + str(name) + '_evaluated'
    savename = os.path.join(path, name)

    dataframetosave.to_json(savename + '_evaluated.json')
    dataframetosave.to_csv(savename + '_evaluated.txt')

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


def plotkde(df, plotarg, grouparg=None, xmin=None, xmax=None, nm=False, plotextension=defextension):
    """Creating a KDE plot for the chose variable. Can be grouped. The x axis range can be defined by the user."""

    print 'Plotting kde of %s' % plotarg

    savename = os.path.join(savedir, name + plotarg + plotextension)
    df[plotarg] = dataunitconversion(df[plotarg], plotarg, nm)
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

    plt.xlim(xmin, xmax)
    plt.xlabel(labelunitconversion(plotarg, nm), alpha=1)
    plt.ylabel('Probability Density', alpha=1)
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
    plt.title('Distribution of ' + labelunitconversion(plotarg, nm))
    plt.savefig(savename)


def plothist(df, plotarg, grouparg=None, xmin=None, xmax=None, bins=20, plotextension=defextension):
    print 'Plotting histogram of %s' % plotarg

    savename = os.path.join(savedir, name + plotarg + "_histogram" + plotextension)
    # Plot and save figures
    fig, ax = plt.subplots(figsize=(15, 10))
    if grouparg is None:
        df = df[plotarg]
        df.plot.hist(ax=ax, alpha=1, linewidth=7.0, bins=bins)

    else:
        df = df[[grouparg, plotarg]]
        df.groupby(grouparg)[plotarg].plot.hist(ax=ax, legend=True, alpha=1, linewidth=7.0, bins=bins)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(reversed(handles), reversed(labels), title=grouparg, loc='upper right')

    plt.xlim(xmin, xmax)
    plt.xlabel(colname2label[plotarg], alpha=1)
    plt.ylabel('Count', alpha=1)
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
    plt.title('Distribution of ' + colname2label[plotarg])
    plt.savefig(savename)


def plotviolin(df, plotarg, grouparg=None, ymin=None, ymax=None, plotextension=defextension):
    print 'Plotting violin of %s' % plotarg

    # Plot and save figures
    savename = os.path.join(savedir, name + plotarg + '_violin' + plotextension)
    fig, ax = plt.subplots(figsize=(10, 7))
    # Plot violinplot
    if grouparg is None:
        df = df[plotarg]
        ax = sns.violinplot(data=df)

    else:
        df = df[[grouparg, plotarg]]
        ax = sns.violinplot(x=grouparg, y=plotarg, data=df)


    ax.invert_xaxis()
    plt.ylim(ymin, ymax)
    plt.ylabel(colname2label[plotarg], alpha=1)
    plt.xlabel(' ')
    plt.savefig(savename)


def plotjoint(df, arg1, arg2, xmin=None, xmax=None, ymin=None, ymax=None, plotextension=defextension):
    print 'Plotting joint plot for %s and %s' % (arg1, arg2)

    savename = os.path.join(savedir, name + arg1 + '_and_' + arg2 + plotextension)

    # Change from m to nm units for plotting
    # df[arg1] = df[arg1] * 1e9
    # df[arg2] = df[arg2] * 1e9

    # Plot data using seaborn
    sns.jointplot(arg1, arg2, data=df, kind='reg')
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel(colname2label[arg1], alpha=1)
    plt.ylabel(colname2label[arg2], alpha=1)
    plt.savefig(savename)


def plotLinearVsCircular(contour_lengths_df):
    pass


if __name__ == '__main__':
    # Set the file path, i.e. the directory where the files are here'
    path = 'C:\\Users\\dumin\\Documents\\PhD\\Data\\Kavit-Top1\\Non-incubation'

    # Set the name of the json file to import here
    name = 'Non-incubation'
    bins = 50

    # Set/create the directory to save the plots
    savedir = os.path.join(path, 'Plots')
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # import data form the json file specified as a dataframe
    df = importfromjson(path, name)
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
    # # savestats(allstats1)
    # Set palette for all plots with length number of topoisomers and reverse
    # # palette = sns.color_palette('PuBu', n_colors=len(topos))

# Setting group argument
grouparg = 'Experimental Conditions'


# Plot one column of the dataframe e.g. 'grain_mean_radius'; grouparg can be specified for plotkde, plothist and
# plotviolin by entering e.g. 'xmin = 0'; xmin and xmax can be specified for plotkde, plothist, and plotjoint;
# ymin and ymax can be specified for plotviolin and plotjoint; bins can be speficied for plothist

# plotkde(df, path, name, 'grain_bound_len',  xmin=0, xmax=1e-7)
# plotkde(df, 'grain_mean_radius')
plotkde(df, 'grain_proj_area', nm=True)
# plotkde (df, 'aspectratio')
plotkde(df, 'grain_min_bound_size', nm=True)
# plotkde(df, 'grain_max_bound_size', xmax=3.5e-8)
# plotkde(df, 'grain_half_height_area', grouparg=grouparg)
# plothist(df, 'grain_min_bound_size', xmax=2.5e-8, bins=bins)
# plothist(df, 'grain_proj_area', xmax=3e-16)
# plothist(df, 'aspectratio')
# plotviolin(df, "grain_proj_area", ymax=6e-16, grouparg=grouparg)
# plotjoint(df, 'grain_bound_len', 'grain_mean_radius')
