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

# Set seaborn to override matplotlib for plot output
sns.set()
sns.set_style("white", {'font.family': ['sans-serif']})
# The four preset contexts, in order of relative size, are paper, notebook, talk, and poster.
# The notebook style is the default
# sns.set_context("notebook", font_scale=1.5)
sns.set_context("poster", font_scale=1.4)
# plt.style.use("dark_background")
sns.set_palette(sns.color_palette('bright'))
defextension = '.png'

colname2label = {
    'grain_bound_len': 'Circumference / m',
    'aspectratio': 'Aspect Ratio',
    'grain_curvature1': 'Smaller Curvature',
    'grain_curvature2': 'Larger Curvature',
    'grain_ellipse_major': 'Ellipse Major Axis Length',  # / m',
    'grain_ellipse_minor': 'Ellipse Minor Axis Length',  # / m',
    'grain_half_height_area': 'Area Above Half Height',
    'grain_maximum': 'Grain maximum',
    'grain_mean': 'Grain mean',
    'grain_median': 'Grain median',
    'grain_min_bound_size': 'Width / m',
    'grain_max_bound_size': 'Length / m',
    'grain_mean_radius': 'Mean Radius / m',
    'grain_pixel_area': 'Area / Pixels',
    'grain_proj_area': 'Area / m^2'
}


def importfromjson(path, name):
    filename = os.path.join(path, name + '.json')
    print (filename)
    importeddata = pd.read_json(filename)

    return importeddata


def savestats(directory, name, dataframetosave):
    directoryname = os.path.splitext(os.path.basename(directory))[0]
    print 'Saving stats for: ' + str(name) + '_evaluated'

    savedir = os.path.join(directory)
    savename = os.path.join(savedir, directoryname)
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    dataframetosave.to_json(savename + '_evaluated.json')
    dataframetosave.to_csv(savename + '_evaluated.txt')


def plotkde(df, directory, name, plotarg, grouparg=None, xmin=None, xmax=None, plotextension=defextension):
    print 'Plotting kde of %s' % plotarg
    # Create a saving name format/directory
    savedir = os.path.join(directory, 'Plots')
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    savename = os.path.join(savedir, name + plotarg + plotextension)
    # Plot and save figures
    fig, ax = plt.subplots(figsize=(15, 10))
    if grouparg is None:
        df = df[plotarg]
        df.plot.kde(ax=ax, alpha=1, linewidth=7.0)
        if xmin is None:
            xmin = df.quantile(.05)
        if xmax is None:
            xmax = df.quantile(.95)
    else:
        df = df[[grouparg, plotarg]]
        df.groupby(grouparg)[plotarg].plot.kde(ax=ax, legend=True, alpha=1, linewidth=7.0)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(reversed(handles), reversed(labels), title=grouparg, loc='upper right')
        if xmin is None:
            xmin = df[plotarg].quantile(.05)
        if xmax is None:
            xmax = df[plotarg].quantile(.95)

    plt.xlim(xmin, xmax)
    plt.xlabel(colname2label[plotarg], alpha=1)
    plt.ylabel('Probability Density', alpha=1)
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
    plt.title('Distribution of ' + colname2label[plotarg])
    plt.savefig(savename)


def plothist(df, directory, name, plotarg, grouparg=None, xmin=None, xmax=None, bins=20, plotextension=defextension):
    print 'Plotting histogram of %s' % plotarg
    savedir = os.path.join(directory, 'Plots')
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    savename = os.path.join(savedir, name + plotarg + "_histogram" + plotextension)
    # Plot and save figures
    fig, ax = plt.subplots(figsize=(15, 10))
    if grouparg is None:
        df = df[plotarg]
        df.plot.hist(ax=ax, alpha=1, linewidth=7.0, bins=bins)
        if xmin is None:
            xmin = df.quantile(.05)
        if xmax is None:
            xmax = df.quantile(.95)
    else:
        df = df[[grouparg, plotarg]]
        df.groupby(grouparg)[plotarg].plot.hist(ax=ax, legend=True, alpha=1, linewidth=7.0, bins=bins)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(reversed(handles), reversed(labels), title=grouparg, loc='upper right')
        if xmin is None:
            xmin = df[plotarg].quantile(.05)
        if xmax is None:
            xmax = df[plotarg].quantile(.95)

    # plt.xlim(xmin, xmax)
    plt.xlabel(colname2label[plotarg], alpha=1)
    plt.ylabel('Count', alpha=1)
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
    plt.title('Distribution of ' + colname2label[plotarg])
    plt.savefig(savename)


def plotviolin(df, directory, name, plotarg, grouparg=None, ymin=None, ymax=None, plotextension=defextension):
    print 'Plotting violin of %s' % plotarg

    # Create a saving name format/directory
    savedir = os.path.join(directory, 'Plots')
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # Plot and save figures
    savename = os.path.join(savedir, name + plotarg + '_violin' + plotextension)
    fig, ax = plt.subplots(figsize=(10, 7))
    # Plot violinplot
    if grouparg is None:
        df = df[plotarg]
        ax = sns.violinplot(data=df)
        if ymin is None:
            ymin = df.quantile(.05)
        if ymax is None:
            ymax = df.quantile(.95)
    else:
        df = df[[grouparg, plotarg]]
        ax = sns.violinplot(x=grouparg, y=plotarg, data=df)
        if ymin is None:
            ymin = df[plotarg].quantile(.05)
        if ymax is None:
            ymax = df[plotarg].quantile(.95)
    ax.invert_xaxis()
    plt.ylim(ymin, ymax)
    plt.ylabel(colname2label[plotarg], alpha=1)
    plt.xlabel(' ')
    plt.savefig(savename)


def plotjoint(df, directory, arg1, arg2, xmin=None, xmax=None, ymin=None, ymax=None, plotextension=defextension):
    print 'Plotting joint plot for %s and %s' % (arg1, arg2)

    # Create a saving name format/directory
    savedir = os.path.join(directory, 'Plots')
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    savename = os.path.join(savedir, name + arg1 + '_and_' + arg2 + plotextension)

    # Change from m to nm units for plotting
    # df[arg1] = df[arg1] * 1e9
    # df[arg2] = df[arg2] * 1e9
    if xmin is None:
        xmin = df[arg1].quantile(.05)
    if xmax is None:
        xmax = df[arg1].quantile(.95)
    if ymin is None:
        ymin = df[arg2].quantile(.05)
    if ymax is None:
        ymax = df[arg2].quantile(.95)

    # Plot data using seaborn
    sns.jointplot(arg1, arg2, data=df, kind='reg')
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel(colname2label[arg1], alpha=1)
    plt.ylabel(colname2label[arg2], alpha=1)
    plt.savefig(savename)


if __name__ == '__main__':
    # Set the file path, i.e. the directory where the files are here'
    path = 'C:\\Users\\dumin\\Documents\\PhD\\Data\\Kavit-Top1\\Non-incubation'

    # Set the name of the json file to import here
    name = 'Non-incubation'
    bins = 50

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
    # # savestats(path, name, allstats1)
    # Set palette for all plots with length number of topoisomers and reverse
    # # palette = sns.color_palette('PuBu', n_colors=len(topos))

# Plot a KDE plot of one column of the dataframe - arg1 e.g. 'aspectratio'
grouparg = 'Experimental Conditions'

# plotkde(df, path, name, 'grain_bound_len',  xmin=0, xmax=1e-7)
# plotkde(df, path, name, 'grain_mean_radius')
# plotkde(df, path, name, 'grain_proj_area', grouparg=grouparg)
# plothist(df, path, name, 'aspectratio')
# plotkde(df, path, name, 'grain_min_bound_size', xmax=3e-8)
# plotkde(df, path, name, 'grain_max_bound_size', xmax=3.5e-8)
# plotkde(df, path, name, 'grain_half_height_area', xmax=1.0e-16, grouparg=grouparg)
# plothist(df, path, name, 'grain_min_bound_size', xmax=2.5e-8, bins=bins)
# plothist(df, path, name, 'grain_proj_area', xmax=3e-16)
# plotviolin(df, path, name, "grain_proj_area", ymax=6e-16, grouparg=grouparg)
plotjoint(df, path, 'grain_bound_len', 'grain_mean_radius')
