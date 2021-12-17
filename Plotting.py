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
defextension = '.png'

colname2label = {
    'grain_bound_len': 'Circumference / %s',
    'aspectratio': 'Aspect Ratio',
    'grain_curvature1': 'Smaller Curvature',
    'grain_curvature2': 'Larger Curvature',
    'grain_ellipse_major': 'Ellipse Major Axis Length / %s',
    'grain_ellipse_minor': 'Ellipse Minor Axis Length / %s',
    'grain_half_height_area': 'Area Above Half Height / $\mathregular{%s^2}$',
    'grain_maximum': 'Maximum Height / %s',
    'grain_mean': 'Mean Height / %s',
    'grain_median': 'Median Height / %s',
    'grain_min_bound_size': 'Width / %s',
    'grain_max_bound_size': 'Length / %s',
    'grain_mean_radius': 'Mean Radius / %s',
    'grain_pixel_area': 'Area / Pixels',
    'grain_proj_area': 'Area / $\mathregular{%s^2}$',
    'grain_min_volume': 'Minimum Volume / $\mathregular{%s^3}$',
    'grain_zero_volume': 'Zero Volume / $\mathregular{%s^3}$',
    'grain_laplace_volume': 'Laplacian Volume / $\mathregular{%s^3}$',
    'End to End Distance': 'End to End Distance / nm',
    'Contour Lengths': 'Contour Lengths / nm',
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
    """Splitting the path into directory and file name; creating or specifying a directory to save the plots"""

    directory = os.path.dirname(path)
    name = os.path.basename(path)[:-5]
    savedir = os.path.join(directory, 'Plots')
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    plotname = os.path.join(savedir, name)
    return plotname


def labelunitconversion(plotarg, nm):
    """Adding units (m or nm) to the axis labels"""

    if plotarg in colname2label:
        label = colname2label[plotarg]
    else:
        label = plotarg

    if '%s' in label:
        if nm is True:
            label = label % 'nm'
        else:
            label = label % 'm'
    elif '/ nm' in label:
        if nm is False:
            label = label[:-2] + 'm'

    return label


def dataunitconversion(data, plotarg, nm):
    """Converting the data based on the unit specified by the user. Only nm and m are supported at the moment."""

    if plotarg in colname2label:
        label = colname2label[plotarg]
    else:
        label = plotarg

    data_new = data
    if nm is True:
        if '%s' in label:
            if '^2' in label:
                data_new = data * 1e18
            elif '^3' in label:
                data_new = data * 1e27
            else:
                data_new = data * 1e9
    else:
        if '/ nm' in label:
            data_new = data * 1e-9

    return data_new


def plotkde(df, plotarg, grouparg=None, xmin=None, xmax=None, nm=False, specpath=None, plotextension=defextension):
    """Creating a KDE plot for the chosen variable. Grouping optional. The x axis range can be defined by the user. The
    default unit is metre, but this can be changed to nanometre by adding 'nm=True'. The default path is the path under
    the if __name__ == '__main__' line, but this can also be changed using the specpath argument."""

    print 'Plotting kde of %s' % plotarg

    # Set the name of the file
    if specpath is None:
        specpath = path
    savename = os.path.join(pathman(specpath) + '_' + plotarg + '_KDE' + plotextension)

    # Convert the unit of the data to nm if specified by the user
    df[plotarg] = dataunitconversion(df[plotarg], plotarg, nm)

    # Plot figure
    fig, ax = plt.subplots(figsize=(15, 12))
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


def plotkde2var(df, plotarg1, plotarg2, xmin=None, xmax=None, nm=False, specpath=None, plotextension=defextension):
    """Creating a KDE plot for the chosen variable. Grouping optional. The x axis range can be defined by the user. The
    default unit is metre, but this can be changed to nanometre by adding 'nm=True'. The default path is the path under
    the if __name__ == '__main__' line, but this can also be changed using the specpath argument."""

    print 'Plotting kde of %s and %s' % (plotarg1, plotarg2)

    # Set the name of the file
    if specpath is None:
        specpath = path
    savename = os.path.join(pathman(specpath) + '_' + plotarg1 + '_' + plotarg2 + '_KDE' + plotextension)

    # Convert the unit of the data to nm if specified by the user
    df[plotarg1] = dataunitconversion(df[plotarg1], plotarg1, nm)
    df[plotarg2] = dataunitconversion(df[plotarg2], plotarg2, nm)

    # Plot figure
    fig, ax = plt.subplots(figsize=(15, 12))
    # Simple KDE plot

    df = df[[plotarg1, plotarg2]]
    df.plot.kde(ax=ax, alpha=1, linewidth=7.0)

    # Label plot and save figure
    plt.xlim(xmin, xmax)
    # plt.xlabel(labelunitconversion(plotarg1, nm), alpha=1)
    plt.xlabel('Bound Size / nm', alpha=1)
    plt.ylabel('Probability Density', alpha=1)
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
    plt.savefig(savename)


def plothist(df, plotarg, grouparg=None, xmin=None, xmax=None, bins=20, nm=False, specpath=None,
             plotextension=defextension):
    """Creating a histogram for the chosen variable. Grouping optional. The x axis range can be defined by the user. The
    default unit is metre, but this can be changed to nanometre by adding 'nm=True'. The default path is the path under
    the if __name__ == '__main__' line, but this can also be changed using the specpath argument."""

    print 'Plotting histogram of %s' % plotarg

    # Set  the name of the file
    if specpath is None:
        specpath = path
    savename = os.path.join(pathman(specpath) + '_' + plotarg + '_histogram' + plotextension)

    # Convert the unit of the data to nm if specified by the user
    df[plotarg] = dataunitconversion(df[plotarg], plotarg, nm)

    # Plot figure
    fig, ax = plt.subplots(figsize=(15, 12))
    # Simple histogram
    if grouparg is None:
        df = df[plotarg]
        df.plot.hist(ax=ax, alpha=1, linewidth=3.0, bins=bins)
    # Grouped histogram
    else:
        df = df[[grouparg, plotarg]]
        df = df.pivot(columns=grouparg, values=plotarg)
        df.plot.hist(ax=ax, legend=True, bins=bins, alpha=1, linewidth=3.0, stacked=True)
        # df.groupby(grouparg)[plotarg].plot.hist(ax=ax, legend=True, alpha=1, linewidth=7.0, bins=bins, stacked=True)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(reversed(handles), reversed(labels), title=grouparg, loc='upper right')

    # Label plot and save figure
    plt.xlim(xmin, xmax)
    plt.xlabel(labelunitconversion(plotarg, nm), alpha=1)
    plt.ylabel('Count', alpha=1)
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
    plt.savefig(savename)


def plotdist(df, xmin=None, xmax=None, bins=20, nm=False, specpath=None,
             plotextension=defextension, *plotargs):
    """An attempt to make dist plots for a customisable number of arguments, but this doesn't work yet"""


    # # Set  the name of the file
    # if specpath is None:
    #     specpath = path
    # savename = os.path.join(pathman(specpath) + '_dist' + plotextension)
    #
    # # Convert the unit of the data to nm if specified by the user
    # for plotarg in plotargs:
    #     df[plotarg] = dataunitconversion(df[plotarg], plotarg, nm)
    #
    # # Plot figure
    # fig, ax = plt.subplots(figsize=(15, 12))
    # # Simple dist plot
    #
    # for plotarg in plotargs:
    #     sns.distplot(df[plotarg], ax=ax, bins=bins)
    #
    # # Label plot and save figure
    # plt.xlim(xmin, xmax)
    # # plt.xlabel(labelunitconversion(plotarg, nm), alpha=1)
    # plt.ylabel('Probability Density', alpha=1)
    # # plt.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
    # plt.savefig(savename)
    pass



def plotdist2var(plotarg1, plotarg2, df1, df2=None, xmin=None, xmax=None, bins=20, nm=False,
                 specpath=None,
                 plotextension=defextension, plotname=None, c1=None, c2=None):

    """Dist plot for 2 variables"""

    print 'Plotting dist plot of %s and %s' % (plotarg1, plotarg2)

    if plotname is None:
        plotname = plotarg1 + '_and_' + plotarg2

    # Set  the name of the file
    if specpath is None:
        specpath = path
    savename = os.path.join(pathman(specpath) + '_' + plotname + '_dist' + plotextension)

    if df2 is None:
        df2 = df1

    # Convert the unit of the data to nm if specified by the user

    dfa = dataunitconversion(df1[plotarg1], plotarg1, nm)
    dfb = dataunitconversion(df2[plotarg2], plotarg2, nm)

    # Plot figure
    fig, ax = plt.subplots(figsize=(15, 12))
    # Simple dist plot

    sns.distplot(dfa, ax=ax, bins=bins, color=c1)
    sns.distplot(dfb, ax=ax, bins=bins, color=c2)

    # Label plot and save figure
    plt.xlim(xmin, xmax)
    plt.xlabel(plotname)
    # plt.xlabel(labelunitconversion(plotarg, nm), alpha=1)
    plt.ylabel('Probability Density', alpha=1)
    plt.ticklabel_format(axis='both', style='sci', scilimits=(-2, 3))
    ax.tick_params(direction='out', bottom=True, left=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.savefig(savename)


def plotviolin(df, plotarg, grouparg=None, ymin=None, ymax=None, nm=False, specpath=None, plotextension=defextension):
    """Creating a violin plot for the chosen variable. Grouping optional. The y axis range can be defined by the user.
    The default unit is metre, but this can be changed to nanometre by adding 'nm=True'. The default path is the path
    under the if __name__ == '__main__' line, but this can also be changed using the specpath argument."""

    print 'Plotting violin of %s' % plotarg

    # Set the name of the file
    if specpath is None:
        specpath = path
    savename = os.path.join(pathman(specpath) + '_' + plotarg + '_violin' + plotextension)

    # Plot and save figures
    df[plotarg] = dataunitconversion(df[plotarg], plotarg, nm)
    fig, ax = plt.subplots(figsize=(15, 12))
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
    plt.xlabel(grouparg)
    plt.savefig(savename)


def plotjoint(df, arg1, arg2, xmin=None, xmax=None, ymin=None, ymax=None, nm=False, specpath=None,
              plotextension=defextension):
    """Creating a joint plot for two chosen variables. The range for both axes can be defined by the user.
    The default unit is metre, but this can be changed to nanometre by adding 'nm=True'. The default path is the path
    under the if __name__ == '__main__' line, but this can also be changed using the specpath argument."""

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


def computeStats(data, columns, min, max):

    """Prints out a table of stats, including the standard deviation, standard error, N value, and peak position"""

    xs = np.linspace(min, max, 1000)

    a = {}
    b = {}
    table = {
        'max': [0] * len(data),
        'std': [0] * len(data),
        'ste': [0] * len(data),
        'N': [0] * len(data),
    }

    for i, x in enumerate(data):
        x = x * 1e9
        a[i] = scipy.stats.gaussian_kde(x)
        b[i] = a[i].pdf(xs)
        table['std'][i] = np.std(x)
        table['ste'][i] = stats.sem(x)
        table['max'][i] = xs[np.argmax(b[i])]
        table['N'][i] = len(x)

    dfmax = pd.DataFrame.from_dict(table, orient='index', columns=columns)
    dfmax.to_csv(pathman(path) + '.csv')


if __name__ == '__main__':
    # Path to the json file, e.g. C:\\Users\\username\\Documents\\Data\\Data.json

    path = ''
    path2 = ''

    # Set the name of the json file to import here
    # name = 'Non-incubation'
    bins = 50

    # import data form the json file specified as a dataframe
    df = importfromjson(path)
    df2 = importfromjson(path2)

    # * df = df[df['End to End Distance'] != 0]
    # * df = df[df['Contour Lengths'] > 100]
    # * df = df[df['Contour Lengths'] < 120]

    # Rename directory column as appropriate
    # df = df.rename(columns={"directory": "Experimental Conditions"})
    # df1 = df1.rename(columns={"directory": "Experimental Conditions"})
    # df2 = df2.rename(columns={"directory": "Experimental Conditions"})
    # df = df.rename(columns={"grain_min_bound_size": "Minimum Bound Size"})
    # df = df.rename(columns={"directory": "Experimental Conditions"})

    # Calculate the aspect ratio for each grain
    # df['aspectratio'] = df['grain_min_bound_size'] / df['grain_max_bound_size']
    # df1['aspectratio'] = df1['grain_min_bound_size'] / df1['grain_max_bound_size']
    # df2['aspectratio'] = df2['grain_min_bound_size'] / df2['grain_max_bound_size']
    # df['Basename'] = os.path.basename(df['Experiment Directory'])

    # df = pd.concat([df1, df2])

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
# grouparg = 'Experiment Directory'
# grouparg = 'Mask'
# * grouparg = 'Basename'
# $ grouparg = 'directory'

# Setting a continuous colour palette; useful for certain grouped plots, but can be commented out if unsuitable.
# sns.set_palette(sns.color_palette('BuPu', n_colors=len(df.groupby(grouparg))))
# print df.pivot(columns=grouparg, values='grain_median')


# Plot one column of the dataframe e.g. 'grain_mean_radius'; grouparg can be specified for plotkde, plothist and
# plotviolin by entering e.g. 'xmin = 0'; xmin and xmax can be specified for plotkde, plothist, and plotjoint;
# ymin and ymax can be specified for plotviolin and plotjoint; bins can be speficied for plothist. The default unit is
# m; add "nm=True" to change from m to nm.


# plotkde(df, 'grain_maximum', xmin=0, xmax=5, nm=True)
# plothist(df, 'grain_maximum', nm=True)
# plotkde(df, 'grain_max_bound_size', xmin=0, xmax=50, nm=True)

# plotkde(df, 'grain_min_bound_size', xmin=0, xmax=40, nm=True)
# plotkde2var(df, 'grain_min_bound_size', 'grain_max_bound_size', xmin=0, xmax=40, nm=True)



plotdist2var('grain_min_bound_size', 'grain_max_bound_size', df2, plotname='Bound Size for Full Protein', nm=True,
             xmin=0, xmax=50, c1='#f9cb9c', c2='#b45f06', plotextension='.tiff')
plotdist2var('grain_min_bound_size', 'grain_max_bound_size', df, df2=df, plotname='Bound Size for CTD', nm=True, xmin=0,
             xmax=50, c1='#9fc5e8', c2='#0b5394', plotextension='.tiff')
plotdist2var('grain_min_bound_size', 'grain_min_bound_size', df, df2=df2,
             plotname='Minimum Bound Size for CTD and Full Protein', nm=True, xmin=0, xmax=50, c1='#9fc5e8',
             c2='#f9cb9c', plotextension='.tiff')
plotdist2var('grain_max_bound_size', 'grain_max_bound_size', df, df2=df2,
             plotname='Maximum Bound Size for CTD and Full Protein', nm=True, xmin=0, xmax=50, c1='#0b5394',
             c2='#b45f06', plotextension='.tiff')
plotdist2var('grain_max_bound_size', 'grain_min_bound_size', df, df2=df2,
             plotname='Mininmum Bound Size for Full Protein and Maximum Bound Size for CTD', nm=True, xmin=0, xmax=50,
             c1='#0b5394', c2='#f9cb9c', plotextension='.tiff')




data = [df['grain_min_bound_size'], df['grain_max_bound_size'], df2['grain_min_bound_size'],
            df2['grain_max_bound_size']]
columns = ['Min for CTD / nm', 'Max for CTD / nm',
               'Min for FL / nm', 'Max for FL / nm']

computeStats(data, columns, 0, 50)













# * plotkde(df, 'End to End Distance', xmin=0, xmax=1.5e2, grouparg=grouparg)
# * plotkde(df, 'Contour Lengths', xmin=0, xmax=250, grouparg=grouparg)
# * plothist(df, 'End to End Distance', bins=20, xmin=0, xmax=2e2, grouparg=grouparg)
# * plothist(df, 'Contour Lengths', bins=20, xmin=0, xmax=250, grouparg=grouparg)
# plotkde (df, 'aspectratio')

# plotkde2var(df, 'grain_min_bound_size', 'grain_max_bound_size', nm=True)
# plotkde(df, 'grain_max_bound_size', xmax=3.5e-8)
# plotkde(df, 'grain_half_height_area', xmin=0, xmax=1.5e2, specpath=path, grouparg=grouparg, nm=True)

# plotkde(df, 'grain_maximum', specpath=path, xmin=0, xmax=1e-8, grouparg=grouparg)
# plotkde(df, 'grain_mean_radius', specpath=path, xmin=0, xmax=1.5e-8, grouparg=grouparg)
# plotkde(df, 'grain_bound_len', specpath=path, xmin=-0.25e-7, xmax=1.25e-7, grouparg=grouparg)

# plotkde(df, 'grain_min_volume', specpath=path, xmin=-1e-25, xmax=2.5e-25, grouparg=grouparg)
# plotkde(df, 'grain_zero_volume', specpath=path, xmin=-1e-25, xmax=5e-25, grouparg=grouparg)
# plotkde(df, 'grain_laplace_volume', specpath=path, xmin=-1e-25, xmax=2.5e-25, grouparg=grouparg)
# plothist(df, 'grain_min_bound_size', xmax=2.5e-8, bins=bins)
# plothist(df, 'grain_proj_area', xmax=3e-16)
# plothist(df, 'grain_half_height_area', grouparg=grouparg)
# plotviolin(df, "grain_proj_area", grouparg=grouparg)
# plotjoint(df, 'grain_bound_len', 'grain_mean_radius', xmax=200, ymax=20, nm=True)
