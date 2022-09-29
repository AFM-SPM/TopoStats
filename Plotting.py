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
import scipy.signal as sig
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
    'Max Curvature': 'Maximum Curvature',
    'Max Curvature Location': 'Maximum Curvature Location / nm',
    'Mean Curvature': 'Mean Curvature / $\ nm^{-1}$',
}


def importfromfile(path):
    """Importing the data needed from the json file specified by the user"""

    print (path)
    filename, filextension = os.path.splitext(path)
    if filextension == '.json':
        importeddata = pd.read_json(path)
    elif filextension == '.csv':
        importeddata = pd.read_csv(path)

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
    dfnew = df.copy()
    dfnew[plotarg] = dataunitconversion(df[plotarg], plotarg, nm)

    # Plot figure
    fig, ax = plt.subplots(figsize=(15, 12))
    # Simple KDE plot
    if grouparg is None:
        dfnew[plotarg].plot.kde(ax=ax, alpha=1, linewidth=7.0)

    # Grouped KDE plots
    else:
        dfnew = dfnew[[grouparg, plotarg]]
        dfnew = dfnew.pivot(columns=grouparg, values=plotarg)
        dfnew.plot.kde(ax=ax, legend=True, alpha=0.7, linewidth=7.0, color=['#B45F06', '#0B5394'])
        handles, labels = ax.get_legend_handles_labels()
        # ax.legend(reversed(handles), reversed(labels), title=grouparg, loc='upper right')
        ax.legend(handles, labels, title=grouparg, loc='upper right')

    # Label plot and save figure
    plt.xlim(xmin, xmax)
    plt.xlabel(labelunitconversion(plotarg, nm), alpha=1)
    plt.ylabel('Probability Density', alpha=1)
    plt.ticklabel_format(axis='both', style='sci', scilimits=(-3, 3))
    plt.savefig(savename)


def plotkde2var(df, plotarg, df2=None, plotarg2=None, label1=None, label2=None, xmin=None, xmax=None, nm=False,
                specpath=None, plotextension=defextension):
    """Creating a KDE plot for the chosen variable. Grouping optional. The x axis range can be defined by the user. The
    default unit is metre, but this can be changed to nanometre by adding 'nm=True'. The default path is the path under
    the if __name__ == '__main__' line, but this can also be changed using the specpath argument."""

    if df2 is None:
        df2 = df
    if plotarg2 is None:
        plotarg2 = plotarg
    if label1 is None:
        label1 = plotarg
    if label2 is None:
        label2 = plotarg2

    print 'Plotting KDE of %s for %s and %s' % (plotarg, label1, label2)

    # Set the name of the file
    if specpath is None:
        specpath = path
    savename = os.path.join(pathman(specpath) + '_' + plotarg + '_' + label1 + '_' + label2 + '_KDE' + plotextension)

    # Convert the unit of the data to nm if specified by the user
    dfnew = df.copy()
    dfnew2 = df2.copy()
    dfnew[plotarg] = dataunitconversion(dfnew[plotarg], plotarg, nm)
    dfnew2[plotarg2] = dataunitconversion(dfnew2[plotarg2], plotarg2, nm)
    dfplot = pd.concat([dfnew[plotarg], dfnew2[plotarg2]], axis=1)
    dfplot.columns = [label1, label2]

    # Plot figure
    fig, ax = plt.subplots(figsize=(15, 12))
    # dfnew[plotarg].plot.hist(ax=ax, alpha=1, linewidth=3.0, bins=bins, color='orange')
    # dfnew2[plotarg2].plot.hist(ax=ax, alpha=1, linewidth=3.0, bins=bins, histtype='barstacked', color='blue')
    dfplot.plot.kde(ax=ax, alpha=1, linewidth=7.0, legend=False)

    # Label plot and save figure
    plt.xlim(xmin, xmax)
    plt.xlabel(labelunitconversion(plotarg, nm), alpha=1)
    plt.ylabel('Probability Density', alpha=1)
    plt.ticklabel_format(axis='both', style='sci', scilimits=(-3, 3))
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
    dfnew = df.copy()
    dfnew[plotarg] = dataunitconversion(df[plotarg], plotarg, nm)

    # Plot figure
    fig, ax = plt.subplots(figsize=(15, 12))
    # Simple histogram
    if grouparg is None:
        dfnew[plotarg].plot.hist(ax=ax, alpha=1, linewidth=3.0, bins=bins)
    # Grouped histogram
    else:
        dfnew = dfnew[[grouparg, plotarg]]
        dfnew = dfnew.pivot(columns=grouparg, values=plotarg)
        dfnew.plot.hist(ax=ax, legend=True, bins=bins, alpha=0.7, linewidth=3.0, stacked=False, density=True,
                        color=['#B45F06', '#0B5394'])
        handles, labels = ax.get_legend_handles_labels()
        # ax.get_legend().remove()
        # ax.legend(reversed(handles), reversed(labels), title=grouparg, loc='upper right')
        # ax.legend(handles, labels, title=grouparg, loc='upper right')

    # Label plot and save figure
    plt.xlim(xmin, xmax)
    plt.xlabel(labelunitconversion(plotarg, nm), alpha=1)
    plt.ylabel('Probability Density', alpha=1)
    plt.ticklabel_format(axis='both', style='sci', scilimits=(-3, 3))
    ax.tick_params(direction='out', bottom=True, left=True)
    plt.savefig(savename)


def plothist2var(df, plotarg, df2=None, plotarg2=None, label1=None, label2=None, xmin=None, xmax=None, nm=False,
                 specpath=None, plotextension=defextension):
    """Creating a histogram for the chosen variable. Grouping optional. The x axis range can be defined by the user. The
    default unit is metre, but this can be changed to nanometre by adding 'nm=True'. The default path is the path under
    the if __name__ == '__main__' line, but this can also be changed using the specpath argument."""

    if df2 is None:
        df2 = df
    if plotarg2 is None:
        plotarg2 = plotarg
    if label1 is None:
        label1 = plotarg
    if label2 is None:
        label2 = plotarg2

    print 'Plotting histogram of %s for %s and %s' % (plotarg, label1, label2)

    # Set the name of the file
    if specpath is None:
        specpath = path
    savename = os.path.join(
        pathman(specpath) + '_' + plotarg + '_' + label1 + '_' + label2 + '_histogram' + plotextension)

    # Convert the unit of the data to nm if specified by the user
    dfnew = df.copy()
    dfnew2 = df2.copy()
    dfnew[plotarg] = dataunitconversion(dfnew[plotarg], plotarg, nm)
    dfnew2[plotarg2] = dataunitconversion(dfnew2[plotarg2], plotarg2, nm)
    dfplot = pd.concat([dfnew[plotarg], dfnew2[plotarg2]], axis=1)
    dfplot.columns = [label1, label2]

    # Plot figure
    fig, ax = plt.subplots(figsize=(15, 12))
    # dfnew[plotarg].plot.hist(ax=ax, alpha=1, linewidth=3.0, bins=bins, color='orange')
    # dfnew2[plotarg2].plot.hist(ax=ax, alpha=1, linewidth=3.0, bins=bins, histtype='barstacked', color='blue')

    # dfplot.plot.hist(ax=ax, alpha=0.5, linewidth=3.0, bins=bins, stacked=False)

    dfnew[plotarg].plot.hist(ax=ax, bins=np.linspace(41, 112, 21), alpha=0.8, linewidth=3.0, stacked=False,
                             density=True,
                             color='#B45F06')
    dfnew2[plotarg2].plot.hist(ax=ax, bins=np.linspace(41, 112, 21), alpha=0.6, linewidth=3.0, stacked=False,
                               density=True, color='#0B5394')

    # dfnew[plotarg].plot.hist(ax=ax, bins=np.linspace(10, 80, 21), alpha=1, linewidth=3.0, stacked=False,
    #                          color='#0B5394')
    # dfnew2[plotarg2].plot.hist(ax=ax, bins=np.linspace(10, 80, 21), alpha=1, linewidth=3.0, stacked=False,
    #                            color='#76719F')

    # Label plot and save figure
    plt.xlim(xmin, xmax)
    plt.xlabel(labelunitconversion(plotarg, nm), alpha=1)
    plt.ylabel('Count', alpha=1)
    plt.ticklabel_format(axis='both', style='sci', scilimits=(-3, 3))
    ax.tick_params(direction='out', bottom=True, left=True)
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
    # # plt.ticklabel_format(axis='both', style='sci', scilimits=(-3, 3))
    # plt.savefig(savename)
    pass


def plotdist2var(df, plotarg, plotarg2=None, df2=None, xmin=None, xmax=None, bins=20, nm=False,
                 specpath=None,
                 plotextension=defextension, plotname=None, c1=None, c2=None):
    """Dist plot for 2 variables"""

    print 'Plotting dist plot of %s and %s' % (plotarg, plotarg2)

    if plotarg2 is None:
        plotarg2 = plotarg

    if plotname is None:
        plotname = plotarg + '_and_' + plotarg2

    # Set the name of the file
    if specpath is None:
        specpath = path
    savename = os.path.join(pathman(specpath) + '_' + plotname + '_dist' + plotextension)

    if df2 is None:
        df2 = df

    # Convert the unit of the data to nm if specified by the user
    dfnew = dataunitconversion(df[plotarg], plotarg, nm)
    dfnew2 = dataunitconversion(df2[plotarg2], plotarg2, nm)

    # Plot figure
    fig, ax = plt.subplots(figsize=(14.5, 12))
    sns.distplot(dfnew, ax=ax, bins=bins, color=c1)
    sns.distplot(dfnew2, ax=ax, bins=bins, color=c2)

    # Label plot and save figure
    plt.xlim(xmin, xmax)
    plt.xlabel('Length / nm')
    # plt.xlabel(labelunitconversion(plotarg, nm), alpha=1)
    plt.ylabel('Probability Density', alpha=1)
    plt.ticklabel_format(axis='both', style='sci', scilimits=(-3, 3))
    ax.tick_params(direction='out', bottom=True, left=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.savefig(savename)


def plotviolin(df, plotarg, grouparg=None, ymin=None, ymax=None, nm=False, specpath=None, plotextension=defextension):
    """Creating a violin plot for the chosen variable. Grouping optional. The y axis range can be defined by the user.
    The default unit is metre, but this can be changed to nanometre by adding 'nm=True'. The default path is the path
    under the if __name__ == '__main__' line, but this can also be changed using the specpath argument."""

    print ('Plotting violin of %s' % plotarg)

    # Set the name of the file
    if specpath is None:
        specpath = path
    savename = os.path.join(pathman(specpath) + '_' + plotarg + '_violin' + plotextension)

    # Convert the unit of the data to nm if specified by the user
    dfnew = df.copy()
    dfnew[plotarg] = dataunitconversion(df[plotarg], plotarg, nm)

    # Plot and save figures
    fig, ax = plt.subplots(figsize=(15, 12))
    # Single violin plot
    if grouparg is None:
        ax = sns.violinplot(data=dfnew[plotarg])
    # Grouped violin plot
    else:
        dfnew = dfnew[[grouparg, plotarg]]
        ax = sns.violinplot(x=grouparg, y=plotarg, data=dfnew,
                            palette={"SKICH": '#3E96CD', "Zinc fingers": '#64B761', "TopoStats": '#76719F'})
        ax.invert_xaxis()  # Useful for topoisomers with negative writhe

    # Label plot and save figure
    plt.ylim(ymin, ymax)
    plt.ylabel(labelunitconversion(plotarg, nm), alpha=1)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
    ax.tick_params(direction='out', bottom=True, left=True)
    plt.xlabel(grouparg)
    plt.savefig(savename)


def plotviolin2var(df, plotarg, df2=None, plotarg2=None, label1=None, label2=None, ymin=None, ymax=None, nm=False,
                   specpath=None, plotextension=defextension):
    """Creating a violin plot for two sets of data. Grouping optional. The y axis range can be defined by the user. The
    default unit is metre, but this can be changed to nanometre by adding 'nm=True'. The default path is the path under
    the if __name__ == '__main__' line, but this can also be changed using the specpath argument."""

    if df2 is None:
        df2 = df
    if plotarg2 is None:
        plotarg2 = plotarg
    if label1 is None:
        label1 = plotarg
    if label2 is None:
        label2 = plotarg2

    print ('Plotting violin plot of %s for %s and %s' % (plotarg, label1, label2))

    # Set the name of the file
    if specpath is None:
        specpath = path
    savename = os.path.join(pathman(specpath) + '_' + plotarg + '_' + label1 + '_violin' + plotextension)
    # savename = os.path.join(pathman(specpath) + '_' + plotarg + '_' + label1 + '_' + label2 + '_violin' + plotextension)

    # Convert the unit of the data to nm if specified by the user
    dfnew = df.copy()
    dfnew2 = df2.copy()
    dfnew[plotarg] = dataunitconversion(dfnew[plotarg], plotarg, nm)
    dfnew2[plotarg2] = dataunitconversion(dfnew2[plotarg2], plotarg2, nm)
    dfplot = pd.concat([dfnew[plotarg], dfnew2[plotarg2]], axis=1)
    dfplot.columns = [label1, label2]

    # Plot figure
    fig, ax = plt.subplots(figsize=(15, 12))
    ax = sns.violinplot(data=dfplot, bw=.25)

    # Label plot and save figure
    plt.ylim(ymin, ymax)
    plt.ylabel(labelunitconversion(plotarg, nm), alpha=1)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
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

    # Convert the unit of the data to nm if specified by the user
    dfnew = df.copy()
    dfnew[arg1] = dataunitconversion(df[arg1], arg1, nm)
    dfnew[arg2] = dataunitconversion(df[arg2], arg2, nm)

    # Plot data using seaborn
    sns.jointplot(arg1, arg2, data=dfnew, kind='reg', height=15)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel(labelunitconversion(arg1, nm), alpha=1)
    plt.ylabel(labelunitconversion(arg2, nm), alpha=1)
    plt.savefig(savename)


def plotLinearVsCircular(contour_lengths_df):
    pass


def computeStats(data, columns, min, max, savename=None):
    """Prints out a table of stats, including the standard deviation, standard error, N value, and peak position"""

    xs = np.linspace(min, max, 6000)

    a = {}
    b = {}
    table = {
        'max': [0] * len(data),
        'std': [0] * len(data),
        'ste': [0] * len(data),
        'N': [0] * len(data),
        'ave': [0] * len(data),
    }

    for i, x in enumerate(data):
        # x = x * 1e9
        a[i] = scipy.stats.gaussian_kde(x)
        b[i] = a[i].pdf(xs)
        peaks = sig.find_peaks(b[i], height=0.3)
        extrema = sig.argrelmax(b[i])
        print (extrema)
        table['std'][i] = np.std(x)
        table['ste'][i] = stats.sem(x)
        table['max'][i] = xs[np.argmax(b[i])]
        table['N'][i] = len(x)
        table['ave'][i] = np.average(x)
    dfmax = pd.DataFrame.from_dict(table, orient='index', columns=columns)
    dfmax.to_csv(pathman(path) + savename +'.csv')


if __name__ == '__main__':

    # Path to the json file, e.g. C:\\Users\\username\\Documents\\Data\\Data.json
    path = ''

    bins = 20

    # import data form the json file specified as a dataframe
    df = importfromfile(path)
    # df2 = importfromfile(path2)

    # fig, ax = plt.subplots(figsize=(15, 12))
    # # df.set_index('Contour length')
    # # df.groupby('Basename').plot('Curvature', alpha=0.5)
    #

    # plt.plot(df1['Contour length'], df1['Curvature'], markersize=10, marker='.', color='b', alpha=0.5, linestyle="None")
    # plt.plot(df2['Contour length'], df2['Curvature'], markersize=10, marker='.', color='y', alpha=0.5, linestyle="None")
    #
    # # scatterplot = sns.scatterplot(data=df, x='Contour length', y='Curvature', hue='Basename', alpha=0.5)
    # # scatterplot.legend(loc='upper right')
    # # plt.plot(df2['Contour length'], df2['Curvature'], color='y', alpha=0.5)
    # # plt.xlim(0, 150)
    # # plt.ylim(-5, 5)
    # # plt.show()
    # curvdir = os.path.join(os.path.dirname(path), 'Curvature_Comparison')
    # if not os.path.exists(curvdir):
    #     os.makedirs(curvdir)
    # plt.savefig(os.path.join(curvdir, 'filtered.png'))

    # df = df[df['Circular'] == False]
    # df = df[df['Contour Lengths'] > 80]
    # df = df[df['Contour Lengths'] < 130]
    # df = df[df['Max Curvature'] < 2]

    # df2 = df2[df2['Circular'] == False]
    # df2 = df2[df2['Contour Lengths'] > 80]
    # df2 = df2[df2['Contour Lengths'] < 130]
    # df2 = df2[df2['Max Curvature'] < 2]

    # df.loc[df['directory'] == 'DNA_20220601', 'directory'] = 'DNA'

    # df = df[df['grain_max_bound_size'] <= 115e-9]
    # df = df[df['grain_max_bound_size'] >= 50e-9]

    # df1 = df[df['directory'] == 'DNA_pure']
    # df2 = df[df['directory'] == 'DNA_NDP']

    # dfDNA = df[df['directory'] == 'DNA_pure']
    # print len(dfDNA)
    # dfDNANDP = df[df['directory'] == 'DNA_NDP']
    # print len(dfDNANDP)

    # Rename directory column as appropriate
    # df1 = df1.rename(columns={"directory": "Experimental Conditions"})
    # df2 = df2.rename(columns={"directory": "Experimental Conditions"})
    # df = df.rename(columns={"grain_min_bound_size": "Minimum Bound Size"})
    # df = df.rename(columns={"directory": "Experimental Conditions"})
    df = df.rename(columns={"Basename": "Immobilisation\ntechnique"})


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

# print len(df)
# print len(df2)
# Setting group argument
grouparg = 'Immobilisation\ntechnique'
# grouparg = 'Mask'
# grouparg = 'Basename'
# grouparg = 'directory'
# grouparg = None
# grouparg = 'Domain'

# Setting a continuous colour palette; useful for certain grouped plots, but can be commented out if unsuitable.
# sns.set_palette(sns.color_palette('BuPu', n_colors=len(df.groupby(grouparg))))
# print df.pivot(columns=grouparg, values='grain_median')


# Plot one column of the dataframe e.g. 'grain_mean_radius'; grouparg can be specified for plotkde, plothist and
# plotviolin by entering e.g. 'xmin = 0'; xmin and xmax can be specified for plotkde, plothist, and plotjoint;
# ymin and ymax can be specified for plotviolin and plotjoint; bins can be speficied for plothist. The default unit is
# m; add "nm=True" to change from m to nm.


# plothist(df, 'Height', grouparg=grouparg)

# plothist(df, 'grain_maximum', xmin=1, xmax=6, nm=True)


# plothist(df, 'grain_max_bound_size', nm=True, grouparg=grouparg)

# plothist2var(df1, 'grain_max_bound_size', df2=df2, nm=True)
# plotkde2var(df2, 'grain_max_bound_size', df2=df1, nm=True, xmin=35, xmax=115)
#
# plotdist2var(df1, 'grain_max_bound_size', df2=df2, nm=True, c1='#B45F06', c2='#0B5394', plotname='Length', xmin=40, xmax=120)


# columns = ['height', 'length', 'width', 'area']
# data = [df['grain_maximum'], df['grain_max_bound_size'], df['grain_min_bound_size'], df['grain_proj_area']]
# plothist(df, 'grain_min_bound_size', nm=True, grouparg=grouparg)
# plothist(df, 'aspectratio', nm=True, grouparg=grouparg)
# plotkde(df, 'grain_max_bound_size', xmin=0, xmax=80, nm=True, grouparg=grouparg)
# plotkde(df, 'grain_min_bound_size', xmin=0, xmax=40, nm=True, grouparg=grouparg)
# plotkde(df, 'grain_maximum', nm=True, xmin=0, xmax=6, grouparg=grouparg)
# plotkde(df, 'grain_proj_area', nm=True, xmin=0, xmax=1000, grouparg=grouparg)
#

# computeStats(data, columns, 0, 1e-15)
# plotkde(df, 'grain_maximum', nm=True, grouparg=grouparg)
# plotkde(df, 'grain_maximum', nm=True, grouparg=grouparg)
# plotkde(df, 'aspectratio', nm=True, grouparg=grouparg)



# plothist(df, 'grain_maximum', xmin=0, xmax=9, nm=True, grouparg=grouparg)
# plothist(df, 'grain_mean', nm=True, grouparg=grouparg)
# plothist(df, 'grain_median', nm=True, grouparg=grouparg)
# # plothist(df, 'grain_proj_area', nm=True, grouparg=grouparg)
# # plothist(df, 'grain_min_volume', nm=True, grouparg=grouparg)
#
#
# plotkde(df, 'grain_maximum', xmin=1, xmax=5, nm=True, grouparg=grouparg)
# plotkde(df, 'grain_mean', xmin=0, xmax=5, nm=True, grouparg=grouparg)
# plotkde(df, 'grain_median', xmin=0, xmax=5, nm=True, grouparg=grouparg)

dfplo= df[df['Immobilisation\ntechnique'] == 'PLO']
dfNi= df[df['Immobilisation\ntechnique'] == 'NiCl2']
columns = ['ete', 'cl', 'meancurvature', 'maxcurvature']
plodata = [dfplo['End to End Distance'], dfplo['Contour Lengths'], dfplo['Mean Curvature'], dfplo['Max Curvature']]
Nidata = [dfNi['End to End Distance'], dfNi['Contour Lengths'], dfNi['Mean Curvature'], dfNi['Max Curvature']]
computeStats(plodata, columns, 0, 200, savename='PLO_ete_cl')
computeStats(plodata, columns, 0, 0.2, savename='PLO_curvature')
computeStats(Nidata, columns, 0, 200, savename='Ni_ete_cl')
computeStats(Nidata, columns, 0, 0.2, savename='Ni_curvature')
#
plotkde(df[df['Circular'] == False], 'End to End Distance', nm=True, xmin=0, xmax=150, grouparg=grouparg)
plotkde(df, 'Contour Lengths', nm=True, xmin=0, xmax=250, grouparg=grouparg)
# plotkde(df, 'Mean Curvature', nm=True, grouparg=grouparg, xmin=0, xmax=0.25)
plotkde(df, 'Mean Curvature', nm=True, grouparg=grouparg, xmin=-0.5, xmax=0.75)

plothist(df[df['Circular'] == False], 'End to End Distance', nm=True, xmin=0, xmax=150, grouparg=grouparg, bins=np.linspace(0, 150, 21))
plothist(df, 'Contour Lengths', nm=True, xmin=0, xmax=250, grouparg=grouparg, bins=np.linspace(0, 250, 21))
# plothist(df, 'Mean Curvature', nm=True, grouparg=grouparg, xmin=0, xmax=0.25)
plothist(df, 'Mean Curvature', nm=True, grouparg=grouparg, bins=np.linspace(0.0, 0.6, 21))


# plotkde(df, 'grain_proj_area', nm=True, grouparg=grouparg)
# plotkde(df, 'grain_min_volume', nm=True, grouparg=grouparg)

# plothist2var(df, 'grain_max_bound_size', df2=df2, label1='All', label2='Oligomers', nm=True)

# plotkde(df, 'grain_min_bound_size', xmin=0, xmax=40, nm=True)
# plotkde2var(df, 'grain_min_bound_size', 'grain_max_bound_size', xmin=0, xmax=40, nm=True)

# plotkde(df, 'grain_min_bound_size', xmin=0, xmax=40, nm=True)

# plotviolin2var(df, 'grain_maximum', df2=df2, plotarg2=['CTD', 'NTD'], label1='TopoStats', label2=['CTD', 'NTD'], nm=True)
# plotviolin(df, 'Height', grouparg='Source')
# plotkde(df, 'Height', xmin=0, xmax=6, nm=True)
# plothist2var(df, 'grain_maximum', df2=df2, plotarg2='Height', label1='TopoStats', label2='Manual Measurements', nm=True)

# plotdist2var('grain_min_bound_size', 'grain_max_bound_size', df2, plotname='Bound Size for Full Protein', nm=True,
#              xmin=0, xmax=50, c1='#f9cb9c', c2='#b45f06', plotextension='.tiff')
# plotdist2var('grain_min_bound_size', 'grain_max_bound_size', df, df2=df, plotname='Bound Size for CTD', nm=True, xmin=0,
#              xmax=50, c1='#9fc5e8', c2='#0b5394', plotextension='.tiff')
# plotdist2var('grain_min_bound_size', 'grain_min_bound_size', df, df2=df2,
#              plotname='Minimum Bound Size for CTD and Full Protein', nm=True, xmin=0, xmax=50, c1='#9fc5e8',
#              c2='#f9cb9c', plotextension='.tiff')
# plotdist2var('grain_max_bound_size', 'grain_max_bound_size', df, df2=df2,
#              plotname='Maximum Bound Size for CTD and Full Protein', nm=True, xmin=0, xmax=50, c1='#0b5394',
#              c2='#b45f06', plotextension='.tiff')
# plotdist2var('grain_max_bound_size', 'grain_min_bound_size', df, df2=df2,
#               plotname='Mininmum Bound Size for Full Protein and Maximum Bound Size for CTD', nm=True, xmin=0, xmax=50,
#               c1='#0b5394', c2='#f9cb9c', plotextension='.tiff')
#
# plothist2var(df, 'grain_maximum', df2=df2, label1='CTD', label2='FL', xmax=10, nm=True)
# plothist2var(df, 'grain_mean', df2=df2, label1='CTD', label2='FL', xmax=4, nm=True)
# plothist2var(df, 'grain_median', df2=df2, label1='CTD', label2='FL', xmax=4, nm=True)
# # plothist2var('grain_mean', 'grain_mean', df, df2=df2, xmin=0, xmax=10, nm=True)
# # plothist2var('grain_median', 'grain_median', df, df2=df2, xmin=0, xmax=10, nm=True)
#
# plotkde(df, 'grain_mean', xmin=0, xmax=4, nm=True)
# #
# #
# plotkde2var(df, 'grain_maximum', df2=df2, label1='CTD', label2='FL', xmin=0, xmax=10, nm=True)
# plotkde2var(df, 'grain_mean', df2=df2, label1='CTD', label2='FL', xmin=0, xmax=4, nm=True)
# plotkde2var(df, 'grain_median', df2=df2, label1='CTD', label2='FL', xmin=0, xmax=4, nm=True)
# #
# data = [df['grain_min_bound_size'], df['grain_max_bound_size'], df2['grain_min_bound_size'],
#             df2['grain_max_bound_size']]
# columns = ['Min for CTD / nm', 'Max for CTD / nm',
#                'Min for FL / nm', 'Max for FL / nm']

# data = [df[df['Source'] == 'SKICH']['Height'], df[df['Source'] == 'Zinc fingers']['Height'],
#         df[df['Source'] == 'TopoStats']['Height']]
# columns = ['SKICH', 'Zinc fingers', 'TopoStats']
#

# computeStats(data, columns, 0, 6)



# plotkde(df, 'End to End Distance', grouparg=grouparg, nm=True)
# plotkde(df, 'Contour Lengths', grouparg=grouparg, nm=True)
# plotkde(df, 'Variance of Curvature', grouparg=grouparg, nm=True)
# plotkde(df, 'Variance of Absolute Curvature', grouparg=grouparg, nm=True)
# plotkde(df, 'Mean Curvature', grouparg=grouparg, nm=True)
# plotkde(df, 'Max Curvature', grouparg=grouparg, nm=True)
# plothist(df, 'End to End Distance', grouparg=grouparg, nm=True)
# plothist(df, 'Contour Lengths', grouparg=grouparg, nm=True)
# plothist(df, 'Variance of Curvature', grouparg=grouparg, nm=True)
# plothist(df, 'Variance of Absolute Curvature', grouparg=grouparg, nm=True)
# plothist(df, 'Mean Curvature', grouparg=grouparg, nm=True)
# plothist(df, 'Max Curvature', grouparg=grouparg, nm=True)

# plotkde (df, 'aspectratio')

# plotkde2var(df, 'grain_min_bound_size', 'grain_max_bound_size', nm=True)
# plotkde(df, 'grain_max_bound_size', xmax=3.5e-8)
# plotkde(df, 'grain_half_height_area', xmin=0, xmax=1.5e2, specpath=path, grouparg=grouparg, nm=True)

# plotkde(df, 'grain_maximum', specpath=path, xmin=0, xmax=1e-8, grouparg=grouparg)
# plotkde(df, 'grain_mean_radius', specpath=path, nm=True, xmin=0, xmax=10)
# plotkde(df, 'grain_bound_len', specpath=path, grouparg=grouparg, xmin=0, xmax=125, nm=True)

# plotkde(df, 'Max Curvature Location', nm=True, grouparg=grouparg)
# plothist(df, 'Max Curvature Location', nm=True, bins=50, grouparg=grouparg)
# plotkde(df, 'Max Curvature', xmin=0, xmax=1.5, grouparg=grouparg)
# plothist(df, 'Max Curvature', grouparg=grouparg)
# plothist(df, 'Contour Lengths', xmin=80, xmax=130, nm=True, bins=20, grouparg=grouparg)
# plotkde(df, 'Contour Lengths', xmin=80, xmax=130, nm=True, grouparg=grouparg)
# plotkde(df, 'Mean Curvature', xmin=0, xmax=0.4, grouparg=grouparg)
# plothist(df, 'Mean Curvature', xmin=0, xmax=0.4, grouparg=grouparg)


# plotdist2var('Mean Curvature', 'Mean Curvature', df, df2=df2)
# plotdist2var('End to End Distance', 'End to End Distance', df, df2=df2, nm=True)
# plotkde2var(df, 'Mean Curvature', df2=df2)
# plotkde(df, 'Mean Curvature')
# plotkde(df, 'Contour Lengths', nm=True)
# plothist(df, 'Mean Curvature', bins=50)


# plotkde(df, 'grain_maximum', nm=True, grouparg='directory', xmin=0, xmax=5)
# plotkde(df, 'grain_mean', nm=True, grouparg='directory', xmin=0, xmax=3)
# plotkde(df, 'grain_median', nm=True, grouparg='directory', xmin=0, xmax=4)

# plotkde(df, 'grain_maximum', nm=True, xmin=1, xmax=5)