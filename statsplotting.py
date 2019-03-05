#!/usr/bin/env python2

import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

# Set seaborn to override matplotlib for plot output
sns.set()
# The four preset contexts, in order of relative size, are paper, notebook, talk, and poster.
# The notebook style is the default
sns.set_context("poster")


def importfromjson(path, name):
    filename = os.path.join(path, name + '.json')
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


def plotkde(df, directory, name, plotextension, grouparg, plotarg):
    print 'Plotting kde of %s' % plotarg

    # Create a saving name format/directory
    savedir = os.path.join(directory, 'Plots')
    savename = os.path.join(savedir, os.path.splitext(os.path.basename(directory))[0])
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # Plot and save figures
    savename = os.path.join(path, name + plotarg + plotextension)
    fig, ax = plt.subplots(figsize=(10, 7))
    df.groupby(grouparg)[plotarg].plot.kde(ax=ax, legend=True)
    plt.xlim(0, 1)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper left')
    plt.savefig(savename)


def plotdfcolumns(df, path, name, grouparg):
    print 'Plotting graphs for all dataframe variables in %s' % name

    # Create a saving name format/directory
    savedir = os.path.join(path, 'Plots')
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # Plot all columns of dataframe and save as graph
    columnstoplot = list(df.select_dtypes(include=['float64', 'int64']).columns)
    for x in columnstoplot:
        savename = os.path.join(savedir, name + '_' + str(x) + plotextension)
        fig, ax = plt.subplots(figsize=(10, 7))
        df.groupby(grouparg)[x].plot.kde(ax=ax, legend=True)
        plt.savefig(savename)


def plothist(dataframe, arg1, grouparg, bins, directory, extension):
    print 'Plotting graph of %s' % (arg1)
    # Create a saving name format/directory
    savedir = os.path.join(directory, 'Plots')
    savename = os.path.join(savedir, os.path.splitext(os.path.basename(directory))[0])
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    df = dataframe

    # Change from m to nm units for plotting
    df[arg1] = df[arg1] * 1e9

    # Generating min and max axes based on datasets
    min_ax = df[arg1].min()
    min_ax = round(min_ax, 9)
    max_ax = df[arg1].max()
    max_ax = round(max_ax, 9)

    # Plot arg1 using MatPlotLib separated by the grouparg
    # Plot with figure with stacking sorted by grouparg
    # Create a figure of given size
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    # Set title
    ttl = 'Histogram of %s' % arg1
    # Pivot dataframe to get required variables in correct format for plotting
    df1 = df.pivot(columns=grouparg, values=arg1)
    # Plot histogram
    df1.plot.hist(ax=ax, legend=True, bins=bins, range=(min_ax, max_ax), alpha=.3, stacked=True)
    # Set x axis label
    plt.xlabel('%s (nm)' % arg1)
    # Set tight borders
    plt.tight_layout()
    # Set legend options
    # plt.legend(ncol=2, loc='upper right')
    # Save plot
    plt.savefig(savename + '_' + arg1 + '_a' + extension)

    # Plot arg1 using MatPlotLib
    # Create a figure of given size
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111)
    # Set title
    ttl = 'Histogram of %s' % arg1
    # Plot histogram
    df[arg1].plot.hist(ax=ax, bins=bins, range=(min_ax, max_ax), alpha=.3)
    plt.xlabel('%s (nm)' % arg1)
    # # Set legend options
    # plt.legend(ncol=2, loc='upper right')
    # Set tight borders
    plt.tight_layout()
    # Save plot
    plt.savefig(savename + '_' + arg1 + '_b' + extension)


def seaplotting(df, arg1, arg2, bins, directory, extension):
    # Create a saving name format/directory
    savedir = os.path.join(directory, 'Plots')
    savename = os.path.join(savedir, os.path.splitext(os.path.basename(directory))[0])
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # Change from m to nm units for plotting
    df[arg1] = df[arg1] * 1e9
    df[arg2] = df[arg2] * 1e9

    # Generating min and max axes based on datasets
    min_ax = min(df[arg1].min(), df[arg2].min())
    min_ax = round(min_ax, 9)
    max_ax = max(df[arg1].max(), df[arg2].max())
    max_ax = round(max_ax, 9)

    # Plot data using seaborn
    with sns.axes_style('white'):
        # sns.jointplot(arg1, arg2, data=df, kind='hex')
        sns.jointplot(arg1, arg2, data=df, kind='reg')
        plt.savefig(savename + '_' + str(arg1) + str(arg2) + '_seaborn' + extension)


# This the main script
if __name__ == '__main__':
    # Set the file path, i.e. the directory where the files are here'
    # path = '/Users/alice/Dropbox/UCL/DNA MiniCircles/Minicircle Data Edited/DNA/339/Nickel'
    # path = '/Users/alice/Dropbox/UCL/DNA MiniCircles/Minicircle Data Edited/DNA/339/Nickel'
    path = '/Users/alice/Dropbox/UCL/DNA MiniCircles/Minicircle Data Edited/New Images/Nickel/Nickel'
    # Set the name of the json file to import here
    name = 'Nickel'
    plotextension = '.pdf'
    bins = 25

    # import data form the json file specified as a dataframe
    df = importfromjson(path, name)

    # Rename directory column as topoisomer
    df = df.rename(columns={"directory": "topoisomer"})
    # df = df.rename(columns={'grain_min_bound_size': 'width', 'grain_max_bound_size': 'length'})

    # Calculate the aspect ratio for each grain
    df['aspectratio'] = df['grain_min_bound_size'] / df['grain_max_bound_size']

    # Get list of unique directory names i.e. topoisomers
    topos = df['topoisomer'].unique()
    sorted(topos, reverse=True)
    # Generate separate dataframes for each topoisomer
    # dfrel = df.loc[df['topoisomer'] == '0']
    # dfnic = df.loc[df['topoisomer'] == 'NIC']
    # dfnat = df.loc[df['topoisomer'] == 'NAT']
    # df6 = df.loc[df['topoisomer'] == '-6']
    # df3 = df.loc[df['topoisomer'] == '-3']
    # df2 = df.loc[df['topoisomer'] == '-2']
    # df1 = df.loc[df['topoisomer'] == '-1']
    # alternative method of generating dataframes for each topoisomer
    nat = df.query("topoisomer == 'native'")

    # Generate a new smaller df from the original df containing only selected columns
    dfaspectratio = df[['topoisomer', 'aspectratio']]

    # Get statistics for different topoisoimers
    allstats = df.groupby('topoisomer').describe()
    # transpose allstats dataframe to get better saving output
    allstats1 = allstats.transpose()
    # Save out statistics file
    savestats(path, name, allstats1)

    # Plot a KDE plot of one column of the dataframe - arg1 e.g. 'aspectratio'
    # grouped by grouparg e.g. 'topoisomer'
    plotkde(df, path, name, plotextension, 'topoisomer', 'aspectratio')


    # # Plot all columns of a dataframe as separate graphs grouped by topoisomer
    # plotdfcolumns(df, path, name, 'topoisomer')

    # Plot a histogram of one column of the dataframe - arg1 e.g. 'aspectratio'
    # grouped by grouparg e.g. 'topoisomer'
    # plothist(df, 'aspectratio', 'topoisomer', bins, path, plotextension)

    # # Plotting indidiviual stats from a dataframe
    # # e.g. Plot the aspect ratio column of the dataframe, grouped by topoisomer as a kde plot
    # savedir = os.path.join(path, 'Plots')
    # savename = os.path.join(savedir, name + '_aspectratio' + plotextension)
    # fig, ax = plt.subplots(figsize=(10, 7))
    # 3.plot.kde(ax=ax, legend=True)
    # plt.savefig(savename)

    # # Plotting a distribution with given fit (e.g. gamma)
    # sns.distplot(df6['aspectratio'], kde=False, fit=stats.gamma)

    # # Plot two variables in the dataframe on a seaborn joint plot to examine dependencies
    # seaplotting(df, 'grain_ellipse_major', 'grain_ellipse_minor', bins, path, plotextension)

    # # Plotting all topoisomers separately as KDE plots using seaborn
    # p1 = sns.kdeplot(dfnicked['aspectratio'], shade=True)
    # p2 = sns.kdeplot(dfrelaxed['aspectratio'], shade=True)
    # p3 = sns.kdeplot(dfnative['aspectratio'], shade=True)
    # p4 = sns.kdeplot(df1['aspectratio'], shade=True)
    # p5 = sns.kdeplot(df2['aspectratio'], shade=True)
    # p6 = sns.kdeplot(df3['aspectratio'], shade=True)
    # p7 = sns.kdeplot(df6['aspectratio'], shade=True)


    # # # Plot bivariate plot using seaborn
    # ax = sns.kdeplot(df.query("topoisomer == '-6'")['grain_max_bound_size'], df.query("topoisomer == '-6'")['grain_min_bound_size'], n_levels=15, shade=True)
    # ax = sns.kdeplot(df.query("topoisomer == 'NAT'")['grain_max_bound_size'], df.query("topoisomer == 'NAT'")['grain_min_bound_size'], n_levels=15, shade=True)
    # ax = sns.kdeplot(df.query("topoisomer == 'REL'")['grain_max_bound_size'], df.query("topoisomer == 'REL'")['grain_min_bound_size'], n_levels=15, shade=True)
    #
    # # # Use seaborn to setup plots for each unique topoisomer
    h = sns.FacetGrid(df, col="topoisomer")
    h.map(sns.kdeplot, "aspectratio")

    ordered_topos = df.topoisomer.value_counts().index
    ordered_topos = sorted(ordered_topos, reverse=True)
    g = sns.FacetGrid(df, row="topoisomer", row_order=ordered_topos,
                      height=1.7, aspect=4)
    g.map(sns.distplot, "aspectratio", hist=False, rug=True);

    # h = sns.FacetGrid(df, col="topoisomer")
    # h.map(plt.scatter, "grain_min_bound_size", "grain_max_bound_size", alpha=.7)

    sns.kdeplot(df.query("topoisomer == '0'")['grain_max_bound_size'],
                df.query("topoisomer == '0'")['grain_min_bound_size'], n_levels=15, shade=True)

    # g = sns.PairGrid(df, vars=['grain_max_bound_size', 'grain_min_bound_size', 'aspectratio'], hue="topoisomer")
    # g.map_diag(sns.kdeplot)
    # g.map_lower(sns.kdeplot)
    # g.map_upper(plt.scatter)

    # Determine KDE for each topoisomer and plot
    # Determine max of each KDE and plot
    xs = np.linspace(0, 1, 100)
    kdemax = dict()
    plt.figure()
    for i in sorted(topos):
        kdemax[i] = i
        x = df.query('topoisomer == @i')['aspectratio']
        a = scipy.stats.gaussian_kde(x)
        b = a.pdf(xs)
        plt.plot(xs, b)
        kdemax[i] = xs[np.argmax(b)]
    plt.figure()
    for i in sorted(topos):
        plt.bar(i, kdemax[i])