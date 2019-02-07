#!/usr/bin/env python2

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set seaborn to override matplotlib for plot output
sns.set()
# The four preset contexts, in order of relative size, are paper, notebook, talk, and poster.
# The notebook style is the default
sns.set_context("poster")


def importfromjson(path, name):
    filename = os.path.join(path, name + '.json')
    importeddata = pd.read_json(filename)

    return importeddata


def plotting(dataframe, arg1, grouparg, bins, directory, extension):
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

    # Plot using MatPlotLib separated by the grouparg on two separate graphs with stacking
    # Create a figure of given size
    fig = plt.figure(figsize=(18, 12))
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

    # Plot each argument together using MatPlotLib
    # Create a figure of given size
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111)
    # Set title
    ttl = 'Histogram of %s' % arg1
    # Melt dataframe to leave only columns we are interested in
    df3 = pd.melt(df, id_vars=[arg1])
    # Plot histogram
    df3.plot.hist(ax=ax, bins=bins, range=(min_ax, max_ax), alpha=.3)
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


def plottingallstats(grainstatsarguments, df, extension, directory):
    # Create a saving name format/directory
    savedir = os.path.join(directory, 'Plots')
    savename = os.path.join(savedir, os.path.splitext(os.path.basename(directory))[0])
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    for key in grainstatsarguments:
        print 'Plotting graph of %s' % (key)
        # Plot each argument together using MatPlotLib
        # Create a figure of given size
        fig = plt.figure(figsize=(18, 12))
        ax = fig.add_subplot(111)
        # Set title
        ttl = 'Histogram of %s' % key
        # Melt dataframe to leave only columns we are interested in
        df3 = pd.melt(df, id_vars=[key])
        # Plot histogram
        df3.plot.hist(ax=ax, alpha=.3)
        plt.xlabel('%s (nm)' % key)
        # # Set legend options
        # plt.legend(ncol=2, loc='upper right')
        # Set tight borders
        plt.tight_layout()
        # Save plot
        plt.savefig(savename + '_' + key + extension)


def plotting2(df, arg1, arg2, grouparg, bins, directory, extension):
    print 'Plotting graph of %s and %s' % (arg1, arg2)
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

    # Plot data

    # Plot each type using MatPlotLib separated by filetype on two separate graphs with stacking
    # Create a figure of given size
    fig = plt.figure(figsize=(28, 8))
    # First dataframe
    # Add a subplot to lot both sets on the same figure
    ax = fig.add_subplot(121)
    # Set title
    ttl = 'Histogram of %s and %s' % (arg1, arg2)
    # Pivot dataframe to get required variables in correct format for plotting
    df1 = df.pivot(columns=grouparg, values=arg1)
    # Plot histogram
    df1.plot.hist(legend=True, ax=ax, bins=bins, range=(min_ax, max_ax), alpha=.3, stacked=True)
    # Set x axis label
    plt.xlabel('%s (nm)' % arg1)
    # Set tight borders
    plt.tight_layout()
    # Set legend options
    # plt.legend(ncol=2, loc='upper right')
    # Second dataframe
    # Add a subplot
    ax = fig.add_subplot(122)
    # Pivot second dataframe to get required variables in correct format for plotting
    df2 = df.pivot(columns=grouparg, values=arg2)
    # Plot histogram
    df2.plot.hist(legend=True, ax=ax, bins=bins, range=(min_ax, max_ax), alpha=.3, stacked=True)
    # Set x axis label
    plt.xlabel('%s (nm)' % arg2)
    # Set tight borders
    plt.tight_layout()
    # Set legend options
    # plt.legend(ncol=2, loc='upper right')
    # Save plot
    plt.savefig(savename + '_' + arg1 + '_' + arg2 + '_' + 'a' + extension)

    # Create a figure of given size
    fig = plt.figure(figsize=(18, 12))
    # Add a subplot
    ax = fig.add_subplot(111)
    # Set title
    ttl = 'Histogram of %s and %s' % (arg1, arg2)
    # Plot each argument together using MatPlotLib
    df3 = pd.melt(df, id_vars=[arg1, arg2])
    df3.plot.hist(legend=True, ax=ax, bins=bins, range=(min_ax, max_ax), alpha=.3)
    # plt.xlabel('%s %s (nm)' % (arg1, arg2))
    plt.xlabel('nm')
    # # Set legend options
    # plt.legend(ncol=2, loc='upper right')
    # Set tight borders
    plt.tight_layout()
    # Save plot
    plt.savefig(savename + '_' + arg1 + '_' + arg2 + '_' + 'b' + extension)


def savestats(directory, dataframetosave):
    directoryname = os.path.splitext(os.path.basename(directory))[0]
    print 'Saving stats for: ' + str(directoryname)

    savedir = os.path.join(directory)
    savename = os.path.join(savedir, directoryname)
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    dataframetosave.to_json(savename + 'evaluated.json')
    dataframetosave.to_csv(savename + 'evaluated.txt')


# This the main script
if __name__ == '__main__':
    # Set the file path, i.e. the directory where the files are here'
    # path = '/Users/alice/Dropbox/UCL/DNA MiniCircles/Minicircle Data Edited/DNA/339/Nickel'
    # path = '/Users/alice/Dropbox/UCL/DNA MiniCircles/Minicircle Data Edited/DNA/339/Nickel'
    path = '/Users/alice/Dropbox/UCL/DNA MiniCircles/Minicircle Data Edited/New Images/Nickel'
    # Set the name of the json file to import here
    name = 'Nickel_Kavit'
    plotextension = '.pdf'
    bins = 25

    # import data form the json file specified as a dataframe
    df = importfromjson(path, name)

    # Rename directory column as topoisomer
    df = df.rename(columns={"directory": "topoisomer"})
    # df = df.rename(columns={'grain_min_bound_size': 'width', 'grain_max_bound_size': 'length'})

    # Evaluate the aspect ratio for each grain
    df['aspectratio'] = df['grain_min_bound_size'] / df['grain_max_bound_size']

    # Get list of unique directory names i.e. topoisomers
    topos = df['topoisomer'].unique()
    sorted(topos, reverse=True)

    # Generate new dataframes for each topoisomer
    dfrel = df.loc[df['topoisomer'] == 'Relaxed']
    dfnic = df.loc[df['topoisomer'] == 'Nicked']
    dfnat = df.loc[df['topoisomer'] == 'Native']
    df6 = df.loc[df['topoisomer'] == '-6']
    df3 = df.loc[df['topoisomer'] == '-3']
    df2 = df.loc[df['topoisomer'] == '-2']
    df1 = df.loc[df['topoisomer'] == '-1']

    topodflist = [dfrel, dfnic, dfnat, df1, df2, df3, df6]

    nat = df.query("topoisomer == 'native'")

    # Generate a new smaller df from the original df containing only the columns topoisomer and mean radius
    dfradius = df[['topoisomer', 'grain_mean_radius']]
    dfaspectratio = df[['topoisomer', 'aspectratio']]

    # Get statistics for different topoisoimers
    allstats = df.groupby('topoisomer').describe()
    # transpose allstats dataframe to get better saving output
    allstats1 = allstats.transpose()
    # Save out statistics file
    savestats(path, allstats1)

    # Plot and save figures
    savename = os.path.join(path, name + '_aspectratio' + plotextension)
    fig, ax = plt.subplots(figsize=(10, 7))
    df.groupby('topoisomer')['aspectratio'].plot.kde(ax=ax, legend=True)
    plt.xlim(0, 1)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper left')
    plt.savefig(savename)

    plotting(df, 'aspectratio', 'topoisomer', bins, path, plotextension)

    # Plot all columns of dataframe and save as graph
    columnstoplot = list(df.select_dtypes(include=['float64', 'int64']).columns)
    for x in columnstoplot:
        savename = os.path.join(path, name + '_' + str(x) + plotextension)
        fig, ax = plt.subplots(figsize=(10, 7))
        df.groupby('topoisomer')[x].plot.kde(ax=ax, legend=True)
        plt.savefig(savename)

    # # Plotting all topoisomers separately as KDE plots using seaborn
    # p1 = sns.kdeplot(dfnicked['aspectratio'], shade=True)
    # p2 = sns.kdeplot(dfrelaxed['aspectratio'], shade=True)
    # p3 = sns.kdeplot(dfnative['aspectratio'], shade=True)
    # p4 = sns.kdeplot(df1['aspectratio'], shade=True)
    # p5 = sns.kdeplot(df2['aspectratio'], shade=True)
    # p6 = sns.kdeplot(df3['aspectratio'], shade=True)
    # p7 = sns.kdeplot(df6['aspectratio'], shade=True)

    # Plotting a distribution with given fit
    # sns.distplot(df6['aspectratio'], kde=False, fit=stats.gamma)

    # plotting(df, 'grain_max_bound_size', 'topoisomer', bins, path, plotextension)

    # plotting2(df, 'grain_min_bound_size', 'grain_max_bound_size', 'topoisomer', bins, path, plotextension)
