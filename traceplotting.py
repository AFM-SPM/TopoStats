#!/usr/bin/env python2

import os
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import seaborn as sns
import numpy as np
import scipy
from scipy import stats
import itertools

# Set seaborn to override matplotlib for plot output
sns.set()
sns.set_style("white")
# The four preset contexts, in order of relative size, are paper, notebook, talk, and poster.
# The notebook style is the default
# sns.set_context("notebook", font_scale=1.5)
sns.set_context("poster", font_scale=1.0)
sns.set_palette(sns.color_palette('BuPu'))
# plt.style.use("dark_background")


def plotHistogramOfTwoDataSets(data_frame_path, dataset_1_name, dataset_2_name):
    pass


def plotAllContourLengthHistograms(json_path):

    contour_lengths_df = pd.read_json(json_path)

    nbins = np.linspace(-10,10,30)

    project_names = set(contour_lengths_df['Experiment Directory'].array)
    #print(project_names)

    for name in project_names:


        temp_df = contour_lengths_df.loc[contour_lengths_df['Experiment Directory'] == name ]

        plt.hist(temp_df['Contour Lengths'], 20, histtype= 'bar')

        try:
            hist_data.append(temp_df['Contour Lengths'])
        except NameError:
            hist_data = [temp_df['Contour Lengths']]

        #plt.hist(hist_data, 50, histtype= 'bar', label = project_names)
            #sns.distplot(temp_df['Contour Lengths'])
    plt.legend(loc='upper right')
    plt.xlabel('Contour Length (nm)')
    plt.ylabel('Occurence')

    save_file_name = json_path.split('/')

    plt.savefig('%s.png' % json_path[:-4])
    plt.savefig('%s.svg' % json_path[:-4])
    plt.close()

def plotLinearVsCircular(contour_lengths_df):
    # Create a saving name format/directory
    savedir = os.path.join(os.path.dirname(file_name), 'Plots')
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    linear_contour_lengths = contour_lengths_df.loc[contour_lengths_df['Circular'] == False]
    circular_contour_lengths = contour_lengths_df.loc[contour_lengths_df['Circular'] == True]
    av_circ_len = circular_contour_lengths['Contour Lengths'].mean()
    av_circ_std = circular_contour_lengths['Contour Lengths'].std()
    av_lin_len = linear_contour_lengths['Contour Lengths'].mean()
    av_lin_std = linear_contour_lengths['Contour Lengths'].std()
    no_linear_mcles = len(linear_contour_lengths)
    no_circular_mcles = len(circular_contour_lengths)

    print 'circular', no_circular_mcles, av_circ_len, av_circ_std
    print 'linear', no_linear_mcles, av_lin_len, av_lin_std

    plt.hist([circular_contour_lengths['Contour Lengths'].array, linear_contour_lengths['Contour Lengths'].array], 25, histtype = 'bar', label = ['Linear', 'Circular'])
    # plt.xlabel('Contour Length (nm)')
    # plt.ylabel('Occurence')
    # plt.legend(loc='upper right')
    # plt.title('%s Linear vs Circular' % json_path[:-4])
    plt.savefig(os.path.join(os.path.dirname(file_name), 'Plots', 'linearVcircularHist.pdf'))

    num_lin_circ_df = pd.DataFrame(data = {'Linear' : [len(circular_contour_lengths)], 'Circular' : [len(linear_contour_lengths)]})

    sns.barplot(data = num_lin_circ_df, order = ['Linear', 'Circular'])
    # plt.xlabel('Conformation')
    # plt.ylabel('Occurence')
    # plt.title('%s Linear vs Circular' % json_path[:-4])
    # plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(file_name), 'Plots', 'barplot.pdf'))
    plt.close()

    return linear_contour_lengths, circular_contour_lengths


def plotkde(df, directory, name, plotextension, grouparg, plotarg):

    print 'Plotting kde of %s' % plotarg

    # Create a saving name format/directory
    savedir = os.path.join(directory, 'Plots')
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    savename = os.path.join(savedir, name + plotarg + plotextension)

    # Plot and save figures
    fig, ax = plt.subplots(figsize=(10, 7))
    df.groupby(grouparg)[plotarg].plot.kde(ax=ax, legend=True, alpha=1, linewidth=3.0)
    plt.xlim(-20, 200)
    plt.legend(loc='upper right')
    # plt.xlabel(' ')
    # plt.ylabel(' ')
    plt.savefig(savename)

def plotdist2(df, directory, name, plotextension, grouparg, plotarg):

    print 'Plotting kde of %s' % plotarg

    # Create a saving name format/directory
    savedir = os.path.join(directory, 'Plots')
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    savename = os.path.join(savedir, name + plotarg + plotextension)

    # Plot and save figures
    fig, ax = plt.subplots(figsize=(10, 7))
    x = df.groupby(grouparg)[plotarg]
    sns.distplot(x, ax=ax, legend=True, alpha=1, linewidth=3.0)
    plt.xlim(-20, 200)
    plt.legend(loc='upper right')
    # plt.xlabel(' ')
    # plt.ylabel(' ')
    plt.savefig(savename)

def plotkdei(df, directory, name, plotextension, grouparg, plotarg, i, color):
    palette = sns.color_palette('BuPu', n_colors=no_expmts)
    print 'Plotting kde of %s' % plotarg

    # Create a saving name format/directory
    savedir = os.path.join(directory, 'Plots')
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    savename = os.path.join(savedir, name + plotarg + i + plotextension)

    # Plot and save figures
    fig, ax = plt.subplots(figsize=(10,10))
    df.groupby(grouparg)[plotarg].plot.kde(ax=ax, legend=True, alpha=1, linewidth=3.0, color=color)
    plt.xlim(-20, 200)
    plt.legend(loc='upper right')
    # plt.xlabel(' ')
    # plt.ylabel(' ')
    plt.tight_layout()
    plt.savefig(savename)

def plothist(df, directory, name, plotextension, grouparg, plotarg):
    palette = sns.color_palette('BuPu', n_colors=no_expmts)
    print 'Plotting histogram of %s' % plotarg

    # Create a saving name format/directory
    savedir = os.path.join(directory, 'Plots')
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # Plot and save figures
    savename = os.path.join(savedir, name + plotarg + '_histogram1' + plotextension)
    fig, ax = plt.subplots(figsize=(10, 7))
    # Pivot dataframe to get required variables in correct format for plotting
    df1 = df.pivot(columns=grouparg, values=plotarg)
    # Plot histogram
    df1.plot.hist(ax=ax, legend=True, bins=bins, alpha=.7, stacked=False)
    plt.xlim(0, 200)
    plt.legend(loc='upper right')
    # plt.xlabel(' ')
    # plt.ylabel(' ')
    plt.savefig(savename)

def plothisti(df, directory, name, plotextension, grouparg, plotarg, i, color):

    print 'Plotting histogram of %s' % plotarg

    # Create a saving name format/directory
    savedir = os.path.join(directory, 'Plots')
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # Plot and save figures
    savename = os.path.join(savedir, name + plotarg + i + '_hist' + plotextension)

    fig, ax = plt.subplots(figsize=(10, 10))
    df.groupby(grouparg)[plotarg].plot.hist(bins=10, ax=ax, legend=True, alpha=0.7, stacked=True, color=color)
    plt.xlim(0, 200)
    plt.legend(loc='upper right')
    # plt.xlabel(' ')
    # plt.ylabel(' ')
    plt.tight_layout()
    plt.savefig(savename)


def plotfacet(df, directory, name, plotextension, grouparg, plotarg):
    print 'Plotting facet of %s' % plotarg

    # Create a saving name format/directory
    savedir = os.path.join(directory, 'Plots')
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # Plot and save figures as facet grid
    savename = os.path.join(savedir, name + plotarg + '_facet' + plotextension)
    fig, ax = plt.subplots()
    bins = np.arange(0, 200, 10)
    g = sns.FacetGrid(df, row='DNA Length', height=4.5, aspect=2)
    g.map(sns.distplot, "Contour Lengths", hist=True, rug=False, bins=bins).set(xlim=(0, 200), xticks=[50, 100, 150, 200])
    plt.savefig(savename)

def plotfacetsearch(df, directory, name, plotextension, grouparg, plotarg, searchfor):
    print 'Plotting reduced facet of %s' % plotarg

    # Create a saving name format/directory
    savedir = os.path.join(directory, 'Plots')
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    df = df[~df['Experiment Directory'].str.contains('|'.join(searchfor))]

    # Plot and save figures as facet grid
    savename = os.path.join(savedir, name + plotarg + '_facet_reduced' + plotextension)
    fig, ax = plt.subplots()
    bins = np.arange(0, 200, 10)
    g = sns.FacetGrid(df, row='DNA Length', height=4.5, aspect=2)
    g.map(sns.distplot, "Contour Lengths", hist=True, rug=False, axlabel=False, bins=bins).set(xlim=(0, 200), xticks=[50, 100, 150, 200])
    plt.xlabel(' ')
    # plt.ylabel(' ')
    plt.savefig(savename)

def plothiststacked2(df, directory, name, plotextension, grouparg, plotarg):
    print 'Plotting histogram of %s' % plotarg

    # Create a saving name format/directory
    savedir = os.path.join(directory, 'Plots')
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # Plot and save figures
    savename = os.path.join(savedir, name + plotarg + '_histogram_stacked' + plotextension)
    fig, ax = plt.subplots(figsize=(10, 7))
    # Pivot dataframe to get required variables in correct format for plotting
    df1 = df.pivot(columns=grouparg, values=plotarg)
    # Plot histogram
    df1.plot.hist(ax=ax, legend=True, bins=bins, alpha=.7, stacked=True)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:])
    plt.xlim(0, 200)
    plt.legend(loc='upper right')
    # plt.xlabel(' ')
    # plt.ylabel(' ')
    plt.savefig(savename)

def plotviolin(df, directory, name, plotextension, grouparg, plotarg):
    print 'Plotting violin of %s' % plotarg

    # Create a saving name format/directory
    savedir = os.path.join(directory, 'Plots')
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # Plot and save figures
    savename = os.path.join(savedir, name + plotarg + '_violin' + plotextension)
    fig, ax = plt.subplots(figsize=(10, 7))
    # Plot violinplot
    # ax.invert_xaxis()
    ax = sns.violinplot(x=grouparg, y=plotarg, data=df)
    # plt.xlim(0, 1)
    plt.xlabel(' ')
    plt.ylabel(' ')
    plt.savefig(savename)


def plotdist(df, directory, name, plotextension, grouparg, plotarg):
    print 'Plotting histogram of %s' % plotarg

    # Create a saving name format/directory
    savedir = os.path.join(directory, 'Plots')
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # Plot and save figures
    savename = os.path.join(savedir, name + plotarg + '_dist' + plotextension)

    nbin = np.linspace(0, 300, 30)

    g = sns.FacetGrid(df, hue="Experiment Directory", height=7, aspect=1.42)
    # g.map(plt.hist, "Contour Lengths", bins=bins)
    g.map(sns.distplot, "Contour Lengths", bins=nbin)
    sns.despine()
    for ax in g.axes.ravel():
        ax.legend()
    # g.add_legend();
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles=handles[1:], labels=labels[1:])
    plt.xlim(0, 200)
    # plt.legend(loc='upper right')
    plt.xlabel(' ')
    plt.ylabel(' ')
    plt.savefig(savename)

def plotkdemax(df, directory, name, plotextension, plotarg, topos):
    print 'Plotting kde and maxima for %s' % plotarg

    # Create a saving name format/directory
    savedir = os.path.join(directory, 'Plots')
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # Plot and save figures
    savename = os.path.join(savedir, name + plotarg + '_KDE_new' + plotextension)

    df = df.rename(columns={"Experiment Directory": "expmt"})

    # Determine KDE for each topoisomer
    # Determine max of each KDE and plot
    xs = np.linspace(0, 400, 100)
    kdemax = dict()
    dfstd = dict()
    dfvar = dict()
    dfste = dict()
    N = dict()
    # plt.figure()
    for i in topos:
        kdemax[i] = i
        dfstd[i] = i
        dfvar[i] = i
        dfste[i] = i
        x = df.query('expmt == @i')[plotarg]
        a = scipy.stats.gaussian_kde(x)
        b = a.pdf(xs)
        dfstd[i] = np.std(x)
        dfstd[i] = x.std()
        dfvar[i] = np.var(x)
        dfste[i] = stats.sem(x)
        N[i] = len(x)
        # plt.plot(xs, b)
        kdemax[i] = xs[np.argmax(b)]
    plt.savefig(savename)

    listofdicts = [kdemax, dfstd, dfste, N]
    dflengths = pd.DataFrame(listofdicts)
    dflengths = dflengths.transpose()
    dflengths.columns = ['length', 'std', 'ste', 'N']
    dflengths.to_csv(os.path.join(savedir, 'lengthsanderrors.txt'))

    savename2 = os.path.join(savedir, name + plotarg + '_KDE_max' + plotextension)
    fig = plt.figure(figsize=(10, 7))
    # plt.xlabel('Topoisomer')
    plt.ylabel(' ')
    # Set an arbitrary value to plot to in x, increasing by one each loop iteration
    order = 0
    # Set a value for the placement of the bars, by creating an array of the length of topos
    bars = np.linspace(0, len(topos), len(topos), endpoint=False, dtype=int)
    for i in sorted(topos, reverse=False):
        # plt.bar(order, kdemax[i], alpha=1)
        plt.bar(order, kdemax[i], yerr=dfstd[i], alpha=1)
        # plt.bar(order, kdemax[i], yerr=dfvar[i], alpha=0.7)
        order = order + 1
        # Set the bar names to be the topoisomer names
        plt.xticks(bars, sorted(topos, reverse=False))
        plt.savefig(savename2)

    return dflengths

if __name__ == '__main__':
    # Set the file path, i.e. the directory where the files are here'
    path = '/Volumes/GoogleDrive/My Drive/AFM research group /Methods paper/Data/Circular'
    # path = '/Volumes/GoogleDrive/My Drive/AFM research group /Methods paper/Data/Fortracing'
    # path2 = '/Volumes/GoogleDrive/My Drive/AFM research group /Methods paper/Data/MAC'
    # path = '/Volumes/GoogleDrive/My Drive/AFM research group /Methods paper/Data/Bea'
    name = 'tracestats.json'
    file_name = os.path.join(path, name)
    # file_name = 'new_data/tracestats.json'

    plotextension = '.pdf'
    bins = 30

    # import data form the json file specified as a dataframe
    df = pd.read_json(file_name)

    # Sort dataframe
    df = df.sort_values(["Experiment Directory", "Image Name", "Molecule number"], ascending=(True, True, True))

    # Obtain list of unique experiments
    expmts = df['Experiment Directory'].unique()
    expmts = sorted(expmts, reverse=False)
    no_expmts = len(expmts)

    # # Add column to calculate length
    # # Use if file directories have words and numbers
    # df['DNA Length'] = df['Experiment Directory'].str.extract('(\d+)').astype(int)
    # df['Length'] = df['DNA Length'] * 0.34

    # Use if file directories have numbers only
    df['DNA Length'] = df['Experiment Directory']

    # Plot proportion of linear and circular molecules
    sns.set_palette(sns.color_palette('BuPu', 2))
    linear_contour_lengths, circular_contour_lengths = plotLinearVsCircular(df)

    sns.set_palette(sns.color_palette('BuPu', no_expmts))
    # Plot data as KDE
    plotkde(df, path, name, plotextension, 'Experiment Directory', 'Contour Lengths')
    plotkde(circular_contour_lengths, path, name, '_circular.pdf', 'Experiment Directory', 'Contour Lengths')
    dflen = plotkdemax(df, path, name, plotextension, 'Contour Lengths', expmts)
    dflencirc = plotkdemax(circular_contour_lengths, path, name, '_circular.pdf', 'Contour Lengths', expmts)
    plotviolin(df, path, name, plotextension, 'Experiment Directory', 'Contour Lengths')
    plotviolin(circular_contour_lengths, path, name, '_circularviolin.pdf', 'Experiment Directory', 'Contour Lengths')

    # Plot data as histograms
    plothist(df, path, name, plotextension, 'Experiment Directory', 'Contour Lengths')
    plothist(circular_contour_lengths, path, name, '_circular.pdf', 'Experiment Directory', 'Contour Lengths')
    plothiststacked2(df, path, name, plotextension, 'Experiment Directory', 'Contour Lengths')
    plothiststacked2(circular_contour_lengths, path, name, '_circular.pdf', 'Experiment Directory', 'Contour Lengths')

    plotdist(df, path, name, plotextension, 'Experiment Directory', 'Contour Lengths')
    plotdist(circular_contour_lengths, path, name, '_circular.pdf', 'Experiment Directory', 'Contour Lengths')

    # Plot all Contour lengths for each minicircle separately
    sns.set_palette(sns.color_palette('deep', 1))
    plotfacet(df, path, name, plotextension, 'Experiment Directory', 'Contour Lengths')

    searchfor = ['116 bp', '357 bp', '398 bp']
    plotfacetsearch(df, path, name, plotextension, 'Experiment Directory', 'Contour Lengths', searchfor)



    # # Plot all Contour Length Histograms
    # plotAllContourLengthHistograms(file_name)

    # # Plot all Contour Length Histograms
    # d = {i: pd.DataFrame() for i in sorted(expmts)}
    # for i in sorted(expmts):
    #     d[i] = df[df.values == i]
    #
    # palette = sns.color_palette('BuPu')
    # new_palette = itertools.cycle(palette)
    #
    # savename = os.path.join(os.path.join(path, 'Plots', 'dist.png'))
    # for i, df in d.iteritems():
    #     print i
    #     color = next(new_palette)
    #     # Plot histogram
    #     plotkdei(df, path, name, plotextension, 'Experiment Directory', 'Contour Lengths', i, 'black')

    # # T test testing
    # df116 = df[df['Experiment Directory'] == '116 bp']
    # df210 = df[df['Experiment Directory'] == '210 bp']
    # df256 = df[df['Experiment Directory'] == '256 bp']
    # df339 = df[df['Experiment Directory'] == '339 bp']
    # df357 = df[df['Experiment Directory'] == '357 bp']
    # df398 = df[df['Experiment Directory'] == '398 bp']
    # df116 = df116['Contour Lengths']
    # df210 = df210['Contour Lengths']
    # df256 = df256['Contour Lengths']
    # df339 = df339['Contour Lengths']
    # df357 = df357['Contour Lengths']
    # df398 = df398['Contour Lengths']
    # scipy.stats.ttest_ind(df116, df210, equal_var=False)


    # # Analysing two data sets together
    # path = '/Volumes/GoogleDrive/My Drive/AFM research group /Methods paper/Data/Bea'
    # file_name = os.path.join(path, name)
    # df = pd.read_json(file_name)
    # df['DNA Length'] = df['Experiment Directory']
    # df = df.replace(["Bea"], "Origami")

    # path2 = '/Volumes/GoogleDrive/My Drive/AFM research group /Methods paper/Data/MAC'
    # file_name2 = os.path.join(path2, name)
    # df2 = pd.read_json(file_name2)
    # df2['DNA Length'] = df2['Experiment Directory']
    # df3 = pd.concat([df, df2], axis=0)

    # df3 = pd.concat([df, df2], axis=0)

    # def plotviolin(df, directory, name, plotextension, grouparg, plotarg):
    #     print 'Plotting violin of %s' % plotarg
    #     # Create a saving name format/directory
    #     savedir = os.path.join(directory, 'Plots')
    #     if not os.path.exists(savedir):
    #         os.makedirs(savedir)
    #     # Plot and save figures
    #     savename = os.path.join(savedir, name + plotarg + '_violin' + plotextension)
    #     fig, ax = plt.subplots(figsize=(15, 5))
    #     # Plot violinplot
    #     ax = sns.violinplot(x=grouparg, y=plotarg, data=df)
    #     ax.invert_xaxis()
    #     plt.xlabel(' ')
    #     plt.ylabel(' ')
    #     plt.savefig(savename)

    # sns.set_palette(sns.color_palette('BuPu',2 ))
    # plotviolin(df3, path, name, '_circularviolinboth.pdf', 'Experiment Directory', 'Contour Lengths')