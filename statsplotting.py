#!/usr/bin/env python2

import os
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import seaborn as sns
import numpy as np
import scipy
from scipy import stats

# Set seaborn to override matplotlib for plot output
sns.set()
sns.set_style("white", {'font.family': ['sans-serif']})
# The four preset contexts, in order of relative size, are paper, notebook, talk, and poster.
# The notebook style is the default
# sns.set_context("notebook", font_scale=1.5)
sns.set_context("poster", font_scale=1.4)
# plt.style.use("dark_background")


def plotHistogramOfTwoDataSets(data_frame_path, dataset_1_name, dataset_2_name):
    pass


def plotAllContourLengthHistograms(json_path):

    contour_lengths_df = pd.read_json(json_path)
    sns.set()

    nbins = np.linspace(-10,10,30)

    project_names = set(contour_lengths_df['Experiment Directory'].array)
    #print(project_names)

    for name in project_names:


        temp_df = contour_lengths_df.loc[contour_lengths_df['Experiment Directory'] == name ]

        #plt.hist(temp_df['Contour Lengths'], 20, histtype= 'bar', label = name)

        try:
            hist_data.append(temp_df['Contour Lengths'])
        except NameError:
            hist_data = [temp_df['Contour Lengths']]

    plt.hist(hist_data, 10, histtype= 'bar', label = project_names)
            #sns.distplot(temp_df['Contour Lengths'])
    plt.legend(loc='upper right')
    plt.xlabel('Contour Length (nm)')
    plt.ylabel('Occurence')

    save_file_name = json_path.split('/')

    plt.savefig('%s.png' % json_path[:-4])
    plt.savefig('%s.svg' % json_path[:-4])
    plt.close()

def plotLinearVsCircular(json_path):
    sns.set(style = 'whitegrid')
    pal = sns.color_palette()

    contour_lengths_df = pd.read_json(json_path)

    linear_contour_lengths = contour_lengths_df.loc[contour_lengths_df['Circular'] == False]
    circular_contour_lengths = contour_lengths_df.loc[contour_lengths_df['Circular'] == True]

    plt.hist([circular_contour_lengths['Contour Lengths'].array, linear_contour_lengths['Contour Lengths'].array], 25, histtype = 'bar', label = ['Linear Molecules', 'Circular Molecules'])
    plt.xlabel('Contour Length (nm)')
    plt.ylabel('Occurence')
    plt.legend(loc='upper right')
    plt.title('%s Linear vs Circular' % json_path[:-4])
    plt.savefig('%s_linearVcircularHist.png' % json_path[:-4])
    plt.savefig('%s_linearVcircularHist.svg' % json_path[:-4])
    plt.close()

    num_lin_circ_df = pd.DataFrame(data = {'Linear' : [len(circular_contour_lengths)], 'Circular' : [len(linear_contour_lengths)]})

    sns.barplot(data = num_lin_circ_df, order = ['Linear', 'Circular'])
    plt.xlabel('Linear or Circular')
    plt.ylabel('Occurence')
    plt.title('%s Linear vs Circular' % json_path[:-4])
    plt.savefig('%s_barplot.png' % json_path[:-4])
    plt.savefig('%s_barplot.svg' % json_path[:-4])
    plt.close()


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
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    savename = os.path.join(savedir, name + plotarg + plotextension)

    # Plot and save figures
    fig, ax = plt.subplots(figsize=(10, 7))
    df.groupby(grouparg)[plotarg].plot.kde(ax=ax, legend=True, alpha=1, linewidth=7.0)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels), title='Topoisomer', loc='upper left')
    plt.xlim(0, 1.2)
    # plt.xlim(1.2e-8, 2.5e-8)
    plt.xlabel(' ')
    plt.ylabel(' ')
    plt.savefig(savename)


def plothist2(df, directory, name, plotextension, grouparg, plotarg):
    print 'Plotting histogram of %s' % plotarg

    # Create a saving name format/directory
    savedir = os.path.join(directory, 'Plots')
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # Plot and save figures
    savename = os.path.join(savedir, name + plotarg + '_histogram' + plotextension)

    fig, ax = plt.subplots(figsize=(10, 7))
    df.groupby(grouparg)[plotarg].plot.hist(ax=ax, legend=True, range=(0,1), bins=bins, alpha=0.3, stacked=True)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels), title='Topoisomer', loc='upper left')
    plt.xlim(0, 1)
    plt.xlabel(' ')
    plt.ylabel(' ')
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
    df1.plot.hist(ax=ax, legend=True, bins=bins, alpha=.3, stacked=True)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels), title='Topoisomer', loc='upper left')
    plt.xlim(0, 1)
    plt.xlabel(' ')
    plt.ylabel(' ')
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
    ax = sns.violinplot(x=grouparg, y=plotarg, data=df)
    ax.invert_xaxis()
    # plt.xlim(0, 1)
    plt.xlabel(' ')
    plt.ylabel(' ')
    plt.savefig(savename)


def plotmarginal(df, directory, name, plotextension, grouparg, plotarg):
    print 'Plotting contour of %s' % plotarg

    # Create a saving name format/directory
    savedir = os.path.join(directory, 'Plots')
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # Plot and save figures
    savename = os.path.join(savedir, name + plotarg + '_marg' + plotextension)
    fig, ax = plt.subplots(figsize=(10, 7))
    # Plot violinplot
    ax = sns.jointplot(x=df[grouparg], y=df[plotarg], kind='kde', color="skyblue")
    # ax = sns.violinplot(x=grouparg, y=plotarg, data=df)
    # ax.invert_xaxis()
    # plt.xlim(0, 1)
    # plt.xlabel(' ')
    plt.ylabel(' ')
    plt.savefig(savename)


def plotkdemax(df, directory, name, plotextension, plotarg, topos):
    print 'Plotting kde and maxima for %s' % plotarg

    # sns.set_context("notebook")

    # Create a saving name format/directory
    savedir = os.path.join(directory, 'Plots')
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # Plot and save figures
    savename = os.path.join(savedir, name + plotarg + '_KDE_new' + plotextension)

    # Determine KDE for each topoisomer
    # Determine max of each KDE and plot
    xs = np.linspace(0, 1, 10)
    kdemax = dict()
    dfstd = dict()
    dfvar = dict()
    dfste = dict()
    # plt.figure()
    for i in sorted(topos, reverse=True):
        kdemax[i] = i
        dfstd[i] = i
        dfvar[i] = i
        dfste[i] = i
        x = df.query('topoisomer == @i')[plotarg]
        a = scipy.stats.gaussian_kde(x)
        b = a.pdf(xs)
        dfstd[i] = np.std(x)
        dfstd[i] = x.std()
        dfvar[i] = np.var(x)
        dfste[i] = stats.sem
        # plt.plot(xs, b)
        kdemax[i] = xs[np.argmax(b)]
    # plt.savefig(savename)

    print kdemax

    savename2 = os.path.join(savedir, name + plotarg + '_KDE_max_var_reversed' + plotextension)
    # palette = sns.color_palette('YlOrRd', n_colors=len(topos))
    palette.reverse()
    with palette:
        fig = plt.figure(figsize=(10, 7))
        # plt.xlabel('Topoisomer')
        plt.ylabel(' ')
        # Set an arbitrary value to plot to in x, increasing by one each loop iteration
        order = 0
        # Set a value for the placement of the bars, by creating an array of the length of topos
        bars = np.linspace(0, len(topos), len(topos), endpoint=False, dtype=int)
        for i in sorted(topos, reverse=True):
            plt.bar(order, kdemax[i], yerr=dfstd[i], alpha=1)
            # plt.bar(order, kdemax[i], yerr=dfstd[i], alpha=0.7)
            # plt.bar(order, kdemax[i], yerr=dfvar[i], alpha=0.7)
            order = order + 1
            # Set the bar names to be the topoisomer names
            plt.xticks(bars, sorted(topos, reverse=True))
        plt.savefig(savename2)
#
    # savename3 = os.path.join(savedir, name + plotarg + '_KDE_max_var' + plotextension)
    # fig = plt.figure(figsize=(10, 7))
    # # sns.set_palette("tab10")
    # # plt.xlabel('Topoisomer')
    # plt.ylabel(' ')
    # # Set an arbitrary value to plot to in x, increasing by one each loop iteration
    # order = 0
    # # Set a value for the placement of the bars, by creating an array of the length of topos
    # bars = np.linspace(0, len(topos), len(topos), endpoint=False, dtype=int)
    # for i in sorted(topos, reverse=False):
    #     plt.bar(order, kdemax[i], yerr=dfvar[i], alpha=0.7)
    #     order = order + 1
    #     plt.xticks(bars, topos)
    # plt.savefig(savename3)


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
    path = '/Users/alicepyne/Dropbox/UCL/DNA MiniCircles/Minicircle Data Edited/Minicircle Manuscript/Nickel'

    # Set the name of the json file to import here
    name = '*.json'
    name = 'Nickel'
    plotextension = '.pdf'
    bins = 10

    # import data form the json file specified as a dataframe
    df = importfromjson(path, name)

    # Rename directory column as topoisomer
    df = df.rename(columns={"directory": "topoisomer"})
    # df = df.rename(columns={'grain_min_bound_size': 'width', 'grain_max_bound_size': 'length'})

    # Calculate the aspect ratio for each grain
    df['aspectratio'] = df['grain_min_bound_size'] / df['grain_max_bound_size']

    # Get list of unique directory names i.e. topoisomers
    topos = df['topoisomer'].unique()
    topos = sorted(topos, reverse=False)

    # Generate a new smaller df from the original df containing only selected columns
    dfaspectratio = df[['topoisomer', 'aspectratio']]

    # Convert original (rounded) delta Lk to correct delta Lk
    dfnew = df
    dfnew['topoisomer'] = df['topoisomer'].astype(str).replace({'-2': '-1.8', '-3': '-2.8', '-6': '-4.9'})
    # Get list of unique directory names i.e. topoisomers
    newtopos = dfnew['topoisomer']
    newtopos = pd.to_numeric(newtopos)
    dfnew['topoisomer'] = newtopos

    # Obtain list of unique topoisomers
    topos = df['topoisomer'].unique()
    topos = sorted(topos, reverse=False)

    # Get statistics for different topoisoimers
    allstats = df.groupby('topoisomer').describe()
    # transpose allstats dataframe to get better saving output
    allstats1 = allstats.transpose()
    # Save out statistics file
    savestats(path, name, allstats1)

    # Set palette for all plots with length number of topoisomers and reverse
    palette = sns.color_palette('BuPu', n_colors=len(topos))
    # palette = sns.color_palette('tab10', n_colors=len(topos))
    palette.reverse()
    with palette:
        # Plot a KDE plot of one column of the dataframe - arg1 e.g. 'aspectratio'
        # grouped by grouparg e.g. 'topoisomer'
        plotkde(df, path, name, plotextension, 'topoisomer', 'aspectratio')
        # plotkde(df, path, name, plotextension, 'topoisomer', 'grain_mean_radius')

        # # Plot a KDE plot of one column of the dataframe - arg1 e.g. 'aspectratio'
        # # grouped by grouparg e.g. 'topoisomer'
        # # Then plot the maxima of each KDE as a bar plot
        plotkdemax(df, path, name, plotextension, 'aspectratio', topos)

        plotviolin(df, path, name, plotextension, 'topoisomer', 'aspectratio')

        plotmarginal(df, path, name, plotextension, 'topoisomer', 'aspectratio')

        # # Plot a histogram of one column of the dataframe - arg1 e.g. 'aspectratio'
        # # grouped by grouparg e.g. 'topoisomer'
        # plothist2(df, path, name, plotextension, 'topoisomer', 'aspectratio')
        #
        # # Plot a histogram of one column of the dataframe - arg1 e.g. 'aspectratio'
        # # grouped by grouparg e.g. 'topoisomer'
        # plothiststacked2(df, path, name, plotextension, 'topoisomer', 'aspectratio')

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

        # # # # Plot bivariate plot using seaborn
        # sns.kdeplot(df.query("topoisomer == '-6'")['grain_max_bound_size'],
        #             df.query("topoisomer == '-6'")['grain_min_bound_size'], n_levels=15, shade=True)

        # # Use seaborn to setup KDE apsect ratio plots for each unique topoisomer on the same page stacked as columns
        # h = sns.FacetGrid(df, col="topoisomer")
        # h.map(sns.kdeplot, "aspectratio")

        # Use seaborn to plot a KDE for each topoisomer separately on the same page, stacked by row
        # ordered_topos = df.topoisomer.value_counts().index
        # ordered_topos = sorted(ordered_topos, reverse=True)
        # g = sns.FacetGrid(df, row="topoisomer", row_order=ordered_topos,
        #                   height=1.7, aspect=4)
        # g.map(sns.distplot, "aspectratio", hist=False, rug=True);
        # plt.xlim(0,1)
        #
        # # Create scatter plot of two variables in seaborn to show correlation
        # h = sns.FacetGrid(df, col="topoisomer")
        # h.map(plt.scatter, "grain_min_bound_size", "grain_max_bound_size", alpha=.7)
        # plt.xlim(0e-7, 0.7e-7)
        # plt.ylim(0e-7, 1.5e-7)
        #
        # # Create bivariate scatter plot of two variables with shading
        # fig, ax = plt.subplots(figsize=(5, 3))
        # sns.kdeplot(df.query("topoisomer == '0'")['grain_max_bound_size'],
        #             df.query("topoisomer == '0'")['grain_min_bound_size'], n_levels=15, shade=True)
        # plt.xlim(4e-8, 8e-8)
        # plt.ylim(2e-8, 6e-8)

        # g = sns.PairGrid(df, vars=['grain_max_bound_size', 'grain_min_bound_size', 'aspectratio'], hue="topoisomer")
        # g.map_diag(sns.kdeplot)
        # g.map_lower(sns.kdeplot)
        # g.map_upper(plt.scatter)

        def plotviolin(df, directory, name, plotextension, grouparg, plotarg):
            print 'Plotting violin of %s' % plotarg

            # Create a saving name format/directory
            savedir = os.path.join(directory, 'Plots')
            if not os.path.exists(savedir):
                os.makedirs(savedir)

            df['topoisomer'] = df['topoisomer'].astype(np.int32)

            # Plot and save figures
            savename = os.path.join(savedir, name + plotarg + '_violin' + plotextension)
            fig, ax = plt.subplots(figsize=(10, 7))
            # Plot violinplot
            ax = sns.violinplot(x=grouparg, y=plotarg, data=df)
            ax.invert_xaxis()
            # plt.xlim(0, 1)
            plt.xlabel(' ')
            plt.ylabel(' ')
            plt.savefig(savename)

# # # import pandas as pd
# # # import os
# # # import matplotlib as plt
# where ='/Users/alicepyne/Dropbox/UCL/DNA MiniCircles/Paper/Pyne et al/SI Violin'
# RGEX=pd.read_csv(os.path.join(where,'radgyrEX.csv'))
# # RGEX.columns = ['-6','-3','-2','-1','0']
# RGIM=pd.read_csv(os.path.join(where,'radgyrIM.csv'))
# # RGIM.columns = ['-6','-3','-2','-1','0']
# WREX=pd.read_csv(os.path.join(where,'writheEX.csv'))
# # WREX.columns = ['-6','-3','-2','-1','0']
# WRIM=pd.read_csv(os.path.join(where,'writheIM.csv'))
# # WRIM.columns = ['-6','-3','-2','-1','0']
# RGEX = RGEX.melt(var_name='groups', value_name='vals')
# # RGEX['groups'] = RGEX['groups'].astype(np.int32)
# RGIM = RGIM.melt(var_name='groups', value_name='vals')
# # RGIM['groups'] = RGIM['groups'].astype(np.int32)
# WREX = WREX.melt(var_name='groups', value_name='vals')
# # WREX['groups'] = WREX['groups'].astype(np.int32)
# WRIM = WRIM.melt(var_name='groups', value_name='vals')
# # WRIM['groups'] = WRIM['groups'].astype(np.int32)
# sns.set_palette(sns.color_palette('BuPu',5))
# savename = os.path.join(where, 'RGEX_violin' + plotextension)
# fig, ax = plt.subplots(figsize=(10, 7))
# ax = sns.violinplot(x="groups", y="vals", data=RGEX)
# ax.invert_xaxis()
# plt.xlabel(' ')
# plt.ylim(8, 18.5)
# plt.ylabel(' ')
# plt.savefig(savename)
# savename = os.path.join(where, 'RGIM_violin' + plotextension)
# fig, ax = plt.subplots(figsize=(10, 7))
# ax = sns.violinplot(x="groups", y="vals", data=RGIM)
# ax.invert_xaxis()
# plt.xlabel(' ')
# plt.ylim(8, 18.5)
# plt.ylabel(' ')
# plt.savefig(savename)
# savename = os.path.join(where, 'WREX_violin' + plotextension)
# fig, ax = plt.subplots(figsize=(10, 7))
# ax = sns.violinplot(x="groups", y="vals", data=WREX)
# ax.invert_xaxis()
# plt.xlabel(' ')
# plt.ylim(-5, 0.5)
# plt.ylabel(' ')
# plt.savefig(savename)
# savename = os.path.join(where, 'WRIM_violin' + plotextension)
# fig, ax = plt.subplots(figsize=(10, 7))
# ax = sns.violinplot(x="groups", y="vals", data=WRIM)
# ax.invert_xaxis()
# plt.xlabel(' ')
# plt.ylim(-5, 0.5)
# plt.ylabel(' ')
# plt.savefig(savename)
# savename = os.path.join(where, 'RG_violin' + plotextension)
# fig, ax = plt.subplots(figsize=(10, 7))
# with sns.color_palette("Blues"):
#     ax = sns.violinplot(x="groups", y="vals", data=RGIM, label='Implicit')
#     ax.invert_xaxis()
# with sns.color_palette("Reds"):
#     ax = sns.violinplot(x="groups", y="vals", data=RGEX, label='Explicit')
#     ax.invert_xaxis()
# plt.xlabel(' ')
# plt.ylim(8,18.5)
# plt.ylabel(' ')
# plt.savefig(savename)
# savename = os.path.join(where, 'WR_violin' + plotextension)
# fig, ax = plt.subplots(figsize=(10, 7))
# with sns.color_palette("Blues"):
#     ax = sns.violinplot(x="groups", y="vals", data=WRIM, label='Implicit')
#     ax.invert_xaxis()
# with sns.color_palette("Reds"):
#     ax = sns.violinplot(x="groups", y="vals", data=WREX, label='Explicit')
#     ax.invert_xaxis()
# plt.xlabel(' ')
# plt.ylim(-5, 1)
# plt.ylabel(' ')
# plt.savefig(savename)
#
# savename = os.path.join(where, 'EX_violin' + plotextension)
# fig, ax = plt.subplots(figsize=(10, 7))
# with sns.color_palette("Blues"):
#     ax1 = sns.violinplot(x="groups", y="vals", data=WREX, label='Writhe', ax=ax, alpha=0.7)
#     # ax.invert_xaxis()
#     plt.xlabel(' ')
#     plt.ylabel(' ')
#     plt.ylim(-8,0.5)
# ax2 = plt.twinx()
# with sns.color_palette("Reds"):
#     ax2 = sns.violinplot(x="groups", y="vals", data=RGEX, label='Rg', ax=ax2, alpha=0.7)
#     # ax.invert_xaxis()
#     plt.xlabel(' ')
#     plt.ylabel(' ')
# plt.ylim(8,20)
# plt.xlabel(' ')
# # plt.ylim(-5, 0.5)
# plt.ylabel(' ')
# plt.savefig(savename)
# savename = os.path.join(where, 'IM_violin' + plotextension)
# fig, ax = plt.subplots(figsize=(10, 7))
# with sns.color_palette("Blues"):
#     ax1 = sns.violinplot(x="groups", y="vals", data=WRIM, label='Writhe', ax=ax, alpha=0.7)
#     ax.invert_xaxis()
#     plt.ylim(-8,0.5)
# ax2 = plt.twinx()
# with sns.color_palette("Reds"):
#     ax2 = sns.violinplot(x="groups", y="vals", data=RGIM, label='Rg', ax=ax2, alpha=0.7)
#     ax.invert_xaxis()
# plt.ylim(8,20)
# plt.xlabel(' ')
# # plt.ylim(-5, 0.5)
# plt.ylabel(' ')
# plt.savefig(savename)
#
#
# savename = os.path.join(where, 'RGEX_violin_sq' + plotextension)
# fig, ax = plt.subplots(figsize=(7, 7))
# ax = sns.violinplot(x="groups", y="vals", data=RGEX)
# ax.invert_xaxis()
# plt.xlabel(' ')
# plt.ylim(8, 18.5)
# plt.ylabel(' ')
# plt.savefig(savename)
#
#
# savename = os.path.join(where, 'WREX_violin_sq' + plotextension)
# fig, ax = plt.subplots(figsize=(7, 7))
# ax = sns.violinplot(x="groups", y="vals", data=WREX)
# ax.invert_xaxis()
# plt.xlabel(' ')
# plt.ylim(-5, 0.5)
# plt.ylabel(' ')
# plt.savefig(savename)
