import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from cycler import cycler

# Set seaborn to override matplotlib for plot output
sns.set()
sns.set_style("white")
# The four preset contexts, in order of relative size, are paper, notebook, talk, and poster.
# The notebook style is the default
sns.set_context("poster", font_scale=1.1)

def importfromjson(path, name):
    filename = os.path.join(path, name + '.json')
    importeddata = pd.read_json(filename)

    return importeddata


def savestats(path, name, dataframetosave):
    print 'Saving JSON file for: ' + str(name)

    savename = os.path.splitext(file)[0]
    if not os.path.exists(path):
        os.makedirs(path)
    dataframetosave.to_json(savename + '.json')


def plotkde(df, directory, name, plotextension, grouparg, plotarg):
    print 'Plotting kde of %s' % plotarg

    # Create a saving name format/directory
    savedir = os.path.join(directory, 'Plots')
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # Plot and save figures
    savename = os.path.join(savedir, name + plotarg + plotextension)
    fig, ax = plt.subplots(figsize=(10, 7))
    df.groupby(grouparg)[plotarg].plot.kde(ax=ax, legend=True, alpha=0.7)
    plt.savefig(savename)

def plotscatter(df, directory, name, plotextension):
    print 'Plotting scatter'

    # Create a saving name format/directory
    savedir = os.path.join(directory, 'Plots')
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # Plot and save figures
    savename = os.path.join(savedir, name + plotextension)
    sns.pairplot(x_vars=["Method"], y_vars=["Angle"], data=df, hue="Type", size=5)
    plt.savefig(savename)


def plot_mean_SD(df, directory, name, plotextension, plotarg, lengths):
    print 'Plotting mean and SD for %s' % plotarg
    # Create a saving name format/directory
    savedir = os.path.join(directory, 'Plots')
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # Plot and save figures
    savename = os.path.join(savedir, name + plotarg + '_mean_std' + plotextension)

    # Determine mean and std for each measurement and plot on bar chart
    TFO = df
    TFOmean = dict()
    TFOstd = dict()
    fig = plt.figure(figsize=(10, 7))
    plt.ylabel(' ')
    color = iter(['blue', 'limegreen', 'mediumvioletred', 'darkturquoise'])
    for i in sorted(lengths, reverse=False):
        TFOmean[i] = i
        TFOstd[i] = i
        x = TFO.query('Obj == @i')[plotarg]
        a = x.mean()
        b = x.std()
        TFOmean[i] = a
        print a
        TFOstd[i] = b
        print b
        c = next(color)
        plt.bar(i, TFOmean[i], yerr=TFOstd[i], alpha=0.7, color=c)
    plt.savefig(savename)


def plothist(df, arg1, grouparg, directory, extension):
    print 'Plotting graph of %s' % (arg1)

    # Create a saving name format/directory
    savedir = os.path.join(directory, 'Plots')
    savename = os.path.join(savedir, os.path.splitext(os.path.basename(directory))[0])
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # Plot arg1 using MatPlotLib separated by the grouparg
    # Plot with figure with stacking sorted by grouparg
    # Plot histogram
    bins = np.arange(0, 130, 3)
    fig = plt.figure(figsize=(10, 7))
    new_prop_cycle = cycler('color', ['blue', 'limegreen', 'mediumvioletred', 'darkturquoise'])
    plt.rc('axes', prop_cycle=new_prop_cycle)
    df.groupby(grouparg)[arg1].plot.hist(legend=True, alpha=0.7, bins=bins)
    plt.ylabel(' ')
    # Save plot
    plt.savefig(savename + '_' + arg1 + '_a' + extension)


# Set the file path, i.e. the directory where the files are here'
path = '/Users/alice/Dropbox/UCL/DNA MiniCircles/Paper/Pyne et al/SI Angles'
# Set the name of the json file to import here
name = 'AnglesV2'
plotextension = '.pdf'
bins = 8
allbins = 20

# # import data from the csv file specified as a dataframe
# angles = pd.read_csv(os.path.join(path, 'AnglesV2short.csv'))
angleslong = pd.read_csv(os.path.join(path, 'AnglesV2.csv'))

df = angleslong.dropna()

details = df.groupby('Type').describe()

plotscatter(df, path, name, plotextension)
