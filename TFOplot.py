import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from cycler import cycler

# Set seaborn to override matplotlib for plot output
sns.set()
# The four preset contexts, in order of relative size, are paper, notebook, talk, and poster.
# The notebook style is the default
sns.set_context("poster")

def importfromjson(path, name):
    filename = os.path.join(path, name + '.json')
    importeddata = pd.read_json(filename)

    return importeddata

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
        TFOstd[i] = b
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
path = '/Users/alice/Dropbox/UCL/DNA MiniCircles/Minicircle Data Edited/TFO/TFOlengthanalysis'
# Set the name of the json file to import here
name = 'TFOlengthanalysis'
plotextension = '.pdf'
bins = 8
allbins = 20

# # import data form the csv file specified as a dataframe
# TFO = pd.read_csv(os.path.join(path, '339LINTFOlengthanalysis.csv'))

# import data from a JSON file
TFO = importfromjson(path, name)

# rename columns to sensible names
TFO['Obj'] = TFO['Obj'].apply(str)
TFO['Obj'] = TFO['Obj'].replace({'1': 'full length', '2': 'short side', '3': 'long side', '4': 'TFO'})

# Get details of different length types
lengths = TFO['Obj'].unique()
lengths = sorted(lengths)

plot_mean_SD(TFO, path, name, plotextension, 'length (nm)', lengths)

plothist(TFO, 'length (nm)', 'Obj', path, plotextension)

# plotkde(TFO, path, name, plotextension, 'Obj', 'length (nm)')
