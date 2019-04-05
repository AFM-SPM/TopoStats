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
sns.set_context("poster", font_scale=1.5)


def traversedirectories(fileend, path):
    # This function finds all the files with the file ending set in the main script as fileend (usually.spm)
    # in the path directory, and all subfolder

    # initialise the list
    sprfiles = []
    # use os.walk to search folders and subfolders and append each file with the correct filetype to the list spmfiles
    for dirpath, subdirs, files in os.walk(path):
        # Looking for files ending in fileend
        for filename in files:
            if filename.endswith((fileend)):
                sprfiles.append(os.path.join(dirpath, filename))
    print 'Files found: ' + str(len(sprfiles))
    # return a list of files including their root and the original path specified
    return sprfiles


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


def plotline(df, directory, name, plotextension):
    print 'Plotting line of %s' % name

    # Create a saving name format/directory
    savedir = os.path.join(directory, 'Plots')
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    savename = os.path.splitext(file)[0] + plotextension

    # Plot and save figures
    savename = os.path.join(savedir, savename)
    fig, ax = plt.subplots(figsize=(10, 7))
    df.plot.line(ax=ax, legend=True, alpha=0.7)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(loc='upper right', ncol=2, fontsize='x-small')
    plt.xlim(-20, 1200)
    plt.ylim(-20, 400)
    plt.xlabel(' ')
    plt.ylabel(' ')
    plt.savefig(savename)


# This the main script
if __name__ == '__main__':
    # Set the file path, i.e. the directory where the files are here'
    # path = '/Users/alice/Dropbox/UCL/DNA MiniCircles/Minicircle Data Edited/Minicircle Manuscript/PLL NaOAc'
    path = '/Users/alice/Dropbox/UCL/DNA MiniCircles/Minicircle Data Edited/SPR'
    # path = '/Users/alice/Dropbox/UCL/DNA MiniCircles/Minicircle Data Edited/DNA/339/Nickel'

    # Set the name of the json file to import here
    name = '339_SC'
    plotextension = '.pdf'
    bins = 10
    fileend = 'Kinetics.json'
    maxax = 300

    # Convert excel files to JSON format
    sprfiles = traversedirectories('Kinetics.xlsx', path)
    for file in sprfiles:
        df = pd.read_excel(file)
        df = df.dropna()
        df = df.set_index('Time')
        savestats(path, file, df)
        plotline(df, path, file, plotextension)
        df.to_json(file + '.json')
