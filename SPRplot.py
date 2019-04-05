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
sns.set_context("poster", font_scale=0.8)

def importfromjson(path, name):
    filename = os.path.join(path, name + '.json')
    importeddata = pd.read_json(filename)

    return importeddata

def savestats(path, name, dataframetosave):
    print 'Saving JSON file for: ' + str(name)

    savename = os.path.join(path, excel_file)
    if not os.path.exists(path):
        os.makedirs(path)
    dataframetosave.to_json(savename + '.json')

def plotline(df, directory, name, plotextension):
    print 'Plotting line of %s' % excel_file

    # Create a saving name format/directory
    savedir = os.path.join(directory, 'Plots')
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # Plot and save figures
    savename = os.path.join(savedir, name + plotextension)
    fig, ax = plt.subplots(figsize=(10, 7))
    df.plot.line(ax=ax, legend=True, alpha=0.7)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(title='Topoisomer', loc='upper left', ncol=2, fontsize='small')
    plt.xlim(0, 1200)
    plt.ylim(-50, 500)
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

    excel_file = '339_SC.xlsx'
    df = pd.read_excel(os.path.join(path, excel_file))
    df = df.set_index('Time')
    df = df.dropna()

    plotline(df, path, name, plotextension)


    savestats(path, name, df)