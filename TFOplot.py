import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set seaborn to override matplotlib for plot output
sns.set()
# The four preset contexts, in order of relative size, are paper, notebook, talk, and poster.
# The notebook style is the default
sns.set_context("talk")

def importfromjson(path, name):
    filename = os.path.join(path, name + '.json')
    importeddata = pd.read_json(filename)

    return importeddata


# Set the file path, i.e. the directory where the files are here'
path = '/Users/alice/Dropbox/UCL/DNA MiniCircles/Minicircle Data Edited/TFO/TFOlengthanalysis'
# Set the name of the json file to import here
name = 'TFOlengthanalysis'
plotextension = '.pdf'
bins = 8

# # import data form the csv file specified as a dataframe
# TFO = pd.read_csv(os.path.join(path, '339LINTFOlengthanalysis.csv'))

# import data from a JSON file
TFO = importfromjson(path, name)

# Get details of different length types
lengths = TFO['Obj'].unique()
sorted(lengths)

# Determine mean and std for measurements
TFO1mean = TFO.query('Obj == 1').loc[:,'length (nm)'].mean()
TFO1std = TFO.query('Obj == 1').loc[:,'length (nm)'].std()
TFO2mean = TFO.query('Obj == 2').loc[:,'length (nm)'].mean()
TFO2std = TFO.query('Obj == 2').loc[:,'length (nm)'].std()
TFO3mean = TFO.query('Obj == 3').loc[:,'length (nm)'].mean()
TFO3std = TFO.query('Obj == 3').loc[:,'length (nm)'].std()
TFO4mean = TFO.query('Obj == 4').loc[:,'length (nm)'].mean()
TFO4std = TFO.query('Obj == 4').loc[:,'length (nm)'].std()

# TFO plotting
savename = os.path.join(path, name + '_1' + plotextension)
fig, ax = plt.subplots()
TFO.query('Obj == 1')['length (nm)'].plot.hist(ax=ax, bins=range(104, 116), color='limegreen')
plt.savefig(savename)

savename = os.path.join(path, name + '_2' + plotextension)
fig, ax = plt.subplots()
TFO.query('Obj == 2')['length (nm)'].plot.hist(ax=ax, bins=range(32, 44), color='darkturquoise')
plt.savefig(savename)

savename = os.path.join(path, name + '_3' + plotextension)
fig, ax = plt.subplots()
TFO.query('Obj == 3')['length (nm)'].plot.hist(ax=ax, bins=range(58, 70), color='mediumvioletred')
plt.savefig(savename)

savename = os.path.join(path, name + '_4' + plotextension)
fig, ax = plt.subplots()
TFO.query('Obj == 4')['length (nm)'].plot.hist(ax=ax, bins=range(0, 12), color='blue')
plt.savefig(savename)
