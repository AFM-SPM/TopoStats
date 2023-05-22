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
from cycler import cycler
from pathlib import Path
from topostats.io import read_yaml

plotting_config = read_yaml(Path("../config/jean_plotting_config.yml"))

# Need to define extension for tests to pass
extension = ".png"

# Set seaborn to override matplotlib for plot output
sns.set()
sns.set_style("white", {"font.family": ["sans-serif"]})
# The four preset contexts, in order of relative size, are paper, notebook, talk, and poster.
# The notebook style is the default
# sns.set_context("notebook", font_scale=1.5)
sns.set_context("poster", font_scale=1.4)
# plt.style.use("dark_background")
sns.set_palette(sns.color_palette("bright"))

colname2label = {
    "grain_bound_len": "Circumference / %s",
    "aspectratio": "Aspect Ratio",
    "grain_curvature1": "Smaller Curvature",
    "grain_curvature2": "Larger Curvature",
    "grain_ellipse_major": "Ellipse Major Axis Length / %s",
    "grain_ellipse_minor": "Ellipse Minor Axis Length / %s",
    "grain_half_height_area": "Area Above Half Height / $\mathregular{%s^2}$",
    "grain_maximum": "Maximum Height / %s",
    "grain_mean": "Mean Height / %s",
    "grain_median": "Median Height / %s",
    "grain_min_bound_size": "Width / %s",
    "grain_max_bound_size": "Length / %s",
    "grain_mean_radius": "Mean Radius / %s",
    "grain_pixel_area": "Area / Pixels",
    "grain_proj_area": "Area / $\mathregular{%s^2}$",
    "grain_min_volume": "Minimum Volume / $\mathregular{%s^3}$",
    "grain_zero_volume": "Zero Volume / $\mathregular{%s^3}$",
    "grain_laplace_volume": "Laplacian Volume / $\mathregular{%s^3}$",
    "end_to_end_distance": "End to End Distance / nm",
    "contour_lengths": "Contour Lengths / nm",
    "raidus_min": "Minimum Radius / nm",
    "radius_max": "Maximum Radius / nm",
    "radius_mean": "Mean Radius / nm",
    "radius_median": "Median Radius / nm",
    "height_min": "Minimum Height / nm",
    "height_max": "Maximum Height / nm",
    "height_median": "Median Height / nm",
    "height_mean": "Mean Height / nm",
    "volume": "Volume / $\mathregular{nm^3}$",
    "area": "Area / $\mathregular{nm^2}$",
    "area_cartesian_bbox": "Cartesian Bounding Box Area / $\mathregular{nm^2}$",
    "smallest_bounding_width": "Smallest Bounding Width / nm",
    "smallest_bounding_length": "Smallest Bounding Length / nm",
    "smallest_bounding_area": "Smallest Bounding Area / $\mathregular{nm^2}$",
    "aspect_ratio": "Aspect Ratio",
    "bending_angle": "Bending Angle / degrees",
}


def importfromfile(path):
    """Importing the data needed from the json or csv file specified by the user"""

    print(path)
    filename, filextension = os.path.splitext(path)
    if filextension == ".json":
        importeddata = pd.read_json(path)
        return importeddata
    elif filextension == ".csv":
        importeddata = pd.read_csv(path)
        return importeddata
    else:
        print("Unsupported file type")


def savestats(path, dataframetosave):
    print("Saving stats for: " + str(os.path.basename(path)[:-5]) + "_evaluated")
    dataframetosave.to_json(path[:-5] + "_evaluated.json")
    dataframetosave.to_csv(path[:-5] + "_evaluated.txt")


def pathman(path):
    """Splitting the path into directory and file name; creating or specifying a directory to save the plots"""

    directory = os.path.dirname(path)
    name = os.path.basename(path)[:-5]
    savedir = os.path.join(directory, "Plots")
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

    if "%s" in label:
        if nm is True:
            label = label % "nm"
        else:
            label = label % "m"
    elif "nm" in label:
        if nm is False:
            label = label.replace("nm", "m")

    return label


def dataunitconversion(data, plotarg, nm):
    """Converting the data based on the unit specified by the user. Only nm and m are supported at the moment."""

    if plotarg in colname2label:
        label = colname2label[plotarg]
    else:
        label = plotarg

    data_new = data
    if nm is True:
        if "%s" in label:
            if "^2" in label:
                data_new = data * 1e18
            elif "^3" in label:
                data_new = data * 1e27
            else:
                data_new = data * 1e9
    else:
        if "/ nm" in label:
            data_new = data * 1e-9
        elif "nm^2" in label:
            data_new = data * 1e-18
        elif "nm^3" in label:
            data_new = data * 1e-27

    return data_new


def plotkde(df, plotarg, grouparg=None, xmin=None, xmax=None, nm=False, specpath=None):
    """Creating a KDE plot for the chosen variable. Grouping optional. The x axis range can be defined by the user. The
    default unit is metre, but this can be changed to nanometre by adding 'nm=True'. The default path is the path under
    the if __name__ == '__main__' line, but this can also be changed using the specpath argument."""

    print("Plotting kde of %s" % plotarg)

    # Set the name of the file
    if specpath is None:
        specpath = path
    savename = os.path.join(pathman(specpath) + "_" + plotarg + "_KDE" + extension)

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
        dfnew.groupby(grouparg)[plotarg].plot.kde(ax=ax, legend=True, alpha=1, linewidth=7.0)
        handles, labels = ax.get_legend_handles_labels()
        # ax.legend(reversed(handles), reversed(labels), title=grouparg, loc='upper right')
        ax.legend(handles, labels, title=grouparg, loc="upper right")

    # Label plot and save figure
    plt.xlim(xmin, xmax)
    plt.xlabel(labelunitconversion(plotarg, nm), alpha=1)
    plt.ylabel("Probability Density", alpha=1)
    plt.ticklabel_format(axis="both", style="sci", scilimits=(-3, 3))
    # Need to return fig in order to test
    plt.savefig(savename)
    # return fig


def plotkde2var(
    df,
    plotarg,
    df2=None,
    plotarg2=None,
    label1=None,
    label2=None,
    xmin=None,
    xmax=None,
    nm=False,
    specpath=None,
    grouparg=None,  # Defined outside scope of function but required for testing
):
    """Creating a KDE plot for the chosen variable. Grouping optional. The x axis range can be defined by the user. The
    default unit is metre, but this can be changed to nanometre by adding 'nm=True'. The default path is the path under
    the if __name__ == '__main__' line, but this can also be changed using the specpath argument."""

    if df2 is None:
        df2 = df
    if plotarg2 is None:
        plotarg2 = plotarg

    print("Plotting kde of %s and %s" % (plotarg, plotarg2))

    # Set the name of the file
    if specpath is None:
        specpath = path
    savename = os.path.join(pathman(specpath) + "_" + plotarg + "_" + plotarg2 + "_KDE" + extension)

    dfnew = df.copy()
    dfnew2 = df2.copy()

    # Convert the unit of the data to nm if specified by the user
    dfnew[plotarg] = dataunitconversion(df[plotarg], plotarg, nm)
    dfnew2[plotarg2] = dataunitconversion(df[plotarg2], plotarg2, nm)

    # Plot figure
    fig, ax = plt.subplots(figsize=(15, 12))
    # Simple KDE plot

    # dfplot = pd.merge(df[plotarg], df2[plotarg2])
    # df = df[[plotarg, plotarg2]]
    # df.plot.kde(ax=ax, alpha=1, linewidth=7.0)

    dfnew[plotarg].plot.kde(ax=ax, alpha=1, linewidth=7.0)
    dfnew2[plotarg2].plot.kde(ax=ax, alpha=1, linewidth=7.0)

    # Label plot and save figure
    plt.xlim(xmin, xmax)
    # plt.xlabel(labelunitconversion(plotarg1, nm), alpha=1)
    plt.xlabel(labelunitconversion(plotarg, nm), alpha=1)
    plt.ylabel("Probability Density", alpha=1)
    plt.ticklabel_format(axis="both", style="sci", scilimits=(-3, 3))
    handles, labels = ax.get_legend_handles_labels()
    # ax.legend(reversed(handles), reversed(labels), , title=grouparg, loc='upper right')
    ax.legend(handles, labels, title=grouparg, loc="upper right")
    # Need to return fig in order to test
    # plt.savefig(savename)
    return fig


def plothist(df, plotarg, grouparg=None, xmin=None, xmax=None, bins=20, nm=False, specpath=None):
    """Creating a histogram for the chosen variable. Grouping optional. The x axis range can be defined by the user. The
    default unit is metre, but this can be changed to nanometre by adding 'nm=True'. The default path is the path under
    the if __name__ == '__main__' line, but this can also be changed using the specpath argument."""

    print("Plotting histogram of %s" % plotarg)

    # Set  the name of the file
    if specpath is None:
        specpath = path
    savename = os.path.join(pathman(specpath) + "_" + plotarg + "_histogram" + extension)

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
        dfnew.plot.hist(ax=ax, legend=True, bins=bins, alpha=1, linewidth=3.0, stacked=True)
        # dfnew.groupby(grouparg)[plotarg].plot.hist(ax=ax, legend=True, alpha=1, linewidth=7.0, bins=bins, stacked=True)
        handles, labels = ax.get_legend_handles_labels()
        # ax.legend(reversed(handles), reversed(labels), title=grouparg, loc='upper right')
        ax.legend(handles, labels, title=grouparg, loc="upper right")

    # Label plot and save figure
    plt.xlim(xmin, xmax)
    plt.xlabel(labelunitconversion(plotarg, nm), alpha=1)
    plt.ylabel("Count", alpha=1)
    plt.ticklabel_format(axis="both", style="sci", scilimits=(-3, 3))
    # Need to return fig in order to test
    plt.savefig(savename)
    # return fig


def plothist2var(
    df,
    plotarg,
    df2=None,
    df3=None,
    plotarg2=None,
    plotarg3=None,
    label1=None,
    label2=None,
    label3='Damaged with protein',
    xmin=None,
    xmax=None,
    color1='#B45F06',
    color2='#0B5394',
    color3='#52C932',
    nm=False,
    specpath=None,
    bins=12,  # Missing argument defined outside scope of function required within for testing
):
    """Creating a histogram for the chosen variable. Grouping optional. The x axis range can be defined by the user. The
    default unit is metre, but this can be changed to nanometre by adding 'nm=True'. The default path is the path under
    the if __name__ == '__main__' line, but this can also be changed using the specpath argument."""

    if label1 is None:
        label1 = plotarg
    if label2 is None:
        label2 = plotarg2
    if df2 is None:
        df2 = df
    if plotarg2 is None:
        plotarg2 = plotarg
    if plotarg3 is None:
        plotarg3 = plotarg

    print("Plotting histogram of %s and %s" % (label1, label2))

    # Set the name of the file
    if specpath is None:
        specpath = path
    savename = os.path.join(pathman(specpath) + "_" + label1 + "_" + label2 + "_histogram" + extension)

    # Convert the unit of the data to nm if specified by the user
    dfnew = df.copy()
    dfnew2 = df2.copy()
    dfnew[plotarg] = dataunitconversion(df[plotarg], plotarg, nm)
    dfnew2[plotarg2] = dataunitconversion(df2[plotarg2], plotarg2, nm)

    # Plot figure
    fig, ax = plt.subplots(figsize=(15, 12))
    dfnew[plotarg].plot.hist(ax=ax, alpha=0.5, linewidth=3.0, bins=bins, color=color1, density=True)
    dfnew2[plotarg2].plot.hist(ax=ax, alpha=0.5, linewidth=3.0, bins=bins, color=color2, density=True, histtype="barstacked")

    # For the third set of data
    if df3 is not None:
        dfnew3 = df3.copy()
        dfnew3[plotarg3] = dataunitconversion(df3[plotarg3], plotarg3, nm)
        dfnew3[plotarg3].plot.hist(ax=ax, alpha=0.5, linewidth=3.0, bins=bins, color=color3, density=True, histtype="barstacked")

    # Label plot and save figure
    plt.xlim(xmin, xmax)
    plt.xlabel(labelunitconversion(plotarg, nm), alpha=1)
    plt.ylabel("Probability Density", alpha=1)
    plt.ticklabel_format(axis="both", style="sci", scilimits=(-3, 3))
    handles, labels = ax.get_legend_handles_labels()
    if label3 is not None:
        ax.legend(labels=[label1, label2, label3])
    else:
        ax.legend(labels=[label1, label2])
    # Need to return fig in order to test
    plt.savefig(savename)
    return fig


def plotdist(df, plotarg, grouparg=None, xmin=None, xmax=None, bins=20, nm=False, specpath=None, plotname=None):

    """Creating a dist plot, which is the combination of a histogram and a KDE plot; doesn't support grouped plots
    yet"""
    # Commenting out caused an error as only one argument provided for the first %s, none for the second.
    # print("Plotting dist plot of %s and %s" % plotarg)

    # Set the name of the file
    if plotname is None:
        plotname = labelunitconversion(plotarg, nm)

    if specpath is None:
        specpath = path
    savename = os.path.join(pathman(specpath) + "_" + plotname + "_dist" + extension)

    # Convert the unit of the data to nm if specified by the user
    dfnew = df.copy()
    dfnew[plotarg] = dataunitconversion(df[plotarg], plotarg, nm)

    # Plot figure
    fig, ax = plt.subplots(figsize=(15, 12))
    sns.distplot(dfnew[plotarg], ax=ax, bins=bins)

    # Label plot and save figure
    plt.xlim(xmin, xmax)
    plt.xlabel(plotname)
    plt.ylabel("Probability Density", alpha=1)
    plt.ticklabel_format(axis="both", style="sci", scilimits=(-3, 3))
    ax.tick_params(direction="out", bottom=True, left=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Need to return fig in order to test
    # plt.savefig(savename)
    return fig


def plotdist2var(
    df,
    plotarg,
    plotarg2,
    df2=None,
    xmin=None,
    xmax=None,
    bins=20,
    nm=False,
    specpath=None,
    plotname=None,
    c1=None,
    c2=None,
    label1=None,
    label2=None,
    extension=".png",  # Defined globally and not within this functions scope, required for testing
):
    """Dist plot for 2 variables"""

    print("Plotting dist plot of %s and %s" % (plotarg, plotarg2))

    if plotname is None:
        plotname = plotarg + "_and_" + plotarg2

    # Set the name of the file
    if specpath is None:
        specpath = path
    savename = os.path.join(pathman(specpath) + "_" + plotname + "_dist" + extension)

    if df2 is None:
        df2 = df

    # Convert the unit of the data to nm if specified by the user
    dfnew = df.copy()
    dfnew2 = df2.copy()
    dfnew[plotarg] = dataunitconversion(df[plotarg], plotarg, nm)
    dfnew2[plotarg2] = dataunitconversion(df2[plotarg2], plotarg2, nm)

    # Plot figure
    fig, ax = plt.subplots(figsize=(15, 12))
    sns.distplot(dfnew[plotarg], ax=ax, bins=bins, color=c1)
    sns.distplot(dfnew2[plotarg2], ax=ax, bins=bins, color=c2)

    # Label plot and save figure
    plt.xlim(xmin, xmax)
    plt.xlabel(plotname)
    # plt.xlabel(labelunitconversion(plotarg, nm), alpha=1)
    plt.ylabel("Probability Density", alpha=1)
    plt.ticklabel_format(axis="both", style="sci", scilimits=(-3, 3))
    ax.tick_params(direction="out", bottom=True, left=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(labels=[label1, label2])
    # Need to return fig in order to test
    plt.savefig(savename)
    # return fig


def plotviolin(df, plotarg, grouparg=None, ymin=None, ymax=None, nm=False, specpath=None):
    """Creating a violin plot for the chosen variable. Grouping optional. The y axis range can be defined by the user.
    The default unit is metre, but this can be changed to nanometre by adding 'nm=True'. The default path is the path
    under the if __name__ == '__main__' line, but this can also be changed using the specpath argument."""

    print("Plotting violin of %s" % plotarg)

    # Set the name of the file
    if specpath is None:
        specpath = path
    savename = os.path.join(pathman(specpath) + "_" + plotarg + "_violin" + extension)

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
        ax = sns.violinplot(x=grouparg, y=plotarg, data=dfnew)
        ax.invert_xaxis()  # Useful for topoisomers with negative writhe

    # Label plot and save figure
    plt.ylim(ymin, ymax)
    plt.ylabel(labelunitconversion(plotarg, nm), alpha=1)
    plt.xlabel(grouparg)
    # Need to return fig in order to test
    # plt.savefig(savename)
    return fig


def plotjoint(df, arg1, arg2, xmin=None, xmax=None, ymin=None, ymax=None, nm=False, specpath=None):
    """Creating a joint plot for two chosen variables. The range for both axes can be defined by the user.
    The default unit is metre, but this can be changed to nanometre by adding 'nm=True'. The default path is the path
    under the if __name__ == '__main__' line, but this can also be changed using the specpath argument."""

    print("Plotting joint plot for %s and %s" % (arg1, arg2))

    # Set the name of the file
    if specpath is None:
        specpath = path
    savename = os.path.join(pathman(specpath) + "_" + arg1 + "_and_" + arg2 + extension)

    # Convert the unit of the data to nm if specified by the user
    dfnew = df.copy()
    dfnew[arg1] = dataunitconversion(df[arg1], arg1, nm)
    dfnew[arg2] = dataunitconversion(df[arg2], arg2, nm)

    # Plot data using seaborn
    fig, ax = plt.subplots(figsize=(15, 12))  # Need a fig to return for tests
    sns.jointplot(arg1, arg2, data=dfnew, kind="reg", height=15)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel(labelunitconversion(arg1, nm), alpha=1)
    plt.ylabel(labelunitconversion(arg2, nm), alpha=1)
    # Need to return fig in order to test
    # plt.savefig(savename)
    return fig


def plotLinearVsCircular(contour_lengths_df):
    pass


def computeStats(data, columns, min, max):
    """Prints out a table of stats, including the standard deviation, standard error, N value, and peak position"""

    xs = np.linspace(min, max, 1000)

    a = {}
    b = {}
    table = {
        "max": [0] * len(data),
        "std": [0] * len(data),
        "ste": [0] * len(data),
        "N": [0] * len(data),
    }
    for i, x in enumerate(data):
        if i != 0:
            x = x * 1e9
            a[i] = scipy.stats.gaussian_kde(x)
            b[i] = a[i].pdf(xs)
            table["std"][i] = np.std(x)
            table["ste"][i] = stats.sem(x)
            table["max"][i] = xs[np.argmax(b[i])]
            table["N"][i] = len(x)

    dfmax = pd.DataFrame.from_dict(table, orient="index", columns=columns)
    # Returning dataframe for regression testing
    # dfmax.to_csv(pathman(path) + ".csv")
    return dfmax


# def computeStats(data: pd.DataFrame, metrics: List[str] = None, groupby: str = None, output_dir: Union[Path, str]) -> pd.DataFrame:
#     """Calculate summary statistics for the desired metrics.

#     Parameters
#     ==========
#     data: pd.DataFrame
#         Pandas dataframe of grain and/or DNA Tracing statistics.
#     metrics: List(str)
#         List of columns to calculate statistics on.
#     groupby: str
#         Optionally summarise statistics by sub-groups
#     output_dir: Union[Path, str]
#         If included writes results to 'summary_stats.csv' in the provided location.
#
#     Returns
#     =======
#     pd.DataFrame
#         Pandas DataFrame of summary statistics. Optionally written to CSV if outputdir is provided.
#     """
#     metrics = data.columns if metrics is None else metrics
#     if groupby is not None:
#         summary = data.groupby(groupby)[metrics].summary()
#     else:
#         summary = data[metrics].summary()


if __name__ == "__main__":

    # import data from the csv file
    path = plotting_config["file"]
    df = importfromfile(path)
    df = df[df['bending_angle'] != 0]
    path2 = plotting_config["file2"]
    path3 = plotting_config["file3"]
    if path2 is not None:
        df2 = importfromfile(path2)
        df2 = df2[df2['bending_angle'] != 0]
    else:
        df2 = None
    if path3 is not None:
        df3 = importfromfile(path3)
        df3 = df3[df3['bending_angle'] != 0]
    else:
        df3 = None
    extension = plotting_config["extension"]
    output_dir = plotting_config["output_dir"]

    for plot in plotting_config["plots"]:
        plottype = plotting_config["plots"][plot]["plottype"]
        parameter = plotting_config["plots"][plot]["parameter"]
        nm = plotting_config["plots"][plot]["nm"]
        grouparg = plotting_config["plots"][plot]["group"]
        xmin = plotting_config["plots"][plot]["xmin"]
        xmax = plotting_config["plots"][plot]["xmax"]
        ymin = plotting_config["plots"][plot]["ymin"]
        ymax = plotting_config["plots"][plot]["ymax"]
        bins = plotting_config["plots"][plot]["bins_number"]
        start = plotting_config["plots"][plot]["bins_start"]
        end = plotting_config["plots"][plot]["bins_end"]
        label1 = plotting_config["plots"][plot]["label1"]
        label2 = plotting_config["plots"][plot]["label2"]
        label3 = plotting_config["plots"][plot]["label3"]
        color1 = plotting_config["plots"][plot]["color1"]
        color2 = plotting_config["plots"][plot]["color2"]
        color3 = plotting_config["plots"][plot]["color3"]
        if plottype == "histogram":
            plothist(df, parameter, nm=nm, grouparg=grouparg, xmin=xmin, xmax=xmax, bins=np.linspace(start, end, bins))
        elif plottype == "histogram2":
            # plothist2var(df, parameter, df2=df2, nm=nm, xmin=xmin, xmax=xmax, label1='01', label2='02',bins=np.linspace(1, 5, 20))
            plothist2var(df, parameter, df2=df2, df3=df3, nm=nm, xmin=xmin, xmax=xmax, label1=label1,
                         label2=label2, label3=label3, color1=color1, color2=color2, color3=color3,
                         bins=np.linspace(start, end, bins))
        elif plottype == "KDE":
            plotkde(df, parameter, nm=nm, grouparg=grouparg, xmin=xmin, xmax=xmax)
        elif plottype == "violin":
            plotviolin(df, parameter, nm=nm, grouparg=grouparg, ymin=ymin, ymax=ymax)
        elif plottype == "dist":
            plotdist(df, parameter, nm=nm, grouparg=grouparg, xmin=xmin, xmax=xmax)
        elif plottype == "dist2":
            plotdist2var(df, parameter, parameter, df2=df2, nm=nm, xmin=xmin, xmax=xmax, c1="#B45F06", c2="#0B5394", label1='01', label2='02')
        elif plottype == "joint":
            plotjoint(df, parameter, nm=nm)
    # Filter data based on the need of specific projects
    # df = df[df['End to End Distance'] != 0]
    # df = df[df['Contour Lengths'] > 100]
    # df = df[df['Contour Lengths'] < 120]

    # Rename directory column as appropriate
    # df = df.rename(columns={"directory": "Experimental Conditions"})
    # df1 = df1.rename(columns={"directory": "Experimental Conditions"})
    # df2 = df2.rename(columns={"directory": "Experimental Conditions"})
    # df = df.rename(columns={"grain_min_bound_size": "Minimum Bound Size"})
    # df = df.rename(columns={"directory": "Experimental Conditions"})

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


# Setting a continuous colour palette; useful for certain grouped plots, but can be commented out if unsuitable.
# sns.set_palette(sns.color_palette('BuPu', n_colors=len(df.groupby(grouparg))))
# print(df.pivot(columns=grouparg, values='grain_median'))


# Plot one column of the dataframe e.g. 'grain_mean_radius'; grouparg can be specified for plotkde, plothist and
# plotviolin by entering e.g. 'xmin = 0'; xmin and xmax can be specified for plotkde, plothist, and plotjoint;
# ymin and ymax can be specified for plotviolin and plotjoint; bins can be speficied for plothist. The default unit is
# m; add "nm=True" to change from m to nm.

# Examples of possible plots
# plotkde(df, 'area', nm=True)
# plothist(df, 'Contour Lengths', nm=False)
