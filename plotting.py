# %matplotlib inline

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pylab


def plot_bar_chart_1(df, title, xlabel, xticklabels, ylabel, legend, colour, name):
    pos = list(range(len(df[0])))
    width = 0.5

    # Plotting the bars
    fig, ax = plt.subplots(figsize=(10, 5))

    # Create a bar with pre_score data,
    # in position pos,
    plt.bar(pos,
            # using df['pre_score'] data,
            df[0],
            # of width
            width,
            # with alpha 0.5
            alpha=0.5,
            # with color
            color=colour,
            # with label the first value in first_name
            label="before")

    # Set the y axis label
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    # Set the chart's title
    ax.set_title(title)

    # Set the position of the x ticks
    ax.set_xticks([p for p in pos])

    # labels = ["{} bps".format(x) for x in range(10)]
    # Set the labels for the x ticks
    ax.set_xticklabels(xticklabels)

    # Setting the x-axis and y-axis limits
    plt.xlim(min(pos) - width, max(pos) + width * 4)
    # plt.ylim([0, max(df['pre_score'] + df['mid_score'] + df['post_score'])])

    # Adding the legend and showing the plot
    plt.legend(legend, loc='upper center',
               bbox_to_anchor=(0.5, -0.095),
               fancybox=False, shadow=False, ncol=5)
    plt.grid()
    plt.savefig(name, dpi=200)
    plt.show()


def plot_bar_chart_2(df, title, xlabel, xticklabels, ylabel, legend, colours, name):
    pos = list(range(len(df[0])))
    width = 0.33

    # Plotting the bars
    fig, ax = plt.subplots(figsize=(10, 5))

    # Create a bar with pre_score data,
    # in position pos,
    plt.bar(pos,
            # using df['pre_score'] data,
            df[0],
            # of width
            width,
            # with alpha 0.5
            alpha=0.5,
            # with color
            color=colours[0],
            # with label the first value in first_name
            label="before")

    # Create a bar with mid_score data,
    # in position pos + some width buffer,
    plt.bar([p + width for p in pos],
            # using df['mid_score'] data,
            df[1],
            # of width
            width,
            # with alpha 0.5
            alpha=0.5,
            # with color
            color=colours[1],
            # with label the second value in first_name
            label="after")

    # Set the y axis label
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    # Set the chart's title
    ax.set_title(title)

    # Set the position of the x ticks
    ax.set_xticks([p + 0.5 * width for p in pos])

    # labels = ["{} bps".format(x) for x in range(10)]
    # Set the labels for the x ticks
    ax.set_xticklabels(xticklabels)

    # Setting the x-axis and y-axis limits
    plt.xlim(min(pos) - width, max(pos) + width * 4)
    # plt.ylim([0, max(df['pre_score'] + df['mid_score'] + df['post_score'])])

    # Adding the legend and showing the plot
    plt.legend(legend, loc='upper center',
               bbox_to_anchor=(0.5, -0.095),
               fancybox=False, shadow=False, ncol=5)
    plt.grid()
    plt.savefig(name, dpi=200)
    plt.show()


def plot_bar_chart_3(df, title, xlabel, xticklabels, ylabel, legend, colours, name):
    pos = list(range(len(df[0])))
    width = 0.25

    # Plotting the bars
    fig, ax = plt.subplots(figsize=(10, 5))

    # Create a bar with pre_score data,
    # in position pos,
    plt.bar(pos,
            # using df['pre_score'] data,
            df[0],
            # of width
            width,
            # with alpha 0.5
            alpha=0.5,
            # with color
            color=colours[0],
            # with label the first value in first_name
            label="before")

    # Create a bar with mid_score data,
    # in position pos + some width buffer,
    plt.bar([p + width for p in pos],
            # using df['mid_score'] data,
            df[1],
            # of width
            width,
            # with alpha 0.5
            alpha=0.5,
            # with color
            color=colours[1],
            # with label the second value in first_name
            label="after")

    # Create a bar with post_score data,
    # in position pos + some width buffer,
    plt.bar([p + width * 2 for p in pos],
            # using df['post_score'] data,
            df[2],
            # of width
            width,
            # with alpha 0.5
            alpha=0.5,
            # with color
            color=colours[2],
            # with label the third value in first_name
            label="nevermind")

    # Set the y axis label
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    # Set the chart's title
    ax.set_title(title)

    # Set the position of the x ticks
    ax.set_xticks([p + 0.5 * width for p in pos])

    # labels = ["{} bps".format(x) for x in range(10)]
    # Set the labels for the x ticks
    ax.set_xticklabels(xticklabels)

    # Setting the x-axis and y-axis limits
    plt.xlim(min(pos) - width, max(pos) + width * 4)
    # plt.ylim([0, max(df['pre_score'] + df['mid_score'] + df['post_score'])])

    # Adding the legend and showing the plot
    plt.legend(legend, loc='upper center',
               bbox_to_anchor=(0.5, -0.095),
               fancybox=False, shadow=False, ncol=5)
    plt.grid()
    plt.savefig(name, dpi=200)
    plt.show()