# %matplotlib inline

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_bar_chart(df):
    # Setting the positions and width for the bars
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
            color='#F78F1E',
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
            color='#EE3224',
            # with label the second value in first_name
            label="after")

    # Set the y axis label
    ax.set_ylabel('Expected probability of implementation')

    # Set the chart's title
    ax.set_title('Expected outcomes before and after adding breaking points')

    # Set the position of the x ticks
    ax.set_xticks([p + 0.5 * width for p in pos])

    labels = ["Issue {}".format(x) for x in range(10)]
    # Set the labels for the x ticks
    ax.set_xticklabels(labels)

    # Setting the x-axis and y-axis limits
    plt.xlim(min(pos) - width, max(pos) + width * 4)
    # plt.ylim([0, max(df['pre_score'] + df['mid_score'] + df['post_score'])])

    # Adding the legend and showing the plot
    plt.legend(['Before adding breaking points', 'After adding breaking points'], loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=False, shadow=False, ncol=5)
    plt.grid()
    plt.show()

def plot_bar_chart_regret(df):
    # Setting the positions and width for the bars
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
            color='#F78F1E',
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
            color='#EE3224',
            # with label the second value in first_name
            label="after")

    # Set the y axis label
    ax.set_ylabel('')

    # Set the chart's title
    ax.set_title('Average voter agreement')

    # Set the position of the x ticks
    ax.set_xticks([p + 0.5 * width for p in pos])

    labels = ["{} bps".format(x) for x in range(10)]
    # Set the labels for the x ticks
    ax.set_xticklabels(labels)

    # Setting the x-axis and y-axis limits
    plt.xlim(min(pos) - width, max(pos) + width * 4)
    # plt.ylim([0, max(df['pre_score'] + df['mid_score'] + df['post_score'])])

    # Adding the legend and showing the plot
    plt.legend(['Rate of agreement with implemented policy', 'Standard deviation'], loc='upper center',
               bbox_to_anchor=(0.5, -0.05),
               fancybox=False, shadow=False, ncol=5)
    plt.grid()
    plt.show()