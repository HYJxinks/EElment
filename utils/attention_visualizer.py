# -*- coding: utf-8 -*-
# file: attention_visualizer.py
# time: 2024/7/25 1004
# author: Feng, Yi <jmyf@21cn.com> (易锋)
# github: https://github.com/zsrainbow/
# BSD 3 clause

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re


def plot_attention(words, attention):
    """Plotting the heat of attention scores between specified words and other words in a sentence

    Args:
        words (list): words in a sentence
        attention (list):  Attention scores corresponding to words in a sentence.         
    """
    data = {}
    for i in range(len(words)):
        data[words[i]] = {'attention': attention[i]}
    df = pd.DataFrame(data, columns=words)

    _, (ax1) = plt.subplots(figsize=(len(words), 3), nrows=1)

    sns.heatmap(df, annot=True, ax=ax1, linewidths=1, vmax=1, fmt='.2f', vmin=0, center=0.3, cmap='Reds')  # YlGnBu,YlOrRd,YlGn,Reds
    plt.show()


def plot_attentions(words, attentions, labels, title):
    """Plotting the heat of attention scores between specified words and other words in a sentence.
       Unlike the above function, if the sentence is too long, it is divided into multiple segments for plotting.
    Args:
        words (list): words in a sentence
        attention (list):  Attention scores corresponding to words in a sentence.                            
        labels (list): graph's labels.
        title (list): graph's titles
    """
    datas = []
    data = {}
    columns = []
    max_word_num_per_subplot = 30
    for i in range(len(words)):
        columns.append(words[i])
        data[words[i]] = {}
        for j in range(len(labels)):
            data[words[i]][labels[j]] = attentions[j][i]
        if (i > 0 and i % max_word_num_per_subplot == 0) or i == len(words) - 1:
            df = pd.DataFrame(data, columns=columns)
            datas.append(df)
            data = {}
            columns = []

    nrows = len(datas)
    _, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(80, len(labels) + 5))
    if nrows == 1:
        axes = [axes]
    # https://seaborn.pydata.org/generated/seaborn.heatmap.html
    axes[0].set_title(title)
    for i, axis in enumerate(axes):        
        label_x = axis.get_xticklabels()
        rotation = 30
        plt.setp(label_x, rotation=rotation)
        #
        # If you want to specify the font within the graph, such as size, whether to bold or italicize, font color, etc., 
        # you can accomplish this by specifying the dictionary parameter annot_kws. 
        # The following is a simple example: annot_kws={'size':12,'weight':'bold', 'color':'red'}；
        #
        # Use the cmap parameter to set the color scheme for the heatmap. 
        # You can refer to the following website for all the available color schemes.
        # https://matplotlib.org/stable/users/explain/colors/colormaps.html
        #
        # cbar：Whether to draw a colorbar. if it's set to True, 
        # you can set colorbar by specifying the dictionary parameter cbar_kws={"orientation": "vertical"}.
        #
        sns_plot=sns.heatmap(datas[i], annot=True, ax=axis, linewidths=1, vmax=1, fmt='.2f', vmin=0, center=0.3, cmap='Reds',  annot_kws={'size': 5},cbar=False)
        # Change the appearance of ticks, tick labels, and gridlines. https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.tick_params.html
        sns_plot.tick_params(labelsize=5)
        # To display the colorbar separately and adjust the scaling. https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.colorbar.html
        cb = sns_plot.figure.colorbar(sns_plot.collections[0],shrink=1)  
        # set the fontsize of colorbar
        cb.ax.tick_params(labelsize=5)  
        plt.tight_layout(pad=5)
        plt.subplots_adjust(left=0.03, right=1, top=0.7, bottom=0.35)
    plt.show()


def plot_multi_attentions_of_sentence(words, attentions_list, labels, titles, savefig_filepath=None):
    """Plotting the heat of attention scores between specified words and other words in a sentence.
       Note that the scores here may be two-dimensional, which means the attention scores calculated for words in multiple models.

    Args:
        words (list): words in a sentence
        attention (list):  Attention scores corresponding to words in a sentence.                            
        labels (list): graph's labels.
        title (list): graph's titles
        savefig_filepath (str, optional):  Defaults to None. Whether to store the drawn image. 
                                           If not None, the image is stored with the path and filename corresponding to the string.
    """
    datas = []
    for i in range(len(titles)):
        data = {}
        columns = range(len(words))
        if i == len(titles) - 1:
            columns = ['%s-%s' % (str(words[k]), str(k)) for k in range(len(words))]
        for j in range(len(words)):
            column_name = columns[j]
            data[column_name] = {labels[i]: attentions_list[i][j]}
        df = pd.DataFrame(data, columns=columns)
        datas.append(df)

    nrows = len(datas)
    _, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(80, len(labels) + 5))
    if nrows == 1:
        axes = [axes]

    for i, axis in enumerate(axes):
        axis.set_title(titles[i], fontsize=15)
        label_x = axis.get_xticklabels()
        rotation = 30
        plt.setp(label_x, rotation=rotation)
        sns_plot = sns.heatmap(datas[i], annot=True, ax=axis, linewidths=1, vmax=1, fmt='.2f', vmin=0, center=0.3, cmap='Reds', annot_kws={'size': 15}, cbar=True, cbar_kws={'shrink': 1})  
        sns_plot.tick_params(labelsize=15)  
        plt.tight_layout(pad=5)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.09)
    if savefig_filepath:
        plt.savefig(savefig_filepath, format='svg')
    else:
        plt.show()

def extract_numbers(text: str):
    """Extract score's values from attention String. 
       I(0.06) want(0.27) -> 0.06 0.27

    Args:
        text (str): string of attention score

    Returns:
        float: float scores of attention
    """
    nums = re.findall('[\d]\\.[\d]+', text)
    result = [float(num) for num in nums]
    return result


if __name__ == '__main__':
    words = 'I go to Sushi Rose for fresh sushi and great portions all at a reasonable price'.split()
    attentions = [
        "I(0.00) go(0.00) to(0.00) Sushi(0.06) Rose(0.00) for(0.00) fresh(0.07) sushi(0.63) and(0.01) great(0.00) portions(0.22) all(0.01) at(0.00) a(0.00) reasonable(0.00) price(0.00)",
        "I(0.01) go(0.00) to(0.00) Sushi(0.23) Rose(0.00) for(0.00) fresh(0.22) sushi(0.23) and(0.00) great(0.00) portions(0.23) all(0.01) at(0.00) a(0.00) reasonable(0.01) price(0.00)"]
    attentions = [extract_numbers(attention) for attention in attentions]
    labels = ['lstm', 'affine']
    titles = ['t1', 't2']
    plot_multi_attentions_of_sentence(words, attentions, labels, ['t1', 't2'])
    plot_multi_attentions_of_sentence(words, attentions, labels, titles)