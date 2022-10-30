import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def setup(nrows=1, ncols=1, figsize=(8, 6), palette='colorblind'):

    sns.set()
    sns.set_style('whitegrid')
    sns.set_context('paper')
    sns.set_palette(palette)
    plt.rcParams['font.family'] = 'serif'
    # plt.rcParams['text.usetex'] = True

    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, tight_layout=True)
    array_marker = ['v', 'x', 'o', '^', 's', '+', 'D', '*', '<', '>']
    array_color = sns.color_palette(palette)

    return fig, ax, array_marker, array_color


def set_ax_property(ax,
                    xscale='linear',
                    yscale='linear',
                    xticks='',
                    yticks='',
                    xlim_left='',
                    xlim_right='',
                    ylim_top='',
                    ylim_bottom='',
                    xlabel='',
                    ylabel='',
                    title='',
                    fontsize=18):

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if xticks != '':
        ax.set_xticks(xticks)
    if yticks != '':
        ax.set_yticks(yticks)
    if xlim_left != '':
        ax.set_xlim(left=xlim_left)
    if xlim_right != '':
        ax.set_xlim(right=xlim_right)
    if ylim_top != '':
        ax.set_ylim(top=ylim_top)
    if ylim_bottom != '':
        ax.set_ylim(bottom=ylim_bottom)
    if xlabel != '':
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel != '':
        ax.set_ylabel(ylabel, fontsize=fontsize)
    if title != '':
        ax.set_title(title, fontsize=fontsize)
    ax.legend(fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    ax.grid(which='major', linestyle=':')
    ax.grid(which='minor', linestyle=':')
