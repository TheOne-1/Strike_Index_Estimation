from const import LINE_WIDTH, SUB_NAMES, FONT_SIZE, FONT_DICT, TRIAL_NAMES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as lines
from Evaluation import Evaluation


def format_plot():
    mpl.rcParams['hatch.linewidth'] = LINE_WIDTH  # previous svg hatch linewidth
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(width=LINE_WIDTH)
    ax.yaxis.set_tick_params(width=LINE_WIDTH)
    ax.spines['left'].set_linewidth(LINE_WIDTH)
    ax.spines['bottom'].set_linewidth(LINE_WIDTH)


def format_errorbar_cap(caplines):
    for i_cap in range(1):
        caplines[i_cap].set_marker('_')
        caplines[i_cap].set_markersize(25)
        caplines[i_cap].set_markeredgewidth(LINE_WIDTH)


def save_fig(name, dpi=600):
    plt.savefig('exports/' + name + '.png', dpi=dpi)




