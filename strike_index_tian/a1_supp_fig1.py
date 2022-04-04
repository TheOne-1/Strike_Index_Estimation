from SharedProcessors.const import SUB_NAMES, LINE_WIDTH, FONT_DICT_SMALL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from strike_index_tian.Drawer import save_fig, format_plot, metric_sub_mean, rmse_fun
from matplotlib import rc


def draw_f8(rmse_mean, rmse_std):
    def format_axis(line_width=LINE_WIDTH):
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_tick_params(width=line_width)
        ax.yaxis.set_tick_params(width=line_width)
        ax.spines['left'].set_linewidth(line_width)
        ax.spines['bottom'].set_linewidth(line_width)

    def draw_lines(rmse_mean, rmse_std):
        ax = plt.gca()
        color_0 = np.array([37, 128, 92]) / 255
        axis_x = range(1, 16)
        plt.plot(axis_x, rmse_mean, linewidth=LINE_WIDTH, color=color_0)
        plt.fill_between(axis_x, rmse_mean - rmse_std, rmse_mean + rmse_std, facecolor=color_0, alpha=0.4)

        ax.tick_params(labelsize=FONT_DICT_SMALL['fontsize'])
        ax.set_xticks(range(1, 16, 2))
        ax.set_xticklabels(range(1, 16, 2), fontdict=FONT_DICT_SMALL)
        ax.set_xlabel('Number of Participant Used for Training', fontdict=FONT_DICT_SMALL)
        ax.set_xlim(1, 15)

        ax = plt.gca()
        ax.set_ylabel('RMSE', fontdict=FONT_DICT_SMALL)
        ax.set_ylim(0, 0.16)
        ticks = [0, 0.04, 0.08, 0.12, 0.16]
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticks, fontdict=FONT_DICT_SMALL)

    rc('font', family='Arial')
    plt.figure(figsize=(3.54, 2.8))
    draw_lines(rmse_mean, rmse_std)
    format_axis()
    plt.tight_layout(rect=[-0.02, -0.02, 1.02, 1.02])
    save_fig('f8', 600)
    plt.show()


if __name__ == '__main__':
    result_date = '220325'
    rmses = []
    for sub_num in range(1, 16):
        if sub_num is not 15:
            condition = f"_training_with_{sub_num}"
        else:
            condition = ""
        rmses.append(metric_sub_mean(result_date, condition, rmse_fun))
    rmse_mean = np.array([np.mean(element) for element in rmses])
    rmse_std = np.array([np.std(element) for element in rmses])
    draw_f8(rmse_mean, rmse_std)












