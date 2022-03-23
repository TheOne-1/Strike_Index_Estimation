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
        axis_x = range(rmse_mean.shape[0])
        plt.plot(axis_x, rmse_mean, label='Training Set', linewidth=LINE_WIDTH, color=color_0)
        plt.fill_between(axis_x, rmse_mean - rmse_std, rmse_mean + rmse_std, facecolor=color_0, alpha=0.4)

        ax.tick_params(labelsize=FONT_DICT_SMALL['fontsize'])
        ax.set_xticks(range(0, rmse_mean.shape[0] + 1, 25))
        ax.set_xticklabels(range(0, rmse_mean.shape[0] + 1, 25), fontdict=FONT_DICT_SMALL)
        ax.set_xlabel('Epoch', fontdict=FONT_DICT_SMALL)
        ax.set_xlim(0, rmse_mean.shape[0])

        ax = plt.gca()
        ax.set_ylabel('RMSE', fontdict=FONT_DICT_SMALL)
        ax.set_ylim(0, 0.2)
        ticks = [0, 0.05, 0.1, 0.15, 0.2]
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticks, fontdict=FONT_DICT_SMALL)

    rc('font', family='Arial')
    plt.figure(figsize=(3.54, 3.2))
    draw_lines(rmse_mean, rmse_std)
    format_axis()
    plt.tight_layout(rect=[-0.02, -0.02, 1.02, 1.02])
    plt.legend(handlelength=3, bbox_to_anchor=(0.98, 0.95), ncol=1, fontsize=FONT_DICT_SMALL['fontsize'],
               frameon=False)
    save_fig('f8', 600)
    plt.show()


if __name__ == '__main__':
    result_date = '220321'
    rmses = []
    for sub_num in [1, 2, 16]:        # range(1, 17)
        if sub_num is not 16:
            condition = f"_training_with_{sub_num}"
        else:
            condition = ""
        rmses.append(metric_sub_mean(result_date, condition, rmse_fun))
    rmse_mean = np.array([np.mean(element) for element in rmses])
    rmse_std = np.array([np.std(element) for element in rmses])
    draw_f8(rmse_mean, rmse_std)












