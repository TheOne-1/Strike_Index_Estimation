from SharedProcessors.const import SUB_NAMES, LINE_WIDTH, FONT_DICT_SMALL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Drawer import save_fig
from matplotlib import rc


def draw_f7(training_mean, training_std, test_mean, test_std):
    def format_axis(line_width=LINE_WIDTH):
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_tick_params(width=line_width)
        ax.yaxis.set_tick_params(width=line_width)
        ax.spines['left'].set_linewidth(line_width)
        ax.spines['bottom'].set_linewidth(line_width)

    def draw_lines(training_mean, training_std, test_mean, test_std):
        ax = plt.gca()
        color_0, color_1 = np.array([37, 128, 92]) / 255, np.array([53, 128, 57]) / 255
        axis_x = range(training_mean.shape[0])
        plt.plot(axis_x, training_mean, label='Training Set', linewidth=LINE_WIDTH)
        plt.fill_between(axis_x, training_mean - training_std, training_mean + training_std, alpha=0.4)
        plt.plot(axis_x, test_mean, label='Test Set', linewidth=LINE_WIDTH, color=color_1)
        plt.fill_between(axis_x, test_mean - test_std, test_mean + test_std, alpha=0.4, facecolor=color_1)

        ax.tick_params(labelsize=FONT_DICT_SMALL['fontsize'])
        ax.set_xticks(range(0, training_mean.shape[0]+1, 25))
        ax.set_xticklabels(range(0, training_mean.shape[0]+1, 25), fontdict=FONT_DICT_SMALL)
        ax.set_xlabel('Epoch', fontdict=FONT_DICT_SMALL)
        ax.set_xlim(0, training_mean.shape[0])

        ax = plt.gca()
        ax.set_ylabel('RMSE', fontdict=FONT_DICT_SMALL)
        ax.set_ylim(0, 0.4)
        ticks = [0, 0.1, 0.2, 0.3, 0.4]
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticks, fontdict=FONT_DICT_SMALL)

    rc('font', family='Arial')
    plt.figure(figsize=(3.54, 3.2))
    draw_lines(training_mean, training_std, test_mean, test_std)
    format_axis()
    plt.tight_layout(rect=[-0.02, -0.02, 1.02, 1.02])
    plt.legend(handlelength=3, bbox_to_anchor=(0.98, 0.95), ncol=1, fontsize=FONT_DICT_SMALL['fontsize'],
               frameon=False)
    save_fig('f7', 600)
    plt.show()


if __name__ == '__main__':
    result_date = '220321'
    training_rmse, test_rmse = [], []
    for sub in SUB_NAMES:  # SUB_NAMES
        training_log = pd.read_csv('./result_conclusion/{}/training_log/{}.csv'.format(result_date, sub), index_col=False)
        training_rmse.append(np.sqrt(training_log['mean_squared_error'].values))
        test_rmse.append(np.sqrt(training_log['val_mean_squared_error'].values))

    training_rmse_mean, training_rmse_std = np.mean(training_rmse, axis=0), np.std(training_rmse, axis=0)
    test_rmse_mean, test_rmse_std = np.mean(test_rmse, axis=0), np.std(test_rmse, axis=0)
    draw_f7(training_rmse_mean, training_rmse_std, test_rmse_mean, test_rmse_std)


