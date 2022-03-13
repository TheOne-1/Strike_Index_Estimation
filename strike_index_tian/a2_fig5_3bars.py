from const import LINE_WIDTH, FONT_DICT_SMALL, SUB_NAMES
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
from scipy.stats import ttest_rel
import matplotlib as mpl
import numpy as np
from sklearn.metrics import mean_squared_error
from strike_index_tian.Drawer import save_fig, format_plot, load_step_data
import pingouin as pg
import prettytable as pt


def format_errorbar_cap(caplines, size):
    for i_cap in range(1):
        caplines[i_cap].set_marker('_')
        caplines[i_cap].set_markersize(size)
        caplines[i_cap].set_markeredgewidth(LINE_WIDTH)


def draw_f5(mean_, std_, sigifi_sign_fun):
    def format_ticks():
        ax = plt.gca()
        ax.set_ylabel('Root Mean Square Error (%)', fontdict=FONT_DICT_SMALL, labelpad=0)
        ax.set_ylim(0, 0.12)
        ax.set_yticks(np.arange(0, 0.13, 0.03))
        ax.set_yticklabels(['0%', '3%', '6%', '9%', '12%'], fontdict=FONT_DICT_SMALL)
        ax.set_xlim(-1, 5)
        ax.set_xticks([0, 2, 4])
        ax.set_xticklabels(['Standard &\nMinimalist', 'Standard ', 'Minimalist'],
                           fontdict=FONT_DICT_SMALL, linespacing=1)
        ax.set_xlabel('Data Used for Model Training', fontdict=FONT_DICT_SMALL)
        # ax.tick_params(axis='x', which='major', pad=10)
        for xtick, color in zip(ax.get_xticklabels(), colors):
            xtick.set_color(color)

    rc('font', family='Arial')
    fig = plt.figure(figsize=(3.54, 2.6))
    base_color = np.array([37, 128, 92]) / 255
    colors = [base_color * x for x in [1.3, 1, 0.7]]
    format_plot()
    bar_locs = [0, 2, 4]
    bar_ = []
    for i_condition in range(3):
        bar_.append(plt.bar(bar_locs[i_condition], mean_[i_condition], color=colors[i_condition], width=1))
    ebar, caplines, barlinecols = plt.errorbar(bar_locs, mean_, std_, capsize=0, ecolor='black',
                                               fmt='none', lolims=True, elinewidth=LINE_WIDTH)

    sigifi_sign_fun(mean_, std_, bar_locs)
    format_errorbar_cap(caplines, 12)
    plt.tight_layout(rect=[0.05, 0.07, 1.01, 1.01])
    format_ticks()


def draw_sigifi_sign(mean_, std_, bar_locs, one_three=False, one_four=False, two_three=False, two_four=False):
    dis = 0.01
    top_line = max([a + b for a, b in zip(mean_, std_)]) + 0.015
    for pair, loc_0, loc_1 in zip([two_three, two_four, one_three, one_four], [1, 1, 0, 0], [2, 3, 2, 3]):
        if not pair:
            continue
        star_x_loc_correction = (loc_1 - loc_0) * 0.02
        coe_0, coe_1 = 0.5 + star_x_loc_correction, 0.5 - star_x_loc_correction
        diff_line_0x = [bar_locs[loc_0], bar_locs[loc_0], bar_locs[loc_1], bar_locs[loc_1]]
        diff_line_0y = [mean_[loc_0] + std_[loc_0] + dis, top_line, top_line, mean_[loc_1] + std_[loc_1] + dis]
        plt.plot(diff_line_0x, diff_line_0y, 'black', linewidth=LINE_WIDTH)
        plt.text(bar_locs[loc_0]*coe_0 + bar_locs[loc_1]*coe_1, top_line - 0.002,
                 '*', fontdict={'fontname': 'Times New Roman'}, color='black', size=17)
        top_line += 0.015


def transfer_data_to_long_df(condition_and_train, conditions, rmses):
    rows = []
    for i_sub in range(len(SUB_NAMES)):
        for condition in conditions:
            train = condition_and_train[condition]
            rmse = float(rmses[condition][i_sub])
            rows.append([i_sub, train, rmse])
    df = pd.DataFrame(rows, columns=['subject id', 'train', 'RMSE'])
    return df


def statistics(result_df):
    anova_result = pg.rm_anova(dv='RMSE', within=['train'], subject='subject id', data=result_df)
    print(anova_result)
    multi_compare_result = pg.pairwise_ttests(result_df, dv='RMSE', within=['train'], subject='subject id', padjust='bonf')
    print(multi_compare_result)


if __name__ == "__main__":
    result_date = '211214'
    condition_and_train = {'': 'all', '_tr_trad_te_all': 'standard', '_tr_mini_te_all': 'minimalist'}
    conditions = list(condition_and_train.keys())
    rmses = {condition: [] for condition in conditions}
    means, stds = [], []
    for condition in conditions:
        si_true, si_pred = load_step_data(result_date, condition)
        for si_true_sub, si_pred_sub in zip(si_true, si_pred):
            rmses[condition].append(np.sqrt(mean_squared_error(si_true_sub, si_pred_sub)))
        means.append(np.mean(rmses[condition]))
        stds.append(np.std(rmses[condition]))

    result_df = transfer_data_to_long_df(condition_and_train, conditions, rmses)
    statistics(result_df)
    draw_f5(means, stds, draw_sigifi_sign)
    save_fig('f5', 600)
    plt.show()
