from const import LINE_WIDTH, FONT_DICT_SMALL, SUB_NAMES
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
from scipy.stats import ttest_rel
import matplotlib as mpl
import numpy as np
from strike_index_tian.Drawer import save_fig, format_plot, metric_sub_mean, rmse_fun
import pingouin as pg
import prettytable as pt
from strike_index_tian.a2_fig5 import draw_sigifi_sign


def format_errorbar_cap(caplines, size=15):
    for i_cap in range(1):
        caplines[i_cap].set_marker('_')
        caplines[i_cap].set_markersize(size)
        caplines[i_cap].set_markeredgewidth(LINE_WIDTH)


def draw_f4(mean_, std_, sigifi_sign_fun):
    def format_ticks():
        ax = plt.gca()
        ax.set_ylabel('Root Mean Square Error (%)', fontdict=FONT_DICT_SMALL, labelpad=0)
        ax.set_ylim(0, 0.1)
        ax.set_yticks(np.arange(0, 0.11, 0.02))
        ax.set_yticklabels(['0%', '2%', '4%', '6%', '8%', '10%'], fontdict=FONT_DICT_SMALL)
        ax.set_xlim(-1, 7)
        ax.set_xticks([0, 2, 4, 6])
        ax.set_xticklabels(['2.4 m/s\n2.4 m/s', '2.8 m/s\n2.8 m/s', '2.4 m/s\n2.8 m/s', '2.8 m/s\n2.4 m/s'],
                           fontdict=FONT_DICT_SMALL, linespacing=2.2)
        ax.tick_params(axis='x', which='major', pad=10)
        for xtick, color in zip(ax.get_xticklabels(), colors):
            xtick.set_color(color)
        base_x, base_y = -0.22, -0.157
        ax.add_patch(mpl.patches.Rectangle((base_x, base_y), ls='--', width=1.23, height=0.08, ec="gray", fill=False,
                                           transform=ax.transAxes, clip_on=False))
        plt.text(base_x+0.02, base_y + 0.018, 'Training', transform=ax.transAxes, fontdict=FONT_DICT_SMALL)
        ax.add_patch(mpl.patches.Rectangle((base_x, base_y-0.112), ls='--', width=1.23, height=0.08, ec="gray", fill=False,
                                           transform=ax.transAxes, clip_on=False))
        plt.text(base_x+0.02, base_y-0.096, 'Test', transform=ax.transAxes, fontdict=FONT_DICT_SMALL)

    rc('font', family='Arial')
    fig = plt.figure(figsize=(3.54, 2.6))
    base_color = np.array([68, 128, 22]) / 255
    colors = [base_color * x for x in [1.2, 1, 0.8, 0.6]]
    format_plot()
    bar_locs = [0, 2, 4, 6]
    bar_ = []
    for i_condition in range(4):
        bar_.append(plt.bar(bar_locs[i_condition], mean_[i_condition], color=colors[i_condition], width=1))
    ebar, caplines, barlinecols = plt.errorbar(bar_locs, mean_, std_, capsize=0, ecolor='black',
                                               fmt='none', lolims=True, elinewidth=LINE_WIDTH)

    sigifi_sign_fun(mean_, std_, bar_locs)
    format_errorbar_cap(caplines, 8)
    plt.tight_layout(rect=[0.05, 0.07, 1.01, 1.01])
    format_ticks()


def transfer_data_to_long_df(condition_and_train_test, conditions, rmses):
    rows = []
    for i_sub in range(len(SUB_NAMES)):
        for condition in conditions:
            train, test = condition_and_train_test[condition]
            rmse = float(rmses[condition][i_sub])
            rows.append([i_sub, train, test, rmse])
    df = pd.DataFrame(rows, columns=['subject id', 'train', 'test', 'RMSE'])
    return df


def statistics(result_df, rmses, conditions):
    anova_result = pg.rm_anova(dv='RMSE', within=['train', 'test'], subject='subject id', data=result_df)
    print(anova_result)

    tb = pt.PrettyTable()
    tb.field_names = conditions
    for i_combo, combo_a in enumerate(conditions):
        p_val_row = []
        for j_combo, combo_b in enumerate(conditions):
            if i_combo == j_combo:
                p_val = 1
            else:
                p_val = round(ttest_rel(rmses[combo_a], rmses[combo_b]).pvalue, 3)
            p_val_row.append(p_val)
        tb.add_row(p_val_row)
    tb.add_column('', conditions, align="l")
    print(tb)


if __name__ == "__main__":
    result_date = '211206'
    condition_and_train_test = {'_24': ('24', '24'), '_28': ('28', '28'),
                                '_tr24te28': ('24', '28'), '_tr28te24': ('28', '24')}
    conditions = list(condition_and_train_test.keys())
    rmses = {condition: [] for condition in conditions}
    means, stds = [], []
    for condition in conditions:
        rmses[condition] = metric_sub_mean(result_date, condition, rmse_fun)
        means.append(np.mean(rmses[condition]))
        stds.append(np.std(rmses[condition]))

    result_df = transfer_data_to_long_df(condition_and_train_test, conditions, rmses)
    statistics(result_df, rmses, conditions)
    print(max(means) - min(means))
    draw_f4(means, stds, draw_sigifi_sign)
    save_fig('f4', 600)
    plt.show()
