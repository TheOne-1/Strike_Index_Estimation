from const import LINE_WIDTH, FONT_DICT_SMALL, SUB_NAMES
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
from scipy.stats import ttest_rel
import matplotlib as mpl
import numpy as np
from strike_index_tian.Drawer import save_fig, format_plot, metric_sub_mean, rmse_fun, cohen_d
import pingouin as pg
import prettytable as pt
import warnings


def draw_f5(rmses, conditions, sigifi_sign_fun):
    def format_ticks():
        ax = plt.gca()
        ax.set_ylabel('Root Mean Square Error (%)', fontdict=FONT_DICT_SMALL, labelpad=5)
        ax.set_ylim(0.02, 0.26)
        ax.set_yticks(np.arange(0.02, 0.28, 0.06))
        ax.set_yticklabels(['2%', '8%', '14%', '20%', '26%'], fontdict=FONT_DICT_SMALL)
        ax.set_xlim(-0.5, 3.5)
        ax.set_xticks(bar_locs)
        ax.set_xticklabels(['Standard\nStandard', 'Minimalist\nMinimalist', 'Standard\nMinimalist', 'Minimalist\nStandard'],
                           fontdict=FONT_DICT_SMALL, linespacing=2.2)
        ax.tick_params(axis='x', which='major', pad=10)
        for xtick, color in zip(ax.get_xticklabels(), colors):
            xtick.set_color(color)
        base_x, base_y = -0.22, -0.123
        ax.add_patch(mpl.patches.Rectangle((base_x, base_y), ls='--', width=1.23, height=0.066, ec="gray", fill=False,
                                           transform=ax.transAxes, clip_on=False))
        plt.text(base_x+0.02, base_y + 0.015, 'Training', transform=ax.transAxes, fontdict=FONT_DICT_SMALL)
        ax.add_patch(mpl.patches.Rectangle((base_x, base_y-0.085), ls='--', width=1.23, height=0.066, ec="gray", fill=False,
                                           transform=ax.transAxes, clip_on=False))
        plt.text(base_x+0.02, base_y-0.071, 'Test', transform=ax.transAxes, fontdict=FONT_DICT_SMALL)

    rc('font', family='Arial')
    fig = plt.figure(figsize=(3.54, 3.2))
    base_color = np.array([37, 128, 92]) / 255
    colors = [base_color * x for x in [1.2, 1, 0.8, 0.6]]
    format_plot()
    bar_locs = [0, 1, 2, 3]
    for i_condition, condition in enumerate(conditions):
        box_ = plt.boxplot(rmses[condition], positions=[i_condition], widths=[0.5], patch_artist=True)
        for field in ['medians', 'whiskers', 'caps', 'boxes']:
            [box_[field][i].set(linewidth=LINE_WIDTH, color=colors[i_condition]) for i in range(len(box_[field]))]
        [box_['fliers'][i].set(marker='D', markeredgecolor=colors[i_condition], markerfacecolor=colors[i_condition], markersize=2.5) for i in range(len(box_['fliers']))]
        box_['medians'][0].set(linewidth=LINE_WIDTH, color=[1, 1, 1])

    plt.tight_layout(rect=[0.02, 0.07, 1.01, 1.01])
    sigifi_sign_fun(rmses, bar_locs, two_four=True, one_four=True)
    format_ticks()


def draw_sigifi_sign(rmses, bar_locs, one_three=False, one_four=False, two_three=False, two_four=False):
    dis = 0.015
    rmses = [max(rmses[condition]) for condition in conditions]
    top_line = max(rmses) + 0.025
    for pair, loc_0, loc_1 in zip([two_three, two_four, one_three, one_four], [1, 1, 0, 0], [2, 3, 2, 3]):
        if not pair:
            continue
        star_x_loc_correction = (loc_1 - loc_0) * 0.02
        coe_0, coe_1 = 0.5 + star_x_loc_correction, 0.5 - star_x_loc_correction
        diff_line_0x = [bar_locs[loc_0], bar_locs[loc_0], bar_locs[loc_1], bar_locs[loc_1]]
        diff_line_0y = [rmses[loc_0] + dis, top_line, top_line, rmses[loc_1] + dis]
        plt.plot(diff_line_0x, diff_line_0y, 'black', linewidth=LINE_WIDTH)
        plt.text(bar_locs[loc_0]*coe_0 + bar_locs[loc_1]*coe_1, top_line - 0.002,
                 '*', fontdict={'fontname': 'Times New Roman'}, color='black', size=17)
        top_line += 0.025


def transfer_data_to_long_df(condition_and_train_test, conditions, rmses):
    rows = []
    for i_sub in range(len(SUB_NAMES)):
        for condition in conditions:
            train, test = condition_and_train_test[condition]
            rmse = float(rmses[condition][i_sub])
            rows.append([i_sub, train, test, rmse])
            # rows.append(rmses[condition])
    df = pd.DataFrame(rows, columns=['subject id', 'train', 'test', 'RMSE'])
    return df


def statistics(result_df, rmses, conditions):
    anova_result = pg.rm_anova(dv='RMSE', within=['train', 'test'], subject='subject id', data=result_df)
    print(anova_result)

    tb_p, tb_d = pt.PrettyTable(), pt.PrettyTable()
    tb_p.field_names, tb_d.field_names = conditions, conditions
    for i_combo, combo_a in enumerate(conditions):
        p_val_row, d_val_row = [], []
        for j_combo, combo_b in enumerate(conditions):
            if i_combo == j_combo:
                p_val, d_val = 1, 0
            else:
                p_val = round(ttest_rel(rmses[combo_a], rmses[combo_b]).pvalue, 3)
                d_val = round(cohen_d(rmses[combo_a], rmses[combo_b]), 3)
            p_val_row.append(p_val)
            d_val_row.append(d_val)
        tb_p.add_row(p_val_row)
        tb_d.add_row(d_val_row)
    tb_p.add_column('', conditions, align="l")
    tb_d.add_column('', conditions, align="l")
    print(tb_p)
    print(tb_d)


if __name__ == "__main__":
    result_date = '211206'
    condition_and_train_test = {'_Trad': ('Standard', 'Standard'), '_Minim': ('Minimalist', 'Minimalist'),
                                '_trTradteMinim': ('Standard', 'Minimalist'), '_trMinimteTrad': ('Minimalist', 'Standard')}
    conditions = list(condition_and_train_test.keys())
    rmses = {condition: metric_sub_mean(result_date, condition, rmse_fun) for condition in conditions}

    result_df = transfer_data_to_long_df(condition_and_train_test, conditions, rmses)
    statistics(result_df, rmses, conditions)
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        draw_f5(rmses, conditions, draw_sigifi_sign)
    save_fig('f5', 600)
    plt.show()
