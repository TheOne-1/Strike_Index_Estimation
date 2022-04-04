from SharedProcessors.const import LINE_WIDTH, FONT_DICT_SMALL, SUB_NAMES
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
from scipy.stats import ttest_rel
import matplotlib as mpl
import numpy as np
from strike_index_tian.Drawer import save_fig, format_plot, metric_sub_mean, rmse_fun
import pingouin as pg
import prettytable as pt
from strike_index_tian.a1_fig7 import draw_sigifi_sign
import warnings
import matplotlib.lines as lines


def draw_f3(rmses, conditions, sigifi_sign_fun):
    def format_ticks():
        ax = plt.gca()
        ax.set_ylabel('Root Mean Square Error (%)', fontdict=FONT_DICT_SMALL, labelpad=5)
        ax.set_ylim(0.04, 0.14)
        ax.set_yticks(np.arange(0.04, 0.15, 0.02))
        ax.set_yticklabels(['4', '6', '8', '10', '12', '14'], fontdict=FONT_DICT_SMALL)
        ax.set_xlim(-0.5, 4.5)
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_xticklabels(['All\nAll', '2.4 m/s\n2.4 m/s', '2.8 m/s\n2.8 m/s', '2.4 m/s\n2.8 m/s', '2.8 m/s\n2.4 m/s'],
                           fontdict=FONT_DICT_SMALL, linespacing=2.2)
        ax.tick_params(axis='x', which='major', pad=10)
        for xtick, color in zip(ax.get_xticklabels(), colors):
            xtick.set_color(color)
        base_x, base_y = -0.15, -0.157
        ax.add_patch(mpl.patches.Rectangle((base_x, base_y), ls='--', width=1.15, height=0.08, ec="gray", fill=False,
                                           transform=ax.transAxes, clip_on=False))
        plt.text(base_x+0.02, base_y + 0.018, 'Training', transform=ax.transAxes, fontdict=FONT_DICT_SMALL)
        ax.add_patch(mpl.patches.Rectangle((base_x, base_y-0.112), ls='--', width=1.15, height=0.08, ec="gray", fill=False,
                                           transform=ax.transAxes, clip_on=False))
        plt.text(base_x+0.02, base_y-0.096, 'Test', transform=ax.transAxes, fontdict=FONT_DICT_SMALL)

    rc('font', family='Arial')
    fig = plt.figure(figsize=(3.54, 2.6))
    base_color = np.array([68, 128, 22]) / 255
    colors = [np.array([0, 0, 0])] + [base_color * x for x in [1.2, 1, 0.8, 0.6]]
    format_plot()
    for i_condition, condition in enumerate(conditions):
        box_ = plt.boxplot(rmses[condition], positions=[i_condition], widths=[0.5], patch_artist=True)
        for field in ['medians', 'whiskers', 'caps', 'boxes']:
            [box_[field][i].set(linewidth=LINE_WIDTH, color=colors[i_condition]) for i in range(len(box_[field]))]
        [box_['fliers'][i].set(marker='D', markeredgecolor=colors[i_condition], markerfacecolor=colors[i_condition], markersize=2.5) for i in range(len(box_['fliers']))]
        box_['medians'][0].set(linewidth=LINE_WIDTH, color=[1, 1, 1])

    plt.tight_layout(rect=[0., 0.07, 1.01, 1.01])
    plt.plot([0.5, 0.5], [0, 1], '--', color='gray')
    # l2 = lines.Line2D([0.31, 0.31], [0.02, 0.81], linestyle='--', transform=fig.transFigure, color='gray')
    # fig.lines.extend([l2])
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
    result_date = '220325'
    condition_and_train_test = {'_24': ('24', '24'), '_28': ('28', '28'),
                                '_tr24te28': ('24', '28'), '_tr28te24': ('28', '24')}
    conditions = list(condition_and_train_test.keys())
    rmses = {condition: metric_sub_mean(result_date, condition, rmse_fun) for condition in conditions}
    result_df = transfer_data_to_long_df(condition_and_train_test, conditions, rmses)
    statistics(result_df, rmses, conditions)
    mean_rmses = [np.mean(rmses[condition]) for condition in conditions]
    print("The RMSE increases were within {:.1f}%".format(100 * (max(mean_rmses) - min(mean_rmses))))
    print("When the training and test data were only from 2.4 m/s trials and 2.8 m/s trials, the RMSE were {:.1f}% and {:.1f}%"
          .format(np.mean(rmses['_24'])*100, np.mean(rmses['_28'])*100))

    conditions = [''] + conditions
    rmses.update({'': metric_sub_mean(result_date, '', rmse_fun)})
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        draw_f3(rmses, conditions, draw_sigifi_sign)
    save_fig('f3', 600)
    plt.show()
