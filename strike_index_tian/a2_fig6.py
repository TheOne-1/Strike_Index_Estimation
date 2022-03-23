import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from SharedProcessors.const import LINE_WIDTH, FONT_DICT_SMALL, SUB_NAMES, FONT_SIZE_SMALL
from Drawer import format_plot, save_fig
from sklearn.metrics import r2_score, mean_squared_error


def set_example_bar_ticks():
    ax = plt.gca()
    min_val, max_val = 0, 1
    ax.set_xlim(min_val, max_val)
    ax.set_xticks(np.arange(min_val, max_val + 0.1, 0.2))
    ax.set_xticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontdict=FONT_DICT_SMALL)
    ax.set_xlabel('Strike Index: Laboratory Measurement', fontdict=FONT_DICT_SMALL)

    ax.set_ylim(min_val, max_val)
    ax.set_yticks(np.arange(min_val, max_val + 0.1, 0.2))
    ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontdict=FONT_DICT_SMALL)
    ax.set_ylabel('Strike Index: A Shoe-Worn IMU (%)', fontdict=FONT_DICT_SMALL)


def show_each_pair(result_all_df, sub_id):
    sub_df = result_all_df[result_all_df['subject id'] == sub_id]
    si_true, si_pred = sub_df['true SI'], sub_df['predicted SI']
    plt.figure(figsize=(3.54, 3.54))
    format_plot()
    sc = plt.scatter(si_true, si_pred, s=40, marker='.', alpha=0.7, edgecolors='none',
                     color=np.array([53, 128, 57])/255)
    set_example_bar_ticks()

    coef = np.polyfit(si_true, si_pred, 1)
    poly1d_fn = np.poly1d(coef)
    black_line, = plt.plot([0, 1], poly1d_fn([0, 1]), color='black', linewidth=LINE_WIDTH)
    ax = plt.gca()
    rmse = 100 * np.sqrt(mean_squared_error(si_true, si_pred))
    R2 = r2_score(si_true, si_pred)
    plt.text(0.508, 0.135, 'RMSE = {:3.1f}%'.format(rmse), fontdict=FONT_DICT_SMALL, transform=ax.transAxes)
    plt.text(0.578, 0.08, '$R^2$ = {:4.2f}'.format(R2), fontdict=FONT_DICT_SMALL, transform=ax.transAxes)
    plt.text(0.6, 0.03, '$y$ = {:4.2f}$x$ + {:4.2f}'.format(coef[0], coef[1]), fontdict=FONT_DICT_SMALL, transform=ax.transAxes)
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.legend([sc, black_line], ['Result of Each Step', 'Regression Line'], fontsize=FONT_SIZE_SMALL,
               frameon=False, bbox_to_anchor=(0.7, 0.99))
    print('Subject {}, RMSE: {:.2f}%, R2: {:.3f}'.format(sub_id+1, rmse, R2))
    print('The RMSE and $R^2$ of this subject were {:.1f}\% and {:.2f}'.format(rmse, R2))
    save_fig('f6', 600)


if __name__ == "__main__":
    result_all_df = pd.read_csv('result_conclusion/{}/step_result/main.csv'.format('211206'))
    # for sub_id in range(len(SUB_NAMES)):
    #     show_each_pair(result_all_df, sub_id)
    show_each_pair(result_all_df, 5)        # could be from 0, 2, 5, 9, 10
    plt.show()

