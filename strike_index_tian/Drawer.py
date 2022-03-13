from const import SI_TRIALS, LINE_WIDTH, SUB_NAMES, FONT_SIZE, FONT_DICT, TRIAL_NAMES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as lines
from sklearn.metrics import mean_squared_error


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


def load_step_data(result_date, test_name):
    all_df = pd.read_csv('result_conclusion/{}/step_result/main{}.csv'.format(result_date, test_name))
    si_true, si_pred = {}, {}
    for sub_name in SUB_NAMES:
        si_true_sub, si_pred_sub = [], []
        sub_id = SUB_NAMES.index(sub_name)
        sub_df = all_df[all_df['subject id'] == sub_id]
        for i_trial in list(set(sub_df['trial id'])):
            trial_df = sub_df[sub_df['trial id'] == i_trial]
            si_true_sub.append(trial_df['true SI'].values)
            si_pred_sub.append(trial_df['predicted SI'].values)
        si_true[sub_name], si_pred[sub_name] = si_true_sub, si_pred_sub
    return si_true, si_pred


def metric_sub_mean(result_date, test_name, metric_fun):
    si_true, si_pred = load_step_data(result_date, test_name)
    metric_sub = []
    for sub_name in list(si_true.keys()):
        si_true_sub, si_pred_sub = si_true[sub_name], si_pred[sub_name]
        metric_trial = [metric_fun(si_true_trial, si_pred_trial) for si_true_trial, si_pred_trial in zip(si_true_sub, si_pred_sub)]
        metric_sub.append(np.mean(metric_trial))
    return metric_sub


def rmse_fun(true, pred):
    return np.sqrt(mean_squared_error(true, pred))


def cohen_d(x,y):
    return (np.mean(x) - np.mean(y)) / np.sqrt((np.std(x, ddof=1) ** 2 + np.std(y, ddof=1) ** 2) / 2.0)

