import matplotlib.pyplot as plt
from strike_index_tian.Drawer import save_fig
from const import SUB_NAMES, FONT_SIZE_SMALL, FONT_DICT_SMALL
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder


def load_step_cm(result_date):
    all_df = pd.read_csv('result_conclusion/{}/step_result/main.csv'.format(result_date))
    bins = pd.IntervalIndex.from_tuples([(2/3, 2), (1/3, 2/3), (-1, 1/3)])
    label_encoder = LabelEncoder()
    label_encoder.fit_transform(bins)
    sub_matrix = []
    for sub_name in SUB_NAMES:
        sub_id = SUB_NAMES.index(sub_name)
        sub_df = all_df[all_df['subject id'] == sub_id]
        true_label = label_encoder.transform(pd.cut(sub_df['true SI'], bins))
        pred_label = label_encoder.transform(pd.cut(sub_df['predicted SI'], bins))
        sub_result = confusion_matrix(true_label, pred_label)
        sub_matrix.append(sub_result.astype('float') / sub_result.sum(axis=1)[:, np.newaxis])
    cm = np.mean(np.array(sub_matrix), axis=0)
    return cm


def plot_confusion_matrix(cm, labels_name):
    plt.figure(figsize=(3.54, 3.54))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens, vmin=0, vmax=1)     # cmap=plt.cm.Blues
    ax = plt.gca()
    cbar = plt.colorbar(fraction=0.04)
    cbar.set_ticks([0, .25, .5, .75, 1])
    cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])
    cbar.ax.tick_params(labelsize=FONT_SIZE_SMALL)

    ax.set_xticks(range(3))
    ax.set_xticklabels(labels_name, fontdict=FONT_DICT_SMALL)
    ax.set_yticks(range(3))
    ax.set_yticklabels(labels_name, fontdict=FONT_DICT_SMALL, rotation=90, va='center')
    # plt.ylabel('Strike Pattern: Laboratory Measurement', fontdict=FONT_DICT_SMALL)
    # plt.xlabel('Strike Pattern: A Shoe-Worn IMU', fontdict=FONT_DICT_SMALL)
    plt.ylabel('Measured Strike Pattern', fontdict=FONT_DICT_SMALL)
    plt.xlabel('Classified Strike Pattern', fontdict=FONT_DICT_SMALL)
    cm = cm.T
    for first_index in range(len(cm)):  # 第几行
        for second_index in range(len(cm[first_index])):  # 第几列
            a = cm[first_index][second_index]
            b = "%.1f%%" % (a * 100)
            if first_index == second_index:
                plt.text(first_index, second_index, b, fontsize=FONT_SIZE_SMALL,  color="w", va='center', ha='center')
            else:
                plt.text(first_index, second_index, b, fontsize=FONT_SIZE_SMALL, va='center', ha='center')
    plt.tight_layout()
    save_fig('f3')
    plt.show()


if __name__ == '__main__':
    result_date = '211206'
    cm = load_step_cm(result_date)
    print('correctly classified {:.1f}\%, {:.1f}\%, and {:.1f}\%'.format(100*cm[0, 0], 100*cm[1, 1], 100*cm[2, 2]))
    plot_confusion_matrix(cm, ['Forefoot', 'Midfoot', 'Rearfoot'])



