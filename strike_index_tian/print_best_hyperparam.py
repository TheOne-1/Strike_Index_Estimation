from const import SUB_NAMES
import pickle
import pandas as pd


""" Print the best paramters """
for sub in SUB_NAMES:
    with open('result_conclusion/211202/hyperparameters/{}.pkl'.format(sub), 'rb') as handle:
        param_set = pickle.load(handle)
        print('{}: {}'.format(sub, param_set))


