from SharedProcessors.const import SUB_NAMES
import pickle
import numpy as np


""" Print the best paramters """
param_set_all = {}
for sub in SUB_NAMES:
    with open('./result_conclusion/211206/hyperparameters/{}.pkl'.format(sub), 'rb') as handle:
        param_set = pickle.load(handle)
        param_set_all[sub] = param_set
        print('{}: {}'.format(sub, param_set))

for key in list(param_set_all[SUB_NAMES[0]].keys()):
    values = [param_set_all[sub][key] for sub in SUB_NAMES]
    if key == 'LR':
        print('{}: {:0.2e}({:0.2e})'.format(key, np.mean(values), np.std(values)))
    else:
        print('{}: {:.1f}({:.1f})'.format(key, np.mean(values), np.std(values)))

