from const import SUB_NAMES
import pickle
import pandas as pd


""" Print the best paramters """
for sub in SUB_NAMES:
    with open('result_conclusion/211202/hyperparameters/{}.pkl'.format(sub), 'rb') as handle:
        param_set = pickle.load(handle)
        print('{}: {}'.format(sub, param_set))

# """ xxx """
# df = pd.read_csv('result_conclusion/211202/trial_summary/main.csv', index_col=False)
# df = df.append({'Subject Name': 'All Subject Mean', **df.mean()}, ignore_index=True)


