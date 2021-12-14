import numpy as np
from sklearn.metrics import mean_squared_error
from strike_index_tian.Drawer import save_fig, load_step_data
from sklearn.metrics import r2_score


if __name__ == "__main__":
    si_true, si_pred = load_step_data('211206', '')
    rmses, r2s = [], []
    for si_true_sub, si_pred_sub in zip(si_true, si_pred):
        rmses.append(100 * np.sqrt(mean_squared_error(si_true_sub, si_pred_sub)))
        r2s.append(r2_score(si_true_sub, si_pred_sub))

    print('a root mean square error of {:.1f}\% and a $R^2$ of {:.2f}'.format(
        np.mean(rmses), np.mean(r2s)))
    print('The overall average RMSE was {:.1f} ± {:.1f}\% and the $R^2$ was {:.2f} ± {:.2f}'.format(
        np.mean(rmses), np.std(rmses), np.mean(r2s), np.std(r2s)))



