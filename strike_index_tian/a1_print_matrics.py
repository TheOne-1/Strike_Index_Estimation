import numpy as np
from sklearn.metrics import mean_squared_error
from strike_index_tian.Drawer import save_fig, format_plot, metric_sub_mean, rmse_fun
from sklearn.metrics import r2_score
from SharedProcessors.const import SUB_ANTHRO, SUB_NAMES


if __name__ == "__main__":
    result_date = '220325'
    rmses = metric_sub_mean(result_date, '', rmse_fun)
    rmses = [rmse * 100 for rmse in rmses]
    r2s = metric_sub_mean(result_date, '', r2_score)
    print('a root mean square error of {:.1f}\% and a $R^2$ of {:.2f}'.format(
        np.mean(rmses), np.mean(r2s)))
    print('The overall average RMSE was {:.1f} ± {:.1f}\% and the $R^2$ was {:.2f} ± {:.2f}'.format(
        np.mean(rmses), np.std(rmses), np.mean(r2s), np.std(r2s)))

    anthro = []
    for field in ['AGE', 'BH', 'BW', 'MA']:
        anthro_subs = [SUB_ANTHRO[name][field] for name in SUB_NAMES]
        anthro.append(np.mean(anthro_subs))
        anthro.append(np.std(anthro_subs))

    print('age: {:.1f} ± {:.1f} years; height: {:.2f} ± {:.2f} m; weight: {:.1f} ± {:.1f} kg; weekly mileage: {:.1f} ± {:.1f} km'.format(*anthro))

