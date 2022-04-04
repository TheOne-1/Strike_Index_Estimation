import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import r2_score, mean_squared_error as mse
from tensorflow.keras.callbacks import CSVLogger
from numpy import sqrt
from scipy.stats import pearsonr
from tensorflow.keras.callbacks import EarlyStopping
from SharedProcessors.const import COLORS, EPOCH_NUM, BATCH_SIZE
import collections
import csv


class MyCSVLogger(CSVLogger):
    def __init__(self, filename, separator=',', append=False):
        super().__init__(filename, separator, append)
        super().on_train_begin()

    def on_train_begin(self, logs=None):
        pass

    def add_row(self, y_true_train, y_pred_train, y_true_test, y_pred_test):
        logs = {'loss': 0, 'mean_squared_error': mse(y_true_train, y_pred_train),
                'val_loss': 0, 'val_mean_squared_error': mse(y_true_test, y_pred_test)}
        if self.keys is None:
            self.keys = sorted(logs.keys())
        if not self.writer:
            class CustomDialect(csv.excel):
                delimiter = self.sep

            fieldnames = ['epoch'] + self.keys

            self.writer = csv.DictWriter(
                self.csv_file,
                fieldnames=fieldnames,
                dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()
        row_dict = collections.OrderedDict({'epoch': -1})
        row_dict.update((key, logs[key]) for key in logs.keys())
        self.writer.writerow(row_dict)
        self.csv_file.flush()


class Evaluation:
    def __init__(self, x_train, x_test, y_train, y_test, x_train_aux=None, x_test_aux=None):
        self._x_train = x_train
        self._x_test = x_test
        self._y_train = y_train
        self._y_test = y_test
        self._x_train_aux = x_train_aux
        self._x_test_aux = x_test_aux

    @staticmethod
    def _get_all_scores(y_test, y_pred, precision=None):
        R2 = r2_score(y_test, y_pred, multioutput='raw_values')
        RMSE = sqrt(mse(y_test, y_pred, multioutput='raw_values'))
        errors = y_test - y_pred
        mean_error = np.mean(errors, axis=0)
        if precision:
            R2 = np.round(R2, precision)
            RMSE = np.round(RMSE, precision)
            mean_error = np.round(mean_error, precision)
        return R2, RMSE, mean_error

    def evaluate_nn(self, model, log_file):
        callbacks = None
        if log_file is not None:
            my_logger = MyCSVLogger(log_file, append=False, separator=',')
            callbacks = [my_logger]
            y_pred_train_pretraining = model.predict(x={'main_input': self._x_train, 'aux_input': self._x_train_aux},
                                                     batch_size=BATCH_SIZE)
            y_pred_test_pretraining = model.predict(x={'main_input': self._x_test, 'aux_input': self._x_test_aux},
                                                    batch_size=BATCH_SIZE)
            # plt.figure()
            # plt.plot(self._y_train, y_pred_train_pretraining, '.')
            # plt.show()
            my_logger.add_row(self._y_train, y_pred_train_pretraining, self._y_test, y_pred_test_pretraining)
        r = model.fit(x={'main_input': self._x_train, 'aux_input': self._x_train_aux}, y=self._y_train,
                      batch_size=BATCH_SIZE, epochs=EPOCH_NUM, verbose=1, callbacks=callbacks,
                      validation_data=({'main_input': self._x_test, 'aux_input': self._x_test_aux}, self._y_test))
        y_pred = model.predict(x={'main_input': self._x_test, 'aux_input': self._x_test_aux},
                               batch_size=BATCH_SIZE)
        return y_pred

    @staticmethod
    def plot_nn_result(y_true, y_pred, title=''):
        # change the shape of data so that no error will be raised during pearsonr analysis
        if y_true.shape != 1:
            y_true = y_true.ravel()
        if y_pred.shape != 1:
            y_pred = y_pred.ravel()

        R2, RMSE, mean_error = Evaluation._get_all_scores(y_true, y_pred, precision=3)
        plt.figure()
        plt.plot(y_true, y_pred, 'b.')
        plt.plot([0, 1], [0, 1], 'r--')
        RMSE_str = str(RMSE[0])
        mean_error_str = str(mean_error)
        pearson_coeff = str(pearsonr(y_true, y_pred))[1:6]
        plt.title(title + '\np_correlation: ' + pearson_coeff + '   RMSE: '
                  + RMSE_str + '  Mean error: ' + mean_error_str)
        plt.xlabel('true value')
        plt.ylabel('predicted value')
        return float(pearson_coeff), RMSE, mean_error

    @staticmethod
    def plot_continuous_result(y_true, y_pred, title=''):
        # change the shape of data so that no error will be raised during pearsonr analysis
        R2, RMSE, mean_error = Evaluation._get_all_scores(y_true, y_pred, precision=3)
        plt.figure()
        plot_true, = plt.plot(y_true[:2000])
        plot_pred, = plt.plot(y_pred[:2000])
        RMSE_str = str(RMSE[0])
        mean_error_str = str(mean_error)
        pearson_coeff = str(pearsonr(y_true, y_pred))[1:6]
        plt.title(title + '\ncorrelation: ' + pearson_coeff + '   RMSE: ' + RMSE_str +
                  '  Mean error: ' + mean_error_str)
        plt.legend([plot_true, plot_pred], ['true values', 'predicted values'])
        plt.xlabel('Sample number')
        plt.ylabel('GRF (body weight)')

    @staticmethod
    def reset_weights(model):
        model.reset_states()

    @staticmethod
    def insert_prediction_result(predict_result_df, sub_name, pearson_coeff, RMSE, mean_error):
        new_data = dict(zip(predict_result_df.columns.values, [sub_name, pearson_coeff, RMSE[0], mean_error]))
        predict_result_df = predict_result_df.append(new_data, ignore_index=True)
        return predict_result_df

    @staticmethod
    def export_prediction_result(predict_result_df, test_date, name):
        predict_result_df = predict_result_df.append({'Subject Name': 'All Subject Mean', **predict_result_df.mean()}, ignore_index=True)
        file_path = 'result_conclusion/{}/trial_summary/{}.csv'.format(test_date, name)
        predict_result_df.to_csv(file_path, index=False)

    @staticmethod
    def export_predicted_values(predicted_value_df, test_date, test_name):
        predicted_value_df.columns = ['subject id', 'trial id', 'true SI', 'predicted SI']
        file_path = 'result_conclusion/' + test_date + '/step_result/' + test_name + '.csv'
        predicted_value_df.to_csv(file_path, index=False)


