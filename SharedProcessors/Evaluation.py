import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from numpy import sqrt
from scipy.stats import pearsonr
from keras import optimizers
from keras.callbacks import EarlyStopping
from const import COLORS, FONT_DICT_SMALL, LINE_WIDTH
import pandas as pd
import os


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
        correlation_coeff = pearsonr(y_test, y_pred)[0]
        RMSE = sqrt(mean_squared_error(y_test, y_pred, multioutput='raw_values'))
        errors = y_test - y_pred
        mean_error = np.mean(errors, axis=0)
        if precision:
            correlation_coeff = np.round(correlation_coeff, precision)
            RMSE = np.round(RMSE, precision)
            mean_error = np.round(mean_error, precision)
        return correlation_coeff, RMSE, mean_error

    def evaluate_sklearn(self, model, title=''):
        model.fit(self._x_train, self._y_train)
        y_pred = model.predict(self._x_test)
        R2, RMSE, mean_error = Evaluation._get_all_scores(self._y_test, y_pred, precision=3)

        plt.figure()
        plt.plot(self._y_test, y_pred, 'b.')
        RMSE_str = str(RMSE[0])
        mean_error_str = str(mean_error)
        pearson_coeff = str(pearsonr(self._y_test, y_pred))[1:6]
        plt.title(title + '\ncorrelation: ' + pearson_coeff + '   RMSE: ' + RMSE_str +
                  '  Mean error: ' + mean_error_str)
        plt.xlabel('true value')
        plt.ylabel('predicted value')

    def evaluate_nn(self, model):
        # train NN
        # lr = learning rate, the other params are default values
        optimizer = optimizers.Nadam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        # val_loss = validation loss, patience is the tolerance
        early_stopping_patience = 5
        early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_patience)
        # epochs is the maximum training round, validation split is the size of the validation set,
        # callback stops the training if the validation was not approved
        batch_size = 32  # the size of data that be trained together
        epoch_num = 200
        if self._x_train_aux is not None:
            r = model.fit(x={'main_input': self._x_train, 'aux_input': self._x_train_aux}, y=self._y_train,
                          batch_size=batch_size, epochs=epoch_num, validation_split=0.2, callbacks=[early_stopping],
                          verbose=2)
            n_epochs = len(r.history['loss'])
            # retrain the model if the model did not converge
            while n_epochs < early_stopping_patience + 7:
                print('Epcohs number was {num}, reset weights and retrain'.format(num=n_epochs))
                model.reset_states()
                r = model.fit(x={'main_input': self._x_train, 'aux_input': self._x_train_aux}, y=self._y_train,
                              batch_size=batch_size, epochs=epoch_num, validation_split=0.2, callbacks=[early_stopping],
                              verbose=2)
                n_epochs = len(r.history['loss'])
            y_pred = model.predict(x={'main_input': self._x_test, 'aux_input': self._x_test_aux},
                                   batch_size=batch_size).ravel()
            # print('Final model, loss = {loss}, epochs = {epochs}'.format(loss=r.history['loss'][-1], epochs=len())
        else:
            model.fit(self._x_train, self._y_train, batch_size=batch_size,
                      epochs=epoch_num, validation_split=0.2, callbacks=[early_stopping])
            y_pred = model.predict(self._x_test, batch_size=batch_size).ravel()
        return y_pred

    @staticmethod
    def plot_fpa_result(y_true, y_pred, sub_id):
        # change the shape of data so that no error will be raised during pearsonr analysis
        if y_true.shape != 1:
            y_true = y_true.ravel()
        if y_pred.shape != 1:
            y_pred = y_pred.ravel()

        R2, RMSE, mean_error = Evaluation._get_all_scores(y_true, y_pred, precision=3)
        plt.figure(figsize=(9, 6))
        Evaluation.format_plot()
        Evaluation.format_fpa_axis()
        plt.plot(y_true, y_pred, 'b.')
        mean_error_str = str(mean_error)[:5]
        pearson_coeff = pearsonr(y_true, y_pred)[0]
        plt.title('Mean error: ' + mean_error_str + ' degree', fontdict=FONT_DICT_SMALL)
        plt.savefig('../2_FPA/fpa_figures/FPA_subject_' + str(sub_id) + '.png')
        return pearson_coeff, RMSE, mean_error

    @staticmethod
    def format_plot():
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_tick_params(width=LINE_WIDTH)
        ax.yaxis.set_tick_params(width=LINE_WIDTH)
        ax.spines['left'].set_linewidth(LINE_WIDTH)
        ax.spines['bottom'].set_linewidth(LINE_WIDTH)

    @staticmethod
    def format_fpa_axis():
        ax = plt.gca()
        ax.set_xlim(-20, 65)
        ax.set_xticks(range(-15, 65, 15))
        ax.set_xticklabels(range(-15, 65, 15), fontdict=FONT_DICT_SMALL)
        ax.set_ylim(-20, 65)
        ax.set_yticks(range(-15, 65, 15))
        ax.set_yticklabels(range(-15, 65, 15), fontdict=FONT_DICT_SMALL)
        plt.xlabel('true value (degree)', fontdict=FONT_DICT_SMALL)
        plt.ylabel('predicted value (degree)', fontdict=FONT_DICT_SMALL)

    @staticmethod
    def plot_nn_result_cate_color(y_true, y_pred, category_id, category_names, title=''):
        # change the shape of data so that no error will be raised during pearsonr analysis
        if y_true.shape != 1:
            y_true = y_true.ravel()
        if y_pred.shape != 1:
            y_pred = y_pred.ravel()
        plt.figure()
        R2, RMSE, mean_error = Evaluation._get_all_scores(y_true, y_pred, precision=3)
        RMSE_str = str(RMSE[0])
        mean_error_str = str(mean_error)
        pearson_coeff = str(pearsonr(y_true, y_pred))[1:6]
        title_extended = title + '\ncorrelation: ' + pearson_coeff + '   RMSE: ' + RMSE_str + '  Mean error: ' + mean_error_str
        plt.title(title_extended)
        print(title_extended)
        plt.plot([0, 250], [0, 250], 'r--')
        category_list = set(category_id)
        category_id_array = np.array(category_id)
        plot_handles, plot_names = [], []
        for category_id in list(category_list):
            category_name = category_names[category_id]
            plot_names.append(category_name)
            if 'mini' in category_name:
                plot_pattern = 'x'
            else:
                plot_pattern = '.'
            category_index = np.where(category_id_array == category_id)[0]
            plt_handle, = plt.plot(y_true[category_index], y_pred[category_index], plot_pattern,
                                   color=COLORS[category_id])
            plot_handles.append(plt_handle)
        plt.legend(plot_handles, plot_names)
        plt.xlabel('true value')
        plt.ylabel('predicted value')

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
        # session = K.get_session()
        # for layer in model.layers:
        #     if hasattr(layer, 'kernel_initializer'):
        #         layer.kernel.initializer.run(session=session)

    @staticmethod
    def insert_prediction_result(predict_result_df, sub_name, pearson_coeff, RMSE, mean_error):
        sub_df = pd.DataFrame([[sub_name, pearson_coeff, RMSE[0], mean_error]])
        predict_result_df = predict_result_df.append(sub_df)
        return predict_result_df

    @staticmethod
    def export_prediction_result(predict_result_df):
        predict_result_df.columns = ['subject_name', 'correlation', 'RMSE', 'mean_error']
        predict_result_df.loc[-1] = ['absolute mean', np.mean(predict_result_df['correlation']),
                                     np.mean(predict_result_df['RMSE']), np.mean(abs(predict_result_df['mean_error']))]
        file_path = 'result_conclusion/predict_result_conclusion.csv'
        i_file = 0
        while os.path.isfile(file_path):
            i_file += 1
            file_path = 'result_conclusion/predict_result_conclusion_' + str(i_file) + '.csv'
        predict_result_df.to_csv(file_path, index=False)
