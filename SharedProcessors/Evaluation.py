import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import r2_score, mean_squared_error
from numpy import sqrt
from scipy.stats import pearsonr
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
from const import COLORS, SUB_NAMES
from tensorflow.keras import backend as K
import pandas as pd
import os
from kerastuner.tuners import BayesianOptimization, RandomSearch, Hyperband
from kerastuner import HyperParameter as hp
import time
from MakeModel import define_model
import tensorflow as tf
from pearsonr import pearson_r
from os.path import normpath, join


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
        RMSE = sqrt(mean_squared_error(y_test, y_pred, multioutput='raw_values'))
        errors = y_test - y_pred
        mean_error = np.mean(errors, axis=0)
        if precision:
            R2 = np.round(R2, precision)
            RMSE = np.round(RMSE, precision)
            mean_error = np.round(mean_error, precision)
        return R2, RMSE, mean_error

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
        verbosity = 2
        # train NN
        # lr = learning rate, the other params are default values
        optimizer = optimizers.Nadam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
        model.compile(loss='mse', optimizer=optimizer,metrics=["mse"])
        # model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        # LR_callback = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=.0001)
        
        # val_loss = validation loss, patience is the tolerance
        early_stopping_patience = 50
        early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_patience)
        # epochs is the maximum training round, validation split is the size of the validation set,
        # callback stops the training if the validation was not approved
        batch_size = 128  # the size of data that be trained together
        epoch_num = 1000
        if self._x_train_aux is not None:
            tic = time.time()
            r = model.fit(x={'main_input': self._x_train, 'aux_input': self._x_train_aux}, y=self._y_train,
                          batch_size=batch_size, epochs=epoch_num, validation_split=0.2, callbacks=[early_stopping],
                          verbose=verbosity)
            n_epochs = len(r.history['loss'])
            # retrain the model if the model did not converge
            while n_epochs < early_stopping_patience + 7:
                print('Epcohs number was {num}, reset weights and retrain'.format(num=n_epochs))
                model.reset_states()
                r = model.fit(x={'main_input': self._x_train, 'aux_input': self._x_train_aux}, y=self._y_train,
                              batch_size=batch_size, epochs=epoch_num, validation_split=0.2, callbacks=[early_stopping],
                              verbose=verbosity)
                n_epochs = len(r.history['loss'])
            toc = time.time()
            print("Training time:{}".format(toc-tic))
            y_pred = model.predict(x={'main_input': self._x_test, 'aux_input': self._x_test_aux},
                                   batch_size=batch_size)
            # print('Final model, loss = {loss}, epochs = {epochs}'.format(loss=r.history['loss'][-1], epochs=len())
        else:
            model.fit(self._x_train, self._y_train, batch_size=batch_size,
                      epochs=epoch_num, validation_split=0.2, callbacks=[early_stopping])
            y_pred = model.predict(self._x_test, batch_size=batch_size)
        return y_pred

    def HyperparameterTuneNN(self,test_sub):
        # train NN
        # lr = learning rate, the other params are default values
        # main_input_shape = self._x_train.shape
        
        
        # val_loss = validation loss, patience is the tolerance
        early_stopping_patience = 30
        name = "BayO_SI"
        early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_patience)
        tb_callback = TensorBoard(f'.\\logs\\{name}\\{SUB_NAMES[test_sub]}', update_freq=1)

        # epochs is the maximum training round, validation split is the size of the validation set,
        # callback stops the training if the validation was not approved
        batch_size = 64  # the size of data that be trained together
        epoch_num = 500
        tuner_type = "BO"

        if tuner_type == "BO":
            tuner = BayesianOptimization(define_model, project_name= name + "\\" + str(SUB_NAMES[test_sub]), objective='mse', metrics = ["mse", pearson_r] , max_trials=30, seed=314, executions_per_trial=1)
        elif tuner_type == "Hyperband":
            tuner = Hyperband(hypermodel=define_model, project_name="Test_Sub-"+str(SUB_NAMES[test_sub])+"-Run_2", objective='mse', max_epochs=epoch_num)
        if self._x_train_aux is not None:
            tuner.search(x={'main_input': self._x_train, 'aux_input': self._x_train_aux}, y=self._y_train,
                              batch_size=batch_size, epochs=epoch_num, validation_split=.3,
                              verbose=2, callbacks=[early_stopping,tb_callback])
            best_hp = tuner.get_best_hyperparameters()[0]
            best_model = tuner.hypermodel.build(best_hp)
            best_model.fit(x={'main_input': self._x_train, 'aux_input': self._x_train_aux}, y=self._y_train,
                              batch_size=batch_size, epochs=epoch_num, validation_split=0.2,
                              verbose=2,callbacks=[early_stopping])            
            y_pred = best_model.predict(x={'main_input': self._x_test, 'aux_input': self._x_test_aux},
                                   batch_size=batch_size).ravel()




            # r = model.fit(x={'main_input': self._x_train, 'aux_input': self._x_train_aux}, y=self._y_train,
            #               batch_size=batch_size, epochs=epoch_num, validation_split=0.2, callbacks=[early_stopping],
            #               verbose=2)
            # n_epochs = len(r.history['loss'])
            # retrain the model if the model did not converge
            # while n_epochs < early_stopping_patience + 7:
            #     print('Epcohs number was {num}, reset weights and retrain'.format(num=n_epochs))
            #     model.reset_states()
            #     r = model.fit(x={'main_input': self._x_train, 'aux_input': self._x_train_aux}, y=self._y_train,
            #                   batch_size=batch_size, epochs=epoch_num, validation_split=0.2, callbacks=[early_stopping],
            #                   verbose=2)
            #     n_epochs = len(r.history['loss'])
            # y_pred = model.predict(x={'main_input': self._x_test, 'aux_input': self._x_test_aux},
            #                        batch_size=batch_size).ravel()
            # print('Final model, loss = {loss}, epochs = {epochs}'.format(loss=r.history['loss'][-1], epochs=len())
        # else:
        #     print("This is not implemented here")
        #     exit()
        #     model.fit(self._x_train, self._y_train, batch_size=batch_size,
        #               epochs=epoch_num, validation_split=0.2, callbacks=[early_stopping])
        #     y_pred = model.predict(self._x_test, batch_size=batch_size).ravel()
        return y_pred , best_hp.values
        # return y_pred 

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
        return pearson_coeff, RMSE, mean_error

    @staticmethod
    def plot_nn_result_cate_color(y_true, y_pred, category_id, category_names, title=''):
        # change the shape of data so that no error will be raised during pearsonr analysis
        if title == "Strike Index":
            bounds = 1
        elif title == "Loading Rate":
            bounds = 250
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
        plt.plot([0, bounds], [0, bounds], 'r--')
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
    def insert_prediction_result(predict_result_df, sub_name, pearson_coeff, RMSE, mean_error, hp_best):
        new_data = dict(zip(predict_result_df.columns.values, [sub_name, pearson_coeff, RMSE[0], mean_error] + list(hp_best.values()))) 
        predict_result_df = predict_result_df.append(new_data, ignore_index=True)
        return predict_result_df

    @staticmethod
    def export_prediction_result(predict_result_df,name):
        file_path = 'result_conclusion/predict_result_conclusion_{}.csv'.format(name)
        i_file = 0
        while os.path.isfile(file_path):
            i_file += 1
            file_path = 'result_conclusion/predict_result_conclusion_{}_{}.csv'.format(name, i_file)
        predict_result_df.to_csv(file_path, index=False)
    
    def random_undersampling(self):
        self._x_train 
        self._y_train 
        self._x_train_aux 


