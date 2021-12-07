"""
cross validation
(1) No resampling
(2) Reduced network size
"""
from Evaluation import Evaluation
from SharedProcessors.MakeModel import define_model
import pickle
from tensorflow.keras.layers import *
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from kerastuner.tuners import BayesianOptimization, RandomSearch, Hyperband
from tensorflow.keras import optimizers
from kerastuner import HyperParameters, Objective
import tensorflow as tf
import random as python_random
import copy
from .ProcessorSI import ProcessorSI
import pandas as pd
from const import SUB_NAMES
import numpy as np
from keras import backend as K
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import os
from random import sample


class ProcessorSICrossVali(ProcessorSI):
    def __init__(self, sub_and_trials, imu_locations, test_date, strike_off_from_IMU=True, do_input_norm=True,
                 start_ratio=.5, end_ratio=.75, pre_samples=5, post_samples=5, tune_hp=False):
        super().__init__(sub_and_trials, None, imu_locations, strike_off_from_IMU, split_train=False, do_input_norm=do_input_norm, 
                         start_ratio=start_ratio, end_ratio=end_ratio, pre_samples=pre_samples,
                         post_samples=post_samples, tune_hp=tune_hp)
        self.test_date = test_date
        self.fix_seed()

    @staticmethod
    def fix_seed():
        os.environ['PYTHONHASHSEED'] = str(0)
        np.random.seed(0)
        python_random.seed(0)
        tf.random.set_random_seed(0)
        # session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        # sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        # K.set_session(sess)

    def prepare_data_cross_vali(self, test_name, test_set_sub_num=1, start_ratio=.5, end_ratio=.7,
                                pre_samples=12, post_samples=20, trials=[2, 5, 9, 12],
                                trials_test=None, train_num=9):
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.pre_samples = pre_samples
        self.post_samples = post_samples
        self.vector_len = (self.pre_samples + self.post_samples + 1)
        train_all_data_list = ProcessorSI.clean_all_data(self.train_all_data_list, self.sensor_sampling_fre)
        input_list, _, output_list = train_all_data_list.get_input_output_list()
        sub_id_list = train_all_data_list.get_sub_id_list()
        trial_id_list = train_all_data_list.get_trial_id_list()
        self.channel_num = input_list[0].shape[1] - 1
        sub_id_set_tuple = tuple(set(sub_id_list))
        sample_num = len(input_list)
        trials_train = trials
        sub_num = len(self.train_sub_and_trials.keys())
        folder_num = int(np.ceil(sub_num / test_set_sub_num)) # the number of cross validation times
        predict_result_df = pd.DataFrame()
        predicted_value_df = pd.DataFrame()  # save all the predicted values in case reviewer ask for more analysis
        self.check_and_create_folders()
        for i_folder in range(folder_num):
            test_id_list = sub_id_set_tuple[test_set_sub_num*i_folder:test_set_sub_num*(i_folder+1)]
            print('\ntest subjects: ')
            for test_id in test_id_list:
                print(SUB_NAMES[test_id] + " who is sub number {} and is {} of {}".format(test_id,i_folder+1, folder_num))
            if trials_test is None:
                trials_test = trials_train
            train_id_list = list(sub_id_set_tuple)
            [train_id_list.remove(x) for x in test_id_list]
            train_id_list = sample(train_id_list, train_num)
            print(str([SUB_NAMES[i] for i in train_id_list]) + " and their numbers are " + str(train_id_list))

            hyper_save_path = 'result_conclusion/{}/hyperparameters/{}.pkl'.format(self.test_date, SUB_NAMES[test_id_list[0]])
            if self.tune_hp:
                if len(train_id_list) < 6:
                    vali_id_list = sample(train_id_list, 1)
                else:
                    vali_id_list = sample(train_id_list, 3)
                train_during_tuning_id_list = copy.deepcopy(train_id_list)
                [train_during_tuning_id_list.remove(x) for x in vali_id_list]
                param_set = self.tune_nn_model(train_during_tuning_id_list, vali_id_list, input_list, output_list,
                                               trials_train, trials_test, trial_id_list, sub_id_list, hyper_save_path)
            else:
                with open(hyper_save_path, 'rb') as handle:
                    param_set = pickle.load(handle)

            input_list_train, input_list_test, output_list_train, output_list_test, test_trial_ids = [], [], [], [], []
            for i_sample in range(sample_num):
                if int(trial_id_list[i_sample]) in trials_train and sub_id_list[i_sample] in train_id_list:
                    input_list_train.append(input_list[i_sample])
                    output_list_train.append(output_list[i_sample])
                elif int(trial_id_list[i_sample]) in trials_test and sub_id_list[i_sample] in test_id_list:
                    input_list_test.append(input_list[i_sample])
                    output_list_test.append(output_list[i_sample])
                    test_trial_ids.append(trial_id_list[i_sample])
            print("Train_size"+str(len(output_list_train)))
            print("test_size"+str(len(output_list_test)))

            self._x_train, self._x_train_aux = self.convert_input_samples(input_list_train, self.sensor_sampling_fre)
            self._y_train = ProcessorSI.convert_output(output_list_train)
            self._x_test, self._x_test_aux = self.convert_input_samples(input_list_test, self.sensor_sampling_fre)
            self._y_test = ProcessorSI.convert_output(output_list_test)

            if self.do_input_norm:
                self.norm_input()

            y_pred = self.define_cnn_model(param_set).reshape([-1, 1])

            pearson_coeff, RMSE, mean_error = Evaluation.plot_nn_result(self._y_test, y_pred, title=SUB_NAMES[test_id_list[0]])
            discrete_prediction_results = pd.DataFrame(columns=["true","pred"])
            discrete_prediction_results.true = self._y_test
            discrete_prediction_results.pred = y_pred

            predicted_value_df = self.save_all_predicted_values(predicted_value_df, self._y_test, y_pred,
                                                                test_id_list[0], test_trial_ids)
            if predict_result_df.empty:
                predict_result_df = pd.DataFrame(columns=["Subject Name", "correlation", "RMSE", "Mean Error"])
            predict_result_df = Evaluation.insert_prediction_result(
                predict_result_df, SUB_NAMES[test_id_list[0]], pearson_coeff, RMSE, mean_error)

        Evaluation.export_predicted_values(predicted_value_df, self.test_date, test_name)
        Evaluation.export_prediction_result(predict_result_df, self.test_date, test_name)
        self.multipage("result_conclusion/{}/charts/{}.pdf".format(self.test_date, test_name))
        plt.show()
        predict_result_df = predict_result_df.replace({",":""}, regex=True)
        return np.nanmean(predict_result_df.correlation.astype("float"))

    def check_and_create_folders(self):
        if not os.path.exists("result_conclusion/{}".format(self.test_date)):
            os.makedirs("result_conclusion/{}".format(self.test_date))
        if not os.path.exists("result_conclusion/{}/step_result".format(self.test_date)):
            os.makedirs("result_conclusion/{}/step_result".format(self.test_date))
        if not os.path.exists("result_conclusion/{}/trial_summary".format(self.test_date)):
            os.makedirs("result_conclusion/{}/trial_summary".format(self.test_date))
        if not os.path.exists("result_conclusion/{}/charts".format(self.test_date)):
            os.makedirs("result_conclusion/{}/charts".format(self.test_date))
        if not os.path.exists("result_conclusion/{}/hyperparameters".format(self.test_date)):
            os.makedirs("result_conclusion/{}/hyperparameters".format(self.test_date))

    @staticmethod
    def save_all_predicted_values(predicted_value_df, y_true, y_pred, sub_id, test_trial_ids):
        data_len = len(test_trial_ids)
        test_sub_ids_np = np.full([data_len], sub_id)
        test_trial_ids_np = np.array(test_trial_ids)
        current_df = pd.DataFrame(np.column_stack([test_sub_ids_np, test_trial_ids_np, y_true, y_pred]))
        predicted_value_df = predicted_value_df.append(current_df)
        return predicted_value_df

    @staticmethod
    def multipage(filename, figs=None, dpi=200):
        pp = PdfPages(filename)
        if figs is None:
            figs = [plt.figure(n) for n in plt.get_fignums()]
        for fig in figs:
            fig.savefig(pp, format='pdf')
            fig.clear()
            plt.close(fig)
        pp.close()

    def define_cnn_model(self, param_set):
        main_input_shape = self._x_train.shape
        main_input = Input((main_input_shape[1:]), name='main_input')
        base_size = int(self.vector_len)

        kernel_regu = regularizers.l2(0.01)
        kernel_num = param_set['filters']
        kernel_size_1 = param_set['T1KS']
        tower_1 = Conv1D(filters=kernel_num, kernel_size=kernel_size_1, kernel_regularizer=kernel_regu)(main_input)
        tower_1 = MaxPool1D(pool_size=base_size-kernel_size_1+1)(tower_1)

        kernel_size_2 = param_set['T2KS']
        tower_2 = Conv1D(filters=kernel_num, kernel_size=kernel_size_2, kernel_regularizer=kernel_regu)(main_input)
        tower_2 = MaxPool1D(pool_size=base_size-kernel_size_2+1)(tower_2)

        joined_outputs = concatenate([tower_1, tower_2], axis=-1)

        joined_outputs = Activation('relu')(joined_outputs)
        main_outputs = Flatten()(joined_outputs)

        aux_input = Input(shape=(2,), name='aux_input')
        aux_joined_outputs = concatenate([main_outputs, aux_input])
        aux_joined_outputs = Dense(param_set['NNL1U'], activation='relu')(aux_joined_outputs)
        aux_joined_outputs = Dense(1, activation='sigmoid')(aux_joined_outputs)
        model = Model(inputs=[main_input, aux_input], outputs=aux_joined_outputs)
        optimizer = optimizers.Nadam(lr=param_set['LR'], beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=["mean_squared_error"])
        my_evaluator = Evaluation(self._x_train, self._x_test, self._y_train, self._y_test, self._x_train_aux,
                                  self._x_test_aux)
        y_pred = my_evaluator.evaluate_nn(model)
        return y_pred

    def tune_nn_model(self, train_during_tuning_id_list, vali_id_list, input_list, output_list, trials_train, trials_test,
                      trial_id_list, sub_id_list, hyper_save_path):
        input_list_train, input_list_vali, output_list_train, output_list_vali = [], [], [], []
        sample_num = len(input_list)
        for i_sample in range(sample_num):
            if int(trial_id_list[i_sample]) in trials_train and sub_id_list[i_sample] in train_during_tuning_id_list:
                input_list_train.append(input_list[i_sample])
                output_list_train.append(output_list[i_sample])
            elif int(trial_id_list[i_sample]) in trials_test and sub_id_list[i_sample] in vali_id_list:
                input_list_vali.append(input_list[i_sample])
                output_list_vali.append(output_list[i_sample])

        self._x_train, self._x_train_aux = self.convert_input_samples(input_list_train, self.sensor_sampling_fre)
        self._y_train = ProcessorSI.convert_output(output_list_train)
        self._x_test, self._x_test_aux = self.convert_input_samples(input_list_vali, self.sensor_sampling_fre)
        self._y_test = ProcessorSI.convert_output(output_list_vali)

        if self.do_input_norm:
            self.norm_input()

        batch_size = 64  # the size of data that be trained together
        epoch_num = 100
        tuner = BayesianOptimization(define_model, objective=Objective('mean_squared_error', 'min'), metrics=['mean_squared_error'],
                                     directory='J:\Projects\HuaWeiProject\codes\job_2022_hyper_search',
                                     seed=0, executions_per_trial=1, project_name='tuner_logs',
                                     overwrite=True, max_trials=10)
        # tuner = RandomSearch(define_model, objective=Objective('mean_squared_error', 'min'), metrics=['mean_squared_error'],
        #                      directory='J:\Projects\HuaWeiProject\codes\job_2022_hyper_search',
        #                      seed=0, executions_per_trial=1, project_name='tuner_logs',
        #                      overwrite=True, max_trials=2)
        print('Hyper search')
        tuner.search(x={'main_input': self._x_train, 'aux_input': self._x_train_aux}, y=self._y_train,
                     batch_size=batch_size, epochs=epoch_num, verbose=1,
                     validation_data=({'main_input': self._x_test, 'aux_input': self._x_test_aux}, self._y_test))
        best_hp = tuner.get_best_hyperparameters()[0]
        print(best_hp.values)
        with open(hyper_save_path, 'wb') as handle:
            pickle.dump(best_hp.values, handle)
        return best_hp.values

    def define_cnn_model_zas_bl(self, param_set):
        """To get Zach's accuracy"""
        main_input_shape = self._x_train.shape
        main_input = Input((main_input_shape[1:]), name='main_input')
        base_size = int(self.vector_len)

        kernel_regu = regularizers.l2(0.01)
        kernel_num = 20
        kernel_size_1 = 19
        tower_1 = Conv1D(filters=kernel_num, kernel_size=kernel_size_1, kernel_regularizer=kernel_regu)(main_input)
        tower_1 = MaxPool1D(pool_size=base_size-kernel_size_1+1)(tower_1)

        kernel_size_2 = 2
        tower_2 = Conv1D(filters=kernel_num, kernel_size=kernel_size_2, kernel_regularizer=kernel_regu)(main_input)
        tower_2 = MaxPool1D(pool_size=base_size-kernel_size_2+1)(tower_2)

        joined_outputs = concatenate([tower_1, tower_2], axis=-1)

        joined_outputs = Activation('relu')(joined_outputs)
        main_outputs = Flatten()(joined_outputs)

        aux_input = Input(shape=(2,), name='aux_input')
        aux_joined_outputs = concatenate([main_outputs, aux_input])
        aux_joined_outputs = Dense(32, activation='relu')(aux_joined_outputs)
        aux_joined_outputs = Dense(1, activation='sigmoid')(aux_joined_outputs)
        model = Model(inputs=[main_input, aux_input], outputs=aux_joined_outputs)
        optimizer = optimizers.Nadam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=["mean_squared_error"])
        my_evaluator = Evaluation(self._x_train, self._x_test, self._y_train, self._y_test, self._x_train_aux,
                                  self._x_test_aux)
        y_pred = my_evaluator.evaluate_nn(model)
        return y_pred






