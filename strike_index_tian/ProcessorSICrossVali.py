"""
cross validation
(1) No resampling
(2) Reduced network size
"""
from SharedProcessors.Evaluation import Evaluation
import pickle
from tensorflow.keras.layers import *
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from keras_tuner.tuners import BayesianOptimization, RandomSearch, Hyperband
from tensorflow.keras import optimizers
from keras_tuner import HyperParameters, Objective
import tensorflow as tf
import random as python_random
import copy
from strike_index_tian.ProcessorSI import ProcessorSI
import pandas as pd
from SharedProcessors.const import SUB_NAMES, MAIN_DATA_PATH, EPOCH_NUM, BATCH_SIZE
import numpy as np
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
        tf.random.set_seed(0)

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

            self._x_train, self._x_train_aux = self.convert_input_samples(input_list_train, self.sensor_sampling_fre)
            self._y_train = ProcessorSI.convert_output(output_list_train)
            self._x_test, self._x_test_aux = self.convert_input_samples(input_list_test, self.sensor_sampling_fre)
            self._y_test = ProcessorSI.convert_output(output_list_test)

            if self.do_input_norm:
                self.norm_input()

            y_pred = self.define_cnn_model(param_set, SUB_NAMES[test_id_list[0]]).reshape([-1, 1])

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
        if not os.path.exists("result_conclusion/{}/model".format(self.test_date)):
            os.makedirs("result_conclusion/{}/model".format(self.test_date))
        if not os.path.exists("result_conclusion/{}/training_log".format(self.test_date)):
            os.makedirs("result_conclusion/{}/training_log".format(self.test_date))

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

    def define_cnn_model(self, param_set, sub):
        main_input_shape = self._x_train.shape
        main_input = Input((main_input_shape[1:]), name='main_input')
        base_size = int(self.vector_len)

        kernel_regu = regularizers.l1(0.01)
        kernel_num = param_set['filters']
        kernel_size_1 = 16
        tower_1 = Conv1D(filters=kernel_num, kernel_size=kernel_size_1, kernel_regularizer=kernel_regu)(main_input)
        tower_1 = MaxPool1D(pool_size=base_size-kernel_size_1+1)(tower_1)

        kernel_size_2 = 4
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
        optimizer = optimizers.Nadam(lr=param_set['LR'], beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=["mean_squared_error"])
        my_evaluator = Evaluation(self._x_train, self._x_test, self._y_train, self._y_test, self._x_train_aux,
                                  self._x_test_aux)
        log_file = 'result_conclusion/{}/training_log/{}.csv'.format(self.test_date, sub)
        y_pred = my_evaluator.evaluate_nn(model, log_file)
        model.save('./result_conclusion/{}/model/{}'.format(self.test_date, sub))
        return y_pred

    @staticmethod
    def define_hp_model(hp):
        base_size = 33
        main_input = Input(shape=(base_size, 6), name='main_input')
        hp_filters = hp.Int("filters", min_value=20, max_value=40, step=1, default=30)
        hp_NN_layer_1_units = hp.Int("NNL1U", min_value=20, max_value=40, step=1, default=30)
        hp_learning_rate = hp.Float("LR", min_value=1e-5, max_value=1e-3, default=1e-4, sampling='log')
        kernel_regu = regularizers.l1(0.01)
        hp_tower_1_kernel_size, hp_tower_2_kernel_size = 16, 4
        tower_1 = Conv1D(filters=hp_filters, kernel_size=hp_tower_1_kernel_size, kernel_regularizer=kernel_regu)(main_input)
        tower_1 = MaxPool1D(pool_size=base_size - hp_tower_1_kernel_size + 1)(tower_1)

        tower_2 = Conv1D(filters=hp_filters, kernel_size=hp_tower_2_kernel_size, kernel_regularizer=kernel_regu)(main_input)
        tower_2 = MaxPool1D(pool_size=base_size-hp_tower_2_kernel_size+1)(tower_2)

        joined_outputs = concatenate([tower_1, tower_2], axis=-1)
        joined_outputs = Activation('relu')(joined_outputs)
        main_outputs = Flatten()(joined_outputs)

        aux_input = Input(shape=(2,), name='aux_input')
        aux_joined_outputs = concatenate([main_outputs, aux_input])
        aux_joined_outputs = Dense(hp_NN_layer_1_units, activation='relu')(aux_joined_outputs)
        aux_joined_outputs = Dense(1, activation='sigmoid')(aux_joined_outputs)
        model = Model(inputs=[main_input, aux_input], outputs=aux_joined_outputs)
        optimizer = optimizers.Nadam(lr=hp_learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=["mean_squared_error"])
        return model

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

        save_path = MAIN_DATA_PATH.split('data\\PhaseIData')[0] + 'codes\job_2022_hyper_search'
        tuner = BayesianOptimization(self.define_hp_model, objective=Objective('mean_squared_error', 'min'), metrics=['mean_squared_error'],
                                     directory=save_path, seed=0, executions_per_trial=1, project_name='tuner_logs',
                                     overwrite=True, max_trials=100)        # !!! HYPER CHANGE
        print('Hyper search')
        tuner.search(x={'main_input': self._x_train, 'aux_input': self._x_train_aux}, y=self._y_train,
                     batch_size=BATCH_SIZE, epochs=EPOCH_NUM, verbose=1,
                     validation_data=({'main_input': self._x_test, 'aux_input': self._x_test_aux}, self._y_test))
        best_hp = tuner.get_best_hyperparameters()[0]
        print(best_hp.values)
        with open(hyper_save_path, 'wb') as handle:
            pickle.dump(best_hp.values, handle)
        return best_hp.values


class ProcessorSICrossValiModelSize(ProcessorSICrossVali):

    def define_cnn_model(self, param_set, sub):
        main_input_shape = self._x_train.shape
        main_input = Input((main_input_shape[1:]), name='main_input')
        base_size = int(self.vector_len)

        kernel_regu = regularizers.l1(0.01)
        kernel_num = int(param_set['filters'] * self.unit_times)
        kernel_size_1 = 16
        tower_1 = Conv1D(filters=kernel_num, kernel_size=kernel_size_1, kernel_regularizer=kernel_regu)(main_input)
        if self.layer_num == 1:
            tower_1 = MaxPool1D(pool_size=base_size-kernel_size_1+1)(tower_1)
        else:
            tower_1 = Conv1D(filters=kernel_num, kernel_size=4, kernel_regularizer=kernel_regu)(tower_1)
            tower_1 = MaxPool1D(pool_size=base_size-kernel_size_1-3)(tower_1)

        kernel_size_2 = 4
        tower_2 = Conv1D(filters=kernel_num, kernel_size=kernel_size_2, kernel_regularizer=kernel_regu)(main_input)
        if self.layer_num == 1:
            tower_2 = MaxPool1D(pool_size=base_size-kernel_size_2+1)(tower_2)
        else:
            tower_2 = Conv1D(filters=kernel_num, kernel_size=4, kernel_regularizer=kernel_regu)(tower_2)
            tower_2 = MaxPool1D(pool_size=base_size-kernel_size_2-3)(tower_2)

        joined_outputs = concatenate([tower_1, tower_2], axis=-1)
        joined_outputs = Activation('relu')(joined_outputs)
        main_outputs = Flatten()(joined_outputs)

        aux_input = Input(shape=(2,), name='aux_input')
        aux_joined_outputs = concatenate([main_outputs, aux_input])
        unit_num = int(param_set['NNL1U'] * self.unit_times)
        for i in range(self.layer_num):
            aux_joined_outputs = Dense(unit_num, activation='relu')(aux_joined_outputs)
        aux_joined_outputs = Dense(1, activation='sigmoid')(aux_joined_outputs)
        model = Model(inputs=[main_input, aux_input], outputs=aux_joined_outputs)
        optimizer = optimizers.Nadam(lr=param_set['LR'], beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=["mean_squared_error"])
        my_evaluator = Evaluation(self._x_train, self._x_test, self._y_train, self._y_test, self._x_train_aux,
                                  self._x_test_aux)
        log_file = 'result_conclusion/{}/training_log/{}.csv'.format(self.test_date, sub)
        y_pred = my_evaluator.evaluate_nn(model, log_file)
        model.save('./result_conclusion/{}/model/{}'.format(self.test_date, sub))
        return y_pred

    def prepare_data_cross_vali(self, unit_times, layer_num, *args, **kwargs):
        self.unit_times = unit_times
        if layer_num not in [1, 2]:
            raise ValueError('Layer number should be 1 or 2.')
        self.layer_num = layer_num
        return super().prepare_data_cross_vali(*args, **kwargs)







