"""
Conv template, improvements:
(1) add input such as subject height, step length, strike occurance time
"""
import sys
sys.path.append("SharedProcessors")
import matplotlib.pyplot as plt
from AllSubData import AllSubData
import scipy.interpolate as interpo
from const import SUB_NAMES, COLORS, DATA_COLUMNS_XSENS, MOCAP_SAMPLE_RATE
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler
from Evaluation import Evaluation
from keras.layers import *
from keras.models import Model
from keras import regularizers
from sklearn.model_selection import train_test_split
from const import TRIAL_NAMES, SUB_NAMES 
import numpy as np
from kerastuner import HyperParameters, BayesianOptimization
import tensorflow as tf
from itertools import compress

gpus = tf.config.list_physical_devices('GPU')
gpu = gpus[0] 
tf.config.experimental.set_memory_growth(gpu, True)


# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class ProcessorLRSI:
    def __init__(self, train_sub_and_trials, test_sub_and_trials, imu_locations, strike_off_from_IMU=False,
                 split_train=False, do_input_norm=True, do_output_norm=False):
        """

        :param train_sub_and_trials:
        :param test_sub_and_trials:
        :param imu_locations:
        :param strike_off_from_IMU: 0 for from plate, 1 for filtfilt, 2 for lfilter
        :param split_train:
        :param do_input_norm:
        :param do_output_norm:
        """
        self.train_sub_and_trials = train_sub_and_trials
        self.test_sub_and_trials = test_sub_and_trials
        self.imu_locations = imu_locations
        self.sensor_sampling_fre = MOCAP_SAMPLE_RATE
        self.strike_off_from_IMU = strike_off_from_IMU
        self.split_train = split_train
        self.do_input_norm = do_input_norm
        self.do_output_norm = do_output_norm
        self.channel_num = 0
        train_all_data = AllSubData(self.train_sub_and_trials, imu_locations,
                                       self.sensor_sampling_fre, self.strike_off_from_IMU)
        self.train_all_data_list = train_all_data.get_all_data()
        if test_sub_and_trials is not None:

            test_all_data = AllSubData(self.test_sub_and_trials, imu_locations,
                                          self.sensor_sampling_fre, self.strike_off_from_IMU)
            self.test_all_data_list = test_all_data.get_all_data()

    def prepare_data(self):
        train_all_data_list = ProcessorLRSI.clean_all_data(self.train_all_data_list, self.sensor_sampling_fre)
        input_list, output_list_LR, output_list_SI = train_all_data_list.get_input_output_list()
        self.channel_num = input_list[0].shape[1] - 1
        self._x_train, self._x_train_aux = self.convert_input(input_list, self.sensor_sampling_fre)
        self._y_LR_train = ProcessorLRSI.convert_output(output_list_LR)
        self._y_SI_train = ProcessorLRSI.convert_output(output_list_SI)
        self._y_train = self._y_SI_train

        if not self.split_train:
            test_all_data_list = ProcessorLRSI.clean_all_data(self.test_all_data_list, self.sensor_sampling_fre)
            input_list, output_list_LR, output_list_SI = test_all_data_list.get_input_output_list()
            self.test_sub_id_list = test_all_data_list.get_sub_id_list()
            self.test_trial_id_list = test_all_data_list.get_trial_id_list()
            self._x_test, self._x_test_aux = self.convert_input(input_list, self.sensor_sampling_fre)
            self._y_LR_test = ProcessorLRSI.convert_output(output_list_LR)
            self._y_SI_test = ProcessorLRSI.convert_output(output_list_SI)
            self._y_test = np.array(tuple(zip(self._y_LR_test,self._y_SI_test)))
            self._y_test = self._y_SI_test
        else:
            # split the train, test set from the train data
            print("This function is not working anymore.")
            exit
            # self._x_train, self._x_test, self._x_train_aux, self._x_test_aux, self._y_train, self._y_test =\
            #     train_test_split(self._x_train, self._x_train_aux, self._y_train, test_size=0.33)

        # do input normalization
        if self.do_input_norm:
            self.norm_input()

        if self.do_output_norm:
            self.norm_output()

    # convert the input from list to ndarray
    def convert_input(self, input_all_list, sampling_fre):
        """
        CNN based algorithm improved
        """
        step_num = len(input_all_list)
        resample_len = self.sensor_sampling_fre
        data_clip_start, data_clip_end = int(resample_len * 0.5), int(resample_len * 0.75)
        step_input = np.zeros([step_num, data_clip_end - data_clip_start, self.channel_num])
        aux_input = np.zeros([step_num, 2])
        for i_step in range(step_num):
            acc_gyr_data = input_all_list[i_step][:, 0:self.channel_num]
            for i_channel in range(self.channel_num):
                channel_resampled = ProcessorLRSI.resample_channel(acc_gyr_data[:, i_channel], resample_len)
                step_input[i_step, :, i_channel] = channel_resampled[data_clip_start:data_clip_end]
                step_len = acc_gyr_data.shape[0]
                aux_input[i_step, 0] = step_len
                strike_sample_num = np.where(input_all_list[i_step][:, -1] == 1)[0]
                aux_input[i_step, 1] = strike_sample_num

        aux_input = ProcessorLRSI.clean_aux_input(aux_input)
        return step_input, aux_input

    @staticmethod
    def clean_aux_input(aux_input):
        """
        replace zeros by the average
        :param aux_input:
        :return:
        """
        # replace zeros
        aux_input_median = np.median(aux_input, axis=0)
        for i_channel in range(aux_input.shape[1]):
            zero_indexes = np.where(aux_input[:, i_channel] == 0)[0]
            aux_input[zero_indexes, i_channel] = aux_input_median[i_channel]
            if len(zero_indexes) != 0:
                print('Zero encountered in aux input. Replaced by the median')
        return aux_input

    def cnn_solution(self):
        self.define_cnn_model_typeCC()
        self.evaluate_cnn_model()
        self.save_cnn_model()
        plt.show()
    
    def cnn_hptune(self):
        # self.define_cnn_model_typeCC()
        self.tune_nn_model()
        # self.save_cnn_model()
        plt.show()

    def define_cnn_model_typeCC(self):
        hp = HyperParameters()
        main_input_shape = self._x_train.shape
        main_input = Input(main_input_shape[1:], name='main_input')
        base_size = int(self.sensor_sampling_fre*0.01)
        hp_filters = hp.Int("filters", min_value=4, max_value=16, step=1, default=10)
        hp_tower_1_kernel_size = hp.Int("T1KS", min_value=10, max_value=16, step=1, default=16)
        hp_tower_3_kernel_size = hp.Int("T3KS", min_value=5, max_value=9, step=1, default=9)
        hp_tower_4_kernel_size = hp.Int("T4KS", min_value=2, max_value=4, step=1, default=2)
        hp_NN_layer_1_units = hp.Int("NNL1U", min_value=10, max_value=30, step=1, default=20)
        hp_NN_layer_2_units = hp.Int("NNL1U", min_value=5, max_value=25, step=1, default=10)
        hp_NN_layer_3_units = hp.Int("NNL1U", min_value=2, max_value=20, step=1, default=5)

        # kernel_init = 'lecun_uniform'
        kernel_regu = regularizers.l2(0.01)
        # for each feature, add 30 * 1 cov kernel
        tower_1 = Conv1D(filters=hp_filters, kernel_size=hp_tower_1_kernel_size*base_size, kernel_regularizer=kernel_regu)(main_input)
        tower_1 = MaxPool1D(pool_size=25*base_size-hp_tower_1_kernel_size*base_size+1)(tower_1)

        # for each feature, add 10 * 1 cov kernel
        tower_3 = Conv1D(filters=hp_filters, kernel_size=hp_tower_3_kernel_size*base_size, kernel_regularizer=kernel_regu)(main_input)
        tower_3 = MaxPool1D(pool_size=25*base_size-hp_tower_3_kernel_size*base_size+1)(tower_3)

        # for each feature, add 4 * 1 cov kernel
        tower_4 = Conv1D(filters=hp_filters, kernel_size=hp_tower_4_kernel_size*base_size, kernel_regularizer=kernel_regu)(main_input)
        tower_4 = MaxPool1D(pool_size=25*base_size-hp_tower_4_kernel_size*base_size+1)(tower_4)

        # for each feature, add 1 * 1 cov kernel
        tower_5 = Conv1D(filters=hp_filters, kernel_size=1, kernel_regularizer=kernel_regu)(main_input)
        tower_5 = MaxPool1D(pool_size=25*base_size-1*base_size+1)(tower_5)

        joined_outputs = Concatenate(axis=-1)([tower_1, tower_3, tower_4, tower_5])
        joined_outputs = Activation('relu')(joined_outputs)
        main_outputs = Flatten()(joined_outputs)

        aux_input = Input(shape=(2,), name='aux_input')
        aux_joined_outputs = Concatenate()([main_outputs, aux_input])

        aux_joined_outputs = Dense(hp_NN_layer_1_units, activation='relu', use_bias=True)(aux_joined_outputs)
        aux_joined_outputs = Dense(hp_NN_layer_2_units, activation='relu')(aux_joined_outputs)
        aux_joined_outputs = Dense(hp_NN_layer_3_units, activation='relu')(aux_joined_outputs)
        aux_joined_outputs = Dense(1, activation='linear')(aux_joined_outputs)
        model = Model(inputs=[main_input, aux_input], outputs=aux_joined_outputs)
        optimizer = optimizers.Nadam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        self.model = model
    
    def define_cnn_model_typeCS(self):
        main_input_shape = self._x_train.shape
        main_input = Input((main_input_shape[1:]), name='main_input')
        base_size = int(self.sensor_sampling_fre*0.01)

        # kernel_init = 'lecun_uniform'
        kernel_regu = regularizers.l2(0.01)
        # for each feature, add 20 * 1 cov kernel
        tower_1 = Conv1D(filters=5, kernel_size=15*base_size, kernel_regularizer=kernel_regu)(main_input)
        tower_1 = MaxPool1D(pool_size=10*base_size+1)(tower_1)

        # for each feature, add 5 * 1 cov kernel
        tower_3 = Conv1D(filters=5, kernel_size=5*base_size, kernel_regularizer=kernel_regu)(main_input)
        tower_3 = MaxPool1D(pool_size=20*base_size+1)(tower_3)

        # for each feature, add 5 * 1 cov kernel
        tower_4 = Conv1D(filters=5, kernel_size=2*base_size, kernel_regularizer=kernel_regu)(main_input)
        tower_4 = MaxPool1D(pool_size=23*base_size+1)(tower_4)

        # for each feature, add 5 * 1 cov kernel
        tower_5 = Conv1D(filters=5, kernel_size=1, kernel_regularizer=kernel_regu)(main_input)
        tower_5 = MaxPool1D(pool_size=50)(tower_5)

        joined_outputs = concatenate([tower_1, tower_3, tower_4, tower_5], axis=-1)
        joined_outputs = Activation('relu')(joined_outputs)
        main_outputs = Flatten()(joined_outputs)

        aux_input = Input(shape=(2,), name='aux_input')
        aux_joined_outputs = concatenate([main_outputs, aux_input])

        aux_joined_outputs_LR = Dense(20, activation='relu')(aux_joined_outputs)
        aux_joined_outputs_LR = Dense(15, activation='relu')(aux_joined_outputs_LR)
        aux_joined_outputs_LR = Dense(10, activation='relu')(aux_joined_outputs_LR)
        aux_joined_outputs_LR = Dense(1, activation='linear')(aux_joined_outputs_LR)

        aux_joined_outputs_SI = Dense(20, activation='relu')(aux_joined_outputs)
        aux_joined_outputs_SI = Dense(15, activation='relu')(aux_joined_outputs_SI)
        aux_joined_outputs_SI = Dense(10, activation='relu')(aux_joined_outputs_SI)
        aux_joined_outputs_SI = Dense(1, activation='linear')(aux_joined_outputs_SI)
        model = Model(inputs=[main_input, aux_input], outputs=[aux_joined_outputs_LR, aux_joined_outputs_SI])
        self.model = model

    def define_cnn_model_typeSS(self):
        main_input_shape = self._x_train.shape
        main_input = Input((main_input_shape[1:]), name='main_input')
        base_size = int(self.sensor_sampling_fre*0.01)

        # kernel_init = 'lecun_uniform'
        kernel_regu = regularizers.l2(0.01)
        # for each feature, add 20 * 1 cov kernel
        tower_1_LR = Conv1D(filters=11, kernel_size=15*base_size, kernel_regularizer=kernel_regu)(main_input)
        tower_1_LR = MaxPool1D(pool_size=10*base_size+1)(tower_1_LR)

        # for each feature, add 5 * 1 cov kernel
        tower_3_LR = Conv1D(filters=11, kernel_size=5*base_size, kernel_regularizer=kernel_regu)(main_input)
        tower_3_LR = MaxPool1D(pool_size=20*base_size+1)(tower_3_LR)

        # for each feature, add 5 * 1 cov kernel
        tower_4_LR = Conv1D(filters=11, kernel_size=2*base_size, kernel_regularizer=kernel_regu)(main_input)
        tower_4_LR = MaxPool1D(pool_size=23*base_size+1)(tower_4_LR)

        # for each feature, add 5 * 1 cov kernel
        tower_5_LR = Conv1D(filters=11, kernel_size=1, kernel_regularizer=kernel_regu)(main_input)
        tower_5_LR = MaxPool1D(pool_size=50)(tower_5_LR)

        joined_outputs_LR = concatenate([tower_1_LR, tower_3_LR, tower_4_LR, tower_5_LR], axis=-1)
        joined_outputs_LR = Activation('relu')(joined_outputs_LR)
        main_outputs_LR = Flatten()(joined_outputs_LR)

        aux_input = Input(shape=(2,), name='aux_input')
        aux_joined_outputs_LR = concatenate([main_outputs_LR, aux_input])

        aux_joined_outputs_LR = Dense(20, activation='relu')(aux_joined_outputs_LR)
        aux_joined_outputs_LR = Dense(15, activation='relu')(aux_joined_outputs_LR)
        aux_joined_outputs_LR = Dense(10, activation='relu')(aux_joined_outputs_LR)
        aux_joined_outputs_LR = Dense(1, activation='linear')(aux_joined_outputs_LR)

        # kernel_init = 'lecun_uniform'
        kernel_regu = regularizers.l2(0.01)
        # for each feature, add 20 * 1 cov kernel
        tower_1_SI = Conv1D(filters=5, kernel_size=15*base_size, kernel_regularizer=kernel_regu)(main_input)
        tower_1_SI = MaxPool1D(pool_size=10*base_size+1)(tower_1_SI)

        # for each feature, add 5 * 1 cov kernel
        tower_3_SI = Conv1D(filters=5, kernel_size=5*base_size, kernel_regularizer=kernel_regu)(main_input)
        tower_3_SI = MaxPool1D(pool_size=20*base_size+1)(tower_3_SI)

        # for each feature, add 5 * 1 cov kernel
        tower_4_SI = Conv1D(filters=5, kernel_size=2*base_size, kernel_regularizer=kernel_regu)(main_input)
        tower_4_SI = MaxPool1D(pool_size=23*base_size+1)(tower_4_SI)

        # for each feature, add 5 * 1 cov kernel
        tower_5_SI = Conv1D(filters=5, kernel_size=1, kernel_regularizer=kernel_regu)(main_input)
        tower_5_SI = MaxPool1D(pool_size=50)(tower_5_SI)

        joined_outputs_SI= concatenate([tower_1_SI, tower_3_SI, tower_4_SI, tower_5_SI], axis=-1)
        joined_outputs_SI= Activation('relu')(joined_outputs_SI)
        main_outputs_SI= Flatten()(joined_outputs_SI)

        aux_input = Input(shape=(2,), name='aux_input')
        aux_joined_outputs_SI= concatenate([main_outputs_SI, aux_input])

        aux_joined_outputs_SI= Dense(20, activation='relu', use_bias=True)(aux_joined_outputs_SI)
        aux_joined_outputs_SI= Dense(15, activation='relu')(aux_joined_outputs_SI)
        aux_joined_outputs_SI= Dense(10, activation='relu')(aux_joined_outputs_SI)
        aux_joined_outputs_SI= Dense(1, activation='linear')(aux_joined_outputs_SI)


        model = Model(inputs=[main_input, aux_input], outputs=[aux_joined_outputs_LR,aux_joined_outputs_SI])
        self.model = model

    def tune_nn_model(self):
        my_evaluator = Evaluation(self._x_train, self._x_test, self._y_train, self._y_test, self._x_train_aux,
                                  self._x_test_aux)
        # y_pred, hp_best = my_evaluator.HyperparameterTuneNN(self.test_sub)
        y_pred = my_evaluator.HyperparameterTuneNN(self.test_sub)
        if self.do_output_norm:
            y_pred = self.norm_output_reverse(y_pred)

        plot_name = 'Strike Index - Sub: ' + str(SUB_NAMES[self.test_sub])
        my_evaluator.plot_nn_result(self._y_test, y_pred, plot_name )

        return y_pred


    def evaluate_cnn_model(self):
        my_evaluator = Evaluation(self._x_train, self._x_test, self._y_train, self._y_test, self._x_train_aux,
                                  self._x_test_aux)
        y_pred = my_evaluator.evaluate_nn(self.model)
        if self.do_output_norm:
            y_pred = self.norm_output_reverse(y_pred)

        if self.split_train:
            my_evaluator.plot_nn_result(self._y_test[:,0], y_pred[:,0], 'Loading Rate')
            my_evaluator.plot_nn_result(self._y_test[:,1], y_pred[:,1], 'Strike Index')
        else:
            # my_evaluator.plot_nn_result_cate_color(self._y_test[:,0], y_pred[0], self.test_trial_id_list, TRIAL_NAMES,
            #                                        'Loading Rate')
            my_evaluator.plot_nn_result_cate_color(self._y_test, y_pred, self.test_trial_id_list, TRIAL_NAMES,
                                                    'Strike Index')

    def save_cnn_model(self, model_name='lr_model'):
        self.model.save(model_name + '.h5', include_optimizer=False)

    def to_generate_figure(self):
        y_pred = self.model.predict(x={'main_input': self._x_test, 'aux_input': self._x_test_aux}).ravel()
        if self.do_output_norm:
            y_pred = self.norm_output_reverse(y_pred)
        return self._y_test, y_pred

    def find_feature(self):
        train_all_data = AllSubData(self.train_sub_and_trials, self.param_name, self.sensor_sampling_fre, self.strike_off_from_IMU)
        train_all_data_list = train_all_data.get_all_data()
        train_all_data_list = ProcessorLRSI.clean_all_data(train_all_data_list, self.sensor_sampling_fre)
        input_list, output_list = train_all_data_list.get_input_output_list()
        x_train, feature_names = self.convert_input(input_list, self.sensor_sampling_fre)
        y_train = ProcessorLRSI.convert_output(output_list)
        sub_id_list = train_all_data_list.get_sub_id_list()
        trial_id_list = train_all_data_list.get_trial_id_list()
        ProcessorLRSI.gait_phase_and_correlation(input_list, y_train, channels=range(self.channel_num))
        ProcessorLRSI.draw_correlation(x_train, y_train, sub_id_list, SUB_NAMES, feature_names)
        ProcessorLRSI.draw_correlation(x_train, y_train, trial_id_list, TRIAL_NAMES, feature_names)
        plt.show()

    @staticmethod
    def gait_phase_and_correlation(input_list, output_array, channels):
        sample_num = len(input_list)
        resample_len = 100
        plt.figure()
        plot_list = []
        for i_channel in channels:
            input_array = np.zeros([sample_num, resample_len])
            for i_sample in range(sample_num):
                channel_data = input_list[i_sample][:, i_channel]
                channel_data_resampled = ProcessorLRSI.resample_channel(channel_data, resample_len)
                input_array[i_sample, :] = channel_data_resampled
            pear_correlations = np.zeros([resample_len])
            for phase in range(resample_len):
                pear_correlations[phase] = stats.pearsonr(input_array[:, phase], output_array)[0]
            channel_plot, = plt.plot(pear_correlations, color=COLORS[i_channel])
            plot_list.append(channel_plot)
        plt.xlabel('gait phase')
        plt.ylabel('correlation')
        plt.legend(plot_list, DATA_COLUMNS_XSENS[0:max(channels)+1])
        plt.grid()
        plt.show()

    @staticmethod
    def draw_correlation(input_array, output_array, category_id_list, category_names, feature_names):
        category_id_set = set(category_id_list)
        category_id_array = np.array(category_id_list)
        for i_feature in range(input_array.shape[1]):
            plt.figure()
            plt.title(feature_names[i_feature])
            plot_list, plot_names = [], []
            i_category = 0
            for category_id in category_id_set:
                category_name = category_names[category_id]
                if 'mini' in category_name:
                    plot_pattern = 'x'
                else:
                    plot_pattern = '.'
                plot_names.append(category_name)
                category_index = np.where(category_id_array == category_id)[0]
                category_plot, = plt.plot(input_array[category_index, i_feature], output_array[category_index],
                                          plot_pattern, color=COLORS[i_category])
                plot_list.append(category_plot)
                i_category += 1
            plt.legend(plot_list, plot_names)

    @staticmethod
    def clean_all_data(all_sub_data_struct, sensor_sampling_fre):
        i_step = 0
        input_list, output_list_LR, output_list_SI = all_sub_data_struct.get_input_output_list()
        min_time_between_strike_off = int(sensor_sampling_fre * 0.15)
        while i_step < len(all_sub_data_struct):
            # delete steps without a valid loading rate
            strikes = np.where(input_list[i_step][:, -1] == 1)[0]
            if np.max(output_list_LR[i_step]) <= 0:
                all_sub_data_struct.pop(i_step)

            # delete steps without a valid strike time
            elif len(strikes) != 1:
                all_sub_data_struct.pop(i_step)

            # delete a step if the duration between strike and off is too short
            elif not min_time_between_strike_off < input_list[i_step].shape[0] - strikes[0]:
                all_sub_data_struct.pop(i_step)

            else:
                # step number only increase when no pop happens
                i_step += 1
        return all_sub_data_struct

    @staticmethod
    def convert_output(output_all_list):
        step_num = len(output_all_list)
        step_output = np.zeros([step_num])
        for i_step in range(step_num):
            step_output[i_step] = np.max(output_all_list[i_step])
        return step_output

    @staticmethod
    def resample_channel(data_array, resampled_len):
        if len(data_array.shape) == 1:
            data_array = data_array.reshape(1, -1)
        data_len = data_array.shape[1]
        data_step = np.arange(0, data_len)
        resampled_step = np.linspace(0, data_len, resampled_len)
        tck, data_step = interpo.splprep(data_array, u=data_step, s=0)
        data_resampled = interpo.splev(resampled_step, tck, der=0)[0]
        return data_resampled

    def norm_input(self):
        channel_num = self._x_train.shape[2]
        # save input scalar parameter
        self.main_max_vals,  self.main_min_vals = [], []
        for i_channel in range(channel_num):
            max_val = np.max(self._x_train[:, :, i_channel]) * 0.99
            min_val = np.min(self._x_train[:, :, i_channel]) * 0.99
            self._x_train[:, :, i_channel] = (self._x_train[:, :, i_channel] - min_val) / (max_val - min_val)
            self._x_test[:, :, i_channel] = (self._x_test[:, :, i_channel] - min_val) / (max_val - min_val)
            self.main_max_vals.append(max_val)
            self.main_min_vals.append(min_val)

        if hasattr(self, '_x_train_aux'):
            # MinMaxScaler is more suitable because StandardScalar will make the input greatly differ from each other
            aux_input_scalar = MinMaxScaler()
            self._x_train_aux = aux_input_scalar.fit_transform(self._x_train_aux)
            self._x_test_aux = aux_input_scalar.transform(self._x_test_aux)
            self.aux_max_vals = aux_input_scalar.data_max_.tolist()
            self.aux_min_vals = aux_input_scalar.data_min_.tolist()

    def norm_output_LRSI(self):
        # self.output_LR_minmax_scalar = MinMaxScaler(feature_range=(1, 3))
        self.output_SI_minmax_scalar = MinMaxScaler(feature_range=(1, 3))
        # self._y_LR_train = self._y_LR_train.reshape(-1, 1)
        self._y_SI_train = self._y_SI_train.reshape(-1, 1)
        # self._y_LR_train = self.output_LR_minmax_scalar.fit_transform(self._y_LR_train)
        self._y_SI_train = self.output_SI_minmax_scalar.fit_transform(self._y_SI_train)
        # self.result_LR_max_vals = self.output_LR_minmax_scalar.data_max_[0]
        # self.result_LR_min_vals = self.output_LR_minmax_scalar.data_min_[0]
        self.result_SI_max_vals = self.output_SI_minmax_scalar.data_max_[0]
        self.result_SI_min_vals = self.output_SI_minmax_scalar.data_min_[0]
        self._y_train = self._y_SI_train

    def norm_output(self):
        # self.output_LR_minmax_scalar = MinMaxScaler(feature_range=(1, 3))
        self.output_minmax_scalar = MinMaxScaler(feature_range=(1, 3))
        # self._y_LR_train = self._y_LR_train.reshape(-1, 1)
        self._y_train = self._y_train.reshape(-1, 1)
        # self._y_LR_train = self.output_LR_minmax_scalar.fit_transform(self._y_LR_train)
        self._y_train = self.output_minmax_scalar.fit_transform(self._y_train)
        # self.result_LR_max_vals = self.output_LR_minmax_scalar.data_max_[0]
        # self.result_LR_min_vals = self.output_LR_minmax_scalar.data_min_[0]
        self.result_max_vals = self.output_minmax_scalar.data_max_[0]
        self.result_min_vals = self.output_minmax_scalar.data_min_[0]

    def norm_output_reverse_LRSI(self, output):
        # output_LR = output[0]
        output_SI = output


        # output_LR = self.output_LR_minmax_scalar.inverse_transform(output_LR.reshape(-1, 1))
        output_SI = self.output_SI_minmax_scalar.inverse_transform(output_SI.reshape(-1, 1))
        # output[0] = output_LR.reshape(-1, )
        output = output_SI.reshape(-1, )
        return output

    def norm_output_reverse(self, output):
        # output_LR = output[0]


        # output_LR = self.output_LR_minmax_scalar.inverse_transform(output_LR.reshape(-1, 1))
        output = self.output_minmax_scalar.inverse_transform(output.reshape(-1, 1))
        # output[0] = output_LR.reshape(-1, )
        output = output.reshape(-1, )
        return output    




