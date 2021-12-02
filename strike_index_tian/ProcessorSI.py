"""
Conv template, improvements:
(1) add input such as subject height, step length, strike occurance time
"""
import matplotlib.pyplot as plt
from AllSubData import AllSubData
import scipy.interpolate as interpo
from const import SUB_NAMES, COLORS, DATA_COLUMNS_XSENS, MOCAP_SAMPLE_RATE, TRIAL_NAMES
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau
from Evaluation import Evaluation
from keras.layers import *
from tensorflow_core.python.keras.models import Model
from tensorflow_core.python.keras import regularizers
from sklearn.model_selection import train_test_split
from const import TRIAL_NAMES
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os


class ProcessorSI:
    def __init__(self, train_sub_and_trials, test_sub_and_trials, imu_locations, strike_off_from_IMU=False,
                 split_train=False, do_input_norm=True, start_ratio=.5, end_ratio=.75, pre_samples=5, post_samples=5, tune_hp=False):
        """

        :param train_sub_and_trials:
        :param test_sub_and_trials:
        :param imu_locations:
        :param strike_off_from_IMU: 0 for from plate, 1 for filtfilt, 2 for lfilter
        :param split_train:
        :param do_input_norm:
        """
        self.train_sub_and_trials = train_sub_and_trials
        self.test_sub_and_trials = test_sub_and_trials
        self.imu_locations = imu_locations
        self.sensor_sampling_fre = MOCAP_SAMPLE_RATE
        self.strike_off_from_IMU = strike_off_from_IMU
        self.split_train = split_train
        self.do_input_norm = do_input_norm
        self.channel_num = 0
        self.param_name = 'SI'
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.pre_samples = pre_samples
        self.post_samples = post_samples
        self.tune_hp = tune_hp
        self.vector_len = self.end_ratio*self.sensor_sampling_fre - self.start_ratio*self.sensor_sampling_fre
        train_all_data = AllSubData(self.train_sub_and_trials, imu_locations, self.sensor_sampling_fre, self.strike_off_from_IMU)
        self.train_all_data_list = train_all_data.get_all_data()
        if test_sub_and_trials is not None:
            test_all_data = AllSubData(self.test_sub_and_trials, imu_locations, self.sensor_sampling_fre, self.strike_off_from_IMU)
            self.test_all_data_list = test_all_data.get_all_data()

    def prepare_data(self):
        train_all_data_list = ProcessorSI.clean_all_data(self.train_all_data_list, self.sensor_sampling_fre)
        input_list, _ ,output_list = train_all_data_list.get_input_output_list()
        self.channel_num = input_list[0].shape[1] - 1
        self._x_train, self._x_train_aux = self.convert_input_samples(input_list, self.sensor_sampling_fre)
        self._y_train = ProcessorSI.convert_output(output_list)

        if not self.split_train:
            test_all_data_list = ProcessorSI.clean_all_data(self.test_all_data_list, self.sensor_sampling_fre)
            input_list,_, output_list = test_all_data_list.get_input_output_list()
            self.test_sub_id_list = test_all_data_list.get_sub_id_list()
            self.test_trial_id_list = test_all_data_list.get_trial_id_list()
            self._x_test, self._x_test_aux = self.convert_input_samples(input_list, self.sensor_sampling_fre)
            self._y_test = ProcessorSI.convert_output(output_list)
        else:
            # split the train, test set from the train data
            self._x_train, self._x_test, self._x_train_aux, self._x_test_aux, self._y_train, self._y_test =\
                train_test_split(self._x_train, self._x_train_aux, self._y_train, test_size=0.33)

        # do input normalization
        if self.do_input_norm:
            self.norm_input()

    # convert the input from list to ndarray
    def convert_input_samples(self, input_all_list, sampling_fre):
        """
        CNN based algorithm improved
        """
        step_num = len(input_all_list)
        step_input = np.zeros([step_num, self.pre_samples + self.post_samples+1, self.channel_num])
        aux_input = np.zeros([step_num, 2])
        for i_step in range(step_num):
            acc_gyr_data = input_all_list[i_step][:, 0:self.channel_num]
            for i_channel in range(self.channel_num):
                strike_sample_num = np.where(input_all_list[i_step][:, -1] == 1)[0]
                step_len = acc_gyr_data.shape[0]
                data_start = int(strike_sample_num - self.pre_samples)
                data_end = int(strike_sample_num + self.post_samples+1)
                step_input[i_step, :, i_channel] = acc_gyr_data[data_start:data_end, i_channel]
                aux_input[i_step, 0] = step_len
                aux_input[i_step, 1] = strike_sample_num

        aux_input = ProcessorSI.clean_aux_input(aux_input)
        return step_input, aux_input

    # convert the input from list to ndarray
    def convert_input(self, input_all_list, sampling_fre):
        """
        CNN based algorithm improved
        """
        step_num = len(input_all_list)
        resample_len = self.sensor_sampling_fre
        data_clip_start, data_clip_end = int(resample_len * self.start_ratio), int(resample_len * self.end_ratio)
        step_input = np.zeros([step_num, data_clip_end - data_clip_start, self.channel_num])
        aux_input = np.zeros([step_num, 2])
        for i_step in range(step_num):
            acc_gyr_data = input_all_list[i_step][:, 0:self.channel_num]
            for i_channel in range(self.channel_num):
                channel_resampled = ProcessorSI.resample_channel(acc_gyr_data[:, i_channel], resample_len)
                step_input[i_step, :, i_channel] = channel_resampled[data_clip_start:data_clip_end]
                step_len = acc_gyr_data.shape[0]
                aux_input[i_step, 0] = step_len
                strike_sample_num = np.where(input_all_list[i_step][:, -1] == 1)[0]
                aux_input[i_step, 1] = strike_sample_num

        aux_input = ProcessorSI.clean_aux_input(aux_input)
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

    @staticmethod
    def clean_all_data(all_sub_data_struct, sensor_sampling_fre):
        i_step = 0
        counter_bad = 0
        input_list, _, output_list = all_sub_data_struct.get_input_output_list()
        sub_list = all_sub_data_struct.get_sub_id_list()
        trial_list = all_sub_data_struct.get_trial_id_list()
        min_time_between_strike_off = int(sensor_sampling_fre * 0.15)
        while i_step < len(all_sub_data_struct):
            # delete steps without a valid strike index
            strikes = np.where(input_list[i_step][:, -1] == 1)[0]
            if np.max(output_list[i_step]) <= 0 or np.max(output_list[i_step]) >= 1:        # !!! count numbers
                if np.max(output_list[i_step]) == 0.0:
                    prntval = np.min(output_list[i_step])
                else:
                    prntval = np.max(output_list[i_step])
                print(f"For {SUB_NAMES[sub_list[i_step]]} and trial {TRIAL_NAMES[trial_list[i_step]]}, the strike index is {prntval}")
                all_sub_data_struct.pop(i_step)
                counter_bad += 1

            # delete steps without a valid strike time
            elif len(strikes) != 1:
                all_sub_data_struct.pop(i_step)
                if "SI" in TRIAL_NAMES[trial_list[i_step]]:
                    print("Bad SI in number of strikes test")
                counter_bad += 1

            # delete a step if the duration between strike and off is too short
            elif not min_time_between_strike_off < input_list[i_step].shape[0] - strikes[0]:
                all_sub_data_struct.pop(i_step)
                if "SI" in TRIAL_NAMES[trial_list[i_step]]:
                    print("Bad SI in strike time test")
                counter_bad += 1

            # delete a step if the strike does not fall into 50% to 85% swing phase
            elif not 0.5 * input_list[i_step].shape[0] < strikes[0] < 0.85 * input_list[i_step].shape[0]:
                all_sub_data_struct.pop(i_step)
                if "SI" in TRIAL_NAMES[trial_list[i_step]]:
                    print("Bad SI in occurance of the strike during a step")
                counter_bad += 1

            else:
                # step number only increase when no pop happens
                i_step += 1
        print(f"There were {counter_bad} bad steps here due to SI problems")
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
        # self.main_max_vals, self.main_min_vals = [], []
        for i_channel in range(channel_num):
            mean_val = np.mean(self._x_train[:, :, i_channel])
            std_val = np.std(self._x_train[:, :, i_channel])
            self._x_train[:, :, i_channel] = (self._x_train[:, :, i_channel] - mean_val) / std_val
            self._x_test[:, :, i_channel] = (self._x_test[:, :, i_channel] - mean_val) / std_val

        if hasattr(self, '_x_train_aux'):
            # MinMaxScaler is more suitable because StandardScalar will make the input greatly differ from each other
            aux_input_scalar = MinMaxScaler()
            self._x_train_aux = aux_input_scalar.fit_transform(self._x_train_aux)
            self._x_test_aux = aux_input_scalar.transform(self._x_test_aux)
            self.aux_max_vals = aux_input_scalar.data_max_.tolist()
            self.aux_min_vals = aux_input_scalar.data_min_.tolist()
