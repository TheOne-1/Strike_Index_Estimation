"""
cross validation
(1) No resampling
(2) Reduced network size
"""
from Evaluation import Evaluation
import matplotlib.pyplot as plt
from keras.layers import *
from ProcessorLRSI import ProcessorLRSI
from keras.models import Model
import pandas as pd
from const import SUB_NAMES
import numpy as np


class ProcessorLRSICrossVali(ProcessorLRSI):
    def __init__(self, sub_and_trials, sensor_sampling_fre, IMU_locations, strike_off_from_IMU=True, do_input_norm=True, do_output_norm=True ):
        super().__init__(sub_and_trials, None,IMU_locations, strike_off_from_IMU,
                         split_train=False, do_input_norm=do_input_norm, do_output_norm=do_output_norm)
        self.test_sub = ""
                                         
    def prepare_data_cross_vali(self, test_set_sub_num=1):
        train_all_data_list = ProcessorLRSI.clean_all_data(self.train_all_data_list, self.sensor_sampling_fre)
        input_list, output_list_LR, output_list_SI = train_all_data_list.get_input_output_list()
        self.channel_num = input_list[0].shape[1] - 1
        output_list = output_list_SI
        sub_id_list = train_all_data_list.get_sub_id_list()
        trial_id_list = train_all_data_list.get_trial_id_list()

        sub_id_set_tuple = tuple(set(sub_id_list))
        sample_num = len(input_list)
        sub_num = len(self.train_sub_and_trials.keys())
        folder_num = int(np.ceil(sub_num / test_set_sub_num))        # the number of cross validation times
        predict_result_df = pd.DataFrame()
        for i_folder in range(folder_num):
            test_id_list = sub_id_set_tuple[test_set_sub_num*i_folder:test_set_sub_num*(i_folder+1)]
            print('\ntest subjects: ')
            for test_id in test_id_list:
                print(SUB_NAMES[test_id])
            input_list_train, input_list_test, output_list_train, output_list_test = [], [], [], []
            for i_sample in range(sample_num):
                if sub_id_list[i_sample] in test_id_list:
                    input_list_test.append(input_list[i_sample])
                    output_list_test.append(output_list[i_sample])
                else:
                    input_list_train.append(input_list[i_sample])
                    output_list_train.append(output_list[i_sample])

            self._x_train, self._x_train_aux = self.convert_input(input_list_train, self.sensor_sampling_fre)
            self._y_train = ProcessorLRSI.convert_output(output_list_train)
            self._x_test, self._x_test_aux = self.convert_input(input_list_test, self.sensor_sampling_fre)
            self._y_test = ProcessorLRSI.convert_output(output_list_test)

            # do input normalization
            if self.do_input_norm:
                self.norm_input()

            if self.do_output_norm:
                self.norm_output()
            self.test_sub = test_id_list[0]
            # y_pred, hp_best = self.tune_nn_model()
            y_pred = self.tune_nn_model()
            y_pred = y_pred.reshape([-1, 1])
            if self.do_output_norm:
                y_pred = self.norm_output_reverse(y_pred)
            pearson_coeff, RMSE, mean_error = Evaluation.plot_nn_result(self._y_test, y_pred, title=SUB_NAMES[test_id_list[0]])

            predict_result_df = Evaluation.insert_prediction_result(
                predict_result_df, SUB_NAMES[test_id_list[0]], pearson_coeff, RMSE, mean_error)#, hp_best)
        Evaluation.export_prediction_result(predict_result_df)










