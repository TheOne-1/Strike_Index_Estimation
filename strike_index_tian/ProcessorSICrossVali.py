"""
cross validation
(1) No resampling
(2) Reduced network size
"""
from Evaluation import Evaluation
import matplotlib.pyplot as plt
from tensorflow.keras.layers import *
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from .ProcessorSI import ProcessorSI
import pandas as pd
from const import SUB_NAMES
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import os
from random import sample


class ProcessorSICrossVali(ProcessorSI):
    def __init__(self, sub_and_trials, imu_locations, test_date, strike_off_from_IMU=True, do_input_norm=True, do_output_norm=True, start_ratio=.5, end_ratio=.75, pre_samples=5, post_samples=5, tune_hp=False):
        super().__init__(sub_and_trials, None, imu_locations, strike_off_from_IMU, split_train=False, do_input_norm=do_input_norm, 
                         do_output_norm=do_output_norm, start_ratio=start_ratio, end_ratio=end_ratio, pre_samples=pre_samples, post_samples=post_samples, tune_hp=tune_hp)
        self.test_date = test_date

    def prepare_data_cross_vali(self, test_name, test_set_sub_num=1, start_ratio=.5, end_ratio=.7,
                                pre_samples=12, post_samples=20, trials=[1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13],
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
        for i_folder in range(folder_num):
            test_id_list = sub_id_set_tuple[test_set_sub_num*i_folder:test_set_sub_num*(i_folder+1)]
            print('\ntest subjects: ')
            for test_id in test_id_list:
                print(SUB_NAMES[test_id] + " who is sub number {} and is {} of {}".format(test_id,i_folder+1, folder_num))
            input_list_train, input_list_test, output_list_train, output_list_test, test_trial_ids = [], [], [], [], []
            train_id_list = list(sub_id_set_tuple)
            [train_id_list.remove(x) for x in test_id_list]
            train_id_list = sample(train_id_list, train_num)
            print(str([SUB_NAMES[i] for i in train_id_list]) + " and their numbers are " + str(train_id_list))

            if trials_test is None:
                trials_test = trials_train
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

            # do input normalization
            if self.do_input_norm:
                self.norm_input()

            if self.do_output_norm:
                self.norm_output()
            
            # self._y_test = self.get_strike_pattern(self._y_test)
            # self._y_train = self.get_strike_pattern(self._y_train)
            if self.tune_hp:
                y_pred, hp_best = self.tune_nn_model(test_id)
            else:
                y_pred = self.define_cnn_model().reshape([-1, 1])
                hp_best = {"Train_Size": len(output_list_train), "Test_Size": len(output_list_test) }
            # y_pred = y_pred.reshape(-1,3)
            
            if self.do_output_norm:
                y_pred = self.norm_output_reverse(y_pred)
            pearson_coeff, RMSE, mean_error = Evaluation.plot_nn_result(self._y_test, y_pred, title=SUB_NAMES[test_id_list[0]])
            discrete_prediction_results = pd.DataFrame(columns=["true","pred"])
            discrete_prediction_results.true = self._y_test
            discrete_prediction_results.pred = y_pred

            predicted_value_df = self.save_all_predicted_values(predicted_value_df, self._y_test, y_pred,
                                                                test_id_list[0], test_trial_ids)
            if predict_result_df.empty:
                predict_result_df = pd.DataFrame(columns=["Subject Name", "correlation", "RMSE", "Mean Error"] + list(hp_best))
            predict_result_df = Evaluation.insert_prediction_result(
                predict_result_df, SUB_NAMES[test_id_list[0]], pearson_coeff, RMSE, mean_error, hp_best)

        self.check_and_create_folders()
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

    def define_cnn_model(self):
        main_input_shape = self._x_train.shape
        main_input = Input((main_input_shape[1:]), name='main_input')
        base_size = int(self.vector_len)

        kernel_regu = regularizers.l2(0.01)
        kernel_size = 18
        tower_4 = Conv1D(filters=18, kernel_size=kernel_size, kernel_regularizer=kernel_regu)(main_input)
        tower_4 = MaxPool1D(pool_size=base_size-kernel_size+1)(tower_4)

        # for each feature, add 5 * 1 cov kernel
        kernel_size = 13
        tower_5 = Conv1D(filters=18, kernel_size=kernel_size, kernel_regularizer=kernel_regu)(main_input)
        tower_5 = MaxPool1D(pool_size=base_size-kernel_size+1)(tower_5)

        # joined_outputs = concatenate([tower_1, tower_3, tower_4, tower_5], axis=-1)
        joined_outputs = concatenate([tower_4, tower_5], axis=-1)

        joined_outputs = Activation('relu')(joined_outputs)
        main_outputs = Flatten()(joined_outputs)

        aux_input = Input(shape=(2,), name='aux_input')
        aux_joined_outputs = concatenate([main_outputs, aux_input])
        aux_joined_outputs = Dense(40, activation='relu')(aux_joined_outputs)
        aux_joined_outputs = Dense(1, activation='sigmoid')(aux_joined_outputs)
        model = Model(inputs=[main_input, aux_input], outputs=aux_joined_outputs)
        my_evaluator = Evaluation(self._x_train, self._x_test, self._y_train, self._y_test, self._x_train_aux,
                                  self._x_test_aux)
        y_pred = my_evaluator.evaluate_nn(model)
        return y_pred

    def get_strike_pattern(self, step_input):
        # the strike pattern is [fore,mid,rear]
        step_num = len(step_input)
        step_output = np.zeros([step_num, 3])
        for i_step in range(step_num):
            SI = np.max(step_input[i_step])
            if SI >= .66:
                step_output[i_step] = [1,0,0] 
            elif SI >=.33 and SI < .66:
                step_output[i_step] = [0,1,0] 
            elif SI < .33:
                step_output[i_step] = [0,0,1]  
        return step_output












