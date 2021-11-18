"""
cross validation
(1) No resampling
(2) Reduced network size
"""
from Evaluation import Evaluation
import matplotlib.pyplot as plt
from tensorflow.keras.layers import *
from tensorflow.keras import regularizers
from ProcessorSI import ProcessorSI
from tensorflow.keras.models import Model
import pandas as pd
from const import SUB_NAMES
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import os
from random import sample



class ProcessorSICrossVali(ProcessorSI):
    def __init__(self, sub_and_trials, imu_locations, sensor_sampling_fre, strike_off_from_IMU=True, do_input_norm=True, do_output_norm=True, start_ratio=.5, end_ratio=.75, pre_samples=5, post_samples=5, tune_hp=False):
        super().__init__(sub_and_trials, None, imu_locations, strike_off_from_IMU, split_train=False, do_input_norm=do_input_norm, 
                         do_output_norm=do_output_norm, start_ratio=start_ratio, end_ratio=end_ratio, pre_samples=pre_samples, post_samples=post_samples, tune_hp=tune_hp)

    def prepare_data_cross_vali(self, trial_include_type, test_set_sub_num=1, start_ratio=.5, end_ratio=.7, pre_samples= 12, post_samples=20, trials = [1,2,3,4,5,6,8,9,10,11,12,13], trials_test = None, train_num=9 ):
        strikePattern = False
        self.start_ratio=start_ratio
        self.end_ratio=end_ratio
        self.pre_samples = pre_samples
        self.post_samples = post_samples
        self.vector_len = (self.pre_samples + self.post_samples + 1)
        train_all_data_list = ProcessorSI.clean_all_data(self.train_all_data_list, self.sensor_sampling_fre)
        input_list,_, output_list = train_all_data_list.get_input_output_list()
        sub_id_list = train_all_data_list.get_sub_id_list()
        trial_id_list = train_all_data_list.get_trial_id_list()
        self.channel_num = input_list[0].shape[1] - 1
        sub_id_set_tuple = tuple(set(sub_id_list))
        sample_num = len(input_list)
        # trial_include_list = [4,5,6,11,12,13]
        trial_include_list = trials
        # trial_include_type = f"{pre_samples=}_{post_samples=}"
        sub_num = len(self.train_sub_and_trials.keys())
        folder_num = int(np.ceil(sub_num / test_set_sub_num)) # the number of cross validation times
        predict_result_df = pd.DataFrame()
        for i_folder in range(folder_num):
            test_id_list = sub_id_set_tuple[test_set_sub_num*i_folder:test_set_sub_num*(i_folder+1)]
            print('\ntest subjects: ')
            for test_id in test_id_list:
                print(SUB_NAMES[test_id] + " who is sub number {} and is {} of {}".format(test_id,i_folder+1, folder_num))
            input_list_train, input_list_test, output_list_train, output_list_test = [], [], [], []
            train_id_list = list(sub_id_set_tuple)
            [train_id_list.remove(x) for x in test_id_list]
            train_id_list = sample(train_id_list, train_num)
            print(str([SUB_NAMES[i] for i in train_id_list]) + " and their numbers are " + str(train_id_list))
            if trials_test is None:
                for i_sample in range(sample_num):
                    if int(trial_id_list[i_sample]) in trial_include_list:
                        if sub_id_list[i_sample] in test_id_list:
                            input_list_test.append(input_list[i_sample])
                            output_list_test.append(output_list[i_sample])
                        elif sub_id_list[i_sample] in train_id_list:
                            input_list_train.append(input_list[i_sample])
                            output_list_train.append(output_list[i_sample])
            else:
                for i_sample in range(sample_num):
                    if int(trial_id_list[i_sample]) in trials_test :
                        if sub_id_list[i_sample] in test_id_list:
                            input_list_test.append(input_list[i_sample])
                            output_list_test.append(output_list[i_sample])
                    if int(trial_id_list[i_sample]) in trial_include_list:
                        if sub_id_list[i_sample] in train_id_list:
                            input_list_train.append(input_list[i_sample])
                            output_list_train.append(output_list[i_sample])
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
            if not os.path.exists("results/{}".format(trial_include_type)):
                os.makedirs("results/{}".format(trial_include_type))
            discrete_prediction_results.to_csv("results/{}/{}.csv".format(trial_include_type, SUB_NAMES[test_id_list[0]]))
            # y_test_classes = np.argmax(self._y_test, axis=1)
            # y_pred_classes = np.argmax(y_pred, axis=1)
            # CM = confusion_matrix(y_test_classes, y_pred_classes, labels=[0, 1, 2])
            # disp = ConfusionMatrixDisplay(CM, ["Forefoot", "Midfoot", "Rearfoot"])
            # print(classification_report(y_test_classes, y_pred_classes,labels=[0, 1,2], target_names=["Forefoot", "Midfoot", "Rearfoot"]))
            # disp.plot()
            if predict_result_df.empty:
                predict_result_df = pd.DataFrame(columns=["Subject Name", "correlation", "RMSE", "Mean Error"] + list(hp_best))
            predict_result_df = Evaluation.insert_prediction_result(
                predict_result_df, SUB_NAMES[test_id_list[0]], pearson_coeff, RMSE, mean_error, hp_best)
        Evaluation.export_prediction_result(predict_result_df, trial_include_type)
        if not os.path.exists("charts"):
            os.makedirs("charts")
        filepath = "charts/{}.pdf".format(trial_include_type)
        self.multipage(filepath)
        plt.show()
        predict_result_df = predict_result_df.replace({",":""}, regex=True)
        return np.nanmean(predict_result_df.correlation.astype("float"))

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

        # kernel_init = 'lecun_uniform'
        kernel_regu = regularizers.l2(0.01)
        # for each feature, add 20 * 1 cov kernel
        # kernel_size = 11
        # tower_1 = Conv1D(filters=11, kernel_size=kernel_size, kernel_regularizer=kernel_regu)(main_input)
        # tower_1 = MaxPool1D(pool_size=base_size-kernel_size+1)(tower_1)

        # # for each feature, add 5 * 1 cov kernel
        # kernel_size = 7 
        # tower_3 = Conv1D(filters=11, kernel_size=kernel_size, kernel_regularizer=kernel_regu)(main_input)
        # tower_3 = MaxPool1D(pool_size=base_size-kernel_size+1)(tower_3)

        # for each feature, add 5 * 1 cov kernel
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
        # aux_joined_outputs = Dense(34, activation='relu')(aux_joined_outputs)
        # aux_joined_outputs = Dense(34, activation='relu')(aux_joined_outputs)
        # aux_joined_outputs = Dense(32, activation='relu')(aux_joined_outputs)
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












