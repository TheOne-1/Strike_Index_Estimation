import numpy as np
import copy
import matplotlib.pyplot as plt
from SharedProcessors.const import SUB_AND_SI_TRIALS, FILTER_WIN_LEN, SI_TRIALS, MOCAP_SAMPLE_RATE, SUB_NAMES
from SharedProcessors.OneTrialData import OneTrialData
from SharedProcessors.AllSubData import AllSubData
from SharedProcessors.AllSubDataStruct import AllSubDataStruct
from strike_index_tian.ProcessorSI import ProcessorSI
from sklearn.metrics import r2_score, mean_squared_error as mse
from scipy.stats import pearsonr
from scipy.signal import butter, filtfilt, find_peaks, lfilter, firwin


class AllSubDataSA(AllSubData):
    def get_all_data(self):
        all_sub_data_struct = AllSubDataStruct()
        for subject_name in self._sub_names:
            print('loading data of: ' + subject_name)
            static_nike_trial = OneTrialDataStaticSA(subject_name, 'nike static', self._sensor_sampling_fre)
            static_nike_df = static_nike_trial.get_multi_IMU_data(self.imu_locations, acc=True, mag=True)
            static_mini_trial = OneTrialDataStaticSA(subject_name, 'mini static', self._sensor_sampling_fre)
            static_mini_df = static_mini_trial.get_multi_IMU_data(self.imu_locations, acc=True, mag=True)
            trials = self._sub_and_trials[subject_name]
            for trial_name in trials:
                if 'nike' in trial_name:
                    trial_processor = OneTrialDataSA(subject_name, trial_name, self._sensor_sampling_fre,
                                                     static_data_df=static_nike_df)
                else:
                    trial_processor = OneTrialDataSA(subject_name, trial_name, self._sensor_sampling_fre,
                                                     static_data_df=static_mini_df)
                trial_input, trial_output_LR, trial_output_SI = trial_processor.get_input_output(
                    self.imu_locations, self._strike_off_from_IMU)
                all_sub_data_struct.append(trial_input, trial_output_LR, trial_output_SI, subject_name, trial_name)
        return all_sub_data_struct


class OneTrialDataSA(OneTrialData):
    def get_input_output(self, imu_locations, from_IMU=1, acc=True, gyr=True, mag=False):
        """
        GRFz: from strike to off
        acc and gyr: from off to off because information before strike might be useful
        :param imu_locations: list of str
        :param from_IMU: 0 for not from IMU, 1 for filtfilt, 2 for left filter
        :param acc:
        :param gyr:
        :param mag:
        :return:
        """
        if from_IMU == 2:
            filter_delay = int(FILTER_WIN_LEN / 2)
        else:
            filter_delay = 0
        if not from_IMU:
            offs, step_num = self.get_offs()
            strikes, step_num = self.get_strikes()
        else:
            offs, strikes, step_num = self.get_offs_strikes_from_IMU(from_IMU)
        lr_data = self.gait_param_df[self._side + "_LR"].values
        SA_data = self.gait_param_df[self._side + "_strike_angle"].values
        IMU_data = self.get_multi_IMU_data(imu_locations, acc, gyr, mag)
        step_lr_data, step_SA_data, step_imu_data = [], [], []
        for i_step in range(step_num):
            strike_in_between = strikes[offs[i_step] < strikes]
            strike_in_between = strike_in_between[strike_in_between < offs[i_step+1]]
            if len(strike_in_between) != 1:
                continue
            step_start = offs[i_step] - filter_delay
            step_end = offs[i_step + 1] - filter_delay

            strikes_array = np.zeros([step_end - step_start, 1])
            strikes_array[strike_in_between - offs[i_step], 0] = 1
            # skip this step if the step_end exceeds the maximum data length
            if step_end > lr_data.shape[0]:
                continue

            step_input = np.column_stack([IMU_data[step_start:step_end, :], strikes_array])
            step_imu_data.append(step_input)
            step_lr_data.append(lr_data[step_start:step_end])
            step_SA_data.append(SA_data[step_start:step_end])
        # for debug
        # plt.figure()
        # for i_step in range(step_num):
        #     # plt.plot(step_imu_data[i_step][:, 0])
        #     plt.plot(step_SA_data[i_step])
        # plt.show()

        step_imu_data, step_lr_data, step_SA_data = self.check_step_input_output(step_imu_data, step_lr_data, step_SA_data)
        return step_imu_data, step_lr_data, step_SA_data


class OneTrialDataStaticSA(OneTrialDataSA):
    def get_one_IMU_data(self, IMU_location, acc=True, gyr=False, mag=False):
        column_names = []
        if acc:
            column_names += [IMU_location + '_acc_' + axis for axis in ['x', 'y', 'z']]
        if gyr:
            column_names += [IMU_location + '_gyr_' + axis for axis in ['x', 'y', 'z']]
        if mag:
            column_names += [IMU_location + '_mag_' + axis for axis in ['x', 'y', 'z']]
        data = self.gait_data_df[column_names]
        return data


def data_filt(data, cut_off_fre, sampling_fre, filter_order=4):
    fre = cut_off_fre / (sampling_fre / 2)
    b, a = butter(filter_order, fre, 'lowpass')
    if len(data.shape) == 1:
        data_filt = filtfilt(b, a, data)
    else:
        data_filt = filtfilt(b, a, data, axis=0)
    return data_filt


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

        # delete steps without a valid strike time
        if len(strikes) != 1:
            all_sub_data_struct.pop(i_step)
            # if "SI" in TRIAL_NAMES[trial_list[i_step]]:
            #     print("Bad SI in number of strikes test")
            counter_bad += 1

        # delete a step if the duration between strike and off is too short
        elif not min_time_between_strike_off < input_list[i_step].shape[0] - strikes[0]:
            all_sub_data_struct.pop(i_step)
            # if "SI" in TRIAL_NAMES[trial_list[i_step]]:
            #     print("Bad SI in strike time test")
            counter_bad += 1

        # delete a step if the strike does not fall into 50% to 85% swing phase
        elif not 0.5 * input_list[i_step].shape[0] < strikes[0] < 0.85 * input_list[i_step].shape[0]:
            all_sub_data_struct.pop(i_step)
            # if "SI" in TRIAL_NAMES[trial_list[i_step]]:
            #     print("Bad SI in occurance of the strike during a step")
            counter_bad += 1

        else:
            # step number only increase when no pop happens
            i_step += 1
    print(f"There were {counter_bad} bad steps here due to SI problems")
    return all_sub_data_struct


def WerkhovenAlgorithm(input_list, output_list):
    step_num = len(input_list)
    # plt.figure()
    angle_esti, angle_vicon = [], []
    for i_step in range(step_num):
        gyro_x, strike_loc, strike_angle = input_list[i_step][:, 3], input_list[i_step][:, 6], output_list[i_step]
        acc_norm = np.linalg.norm(input_list[i_step][:, :3], axis=1)
        # acc_norm_filted = data_filt(copy.deepcopy(acc_norm), 5, 200)
        # inte_start_loc = np.argmax(acc_norm_filted[30:120])
        # inte_start_loc += 30
        inte_start_loc = np.where(strike_loc==1)[0][0] - 1

        stationary_check_win = 25
        stationary_check_period = int(50 / (1000 / MOCAP_SAMPLE_RATE))
        stationary_check = np.zeros([stationary_check_win])
        for i in range(stationary_check_win):
            stationary_check[i] = np.sum(acc_norm[inte_start_loc+i:inte_start_loc+i+stationary_check_period])
        stationary_start_loc = np.argmin(stationary_check)
        angle_esti_step = - np.sum(gyro_x[inte_start_loc:inte_start_loc+stationary_start_loc]) / MOCAP_SAMPLE_RATE * 180 / np.pi

        angle_esti.append(angle_esti_step)
        angle_vicon.append(strike_angle[np.where(strike_loc==1)[0][0]])

    # plt.plot(angle_vicon, angle_esti, '.')
    r2 = r2_score(angle_vicon, angle_esti)
    # r2 = pearsonr(angle_vicon, angle_esti)[0] ** 2
    RMSE = np.sqrt(mse(angle_vicon, angle_esti, multioutput='raw_values'))
    return r2, RMSE


def ShiangAlgorithm(input_list, output_list):
    step_num = len(input_list)
    plt.figure()
    for i_step in range(step_num):
        gyro, strike_loc, strike_angle = input_list[i_step][:, 3], input_list[i_step][:, 6], output_list[i_step]
        acc_norm = np.linalg.norm(input_list[i_step][:, :3], axis=1)
        max_acc_loc = np.argmax(acc_norm)
        gyro_win = gyro[max_acc_loc-int(MOCAP_SAMPLE_RATE*0.15):max_acc_loc+int(MOCAP_SAMPLE_RATE*0.2)]
        angle_win = np.zeros(gyro_win.shape)
        for i_sample, gyro_sample in enumerate(gyro_win):
            angle_win[i_sample] = angle_win[i_sample-1] + 1/MOCAP_SAMPLE_RATE*gyro_sample

        plt.plot(angle_win)
    plt.show()



test_date = '220325'
imu_locations = ['l_foot']

if __name__ == "__main__":
    r2_all, RMSE_all = [], []
    for sub in SUB_NAMES:
        all_data = AllSubDataSA({sub: SI_TRIALS}, imu_locations, MOCAP_SAMPLE_RATE, strike_off_from_IMU=1)
        all_data_list = all_data.get_all_data()
        train_all_data_list = clean_all_data(all_data_list, MOCAP_SAMPLE_RATE)
        input_list, _, output_list = train_all_data_list.get_input_output_list()
        r2, RMSE = WerkhovenAlgorithm(input_list, output_list)
        r2_all.append(r2)
        RMSE_all.append(RMSE)
    print("{:.2f}".format(np.mean(r2_all)))
    print(r2_all)
    plt.show()
















