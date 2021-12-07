import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from SharedProcessors.const import TRIAL_NAMES, PLATE_SAMPLE_RATE, MOCAP_SAMPLE_RATE, \
    LOADING_RATE_NORMALIZATION, SI_TRIALS, SUB_WEIGHTS, MININ_SHOE_LENGTHS, TRAD_SHOE_LENGTHS, GOOD_SUBS
from scipy.signal import find_peaks
from numpy.linalg import norm
from a_processing.StrikeOffDetectorIMU import StrikeOffDetectorIMU, StrikeOffDetectorIMUFilter
import vg


class ParamProcessor:
    def __init__(self, sub_name, trials, check_steps=False, plot_strike_off=False, initialize_200Hz=True):
        self._sub_name = sub_name
        self._weight = SUB_WEIGHTS[sub_name]  # in kilos
        self._check_steps = check_steps
        self._plot_strike_off = plot_strike_off
        self._initialize_thresholds()
        self._initialize_200Hz = initialize_200Hz
        if trials is TRIAL_NAMES:
            self._trials = SI_TRIALS
        else:
            self._trials = list(trials)
            # remove static trials
            if TRIAL_NAMES[0] in self._trials:
                self._trials.remove(TRIAL_NAMES[0])
            if TRIAL_NAMES[7] in self._trials:
                self._trials.remove(TRIAL_NAMES[7])

    def start_initalization(self, path):
        print('\n' + self._sub_name)
        fre_200_path = path + '\\' + self._sub_name + '\\200Hz\\'
        fre_1000_path = path + '\\' + self._sub_name + '\\1000Hz\\'

        if self._initialize_200Hz:
            self.nike_static_200_df = pd.read_csv(fre_200_path + TRIAL_NAMES[0] + '.csv', index_col=False)
            self.mini_static_200_df = pd.read_csv(fre_200_path + TRIAL_NAMES[7] + '.csv', index_col=False)
            for trial_name in self._trials:
                print('\n' + trial_name + ' trial')
                self._current_trial = trial_name
                # initialize 200 Hz parameter
                print('200Hz')
                self._current_fre = 200
                gait_data_200_df = pd.read_csv(fre_200_path + trial_name + '.csv', index_col=False)
                grf_1000_df = pd.read_csv(fre_1000_path + trial_name + '.csv', index_col=False)
                trial_param_df_200, l_steps_1000, r_steps_1000 = self.init_trial_params(gait_data_200_df, grf_1000_df,
                                                                                        MOCAP_SAMPLE_RATE)
                l_steps, r_steps = self.resample_steps(l_steps_1000, 200), self.resample_steps(r_steps_1000, 200)
                self.__save_data(fre_200_path, trial_name, trial_param_df_200, l_steps, r_steps)
                plt.show()

    @staticmethod
    def resample_steps(steps_1000, sample_fre):
        ratio = int(1000 / sample_fre)
        steps_resampled = []
        for step in steps_1000:
            step_resampled = [round(step[0] / ratio), round(step[1] / ratio)]
            steps_resampled.append(step_resampled)
        return steps_resampled

    def _initialize_thresholds(self):
        self._stance_phase_sample_thd_lower = 180
        self._stance_phase_sample_thd_higher = 380
        self._impact_peak_sample_num_lower = 15
        self._impact_peak_sample_num_higher = 80
        self._20_80_sample_len_lower = 5
        self._20_80_sample_len_higher = 40

    def init_trial_params(self, gait_data_df, grf_1000_df, sensor_sampling_rate):
        # get the corresponding plate data period
        marker_frame = gait_data_df['marker_frame']
        start_vicon, end_vicon = min(marker_frame), max(marker_frame)
        vicon_force_ratio = int(PLATE_SAMPLE_RATE / MOCAP_SAMPLE_RATE)
        start_row_grf = int((start_vicon - 1) * vicon_force_ratio)
        end_row_grf = int(end_vicon * vicon_force_ratio) - 1
        plate_data_1000 = grf_1000_df.loc[start_row_grf:end_row_grf].reset_index(drop=True)

        # get strikes and offs
        l_strikes, r_strikes, l_offs, r_offs = self.get_strike_off(gait_data_df)
        l_strikes_1000, r_strikes_1000, l_offs_1000, r_offs_1000 = self.get_strike_off_1000(
            gait_data_df, plate_data_1000, sensor_sampling_rate)

        # get strike index and FPA
        strike_index_all = self.get_strike_index_all(gait_data_df,l_strikes, r_strikes)
        strike_angle_all = self.get_foot_strike_angle(gait_data_df)
        FPA_all = self.get_FPA_all(gait_data_df)  # FPA of all the samples
        param_data = np.column_stack(
            [l_strikes, r_strikes, l_offs, r_offs, strike_index_all, strike_angle_all, FPA_all])
        param_data_df = pd.DataFrame(param_data)
        param_data_df.columns = ['l_strikes', 'r_strikes', 'l_offs', 'r_offs', 'l_strike_index', 'r_strike_index',
                                 'l_strike_angle', 'r_strike_angle', 'l_FPA', 'r_FPA']
        param_data_df.insert(0, 'marker_frame', gait_data_df['marker_frame'])

        # get strikes and offs from IMU data
        estimated_strikes, estimated_offs = self.get_strike_off_from_imu(
            gait_data_df, param_data_df, sensor_sampling_rate, check_strike_off=False, plot_the_strike_off=False)
        param_data_df.insert(len(param_data_df.columns), 'strikes_IMU', estimated_strikes)
        param_data_df.insert(len(param_data_df.columns), 'offs_IMU', estimated_offs)

        estimated_strikes_lfilter, estimated_offs_lfilter = self.get_strike_off_from_imu_lfilter(
            gait_data_df, param_data_df, sensor_sampling_rate, check_strike_off=True)
        param_data_df.insert(len(param_data_df.columns), 'strikes_IMU_lfilter', estimated_strikes_lfilter)
        param_data_df.insert(len(param_data_df.columns), 'offs_IMU_lfilter', estimated_offs_lfilter)

        # get loading rate
        l_steps_1000 = self.get_legal_steps(l_strikes_1000, l_offs_1000, 'left', plate_data_1000)
        l_LR = self.get_loading_rate(plate_data_1000, l_steps_1000)
        self.insert_LR_to_param_data(param_data_df, l_LR, 'l_LR')
        r_steps_1000 = self.get_legal_steps(r_strikes_1000, r_offs_1000, 'right', plate_data_1000)
        r_LR = self.get_loading_rate(plate_data_1000, r_steps_1000)
        self.insert_LR_to_param_data(param_data_df, r_LR, 'r_LR')

        # plt.figure()
        # plt.plot(gait_data_df['c_1_y'])
        # plt.plot(gait_data_df['LFM2_y'])
        # plt.plot(-gait_data_df['f_1_z'])
        # plt.grid()
        # plt.show()

        return param_data_df, l_steps_1000, r_steps_1000

    @staticmethod
    def check_loading_rate_all(l_loading_rate):
        plt.figure()
        plt.plot(l_loading_rate)

    def get_strike_off(self, gait_data_df, threshold=20):
        force = gait_data_df[['f_1_x', 'f_1_y', 'f_1_z']].values
        force_norm = norm(force, axis=1)
        strikes, offs = self.get_raw_strikes_offs(force_norm, threshold)
        self.check_strikes_offs(force_norm, strikes, offs)

        # distribute strikes offs to left and right foot
        data_len = len(strikes)
        l_heel_y, r_heel_y = gait_data_df['LFCC_y'], gait_data_df['RFCC_y']
        l_strikes, r_strikes = np.zeros(data_len), np.zeros(data_len)
        l_offs, r_offs = np.zeros(data_len), np.zeros(data_len)
        for i_sample in range(data_len):
            if strikes[i_sample] == 1:
                try:
                    if l_heel_y[i_sample] > r_heel_y[i_sample]:
                        l_strikes[i_sample] = 1
                    else:
                        r_strikes[i_sample] = 1
                except KeyError:
                    if l_heel_y[i_sample] > r_heel_y[i_sample]:
                        l_strikes[i_sample] = 1
                    else:
                        r_strikes[i_sample] = 1

            if offs[i_sample] == 1:
                try:
                    if l_heel_y[i_sample] < r_heel_y[i_sample]:
                        l_offs[i_sample] = 1
                    else:
                        r_offs[i_sample] = 1
                except KeyError:
                    if l_heel_y[i_sample] < r_heel_y[i_sample]:
                        l_offs[i_sample] = 1
                    else:
                        r_offs[i_sample] = 1
        return l_strikes, r_strikes, l_offs, r_offs

    def get_strike_off_1000(self, gait_data_df, plate_data_1000, sensor_sampling_rate, threshold=20):
        force = plate_data_1000[['f_1_x', 'f_1_y', 'f_1_z']].values
        force_norm = norm(force, axis=1)
        strikes, offs = self.get_raw_strikes_offs(force_norm, threshold, comparison_len=20)
        self.check_strikes_offs(force_norm, strikes, offs)

        # distribute strikes offs to left and right foot
        data_len = len(strikes)
        ratio = sensor_sampling_rate / PLATE_SAMPLE_RATE

        l_strikes, r_strikes = np.zeros(data_len), np.zeros(data_len)
        l_offs, r_offs = np.zeros(data_len), np.zeros(data_len)
        for i_sample in range(data_len):
            if strikes[i_sample] == 1:
                l_heel_y = gait_data_df.loc[round(i_sample * ratio), 'LFCC_y']
                r_heel_y = gait_data_df.loc[round(i_sample * ratio), 'RFCC_y']
                if l_heel_y == 0 or r_heel_y == 0:
                    raise ValueError('Marker missing')
                if l_heel_y > r_heel_y:
                    l_strikes[i_sample] = 1
                else:
                    r_strikes[i_sample] = 1
            if offs[i_sample] == 1:
                l_heel_y = gait_data_df.loc[round(i_sample * ratio), 'LFCC_y']
                r_heel_y = gait_data_df.loc[round(i_sample * ratio), 'RFCC_y']
                if l_heel_y == 0 or r_heel_y == 0:
                    raise ValueError('Marker missing')
                if l_heel_y < r_heel_y:
                    l_offs[i_sample] = 1
                else:
                    r_offs[i_sample] = 1
        return l_strikes, r_strikes, l_offs, r_offs

    @staticmethod
    def get_raw_strikes_offs(force_norm, threshold, comparison_len=4):
        data_len = force_norm.shape[0]
        strikes, offs = np.zeros(data_len, dtype=np.int8), np.zeros(data_len, dtype=np.int8)
        i_point = comparison_len
        # go to the first swing phase
        while i_point < data_len and force_norm[i_point] < 300:  # go to the middle of the first stance phase
            i_point += 1
        swing_phase = False
        while i_point < data_len - comparison_len:
            # for swing phase
            if swing_phase:
                while i_point < data_len - comparison_len:
                    i_point += 1
                    lower_than_threshold_num = len(
                        np.where(force_norm[i_point:i_point + comparison_len] < threshold)[0])
                    if lower_than_threshold_num >= round(0.8 * comparison_len):
                        continue
                    else:
                        strikes[i_point + round(0.8 * comparison_len) - 1] = 1
                        swing_phase = False
                        break
            # for stance phase
            else:
                while i_point < data_len and force_norm[i_point] > 300:  # go to the next stance phase
                    i_point += 1
                while i_point < data_len - comparison_len:
                    i_point += 1
                    lower_than_threshold_num = len(
                        np.where(force_norm[i_point:i_point + comparison_len] < threshold)[0])
                    if lower_than_threshold_num >= round(0.8 * comparison_len):
                        offs[i_point + round(0.2 * comparison_len)] = 1
                        swing_phase = True
                        break
        return strikes, offs

    def check_strikes_offs(self, force_norm, strikes, offs):
        strike_indexes = np.where(strikes == 1)[0]
        off_indexes = np.where(offs == 1)[0]
        data_len = min(strike_indexes.shape[0], off_indexes.shape[0])

        # check strike off by checking if each strike is followed by a off
        strike_off_detection_flaw = False
        if strike_indexes[0] > off_indexes[0]:
            diffs_0 = np.array(strike_indexes[:data_len]) - np.array(off_indexes[:data_len])
            diffs_1 = np.array(strike_indexes[:data_len - 1]) - np.array(off_indexes[1:data_len])
        else:
            diffs_0 = np.array(off_indexes[:data_len]) - np.array(strike_indexes[:data_len])
            diffs_1 = np.array(off_indexes[:data_len - 1]) - np.array(strike_indexes[1:data_len])
        if np.min(diffs_0) < 0 or np.max(diffs_1) > 0:
            strike_off_detection_flaw = True

        try:
            if strike_off_detection_flaw:
                raise ValueError('For trial {trial_name}, strike off detection result are wrong.'.format(
                    trial_name=self._current_trial))
            if self._plot_strike_off:
                raise ValueError
        except ValueError as value_error:
            if len(value_error.args) != 0:
                print(value_error.args[0])
            plt.figure()
            plt.plot(force_norm)
            plt.grid()
            plt.plot(strike_indexes, force_norm[strike_indexes], 'g*')
            plt.plot(off_indexes, force_norm[off_indexes], 'gx')

    def get_trunk_swag(self, gait_data_df):
        C7 = gait_data_df.as_matrix(columns=['C7_x', 'C7_y', 'C7_z'])
        l_PSIS = gait_data_df.as_matrix(columns=['l_PSIS_x', 'l_PSIS_y', 'l_PSIS_z'])
        r_PSIS = gait_data_df.as_matrix(columns=['r_PSIS_x', 'r_PSIS_y', 'r_PSIS_z'])
        middle_PSIS = (l_PSIS + r_PSIS) / 2
        vertical_vector = C7 - middle_PSIS
        return - 180 / np.pi * np.arctan(vertical_vector[:, 0] / vertical_vector[:, 2])

    @staticmethod
    def check_trunk_swag(data_trunk_swag):
        plt.figure()
        plt.plot(data_trunk_swag)
        plt.title('trunk swag')

    def get_strike_index_all(self, gait_data_df, l_strikes, r_strikes):
        """ This is Zach's version """
        delay = 4       # why needed?
        if self._sub_name in GOOD_SUBS:
            if "mini" in self._current_trial:
                foot_len = MININ_SHOE_LENGTHS[self._sub_name]
            elif "nike" in self._current_trial: 
                foot_len = TRAD_SHOE_LENGTHS[self._sub_name]
        else:
            if "mini" in self._current_trial:
                foot_len = 250
            elif "nike" in self._current_trial: 
                foot_len = 276
        l_foot_toe_data = gait_data_df[['LFM2_x', 'LFM2_y', 'LFM2_z']].values
        l_foot_heel_data = gait_data_df[['LFCC_x', 'LFCC_y', 'LFCC_z']].values
        l_foot_cop_data = gait_data_df[['c_1_x', 'c_1_y', 'c_1_z']].values
        l_foot_length = (l_foot_toe_data - l_foot_heel_data)
        l_foot_length = l_foot_length / np.expand_dims(np.linalg.norm(l_foot_length, axis=1), 1) * foot_len
        l_cop_length = l_foot_cop_data - l_foot_heel_data

        l_index = vg.scalar_projection(l_cop_length, l_foot_length)
        l_index = l_index / foot_len

        if delay != 0:
            l_strikes = np.pad(l_strikes, (delay, 0), mode='constant')[:-delay]
        l_index = l_index * l_strikes

        r_foot_toe_data = gait_data_df[['RFM2_x', 'RFM2_y', 'RFM2_z']].values
        r_foot_heel_data = gait_data_df[['RFCC_x', 'RFCC_y', 'RFCC_z']].values
        r_foot_cop_data = gait_data_df[['c_1_x', 'c_1_y', 'c_1_z']].values
        r_foot_length = (r_foot_toe_data - r_foot_heel_data)
        r_cop_length = (r_foot_cop_data - r_foot_heel_data)
        r_foot_length = r_foot_length / np.expand_dims(np.linalg.norm(r_foot_length, axis=1), 1) * foot_len
        r_index = vg.scalar_projection(r_cop_length, r_foot_length)
        r_index = r_index / foot_len
        if delay != 0:
            r_strikes = np.pad(r_strikes, (delay, 0), mode='constant')[:-delay]
        r_index = r_index * r_strikes
        return np.column_stack([l_index, r_index])

    # this program use 1000Hz force plate data
    def get_loading_rate(self, plate_data, steps):
        # 20% to 80% from strike to impact peak
        loading_rates = []
        grf_z = plate_data['f_1_z'].values
        for step in steps:
            grf_z_step = grf_z[step[0]:step[1]]
            peaks, _ = find_peaks(-grf_z_step, height=200, prominence=150)
            try:  # find legal peaks
                if len(peaks) == 1:  # case 0, no impact peak, only one max peak
                    impact_peak_sample_num = 0.13 * (step[1] - step[0])
                elif len(peaks) == 2:  # case 1, impact peak exists
                    impact_peak_sample_num = peaks[0]
                else:
                    raise ValueError('Wrong peak number, please check the plot.')
                if impact_peak_sample_num < self._impact_peak_sample_num_lower or \
                        impact_peak_sample_num > self._impact_peak_sample_num_higher:
                    raise ValueError('Wrong impact peak location, please check the plot.')
            except ValueError as value_error:
                # # if the user close the plot, the calculation will continue
                # plt.figure()
                # plt.plot(grf_z_step)
                # plt.plot(peaks, grf_z_step[peaks], 'r*')
                # plt.show()
                continue  # continue without recording loading rate
            peak_index = int(round(impact_peak_sample_num))
            impact_peak_force = grf_z_step[peak_index]
            force_start = 0.2 * impact_peak_force
            start_index = np.abs(grf_z_step[:peak_index] - force_start).argmin()
            force_end = 0.8 * impact_peak_force
            end_index = np.abs(grf_z_step[:peak_index] - force_end).argmin()
            try:
                if end_index - start_index < self._20_80_sample_len_lower or \
                        end_index - start_index > self._20_80_sample_len_higher:
                    raise ValueError('Wrong 20% - 80% sample num, found {num} in total, please check the plot'.
                                     format(num=end_index - start_index))
            except ValueError as value_error:
                print(value_error.args[0])
                print('From sample {start} to sample {end}\n'.format(start=step[0], end=step[1]))
                # plt.figure()
                # plt.plot(grf_z_step)
                # plt.plot([start_index, end_index], [grf_z_step[start_index], grf_z_step[end_index]], 'r-')
                # plt.show()
                continue  # continue without recording loading rate
            loading_rate = (grf_z_step[end_index] - grf_z_step[start_index]) * PLATE_SAMPLE_RATE / (end_index - start_index)
            if LOADING_RATE_NORMALIZATION:
                loading_rate = - loading_rate / (self._weight * 10)
            marker_frame = plate_data.loc[round((step[0] + step[1]) / 2), 'marker_frame']
            loading_rates.append([loading_rate, marker_frame])
        return loading_rates

    def insert_params_to_param_data(self, param_data_df, params, param_names):
        if len(params) != len(param_names):
            raise RuntimeError('The param number and param name number doesn\'t match')
        for param, param_name in params, param_names:
            param_data_df.insert(len(param_data_df.columns), param_name, param)

    def insert_LR_to_param_data(self, gait_data_df, insert_list, column_name):
        data_len = gait_data_df.shape[0]
        insert_data = np.zeros([data_len])
        for item in insert_list:
            row_index = gait_data_df.index[gait_data_df['marker_frame'] == item[1]]
            if len(row_index) == 0:
                row_index = gait_data_df.index[gait_data_df['marker_frame'] == item[1] + 1]
            insert_data[row_index[0]] = item[0]
        gait_data_df.insert(len(gait_data_df.columns), column_name, insert_data)

    def get_legal_steps(self, strikes, offs, side, plate_data=None):
        """
            Sometimes subjects have their both feet on the ground so it is necessary to do step check.
        """
        strike_tuple = np.where(strikes == 1)[0]
        off_tuple = np.where(offs == 1)[0]
        off_tuple = off_tuple[off_tuple > strike_tuple[0]]
        steps = []
        abandoned_step_nam = 0
        i_step = -1
        while i_step < min(len(strike_tuple), len(off_tuple)) - 1:
            i_step += 1
            stance_start = strike_tuple[i_step]
            stance_end = off_tuple[i_step]
            step_len = stance_end - stance_start
            # pop out illegal steps
            if not self._stance_phase_sample_thd_lower < step_len < self._stance_phase_sample_thd_higher:
                abandoned_step_nam += 1
                if step_len > self._stance_phase_sample_thd_higher:
                    strike_tuple = np.delete(strike_tuple, i_step)
                else:
                    off_tuple = np.delete(off_tuple, i_step)
                continue
            steps.append([strike_tuple[i_step], off_tuple[i_step]])
        print('For {side} foot steps, {step_num} steps abandonded'.format(side=side, step_num=abandoned_step_nam))
        if self._check_steps:
            plt.figure()
            grf_z = plate_data['f_1_z'].values
            for step in steps:
                plt.plot(grf_z[step[0]:step[1]])
            plt.show()
        return steps

    @staticmethod
    def __save_data(folder_path, trial_name, data_all_df, l_steps, r_steps):
        # save param data
        data_file_str = '{folder_path}\\param_of_{trial_name}_si_paper.csv'.format(
            folder_path=folder_path, trial_name=trial_name)
        data_all_df.to_csv(data_file_str, index=False)

    def get_strike_off_from_imu(self, gait_data_df, param_data_df, sensor_sampling_rate, check_strike_off=True,
                                plot_the_strike_off=False):
        if sensor_sampling_rate == MOCAP_SAMPLE_RATE:
            my_detector = StrikeOffDetectorIMU(self._current_trial, gait_data_df, param_data_df, 'l_foot',
                                               MOCAP_SAMPLE_RATE)
            strike_delay, off_delay = 8, 6  # delay from the peak
        else:
            raise ValueError('Wrong sensor sampling rate value')
        estimated_strike_indexes, estimated_off_indexes = my_detector.get_jogging_strike_off(strike_delay, off_delay)
        if plot_the_strike_off:
            my_detector.show_IMU_data_and_strike_off(estimated_strike_indexes, estimated_off_indexes)
        data_len = gait_data_df.shape[0]
        estimated_strikes, estimated_offs = np.zeros([data_len]), np.zeros([data_len])
        estimated_strikes[estimated_strike_indexes] = 1
        estimated_offs[estimated_off_indexes] = 1
        if check_strike_off:
            my_detector.true_esti_diff(estimated_strike_indexes, 'strikes')
            my_detector.true_esti_diff(estimated_off_indexes, 'offs')
        return estimated_strikes, estimated_offs

    def get_strike_off_from_imu_lfilter(self, gait_data_df, param_data_df, sensor_sampling_rate, check_strike_off=True):
        """
        In the filter, lfilter was used so there are delays in the detected events (about 50 samples)
        """
        if sensor_sampling_rate == MOCAP_SAMPLE_RATE:
            my_detector = StrikeOffDetectorIMUFilter(self._current_trial, gait_data_df, param_data_df, 'l_foot',
                                                     MOCAP_SAMPLE_RATE)
            strike_delay, off_delay = 8, 6  # delay from the peak
        else:
            raise ValueError('Wrong sensor sampling rate value')
        estimated_strike_indexes, estimated_off_indexes = my_detector.get_jogging_strike_off(strike_delay, off_delay)
        if self._plot_strike_off:
            my_detector.show_IMU_data_and_strike_off(estimated_strike_indexes, estimated_off_indexes)
        data_len = gait_data_df.shape[0]
        estimated_strikes, estimated_offs = np.zeros([data_len]), np.zeros([data_len])
        estimated_strikes[estimated_strike_indexes] = 1
        estimated_offs[estimated_off_indexes] = 1
        if check_strike_off:
            my_detector.true_esti_diff(estimated_strike_indexes, 'strikes')
            my_detector.true_esti_diff(estimated_off_indexes, 'offs')
        return estimated_strikes, estimated_offs

    # FPA of all the samples
    def get_FPA_all(self, gait_data_df):
        l_toe = gait_data_df[['LFM2_x', 'LFM2_y', 'LFM2_z']].values
        l_heel = gait_data_df[['LFCC_x', 'LFCC_y', 'LFCC_z']].values
        data_len = l_toe.shape[0]
        left_FPAs = np.zeros(data_len)
        for i_point in range(0, data_len):
            forward_vector = l_toe[i_point, :] - l_heel[i_point, :]
            left_FPAs[i_point] = - 180 / np.pi * np.arctan(forward_vector[0] / forward_vector[1])

        r_toe = gait_data_df[['RFM2_x', 'RFM2_y', 'RFM2_z']].values
        r_heel = gait_data_df[['RFCC_x', 'RFCC_y', 'RFCC_z']].values
        right_FPAs = np.zeros(data_len)
        for i_point in range(0, data_len):
            forward_vector = r_toe[i_point, :] - r_heel[i_point, :]
            right_FPAs[i_point] = 180 / np.pi * np.arctan(forward_vector[0] / forward_vector[1])

        return np.column_stack([left_FPAs, right_FPAs])

    def get_foot_strike_angle(self, gait_data_df):
        static_data_df = self.get_current_static_data_df()
        l_toe_static = static_data_df[['LFM2_x', 'LFM2_y', 'LFM2_z']].values
        l_heel_static = static_data_df[['LFCC_x', 'LFCC_y', 'LFCC_z']].values
        l_foot_diff = l_toe_static - l_heel_static
        l_foot_len = np.mean(norm(l_foot_diff, axis=1))
        l_toe_z_static = np.mean(l_toe_static[:, 2])
        l_heel_z_static = np.mean(l_heel_static[:, 2])
        l_toe_z = gait_data_df[['LFM2_z']].values
        l_heel_z = gait_data_df[['LFCC_z']].values
        z_diff = (l_toe_z - l_toe_z_static) - (l_heel_z - l_heel_z_static)
        l_foot_strike_angle = np.rad2deg(np.arcsin(z_diff / l_foot_len))

        r_toe_static = static_data_df[['RFM2_x', 'RFM2_y', 'RFM2_z']].values
        r_heel_static = static_data_df[['RFCC_x', 'RFCC_y', 'RFCC_z']].values
        r_foot_diff = r_toe_static - r_heel_static
        r_foot_len = np.mean(norm(r_foot_diff, axis=1))
        r_toe_z_static = np.mean(r_toe_static[:, 2])
        r_heel_z_static = np.mean(r_heel_static[:, 2])
        r_toe_z = gait_data_df[['RFM2_z']].values
        r_heel_z = gait_data_df[['RFCC_z']].values
        z_diff = (r_toe_z - r_toe_z_static) - (r_heel_z - r_heel_z_static)
        r_foot_strike_angle = np.rad2deg(np.arcsin(z_diff / r_foot_len))

        return np.column_stack([l_foot_strike_angle, r_foot_strike_angle])

    def get_current_static_data_df(self):
        if 'nike' in self._current_trial:
            if self._current_fre == 100:
                static_data_df = self.nike_static_100_df
            else:
                static_data_df = self.nike_static_200_df
        else:
            if self._current_fre == 100:
                static_data_df = self.mini_static_100_df
            else:
                static_data_df = self.mini_static_200_df
        return static_data_df


# class ParamProcessor2021(ParamProcessor):
#     def init_trial_params(self, gait_data_df, grf_1000_df, sensor_sampling_rate):
#         # get the corresponding plate data period
#         marker_frame = gait_data_df['marker_frame']
#         start_vicon, end_vicon = min(marker_frame), max(marker_frame)
#         vicon_force_ratio = int(PLATE_SAMPLE_RATE / MOCAP_SAMPLE_RATE)
#         start_row_grf = int((start_vicon - 1) * vicon_force_ratio)
#         end_row_grf = int(end_vicon * vicon_force_ratio) - 1
#         plate_data_1000 = grf_1000_df.loc[start_row_grf:end_row_grf].reset_index(drop=True)
#
#         # get strikes and offs
#         l_strikes, r_strikes, l_offs, r_offs = self.get_strike_off(gait_data_df)
#         l_strikes_1000, r_strikes_1000, l_offs_1000, r_offs_1000 = self.get_strike_off_1000(
#             gait_data_df, plate_data_1000, sensor_sampling_rate)
#
#         # get strike index and FPA
#         strike_index_all = self.get_strike_index_all(gait_data_df, l_strikes, r_strikes)
#         strike_angle_all = self.get_foot_strike_angle(gait_data_df)
#         FPA_all = self.get_FPA_all(gait_data_df)  # FPA of all the samples
#         param_data = np.column_stack(
#             [l_strikes, r_strikes, l_offs, r_offs, strike_index_all, strike_angle_all, FPA_all])
#         param_data_df = pd.DataFrame(param_data)
#         param_data_df.columns = ['l_strikes', 'r_strikes', 'l_offs', 'r_offs', 'l_strike_index', 'r_strike_index',
#                                  'l_strike_angle', 'r_strike_angle', 'l_FPA', 'r_FPA']
#         param_data_df.insert(0, 'marker_frame', gait_data_df['marker_frame'])
#
#         # get strikes and offs from IMU data
#         estimated_strikes, estimated_offs = self.get_strike_off_from_imu(gait_data_df, param_data_df,
#                                                                          sensor_sampling_rate, check_strike_off=False,
#                                                                          plot_the_strike_off=False)
#         param_data_df.insert(len(param_data_df.columns), 'strikes_IMU', estimated_strikes)
#         param_data_df.insert(len(param_data_df.columns), 'offs_IMU', estimated_offs)
#
#         estimated_strikes_lfilter, estimated_offs_lfilter = self.get_strike_off_from_imu_lfilter(
#             gait_data_df, param_data_df, sensor_sampling_rate, check_strike_off=True)
#         param_data_df.insert(len(param_data_df.columns), 'strikes_IMU_lfilter', estimated_strikes_lfilter)
#         param_data_df.insert(len(param_data_df.columns), 'offs_IMU_lfilter', estimated_offs_lfilter)
#
#         # get loading rate
#         l_steps_1000 = self.get_legal_steps(l_strikes_1000, l_offs_1000, 'left', plate_data_1000)
#         l_LR = self.get_loading_rate(plate_data_1000, l_steps_1000)
#         self.insert_LR_to_param_data(param_data_df, l_LR, 'l_LR')
#         r_steps_1000 = self.get_legal_steps(r_strikes_1000, r_offs_1000, 'right', plate_data_1000)
#         r_LR = self.get_loading_rate(plate_data_1000, r_steps_1000)
#         self.insert_LR_to_param_data(param_data_df, r_LR, 'r_LR')
#         return param_data_df, l_steps_1000, r_steps_1000





