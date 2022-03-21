import numpy as np
from const import RAW_DATA_PATH, TRIAL_NAMES, MOCAP_SAMPLE_RATE,SEGMENT_MARKERS
import matplotlib.pyplot as plt
import scipy.interpolate as interpo
from ViconReader import ViconReader


class GyrSimulator:
    def __init__(self, subject_folder, segment, sampling_fre, static_start=0, static_end=8):
        """

        :param subject_folder: str
        :param segment: str, acceptable segments are : 'trunk', 'pelvis', 'l_thigh', 'r_thigh', 'l_shank', 'r_shank',
        'l_foot', 'r_foot'
        :param static_start: int, the start second of calibration
        :param static_end: int, the end time of calibration
        """
        self._subject_folder = subject_folder
        self._sampling_fre = sampling_fre
        static_mat = self.initialize_static_marker_df(TRIAL_NAMES[0], segment).values
        self._cali_matrix = GyrSimulator.get_marker_cali_matrix(static_mat, static_start, static_end)

    @staticmethod
    def __sync_via_correlation(vicon_norm, sensor_norm, check, segment):
        vicon_norm_zero_mean = vicon_norm - np.mean(vicon_norm)
        sensor_norm_zero_mean = sensor_norm - np.mean(sensor_norm)
        correlation = np.correlate(vicon_norm_zero_mean, sensor_norm_zero_mean, 'full')
        vicon_delay = len(sensor_norm_zero_mean) - np.argmax(correlation) - 1
        if check:
            plt.figure()
            if vicon_delay > 0:
                plt.plot(sensor_norm[vicon_delay:])
                plt.plot(vicon_norm)
            else:
                plt.plot(sensor_norm)
                plt.plot(vicon_norm[-vicon_delay:])
            plt.title('Motion sync check: ' + segment)
            plt.show()
        return vicon_delay

    @staticmethod
    def sync_vicon_sensor(trial_name, sensor_loc, vicon_norm, sensor_norm, start_vicon, check=False):
        if 'static' in trial_name:
            vicon_delay = GyrSimulator._sync_standing(vicon_norm, sensor_norm, sensor_loc, check)
        else:
            vicon_delay = GyrSimulator.__sync_via_correlation(
                vicon_norm[:start_vicon], sensor_norm[:start_vicon*2], check, sensor_loc)
        return vicon_delay

    @staticmethod
    def _sync_standing(vicon_norm, sensor_norm, sensor_loc, check=False):
        vicon_delay = GyrSimulator.__sync_via_correlation(vicon_norm, sensor_norm, check, sensor_loc)
        return vicon_delay

    def initialize_static_marker_df(self, file_name, segment):
        my_vicon_static_reader = ViconReader('{path}{sub_folder}\\{sensor}\\{file_name}.csv'.format(
            path=RAW_DATA_PATH, sub_folder=self._subject_folder, sensor='vicon', file_name=file_name))
        marker_static_df = my_vicon_static_reader.get_marker_data_processed_segment(segment)
        return marker_static_df

    @staticmethod
    def get_marker_cali_matrix(vicon_data, static_start, static_end):
        """
        standing marker data for IMU simulation calibration
        :param vicon_data:
        :param static_start: unit: second
        :param static_end: unit: second
        :return:
        """
        static_start_sample = MOCAP_SAMPLE_RATE * static_start
        static_end_sample = MOCAP_SAMPLE_RATE * static_end
        cali_period_data = vicon_data[static_start_sample:static_end_sample, :]
        vicon_data_average = np.mean(cali_period_data, axis=0)
        cali_matrix = vicon_data_average.reshape([-1, 3])
        return cali_matrix

    def get_gyr(self, segment, vicon_all_df, R_standing_to_ground=None, sampling_rate=MOCAP_SAMPLE_RATE,
                sync_max_len=60):
        """

        :param segment:
        :param vicon_all_df:
        :param R_standing_to_ground:
        :param sampling_rate:
        :param sync_max_len: The maximum length of gyroscope simulation. (60 seconds by default)
        :return:
        """
        # get segment dataframe
        marker_names = [marker + axis for marker in SEGMENT_MARKERS[segment] for axis in ['_x', '_y', '_z']]
        segment_data_df = vicon_all_df[marker_names]

        walking_data = segment_data_df.values[:sync_max_len*self._sampling_fre, :]
        data_len = walking_data.shape[0]
        R_IMU_transform = np.zeros([3, 3, data_len])
        marker_number = int(walking_data.shape[1] / 3)
        next_marker_matrix = walking_data[0, :].reshape([marker_number, 3])
        gyr_middle = np.zeros([data_len, 3])
        for i_frame in range(data_len - 1):
            current_marker_matrix = next_marker_matrix
            next_marker_matrix = walking_data[i_frame + 1, :].reshape([marker_number, 3])
            [R_one_sample, t] = GyrSimulator.rigid_transform_3D(current_marker_matrix, next_marker_matrix)
            theta = np.math.acos((np.matrix.trace(R_one_sample) - 1) / 2)
            a, b = np.linalg.eig(R_one_sample)
            for i_eig in range(a.__len__()):
                if abs(a[i_eig].imag) < 1e-12:
                    vector = b[:, i_eig].real
                    break
                if i_eig == a.__len__():
                    raise RuntimeError('no eig')

            if (R_one_sample[2, 1] - R_one_sample[1, 2]) * vector[0] < 0:  # check the direction of the rotation axis
                vector = -vector

            [R_from_static_cali, _] = GyrSimulator.rigid_transform_3D(self._cali_matrix, current_marker_matrix)
            if R_standing_to_ground is not None:
                R_IMU_transform[:, :, i_frame] = np.matmul(R_standing_to_ground, R_from_static_cali.T)
            else:
                R_IMU_transform[:, :, i_frame] = R_from_static_cali
            vector = np.dot(R_IMU_transform[:, :, i_frame].T, vector)
            gyr_middle[i_frame, :] = theta * vector * sampling_rate

        step_middle = np.arange(0.5 / sampling_rate, data_len / sampling_rate + 1e-6,
                                1 / sampling_rate)
        step_gyr = np.arange(0, data_len / sampling_rate, 1 / sampling_rate)
        # in splprep, s the amount of smoothness. 6700 might be appropriate
        tck, step = interpo.splprep(gyr_middle.T, u=step_middle, s=0)
        gyr = interpo.splev(step_gyr, tck, der=0)
        gyr = np.column_stack([gyr[0], gyr[1], gyr[2]])
        return gyr

    # get virtual marker and R_IMU_transform
    @staticmethod
    def get_virtual_marker(simulated_marker, vicon_data, marker_cali_matrix, R_standing_to_ground):
        segment_marker_num = marker_cali_matrix.shape[0]
        data_len = vicon_data.shape[0]
        virtual_marker = np.zeros([data_len, 3])
        R_IMU_transform = np.zeros([3, 3, data_len])
        for i_frame in range(data_len):
            current_marker_matrix = vicon_data[i_frame, :].reshape([segment_marker_num, 3])
            [R_between_frames, t] = GyrSimulator.rigid_transform_3D(marker_cali_matrix, current_marker_matrix)
            virtual_marker[i_frame, :] = (np.dot(R_between_frames, simulated_marker) + t)
            R_IMU_transform[:, :, i_frame] = np.matmul(R_standing_to_ground, R_between_frames.T)
        return virtual_marker, R_IMU_transform

    @staticmethod
    def rigid_transform_3D(A, B):
        assert len(A) == len(B)

        N = A.shape[0]  # total points
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        # centre the points
        AA = A - np.tile(centroid_A, (N, 1))
        BB = B - np.tile(centroid_B, (N, 1))
        # dot is matrix multiplication for array
        H = np.dot(AA.T, BB)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        # special reflection case
        if np.linalg.det(R) < 0:
            # print
            # "Reflection detected"
            Vt[2, :] *= -1
            R = np.dot(Vt.T, U.T)
        t = -np.dot(R, centroid_A.T) + centroid_B.T
        return R, t
