import numpy as np
import copy

# in Haisheng sensor's column names, x and y are switched to make it the same as Xsens column
COLUMN_NAMES_HAISHENG = ['hour', 'minute', 'second', 'millisecond', 'acc_y', 'acc_x', 'acc_z', 'gyr_y', 'gyr_x',
                         'gyr_z', 'mag_y', 'mag_x', 'mag_z']

SEGMENT_MARKERS = {'trunk': ['RAC', 'LAC', 'C7'], 'pelvis': ['RIAS', 'LIAS', 'LIPS', 'RIPS'],
                   'l_thigh': ['LTC1', 'LTC2', 'LTC3', 'LTC4', 'LFME', 'LFLE'],
                   'r_thigh': ['RTC1', 'RTC2', 'RTC3', 'RTC4', 'RFME', 'RFLE'],
                   'l_shank': ['LSC1', 'LSC2', 'LSC3', 'LSC4', 'LTAM', 'LFAL'],
                   'r_shank': ['RSC1', 'RSC2', 'RSC3', 'RSC4', 'RTAM', 'RFAL'],
                   'l_foot': ['LFM2', 'LFM5', 'LFCC'], 'r_foot': ['RFM2', 'RFM5', 'RFCC']}

FORCE_NAMES = ['marker_frame', 'f_1_x', 'f_1_y', 'f_1_z', 'c_1_x', 'c_1_y', 'c_1_z',
               'f_2_x', 'f_2_y', 'f_2_z', 'c_2_x', 'c_2_y', 'c_2_z']

DATA_COLUMNS_XSENS = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'mag_x', 'mag_y', 'mag_z', 'q0', 'q1',
                      'q2', 'q3']

DATA_COLUMNS_IMU = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'mag_x', 'mag_y', 'mag_z']

XSENS_SENSOR_LOACTIONS = ['trunk', 'pelvis', 'l_thigh', 'l_shank', 'l_foot']

XSENS_FILE_NAME_DIC = {'trunk': 'MT_0370064E_000.mtb', 'pelvis': 'MT_0370064C_000.mtb',
                       'l_thigh': 'MT_0370064B_000.mtb', 'l_shank': 'MT_0370064A_000.mtb',
                       'l_foot': 'MT_03700647_000.mtb'}

HAISHENG_SENSOR_SAMPLE_RATE = 100
MOCAP_SAMPLE_RATE = 200
PLATE_SAMPLE_RATE = 1000
STATIC_STANDING_PERIOD = 10  # unit: second

with open('../configuration.txt', 'r') as config:
    RAW_DATA_PATH = config.readline()

path_index = RAW_DATA_PATH.rfind('\\', 0, len(RAW_DATA_PATH) - 2)
MAIN_DATA_PATH = RAW_DATA_PATH[:path_index]
PROCESSED_DATA_PATH = RAW_DATA_PATH[:path_index] + '\\ProcessedData'

LOADING_RATE_NORMALIZATION = True

COP_DIFFERENCE = np.array([279.4, 784, 0])  # reset coordinate difference

COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'gray', 'rosybrown', 'firebrick', 'olive', 'darkgreen',
          'slategray', 'navy', 'slateblue', 'm', 'indigo', 'maroon', 'peru', 'seagreen']

TRIAL_NAMES = ['nike static', 'nike baseline 24', 'nike SI 24', 'nike SR 24', 'nike baseline 28', 'nike SI 28',
               'nike SR 28', 'mini static', 'mini baseline 24', 'mini SI 24', 'mini SR 24', 'mini baseline 28',
               'mini SI 28', 'mini SR 28']
SI_TRIALS = ('nike SI 24', 'nike SI 28', 'mini SI 24', 'mini SI 28')

SUB_AND_TRIALS = {'190521GongChangyang': TRIAL_NAMES, '190523ZengJia': TRIAL_NAMES,
                  '190522QinZhun': TRIAL_NAMES, '190522YangCan': TRIAL_NAMES, '190521LiangJie': TRIAL_NAMES,
                  '190517ZhangYaqian': TRIAL_NAMES, '190518MouRongzi': TRIAL_NAMES, '190518FuZhinan': TRIAL_NAMES,
                  '190522SunDongxiao': TRIAL_NAMES, '190513YangYicheng':  TRIAL_NAMES}

SUB_NAMES = tuple(SUB_AND_TRIALS.keys())
SUB_AND_SI_TRIALS = {sub_name: SI_TRIALS for sub_name in SUB_NAMES}

SUB_AND_NIKE_SI_TRIALS = {sub_name: ('nike SI 24', 'nike SI 28') for sub_name in SUB_NAMES}
SUB_AND_MINI_SI_TRIALS = {sub_name: ('mini SI 24', 'mini SI 28') for sub_name in SUB_NAMES}

# The orientation of left foot xsens sensor was wrong
XSENS_ROTATION_CORRECTION_NIKE = {
    '190511ZhuJiayi': {'l_foot': [[-1, 0, 0],
                                  [0, -1, 0],
                                  [0, 0, 1]]}}

# magnetic field interference occurred in Wang Dianxin's data, so YuHongzhe's data were used instead
SPECIFIC_CALI_MATRIX = {
    '190414WangDianxin': {'r_foot': [[0.92751222, 0.34553155, -0.14257993],
                                     [-0.37081009, 0.80245287, -0.46751393],
                                     [-0.04712714, 0.48649496, 0.87241142]]}}

ROTATION_VIA_STATIC_CALIBRATION = False

TRIAL_START_BUFFER = 3       # 3 seconds filter buffer
FILTER_WIN_LEN = 100        # The length of FIR filter window


FONT_SIZE = 18
FONT_DICT = {'fontsize': FONT_SIZE, 'fontname': 'DejaVu Sans'}
FONT_DICT_SMALL = {'fontsize': 16, 'fontname': 'DejaVu Sans'}
LINE_WIDTH = 2
MININ_SHOE_LENGTHS = {'190521GongChangyang': 262, '190523ZengJia': 252, '190522QinZhun': 252,
                      '190522YangCan': 252, '190521LiangJie': 232, '190517ZhangYaqian': 252,
                      '190518MouRongzi': 252, '190518FuZhinan': 252, '190522SunDongxiao': 262,
                      '190510HeMing': 252, '190513YangYicheng': 262}

TRAD_SHOE_LENGTHS = {'190521GongChangyang': 302, '190523ZengJia': 288, '190522QinZhun': 288,
                     '190522YangCan': 266, '190521LiangJie': 254, '190517ZhangYaqian': 266,
                     '190518MouRongzi': 266, '190518FuZhinan': 266, '190522SunDongxiao': 302,
                     '190510HeMing': 266, '190513YangYicheng': 302}

GOOD_SUBS = ['190521GongChangyang', '190523ZengJia', '190522QinZhun', '190522YangCan', '190521LiangJie', '190517ZhangYaqian', '190518MouRongzi', '190518FuZhinan', '190522SunDongxiao', '190510HeMing']

