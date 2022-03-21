
from ProcessorSI import ProcessorSI
from ProcessorSICrossVali import ProcessorSICrossVali
from const import SUB_AND_RUNNING_TRIALS, RUNNING_TRIALS
import copy
import tensorflow as tf

try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass



import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'



IMU_locations = ['l_foot']


# train = copy.deepcopy(SUB_AND_RUNNING_TRIALS)
# del train['190522QinZhun']
# del train['190513YangYicheng']
# del train['190513OuYangjue']



train = {'190521GongChangyang': RUNNING_TRIALS[:1]}

test = {'190522QinZhun':  RUNNING_TRIALS[:1]}

my_SI_processor = ProcessorSI(train, test, IMU_locations, strike_off_from_IMU=2, do_output_norm=True)
predict_result_all = my_SI_processor.prepare_data()
my_SI_processor.cnn_solution()


# train = copy.deepcopy(SUB_AND_RUNNING_TRIALS)
# del train['190513YangYicheng']
# del train['190513OuYangjue']
# # train = {'190521GongChangyang': RUNNING_TRIALS,
# #          '190522YangCan': RUNNING_TRIALS}
# # cross validation
# my_LR_processor = ProcessorLRCNNv5(train, 200, strike_off_from_IMU=2, do_input_norm=True, do_output_norm=True)
# my_LR_processor.prepare_data_cross_vali()
