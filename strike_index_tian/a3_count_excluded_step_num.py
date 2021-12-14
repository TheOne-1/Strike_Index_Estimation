from ProcessorSICrossVali import ProcessorSICrossVali
from SharedProcessors.const import SUB_AND_SI_TRIALS, SI_TRIALS
import copy


train = copy.deepcopy(SUB_AND_SI_TRIALS)
# train = {'190521GongChangyang': SI_TRIALS}

if __name__ == "__main__":
    date = '211201'
    my_SI_processor = ProcessorSICrossVali(train, ['l_foot'], date, strike_off_from_IMU=1,
                                           do_input_norm=True, tune_hp=True)
    step_num = len(my_SI_processor.train_all_data_list)
    print('Number of steps before removing: {}'.format(step_num))
    train_all_data_list = my_SI_processor.clean_all_data(my_SI_processor.train_all_data_list,
                                                         my_SI_processor.sensor_sampling_fre)

    step_num = len(my_SI_processor.train_all_data_list)
    print('Number of steps after removing: {}'.format(step_num))

