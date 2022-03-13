from ProcessorSICrossVali import ProcessorSICrossVali
from SharedProcessors.const import SUB_AND_SI_TRIALS, SI_TRIALS
import matplotlib.pyplot as plt
import numpy as np
import copy


train = copy.deepcopy(SUB_AND_SI_TRIALS)
# train = {'190521GongChangyang': SI_TRIALS}

if __name__ == "__main__":
    date = '211206'
    my_SI_processor = ProcessorSICrossVali(train, ['l_foot'], date, strike_off_from_IMU=1,
                                           do_input_norm=True, tune_hp=True)
    step_num = len(my_SI_processor.train_all_data_list)
    print('Number of steps before removing: {}'.format(step_num))

    i_step = 0
    counter_bad = 0
    bad_si = []
    input_list, _, output_list = my_SI_processor.train_all_data_list.get_input_output_list()
    sub_list = my_SI_processor.train_all_data_list.get_sub_id_list()
    trial_list = my_SI_processor.train_all_data_list.get_trial_id_list()
    while i_step < len(my_SI_processor.train_all_data_list):
        # delete steps without a valid strike index
        strikes = np.where(input_list[i_step][:, -1] == 1)[0]
        if np.max(output_list[i_step]) <= 0 or np.max(output_list[i_step]) >= 1:
            if np.max(output_list[i_step]) == 0.0:
                prntval = np.min(output_list[i_step])
            else:
                prntval = np.max(output_list[i_step])
            bad_si.append(prntval)
            my_SI_processor.train_all_data_list.pop(i_step)
            counter_bad += 1

        else:
            i_step += 1
    print(f"There were {counter_bad} bad steps here due to SI problems")
    # train_all_data_list = my_SI_processor.clean_all_data(my_SI_processor.train_all_data_list,
    #                                                      my_SI_processor.sensor_sampling_fre)

    step_num = len(my_SI_processor.train_all_data_list)
    print('Number of steps after removing: {}'.format(step_num))

    bad_si = [round(t * 100, 1) for t in bad_si if t > -100]
    print(max(bad_si))
    print(min(bad_si))
    print(bad_si)