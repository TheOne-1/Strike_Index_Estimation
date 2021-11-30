from strike_index_tian.ProcessorSICrossVali import ProcessorSICrossVali
from SharedProcessors.const import SUB_AND_SI_TRIALS
import copy
import tensorflow as tf
import pandas as pd
import itertools

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
IMU_locations = ['l_foot']

train = copy.deepcopy(SUB_AND_SI_TRIALS)

train = {'190521GongChangyang': SUB_AND_SI_TRIALS['190521GongChangyang'],
         '190513YangYicheng': SUB_AND_SI_TRIALS['190513YangYicheng']}


def regular_run(basename):
    my_SI_processor = ProcessorSICrossVali(train, IMU_locations, date, strike_off_from_IMU=1, do_output_norm=False,
                                           do_input_norm=True, tune_hp=False)
    trials = [2, 5, 9, 12]
    print("starting regular run")
    trial_name = basename
    avg_correlation = my_SI_processor.prepare_data_cross_vali(
        pre_samples=12, post_samples=20, test_name=trial_name, trials=trials, train_num=len(train) - 1)
    print(avg_correlation)


def subject_sufficiently(basename):
    my_SI_processor = ProcessorSICrossVali(train, IMU_locations, date, strike_off_from_IMU=1, do_output_norm=False,
                                           do_input_norm=True, tune_hp=False)
    trials = [2, 5, 9, 12]
    print("starting subject sufficiently")
    for j in range(7, 11):
        SS_corr_DF = pd.DataFrame(columns=["correlation", "Train_num"])
        for i in range(1, 10):
            trial_name = f"{basename}_{j}_training_with_{i}"
            print(trial_name)
            avg_correlation = my_SI_processor.prepare_data_cross_vali(pre_samples=12, post_samples=20,
                                                                      test_name=trial_name, trials=trials, train_num=i)
            new_data = {"correlation": avg_correlation, "Train_num": i}
            SS_corr_DF = SS_corr_DF.append(new_data, ignore_index=True)
            print(f"For number of training sub = {i}, {avg_correlation}")
        SS_corr_DF.to_csv(f"results/SubCalc/{basename}_bout{j}.csv")


def window_opt(basename):
    my_SI_processor = ProcessorSICrossVali(train, IMU_locations, date, strike_off_from_IMU=1, do_output_norm=False,
                                           do_input_norm=True, tune_hp=False)
    trials = [2, 5, 9, 12]
    print("starting window opt")
    basis_vector = [2, 4, 6, 8, 10, 12, 16, 18, 20, 24]
    samples = [x for x in itertools.product(basis_vector, basis_vector)]
    end_start_corr_DF = pd.DataFrame(columns=["correlation", "start", "end"])
    for i, (pre_sample, post_sample) in enumerate(samples):
        number_of_trials = len(samples)
        print(f"Working with {pre_sample}, {post_sample}, which is {i + 1} of {number_of_trials}")
        trial_name = f"{pre_sample}_{post_sample}_{basename}"
        avg_correlation = my_SI_processor.prepare_data_cross_vali(pre_samples=pre_sample, post_samples=post_sample,
                                                                  test_name=trial_name, trials=trials)
        new_data = {"correlation": avg_correlation, "start": pre_sample, "end": post_sample}
        end_start_corr_DF = end_start_corr_DF.append(new_data, ignore_index=True)
        print(f"For {pre_sample} and {post_sample}, {avg_correlation}")
    end_start_corr_DF.to_csv(f"results/window_opt/{basename}_{pre_sample}_{post_sample}.csv")
    print(end_start_corr_DF)


def four_cond(basename):
    my_SI_processor = ProcessorSICrossVali(train, IMU_locations, date, strike_off_from_IMU=1, do_output_norm=False,
                                           do_input_norm=True, tune_hp=False)
    print("starting four cond")
    pre_samples = 12
    post_samples = 20
    trials = [2, 9]
    trial_name = basename + "_24"
    avg_correlation1 = my_SI_processor.prepare_data_cross_vali(
        pre_samples=pre_samples, post_samples=post_samples, test_name=trial_name, trials=trials,
        train_num=len(train) - 1)
    print(f"{trial_name}:{avg_correlation1}")
    trials = [5, 12]
    trial_name = basename + "_28"
    avg_correlation2 = my_SI_processor.prepare_data_cross_vali(
        pre_samples=pre_samples, post_samples=post_samples, test_name=trial_name, trials=trials,
        train_num=len(train) - 1)
    print(f"{trial_name}:{avg_correlation2}")
    trials = [2, 5]
    trial_name = basename + "_Trad"
    avg_correlation3 = my_SI_processor.prepare_data_cross_vali(
        pre_samples=pre_samples, post_samples=post_samples, test_name=trial_name, trials=trials,
        train_num=len(train) - 1)
    print(f"{trial_name}:{avg_correlation3}")
    trials = [9, 12]
    trial_name = basename + "_Minim"
    avg_correlation4 = my_SI_processor.prepare_data_cross_vali(
        pre_samples=pre_samples, post_samples=post_samples, test_name=trial_name, trials=trials,
        train_num=len(train) - 1)
    print(f"{trial_name}:{avg_correlation4}")
    print(f"{avg_correlation1}:{avg_correlation1}:{avg_correlation3}:{avg_correlation4}")


def hp_opt(trial_name):
    trials = [2, 5, 9, 12]
    print("starting hp opt")
    pre_samples = 12
    post_samples = 20
    my_SI_processor = ProcessorSICrossVali(train, IMU_locations, date, strike_off_from_IMU=2, do_output_norm=False,
                                           do_input_norm=True, hp_tune=True)
    avg_correlation = my_SI_processor.prepare_data_cross_vali(pre_samples=pre_samples, post_samples=post_samples,
                                                              test_name=trial_name, trials=trials,
                                                              train_num=len(train) - 1)


def cross_test(basename):
    my_SI_processor = ProcessorSICrossVali(train, IMU_locations, date, strike_off_from_IMU=1, do_output_norm=False,
                                           do_input_norm=True, tune_hp=False)
    print("starting cross test")
    pre_samples = 12
    post_samples = 20
    trials = [2, 9]
    trials_test = [5, 12]
    trial_name = basename + "_tr24te28"
    avg_correlation1 = my_SI_processor.prepare_data_cross_vali(
        pre_samples=pre_samples, post_samples=post_samples, test_name=trial_name, trials=trials,
        trials_test=trials_test, train_num=len(train) - 1)
    print(f"{trial_name}:{avg_correlation1}")
    trials = [5, 12]
    trials_test = [2, 9]
    trial_name = basename + "_tr28te24"
    avg_correlation2 = my_SI_processor.prepare_data_cross_vali(
        pre_samples=pre_samples, post_samples=post_samples, test_name=trial_name, trials=trials,
        trials_test=trials_test, train_num=len(train) - 1)
    print(f"{trial_name}:{avg_correlation2}")
    trials = [2, 5]
    trials_test = [9, 12]
    trial_name = basename + "_trTradteMinim"
    avg_correlation3 = my_SI_processor.prepare_data_cross_vali(
        pre_samples=pre_samples, post_samples=post_samples, test_name=trial_name, trials=trials,
        trials_test=trials_test, train_num=len(train) - 1)
    print(f"{trial_name}:{avg_correlation3}")
    trials = [9, 12]
    trials_test = [2, 5]
    trial_name = basename + "_trMinimteTrad"
    avg_correlation4 = my_SI_processor.prepare_data_cross_vali(
        pre_samples=pre_samples, post_samples=post_samples, test_name=trial_name, trials=trials,
        trials_test=trials_test, train_num=len(train) - 1)
    print(f"{trial_name}:{avg_correlation4}")

    print(f"{avg_correlation1}:{avg_correlation1}:{avg_correlation3}:{avg_correlation4}")


if __name__ == "__main__":
    date = '211201'
    regular_run("main")
    # four_cond("main")
    # cross_test("main")
