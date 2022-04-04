from ProcessorSICrossVali import ProcessorSICrossVali, ProcessorSICrossValiModelSize
from SharedProcessors.const import SUB_AND_SI_TRIALS, SI_TRIALS
import copy
import pandas as pd
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'        # seems CPU ('-1') is faster than GPU ('-0') for this CNN model
IMU_locations = ['l_foot']

train = copy.deepcopy(SUB_AND_SI_TRIALS)

# train = {'190521GongChangyang': SI_TRIALS,
#          'Z211208DingYuechen': SI_TRIALS,
#          '211204WangDianxin': SI_TRIALS}


def regular_run(basename):
    my_SI_processor = ProcessorSICrossVali(train, IMU_locations, date, strike_off_from_IMU=1,
                                           do_input_norm=True, tune_hp=False)       # !!!
    print("starting regular run")
    trial_name = basename
    avg_correlation = my_SI_processor.prepare_data_cross_vali(
        test_name=trial_name, train_num=len(train) - 1)
    print(avg_correlation)


def subject_sufficiently(basename):
    my_SI_processor = ProcessorSICrossVali(train, IMU_locations, date, strike_off_from_IMU=1,
                                           do_input_norm=True)
    print("starting subject sufficiently")
    for j in range(7, 11):
        SS_corr_DF = pd.DataFrame(columns=["correlation", "Train_num"])
        for i in range(1, 10):
            trial_name = f"{basename}_{j}_training_with_{i}"
            print(trial_name)
            avg_correlation = my_SI_processor.prepare_data_cross_vali(test_name=trial_name, train_num=i)
            new_data = {"correlation": avg_correlation, "Train_num": i}
            SS_corr_DF = SS_corr_DF.append(new_data, ignore_index=True)
            print(f"For number of training sub = {i}, {avg_correlation}")
        SS_corr_DF.to_csv(f"results/SubCalc/{basename}_bout{j}.csv")


def four_cond(basename):
    my_SI_processor = ProcessorSICrossVali(train, IMU_locations, date, strike_off_from_IMU=1,
                                           do_input_norm=True)
    print("starting four cond")
    trials = [2, 9]
    trial_name = basename + "_24"
    avg_correlation1 = my_SI_processor.prepare_data_cross_vali(
        test_name=trial_name, trials=trials,
        train_num=len(train) - 1)
    print(f"{trial_name}:{avg_correlation1}")
    trials = [5, 12]
    trial_name = basename + "_28"
    avg_correlation2 = my_SI_processor.prepare_data_cross_vali(
        test_name=trial_name, trials=trials,
        train_num=len(train) - 1)
    print(f"{trial_name}:{avg_correlation2}")
    trials = [2, 5]
    trial_name = basename + "_Trad"
    avg_correlation3 = my_SI_processor.prepare_data_cross_vali(
        test_name=trial_name, trials=trials,
        train_num=len(train) - 1)
    print(f"{trial_name}:{avg_correlation3}")
    trials = [9, 12]
    trial_name = basename + "_Minim"
    avg_correlation4 = my_SI_processor.prepare_data_cross_vali(
        test_name=trial_name, trials=trials,
        train_num=len(train) - 1)
    print(f"{trial_name}:{avg_correlation4}")
    print(f"{avg_correlation1}:{avg_correlation1}:{avg_correlation3}:{avg_correlation4}")


def cross_test(basename):
    my_SI_processor = ProcessorSICrossVali(train, IMU_locations, date, strike_off_from_IMU=1,
                                           do_input_norm=True)
    print("starting cross test")
    trials = [2, 9]
    trials_test = [5, 12]
    trial_name = basename + "_tr24te28"
    avg_correlation1 = my_SI_processor.prepare_data_cross_vali(
        test_name=trial_name, trials=trials,
        trials_test=trials_test, train_num=len(train) - 1)
    print(f"{trial_name}:{avg_correlation1}")
    trials = [5, 12]
    trials_test = [2, 9]
    trial_name = basename + "_tr28te24"
    avg_correlation2 = my_SI_processor.prepare_data_cross_vali(
        test_name=trial_name, trials=trials,
        trials_test=trials_test, train_num=len(train) - 1)
    print(f"{trial_name}:{avg_correlation2}")
    trials = [2, 5]
    trials_test = [9, 12]
    trial_name = basename + "_trTradteMinim"
    avg_correlation3 = my_SI_processor.prepare_data_cross_vali(
        test_name=trial_name, trials=trials,
        trials_test=trials_test, train_num=len(train) - 1)
    print(f"{trial_name}:{avg_correlation3}")
    trials = [9, 12]
    trials_test = [2, 5]
    trial_name = basename + "_trMinimteTrad"
    avg_correlation4 = my_SI_processor.prepare_data_cross_vali(
        test_name=trial_name, trials=trials,
        trials_test=trials_test, train_num=len(train) - 1)
    print(f"{trial_name}:{avg_correlation4}")

    print(f"{avg_correlation1}:{avg_correlation1}:{avg_correlation3}:{avg_correlation4}")


def evaluate_train_set_influence(basename):
    my_SI_processor = ProcessorSICrossVali(train, IMU_locations, date, strike_off_from_IMU=1, do_input_norm=True)
    trials_test = [2, 5, 9, 12]
    trials = [2, 9]
    trial_name = basename + "_tr_24_te_all"
    my_SI_processor.prepare_data_cross_vali(test_name=trial_name, trials=trials, trials_test=trials_test,
                                            train_num=len(train) - 1)
    trials = [5, 12]
    trial_name = basename + "_tr_28_te_all"
    my_SI_processor.prepare_data_cross_vali(test_name=trial_name, trials=trials, trials_test=trials_test,
                                            train_num=len(train) - 1)
    trials = [2, 5]
    trial_name = basename + "_tr_trad_te_all"
    my_SI_processor.prepare_data_cross_vali(test_name=trial_name, trials=trials, trials_test=trials_test,
                                            train_num=len(train) - 1)
    trials = [9, 12]
    trial_name = basename + "_tr_mini_te_all"
    my_SI_processor.prepare_data_cross_vali(test_name=trial_name, trials=trials, trials_test=trials_test,
                                            train_num=len(train) - 1)


def evaluate_subject_sufficiency(basename):
    my_SI_processor = ProcessorSICrossVali(train, IMU_locations, date, strike_off_from_IMU=1, do_input_norm=True)
    print("starting subject sufficiently")
    for i in range(1, len(train)-1):
        trial_name = f"{basename}_training_with_{i}"
        my_SI_processor.prepare_data_cross_vali(test_name=trial_name, train_num=i)


def evaluate_cnn_size(basename):
    my_SI_processor = ProcessorSICrossValiModelSize(train, IMU_locations, date, strike_off_from_IMU=1, do_input_norm=True)
    print("starting evaluating cnn size")
    my_SI_processor.prepare_data_cross_vali(1, 8, test_name=basename+'_size_8_conv', train_num=len(train)-1)
    my_SI_processor.prepare_data_cross_vali(1, 4, test_name=basename+'_size_4_conv', train_num=len(train)-1)
    my_SI_processor.prepare_data_cross_vali(1, 2, test_name=basename+'_size_2_conv', train_num=len(train)-1)


if __name__ == "__main__":
    date = '220325'
    regular_run("main")
    four_cond("main")
    cross_test("main")
    evaluate_cnn_size("main")
    evaluate_subject_sufficiency("main")











