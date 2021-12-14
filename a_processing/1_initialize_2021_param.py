from SharedProcessors.const import PROCESSED_DATA_PATH, SUB_NAMES_2021, SUB_AND_SI_TRIALS
from a_processing.ParameterProcessor import ParamProcessor


def initialize_param(sub_names):
    for subject_folder in sub_names:
        trials = SUB_AND_SI_TRIALS[subject_folder]
        my_initializer = ParamProcessor(subject_folder, trials, check_steps=False, plot_strike_off=False,
                                        initialize_200Hz=True)
        my_initializer.start_initalization(PROCESSED_DATA_PATH)


if __name__ == '__main__':
    initialize_param(SUB_NAMES_2021)

