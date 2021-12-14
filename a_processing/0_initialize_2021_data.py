from a_processing.Initializer import SubjectDataInitializer
from const import RAW_DATA_PATH, PROCESSED_DATA_PATH, SUB_NAMES_2021, SUB_AND_TRIALS, TRIAL_NAMES

for subject_folder in SUB_NAMES_2021[3:]:
    trials_to_init = ('nike static', 'nike baseline 24', 'nike SI 24', 'nike baseline 28', 'nike SI 28',
                      'mini static', 'mini baseline 24', 'mini SI 24', 'mini baseline 28', 'mini SI 28')
    readme_xls = RAW_DATA_PATH + subject_folder + '\\readme\\readme.xlsx'
    my_initializer = SubjectDataInitializer(PROCESSED_DATA_PATH, subject_folder, trials_to_init, readme_xls,
                                            initialize_200Hz=True, initialize_1000Hz=True, check_sync=True,
                                            check_running_period=True)
