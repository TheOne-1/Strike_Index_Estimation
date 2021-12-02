import matplotlib.pyplot as plt
from AllSubData import AllSubData
import scipy.interpolate as interpo
from const import SUB_NAMES, COLORS, DATA_COLUMNS_XSENS, MOCAP_SAMPLE_RATE
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from const import TRIAL_NAMES
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras import optimizers
# from keras.callbacks import EarlyStopping
import numpy as np
from pearsonr import pearson_r


def define_model(hp):
    base_size = 33
    main_input = Input(shape=(base_size, 6), name='main_input')
    hp_filters = hp.Int("filters", min_value=20, max_value=40, step=1, default=30)
    hp_tower_1_kernel_size = hp.Int("T1KS", min_value=10, max_value=20, step=2, default=20)
    hp_tower_2_kernel_size = hp.Int("T2KS", min_value=2, max_value=8, step=2, default=5)
    hp_NN_layer_1_units = hp.Int("NNL1U", min_value=20, max_value=40, step=1, default=30)
    hp_learning_rate = hp.Float("LR", min_value=1e-5, max_value=1e-3, default=1e-4, sampling='log')
    # kernel_init = 'lecun_uniform'
    kernel_regu = regularizers.l1(0.01)
    # for each feature, add 30 * 1 cov kernel
    tower_1 = Conv1D(filters=hp_filters, kernel_size=hp_tower_1_kernel_size, kernel_regularizer=kernel_regu)(main_input)
    tower_1 = MaxPool1D(pool_size=base_size - hp_tower_1_kernel_size + 1)(tower_1)

    # for each feature, add 10 * 1 cov kernel
    tower_2 = Conv1D(filters=hp_filters, kernel_size=hp_tower_2_kernel_size, kernel_regularizer=kernel_regu)(main_input)
    tower_2 = MaxPool1D(pool_size=base_size-hp_tower_2_kernel_size+1)(tower_2)

    joined_outputs = Concatenate(axis=-1)([tower_1, tower_2])
    joined_outputs = Activation('relu')(joined_outputs)
    main_outputs = Flatten()(joined_outputs)

    aux_input = Input(shape=(2,), name='aux_input')
    aux_joined_outputs = Concatenate()([main_outputs, aux_input])
    aux_joined_outputs = Dense(hp_NN_layer_1_units, activation='relu')(aux_joined_outputs)
    aux_joined_outputs = Dense(1, activation='linear')(aux_joined_outputs)
    model = Model(inputs=[main_input, aux_input], outputs=aux_joined_outputs)
    optimizer = optimizers.Nadam(lr=hp_learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

