import matplotlib.pyplot as plt
from AllSubData import AllSubData
import scipy.interpolate as interpo
from const import SUB_NAMES, COLORS, DATA_COLUMNS_XSENS, MOCAP_SAMPLE_RATE
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler
from keras.layers import *
from keras.models import Model
from keras import regularizers
from const import TRIAL_NAMES
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from keras import optimizers
# from keras.callbacks import EarlyStopping
import numpy as np

def define_cnn_model_typeCC(hp):
    main_input = Input(shape=(50,6), name='main_input')
    base_size = int(200*0.01)
    hp_filters = 10 #hp.Int("filters", min_value=1, max_value=20, step=1, default=10)
    hp_tower_1_kernel_size = 10 #hp.Int("T1KS", min_value=2, max_value=16, step=1, default=10)
    hp_tower_3_kernel_size = 8 #  hp.Int("T3KS", min_value=2, max_value=16, step=1, default= 8)
    hp_tower_4_kernel_size = 7 # hp.Int("T4KS", min_value=2, max_value=16, step=1, default=7)
    hp_NN_layer_1_units = 30 # hp.Int("NNL1U", min_value=2, max_value=40, step=1, default=20)
    hp_NN_layer_2_units = 35 #hp.Int("NNL2U", min_value=2, max_value=40, step=1, default=20)
    hp_NN_layer_3_units = 30 # hp.Int("NNL3U", min_value=2, max_value=40, step=1, default=20)
    hp_NN_layer_4_units = 25 # hp.Int("NNL4U", min_value=2, max_value=40, step=1, default=20)
    hp_NN_layer_5_units = hp.Int("NNL5U", min_value=2, max_value=40, step=1, default=20)
    hp_use_bias = False # hp.Boolean("use_bias", default=False)
    hp_number_of_hidden_layers = 4 #hp.Int("numberOfHiddenLayers", min_value=1, max_value=5, default=3)
    # kernel_init = 'lecun_uniform'
    kernel_regu = regularizers.l2(0.01)
    # for each feature, add 30 * 1 cov kernel
    tower_1 = Conv1D(filters=hp_filters, kernel_size=hp_tower_1_kernel_size*base_size, kernel_regularizer=kernel_regu)(main_input)
    tower_1 = MaxPool1D(pool_size=25*base_size-hp_tower_1_kernel_size*base_size+1)(tower_1)

    # for each feature, add 10 * 1 cov kernel
    tower_3 = Conv1D(filters=hp_filters, kernel_size=hp_tower_3_kernel_size*base_size, kernel_regularizer=kernel_regu)(main_input)
    tower_3 = MaxPool1D(pool_size=25*base_size-hp_tower_3_kernel_size*base_size+1)(tower_3)

    # for each feature, add 4 * 1 cov kernel
    tower_4 = Conv1D(filters=hp_filters, kernel_size=hp_tower_4_kernel_size*base_size, kernel_regularizer=kernel_regu)(main_input)
    tower_4 = MaxPool1D(pool_size=25*base_size-hp_tower_4_kernel_size*base_size+1)(tower_4)

    # for each feature, add 1 * 1 cov kernel
    tower_5 = Conv1D(filters=hp_filters, kernel_size=1, kernel_regularizer=kernel_regu)(main_input)
    tower_5 = MaxPool1D(pool_size=25*base_size-1*base_size+1)(tower_5)

    joined_outputs = Concatenate(axis=-1)([tower_1, tower_3, tower_4, tower_5])
    joined_outputs = Activation('relu')(joined_outputs)
    main_outputs = Flatten()(joined_outputs)

    aux_input = Input(shape=(2,), name='aux_input')
    aux_joined_outputs = Concatenate()([main_outputs, aux_input])
    if hp_number_of_hidden_layers >= 1:
        aux_joined_outputs = Dense(hp_NN_layer_1_units, activation='relu', use_bias=hp_use_bias)(aux_joined_outputs)
    if hp_number_of_hidden_layers >= 2:    
        aux_joined_outputs = Dense(hp_NN_layer_2_units, activation='relu', use_bias=hp_use_bias)(aux_joined_outputs)
    if hp_number_of_hidden_layers >= 3:
        aux_joined_outputs = Dense(hp_NN_layer_3_units, activation='relu', use_bias=hp_use_bias)(aux_joined_outputs)
    if hp_number_of_hidden_layers >= 4:
        aux_joined_outputs = Dense(hp_NN_layer_4_units, activation='relu', use_bias=hp_use_bias)(aux_joined_outputs)
    if hp_number_of_hidden_layers >= 5:
        aux_joined_outputs = Dense(hp_NN_layer_5_units, activation='relu', use_bias=hp_use_bias)(aux_joined_outputs)
    aux_joined_outputs = Dense(1, activation='linear')(aux_joined_outputs)
    model = Model(inputs=[main_input, aux_input], outputs=aux_joined_outputs)
    optimizer = optimizers.Nadam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    model.compile(loss='mse', metrics=['mse'], optimizer=optimizer)
    return model
