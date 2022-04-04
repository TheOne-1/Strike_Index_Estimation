import tensorflow as tf
from SharedProcessors.const import SUB_NAMES
import numpy as np


result_date = '220325'
main_feature_weights, aux_1_weights, aux_2_weights, nums_of_main_feature = [], [], [], []
for sub in SUB_NAMES:
    loaded_model = tf.keras.models.load_model('./result_conclusion/{}/model/{}'.format(result_date, sub))
    the_layer_weights = loaded_model.layers[10].weights[0].__array__()
    num_of_main_feature = the_layer_weights.shape[0] - 2
    for i_feature in range(num_of_main_feature):
        main_feature_weights.append(np.mean(np.abs(the_layer_weights[i_feature])))

    aux_1_weights.append(np.mean(np.abs(the_layer_weights[-2])))
    aux_2_weights.append(np.mean(np.abs(the_layer_weights[-1])))
    nums_of_main_feature.append(num_of_main_feature)

print(np.mean(main_feature_weights))
print(np.mean(aux_1_weights))
print(np.mean(aux_2_weights))
print(np.mean(nums_of_main_feature))
