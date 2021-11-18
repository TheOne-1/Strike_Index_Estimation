from tensorflow.keras import backend as K

def pearson_r(y_true, y_pred):
    n_y_true = (y_true - K.mean(y_true[:])) / K.std(y_true[:])
    n_y_pred = (y_pred - K.mean(y_pred[:])) / K.std(y_pred[:])  
    top=K.sum((n_y_true[:]-K.mean(n_y_true[:]))*(n_y_pred[:]-K.mean(n_y_pred[:])),axis=[-1,-2])
    bottom=K.sqrt(K.sum(K.pow((n_y_true[:]-K.mean(n_y_true[:])),2),axis=[-1,-2])*K.sum(K.pow(n_y_pred[:]-K.mean(n_y_pred[:]),2),axis=[-1,-2]))
    result=top/bottom
    return K.mean(result)
    # return tfp.stats.correlation(y_pred,y_true, sample_axis=None, event_axis=None)
    # x = y_true
    # y = y_pred
    # mx = K.mean(x, axis=0)
    # my = K.mean(y, axis=0)
    # xm, ym = x - mx, y - my
    # r_num = K.sum(xm * ym)
    # x_square_sum = K.sum(xm * xm)
    # y_square_sum = K.sum(ym * ym)
    # r_den = K.sqrt(x_square_sum * y_square_sum)
    # r = r_num / r_den
    # return K.mean(r)