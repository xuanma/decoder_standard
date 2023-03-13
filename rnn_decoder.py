import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

def format_data_rnn(x, y, n_lag):
    """
    Here x, y are arrays for a single trial, not lists, like neural data and EMGs
    If y is [], the function will only take care of the x input
    """
    x_ = [x[i:i+n_lag, :] for i in range(x.shape[0]-n_lag+1)]
    if len(y)>0:
        return np.asarray(x_), y[n_lag-1:, :]
    else:
        return np.asarray(x_)
    

def format_data_from_trials_rnn(x, y, n_lag):
    """
    Here x, y are lists, for multiple trials. They can be arrays, as the function checks
    the type at the beginning.
    y can be empty ([]), if so the function will only take care of the x input
    """
    if type(x) == np.ndarray:
        x = [x]
    if type(y) == np.ndarray:
        y = [y]
    x_, y_ = [], []
    if len(y) > 0:
        for each in zip(x, y):
            temp = format_data_rnn(each[0], each[1], n_lag)
            x_.append(temp[0])
            y_.append(temp[1])
        return np.concatenate(x_), np.concatenate(y_)
    else:
        for each in x:
            temp = format_data_rnn(each, [], n_lag)
            x_.append(temp)
        return np.concatenate(x_)