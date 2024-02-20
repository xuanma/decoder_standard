import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from scipy.optimize import least_squares

def relu(x):
    return (np.maximum(0, x))

def flatten_list(X):
    """
    Converting list containing multiple ndarrays into a large ndarray
    X: a list
    return: a numpy ndarray
    """
    n_col = np.size(X[0],1)
    Y = np.empty((0, n_col))
    for each in X:
        Y = np.vstack((Y, each))
    return Y

def format_data(x, y, n_lag):
    """
    To reshape the numpy arrays for Wiener filter fitting
    Parameters
        x: the input data for Wiener filter fitting, an ndarray
        y: the output data for Wiener filter fitting, an ndarray
        n_lag: the number of time lags, an int number
    Returns:
        out1: the reshaped array for x, an ndarray
        out2: the trimmed array for y, an ndarray
    """
    x_ = [x[i:i+n_lag, :].reshape(n_lag*x.shape[1]) for i in range(x.shape[0]-n_lag+1)]
    return np.asarray(x_), y[n_lag-1:, :]

def format_data_from_list(x, y, n_lag):
    if type(x) == np.ndarray:
        x = [x]
    if type(y) == np.ndarray:
        y = [y]
    x_, y_ = [], []
    for each in zip(x, y):
        temp = format_data(each[0], each[1], n_lag)
        x_.append(temp[0])
        y_.append(temp[1])
    return np.concatenate(x_), np.concatenate(y_)

def format_data_from_trials(x, y, n_lag):
    """
    To reshape lists containing multiple trials into a big array so as to form 
    the training data for Wiener filter fitting
    Parameters
        x: a list containing multiple trials, as the inputs for Wiener filter fitting
        y: a list containing multiple trials, as the outputs for Wiener filter fitting
        n_lag: the number of time lags, an int number
    Returns
        out1: the reshaped data for the input list x, an ndarray
        out2: the reshaped data for the input list y, an ndarray
    """
    if type(x) == np.ndarray:
        x = [x]
    if type(y) == np.ndarray:
        y = [y]
    x_, y_ = [], []
    for each in zip(x, y):
        temp = format_data(each[0], each[1], n_lag)
        x_.append(temp[0])
        y_.append(temp[1])
    return np.concatenate(x_), np.concatenate(y_)

def parameter_fit(x, y, c):
    """
    c : L2 regularization coefficient
    I : Identity Matrix
    Linear Least Squares (code defaults to this if c is not passed)
    H = ( X^T * X )^-1 * X^T * Y
    Ridge Regression
    R = c * I
    ridge regression doesn't penalize x
    R[0,0] = 0
    H = ( (X^T * X) + R )^-1 * X^T * Y
    """
    x_plus_bias = np.c_[np.ones((np.size(x, 0), 1)), x]
    R = c * np.eye( x_plus_bias.shape[1] )
    R[0,0] = 0;
    temp = np.linalg.inv(np.dot(x_plus_bias.T, x_plus_bias) + R)
    temp2 = np.dot(temp,x_plus_bias.T)
    H = np.dot(temp2,y)
    return H

def parameter_fit_with_sweep( x, y, C, kf ):
    reg_r2 = []
    print ('Sweeping ridge regularization using CV decoding on train data' )
    for c in C:
        print( 'Testing c= ' + str(c) )
        cv_r2 = []
        for train_indices, test_indices in kf.split(x):
            # split data into train and test
            train_x, test_x = x[train_indices,:], x[test_indices,:]
            train_y, test_y = y[train_indices,:], y[test_indices,:]
            # fit decoder
            H = parameter_fit(train_x, train_y, c)
            #print( H.shape )
            # predict
            test_y_pred = test_wiener_filter(test_x, H)
            # evaluate performance
            cv_r2.append(r2_score(test_y, test_y_pred, multioutput='raw_values'))
        # append mean of CV decoding for output
        cv_r2 = np.asarray(cv_r2)
        reg_r2.append( np.mean( cv_r2, axis=0 ) )

    reg_r2 = np.asarray(reg_r2)        
    reg_r2 = np.mean( reg_r2, axis=1 )
    best_c = C[ np.argmax( reg_r2 ) ] 
    return best_c

def train_wiener_filter(x, y, l2 = 0):
    """
    To train a linear decoder
    x: input data, e.g. neural firing rates
    y: expected results, e.g. true EMG values
    l2: 0 or 1, switch for turning L2 regularization on or off
    """
    if l2 == 1:
        n_l2 = 20
        C = np.logspace( 1, 5, n_l2 )
        kfolds = 4
        kf = KFold( n_splits = kfolds )
        best_c = parameter_fit_with_sweep( x, y, C, kf )
        print(best_c)
    else:
        best_c = 0
    H_reg = parameter_fit( x, y, best_c )
    return H_reg
   
def test_wiener_filter(x, H):
    """
    To get predictions from input data x with linear decoder
    x: input data
    H: parameter vector obtained by training
    """
    x_plus_bias = np.c_[np.ones((np.size(x, 0), 1)), x]
    y_pred = np.dot(x_plus_bias, H)
    return y_pred    
      
def nonlinearity(p, y, nonlinear_type = 'poly2'):
    if nonlinear_type == 'poly':
        print('Version updated, please specify if you need poly 2 or poly 3')
        return None
    elif nonlinear_type == 'poly2':
        return p[0]+p[1]*y+p[2]*y*y
    elif nonlinear_type == 'poly3':
        return p[0]+p[1]*y+p[2]*y**2+p[3]*y**3
    elif nonlinear_type == 'sigmoid':
        return 1/( 1+np.exp(-10*(y-p[0])) )
    
def nonlinearity_residue(p, y, z, nonlinear_type = 'poly2'):
    return (nonlinearity(p, y, nonlinear_type) - z).reshape((-1,))

def train_nonlinear_wiener_filter(x, y, l2 = 0, nonlinear_type = 'poly2'):
    """
    To train a nonlinear decoder
    x: input data, e.g. neural firing rates
    y: expected results, e.g. true EMG values
    l2: 0 or 1, switch for turning L2 regularization on or off
    """
    if l2 == 1:
        n_l2 = 20
        C = np.logspace( 1, 5, n_l2 )
        kfolds = 4
        kf = KFold( n_splits = kfolds )
        best_c = parameter_fit_with_sweep( x, y, C, kf )
        print(best_c)
    else:
        best_c = 0
    H_reg = parameter_fit( x, y, best_c )
    y_pred = test_wiener_filter(x, H_reg)
    if nonlinear_type == 'relu':
        return H_reg
    else:
        if nonlinear_type == 'poly':
            print('Version updated. Please specify if you want poly-2 or poly-3')
            init = [0, 0]
        elif nonlinear_type == 'poly2':
            init = [0.1, 0.1, 0.1]
        elif nonlinear_type == 'poly3':
            init = [0.1, 0.1, 0.1, 0.1]
        elif nonlinear_type == 'sigmoid':
            init = [0.5]
        res_lsq = least_squares(nonlinearity_residue, init, args = (y_pred, y, nonlinear_type))
        return H_reg, res_lsq

def test_nonlinear_wiener_filter(x, H, res_lsq, nonlinear_type = 'poly2'):  
    """
    To get predictions from input data x with nonlinear decoder
    x: input data
    H: parameter vector obtained by training
    res_lsq: nonlinear components obtained by training
    """
    y1 = test_wiener_filter(x, H)
    if nonlinear_type == 'relu':
        y2 = relu(y1)
    else:
        y2 = nonlinearity(res_lsq.x, y1, nonlinear_type)
    return y2    
    
    
    
    


