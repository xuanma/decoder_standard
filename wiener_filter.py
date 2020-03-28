import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from scipy.optimize import least_squares

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

def vaf(x,xhat):
    """
    Calculating vaf value
    x: actual values, a numpy array
    xhat: predicted values, a numpy array
    """
    x = x - x.mean(axis=0)
    xhat = xhat - xhat.mean(axis=0)
    return (1-(np.sum(np.square(x - xhat))/np.sum(np.square(x))))

def format_data(x, y, N):
    spike_N_lag = []
    emg_N_lag = []
    for i in range(np.size(x, 0) - N):
        temp = x[i:i+N, :]
        temp = temp.reshape((np.size(temp)))
        spike_N_lag.append(temp)
        emg_N_lag.append(y[i+N-1, :])
    return np.asarray(spike_N_lag), np.asarray(emg_N_lag)

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
      
def nonlinearity(p, y):
    return p[0]+p[1]*y+p[2]*y*y
    
def nonlinearity_residue(p, y, z):
    return (nonlinearity(p, y) - z).reshape((-1,))

def train_nonlinear_wiener_filter(x, y, l2 = 0):
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
    else:
        best_c = 0
    H_reg = parameter_fit( x, y, best_c )
    y_pred = test_wiener_filter(x, H_reg)
    res_lsq = least_squares(nonlinearity_residue, [0.1,0.1,0.1], args = (y_pred, y))
    return H_reg, res_lsq

def test_nonlinear_wiener_filter(x, H, res_lsq):  
    """
    To get predictions from input data x with nonlinear decoder
    x: input data
    H: parameter vector obtained by training
    res_lsq: nonlinear components obtained by training
    """
    y1 = test_wiener_filter(x, H)
    y2 = nonlinearity(res_lsq.x, y1)
    return y2    
    
    
    
    


