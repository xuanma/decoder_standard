import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def format_data_fnn(x, y, n_lag):
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

def format_data_from_trials_fnn(x, y, n_lag):
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
        temp = format_data_fnn(each[0], each[1], n_lag)
        x_.append(temp[0])
        y_.append(temp[1])
    return np.concatenate(x_), np.concatenate(y_)

class fnn_decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, drop_out):
        super(fnn_decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.drop_out = drop_out
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Dropout(self.drop_out),
            nn.LeakyReLU(0.5),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
    
    def forward(self, input):
        y = self.model(input)
        return y

def train_fnn_decoder(x, y, hidden_dim, training_params):
    loss_type = training_params['loss_type']
    optim_type = training_params['optim_type']
    epochs = training_params['epochs']
    batch_size = training_params['batch_size']
    lr = training_params['lr']
    drop_out = training_params['drop_out']
    use_cuda = training_params['use_cuda']
    #---- define the network ----#
    x_dim = x.shape[1]
    y_dim = y.shape[1]
    network = fnn_decoder(x_dim, hidden_dim, y_dim, drop_out)
    #---- specifying hyperparameters ----#
    if loss_type == 'L1':
        criterion = torch.nn.L1Loss()
    elif loss_type == 'MSE':
        criterion = torch.nn.MSELoss()
    if optim_type == 'SGD':
        optimizer = optim.SGD(network.parameters(), lr = lr)
    elif optim_type == 'Adam':
        optimizer = optim.Adam(network.parameters(), lr = lr)
    elif optim_type == 'RMSprop':
        optimizer = optim.RMSprop(network.parameters(), lr = lr)
    # ------ define tensor dataset and build dataloader ----- #
    train_set = torch.utils.data.TensorDataset(torch.Tensor(x), 
                                              torch.Tensor(y))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # ------------------------------------------------------- #
    if use_cuda == True:
        network = network.cuda()
    #---- train loop ----#
    network = network.train()
    for i in range(epochs):
        for batch_idx, (x_, y_) in enumerate(train_loader):
            if use_cuda:
                x_ = x_.cuda()
                y_ = y_.cuda()
            optimizer.zero_grad()
            y_pred = network.forward(x_)
            loss = criterion(y_pred, y_)
            loss.backward()
            optimizer.step()
        if i%100 == 0:
            print('The %d-th epoch was finished'%(i))
    
    return network
            
def test_fnn_decoder(network, x, use_cuda = False):
    network = network.eval()
    with torch.no_grad():
        x_ = torch.Tensor(x)
        if use_cuda:
            x_ = x_.cuda()
        y_pred = network(x_)
        if use_cuda:
            y_pred = y_pred.cpu()
    return y_pred.data.numpy()


