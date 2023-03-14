import numpy as np
import time
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim

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
    
# Base class, for specifying model type and parameters
class BaseModel(nn.Module):
    def __init__(self, inputDim, hiddenNum, outputDim, layerNum, cell, use_cuda=False, dropout_rnn = 0.0, dropout_fc = 0.05):
        super(BaseModel, self).__init__()
        self.hiddenNum = hiddenNum
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.layerNum = layerNum
        self.use_cuda = use_cuda
        if cell == "RNN":
            self.cell = nn.RNN(input_size=self.inputDim, hidden_size=self.hiddenNum,
                        num_layers=self.layerNum, dropout=dropout_rnn,
                         nonlinearity="relu", batch_first=True,)
        if cell == "LSTM":
            self.cell = nn.LSTM(input_size=self.inputDim, hidden_size=self.hiddenNum,
                               num_layers=self.layerNum, dropout=dropout_rnn,
                               batch_first=True, )
        if cell == "GRU":
            self.cell = nn.GRU(input_size=self.inputDim, hidden_size=self.hiddenNum,
                                num_layers=self.layerNum, dropout=dropout_rnn,
                                 batch_first=True, )
        print(self.cell)
        self.fc = nn.Linear(self.hiddenNum, self.outputDim)
        self.dp = nn.Dropout(p = dropout_fc)

# Vanilla RNN
class RNNModel(BaseModel):
    def __init__(self, inputDim, hiddenNum, outputDim, layerNum, cell, use_cuda, dropout = 0.0):
        super(RNNModel, self).__init__(inputDim, hiddenNum, outputDim, layerNum, cell, use_cuda, dropout)

    def forward(self, x):
        batchSize = x.size(0)
        h0 = torch.zeros(self.layerNum * 1, batchSize , self.hiddenNum).type('torch.FloatTensor')
        if self.use_cuda:
            h0 = h0.cuda()
        rnnOutput, hn = self.cell(x, h0)
        hn = hn[-1,:,:].view(batchSize, self.hiddenNum)
        fcOutput = self.fc(hn)

        return fcOutput

# LSTM
class LSTMModel(BaseModel):
    def __init__(self, inputDim, hiddenNum, outputDim, layerNum, cell, use_cuda, dropout = 0.0, dropout_fc = 0.02, nonlinear_output = 1):
        super(LSTMModel, self).__init__(inputDim, hiddenNum, outputDim, layerNum, cell, use_cuda, dropout, dropout_fc)
        self.nonlinear_output = nonlinear_output

    def forward(self, x):
        batchSize = x.size(0)
        # h0 = torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum)
        # c0 = torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum)
        h0 = torch.randn(self.layerNum * 1, batchSize, self.hiddenNum)
        c0 = torch.randn(self.layerNum * 1, batchSize, self.hiddenNum)
        if self.use_cuda:
            h0 = h0.cuda()
            c0 = c0.cuda()
        rnnOutput, hn = self.cell(x, (h0, c0))
        ho = hn[0][-1,:,:].view(batchSize, self.hiddenNum)
        fcOutput = self.fc(ho)
        fcOutput = self.dp(fcOutput)
        if self.nonlinear_output == 1:
            fcOutput = nn.functional.relu(fcOutput)
        return fcOutput

# GRU
class GRUModel(BaseModel):
    def __init__(self, inputDim, hiddenNum, outputDim, layerNum, cell, use_cuda, dropout = 0.0, dropout_fc = 0.02, nonlinear_output = 1):
        super(GRUModel, self).__init__(inputDim, hiddenNum, outputDim, layerNum, cell, use_cuda, dropout, dropout_fc)
        self.nonlinear_output = nonlinear_output
        
    def forward(self, x):
        batchSize = x.size(0)
        h0 = torch.randn(self.layerNum * 1, batchSize, self.hiddenNum)
        if self.use_cuda:
            h0 = h0.cuda()
        rnnOutput, hn = self.cell(x, h0)
        ho = hn[-1,:,:].view(batchSize, self.hiddenNum)
        fcOutput = self.fc(ho)
        fcOutput = self.dp(fcOutput)
        if self.nonlinear_output == 1:
            fcOutput = nn.functional.relu(fcOutput)
        return fcOutput    

def train_rnn_decoder(train_x, train_y, method, decoder_param, check_point = 10):
    """
    train_x, train_y are numpy formatted by the format_data_rnn functions
    """
    loss_list = []
    
    # -------- decoder params -------- #
    lr = decoder_param['lr']
    hidden_num = decoder_param['hidden_num']
    n_layer = decoder_param['n_layer']
    epoch = decoder_param['epoch']
    batch_size = decoder_param['batch_size']
    use_cuda = decoder_param['use_cuda']
    dropout = decoder_param['dropout']
    dropout_fc = decoder_param['dropout_fc']
    nonlinear_output = decoder_param['nonlinear_output']
    loss_type = decoder_param['loss_type']
    # -------------------------------- # 
    
    
    # ------ define tensor dataset and build dataloader ----- #
    train_set = torch.utils.data.TensorDataset(torch.Tensor(train_x), 
                                              torch.Tensor(train_y))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # ------------------------------------------------------- #
    
    # --------- define the decoder specified by the method parameter -------- #
    net = None
    D_input, D_output = np.size(train_x, 2), np.size(train_y, 1)
    if method == "RNN":
        net = RNNModel(inputDim=D_input, hiddenNum=hidden_num, outputDim=D_output, 
                       layerNum=n_layer, cell="RNN", use_cuda=use_cuda, dropout = dropout)
    if method == "LSTM":
        net = LSTMModel(inputDim=D_input, hiddenNum=hidden_num, outputDim=D_output, layerNum=n_layer, 
                        cell="LSTM", use_cuda=use_cuda, dropout = dropout, dropout_fc = dropout_fc, nonlinear_output= nonlinear_output)
    if method == "GRU":
        net = GRUModel(inputDim=D_input, hiddenNum=hidden_num, outputDim=D_output, layerNum=n_layer, cell="GRU", 
                       use_cuda=use_cuda, dropout = dropout, dropout_fc = dropout_fc, nonlinear_output = nonlinear_output)
    if use_cuda:
        net = net.cuda()
    # ----------------------------------------------------------------------- #
    
    # ------ Define the key elements for decoder training ------- #
    optimizer = optim.Adam(net.parameters(), lr=lr)
    if loss_type == 'Huber':
        criterion = nn.HuberLoss()
    elif loss_type == 'MSE':
        criterion = nn.MSELoss()
    elif loss_type == 'L1':
        criterion = nn.L1Loss()
    # ----------------------------------------------- #
    t1 = time.time()
    loss_sum = 0
    
    # -------- Train loop --------- #
    net = net.train()
    for i in range(epoch):
        for batch_idx, (x, y) in enumerate(train_loader):
            if use_cuda:
                x = x.cuda()
                y = y.cuda()
            optimizer.zero_grad()
            pred = net.forward(x)
            loss = criterion(pred, y)
            loss_sum += loss.item()
            loss.backward()
            optimizer.step()
            if i % check_point == 0:
                if batch_idx % check_point == 0 and batch_idx != 0:
                   print("batch: %d , loss is:%f" % (batch_idx, loss_sum / check_point))
                   loss_list.append(loss_sum / check_point)
                   loss_sum = 0
        if i % check_point == 0:
            print("%d epoch is finished!" % (i+1))
            
    t2 = time.time()
    print("train time:", t2-t1)
    return net

def test_rnn_decoder(net, test_x, use_cuda=False):
    net = net.eval()
    with torch.no_grad():
        test_x = torch.Tensor(test_x)
        if use_cuda:
            test_x = test_x.cuda()
        pred = net(test_x)
        if use_cuda:
            pred = pred.cpu()
    return pred.data.numpy()

















