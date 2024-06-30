# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math

import pandas as pd
import numpy as np
import torch
import scipy.optimize as optimize
from torch import nn
from sklearn import preprocessing
from matplotlib import pyplot as plt
from My_utils.evaluation_scheme import evaluation
from My_utils.preprocess_data import divide_data
from My_utils.train_model import train_model
from My_utils.forecasting_model import multi_outputs_forecasting
import warnings
import torch.utils.data as Data
import time

warnings.filterwarnings("ignore")
# %%
# 数据读取与预处理
# data_csv = pd.read_csv('0-EEMD-Pollutants\Roadside data-SH.csv')
data_csv = pd.read_csv('Roadside data-SH.csv')
pd.set_option('display.max_columns', 20)
# divide 4 stations
DF_station = data_csv[data_csv['站名'].isin(['浦东东方路交通站'])]
DF_station = DF_station.fillna(DF_station.interpolate())

JY_station = data_csv[data_csv['站名'].isin(['静安延安西路交通站'])]
JY_station = JY_station.fillna(JY_station.interpolate())

JG_station = data_csv[data_csv['站名'].isin(['静安共和新路交通站'])]
JG_station = JG_station.fillna(JG_station.interpolate())

CX_station = data_csv[data_csv['站名'].isin(['徐汇漕溪路交通站'])]
CX_station = CX_station.fillna(CX_station.interpolate())

#%%
class CNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, kernel_size, pool_size):
        super(CNN_LSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=pool_size)
        self.lstm = nn.LSTM(input_size=32,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=0.2,
                            bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, 168)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)  # 重新排列维度以适应LSTM输入要求
        temp, _ = self.lstm(x)
        b, s, h = temp.size()

        temp = temp.reshape(-1,64*2)

        # temp=self.relu(temp)
        outs = self.fc(temp)

        outs=outs.view(b,-1,168)#outs[:,-1,:]

        return outs[:,-1,:]

#%%
# 定义模型参数
input_size = 1  # 输入特征维度
hidden_size = 64  # LSTM隐藏层大小
num_layers = 1  # LSTM层数
kernel_size = 4  # 卷积核大小
pool_size = 3  # 池化核大小
seq_len = 168
pre_len = 168
batch_size = 128

#set seed for reproductive 42 is the answer to the universe
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

S = []
R = []
MMMM=[]
all_results = []
# for station in [CX_station, JY_station, JG_station, DF_station]:
for idx in [3,4,5,6,7,8,9,10]:

    TS = CX_station.iloc[:,idx].values

    Normalization = preprocessing.MinMaxScaler()
    Norm_TS = Normalization.fit_transform(TS.reshape(-1, 1))  # warning: should be reshape(-1, 1) not (none,1) column
    #      ##           divide data

    trainX, trainY, _,_,testX, testY = divide_data(data=Norm_TS, train_size=(365 + 180) * 24,val_size=0,
                                               seq_len=seq_len, pre_len=pre_len)
    trainX, trainY = trainX.transpose(1, 2).float(), trainY.transpose(1, 2).float()
    testX, testY = testX.transpose(1, 2).float(), testY.transpose(1, 2).float()

    train_dataset = Data.TensorDataset(trainX, trainY)
    test_dataset = Data.TensorDataset(trainX, trainY)

    train_loader = Data.DataLoader(dataset=train_dataset,
                                   batch_size=batch_size, shuffle=False)
    test_loader = Data.DataLoader(dataset=test_dataset,
                                  batch_size=batch_size, shuffle=False)

    # init
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CNN_LSTM_net = CNN_LSTM(input_size, hidden_size, num_layers, kernel_size, pool_size).to(device)
    # print(fed_lstm)

    optimizer = torch.optim.RMSprop(CNN_LSTM_net.parameters(), lr=0.00005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.99)
    loss_func = nn.MSELoss()

    best_val_loss = float("inf")
    best_model = None
    #           train
    train_loss_all = []
    CNN_LSTM_net.train()  # Turn on the train mode
    total_loss = 0.

    for epoch in range(200):
        train_loss = 0
        train_num = 0
        for step, (x, y) in enumerate(train_loader):

            time_start = time.time()

            optimizer.zero_grad()

            x, y = x.to(device), y.to(device)

            pre_y = CNN_LSTM_net(x)

            loss = loss_func(pre_y, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(CNN_LSTM_net.parameters(), 0.1)  # 梯度裁剪，放backward和step直接，小模型可以不考虑用于缓解梯度爆炸
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            train_num += x.size(0)

            time_end = time.time()
            time_c = time_end - time_start

            total_loss += loss.item()
            log_interval = int(len(trainX) / batch_size / 5)
            if (step + 1) % log_interval == 0 and (step + 1) > 0:
                cur_loss = total_loss / log_interval
                print('| epoch {:3d} | {:5d}/{:5d} batches | '
                      'lr {:02.6f} | '
                      'loss {:5.5f} | time {:8.2f}'.format(
                    epoch, (step + 1), len(trainX) // batch_size, scheduler.get_lr()[0],
                    cur_loss, time_c))
                total_loss = 0

        if (epoch + 1) % 5 == 0:
            print('-' * 89)
            print('end of epoch: {}, Loss:{:.7f}'.format(epoch + 1, loss.item()))
            print('-' * 89)
            train_loss_all.append(train_loss / train_num)

        # if train_loss < best_val_loss:
        #     best_val_loss = train_loss
        #     best_model = GRU_net

        scheduler.step()

    best_model = CNN_LSTM_net.eval()  # 转换成测试模式
    pred = best_model(testX.float().to(device))
    Norm_pred = pred.data.cpu().numpy()

    del CNN_LSTM_net

    all_simu = Normalization.inverse_transform(Norm_pred)
    all_real = Normalization.inverse_transform(testY[:, 0, 0:168].data.numpy())

    Metric = []
    for i in range(168):
        MAE, RMSE, MAPE, R2 = evaluation(all_real[:, i], all_simu[:, i])
        Metric.append([MAE, RMSE, MAPE, R2])

    M = np.mean(np.array(Metric), axis=0)

    Deg_M = np.array(Metric)
    MMMM.append(Deg_M)
    print(M)

#%%
all_simu=Normalization.inverse_transform(Norm_pred)
all_real=Normalization.inverse_transform(testY[:, 0, 0:168].data.numpy())

Metric=[]
for i in range(168):
    MAE, RMSE, MAPE, R2=evaluation(all_real[:,i], all_simu[:,i])
    Metric.append([MAE, RMSE, MAPE, R2])

M=np.mean(np.array(Metric),axis=0)

Deg_M=np.array(Metric)
print(M)