# -*- coding: utf-8 -*-
import math

import pandas as pd
import numpy as np
import torch
import scipy.optimize as optimize
from torch import nn
from sklearn import preprocessing
from matplotlib import pyplot as plt
from My_utils.evaluation_scheme import evaluation
from My_utils.preprocess_data import  divide_data
from My_utils.train_model import train_model
from My_utils.forecasting_model import multi_outputs_forecasting
import warnings
import torch.utils.data as Data
import time
import statsmodels.api as sm
from matplotlib import rcParams
config = {
            "font.family": 'serif',
            "font.size": 12,
            "mathtext.fontset": 'stix',
            "font.serif": ['SimSun'],
         }
rcParams.update(config)

warnings.filterwarnings("ignore")
# %%
# 数据读取与预处理
data_csv = pd.read_csv('0-EEMD-Pollutants\Roadside data-SH.csv')
# data_csv = pd.read_csv('Roadside data-SH.csv')
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


# %%####################  G bLRUOCK ####################
class LagFLSTM(nn.Module):
    def __init__(self, seq_len, hidden_size, num_layers, pre_len, acf_threshold):
        super(LagFLSTM, self).__init__()
        self.acf_threshold = acf_threshold
        self.lstm = nn.LSTM(
            input_size=seq_len,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.3,
            batch_first=True,
            bidirectional=True)
        self.out = nn.Linear(hidden_size*2, pre_len)

    def forward(self, x):
        device = next(self.parameters()).device  # 获取模型参数所在的设备
        acf_values = []
        for i in range(x.size(0)):
            sequence = x[i, 0, :].detach().cpu().numpy()  # 仅考虑第一个预测维度
            acf = sm.tsa.acf(sequence, nlags=len(sequence) - 1)
            acf_values.append(acf)
        acf_values_tensor = torch.tensor(acf_values, dtype=torch.float32).unsqueeze(1)

        # 根据阈值将自相关系数低于阈值的部分置为 0
        acf_mask = torch.where(acf_values_tensor < self.acf_threshold, torch.tensor(0), torch.tensor(1))

        acf_mask = acf_mask.to(device)  # 移动到与输入数据相同的设备上

        # 将原始序列与自相关系数阈值处理后的结果相乘
        x_processed = x * acf_mask

        temp, _ = self.lstm(x_processed)
        s, b, h = temp.size()
        temp = temp.view(s * b, h)
        outs = self.out(temp)
        lstm_out = outs.view(s, b, -1)
        return lstm_out
# %% Hyper Parameters
# 定义模型参数
# 示例使用
seq_len = 168  # 输入时间序列的特征数
hidden_size = 128  # LSTM隐藏层的大小
num_layers = 2  # LSTM层数
lag = 6  # 滞后层的时间步数
acf_threshold = -0.6  # 相关阈值
batch_size = 144
pre_len = 168

#set seed for reproductive 42 is the answer to the universe
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


S = []
R = []
MMMM=[]
all_results = []
# for station in [CX_station, JY_station, JG_station, DF_station]:
for idx in [3]:#[3,4,5,6,7,8,9,10]

    TS = CX_station.iloc[:,idx].values

    Normalization = preprocessing.MinMaxScaler()
    Norm_TS = Normalization.fit_transform(TS.reshape(-1, 1))  # warning: should be reshape(-1, 1) not (none,1) column
    #      ##           divide data

    trainX, trainY,_,_, testX, testY = divide_data(data=Norm_TS, train_size=(365 + 180) * 24,val_size=0,
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
    LagFLSTM_net = LagFLSTM(seq_len=seq_len, hidden_size=128, num_layers=2,
                            pre_len=pre_len, acf_threshold=acf_threshold).to(device)
    # print(fed_lstm)

    optimizer = torch.optim.RMSprop(LagFLSTM_net.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.99)
    loss_func = nn.MSELoss()

    best_val_loss = float("inf")
    best_model = None
    #           train
    train_loss_all = []
    LagFLSTM_net.train()  # Turn on the train mode
    total_loss = 0.

    for epoch in range(100):
        train_loss = 0
        train_num = 0
        for step, (x, y) in enumerate(train_loader):

            time_start = time.time()

            optimizer.zero_grad()

            x, y = x.to(device), y.to(device)

            pre_y = LagFLSTM_net(x)

            loss = loss_func(pre_y, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(LagFLSTM_net.parameters(), 0.1)  # 梯度裁剪，放backward和step直接，小模型可以不考虑用于缓解梯度爆炸
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

        if train_loss < best_val_loss:
            best_val_loss = train_loss
            best_model = LagFLSTM_net

        scheduler.step()

    # best_model = LagFLSTM_net.eval()  # 转换成测试模式
    pred = best_model(testX.float().to(device))
    Norm_pred = pred.data.cpu().numpy()

    del LagFLSTM_net

    all_simu=Normalization.inverse_transform(Norm_pred.squeeze(1))
    all_real = Normalization.inverse_transform(testY[:, 0, 0:168].data.numpy())

    Metric = []
    for i in range(168):
        MAE, RMSE, MAPE, R2 = evaluation(all_real[:, i], all_simu[:, i])
        Metric.append([MAE, RMSE, MAPE, R2])

    M = np.mean(np.array(Metric), axis=0)

    Deg_M = np.array(Metric)
    MMMM.append(Deg_M)
    print(M)


