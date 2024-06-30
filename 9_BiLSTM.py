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
class LSTM(nn.Module):
    def __init__(self, seq_len, hidden_size, num_layers, pre_len):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=seq_len,  # 输入纬度   记得加逗号
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.3,
            batch_first=True,
            bidirectional=True)
        self.out = nn.Linear(hidden_size*2, pre_len)

    def forward(self, x):


        temp, _ = self.lstm(x)
        s, b, h = temp.size()
        temp = temp.view(s * b, h)
        outs = self.out(temp)
        lstm_out = outs.view(s, b, -1)
        return lstm_out


# %%
def cul_WIA(oriList, preList):
    if len(oriList) != len(preList):
        raise Exception("len(ori_list) != len(pre_list) !")
    temp1 = 0  # 分子
    barO = 0  # 平均观测值
    for i in range(len(oriList)):
        temp1 += math.pow(oriList[i] - preList[i], 2)
        barO += oriList[i] / len(oriList)
    temp2 = 0  # 分母
    for i in range(len(oriList)):
        temp2 += math.pow(math.fabs(preList[i] - barO) + math.fabs(oriList[i] - barO), 2)

    return 1 - temp1 / temp2


# %% Hyper Parameters
seq_len = 168
pre_len = 168
batch_size = 144
# %%set seed for reproductive 42 is the answer to the universe
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
# %%
S = []
R = []
all_results = []
# for station in [CX_station, JY_station, JG_station, DF_station]:
for station in [CX_station]:

    TS = station.iloc[:,3].values

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
    LSTM_net = LSTM(seq_len=seq_len, hidden_size=128, num_layers=2, pre_len=pre_len).to(device)
    # print(fed_lstm)

    optimizer = torch.optim.RMSprop(LSTM_net.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.99)
    loss_func = nn.MSELoss()

    best_val_loss = float("inf")
    best_model = None
    #           train
    train_loss_all = []
    LSTM_net.train()  # Turn on the train mode
    total_loss = 0.

    for epoch in range(100):
        train_loss = 0
        train_num = 0
        for step, (x, y) in enumerate(train_loader):

            time_start = time.time()

            optimizer.zero_grad()

            x, y = x.to(device), y.to(device)

            pre_y = LSTM_net(x)

            loss = loss_func(pre_y, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(LSTM_net.parameters(), 0.1)  # 梯度裁剪，放backward和step直接，小模型可以不考虑用于缓解梯度爆炸
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

    best_model = LSTM_net.eval()  # 转换成测试模式
    pred = best_model(testX.float().to(device))
    Norm_pred = pred.data.cpu().numpy()

    # Metric = []
    # for forecasting_step in [24, 48, 96, 120, 168]:
    #     all_simu = []
    #     all_real = []
    #     results = []
    #     for i in range(len(testX)):
    #         simu = Normalization.inverse_transform(Norm_pred[i, :, :forecasting_step])
    #         all_simu.append(simu)
    #         real = Normalization.inverse_transform(testY[i, :, :forecasting_step].data.numpy())
    #         all_real.append(real)
    #         results.append(evaluation(real, simu))
    #
    #     MAE, RMSE, MAPE, _ = np.cumsum(results, axis=0)[-1] / len(testX)
    #     final_simu = np.cumsum(np.array(all_simu).squeeze(), axis=1)[:, -1] / forecasting_step
    #     final_real = np.cumsum(np.array(all_real).squeeze(), axis=1)[:, -1] / forecasting_step
    #     IA = cul_WIA(final_real, final_simu)
    #     Metric.append(np.array([MAE, RMSE, MAPE / 100, IA]))
    #
    # all_results.append(Metric)
    # S.append(all_simu)
    # R.append(all_real)
# %%
# GRU_M = pd.DataFrame(np.vstack((np.array(all_results[0]), np.array(all_results[1]),
#                                 np.array(all_results[2]), np.array(all_results[3]))))

#%%
# ERR=S+R
# np.save('EEMD-Pollutants/EEMD_results/GRU_ERR.npy',ERR)
# %%
all_simu=Normalization.inverse_transform(Norm_pred.squeeze(1))
all_real=Normalization.inverse_transform(testY[:, 0, 0:168].data.numpy())
#%%
Metric=[]
for i in range(168):
    MAE, RMSE, MAPE, R2=evaluation(all_real[:,i], all_simu[:,i])
    IA = cul_WIA(all_real[:,i], all_simu[:,i])
    Metric.append([MAE, RMSE, MAPE, R2,IA])

M=np.mean(np.array(Metric),axis=0)

Deg_M=np.array(Metric)
print(M)
