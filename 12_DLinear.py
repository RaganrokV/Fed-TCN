# -*- coding: utf-8 -*-
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
#%%
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Module):
    """
    Decomposition-Linear
    """

    def __init__(self,seq_len,pred_len,individual,enc_in):
        super(Model, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = individual
        self.channels = enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)  # to [Batch, Output length, Channel]

#%%
def cul_WIA(oriList, preList):
    if len(oriList) != len(preList):
        raise Exception("len(ori_list) != len(pre_list) !")
    temp1 = 0  # 分子
    barO = 0  # 平均观测值
    for i in range(len(oriList)):
        temp1 += math.pow(oriList[i] - preList[i], 2)
        barO += oriList[i]/len(oriList)
    temp2 = 0  # 分母
    for i in range(len(oriList)):
        temp2 += math.pow(math.fabs(preList[i]-barO) +math.fabs(oriList[i]-barO),2)

    return 1 - temp1 / temp2
#%%
seq_len = 168
pre_len = 168
batch_size = 144

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

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
    trainX, trainY = trainX.float(), trainY.float()
    testX, testY = testX.float(), testY.float()

    train_dataset = Data.TensorDataset(trainX, trainY)
    test_dataset = Data.TensorDataset(trainX, trainY)

    train_loader = Data.DataLoader(dataset=train_dataset,
                                   batch_size=batch_size, shuffle=False)
    test_loader = Data.DataLoader(dataset=test_dataset,
                                  batch_size=batch_size, shuffle=False)

    # init
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DLinear = Model(seq_len=168,pred_len=168,individual=1,enc_in=1).to(device)
    # print(fed_lstm)

    optimizer = torch.optim.RMSprop(DLinear.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.99)
    loss_func = nn.MSELoss()

    best_val_loss = float("inf")
    best_model = None
    #           train
    train_loss_all = []
    DLinear.train()  # Turn on the train mode
    total_loss = 0.

    for epoch in range(1000):
        train_loss = 0
        train_num = 0
        for step, (x, y) in enumerate(train_loader):

            time_start = time.time()

            optimizer.zero_grad()

            x, y = x.to(device), y.to(device)

            pre_y = DLinear(x)

            loss = loss_func(pre_y, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(DLinear.parameters(), 0.1)  # 梯度裁剪，放backward和step直接，小模型可以不考虑用于缓解梯度爆炸
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

    best_model = DLinear.eval()  # 转换成测试模式
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


#%%
all_simu=Normalization.inverse_transform(Norm_pred.squeeze())
all_real=Normalization.inverse_transform(testY[:, 0:168,0 ].data.numpy())
Metric=[]
for i in range(168):
    MAE, RMSE, MAPE, R2=evaluation(all_real[:,i], all_simu[:,i])
    IA = cul_WIA(all_real[:,i], all_simu[:,i])
    Metric.append([MAE, RMSE, MAPE, R2,IA])

M=np.mean(np.array(Metric),axis=0)

Deg_M=np.array(Metric)
print(M)