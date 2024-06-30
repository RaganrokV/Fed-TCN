# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# #!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

import torch
from torch import nn
from sklearn import preprocessing
import math
import torch.utils.data as Data
from matplotlib import pyplot as plt
from My_utils.evaluation_scheme import evaluation
from My_utils.preprocess_data import divide_data
import scipy.optimize as optimize
import warnings
import torch.nn.functional as F
import time
from matplotlib import rcParams
config = {
            "font.family": 'serif',
            "font.size": 18,
            "mathtext.fontset": 'stix',
            "font.serif": ['SimSun'],
         }
rcParams.update(config)

warnings.filterwarnings("ignore")
#%%# %%数据读取与预处理
data_csv = pd.read_csv('0-EEMD-Pollutants/Roadside data-SH.csv')
# data_csv = pd.read_csv('Roadside data-SH.csv')#####for debug
# pd.set_option('display.max_columns', 20)
# divide 4 stations
DF_station = data_csv[data_csv['站名'].isin(['浦东东方路交通站'])]
DF_station = DF_station.fillna(DF_station.interpolate())

JY_station = data_csv[data_csv['站名'].isin(['静安延安西路交通站'])]
JY_station = JY_station.fillna(JY_station.interpolate())

JG_station = data_csv[data_csv['站名'].isin(['静安共和新路交通站'])]
JG_station = JG_station.fillna(JG_station.interpolate())

CX_station = data_csv[data_csv['站名'].isin(['徐汇漕溪路交通站'])]
CX_station = CX_station.fillna(CX_station.interpolate())
#%% MO-TCNA

from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        其实这就是一个裁剪的模块，裁剪多出来的padding
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs=360, n_outputs=168, kernel_size=2, stride=1, dilation=None, padding=3, dropout=0.2):
        """
        相当于一个Residual block

        :param n_inputs: int, 输入通道数
        :param n_outputs: int, 输出通道数
        :param kernel_size: int, 卷积核尺寸
        :param stride: int, 步长，一般为1
        :param dilation: int, 膨胀系数
        :param padding: int, 填充系数
        :param dropout: float, dropout比率
        """
        super(TemporalBlock, self).__init__()
        if dilation is None:
            dilation = 1
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)
        self.chomp1 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        参数初始化

        :return:
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return:
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout):
        """
        TCN，目前paper给出的TCN结构很好的支持每个时刻为一个数的情况，即sequence结构，
        对于每个时刻为一个向量这种一维结构，勉强可以把向量拆成若干该时刻的输入通道，
        对于每个时刻为一个矩阵或更高维图像的情况，就不太好办。

        :param num_inputs: int， 输入通道数
        :param num_channels: list，每层的hidden_channel数，例如[25,25,25,25]表示有4个隐层，每层hidden_channel数为25
        :param kernel_size: int, 卷积核尺寸
        :param dropout: float, drop_out比率
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i  # 膨胀系数：1，2，4，8……
            in_channels = num_inputs if i == 0 else num_channels[i - 1]  # 确定每一层的输入通道数
            out_channels = num_channels[i]  # 确定每一层的输出通道数
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        输入x的结构不同于RNN，一般RNN的size为(Batch, seq_len, channels)或者(seq_len, Batch, channels)，
        这里把seq_len放在channels后面，把所有时间步的数据拼起来，当做Conv1d的输入尺寸，实现卷积跨时间步的操作，
        很巧妙的设计。

        :param x: size of (Batch, input_channel, seq_len)
        :return: size of (Batch, output_channel, seq_len)
        """
        # return self.network(x)
        return torch.mean(self.network(x),dim=2)
        # return torch.sum(self.network(x), dim=2,keepdim=True)


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

#%%
S = []
R = []
MMMM=[]
all_results = []
# for station in [CX_station, JY_station, JG_station, DF_station]:
for idx in [3,4,5,6,7,8,9,10]:#[3,4,5,6,7,8,9,10]

    TS = CX_station.iloc[:,idx].values
    Normalization = preprocessing.MinMaxScaler()
    Norm_TS = Normalization.fit_transform(TS.reshape(-1, 1))  # warning: should be reshape(-1, 1) not (none,1) column
    #      ##           divide data

    trainX, trainY, _, _, testX, testY = divide_data(data=Norm_TS, train_size=(365 + 180) * 24, val_size=0,
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
    # %%
    # fed = Fed_enhance_LSTM(seq_len=seq_len,hidden_size=64, pre_len=pre_len).to(device)
    MO_TCNA = TemporalConvNet(num_inputs=seq_len, num_channels=[64, 168], kernel_size=2, dropout=0.2).to(device)

    optimizer = torch.optim.RMSprop(MO_TCNA.parameters(), lr=0.001)
    # optimizer = torch.optim.AdamW(fed .parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.99)
    # loss_func = weight_loss
    loss_func = nn.MSELoss()
    # loss_func = R2_loss

    best_val_loss = float("inf")
    best_model = None

    train_loss_all = []
    # net.train()  # Turn on the train mode
    MO_TCNA.train()  # Turn on the train mode
    total_loss = 0.

    for epoch in range(200):
        train_loss = 0
        train_num = 0
        for step, (x, y) in enumerate(train_loader):

            time_start = time.time()

            optimizer.zero_grad()

            x, y = x.to(device), y.to(device)

            # pre_y = net(data)  # fed

            pre_y = MO_TCNA(x.permute(0, 2, 1))
            # pre_y = fed_TCN(IPT.permute(0, 2, 1).to(device)) #NO X
            # pre_y = fed_TCN(x.permute(0, 2, 1))  # just TCN

            loss = loss_func(y, pre_y.unsqueeze(1))
            # loss1=weight_loss(y, pre_y.unsqueeze(1))
            # loss2 = R2_loss(y, y.unsqueeze(1))
            # loss = loss1 + loss2 / (loss2 / loss1).detach()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(fed_TCN.parameters(), 0.05)  # 梯度裁剪，放backward和step直接，小模型可以不考虑用于缓解梯度爆炸
            torch.nn.utils.clip_grad_norm_(MO_TCNA.parameters(), 0.05)
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

            best_model = MO_TCNA

        scheduler.step()

    best_model = best_model.eval()  # 转换成测试模式
    # best_model = fed_TCN.eval()
    pred = MO_TCNA(testX.permute(0, 2, 1).float().to(device))
    Norm_pred = pred.data.cpu().numpy()
    all_simu = Normalization.inverse_transform(Norm_pred)
    all_real = Normalization.inverse_transform(testY[:, 0, 0:168].data.numpy())

    del best_model,MO_TCNA

    Metric = []
    for i in range(168):
        MAE, RMSE, MAPE, R2 = evaluation(all_real[:, i], all_simu[:, i])
        Metric.append([MAE, RMSE, MAPE, R2])

    M = np.mean(np.array(Metric), axis=0)

    Deg_M = np.array(Metric)
    MMMM.append(Deg_M)
    print(M)



