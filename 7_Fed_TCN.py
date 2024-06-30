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
# %%######## FED_ltsm not good
# class Fed_enhance_LSTM(nn.Module):
#     def __init__(self, seq_len, hidden_size, pre_len):
#         super(Fed_enhance_LSTM, self).__init__()
#
#         self.lstm = nn.LSTM(seq_len, hidden_size, 3, batch_first=True, bidirectional=True, dropout=0.3)
#
#         self.fc = nn.Linear(hidden_size * 2, pre_len)
#
#     def forward(self, x):
#
#         # batch_size, seq_len, input_size= x.shape[0], x.shape[1], x.shape[2] #seq_len是特征 input_size是长度
#
#         state, _ = self.lstm(x)
#         b, s, h = state.size() #shape(batch,seq_len , hidden_size * num_directions------9,144,128*2)
#         # print(s, b, h)
#
#         final_out = self.fc(state.reshape(b * s, h)).reshape(b, s, -1)#(9*144,128*2)--(9*144,168)这个168是pre_len
#         # print(final_out.size())
#
#
#         return torch.mean(final_out,dim=1)
#%% TCN

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


#%%# %% functions
def approximate_pattern2(time_series, period):  # time_series shape(1,1,timestep)
    time_series = time_series.cpu()
    x_list = np.arange(1, time_series.size(2) + 1, 1)
    y_list = time_series.squeeze().numpy()
    pattern = 0
    para_list = []
    pattern_list = []
    res_list = []
    for i in range(len(period) - 1, -1, -1):
        global T
        T = period[i]
        a = [5, 5, 5, 7, 7, 7, 9, 9]
        p = np.random.rand(a[i])
        para, _ = optimize.curve_fit(func_fourier, x_list, y_list, p0=p)
        y_fit = [func_fourier(a, *para) for a in x_list]
        pattern = np.array(y_fit) + np.array(pattern)
        time_series = time_series - np.array(y_fit)
        res = time_series
        para_list.append(para)
        pattern_list.append(np.array(y_fit))
        res_list.append(res)

    return pattern_list, res_list[-1]
def func_fourier(x, *b):
    w = T
    ret = 0
    a = b[:-2]
    for deg in range(0, int(len(a) / 2) + 1):
        ret += a[deg] * np.cos(deg * 2 * math.pi / w * x) + a[len(a) - deg - 1] * np.sin(deg * 2 * math.pi / w * x)
    return ret + b[-1]
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
# %%set seed for reproductive 42 is the answer to the universe
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
#%%  period info , Save ahead to accelerate training
period_info = pd.read_csv('0-EEMD-Pollutants/period_info.csv')
# period_info = pd.read_csv('period_info.csv') # FOR DEBUG
# period=period_info[period_info['pollutant'].isin(['PM10'])].\
#     sort_values('station').iloc[:,2:9].values
period=np.array([[4.0,6.6,13.1,25.0,50.1,104.5,237.7],
                [4.0 ,7.8 ,12.6,23.9,47.6,	108.2,	243.7 ],
                [3.9 ,6.8 ,	12.4 ,	23.7 ,	45.9 ,	105.6 ,	213.7 ]])
#%%
TS = CX_station.iloc[:,10].values

Normalization = preprocessing.MinMaxScaler()
Norm_TS = Normalization.fit_transform(TS.reshape(-1, 1))  # warning: should be reshape(-1, 1) not (none,1) column
#      ##           divide data

trainX, trainY, _,_, testX, testY = divide_data(data=Norm_TS, train_size=(365 + 180) * 24,val_size=0,
                                           seq_len=seq_len, pre_len=pre_len)
trainX, trainY = trainX.transpose(1, 2).float(), trainY.transpose(1, 2).float()
testX, testY = testX.transpose(1, 2).float(), testY.transpose(1, 2).float()

start_time = time.time()
#   pattern info
train_patterns = []
train_res = []
for i in range(len(trainX)):
    train_P, train_R = approximate_pattern2(trainX[i].unsqueeze(0), period[2, :])
    train_patterns.append(np.array(train_P))
    train_res.append(np.array(train_R))

test_patterns = []
test_res = []
for i in range(len(testX)):
    test_P, test_R = approximate_pattern2(testX[i].unsqueeze(0), period[2, :])
    test_patterns.append(np.array(test_P))
    test_res.append(np.array(test_R))
# unify shape
train_intrinsic_pattern = torch.tensor(np.array(train_patterns)).float()
train_intrinsic_res = torch.tensor(np.array(train_res)).squeeze(1).float()

test_intrinsic_pattern = torch.tensor(np.array(test_patterns)).float()
test_intrinsic_res = torch.tensor(np.array(test_res)).squeeze(1).float()

train_IPT = torch.cat((train_intrinsic_pattern, train_intrinsic_res), dim=1)
test_IPT = torch.cat((test_intrinsic_pattern, test_intrinsic_res), dim=1)
#  nomalize data
train_IPT = nn.functional.normalize(train_IPT, dim=2)  # dim=2，是对第三个维度，也就是每一行操作
test_IPT = nn.functional.normalize(test_IPT, dim=2)

#  loader
train_dataset = Data.TensorDataset(trainX, trainY, train_IPT)
test_dataset = Data.TensorDataset(testX, testY, test_IPT)

# put into loader
train_loader = Data.DataLoader(dataset=train_dataset,
                               batch_size=batch_size, shuffle=False)
test_loader = Data.DataLoader(dataset=test_dataset,
                              batch_size=batch_size, shuffle=False)
end_time = time.time()

# 计算训练时间
training_time = end_time - start_time

print(f"IPT时间: {training_time} 秒")
#%%
def weight_loss(y, pre_y):

    func = nn.MSELoss(reduce = False)  #return vector
    weight=torch.tensor(np.linspace(1, 0.0001, y.size(2))).float().to(device)
    b_weight=weight.repeat(y.size(0),1).unsqueeze(1)
    loss=func(y, pre_y)
    adj_loss=torch.mean(b_weight*loss)

    return adj_loss

def R2_loss(y, pre_y):

    SSR=torch.sum(torch.square(y-torch.mean(y)))
    SSE=torch.sum(torch.square(y-pre_y))

    # R2=1-SSE/(SSE+SSR)

    return SSE/(SSE+SSR)


#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# fed = Fed_enhance_LSTM(seq_len=seq_len,hidden_size=64, pre_len=pre_len).to(device)
fed_TCN = TemporalConvNet(num_inputs=seq_len, num_channels=[32,168], kernel_size=2, dropout=0.2).to(device)

optimizer = torch.optim.RMSprop(fed_TCN .parameters(), lr=0.001)
# optimizer = torch.optim.AdamW(fed .parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.99)
# loss_func = weight_loss
loss_func=nn.MSELoss()
# loss_func = R2_loss

#%%
start_time = time.time()
best_val_loss = float("inf")
best_model = None

train_loss_all = []
# net.train()  # Turn on the train mode
fed_TCN.train()  # Turn on the train mode
total_loss = 0.

for epoch in range(200):
    train_loss = 0
    train_num = 0
    for step, (x, y, IPT) in enumerate(train_loader):

        time_start = time.time()

        optimizer.zero_grad()

        x, y = x.to(device), y.to(device)

        data = torch.cat((IPT.to(device), x), dim=1)

        # pre_y = net(data)  # fed

        pre_y = fed_TCN(data.permute(0,2,1))
        # pre_y = fed_TCN(IPT.permute(0, 2, 1).to(device)) #NO X
        # pre_y = fed_TCN(x.permute(0, 2, 1))  # just TCN

        loss = loss_func(y, pre_y.unsqueeze(1))
        # loss1=weight_loss(y, pre_y.unsqueeze(1))
        # loss2 = R2_loss(y, y.unsqueeze(1))
        # loss = loss1 + loss2 / (loss2 / loss1).detach()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(fed_TCN.parameters(), 0.05)  # 梯度裁剪，放backward和step直接，小模型可以不考虑用于缓解梯度爆炸
        torch.nn.utils.clip_grad_norm_(fed_TCN.parameters(), 0.05)
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

        best_model = fed_TCN

    scheduler.step()

end_time = time.time()

# 计算训练时间
training_time = end_time - start_time

print(f"模型训练时间为: {training_time} 秒")
 #%%
start_time = time.time()
best_model = best_model.eval()  # 转换成测试模式
# best_model = fed_TCN.eval()
data_test = torch.cat((test_IPT.float().to(device),testX.float().to(device) ), dim=1)
pred = best_model(data_test.permute(0,2,1).to(device))  #
# pred = best_model(test_IPT.permute(0,2,1).to(device))  #no X
# pred = best_model(testX.float().permute(0,2,1).to(device))  #just tcn

# pred = best_model(data_test)  #fed mean

Norm_pred = pred.squeeze().data.cpu().numpy()
# Norm_pred = pred.data.cpu().numpy()

all_simu = Normalization.inverse_transform(Norm_pred)
all_real = Normalization.inverse_transform(testY[:, 0, 0:168].data.numpy())

# 模型推断结束时记录时间戳
end_time = time.time()

# 计算推断时间
inference_time = end_time - start_time

print(f"模型推断时间为: {inference_time} 秒")

# evaluation(all_real.reshape(-1,1), all_simu.reshape(-1,1))
#%%
# np.savez('0-EEMD-Pollutants/error data/NO.npz', all_simu=all_simu, all_real=all_real)
# np.savez('0-EEMD-Pollutants/error data/NO2.npz', all_simu=all_simu, all_real=all_real)
# np.savez('0-EEMD-Pollutants/error data/NOX.npz', all_simu=all_simu, all_real=all_real)
# np.savez('0-EEMD-Pollutants/error data/CO.npz', all_simu=all_simu, all_real=all_real)
# np.savez('0-EEMD-Pollutants/error data/PM25.npz', all_simu=all_simu, all_real=all_real)
# np.savez('0-EEMD-Pollutants/error data/PM10.npz', all_simu=all_simu, all_real=all_real)
# np.savez('0-EEMD-Pollutants/error data/SO2.npz', all_simu=all_simu, all_real=all_real)
np.savez('0-EEMD-Pollutants/error data/O3.npz', all_simu=all_simu, all_real=all_real)
#%%  mean 168steps
Metric=[]
for i in range(168):
    MAE, RMSE, MAPE, R2=evaluation(all_real[:,i], all_simu[:,i])
    IA = cul_WIA(all_real[:,i], all_simu[:,i])
    Metric.append([MAE, RMSE, MAPE, R2,IA])

M=np.mean(np.array(Metric),axis=0)
Deg_M=pd.DataFrame(Metric)
print(M)
#%%
from thop import profile
from thop import clever_format
# 估算 FLOPs
flops, params = profile(fed_TCN, inputs=(data_test.permute(0,2,1).to(device),))

# 格式化输出
flops, params = clever_format([flops, params], "%.3f")

print(f"FLOPs: {flops}")
print(f"Parameters: {params}")

#%%
best_model = best_model.eval()  # 转换成测试模式
data_test = torch.cat((train_IPT.float().to(device),trainX.float().to(device) ), dim=1)
pred = best_model(data_test.permute(0,2,1).to(device))  #
Norm_pred = pred.squeeze().data.cpu().numpy()


all_simu = Normalization.inverse_transform(Norm_pred)
all_real = Normalization.inverse_transform(trainY[:, 0, 0:168].data.numpy())


Metric=[]
for i in range(168):
    MAE, RMSE, MAPE, R2=evaluation(all_real[:,i], all_simu[:,i])
    IA = cul_WIA(all_real[:,i], all_simu[:,i])
    Metric.append([MAE, RMSE, MAPE, R2,IA])

M=np.mean(np.array(Metric),axis=0)
Deg_M=pd.DataFrame(Metric)
print(M)

#%% probability range
alpha=3
Rate=[]
for i in range(len(all_real)):
    Uper = all_simu[i] +alpha*Deg_M.iloc[:, 1].values
    Lower = all_simu[i] - alpha*Deg_M.iloc[:, 1].values
    more=np.sum(all_real[i]-Uper>0)
    less = np.sum(all_real[i] - Lower < 0)
    Rate.append((more+less)/168)

OOR_ALL=np.mean(np.array(Rate[::168]))

Uper = all_simu[:,0] + alpha*Deg_M.iloc[0, 1]
Lower = all_simu[:,0] - alpha*Deg_M.iloc[0, 1]
more = np.sum(all_real[:,0] - Uper > 0)
less = np.sum(all_real[:,0] - Lower < 0)
OOR1=(more + less) / len(all_real)

Uper = all_simu[:,23] + alpha*Deg_M.iloc[23, 1]
Lower = all_simu[:,23] - alpha*Deg_M.iloc[23, 1]
more = np.sum(all_real[:,23] - Uper > 0)
less = np.sum(all_real[:,23] - Lower < 0)
OOR24=(more + less) / len(all_real)

Uper = all_simu[:,-1] + alpha*Deg_M.iloc[-1, 1]
Lower = all_simu[:,-1] - alpha*Deg_M.iloc[-1, 1]
more = np.sum(all_real[:,-1] - Uper > 0)
less = np.sum(all_real[:,-1] - Lower < 0)
OOR168=(more + less) / len(all_real)

OOR=np.array([OOR1,OOR24,OOR168,OOR_ALL])
1-OOR[-1]
