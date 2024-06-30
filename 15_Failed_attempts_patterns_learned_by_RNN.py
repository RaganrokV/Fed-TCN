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
            "font.size": 12,
            "mathtext.fontset": 'stix',
            "font.serif": ['SimSun'],
         }
rcParams.update(config)

warnings.filterwarnings("ignore")
#%%# %%数据读取与预处理
data_csv = pd.read_csv('EEMD-Pollutants/Roadside data-SH.csv')
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
# %%######## FED
class Fed_enhance_LSTM(nn.Module):
    def __init__(self, seq_len, data_hidden_size, res_hidden_size, hidden_size, pre_len):
        super(Fed_enhance_LSTM, self).__init__()

        # lstm layers
        self.lstm = nn.GRU(seq_len, data_hidden_size, 2, batch_first=True, bidirectional=True, dropout=0.5)
        self.lstm_res = nn.GRU(seq_len, res_hidden_size, 3, batch_first=True, bidirectional=True, dropout=0.5)
        self.lstm_pattern0 = nn.GRU(seq_len, hidden_size, 1, batch_first=True,  dropout=0.3)
        self.lstm_pattern1 = nn.GRU(seq_len, hidden_size, 1, batch_first=True,  dropout=0.3)
        self.lstm_pattern2 = nn.GRU(seq_len, hidden_size, 1, batch_first=True,  dropout=0.3)
        self.lstm_pattern3 = nn.GRU(seq_len, hidden_size, 1, batch_first=True, dropout=0.3)
        self.lstm_pattern4 = nn.GRU(seq_len, hidden_size, 1, batch_first=True, dropout=0.3)
        self.lstm_pattern5 = nn.GRU(seq_len, hidden_size, 1, batch_first=True,  dropout=0.3)
        self.lstm_pattern6 = nn.GRU(seq_len, hidden_size, 1, batch_first=True,  dropout=0.3)
        self.lstm_pattern7 = nn.GRU(seq_len, hidden_size, 1, batch_first=True, dropout=0.3)

        self.out_lstm = nn.Linear(data_hidden_size * 2, pre_len)
        self.out_res = nn.Linear(res_hidden_size * 2, pre_len)
        self.out_state = nn.Linear(hidden_size, pre_len)

        self.out_final = nn.Linear(pre_len * 2, pre_len)

        self.drop=nn.Dropout(0.2)
        self.relu = nn.ReLU()


    def forward(self, x, patterns, res):
        state, _ = self.lstm(x)
        s_d, b_d, h_d = state.size()
        outs_data = self.out_lstm(state.view(s_d * b_d, h_d)).view(s_d, b_d, -1)

        state0, _ = self.lstm_res(res)
        s_r, b_r, h_r = state0.size()
        outs_r = self.out_res(state0.view(s_r * b_r, h_r)).view(s_r, b_r, -1)

        state1, _ = self.lstm_pattern0(patterns[:,0,].unsqueeze(1))
        s, b, h = state1.size()

        state2, _ = self.lstm_pattern1(patterns[:,1,].unsqueeze(1))

        state3, _ = self.lstm_pattern2(patterns[:,2,].unsqueeze(1))

        state4, _ = self.lstm_pattern3(patterns[:,3,].unsqueeze(1))

        state5, _ = self.lstm_pattern4(patterns[:,4,].unsqueeze(1))

        state6, _ = self.lstm_pattern5(patterns[:,5,].unsqueeze(1))

        state7, _ = self.lstm_pattern6(patterns[:,6,].unsqueeze(1))



        # should add state or outs?
        state_pattern =  state1 + state2 + state3 + state4 + state5 + state6 + state7

        outs_pattern=self.out_state(state_pattern.view(s * b, h)).view(s, b, -1)

        fusion = torch.cat((outs_data, outs_pattern+outs_r), 2)
        fusion = self.drop(fusion)

        final_out = self.out_final(fusion)

        return final_out
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
epochs = 100
batch_size = 144
# %%set seed for reproductive 42 is the answer to the universe
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
#%%
period_info = pd.read_csv('EEMD-Pollutants/period_info.csv')
# period_info = pd.read_csv('period_info.csv') # FOR DEBUG
period=period_info[period_info['pollutant'].isin(['NO'])].\
    sort_values('station').iloc[:,2:9].values
#%%
TS = CX_station['NO(μg/m³)'].values

Normalization = preprocessing.MinMaxScaler()
Norm_TS = Normalization.fit_transform(TS.reshape(-1, 1))  # warning: should be reshape(-1, 1) not (none,1) column
#      ##           divide data

trainX, trainY, testX, testY = divide_data(data=Norm_TS, train_size=(365 + 180) * 24,
                                           seq_len=seq_len, pre_len=pre_len)
trainX, trainY = trainX.transpose(1, 2).float(), trainY.transpose(1, 2).float()
testX, testY = testX.transpose(1, 2).float(), testY.transpose(1, 2).float()

#   pattern info
train_patterns = []
train_res = []
for i in range(len(trainX)):
    train_P, train_R = approximate_pattern2(trainX[i].unsqueeze(0), period[0, :])
    train_patterns.append(np.array(train_P))
    train_res.append(np.array(train_R))

test_patterns = []
test_res = []
for i in range(len(testX)):
    test_P, test_R = approximate_pattern2(testX[i].unsqueeze(0), period[0, :])
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


#%%  LOOP FOR
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fed = Fed_enhance_LSTM(seq_len=seq_len, data_hidden_size=64, res_hidden_size=128,
                            hidden_size=32, pre_len=pre_len).to(device)

optimizer = torch.optim.AdamW(fed.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
loss_func = nn.MSELoss()

#%%  no masking
best_val_loss = float("inf")
best_model = None

train_loss_all = []
fed.train()  # Turn on the train mode
total_loss = 0.

for epoch in range(100):
    train_loss = 0
    train_num = 0
    for step, (x, y, IPT) in enumerate(train_loader):

        time_start = time.time()

        optimizer.zero_grad()

        x, y = x.to(device), y.to(device)

        patterns, res = IPT[:, 0:-1, :].to(device), IPT[:, -1, :].unsqueeze(1).to(device)

        pre_y = fed(x, patterns, res)

        loss = loss_func(pre_y, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fed.parameters(), 0.05)  # 梯度裁剪，放backward和step直接，小模型可以不考虑用于缓解梯度爆炸
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

        best_model = fed

    scheduler.step()
#%%
best_model = best_model.eval()  # 转换成测试模式
pred = best_model(testX.float().to(device), test_IPT[:, 0:7, :].float().to(device),
                  test_IPT[:, 7, :].unsqueeze(1).float().to(device))  #FED

Norm_pred = pred.data.cpu().numpy()

all_simu = Normalization.inverse_transform(Norm_pred.squeeze(1))
all_real = Normalization.inverse_transform(testY[:, 0, 0:168].data.numpy())
#%%
Metric=[]
for i in range(168):
    MAE, RMSE, MAPE, R2=evaluation(all_real[:,i].reshape(-1,1), all_simu[:,i].reshape(-1,1))
    IA = cul_WIA(all_real[:,i].reshape(-1,1), all_simu[:,i].reshape(-1,1))
    Metric.append([MAE, RMSE, MAPE, R2,IA])

M=np.mean(np.array(Metric),axis=0)
Deg_M=pd.DataFrame(Metric)
#%%  ################## GRU  #############




# %%####################  G bLRUOCK ####################
class GRU(nn.Module):
    def __init__(self, seq_len, hidden_size, num_layers, pre_len):
        super(GRU, self).__init__()
        # self.gru = nn.GRU(
        #     input_size=seq_len,  # 输入纬度   记得加逗号
        #     hidden_size=hidden_size,
        #     num_layers=num_layers,
        #     dropout=0.3,
        #     batch_first=True)
        # self.out = nn.Linear(hidden_size , pre_len)
        self.gru = nn.GRU(
            input_size=seq_len,  # 输入纬度   记得加逗号
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.3,
            batch_first=True,
            bidirectional=True)
        self.out = nn.Linear(hidden_size*2, pre_len)

    def forward(self, x):
        temp, _ = self.gru(x)
        s, b, h = temp.size()
        temp = temp.view(s * b, h)
        outs = self.out(temp)
        lstm_out = outs.view(s, b, -1)
        return lstm_out

#%%
TS = CX_station['NO(μg/m³)'].values

Normalization = preprocessing.MinMaxScaler()
Norm_TS = Normalization.fit_transform(TS.reshape(-1, 1))  # warning: should be reshape(-1, 1) not (none,1) column
#      ##           divide data

trainX, trainY, testX, testY = divide_data(data=Norm_TS, train_size=(365 + 180) * 24,
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
GRU_net = GRU(seq_len=seq_len, hidden_size=128, num_layers=2, pre_len=pre_len).to(device)
# print(fed_lstm)

optimizer = torch.optim.RMSprop(GRU_net.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
loss_func = nn.MSELoss()

best_val_loss = float("inf")
best_model = None
#           train
train_loss_all = []
GRU_net.train()  # Turn on the train mode
total_loss = 0.

for epoch in range(100):
    train_loss = 0
    train_num = 0
    for step, (x, y) in enumerate(train_loader):

        time_start = time.time()

        optimizer.zero_grad()

        x, y = x.to(device), y.to(device)

        pre_y = GRU_net(x)

        loss = loss_func(pre_y, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(GRU_net.parameters(), 0.05)  # 梯度裁剪，放backward和step直接，小模型可以不考虑用于缓解梯度爆炸
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
        best_model = GRU_net

    scheduler.step()
#%%
best_model = best_model.eval()  # 转换成测试模式
pred = best_model(testX.float().to(device))
Norm_pred = pred.data.cpu().numpy()


all_simu = Normalization.inverse_transform(Norm_pred.squeeze(1))
all_real = Normalization.inverse_transform(testY[:, 0, 0:168].data.numpy())
#%%
Metric=[]
for i in range(168):
    MAE, RMSE, MAPE, R2=evaluation(all_real[:,i].reshape(-1,1), all_simu[:,i].reshape(-1,1))
    IA = cul_WIA(all_real[:,i].reshape(-1,1), all_simu[:,i].reshape(-1,1))
    Metric.append([MAE, RMSE, MAPE, R2,IA])

M=np.mean(np.array(Metric),axis=0)
Deg_M=pd.DataFrame(Metric)
#%%
Metric=pd.DataFrame()



