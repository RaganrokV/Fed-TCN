# -*- coding: utf-8 -*-
import math
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import scipy.optimize as optimize
from torch import nn
from sklearn import preprocessing
from matplotlib import pyplot as plt
from My_utils.preprocess_data import  divide_data
import warnings

from matplotlib import rcParams
config = {
            "font.family": 'serif',
            "font.size": 13,
            "mathtext.fontset": 'stix',
            "font.serif": ['SimSun'],
         }
rcParams.update(config)


warnings.filterwarnings("ignore")
#%%                 introduction figure
#%%  differernt step in introduction
x=np.array(list(range(120)))
y0=np.cos(0.2*x)
y1=np.sin(0.5*x)

y=y0+y1
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,figsize=(20, 8))

ax2.grid(ls="--", lw=0.3, dashes=(8, 4), color="lightgray")
ax2.plot(x,y,linewidth=0.5, ls="--", mfc="white", ms=5,color="k", label='ground truth')
ax2.plot(x[:24],y[:24],linewidth=0.5, ls="--", marker="o",color="k",
         mfc="white", ms=5 ,label='historical values')
ax2.plot(x[23:28],y[23:28],linewidth=0.5, ls="--", marker="o",color="r",
         mfc="white", ms=5,label='forecasting values')
ax2.set_yticks([-2,-1,0,1,2,],
           ['-2','-1','0','1','2'],rotation=0,fontsize=20)
ax2.set_xticks([0*24,1*24,2*24,3*24,4*24,5*24],
           ['0h','24h','48h','72h','96h','120h'],rotation=0,fontsize=20)
# ax2.legend()


ax3.grid(ls="--", lw=0.3, dashes=(8, 4), color="lightgray")
ax3.plot(x,y,linewidth=0.5, ls="--", mfc="white", ms=5,color="k", label='ground truth')
ax3.plot(x[:24],y[:24],linewidth=0.5, ls="--", marker="o",color="k",
         mfc="white", ms=5 ,label='historical values')
ax3.plot(x[23:72],y[23:72],linewidth=0.5, ls="--", marker="o",color="r",
         mfc="white", ms=5,label='forecasting values')
ax3.set_xticks([0*24,1*24,2*24,3*24,4*24,5*24],
           ['0h','24h','48h','72h','96h','120h'],rotation=0,fontsize=20)
ax3.set_yticks([-2,-1,0,1,2,],
           ['-2','-1','0','1','2'],rotation=0,fontsize=20)
# ax3.legend()


ax1.grid(ls="--", lw=0.3, dashes=(8, 4), color="lightgray")
ax1.plot(x,y,linewidth=0.5, ls="--", mfc="white", ms=5,color="k", label='ground truth')
ax1.plot(x[:24],y[:24],linewidth=0.5, ls="--", marker="o",color="k",
         mfc="white", ms=5 ,label='historical values')
ax1.plot(x[71],y[71],linewidth=0.5, ls="--", marker="o",color="r",
         mfc="white", ms=5,label='forecasting values')
ax1.set_yticks([-2,-1,0,1,2,],
           ['-2','-1','0','1','2'],rotation=0,fontsize=20)

ax1.set_xticks([0*24,1*24,2*24,3*24,4*24,5*24],
           ['0h','24h','48h','72h','96h','120h'],rotation=0,fontsize=20)
# ax1.legend()


ax4.grid(ls="--", lw=0.3, dashes=(8, 4), color="lightgray")
ax4.plot(x,y,linewidth=0.5, ls="--", mfc="white", ms=5,color="k", label='ground truth')
ax4.plot(x[:24],y[:24],linewidth=0.5, ls="--", marker="o",color="k",
         mfc="white", ms=5 ,label='historical values')
ax4.plot(x[23:-1],y[23:-1],linewidth=0.5, ls="--", marker="o",color="r",
         mfc="white", ms=5,label='forecasting values')
ax4.set_xticks([0*24,1*24,2*24,3*24,4*24,5*24],
           ['0h','24h','48h','72h','96h','120h'],rotation=0,fontsize=20)
ax4.set_yticks([-2,-1,0,1,2,],
           ['-2','-1','0','1','2'],rotation=0,fontsize=20)

# ax4.legend()

# box = ax1.get_position()
# ax1.set_position([box.x0, box.y0, box.width , box.height* 0.8])
# ax1.legend(loc='center left',bbox_to_anchor=(0.2, 1.12),ncol=3)
# plt.savefig(r"EEMD-Pollutants/Figs/strategy.svg", dpi=600)
plt.show()

#%%                 long-term pattern
#%% get pattern
data_csv = pd.read_csv('Roadside data-SH.csv')
CX_station = data_csv[data_csv['站名'].isin(['徐汇漕溪路交通站'])]
CX_station = CX_station.fillna(CX_station.interpolate())
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

seq_len = 168
pre_len = 168
epochs = 100
batch_size = 144

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

period_info = pd.read_csv('period_info.csv')
period=period_info[period_info['pollutant'].isin(['NO'])].\
    sort_values('station').iloc[:,2:9].values

TS = CX_station.iloc[:,3].values

Normalization = preprocessing.MinMaxScaler()
Norm_TS = Normalization.fit_transform(TS.reshape(-1, 1))  # warning: should be reshape(-1, 1) not (none,1) column
#      ##           divide data

trainX, trainY, _, _, testX, testY = divide_data(data=Norm_TS, train_size=(365 + 180) * 24, val_size=0,
                                                 seq_len=seq_len, pre_len=pre_len)
trainX, trainY = trainX.transpose(1, 2).float(), trainY.transpose(1, 2).float()
testX, testY = testX.transpose(1, 2).float(), testY.transpose(1, 2).float()

#   pattern info

test_patterns = []
test_res = []
for i in range(len(testX)):
    test_P, test_R = approximate_pattern2(testX[i].unsqueeze(0), period[2, :])
    test_patterns.append(np.array(test_P))
    test_res.append(np.array(test_R))
# unify shape

test_intrinsic_pattern = torch.tensor(np.array(test_patterns)).float()
test_intrinsic_res = torch.tensor(np.array(test_res)).squeeze(1).float()

test_IPT = torch.cat((test_intrinsic_pattern, test_intrinsic_res), dim=1)
#  nomalize data
 # dim=2，是对第三个维度，也就是每一行操作
test_IPT = nn.functional.normalize(test_IPT, dim=2)
#%%
pos=-128
fig, ax = plt.subplots(2, 4,figsize=(20, 8))

plt.subplot(241)
plt.grid(ls="--", lw=0.3, dashes=(8, 4), color="lightgray")
plt.plot(test_IPT[pos,0,:])

plt.subplot(242)
plt.grid(ls="--", lw=0.3, dashes=(8, 4), color="lightgray")
plt.plot(test_IPT[pos,1,:])

plt.subplot(243)
plt.plot(test_IPT[pos,2,:])
plt.grid(ls="--", lw=0.3, dashes=(8, 4), color="lightgray")

plt.subplot(244)
plt.plot(test_IPT[pos,3,:])
plt.grid(ls="--", lw=0.3, dashes=(8, 4), color="lightgray")

plt.subplot(245)
plt.plot(test_IPT[pos,4,:])
plt.grid(ls="--", lw=0.3, dashes=(8, 4), color="lightgray")

plt.subplot(246)
plt.plot(test_IPT[pos,5,:])
plt.grid(ls="--", lw=0.3, dashes=(8, 4), color="lightgray")

plt.subplot(247)
plt.plot(test_IPT[pos,6,:])
plt.grid(ls="--", lw=0.3, dashes=(8, 4), color="lightgray")

plt.subplot(248)
plt.plot(test_IPT[pos,7,:]+0.08)
plt.grid(ls="--", lw=0.3, dashes=(8, 4), color="lightgray")

# plt.savefig(r"EEMD-Pollutants/Figs/disp pattern.svg", dpi=600)

plt.show()



#%%                 practical implications
#%%
config = {
            "font.family": 'serif',
            "font.size": 25,
            "mathtext.fontset": 'stix',
            "font.serif": ['SimSun'],
         }
rcParams.update(config)

time_list=['1/1/2018','1/2/2018','1/3/2018','1/4/2018','1/5/2018','1/6/2018','1/7/2018']
y=CX_station['NO(μg/m³)'].values
x=range(0,len(y),1)
fig, ax=plt.subplots(figsize=(20, 6))
# ax.spines["left"].set_visible(False)
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)

ax.grid(ls="--", lw=0.3, dashes=(8, 4), color="lightgray")

# ax.fill_between(x = range(0,168,1), y1 =y[0:168], y2=20, color="b",alpha = 0.3)


plt.plot(y[0*24:7*24],color='k',marker = "o", mfc = "white", ms = 0)

# plt.margins(y=0)
plt.ylim([0,200])
plt.xticks([0*24+12,1*24+12,2*24+12,3*24+12,4*24+12,5*24+12,6*24+12], time_list,color='k',rotation=0)
# plt.xlabel('2018-1-1')
plt.ylabel('NO(μg/m³)')
# plt.savefig("EEMD-Pollutants/Figs/whole-week.svg", dpi=600)
plt.show()
#%%  lineplot with plot     one_day
# time_list=['1:00','2:00','3:00','4:00','5:00','6:00','7:00','8:00','9:00','10:00'
#            ,'11:00','12:00','13:00','14:00','15:00','16:00','17:00','18:00','19:00'
#            ,'20:00','21:00','22:00','23:00','24:00']
time_list=['0:00','4:00','8:00','12:00','16:00','20:00','0:00']
y=CX_station['NO(μg/m³)'].values
x=range(0,len(y),1)
fig, ax=plt.subplots(figsize=(12, 6))
# ax.spines["left"].set_visible(False)
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)

ax.grid(ls="--", lw=0.3, dashes=(8, 4), color="lightgray")

# ax.fill_between(x = range(5,10,1), y1 =y[5:10], y2=20, color="r",alpha = 0.3)
# ax.fill_between(x = range(17,22,1), y1 =y[17:22],y2=20, color="r", alpha = 0.3)

plt.plot(y[1*24:2*24],color='k',marker = "o", mfc = "white", ms =8)

# plt.margins(y=0)
plt.ylim([0,200])
plt.xticks(x[0:25:4], time_list,color='k',rotation=0)
# plt.xlabel('2018-1-1')
plt.ylabel('NO(μg/m³)')
# plt.savefig("EEMD-Pollutants/Figs/one_day.svg", dpi=600)
plt.show()


#%%  lineplot with plot     two_day
# time_list=['1:00','2:00','3:00','4:00','5:00','6:00','7:00','8:00','9:00','10:00'
#            ,'11:00','12:00','13:00','14:00','15:00','16:00','17:00','18:00','19:00'
#            ,'20:00','21:00','22:00','23:00','24:00']
time_list=['0:00','4:00','8:00','12:00','16:00','20:00','0:00','4:00','8:00','12:00','16:00','20:00','0:00']
y=CX_station['NO(μg/m³)'].values
x=range(0,len(y),1)
fig, ax=plt.subplots(figsize=(18, 6))
# ax.spines["left"].set_visible(False)
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)

ax.grid(ls="--", lw=0.3, dashes=(8, 4), color="lightgray")

# ax.fill_between(x = range(5,10,1), y1 =y[5:10], y2=20, color="r",alpha = 0.3)
# ax.fill_between(x = range(17,22,1), y1 =y[17:22],y2=20, color="r", alpha = 0.3)

plt.plot(y[4*24:6*24],color='k',marker = "o", mfc = "white", ms =8)

# plt.margins(y=0)
plt.ylim([0,200])
plt.xticks(x[0:49:4], time_list,color='k',rotation=0)
# plt.xlabel('2018-1-1')
plt.ylabel('NO(μg/m³)')
# plt.savefig("EEMD-Pollutants/Figs/TWO_day.svg", dpi=600)
plt.show()
#%%  half day
# time_list=['2018-1-1 1:00','2018-1-1 5:00','2018-1-1 9:00','2018-1-1 13:00','2018-1-1 17:00'
#            ,'2018-1-1 21:00','2018-1-2 1:00','2018-1-2 5:00','2018-1-2 9:00','2018-1-3 13:00'
#            ,'2018-1-2 17:00','2018-1-2 21:00','2018-1-3 1:00','2018-1-3 5:00','2018-1-3 9:00',
#            '2018-1-3 13:00','2018-1-3 17:00','2018-1-3 21:00']
time_list=['1/1/2018','1/2/2018','1/3/2018','1/4/2018','1/5/2018','1/6/2018','1/7/2018']
y=CX_station['NO(μg/m³)'].values
x=range(0,len(y),1)
fig, ax=plt.subplots(figsize=(12, 6))
# ax.spines["left"].set_visible(False)
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)

ax.grid(ls = "--", lw = 0.5, color = "#4E616C")

ax.fill_between(x = range(3*24+1,4*24+1,1), y1 =y[3*24+1:4*24+1], y2=0, color="r",alpha = 0.3)
ax.fill_between(x = range(5*24+1,6*24+1,1), y1 =y[5*24+1:6*24+1], y2=0, color="r",alpha = 0.3)
ax.fill_between(x = range(0*24,1*12,1), y1 =y[0:12],y2=0, color="g", alpha = 0.3)
ax.fill_between(x = range(1*24,36,1), y1 =y[24:36],y2=0, color="g", alpha = 0.3)

# plt.plot(y[0*24:7*24],color='k',marker = "o", mfc = "white", ms = 2)
plt.plot(y[0*24:7*24],color='k',marker = "o", mfc = "white", ms = 1)

plt.margins(y=0)
# plt.ylim([20,200])

plt.xticks(range(0,30*24,24), time_list,color='k',rotation=30)
# plt.xlabel('2018-1-1')
plt.ylabel('NO(μg/m³)')
# plt.savefig("EEMD-Pollutants/Figs/day and semi-day.svg", dpi=600)
plt.show()
