import pandas as pd
import numpy as np
from PyEMD import EEMD
from scipy.signal import hilbert

from matplotlib import pyplot as plt

import warnings

warnings.filterwarnings("ignore")
from matplotlib import rcParams
config = {
            "font.family": 'serif',
            "font.size": 15,# 相当于小四大小
            "mathtext.fontset": 'stix',#matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
            "font.serif": ['SimSun'],#宋体
            'axes.unicode_minus': False # 处理负号，即-号
         }
rcParams.update(config)
#%%
# 数据读取与预处理
data_csv = pd.read_csv('EEMD-Pollutants\Roadside data-SH.csv')
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

print(len(DF_station), len(JY_station), len(JG_station), len(CX_station))
# data_csv.head()
#%%
# find_length=365
# eemd = EEMD()
# extraction = []
# for station in ([DF_station, JY_station, JG_station, CX_station]):
#     eemd.eemd(station['NO₂(μg/m³)'].iloc[24 * 0:24 * (0 + find_length)].values)  # need array
#     t = np.arange(0, len(data_csv['NO₂(μg/m³)'].iloc[24 * 0:24 * (0 + find_length)]), 1)
#     imfs, res = eemd.get_imfs_and_residue()
#
#     # calculate the mean instaneous frequency##############
#     analytic_signal = hilbert(imfs)
#     inst_phase = np.unwrap(np.angle(analytic_signal))
#     inst_freqs = np.diff(inst_phase) / (2 * np.pi * (t[1] - t[0]))
#     inst_freqs = np.concatenate((inst_freqs, inst_freqs[:, -1].reshape(inst_freqs[:, -1].shape[0], 1)),
#                                 axis=1)  ##row vector!!
#     MIF = np.mean(inst_freqs, axis=1)
#
#     # transform to period
#     sample_interval = 60  # min
#     period = sample_interval / (60 * MIF)
#     extraction.append(period)
# # %% pedding zero
# max_len = max((len(l) for l in extraction))
# new_extraction = list(map(lambda l: list(l) + [0] * (max_len - len(l)), extraction))
# results = pd.DataFrame(new_extraction[:])
# results.index = ['浦东东方路交通站', '静安延安西路交通站', '静安共和新路交通站', '徐汇漕溪路交通站']
# results.head()
# np.save('EEMD-Pollutants/EEMD_results/NO2_pattern.npy',results)
#%%
eemd = EEMD()
daily1=[]
daily2=[]
daily3=[]
for step in range(18):
    find_length=30*(step+1)
    extraction = []
    # for station in ([DF_station, JY_station, JG_station, CX_station]):
    for station in ([CX_station]):
        eemd.eemd(station['PM₁₀(μg/m³)'].iloc[24 * 0:24 * (0 + find_length)].values)  # need array
        t = np.arange(0, len(data_csv['PM₁₀(μg/m³)'].iloc[24 * 0:24 * (0 + find_length)]), 1)
        imfs, res = eemd.get_imfs_and_residue()

        # calculate the mean instaneous frequency##############
        analytic_signal = hilbert(imfs)
        inst_phase = np.unwrap(np.angle(analytic_signal))
        inst_freqs = np.diff(inst_phase) / (2 * np.pi * (t[1] - t[0]))
        inst_freqs = np.concatenate((inst_freqs, inst_freqs[:, -1].reshape(inst_freqs[:, -1].shape[0], 1)),
                                    axis=1)  ##row vector!!
        MIF = np.mean(inst_freqs, axis=1)

        # transform to period
        sample_interval = 60  # min
        period = sample_interval / (60 * MIF)
        extraction.append(period)

        # pedding zero
        max_len = max((len(l) for l in extraction))
        new_extraction = list(map(lambda l: list(l) + [0] * (max_len - len(l)), extraction))
        results = pd.DataFrame(new_extraction[:])


    daily1.append(results[2])
    daily2.append(results[3])
    daily3.append(results[4])

#%%
daily1 = pd.DataFrame(daily1[:]).T
daily2 = pd.DataFrame(daily2[:]).T
daily3 = pd.DataFrame(daily3[:]).T
print(daily1)
#%%  PLOT

# 数据读取与预处理
OPT_LEN = pd.read_csv('EEMD-Pollutants/OPT_LEN.csv')

A1_station = OPT_LEN[OPT_LEN['stations'].isin(['A1'])]
A1_station = A1_station.iloc[:,1:19]
A1_station.index = ['NO', 'NO$_{2}$', 'NO$_X$', 'CO', 'PM_2.5']

A2_station = OPT_LEN[OPT_LEN['stations'].isin(['A2'])]
A2_station = A2_station.iloc[:,1:19]
A2_station.index = ['NO', 'NO_2', 'NO_X', 'CO', 'PM_2.5']

A3_station = OPT_LEN[OPT_LEN['stations'].isin(['A3'])]
A3_station = A3_station.iloc[:,1:19]
A3_station.index = ['NO', 'NO_2', 'NO_X', 'CO', 'PM_2.5']

A4_station = OPT_LEN[OPT_LEN['stations'].isin(['A4'])]
A4_station = A4_station.iloc[:,1:19]
A4_station.index = ['NO', 'NO_2', 'NO_X', 'CO', 'PM_2.5']
#%%
length=["30 days","60 days","90 days","120 days","150 days","180 days","210 days"
        ,"240 days","270 days","300 days","330 days","360 days","390 days","420 days"
        ,"450 days","480 days","510 days","540 days"]
i=1
x = np.array(range(0, 18, 1))
for station in [A1_station,A2_station,A3_station,A4_station]:
    # plt.subplot(2, 2, i+1, label='Station A{}'.format(i+1))
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.grid(ls = "--", lw = 0.3,dashes=(8, 4), color = "lightgray")
    plt.plot(station.T, linewidth=0, ls="--", marker="o", mfc="white", ms=10)
    plt.legend(['NO', 'NO$_{2}$', 'NO$_X$', 'CO', 'PM$_2$$_.$$_5$'])
    plt.axhline(y=24, c="r", ls="--", lw=1)
    plt.xticks(np.array(range(0, 18, 1)), length, rotation=40)
    plt.ylabel('Daily pattern(h/cycle)')

    x_max = station.max()
    x_min = station.min()
    for j in range(len(x_max)):
        plt.plot([x[j], x[j]], [x_min[j], x_max[j]], c='purple', ls="--", lw=1.5, alpha=0.8)
    plt.savefig("EEMD-Pollutants/Figs/opt_len_stationA{}.svg".format(i), dpi=600)
    plt.show()
    i=i+1

    # plt.tick_params(labelsize=11)
# plt.savefig("EEMD-Pollutants/Figs/MIF_NO.svg", dpi=600)
# plt.show()
#%%
# from numpy import polyfit, poly1d
# from scipy.interpolate import make_interp_spline
length=["30 days","60 days","90 days","120 days","150 days","180 days","210 days"
        ,"240 days","270 days","300 days","330 days","360 days","390 days","420 days"
        ,"450 days","480 days","510 days","540 days"]
fig, ax=plt.subplots(figsize=(12, 8))
ax.grid(axis="y",ls = "--", lw = 0.3,dashes=(8, 4), color = "#4E616C")
plt.plot(A1_station.T, linewidth=0.1,ls="--",marker = "o", mfc = "white", ms = 10)
plt.legend(['NO', 'NO$_{2}$', 'NO$_X$', 'CO', 'PM$_2$$_.$$_5$'])
plt.axhline(y=24, c="r", ls="--", lw=1)
plt.xticks(np.array(range(0,18,1)),length,rotation=40)
x=np.array(range(0,18,1))
x_max=A1_station.max()
x_min=A1_station.min()
for i in range(len(x_max)):
    plt.plot([x[i],x[i]], [x_min[i],x_max[i]],c='purple',ls="--", lw=2.5,alpha=0.6)
# plt.xlabel("Decomposition length")
plt.ylabel('Daily pattern(h/cycle)')


# smooth line
# x=np.array(range(0,17,1))
# y=np.array(A1_station.mean(axis=0))
# x_smooth = np.linspace(x.min(), x.max(), 1000)#list没有min()功能调用
# y_smooth = make_interp_spline(x, y)(x_smooth)
# plt.plot(x_smooth, y_smooth)

# plt.scatter(x, y, marker='o')#绘制散点图
# plt.show()
# coeff = polyfit(x,y, 2)
# p1 = np.poly1d(coeff)
# y_pre = p1(x)
# plt.plot(x,y,'r')
# plt.plot(x,y_pre)
plt.show()

#%%
from numpy import polyfit, poly1d
from scipy.linalg import lstsq
from scipy.stats import linregress
# x=np.array(range(0,17,1))
# M_X=np.tile(x,5).reshape(-1,17)
# y=np.array(A1_station)
# coeff = polyfit(M_X.flatten(),y.flatten(), 2)
# p1 = np.poly1d(coeff)
# y_pre = p1(x)
# plt.plot(x,y_pre)
# plt.show()
#%%

# from scipy.interpolate import make_interp_spline
# x_smooth = np.linspace(x.min(), x.max(), 1000)#list没有min()功能调用
# y_smooth = make_interp_spline(x, y)(x_smooth)
# plt.plot(x_smooth, y_smooth)
# plt.scatter(x, y, marker='o')#绘制散点图
# plt.show()
