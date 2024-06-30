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
#%%
train_size=(365 + 180) * 24
train=CX_station.iloc[:train_size,3:]
test=CX_station.iloc[train_size:,3:]
train_mean = train.mean()
train_std = train.std()

test_mean = test.mean()
test_std = test.std()