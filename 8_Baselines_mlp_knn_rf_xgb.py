# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import math
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import svm
import scipy.optimize as optimize
import xgboost as xgb
from sklearn import preprocessing
from matplotlib import pyplot as plt
from My_utils.evaluation_scheme import evaluation
from My_utils.preprocess_data import divide_data
import warnings
from matplotlib import rcParams
config = {
            "font.family": 'serif',
            "font.size": 12,
            "mathtext.fontset": 'stix',
            "font.serif": ['SimSun'],
         }
rcParams.update(config)
warnings.filterwarnings("ignore")

#%%
def Baselines(method,trainX,trainY,testX):
    # 预测
    if method == 'LR':  # 线性回归
        LR = LinearRegression()
        Norm_pred = LR.fit(trainX, trainY).predict(testX)
    elif method == 'MLP':  # Mlp回归
        MLP = MLPRegressor(hidden_layer_sizes=(128,128), activation='relu',
                           batch_size='auto',solver='sgd', alpha=1e-04,
                           learning_rate_init=0.001, max_iter=300,beta_1=0.9,
                           beta_2=0.999, epsilon=1e-08)
        Norm_pred = MLP.fit(trainX, trainY).predict(testX)
    elif method == 'DT':  # 决策树回归
        DT = DecisionTreeRegressor()
        Norm_pred = DT.fit(trainX, trainY).predict(testX)
    elif method == 'SVR':  # 支持向量机回归
        SVR_MODEL = svm.SVR()
        Norm_pred = SVR_MODEL.fit(trainX, trainY).predict(testX)
    elif method == 'KNN':  # K近邻回归
        KNN = KNeighborsRegressor(10, weights="distance", leaf_size=30,
                                  algorithm='auto', p=1,)
                                  # metric='chebyshev')
        Norm_pred = KNN.fit(trainX, trainY).predict(testX)
    elif method == 'RF':  # 随机森林回归
        RF = RandomForestRegressor(n_estimators=100,max_depth=8,
                                   random_state=42,criterion='mse')
        Norm_pred = RF.fit(trainX, trainY).predict(testX)
    elif method == 'AdaBoost':  # Adaboost回归
        AdaBoost = AdaBoostRegressor(n_estimators=50)
        Norm_pred = AdaBoost.fit(trainX, trainY).predict(testX)
    elif method == 'XGB':  # XGB回归
        XGB_params = {'learning_rate': 0.1, 'n_estimators': 40,
                      'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                      'objective': 'reg:squarederror', 'subsample': 0.7,
                      'colsample_bytree': 0.7, 'gamma': 0,
                      'reg_alpha': 0.1, 'reg_lambda': 0.1}
        XGB = xgb.XGBRegressor(**XGB_params)
        Norm_pred = XGB.fit(trainX, trainY).predict(testX)
    elif method == 'GBRT':  # GBRT回归
        GBRT = GradientBoostingRegressor(n_estimators=100)
        Norm_pred = GBRT.fit(trainX, trainY).predict(testX)
    elif method == 'BR':  # Bagging回归
        BR = BaggingRegressor()
        Norm_pred = BR.fit(trainX, trainY).predict(testX)
    elif method == 'ETR':  # ExtraTree极端随机树回归
        ETR = ExtraTreeRegressor()
        Norm_pred = ETR.fit(trainX, trainY).predict(testX)

    return Norm_pred
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
# %% # 数据读取与预处理
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
# %%set seed for reproductive 42 is the answer to the universe
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

#%%        Normalization
# TS=CX_station['PM₂.₅(μg/m³)'].values
MMMM=[]
for idx in [3]:#[3,4,5,6,7,8,9,10]

    TS = CX_station.iloc[:,idx].values
    Normalization = preprocessing.MinMaxScaler()
    # warning: should be reshape(-1, 1) not (none,1) column
    Norm_TS = Normalization.fit_transform(TS.reshape(-1, 1))
    # %%
    seq_len = 168
    pre_len = 168

    trainX, trainY, _, _, testX, testY = divide_data(data=Norm_TS, train_size=(365 + 180) * 24, val_size=0,
                                                     seq_len=seq_len, pre_len=pre_len)

    trainX, trainY = np.array(trainX).squeeze(), np.array(trainY).squeeze()
    testX, testY = np.array(testX).squeeze(), np.array(testY).squeeze()

    Norm_pred = Baselines(method='XGB', trainX=trainX, trainY=trainY, testX=testX)
    all_simu = Normalization.inverse_transform(Norm_pred)
    all_real = Normalization.inverse_transform(testY)

    Metric = []
    for i in range(168):
        MAE, RMSE, MAPE, R2 = evaluation(all_real[:, i], all_simu[:, i])
        IA = cul_WIA(all_real[:, i], all_simu[:, i])
        Metric.append([MAE, RMSE, MAPE, R2, IA])

    M = np.mean(np.array(Metric), axis=0)

    Deg_M = np.array(Metric)
    MMMM.append(Deg_M)
    print(M)
