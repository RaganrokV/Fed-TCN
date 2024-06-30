import pandas as pd
import numpy as np
from PyEMD import EMD, Visualisation, EEMD, CEEMDAN
from scipy.signal import hilbert
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
warnings.filterwarnings("ignore")



class IPT:
    def __init__(self,time_series,t):
        self.time_series=time_series
        self.sample_interval = t

    def Extract_period(self,time_series):
        eemd = EEMD()
        extraction = []
        eemd.eemd(time_series)  # need array
        t = np.arange(0, len(time_series), 1)
        imfs, res = eemd.get_imfs_and_residue()
        # calculate the mean instaneous frequency##############
        analytic_signal = hilbert(imfs)
        inst_phase = np.unwrap(np.angle(analytic_signal))
        inst_freqs = np.diff(inst_phase) / (2 * np.pi * (t[1] - t[0]))
        inst_freqs = np.concatenate((inst_freqs, inst_freqs[:, -1].reshape(inst_freqs[:, -1].shape[0], 1)),
                                    axis=1)  ##row vector!!
        MIF = np.mean(inst_freqs, axis=1)

        # transform to period`
        sample_interval = 60  # min
        period = sample_interval / (60 * MIF)
        extraction.append(period)

        return extraction

    def pedding_zore(self,extraction):
        max_len = max((len(l) for l in extraction))
        new_extraction = list(map(lambda l: list(l) + [0] * (max_len - len(l)), extraction))
        period_info = pd.DataFrame(new_extraction[:])
        period_info.index = ['浦东东方路交通站', '静安延安西路交通站', '静安共和新路交通站', '徐汇漕溪路交通站']

        return period_info


    def func_fourier(self,x, *b):
        global T
        w = T
        ret = 0
        a = b[:-2]
        for deg in range(0, int(len(a) / 2) + 1):
            ret += a[deg] * np.cos(deg * 2 * math.pi / w * x) + a[len(a) - deg - 1] * np.sin(deg * 2 * math.pi / w * x)
        return ret + b[-1]


    def approximate_pattern(self,time_series,period_info):
        Period = period_info.iloc[[0], 0:8].values
        original = time_series
        x = list(range(1, len(time_series) + 1))
        pattern = 0
        fig1 = plt.figure(figsize=(15, 8))
        para_list = []
        pattern_list = []
        res_list = []
        for i in range(len(Period[0]) - 1, -1, -1):
            T = Period[0][i]
            a = [7, 7, 7, 9, 9, 9, 11, 11]
            p = np.random.rand(a[i])
            para, _ = optimize.curve_fit(self.func_fourier, x, y, p0=p)
            y_fit = [self.func_fourier(a, *para) for a in x]
            pattern = np.array(y_fit) + np.array(pattern)
            time_series = time_series - np.array(y_fit)
            res = time_series
            para_list.append(para)
            pattern_list.append(y_fit)
            res_list.append(res)

            plt.subplot(3, 3, 8 - i)
            plt.plot(x, original, 'b-', label='data')
            plt.plot(x, pattern, 'r-', label='pattern')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()
        plt.show()

        fig2 = plt.figure(figsize=(15, 8))
        for i in range(len(Period[0])):
            patterns = pattern_list[i]

            plt.subplot(3, 3, i + 1)
            plt.plot(x, patterns, 'b-', label='patterns %i' % i)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()

        plt.subplot(3, 3, 9)
        plt.plot(x, res_list[-1], 'b-', label='res')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show(fig2)





