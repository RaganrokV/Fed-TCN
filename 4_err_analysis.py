# -*- coding: utf-8 -*-
import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
#%%
# 计算残差

loaded_data = np.load('0-EEMD-Pollutants/error data/NO.npz')
all_simu = loaded_data['all_simu'][:,1]
all_real = loaded_data['all_real'][:,1]

# Calculate residuals
residuals = all_real - all_simu
skewness = skew(residuals)
kurt = kurtosis(residuals)

print("偏度：", skewness)
print("峰度：", kurt)

#%%
loaded_data = np.load('0-EEMD-Pollutants/error data/NO2.npz')
all_simu = loaded_data['all_simu'][:,1]
all_real = loaded_data['all_real'][:,1]

# Calculate residuals
residuals = all_real - all_simu
skewness = skew(residuals)
kurt = kurtosis(residuals)

print("偏度：", skewness)
print("峰度：", kurt)
#%%
loaded_data = np.load('0-EEMD-Pollutants/error data/NOX.npz')
all_simu = loaded_data['all_simu'][:,1]
all_real = loaded_data['all_real'][:,1]

# Calculate residuals
residuals = all_real - all_simu
skewness = skew(residuals)
kurt = kurtosis(residuals)

print("偏度：", skewness)
print("峰度：", kurt)
#%%
loaded_data = np.load('0-EEMD-Pollutants/error data/CO.npz')
all_simu = loaded_data['all_simu'][:,1]
all_real = loaded_data['all_real'][:,1]

# Calculate residuals
residuals = all_real - all_simu
skewness = skew(residuals)
kurt = kurtosis(residuals)

print("偏度：", skewness)
print("峰度：", kurt)
#%%
loaded_data = np.load('0-EEMD-Pollutants/error data/PM25.npz')
all_simu = loaded_data['all_simu'][:,1]
all_real = loaded_data['all_real'][:,1]

# Calculate residuals
residuals = all_real - all_simu
skewness = skew(residuals)
kurt = kurtosis(residuals)

print("偏度：", skewness)
print("峰度：", kurt)
#%%
loaded_data = np.load('0-EEMD-Pollutants/error data/PM10.npz')
all_simu = loaded_data['all_simu'][:,1]
all_real = loaded_data['all_real'][:,1]

# Calculate residuals
residuals = all_real - all_simu
skewness = skew(residuals)
kurt = kurtosis(residuals)

print("偏度：", skewness)
print("峰度：", kurt)
#%%
loaded_data = np.load('0-EEMD-Pollutants/error data/SO2.npz')
all_simu = loaded_data['all_simu'][:,1]
all_real = loaded_data['all_real'][:,1]

# Calculate residuals
residuals = all_real - all_simu
skewness = skew(residuals)
kurt = kurtosis(residuals)

print("偏度：", skewness)
print("峰度：", kurt)
#%%
loaded_data = np.load('0-EEMD-Pollutants/error data/O3.npz')
all_simu = loaded_data['all_simu'][:,1]
all_real = loaded_data['all_real'][:,1]

# Calculate residuals
residuals = all_real - all_simu
skewness = skew(residuals)
kurt = kurtosis(residuals)

print("偏度：", skewness)
print("峰度：", kurt)